"""DeepLabV3+ segmentation 어댑터 ([B,C,H,W] logit → VSC seg 컨트랙트).

컨트랙트 일관성: HBB 가 pre-NMS, CLS 가 pre-softmax logit 을 내보내듯 SEG 도 raw logit
[B,C,H,W] 을 내보내고 per-pixel softmax/argmax(또는 채널별 heatmap)는 VSC 가 수행한다.
seg_mode/seg_background_class 는 model.yaml decode 메타로만 전달(그래프에 활성화 미주입).

heavy lib(torch/torchvision/segmentation_models_pytorch)는 메서드 내부 lazy import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer
from .base import BaseTrainer


@register_trainer("segmentation", "deeplab3pp", "deeplab3pp_one_channel")
class DeepLabTrainer(BaseTrainer):

    def prepare_data(self) -> Any:
        from ..data import iter_labelme_dir, split_samples
        from ..data.writers import write_masks

        bg = self.cfg.export.seg_background_class
        samples = iter_labelme_dir(self.cfg.dataset.root)
        by_split = split_samples(samples, self.cfg.dataset.split)
        mask_root = write_masks(by_split, self.names, self.out_dir / "seg_cache", background=bg)
        # split 별 (이미지, 마스크) 쌍 — train() 이 곧장 Dataset 으로 사용
        pairs: dict[str, list[tuple[str, str]]] = {}
        for split, samples_s in by_split.items():
            m_dir = Path(mask_root) / "masks" / split
            rows = []
            for s in samples_s:
                mp = m_dir / f"{Path(s.image_path).stem}.png"
                if Path(s.image_path).exists() and mp.exists():
                    rows.append((s.image_path, str(mp)))
            pairs[split] = rows
        return pairs

    def _model(self, num_classes: int):
        av = self.cfg.arch_variant
        lib = av.get("lib", "smp")
        if lib == "torchvision":
            from torchvision.models import segmentation as seg
            key = av.get("backbone", av.get("encoder", "resnet50"))   # 두 키 모두 허용(통일)
            ctor = {
                "resnet50": seg.deeplabv3_resnet50,
                "resnet101": seg.deeplabv3_resnet101,
                "mobilenet": seg.deeplabv3_mobilenet_v3_large,
            }.get(key)
            if ctor is None:
                raise ValueError(f"torchvision deeplab backbone '{key}' 미지원"
                                 "(resnet50/resnet101/mobilenet). 그 외 백본은 lib: smp 사용")
            return ctor(num_classes=num_classes, aux_loss=True)
        import segmentation_models_pytorch as smp  # DeepLabV3+ (= deeplab3pp)
        enc = av.get("encoder", av.get("backbone", "resnet50"))
        weights = av.get("encoder_weights", "imagenet")
        return smp.DeepLabV3Plus(encoder_name=enc, encoder_weights=weights,
                                 in_channels=self.cfg.export.input.c, classes=num_classes)

    def train(self, data: Any) -> Path:
        import cv2
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        t = self.cfg.train
        dev = f"cuda:{t.gpus[0]}" if (t.gpus and torch.cuda.is_available()) else "cpu"
        H, W = self.cfg.export.input.h, self.cfg.export.input.w
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)

        class SegDS(Dataset):
            def __init__(self, rows): self.rows = rows
            def __len__(self): return len(self.rows)
            def __getitem__(self, i):
                ip, mp = self.rows[i]
                img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                x = ((img.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
                return torch.from_numpy(x), torch.from_numpy(msk.astype(np.int64))

        train_ds = SegDS(data.get("train", []))
        loader = DataLoader(train_ds, batch_size=t.batch, shuffle=True, num_workers=t.workers)

        model = self._model(len(self.names)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
        crit = nn.CrossEntropyLoss()
        model.train()
        for ep in range(t.epochs):
            loss = torch.tensor(0.0)
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                out = model(x)
                logits = out["out"] if isinstance(out, dict) else out   # torchvision=dict / smp=tensor
                loss = crit(logits, y)
                loss.backward()
                opt.step()
            print(f"[deeplab] epoch {ep+1}/{t.epochs} loss={float(loss):.4f}")
        ckpt = self.out_dir / "best.pt"
        torch.save({"state_dict": model.state_dict(),
                    "arch_variant": dict(self.cfg.arch_variant),
                    "num_classes": len(self.names)}, ckpt)   # ckpt 자기기술(재export 안전)
        return ckpt

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        import torch
        from torch import nn

        from ..export import VscExporter, introspect_output_shape

        e = self.cfg.export
        sd = torch.load(str(ckpt), map_location="cpu")
        ck_nc = sd.get("num_classes", len(self.names)) if isinstance(sd, dict) else len(self.names)
        if ck_nc != len(self.names):   # 학습/export config 클래스 수 불일치 → fail-fast
            raise ValueError(f"ckpt num_classes({ck_nc}) != dataset.names({len(self.names)})")
        model = self._model(len(self.names))
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        model.eval()

        from torch.nn import functional as F

        class Wrap(nn.Module):   # dict('out')→텐서 평탄화 + softmax(dim=1) 베이킹(참조 코어 동형)
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                o = self.m(x)
                logits = o["out"] if isinstance(o, dict) else o
                return F.softmax(logits, dim=1)

        wrapped = Wrap(model).eval()
        dst = self.out_dir / "model.onnx"
        dummy = torch.randn(1, e.input.c, e.input.h, e.input.w)
        dyn = ({e.io_names.input: {0: "B"}, e.io_names.output: {0: "B"}}
               if e.dynamic_axes else None)
        torch.onnx.export(wrapped, dummy, str(dst), opset_version=e.opset,
                          input_names=[e.io_names.input], output_names=[e.io_names.output],
                          dynamic_axes=dyn)   # ★ softmax(dim=1) BAKED → per-pixel 확률[0,1](실 manifest 정합)
        out_shape = introspect_output_shape(dst)
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
