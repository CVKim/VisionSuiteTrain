"""timm EfficientNet classification 어댑터 ([B,NC] logit → VSC cls 컨트랙트).

heavy lib(torch/timm/torchvision)는 메서드 내부 lazy import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer
from .base import BaseTrainer


class _RemapToNames:
    """ImageFolder 알파벳순 타깃 → dataset.names 순서 인덱스.

    DataLoader 워커(Windows spawn)에서 피클 가능해야 하므로 lambda 가 아닌 클래스.
    """
    def __init__(self, mapping: dict): self.mapping = mapping
    def __call__(self, t: int) -> int: return self.mapping[t]


@register_trainer("classification", "efficientnet", "classifier")
class EfficientNetTrainer(BaseTrainer):

    def prepare_data(self) -> Any:
        from ..data import iter_labelme_dir, split_samples
        from ..data.writers import write_imagefolder

        samples = iter_labelme_dir(self.cfg.dataset.root)
        by_split = split_samples(samples, self.cfg.dataset.split)
        root = write_imagefolder(by_split, self.names, self.out_dir / "imagefolder")
        return root

    def _model(self, num_classes: int):
        import timm
        size = self.cfg.arch_variant.get("size", "b0")
        name = self.cfg.arch_variant.get("timm_name", f"efficientnet_{size}")
        pretrained = self.cfg.arch_variant.get("pretrained", "imagenet") != "none"
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    def train(self, data: Any) -> Path:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

        t = self.cfg.train
        dev = f"cuda:{t.gpus[0]}" if (t.gpus and torch.cuda.is_available()) else "cpu"
        H, W = self.cfg.export.input.h, self.cfg.export.input.w
        # timm 기본 ImageNet 정규화(VSC imagenet_std:true 와 정합)
        tf = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds = datasets.ImageFolder(str(Path(data) / "train"), transform=tf)
        # ⚠ ImageFolder 는 디렉터리명을 알파벳순 인덱싱 → names 순서로 재매핑해야
        #   logit 채널 i == names[i] == manifest.label_map[i] 성립(클래스 뒤바뀜 방지).
        name_to_idx = {n: i for i, n in enumerate(self.names)}
        unknown = [c for c in train_ds.classes if c not in name_to_idx]
        if unknown:
            raise ValueError(f"학습 데이터에 dataset.names 에 없는 클래스: {unknown}")
        idx_to_name = {i: c for c, i in train_ds.class_to_idx.items()}
        train_ds.target_transform = _RemapToNames(
            {i: name_to_idx[idx_to_name[i]] for i in idx_to_name})
        loader = DataLoader(train_ds, batch_size=t.batch, shuffle=True,
                            num_workers=t.data_module.num_workers)

        # head 는 항상 '선언된 전체 클래스 수' → 채널 순서=names, NC==len(names)
        # (train↔export 동일 크기라 state_dict 로드 mismatch 불가; 0-sample 클래스도 채널 점유)
        model = self._model(len(self.names)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=t.optimizer.lr,
                                weight_decay=t.optimizer.weight_decay)
        crit = nn.CrossEntropyLoss()
        model.train()
        loss = torch.tensor(0.0)
        for ep in range(t.epochs):
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
            print(f"[effnet] epoch {ep+1}/{t.epochs} loss={float(loss):.4f}")
        ckpt = self.out_dir / "best.pt"
        torch.save({"state_dict": model.state_dict(), "classes": list(self.names),
                    "arch_variant": dict(self.cfg.arch_variant),
                    "num_classes": len(self.names)}, ckpt)
        return ckpt

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        import torch

        from ..export import VscExporter, introspect_output_shape

        e = self.cfg.export
        sd = torch.load(str(ckpt), map_location="cpu")
        ck_nc = sd.get("num_classes", len(self.names)) if isinstance(sd, dict) else len(self.names)
        if ck_nc != len(self.names):   # 학습/export config 클래스 수 불일치 → fail-fast
            raise ValueError(f"ckpt num_classes({ck_nc}) != dataset.names({len(self.names)})")
        from torch import nn
        from torch.nn import functional as F

        base = self._model(len(self.names))
        base.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        base.eval()

        class _ClsExport(nn.Module):   # softmax 베이킹(참조 코어 동형) → 출력=확률[0,1]
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return F.softmax(self.m(x), dim=1)

        model = _ClsExport(base).eval()
        dst = self.out_dir / "model.onnx"
        dummy = torch.randn(1, e.input.c, e.input.h, e.input.w)
        dyn = {e.io_names.input: {0: "B"}, e.io_names.output: {0: "B"}} if e.dynamic_axes else None
        torch.onnx.export(model, dummy, str(dst), opset_version=e.opset,
                          input_names=[e.io_names.input], output_names=[e.io_names.output],
                          dynamic_axes=dyn)   # ★ softmax BAKED → 출력 확률(실 manifest value_range[0,1] 정합)
        out_shape = introspect_output_shape(dst)
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
