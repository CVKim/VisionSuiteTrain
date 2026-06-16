"""이상탐지 어댑터 (feature-reconstruction → per-pixel 오차 heatmap [1,1,H,W] → VSC foundation_anomaly).

비지도(정상 이미지만). pixel-AE 는 결함도 복원해 분리력이 약함 → **frozen ImageNet 백본(resnet18)
특징 공간**에서 디코더가 정상 특징을 복원하도록 학습, 결함은 특징 복원오차가 커진다(dinomaly 계열).
forward: x/255 → imagenet 정규화 bake → enc(frozen) → dec → ||f-dec||² 채널평균 → upsample → clamp[0,1].
입력은 RGB(c=3). (DINOv2 백본/메모리뱅크 등은 동일 컨트랙트 뒤 엔진 교체로 확장.)
heavy lib(torch/torchvision)는 메서드 내부 lazy import — nn.Module 정의도 메서드 내부.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer, build_optimizer
from .base import BaseTrainer


@register_trainer("anomaly_detection", "foundation_anomaly")
class FoundationAnomalyTrainer(BaseTrainer):

    def prepare_data(self) -> Any:
        # 비지도 — 정상 이미지만 직접 글롭(라벨 무관). MVTec-AD(<cat>/train/good) 자동 인식,
        # 그 외엔 dataset.root 재귀 글롭. (labelme dir 도 이미지 그대로 수집됨)
        from ..data.readers.images import iter_images, mvtec_normal_dir
        base = mvtec_normal_dir(self.cfg.dataset.root)
        rows = iter_images(base)
        if not rows:
            raise ValueError(f"이상탐지 정상 이미지 0장: {base} "
                             "(MVTec 은 dataset.root=<category>, 또는 정상 이미지 폴더)")
        print(f"[ad] normal images: {len(rows)} from {base}")
        return rows

    def _model(self):
        import torch
        import torchvision as tv
        from torch import nn

        class _FeatRecon(nn.Module):
            def __init__(self):
                super().__init__()
                bb = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
                self.enc = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                         bb.layer1, bb.layer2)        # 128ch @ H/8
                for p in self.enc.parameters():
                    p.requires_grad_(False)
                self.dec = nn.Sequential(
                    nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                    nn.Conv2d(256, 128, 3, 1, 1),
                )
                self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def feats(self, x01):                # x01 in [0,1] → imagenet 정규화 → frozen enc
                return self.enc((x01 - self.mean) / self.std)

            def forward(self, x01):              # 학습용: (정상특징, 복원특징)
                f = self.feats(x01)
                return f, self.dec(f)

        return _FeatRecon()

    def train(self, data: Any) -> Path:
        import cv2
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        t, e = self.cfg.train, self.cfg.export
        if e.input.c != 3:
            raise ValueError("foundation_anomaly(feature-recon)는 RGB(c=3) 입력 필요")
        dev = f"cuda:{t.gpus[0]}" if (t.gpus and torch.cuda.is_available()) else "cpu"
        H, W = e.input.h, e.input.w
        rows = list(data)

        class NormDS(Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self, i):
                img = cv2.cvtColor(cv2.imread(rows[i]), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                return torch.from_numpy((img.astype(np.float32) / 255.0).transpose(2, 0, 1))

        loader = DataLoader(NormDS(), batch_size=t.batch, shuffle=True, num_workers=0)
        model = self._model().to(dev)
        model.train()
        model.enc.eval()                          # frozen 백본 — BN running stat 고정
        opt = build_optimizer(model.dec.parameters(), t.optimizer)   # 디코더만 학습
        mse = nn.MSELoss()
        loss = torch.tensor(0.0)
        for ep in range(t.epochs):
            for x in loader:
                x = x.to(dev)
                f, rec = model(x)
                opt.zero_grad()
                loss = mse(rec, f.detach())       # 정상 특징 복원
                loss.backward()
                opt.step()
            print(f"[ad] epoch {ep+1}/{t.epochs} loss={float(loss):.5f}")
        # 정규화 상수: 정상셋 특징복원오차 분포의 상위(99퍼센타일 근사 = per-image max 의 max)
        model.eval()
        max_score = 1e-6
        with torch.no_grad():
            for x in loader:
                x = x.to(dev)
                f, rec = model(x)
                err = ((f - rec) ** 2).mean(dim=1)         # [B,h',w']
                max_score = max(max_score, float(err.max()))
        ckpt = self.out_dir / "best.pt"
        torch.save({"state_dict": model.state_dict(), "max_score": max_score,
                    "num_classes": len(self.names)}, ckpt)
        return ckpt

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        import torch
        from torch import nn
        from torch.nn import functional as F

        from ..export import VscExporter, introspect_output_shape

        e = self.cfg.export
        sd = torch.load(str(ckpt), map_location="cpu")
        model = self._model()
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        model.eval()
        max_score = float(sd.get("max_score", 1.0)) if isinstance(sd, dict) else 1.0
        H, W = e.input.h, e.input.w

        class _ADExport(nn.Module):   # forward = feature-recon-error heatmap [B,1,H,W] in [0,1]
            def __init__(self, m, ms: float):
                super().__init__()
                self.m = m
                self.register_buffer("ms", torch.tensor(max(ms, 1e-6)))

            def forward(self, x):
                f, rec = self.m(x / 255.0)                 # VSC 0..255 → /255 + (내부)imagenet 정규화
                err = ((f - rec) ** 2).mean(dim=1, keepdim=True)
                err = F.interpolate(err, size=(H, W), mode="bilinear", align_corners=False)
                return torch.clamp(err / self.ms, 0.0, 1.0)

        exp = _ADExport(model, max_score).eval()
        dst = self.out_dir / "model.onnx"
        dummy = torch.randn(1, e.input.c, e.input.h, e.input.w)
        dyn = ({e.io_names.input: {0: "B"}, e.io_names.output: {0: "B"}}
               if e.dynamic_axes else None)
        torch.onnx.export(exp, dummy, str(dst), opset_version=e.opset,
                          input_names=[e.io_names.input], output_names=[e.io_names.output],
                          dynamic_axes=dyn)
        out_shape = introspect_output_shape(dst)      # [1, 1, H, W]
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
