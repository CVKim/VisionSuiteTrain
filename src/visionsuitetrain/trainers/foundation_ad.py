"""이상탐지 어댑터 (1차: 재구성 AE → per-pixel 오차 heatmap [1,1,H,W] → VSC foundation_anomaly).

비지도(정상 이미지만 학습). export 모델 forward = |x/255 - recon| 채널평균 → max_score 로 정규화
→ clamp[0,1] → [1,1,H,W]. (foundation/DINOv2 변형은 차후 — 출력 컨트랙트 동일하므로 _ae 교체만.)
heavy lib(torch)는 메서드 내부 lazy import — nn.Module 정의도 메서드 내부.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer
from .base import BaseTrainer


@register_trainer("anomaly_detection", "foundation_anomaly")
class FoundationAnomalyTrainer(BaseTrainer):

    def prepare_data(self) -> Any:
        from ..data import iter_labelme_dir, split_samples
        samples = iter_labelme_dir(self.cfg.dataset.root)
        by_split = split_samples(samples, self.cfg.dataset.split, self.cfg.run.seed)
        # 비지도 — 정상 이미지 경로만 사용(라벨 무관)
        return [s.image_path for s in by_split.get("train", []) if Path(s.image_path).exists()]

    def _ae(self, c: int):
        from torch import nn
        return nn.Sequential(
            nn.Conv2d(c, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, c, 4, 2, 1), nn.Sigmoid(),
        )

    def train(self, data: Any) -> Path:
        import cv2
        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset

        t = self.cfg.train
        e = self.cfg.export
        dev = f"cuda:{t.gpus[0]}" if (t.gpus and torch.cuda.is_available()) else "cpu"
        H, W, C = e.input.h, e.input.w, e.input.c
        rows = list(data)
        if not rows:
            raise ValueError("이상탐지 학습 이미지 0장 (정상 이미지 필요)")
        from torch.nn import functional as F

        class NormDS(Dataset):
            def __len__(self): return max(1, len(rows))
            def __getitem__(self, i):
                img = cv2.imread(rows[i])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if C == 3 else \
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                x = (img.astype(np.float32) / 255.0).reshape(H, W, C).transpose(2, 0, 1)
                return torch.from_numpy(x)

        # NormDS 가 내부 클래스 → Windows spawn 피클 회피 위해 num_workers=0 고정
        loader = DataLoader(NormDS(), batch_size=t.batch, shuffle=True, num_workers=0)
        ae = self._ae(C).to(dev)
        opt = torch.optim.AdamW(ae.parameters(), lr=t.optimizer.lr,
                                weight_decay=t.optimizer.weight_decay)
        crit = nn.MSELoss()

        def _recon(z):   # AE 출력이 입력과 다르면(H/W 8배수 아님) bilinear 강제 정렬
            y = ae(z)
            return y if y.shape[-2:] == z.shape[-2:] else \
                F.interpolate(y, size=z.shape[-2:], mode="bilinear", align_corners=False)

        ae.train()
        loss = torch.tensor(0.0)
        for ep in range(t.epochs):
            for x in loader:
                x = x.to(dev)
                opt.zero_grad()
                loss = crit(_recon(x), x)
                loss.backward()
                opt.step()
            print(f"[ad] epoch {ep+1}/{t.epochs} loss={float(loss):.5f}")
        # 정규화 상수: 정상셋 per-pixel 최대 재구성오차(없으면 1.0)
        ae.eval()
        max_score = 1e-6
        with torch.no_grad():
            for x in loader:
                x = x.to(dev)
                err = (x - _recon(x)).abs().mean(dim=1)
                max_score = max(max_score, float(err.max()))
        ckpt = self.out_dir / "best.pt"
        torch.save({"state_dict": ae.state_dict(), "max_score": max_score,
                    "num_classes": len(self.names)}, ckpt)
        return ckpt

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        import torch
        from torch import nn

        from ..export import VscExporter, introspect_output_shape

        e = self.cfg.export
        sd = torch.load(str(ckpt), map_location="cpu")
        ae = self._ae(e.input.c)
        ae.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        ae.eval()
        max_score = float(sd.get("max_score", 1.0)) if isinstance(sd, dict) else 1.0

        from torch.nn import functional as F

        class _ADExport(nn.Module):   # forward = recon-error heatmap [B,1,H,W] in [0,1]
            def __init__(self, ae_, ms: float):
                super().__init__()
                self.ae = ae_
                self.register_buffer("ms", torch.tensor(max(ms, 1e-6)))

            def forward(self, x):
                x01 = x / 255.0                       # VSC 가 0..255 공급 → 내부 /255 bake
                rec = self.ae(x01)
                if rec.shape[-2:] != x01.shape[-2:]:   # 입력 H/W 8배수 아니면 정렬
                    rec = F.interpolate(rec, size=x01.shape[-2:], mode="bilinear",
                                        align_corners=False)
                err = (x01 - rec).abs().mean(dim=1, keepdim=True)
                return torch.clamp(err / self.ms, 0.0, 1.0)

        model = _ADExport(ae, max_score).eval()
        dst = self.out_dir / "model.onnx"
        dummy = torch.randn(1, e.input.c, e.input.h, e.input.w)
        dyn = ({e.io_names.input: {0: "B"}, e.io_names.output: {0: "B"}}
               if e.dynamic_axes else None)
        torch.onnx.export(model, dummy, str(dst), opset_version=e.opset,
                          input_names=[e.io_names.input], output_names=[e.io_names.output],
                          dynamic_axes=dyn)
        out_shape = introspect_output_shape(dst)      # [1, 1, H, W]
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
