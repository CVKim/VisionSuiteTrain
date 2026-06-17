"""이상탐지 어댑터 (멀티스케일 feature-reconstruction → per-pixel 오차 heatmap [1,1,H,W] → VSC foundation_anomaly).

비지도(정상 이미지만). 단일 스케일 pixel/feature AE 는 미세 결함과 대형 결함을 동시에 못 잡는다
→ **frozen ImageNet 백본(wide_resnet50_2)의 layer1/2/3 다중 스케일 특징**을 공유 병목(OCBE)으로
압축한 뒤 디코더가 각 스케일 정상 특징을 복원하도록 학습(reverse-distillation 계열, RD4AD).
결함은 특징 복원오차(코사인 거리)가 커지며, 얕은 고해상도 층은 미세 결함(오염·점)·깊은 층은
대형 결함(파손)을 포착한다. 추론 forward:
  x/255 → imagenet 정규화 bake → enc(frozen) {f1,f2,f3} → OCBE 융합 z → dec {r1,r2,r3}
  → Σ_k upsample(1 - cos(f_k, r_k)) → Gaussian 평활 → clamp(/ms)[0,1].
입력은 RGB(c=3). heavy lib(torch/torchvision)는 메서드 내부 lazy import — nn.Module 정의도 내부.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer, build_optimizer
from .base import BaseTrainer

# ── 알고리즘 내부 상수(JSON 설정 아님) ──────────────────────────────────────
_AD_BACKBONE = "wide_resnet50_2"   # frozen ImageNet 백본(캐시됨, TRT 친화 CNN). resnet18 폴백 가능.
_AD_FUSE_DIM = 256                 # OCBE 융합 병목 채널폭(공간 H/16 병목과 함께 identity 단축 차단)
_AD_SMOOTH_SIGMA = 4.0             # 이상맵 Gaussian 평활 σ (MVTec 관례)
_AD_SMOOTH_KSIZE = 33              # 평활 커널 크기(≈ 2·round(4σ)+1)
_BACKBONE_CHANNELS = {"wide_resnet50_2": (256, 512, 1024), "resnet18": (64, 128, 256)}


def _gaussian_kernel(ksize: int, sigma: float):
    """2D 정규화 Gaussian 커널 [1,1,k,k] (depthwise conv 평활용)."""
    import torch
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k = torch.outer(g, g)
    return k.view(1, 1, ksize, ksize)


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

    def _model(self, backbone: str = _AD_BACKBONE, fuse_dim: int = _AD_FUSE_DIM):
        import torch
        import torchvision as tv
        from torch import nn
        from torch.nn import functional as F

        chs = _BACKBONE_CHANNELS[backbone]

        class _MSFeatRecon(nn.Module):
            """다중 스케일 특징 복원(OCBE 병목). enc 고정, proj/fuse/dec 만 학습."""

            def __init__(self):
                super().__init__()
                if backbone == "wide_resnet50_2":
                    bb = tv.models.wide_resnet50_2(
                        weights=tv.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
                else:
                    bb = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
                self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
                self.layer1, self.layer2, self.layer3 = bb.layer1, bb.layer2, bb.layer3
                for p in self.parameters():
                    p.requires_grad_(False)                       # 백본 전체 freeze
                self.D = fuse_dim
                # 각 스케일 → D 채널 1x1 투영 후 H/16 으로 정렬·융합(OCBE)
                self.proj = nn.ModuleList([nn.Conv2d(c, self.D, 1) for c in chs])
                self.fuse = nn.Sequential(
                    nn.Conv2d(self.D * 3, self.D, 3, 1, 1), nn.BatchNorm2d(self.D), nn.ReLU(inplace=True),
                    nn.Conv2d(self.D, self.D, 3, 1, 1), nn.BatchNorm2d(self.D), nn.ReLU(inplace=True))
                # 융합 코드 z(H/16) → 각 스케일 특징 복원 (layer1 2업, layer2 1업, layer3 0업)
                self.dec = nn.ModuleList([self._dec(chs[0], 2), self._dec(chs[1], 1), self._dec(chs[2], 0)])
                self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def _dec(self, out_c, ups):
                layers = []
                cin = self.D
                for _ in range(ups):
                    layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                               nn.Conv2d(cin, self.D, 3, 1, 1), nn.BatchNorm2d(self.D), nn.ReLU(inplace=True)]
                    cin = self.D
                layers += [nn.Conv2d(cin, out_c, 3, 1, 1)]
                return nn.Sequential(*layers)

            def feats(self, x01):                # x01 in [0,1] → imagenet 정규화 → frozen enc
                x = (x01 - self.mean) / self.std
                x = self.stem(x)
                f1 = self.layer1(x)
                f2 = self.layer2(f1)
                f3 = self.layer3(f2)
                return [f1, f2, f3]

            def reconstruct(self, feats):
                h3 = (int(feats[2].shape[-2]), int(feats[2].shape[-1]))   # H/16 (trace 시 상수화)
                proj = []
                for i in range(3):
                    p = self.proj[i](feats[i])
                    if p.shape[-2:] != feats[2].shape[-2:]:
                        p = F.interpolate(p, size=h3, mode="bilinear", align_corners=False)
                    proj.append(p)
                z = self.fuse(torch.cat(proj, dim=1))
                return [self.dec[i](z) for i in range(3)]

            @staticmethod
            def _cos_dist(f, r, keepdim):        # 1 - cosine (채널축) — norm 은 TRT 안전하게 수동 분해
                num = (f * r).sum(1, keepdim=keepdim)
                fn = (f * f).sum(1, keepdim=keepdim).clamp_min(1e-12).sqrt()
                rn = (r * r).sum(1, keepdim=keepdim).clamp_min(1e-12).sqrt()
                return 1.0 - num / (fn * rn + 1e-6)

            def recon_loss(self, x01):           # 학습 손실: 스케일별 코사인 거리 평균 합
                feats = self.feats(x01)
                recs = self.reconstruct(feats)
                loss = x01.new_zeros(())
                for f, r in zip(feats, recs):
                    loss = loss + self._cos_dist(f, r, keepdim=False).mean()
                return loss

            def anomaly_map(self, x01, H, W):    # Σ_k upsample(1 - cos) → [B,1,H,W]
                feats = self.feats(x01)
                recs = self.reconstruct(feats)
                amap = None
                for f, r in zip(feats, recs):
                    a = self._cos_dist(f, r, keepdim=True)
                    a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
                    amap = a if amap is None else amap + a
                return amap

        return _MSFeatRecon()

    @staticmethod
    def _freeze_encoder(model):
        for m in (model.stem, model.layer1, model.layer2, model.layer3):
            m.eval()                              # frozen 백본 — BN running stat 고정(train() 호출 후에도)
            for p in m.parameters():
                p.requires_grad_(False)

    def _smoother(self, dev):
        import torch  # noqa: F401
        from torch.nn import functional as F
        k = _gaussian_kernel(_AD_SMOOTH_KSIZE, _AD_SMOOTH_SIGMA).to(dev)
        pad = _AD_SMOOTH_KSIZE // 2
        return lambda a: F.conv2d(a, k, padding=pad)

    def _infer_module(self, model, max_score: float):
        """추론/ONNX export 공용 모듈(VSC 0..255 RGB → [B,1,H,W] heatmap[0,1]). train 의 ms 와 동일 평활."""
        import torch
        from torch import nn
        from torch.nn import functional as F

        H, W = self.cfg.export.input.h, self.cfg.export.input.w
        kernel = _gaussian_kernel(_AD_SMOOTH_KSIZE, _AD_SMOOTH_SIGMA)
        pad = _AD_SMOOTH_KSIZE // 2

        class _ADExport(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = model
                self.register_buffer("gk", kernel)
                self.register_buffer("ms", torch.tensor(max(max_score, 1e-6)))
                self.pad, self.H, self.W = pad, H, W

            def forward(self, x):                 # x: VSC 0..255 RGB [B,3,H,W]
                a = self.m.anomaly_map(x / 255.0, self.H, self.W)   # [B,1,H,W] 코사인거리 합
                a = F.conv2d(a, self.gk, padding=self.pad)          # Gaussian 평활
                return torch.clamp(a / self.ms, 0.0, 1.0)

        return _ADExport().eval()

    def train(self, data: Any) -> Path:
        import cv2
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset

        t, e = self.cfg.train, self.cfg.export
        if e.input.c != 3:
            raise ValueError("foundation_anomaly(multi-scale feature-recon)는 RGB(c=3) 입력 필요")
        dev = f"cuda:{t.gpus[0]}" if (t.gpus and torch.cuda.is_available()) else "cpu"
        H, W = e.input.h, e.input.w
        rows = list(data)

        class NormDS(Dataset):
            def __init__(self, aug): self.aug = aug
            def __len__(self): return len(rows)
            def __getitem__(self, i):
                img = cv2.cvtColor(cv2.imread(rows[i]), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                if self.aug:                       # dihedral 증강(무손실, 정상 manifold 확장)
                    if np.random.rand() < 0.5: img = img[:, ::-1]
                    if np.random.rand() < 0.5: img = img[::-1, :]
                    img = np.rot90(img, k=int(np.random.randint(0, 4)))
                img = np.ascontiguousarray(img)
                return torch.from_numpy((img.astype(np.float32) / 255.0).transpose(2, 0, 1))

        nw = t.data_module.num_workers
        loader = DataLoader(NormDS(aug=True), batch_size=t.batch, shuffle=True, num_workers=nw, drop_last=False)
        model = self._model().to(dev)
        model.train()
        self._freeze_encoder(model)
        params = [p for p in model.parameters() if p.requires_grad]   # proj/fuse/dec 만
        opt = build_optimizer(params, t.optimizer)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, t.epochs))
        for ep in range(t.epochs):
            run_loss, n = 0.0, 0
            for x in loader:
                x = x.to(dev)
                loss = model.recon_loss(x)
                opt.zero_grad(); loss.backward(); opt.step()
                run_loss += float(loss) * x.size(0); n += x.size(0)
            sched.step()
            if (ep + 1) % 10 == 0 or ep == 0 or ep == t.epochs - 1:
                print(f"[ad] epoch {ep+1}/{t.epochs} loss={run_loss/max(1,n):.5f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        # 정규화 상수 ms: 정상셋의 (평활)이상맵 최대값 → 정상≈경계, 결함은 초과→clamp[0,1]
        model.eval()
        eval_loader = DataLoader(NormDS(aug=False), batch_size=t.batch, shuffle=False, num_workers=nw)
        smooth = self._smoother(dev)
        max_score = 1e-6
        with torch.no_grad():
            for x in eval_loader:
                a = smooth(model.anomaly_map(x.to(dev), H, W))
                max_score = max(max_score, float(a.max()))
        print(f"[ad] normalize max_score={max_score:.5f}", flush=True)
        ckpt = self.out_dir / "best.pt"
        torch.save({"state_dict": model.state_dict(), "max_score": max_score,
                    "num_classes": len(self.names), "backbone": _AD_BACKBONE,
                    "fuse_dim": _AD_FUSE_DIM}, ckpt)
        return ckpt

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        import torch

        from ..export import VscExporter, introspect_output_shape

        e = self.cfg.export
        sd = torch.load(str(ckpt), map_location="cpu")
        bb = sd.get("backbone", _AD_BACKBONE) if isinstance(sd, dict) else _AD_BACKBONE
        fd = sd.get("fuse_dim", _AD_FUSE_DIM) if isinstance(sd, dict) else _AD_FUSE_DIM
        model = self._model(backbone=bb, fuse_dim=fd)
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        model.eval()
        max_score = float(sd.get("max_score", 1.0)) if isinstance(sd, dict) else 1.0

        exp = self._infer_module(model, max_score)
        dst = self.out_dir / "model.onnx"
        dummy = torch.randn(1, e.input.c, e.input.h, e.input.w)
        dyn = ({e.io_names.input: {0: "B"}, e.io_names.output: {0: "B"}}
               if e.dynamic_axes else None)
        torch.onnx.export(exp, dummy, str(dst), opset_version=e.opset,
                          input_names=[e.io_names.input], output_names=[e.io_names.output],
                          dynamic_axes=dyn)
        out_shape = introspect_output_shape(dst)      # [1, 1, H, W]
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
