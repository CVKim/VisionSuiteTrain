"""timm EfficientNet classification 어댑터 ([B,NC] logit → VSC cls 컨트랙트).

heavy lib(torch/timm/torchvision)는 메서드 내부 lazy import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer
from .base import BaseTrainer


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
        loader = DataLoader(train_ds, batch_size=t.batch, shuffle=True, num_workers=t.workers)

        model = self._model(len(train_ds.classes)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
        crit = nn.CrossEntropyLoss()
        model.train()
        for ep in range(t.epochs):
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad()
                loss = crit(model(x), y)
                loss.backward()
                opt.step()
            print(f"[effnet] epoch {ep+1}/{t.epochs} loss={loss.item():.4f}")
        ckpt = self.out_dir / "best.pt"
        torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, ckpt)
        return ckpt

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        import torch

        from ..export import VscExporter, introspect_output_shape

        e = self.cfg.export
        model = self._model(len(self.names))
        sd = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        model.eval()
        dst = self.out_dir / "model.onnx"
        dummy = torch.randn(1, e.input.c, e.input.h, e.input.w)
        dyn = {e.io_names.input: {0: "B"}, e.io_names.output: {0: "B"}} if e.dynamic_axes else None
        torch.onnx.export(model, dummy, str(dst), opset_version=e.opset,
                          input_names=[e.io_names.input], output_names=[e.io_names.output],
                          dynamic_axes=dyn)   # ⚠ head 에 softmax 없음(logit export)
        out_shape = introspect_output_shape(dst)
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
