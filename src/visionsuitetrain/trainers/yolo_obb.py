"""ultralytics YOLO-OBB 어댑터 (→ VSC yolov8_obb 컨트랙트 [1, 4+NC+1, A], angle 맨끝 채널).

heavy lib(ultralytics)는 train/export 메서드 내부 lazy import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer, resolve_devices
from .base import BaseTrainer


@register_trainer("obbdetection", "yolov8_obb", "yolov11_obb")
class YoloObbTrainer(BaseTrainer):

    def prepare_data(self) -> Any:
        from ..data import iter_labelme_dir, validate_samples, split_samples, summarize
        from ..data.writers.yolo_obb import write_yolo_obb

        samples = iter_labelme_dir(self.cfg.dataset.root)
        print(summarize(validate_samples(samples, self.names, self.task)))
        by_split = split_samples(samples, self.cfg.dataset.split, self.cfg.run.seed)
        return write_yolo_obb(by_split, self.names, self.out_dir / "obb_cache")

    def train(self, data: Any) -> Path:
        from ultralytics import YOLO

        size = self.cfg.arch_variant.get("size", "m")
        t = self.cfg.train
        m = YOLO(f"yolov8{size}-obb.pt")
        m.train(
            data=str(data), epochs=t.epochs, imgsz=t.imgsz, batch=t.batch,
            optimizer=("AdamW" if t.optimizer == "adamw" else "SGD"),
            lr0=t.lr, weight_decay=t.weight_decay, amp=t.amp,
            device=resolve_devices(t.gpus), workers=t.workers,
            project=str(self.out_dir), name="train", exist_ok=True,
            seed=self.cfg.run.seed,
        )
        return self.out_dir / "train" / "weights" / "best.pt"

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        from ultralytics import YOLO

        from ..export import VscExporter, rename_io, introspect_output_shape

        e = self.cfg.export
        m = YOLO(str(ckpt))
        # nms=False → VSC 가 conf+NMS(rotated). OBB head export 는 [1,4+NC+1,A](angle 맨끝).
        onnx_path = Path(m.export(format="onnx", opset=e.opset, dynamic=e.dynamic_axes,
                                  simplify=True, nms=False, imgsz=[e.input.h, e.input.w]))
        dst = self.out_dir / "model.onnx"
        onnx_path.replace(dst)
        rename_io(dst, e.io_names.input, e.io_names.output)
        out_shape = introspect_output_shape(dst)
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
