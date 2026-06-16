"""ultralytics YOLOv8 detection 어댑터 (drop-in → VSC eYolov8Hbb 컨트랙트).

heavy lib(ultralytics)는 train/export 메서드 내부에서 lazy import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer, resolve_devices
from .base import BaseTrainer


@register_trainer("hbbdetection", "yolov8_hbb", "yolov7_hbb")
class YoloHbbTrainer(BaseTrainer):

    def prepare_data(self) -> Any:
        from ..data import iter_labelme_dir, validate_samples, split_samples, summarize
        from ..data.writers import write_yolo

        samples = iter_labelme_dir(self.cfg.dataset.root)
        print(summarize(validate_samples(samples, self.names, self.task)))
        by_split = split_samples(samples, self.cfg.dataset.split)
        data_yaml = write_yolo(by_split, self.names, self.out_dir / "yolo_cache")
        return data_yaml

    def train(self, data: Any) -> Path:
        from ultralytics import YOLO

        size = self.cfg.arch_variant.get("size", "m")
        t = self.cfg.train
        o, tr = t.optimizer, t.data_module.transforms
        m = YOLO(f"yolov8{size}.pt")
        m.train(
            data=str(data), epochs=t.epochs, imgsz=t.imgsz, batch=t.batch,
            optimizer=("AdamW" if o.type == "adamw" else "SGD"),
            lr0=o.lr, weight_decay=o.weight_decay, amp=t.amp,
            device=resolve_devices(t.gpus), workers=t.data_module.num_workers,
            hsv_h=tr.hsv_h, hsv_s=tr.hsv_s, hsv_v=tr.hsv_v, degrees=tr.degrees,
            translate=tr.translate, scale=tr.scale, shear=tr.shear,
            perspective=tr.perspective, flipud=tr.flipud, fliplr=tr.fliplr,
            mosaic=tr.mosaic, mixup=tr.mixup,
            project=str(self.out_dir), name="train", exist_ok=True,
            seed=self.cfg.run.seed,
        )
        return self.out_dir / "train" / "weights" / "best.pt"

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        from ultralytics import YOLO

        from ..export import VscExporter, rename_io, introspect_output_shape

        e = self.cfg.export
        m = YOLO(str(ckpt))
        # ⚠ nms=False 필수(VSC 가 conf+NMS 수행). simplify 로 그래프 정리.
        # imgsz=[h,w] 로 전달(스칼라면 ultralytics 가 w×w 정사각으로 붕괴 → 비정사각 깨짐).
        onnx_path = Path(m.export(format="onnx", opset=e.opset, dynamic=e.dynamic_axes,
                                  simplify=e.simplify, nms=False, imgsz=[e.input.h, e.input.w]))
        dst = self.out_dir / "model.onnx"
        onnx_path.replace(dst)
        # io 이름을 VSC 컨트랙트(data/output)로 고정 후 출력 shape introspect([1,4+NC,A]).
        rename_io(dst, e.io_names.input, e.io_names.output)
        out_shape = introspect_output_shape(dst)
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
