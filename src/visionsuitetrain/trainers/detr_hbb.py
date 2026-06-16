"""DETR-family HBB 검출 어댑터 (ultralytics RT-DETR → 통일 [1,4+NC,A] 컨트랙트).

DETR 계열(D-FINE/RF-DETR/RT-DETR)은 모두 동일 채널-우선 [1,4+NC,A] export 컨트랙트(A=query)라
VSC yolov8_hbb 핸들러로 디코드한다. 학습 엔진은 현재 가용한 ultralytics RT-DETR 사용.
(정확한 D-FINE/RF-DETR 가중치는 upstream repo 필요 — 동일 어댑터 뒤 엔진 교체로 확장.)
RT-DETR export 가 query-major([1,A,4+NC])면 export 단계에서 channel-first 로 transpose 보정.

heavy lib(ultralytics)는 메서드 내부 lazy import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import register_trainer, resolve_devices
from .base import BaseTrainer


@register_trainer("hbbdetection", "rtdetr_hbb", "dfine_hbb", "rfdetr_hbb")
class DetrHbbTrainer(BaseTrainer):

    def prepare_data(self) -> Any:
        from ..data import iter_labelme_dir, validate_samples, split_samples, summarize
        from ..data.writers import write_yolo

        samples = iter_labelme_dir(self.cfg.dataset.root)
        print(summarize(validate_samples(samples, self.names, self.task)))
        by_split = split_samples(samples, self.cfg.dataset.split, self.cfg.run.seed)
        return write_yolo(by_split, self.names, self.out_dir / "yolo_cache")

    def train(self, data: Any) -> Path:
        from ultralytics import RTDETR

        size = self.cfg.arch_variant.get("size", "l")   # rtdetr-l | rtdetr-x
        t, o = self.cfg.train, self.cfg.train.optimizer
        m = RTDETR(f"rtdetr-{size}.pt")
        m.train(
            data=str(data), epochs=t.epochs, imgsz=t.imgsz, batch=t.batch,
            optimizer=("AdamW" if o.type == "adamw" else "SGD"),
            lr0=o.lr, weight_decay=o.weight_decay, amp=t.amp,
            device=resolve_devices(t.gpus), workers=t.data_module.num_workers,
            project=str(self.out_dir), name="train", exist_ok=True,
            seed=self.cfg.run.seed,
        )
        return self.out_dir / "train" / "weights" / "best.pt"

    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        from ultralytics import RTDETR

        from ..export import (VscExporter, rename_io, introspect_output_shape,
                              ensure_channel_first_det, scale_det_boxes_to_pixels)

        e = self.cfg.export
        m = RTDETR(str(ckpt))
        onnx_path = Path(m.export(format="onnx", opset=e.opset, dynamic=e.dynamic_axes,
                                  simplify=e.simplify, imgsz=[e.input.h, e.input.w]))
        dst = self.out_dir / "model.onnx"
        onnx_path.replace(dst)
        rename_io(dst, e.io_names.input, e.io_names.output)
        ensure_channel_first_det(dst, force=True)   # RT-DETR query-major([1,A,4+NC]) → [1,4+NC,A]
        # ⚠ RT-DETR 박스는 정규화[0,1] export(px 스케일은 그래프 밖) → 입력px 로 bake(VSC [0,H] 정합)
        scale_det_boxes_to_pixels(dst, e.input.w, e.input.h, len(self.names))
        out_shape = introspect_output_shape(dst)
        return VscExporter(self.cfg).write(dst, out_shape, weights_name="model.onnx")
