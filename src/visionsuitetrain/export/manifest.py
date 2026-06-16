"""<NAME>_manifest.yaml 빌더 — VSC I/O 계약(environment/inputs/outputs/preprocessing/task).

키는 VSC manifest 파서 기준(사용자 공유 manifest 2건과 동일 on-disk 스키마).
"""
from __future__ import annotations

import sys
from typing import Any, Optional

from ..config.schema import TrainConfig


def _env() -> dict[str, Any]:
    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        import torch  # noqa
        cuda = getattr(__import__("torch").version, "cuda", None)
    except Exception:
        cuda = None
    try:
        import onnx  # noqa
        onnx_ver = __import__("onnx").__version__
    except Exception:
        onnx_ver = None
    return {"cuda": cuda, "onnx": onnx_ver, "opset": None, "python": py}


def build_manifest(cfg: TrainConfig, output_shape: list, *,
                   thresholds: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """resolved cfg + ONNX 출력 shape → manifest dict."""
    e = cfg.export
    names = list(cfg.dataset.names)
    in_name = e.io_names.input
    out_name = e.io_names.output
    H, W, C = e.input.h, e.input.w, e.input.c

    env = _env()
    env["opset"] = e.opset

    # 전처리 steps — det:letterbox / cls,seg:resize (관행)
    resize_mode = e.preprocess_carry.get("resize_mode",
                                         "letterbox" if cfg.task.endswith("detection") else "resize")
    step = {
        "type": resize_mode,
        "height": H, "width": W,
        "interpolation": "bilinear", "antialias": False, "library": "cv2",
    }
    if resize_mode == "letterbox":
        step["padding_value"] = int(e.preprocess_carry.get("letterbox_pad_value", 0))

    manifest: dict[str, Any] = {
        "environment": env,
        "inputs": {
            in_name: {
                "shape": [1, C, H, W],
                "dtype": e.input.dtype,
                "dynamic_axes": [] if not e.dynamic_axes else [0],
                "value_range": list(e.input.value_range),
            }
        },
        "outputs": {
            out_name: {
                "shape": list(output_shape),
                "dtype": "FLOAT",
                "dynamic_axes": [] if not e.dynamic_axes else [0],
                "value_range": [0.0, float(H)] if cfg.task.endswith("detection") else [0.0, 1.0],
            }
        },
        "preprocessing": {
            "rois": [],
            "maskings": [],
            "patch": {"enabled": False, "height": None, "width": None},
            "steps": [step],
        },
        "task": {
            "task_type": cfg.task,
            "label_map": {i: n for i, n in enumerate(names)},
        },
    }
    if thresholds:
        manifest["task"]["thresholds"] = dict(thresholds)
    return manifest
