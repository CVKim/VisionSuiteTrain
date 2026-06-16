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


def _seg_output_axes(num_classes: int, activation: str, background_index: int) -> list:
    """SEG output 의 axes 메타 — talos manifest 와 동형(VSC 가 decode 활성화/배경 결정)."""
    return [
        {"type": "batch"},
        {"type": "features", "structure": [{
            "type": "class", "activation": activation,
            "background_index": background_index, "size": num_classes,
            "value_range": None,
        }]},
        {"type": "spatial", "dim": "height"},
        {"type": "spatial", "dim": "width"},
    ]


def build_manifest(cfg: TrainConfig, output_shape: list, *,
                   thresholds: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """resolved cfg + ONNX 출력 shape → manifest dict (talos export manifest 스키마 정합)."""
    e = cfg.export
    names = list(cfg.dataset.names)
    in_name = e.io_names.input
    out_name = e.io_names.output
    H, W, C = e.input.h, e.input.w, e.input.c
    is_det = cfg.task.endswith("detection")
    is_seg = cfg.task == "segmentation"

    env = _env()
    env["opset"] = e.opset

    # 전처리 steps — det:letterbox / cls,seg:resize (관행)
    resize_mode = e.preprocess_carry.get("resize_mode", "letterbox" if is_det else "resize")
    step = {
        "type": resize_mode,
        "height": H, "width": W,
        "interpolation": "bilinear", "antialias": False, "library": "cv2",
    }
    if resize_mode == "letterbox":
        step["padding_value"] = int(e.preprocess_carry.get("letterbox_pad_value", 0))

    out_spec: dict[str, Any] = {
        "shape": list(output_shape),
        "dtype": "FLOAT",
        "dynamic_axes": [] if not e.dynamic_axes else [0],
        "value_range": [0.0, float(max(H, W))] if is_det else [0.0, 1.0],
    }
    if is_seg:
        # 활성화/배경 메타 선언 → VSC 가 per-pixel softmax/argmax 수행(그래프엔 미주입)
        seg_act = "softmax" if e.seg_mode == "multi_channel" else "sigmoid"
        out_spec["axes"] = _seg_output_axes(len(names), seg_act, e.seg_background_class)

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
        "outputs": {out_name: out_spec},
        # talos manifest 정합: maskings 키 없음, rois 항상 존재, patch 항상 존재
        "preprocessing": {
            "patch": {"enabled": False, "height": None, "width": None},
            "rois": [],
            "steps": [step],
        },
        "task": {
            "task_type": cfg.task,
            "label_map": {i: n for i, n in enumerate(names)},
        },
    }
    # 검출: per-class threshold 항상 기록(real manifest 동형). 명시값 우선, 없으면 기본 conf.
    if is_det:
        default_conf = float(e.preprocess_carry.get("conf_threshold", 0.25))
        manifest["task"]["thresholds"] = dict(thresholds) if thresholds else {
            n: default_conf for n in names}
    elif thresholds:
        manifest["task"]["thresholds"] = dict(thresholds)
    # 세그멘테이션: 배경 채널 인덱스 명시(label_map 은 0..N-1 전체 유지)
    if is_seg:
        manifest["task"]["background_index"] = e.seg_background_class
    return manifest
