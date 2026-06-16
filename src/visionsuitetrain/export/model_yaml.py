"""model.yaml 빌더 — VSC ModelConfig 매핑(runtime/preprocess/postprocess.decision)."""
from __future__ import annotations

from typing import Any, Optional

from ..config.canonical import model_type_of
from ..config.schema import TrainConfig


def build_model_yaml(cfg: TrainConfig, *, weights: str = "",
                     nms_conf_vector: Optional[list[float]] = None) -> dict[str, Any]:
    e = cfg.export
    names = list(cfg.dataset.names)
    mtype = model_type_of(cfg.arch)
    pc = e.preprocess_carry

    model: dict[str, Any] = {
        "id": cfg.run.name,
        "name": cfg.run.name,
        "type": mtype,
        "weights": weights,
        "trt_cache": "",
        "input": {"w": e.input.w, "h": e.input.h, "c": e.input.c},
        "input_tensors": [e.io_names.input],
        "output_tensors": [e.io_names.output],
    }
    runtime = {
        "backend": e.backend,
        "gpu_idx": cfg.train.gpus[0] if cfg.train.gpus else 0,
        "fp16": e.fp16,
        "instance_count": 1,
        "on_memory": True,
        "warmup": 1,
    }
    preprocess = {
        "normalize": bool(pc.get("normalize", True)),
        "imagenet_std": bool(pc.get("imagenet_std", False)),
        "resize": {"w": e.input.w, "h": e.input.h, "mode": "bilinear"},
        "letterbox": pc.get("resize_mode", "letterbox") == "letterbox",
        "letterbox_pad_value": int(pc.get("letterbox_pad_value", 0)),
        "rgb": bool(pc.get("rgb", True)),
        "channel_first": True,
    }
    # global_std 우회(cls 변종 mean/std ≠ ImageNet)
    if pc.get("global_std_mean") and pc.get("global_std_std"):
        preprocess["global_std"] = True
        preprocess["global_std_mean"] = list(pc["global_std_mean"])
        preprocess["global_std_std"] = list(pc["global_std_std"])

    patch_cfg = pc.get("patch")
    inf_patch: dict[str, Any] = {"enable": False}
    if patch_cfg:                      # HBB 대형 이미지 패치 타일링 → VSC inference.patch
        inf_patch = {"enable": True,
                     "w": int(patch_cfg.get("width", e.input.w)),
                     "h": int(patch_cfg.get("height", e.input.h)),
                     "overlap": float(patch_cfg.get("overlap", 0.5)),
                     "assemble": "nms_global"}
        if pc.get("border_suppression") is not None:
            inf_patch["border_suppress"] = float(pc["border_suppression"])
    inference = {"batch_size": 1, "patch": inf_patch, "tta": {"enable": False}}

    postprocess: dict[str, Any] = {"type": f"{mtype}_decode"}
    if cfg.task in ("hbbdetection", "obbdetection"):
        ncv = nms_conf_vector or [0.25] * len(names)
        if len(ncv) != len(names):     # decision 벡터는 names 수에 정합해야(VSC per-class)
            raise ValueError(f"nms_conf_vector 길이({len(ncv)}) != names({len(names)})")
        postprocess.update({
            "conf_thres": 0.25, "iou_thres": 0.45, "max_det": 300,
            "classes": names,
            "decision": {
                "names": names,
                "nms_conf_vector": list(ncv),
                "nms_iou_th": 0.45,
                "width_th": [0] * len(names),
                "height_th": [0] * len(names),
                "judge_by_feret": [0] * len(names),
            },
        })
    elif cfg.task == "classification":
        postprocess.update({"classes": names, "cls_activation": e.cls_activation})
    elif cfg.task == "segmentation":
        postprocess.update({"classes": names, "seg_background_class": e.seg_background_class})
        if e.seg_mode == "one_channel":
            postprocess["confidence_threshold_one_channel_seg"] = [0.5] * len(names)
    elif cfg.task == "ocr":                 # VSC OCR 인식 디코드(CTC argmax+collapse 는 호스트)
        postprocess.update({
            "classes": names,
            "charset": "".join(names),
            "blank_index": len(names),      # CTC blank = NC(마지막 채널)
            "decode": "ctc",
        })
    elif cfg.task == "anomaly_detection":   # VSC foundation_anomaly 디코드 메타
        # ★ norm_min/max 를 출력 범위[0,1]로 명시(미지정 시 VSC 기본 [1,255]→heatmap 0 붕괴)
        preprocess["normalize"] = False     # 그래프에서 /255 bake → VSC 재정규화 금지
        postprocess.update({
            "classes": names,
            "anomaly_norm_min": float(pc.get("anomaly_norm_min", 0.0)),
            "anomaly_norm_max": float(pc.get("anomaly_norm_max", 1.0)),
            "anomaly_sigmoid": False,        # 이미 [0,1] heatmap (그래프에서 clamp+scale)
            "anomaly_score_mode": pc.get("anomaly_score_mode", "top_k_mean"),
            "anomaly_top_k": int(pc.get("anomaly_top_k", 16)),
            "anomaly_bin_threshold": int(pc.get("anomaly_bin_threshold", 128)),
            "anomaly_min_area": int(pc.get("anomaly_min_area", 4)),
            "decision": {"names": names},
        })

    return {"model": model, "runtime": runtime, "preprocess": preprocess,
            "inference": inference, "postprocess": postprocess}
