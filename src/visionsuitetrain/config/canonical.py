"""VisionSuiteCore canonical 문자열 룩업.

학습 config 의 task/arch enum 을 VSC 의 model.type / manifest task_type 과 동일하게 채택해
export 무변환 정합을 보장한다. (근거: VSC TV_REGISTER_MODEL_TYPES, ModelConfig, manifest 스키마)
"""
from __future__ import annotations

# manifest task.task_type (VSC TvTaskType 와 1:1)
TASK_TYPES = {
    "hbbdetection",
    "obbdetection",
    "classification",
    "segmentation",
    "ocr",
    "anomaly",
}

# arch alias → 정규 arch (model.type)
ARCH_ALIASES = {
    "classifier": "efficientnet",
    "yolov11_obb": "yolov8_obb",
}


def canonical_arch(arch: str) -> str:
    return ARCH_ALIASES.get(arch, arch)


# 정규 arch → VSC model.type. postprocess.type 은 f"{model_type}_decode".
ARCH_TO_MODEL_TYPE = {
    "yolov8_hbb": "yolov8_hbb",
    "yolov7_hbb": "yolov7_hbb",
    "yolov8_obb": "yolov8_obb",
    "efficientnet": "efficientnet",
    "deeplab3pp": "deeplab3pp",
    "deeplab3pp_one_channel": "deeplab3pp_one_channel",
    "dfine_hbb": "dfine_hbb",       # 2차 — VSC 신규 핸들러 선결
    "rfdetr_hbb": "rfdetr_hbb",     # 2차
}

# 정규 arch → 허용 task (config 정합 검증용)
ARCH_TASK = {
    "yolov8_hbb": "hbbdetection",
    "yolov7_hbb": "hbbdetection",
    "yolov8_obb": "obbdetection",
    "efficientnet": "classification",
    "deeplab3pp": "segmentation",
    "deeplab3pp_one_channel": "segmentation",
    "dfine_hbb": "hbbdetection",
    "rfdetr_hbb": "hbbdetection",
}

# 1차 범위에서 어댑터가 구현된 arch
IMPLEMENTED = {"yolov8_hbb", "efficientnet", "deeplab3pp"}


def model_type_of(arch: str) -> str:
    a = canonical_arch(arch)
    if a not in ARCH_TO_MODEL_TYPE:
        raise KeyError(f"unknown arch '{arch}' (canonical='{a}'). 등록: {sorted(ARCH_TO_MODEL_TYPE)}")
    return ARCH_TO_MODEL_TYPE[a]
