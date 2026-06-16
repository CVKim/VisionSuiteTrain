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
    "anomaly_detection",   # talos manifest task_type (VSC TV_TASK_ANOMALY)
}

# arch alias → 정규 arch (model.type)
ARCH_ALIASES = {
    "classifier": "efficientnet",
    "yolov11_obb": "yolov8_obb",
}


def canonical_arch(arch: str) -> str:
    return ARCH_ALIASES.get(arch, arch)


# 정규 arch → VSC model.type. postprocess.type 은 f"{model_type}_decode".
# ★ HBB 검출은 v7/v8/D-FINE/RF-DETR 가 모두 동일 채널-우선 [1,4+NC,A] 컨트랙트로 export
#   (eYolov8Hbb 통일, A=anchor 또는 query). 따라서 전부 동일 yolov8_hbb 핸들러로 디코드 →
#   신규 VSC 핸들러 불요. (실측: RF-DETR [1,27,300], D-FINE [1,16,300], value_range[0,H])
ARCH_TO_MODEL_TYPE = {
    "yolov8_hbb": "yolov8_hbb",
    "yolov7_hbb": "yolov7_hbb",
    "yolov8_obb": "yolov8_obb",
    "efficientnet": "efficientnet",
    "deeplab3pp": "deeplab3pp",
    "deeplab3pp_one_channel": "deeplab3pp_one_channel",
    "rtdetr_hbb": "yolov8_hbb",         # DETR-family(ultralytics RT-DETR) — [1,4+NC,A] 통일
    "dfine_hbb": "yolov8_hbb",          # DETR-family — 동일 컨트랙트(엔진=RT-DETR, 정확 가중치는 upstream)
    "rfdetr_hbb": "yolov8_hbb",         # DETR-family — A=쿼리, 디코드(conf+NMS) 동일
    "foundation_anomaly": "foundation_anomaly",   # [1,1,H,W] heatmap, scalar threshold
    "crnn_ctc": "parseq",               # OCR 인식 — [B,T,NC+1] CTC, VSC parseq 핸들러로 디코드
    "parseq": "parseq",
}

# 정규 arch → 허용 task (config 정합 검증용)
ARCH_TASK = {
    "yolov8_hbb": "hbbdetection",
    "yolov7_hbb": "hbbdetection",
    "yolov8_obb": "obbdetection",
    "efficientnet": "classification",
    "deeplab3pp": "segmentation",
    "deeplab3pp_one_channel": "segmentation",
    "rtdetr_hbb": "hbbdetection",
    "dfine_hbb": "hbbdetection",
    "rfdetr_hbb": "hbbdetection",
    "foundation_anomaly": "anomaly_detection",
    "crnn_ctc": "ocr",
    "parseq": "ocr",
}

# 어댑터가 실제 등록된 canonical arch (정보용; 실제 게이트는 registry/build_trainer)
IMPLEMENTED = {"yolov8_hbb", "yolov7_hbb", "yolov8_obb", "rtdetr_hbb", "dfine_hbb",
               "rfdetr_hbb", "efficientnet", "deeplab3pp", "deeplab3pp_one_channel",
               "foundation_anomaly", "crnn_ctc", "parseq"}
# 매핑·컨트랙트 확정 + 학습 엔진 가용(전 arch 어댑터 등록). 정확한 upstream 가중치
#   (D-FINE/RF-DETR repo, PaddleOCR det+cls)는 동일 어댑터 뒤 엔진 교체로 확장.
PLANNED: set[str] = set()


def model_type_of(arch: str) -> str:
    a = canonical_arch(arch)
    if a not in ARCH_TO_MODEL_TYPE:
        raise KeyError(f"unknown arch '{arch}' (canonical='{a}'). 등록: {sorted(ARCH_TO_MODEL_TYPE)}")
    return ARCH_TO_MODEL_TYPE[a]
