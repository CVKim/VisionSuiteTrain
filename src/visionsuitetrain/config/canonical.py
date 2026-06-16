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
    "dfine_hbb": "yolov8_hbb",          # 2차(학습 어댑터 미구현) — export 는 [1,4+NC,A] drop-in
    "rfdetr_hbb": "yolov8_hbb",         # 2차 — A=쿼리(300), 디코드(conf+NMS) 동일
    "foundation_anomaly": "foundation_anomaly",   # 2차 — [1,1,H,W] heatmap, scalar threshold
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
    "foundation_anomaly": "anomaly_detection",
}

# 1차 범위에서 어댑터가 실제 등록된 canonical arch (정보용; 실제 게이트는 registry/build_trainer)
IMPLEMENTED = {"yolov8_hbb", "yolov7_hbb", "efficientnet", "deeplab3pp", "deeplab3pp_one_channel"}
# export 컨트랙트·model.type 매핑은 확정(drop-in)이나 학습 어댑터 미구현(2차):
#   dfine_hbb/rfdetr_hbb = labelme_hbb 입력 → [1,4+NC,A] export(=yolov8_hbb 디코드),
#   foundation_anomaly = [1,1,H,W] heatmap, yolov8_obb = OBB. build_trainer 에서 '어댑터 없음'.
PLANNED = {"dfine_hbb", "rfdetr_hbb", "yolov8_obb", "foundation_anomaly"}


def model_type_of(arch: str) -> str:
    a = canonical_arch(arch)
    if a not in ARCH_TO_MODEL_TYPE:
        raise KeyError(f"unknown arch '{arch}' (canonical='{a}'). 등록: {sorted(ARCH_TO_MODEL_TYPE)}")
    return ARCH_TO_MODEL_TYPE[a]
