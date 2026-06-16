"""어댑터 패키지.

BaseTrainer 를 노출하고, 구현된 어댑터를 import 해 @register_trainer 자동등록을 트리거한다.
어댑터는 heavy lib 를 메서드 내부에서 lazy import 하므로 여기 import 는 lib 없이도 안전하다.
"""
from .base import BaseTrainer

# 등록 트리거. 어댑터 추가 시 여기에 import 추가.
from . import yolo_hbb      # noqa: F401  (hbbdetection, yolov8_hbb/yolov7_hbb)
from . import yolo_obb      # noqa: F401  (obbdetection, yolov8_obb/yolov11_obb)
from . import detr_hbb      # noqa: F401  (hbbdetection, rtdetr_hbb/dfine_hbb/rfdetr_hbb)
from . import efficientnet  # noqa: F401  (classification, efficientnet/classifier)
from . import deeplab       # noqa: F401  (segmentation, deeplab3pp/deeplab3pp_one_channel)
from . import foundation_ad  # noqa: F401  (anomaly_detection, foundation_anomaly)
from . import ocr_crnn      # noqa: F401  (ocr, crnn_ctc/parseq)

__all__ = ["BaseTrainer"]
