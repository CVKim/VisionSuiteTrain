"""DETR(RT-DETR) + OCR(CRNN-CTC) 어댑터 — 등록/빌드/컨트랙트 (GPU/heavy-lib 불요)."""
from __future__ import annotations

from visionsuitetrain.config.canonical import model_type_of, ARCH_TASK, PLANNED
from visionsuitetrain.config.preset import build_config_from_preset
from visionsuitetrain.export import build_model_yaml
from visionsuitetrain.registry import build_trainer, TRAINER_REGISTRY
import visionsuitetrain.trainers  # noqa: F401  (registry 자동등록)


def test_detr_and_ocr_registered_no_planned():
    for a in ("rtdetr_hbb", "dfine_hbb", "rfdetr_hbb"):
        assert ("hbbdetection", a) in TRAINER_REGISTRY
        assert model_type_of(a) == "yolov8_hbb"           # 통일 디코드
    assert ("ocr", "crnn_ctc") in TRAINER_REGISTRY
    assert model_type_of("crnn_ctc") == "crnn_ctc"        # parseq 핸들러 비호환 → 별도 model.type
    assert ARCH_TASK["crnn_ctc"] == "ocr"
    assert PLANNED == set()                               # 전 arch 어댑터 등록


def test_detr_preset_builds(tmp_path):
    cfg = build_config_from_preset("detr_default", root=str(tmp_path), names=["a", "b"],
                                   out_dir=str(tmp_path / "run"))
    assert cfg.task == "hbbdetection" and cfg.resolved_arch == "rtdetr_hbb"
    assert build_trainer(cfg).__class__.__name__ == "DetrHbbTrainer"
    assert build_model_yaml(cfg, weights="m.onnx")["model"]["type"] == "yolov8_hbb"


def test_ocr_preset_builds_with_charset(tmp_path):
    charset = list("0123456789")
    cfg = build_config_from_preset("ocr_default", root=str(tmp_path), names=charset,
                                   out_dir=str(tmp_path / "run"))
    assert cfg.task == "ocr" and cfg.resolved_arch == "crnn_ctc"
    assert build_trainer(cfg).num_classes == 10
    my = build_model_yaml(cfg, weights="m.onnx")
    assert my["postprocess"]["type"] == "crnn_ctc_decode"
    assert my["postprocess"]["ocr_charset"] == charset     # VSC 가 읽는 키(시퀀스)
    assert my["postprocess"]["blank_index"] == 10          # CTC blank = NC
