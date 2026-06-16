"""export 정합: manifest/model.yaml 빌더 + VscExporter.assert_consistency 의 fail-fast.

ONNX/torch 불요 — output_shape 를 손으로 주입해 컨트랙트 로직만 검증.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from visionsuitetrain.config.schema import load_train_config
from visionsuitetrain.config.canonical import model_type_of, ARCH_TASK, PLANNED
from visionsuitetrain.export import build_manifest, build_model_yaml, VscExporter

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "train"

# task → (config, 정상 출력 shape, NC)
CASES = {
    "hbb": ("yolov8_hbb.yaml", [1, 7, 8400], 3),       # 4+NC, NC=3
    "cls": ("efficientnet.yaml", [1, 2], 2),           # [B, NC]
    "seg": ("deeplab3pp.yaml", [1, 3, 512, 512], 3),   # [B, C, H, W]
}


@pytest.mark.parametrize("key", list(CASES))
def test_manifest_and_model_yaml_consistent(key):
    fname, out_shape, nc = CASES[key]
    cfg = load_train_config(CFG_DIR / fname)
    manifest = build_manifest(cfg, out_shape)
    model_yaml = build_model_yaml(cfg, weights="model.onnx")

    # label_map 키 = 0..NC-1
    assert sorted(manifest["task"]["label_map"].keys()) == list(range(nc))
    assert manifest["task"]["task_type"] == cfg.task
    # input shape 정합
    in_name = cfg.export.io_names.input
    assert manifest["inputs"][in_name]["shape"] == [1, cfg.export.input.c,
                                                     cfg.export.input.h, cfg.export.input.w]
    # postprocess.type = "{model_type}_decode"
    assert model_yaml["postprocess"]["type"].endswith("_decode")
    # fail-fast assert 통과
    VscExporter(cfg).assert_consistency(manifest, model_yaml, out_shape)


def test_nc_mismatch_raises():
    cfg = load_train_config(CFG_DIR / "yolov8_hbb.yaml")   # names=3 → 기대 4+3=7
    bad_shape = [1, 9, 8400]                                # 4+5=9 → NC=5 != 3
    manifest = build_manifest(cfg, bad_shape)
    model_yaml = build_model_yaml(cfg, weights="model.onnx")
    with pytest.raises(AssertionError):
        VscExporter(cfg).assert_consistency(manifest, model_yaml, bad_shape)


def test_symbolic_dim_skips_nc_check():
    cfg = load_train_config(CFG_DIR / "yolov8_hbb.yaml")
    sym_shape = [1, -1, -1]                                 # dynamic → NC 검증 skip
    manifest = build_manifest(cfg, sym_shape)
    model_yaml = build_model_yaml(cfg, weights="model.onnx")
    VscExporter(cfg).assert_consistency(manifest, model_yaml, sym_shape)   # 예외 없어야


def test_seg_one_channel_threshold_emitted(tmp_path):
    cfg = load_train_config(CFG_DIR / "deeplab3pp.yaml")
    cfg.export.seg_mode = "one_channel"
    my = build_model_yaml(cfg, weights="model.onnx")
    assert "confidence_threshold_one_channel_seg" in my["postprocess"]
    assert my["postprocess"]["seg_background_class"] == 0


def test_cls_activation_carried():
    cfg = load_train_config(CFG_DIR / "efficientnet.yaml")
    my = build_model_yaml(cfg, weights="model.onnx")
    assert my["postprocess"]["cls_activation"] == "softmax"
    assert my["preprocess"]["imagenet_std"] is True


# ── talos export manifest 스키마 정합 ──
def test_manifest_no_maskings_has_rois():
    cfg = load_train_config(CFG_DIR / "yolov8_hbb.yaml")
    m = build_manifest(cfg, [1, 7, 8400])
    assert "maskings" not in m["preprocessing"]      # real manifest 엔 없음
    assert m["preprocessing"]["rois"] == []
    assert "patch" in m["preprocessing"]


def test_detection_thresholds_always_emitted():
    cfg = load_train_config(CFG_DIR / "yolov8_hbb.yaml")
    m = build_manifest(cfg, [1, 7, 8400])
    th = m["task"]["thresholds"]
    assert set(th.keys()) == set(cfg.dataset.names)   # per-class 전부
    # 명시 thresholds 우선
    m2 = build_manifest(cfg, [1, 7, 8400], thresholds={"scratch": 0.4, "dent": 0.5, "stain": 0.6})
    assert m2["task"]["thresholds"]["dent"] == 0.5


def test_seg_output_axes_declares_activation_and_background():
    cfg = load_train_config(CFG_DIR / "deeplab3pp.yaml")   # multi_channel
    m = build_manifest(cfg, [1, 3, 512, 512])
    axes = m["outputs"]["output"]["axes"]
    feat = next(a for a in axes if a["type"] == "features")["structure"][0]
    assert feat["activation"] == "softmax"
    assert feat["background_index"] == 0
    assert feat["size"] == 3
    assert m["task"]["background_index"] == 0


def test_seg_one_channel_uses_sigmoid_activation():
    cfg = load_train_config(CFG_DIR / "deeplab3pp.yaml")
    cfg.export.seg_mode = "one_channel"
    m = build_manifest(cfg, [1, 3, 512, 512])
    feat = next(a for a in m["outputs"]["output"]["axes"] if a["type"] == "features")["structure"][0]
    assert feat["activation"] == "sigmoid"


# ── 2차 범위 컨트랙트 (레퍼런스 export 대조: D-FINE/RF-DETR/Foundation AD) ──
def test_detr_archs_decode_via_unified_yolov8_hbb():
    # D-FINE/RF-DETR export = [1,4+NC,A](A=쿼리) → 동일 yolov8_hbb 디코드(신규 핸들러 불요)
    assert model_type_of("dfine_hbb") == "yolov8_hbb"
    assert model_type_of("rfdetr_hbb") == "yolov8_hbb"


def test_foundation_anomaly_mapping():
    assert model_type_of("foundation_anomaly") == "foundation_anomaly"
    assert ARCH_TASK["foundation_anomaly"] == "anomaly_detection"
    assert {"dfine_hbb", "rfdetr_hbb"} <= PLANNED      # foundation_anomaly 는 이제 구현됨
    assert "foundation_anomaly" not in PLANNED


def test_dfine_sample_config_decodes_as_yolov8_hbb():
    cfg = load_train_config(CFG_DIR / "dfine_hbb.yaml")
    assert cfg.task == "hbbdetection" and cfg.resolved_arch == "dfine_hbb"
    my = build_model_yaml(cfg, weights="model.onnx")
    assert my["model"]["type"] == "yolov8_hbb"            # 통일 디코드
    assert my["postprocess"]["type"] == "yolov8_hbb_decode"


def test_manifest_and_model_yaml_emit_patch_and_border_suppression():
    cfg = load_train_config(CFG_DIR / "dfine_hbb.yaml")   # patch + border_suppression 포함
    m = build_manifest(cfg, [1, 7, 300])
    assert m["preprocessing"]["patch"] == {"enabled": True, "height": 1024, "width": 1024}
    assert m["postprocessing"]["border_suppression"] == 0.2
    my = build_model_yaml(cfg, weights="model.onnx")
    assert my["inference"]["patch"]["enable"] is True
    assert my["inference"]["patch"]["border_suppress"] == 0.2
    assert my["inference"]["patch"]["overlap"] == 0.5


def test_no_patch_block_when_unconfigured():
    cfg = load_train_config(CFG_DIR / "yolov8_hbb.yaml")  # patch 미설정
    m = build_manifest(cfg, [1, 7, 8400])
    assert m["preprocessing"]["patch"]["enabled"] is False
    assert "postprocessing" not in m
    assert build_model_yaml(cfg, weights="m.onnx")["inference"]["patch"]["enable"] is False
