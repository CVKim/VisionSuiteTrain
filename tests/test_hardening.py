"""리뷰 하네스가 확정한 결함들의 회귀 방지 + 코어 커버리지 보강.

전부 순수-코어(GPU/torch/onnx 불요).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from visionsuitetrain.config.schema import load_train_config
from visionsuitetrain.data.ir import Sample, Region
from visionsuitetrain.data.readers import read_labelme
from visionsuitetrain.data.validate import validate_samples
from visionsuitetrain.data.split import split_samples
from visionsuitetrain.data.writers.yolo_txt import to_yolo_lines, write_yolo
from visionsuitetrain.data.writers.mask_png import to_mask
from visionsuitetrain.data.writers.imagefolder import label_of
from visionsuitetrain.export import build_model_yaml, VscExporter
from visionsuitetrain.registry import resolve_devices

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "train"


# ── HIGH: circle aabb (ir) ──
def test_circle_aabb_is_circumscribed_box():
    r = Region("hole", "circle", [(50, 50), (50, 80)])   # center(50,50), edge → r=30
    assert r.aabb() == (20, 20, 60, 60)


# ── HIGH: 부분 화면밖 박스 픽셀공간 clip (yolo_txt) ──
def test_to_yolo_clips_partial_box_in_pixel_space():
    s = Sample("x.jpg", 100, 100, regions=[Region("a", "rectangle", [(80, 80), (140, 140)])])
    cid, cx, cy, w, h = to_yolo_lines(s, {"a": 0})[0].split()
    assert abs(float(cx) - 0.9) < 1e-6 and abs(float(w) - 0.2) < 1e-6   # 평행이동 없이 clip


def test_to_yolo_skips_fully_offscreen_box():
    s = Sample("x.jpg", 100, 100, regions=[Region("a", "rectangle", [(120, 120), (140, 140)])])
    assert to_yolo_lines(s, {"a": 0}) == []


# ── MED: to_mask shape-aware (mask_png) ──
def test_to_mask_renders_circle_and_rectangle():
    circ = Sample("x.jpg", 50, 50, regions=[Region("c", "circle", [(25, 25), (25, 40)])])
    m = to_mask(circ, {"bg": 0, "c": 1}, background=0)
    assert m[25, 25] == 1 and m[0, 0] == 0
    rect = Sample("x.jpg", 50, 50, regions=[Region("c", "rectangle", [(10, 10), (30, 30)])])
    m2 = to_mask(rect, {"bg": 0, "c": 1}, background=0)
    assert m2[20, 20] == 1 and m2[40, 40] == 0


# ── MED: image_labels 다중 매칭 모호 → None (imagefolder) ──
def test_label_of_multi_flag_is_ambiguous():
    s = Sample("x.jpg", 10, 10, image_labels=["ok", "ng"])
    assert label_of(s, ["ok", "ng"]) is None


# ── LOW: 빈 라벨 도형은 reader 에서 제외 (labelme) ──
def test_labelme_skips_empty_label(tmp_path):
    import json
    doc = {"shapes": [
        {"label": "", "shape_type": "polygon", "points": [[0, 0], [1, 1], [2, 2]]},
        {"label": "crack", "shape_type": "polygon", "points": [[0, 0], [1, 1], [2, 2]]},
    ], "imageWidth": 20, "imageHeight": 20, "imagePath": "x.jpg"}
    p = tmp_path / "x.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    s = read_labelme(p)
    assert [r.class_name for r in s.regions] == ["crack"]


# ── MED: validate 기하/라벨 카테고리 (validate) ──
def test_validate_flags_missing_label_and_circle_geometry():
    samples = [
        Sample("a.jpg", 10, 10, regions=[Region("", "polygon", [(0, 0), (1, 1), (2, 2)])]),
        Sample("b.jpg", 10, 10, regions=[Region("crack", "circle", [(0, 0)])]),   # circle 1pt
    ]
    issues = validate_samples(samples, ["crack"], "segmentation", check_image_size=False)
    assert "missing_label" in issues
    assert any("circle" in g[1] for g in issues["bad_geometry"])


# ── HIGH: split 재현가능 셔플 + ratio 크기 (split) ──
def test_split_is_seeded_and_deterministic():
    samples = [Sample(f"{i}.jpg", 8, 8) for i in range(100)]
    cfg = {"ratio": {"train": 0.8, "val": 0.1, "test": 0.1}}
    a = split_samples(samples, cfg, seed=7)
    b = split_samples(samples, cfg, seed=7)
    assert [s.image_path for s in a["train"]] == [s.image_path for s in b["train"]]   # 결정적
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (80, 10, 10)
    # 셔플되어 파일명 정렬 순서를 그대로 자르지 않음
    assert [s.image_path for s in a["train"]] != [f"{i}.jpg" for i in range(80)]


def test_split_scalar_ratio_raises():
    with pytest.raises(ValueError):
        split_samples([Sample("a.jpg", 8, 8)], {"ratio": 0.8})


# ── 코어 커버리지: resolve_devices / write_yolo data.yaml ──
def test_resolve_devices():
    assert resolve_devices([]) == "cpu"
    assert resolve_devices([0, 1]) == [0, 1]


def test_write_yolo_emits_data_yaml(tmp_path):
    s = Sample("nope.jpg", 100, 100, regions=[Region("a", "rectangle", [(10, 10), (50, 50)])])
    data_yaml = write_yolo({"train": [s]}, ["a", "b"], tmp_path, link_images=False)
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    assert d["nc"] == 2 and d["names"] == {0: "a", 1: "b"}
    assert (tmp_path / "labels" / "train" / "nope.txt").exists()


# ── MED: 미등록 arch fail-fast (schema) ──
def test_unknown_arch_rejected_at_load(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("task: classification\narch: bogus_arch\n"
                 "dataset: { root: ./x, names: [a, b] }\n"
                 "export: { input: { w: 64, h: 64 } }\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_train_config(p)


# ── MED: seg_mode ↔ _one_channel arch 정합 (schema) ──
def _seg_cfg(tmp_path, arch, seg_mode):
    p = tmp_path / f"{arch}_{seg_mode}.yaml"
    p.write_text(
        f"task: segmentation\narch: {arch}\n"
        "dataset: { root: ./x, names: [background, defect] }\n"
        f"export: {{ input: {{ w: 128, h: 128 }}, seg_mode: {seg_mode} }}\n",
        encoding="utf-8")
    return p


def test_one_channel_arch_requires_one_channel_mode(tmp_path):
    with pytest.raises(ValueError):
        load_train_config(_seg_cfg(tmp_path, "deeplab3pp_one_channel", "multi_channel"))
    with pytest.raises(ValueError):
        load_train_config(_seg_cfg(tmp_path, "deeplab3pp", "one_channel"))
    # 정합하면 통과
    cfg = load_train_config(_seg_cfg(tmp_path, "deeplab3pp_one_channel", "one_channel"))
    assert cfg.export.seg_mode == "one_channel"


# ── MED: CLS NC 가 [B,NC,1,1] 도 동일취급 (base_exporter) ──
def test_infer_nc_handles_4d_cls_logit():
    exp = VscExporter(load_train_config(CFG_DIR / "efficientnet.yaml"))   # names=2
    assert exp._infer_nc([1, 2]) == 2
    assert exp._infer_nc([1, 2, 1, 1]) == 2     # trailing spatial 1 → 채널 dim 사용


# ── MED: nms_conf_vector 길이 검증 (model_yaml) ──
def test_model_yaml_rejects_bad_nms_vector_length():
    cfg = load_train_config(CFG_DIR / "yolov8_hbb.yaml")   # names=3
    with pytest.raises(ValueError):
        build_model_yaml(cfg, nms_conf_vector=[0.1, 0.2])   # 길이 2 != 3
