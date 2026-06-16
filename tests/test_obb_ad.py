"""OBB + AD(foundation_anomaly) 어댑터 — 등록/빌드/컨트랙트 (GPU/heavy-lib 불요)."""
from __future__ import annotations

from visionsuitetrain.config.canonical import model_type_of, PLANNED, IMPLEMENTED
from visionsuitetrain.config.preset import build_config_from_preset
from visionsuitetrain.data.ir import Sample, Region
from visionsuitetrain.data.writers.yolo_obb import to_obb_lines
from visionsuitetrain.export import build_manifest, build_model_yaml, VscExporter
from visionsuitetrain.registry import build_trainer, TRAINER_REGISTRY
import visionsuitetrain.trainers  # noqa: F401  (registry 자동등록)


def test_obb_and_ad_registered_not_planned():
    assert ("obbdetection", "yolov8_obb") in TRAINER_REGISTRY
    assert ("obbdetection", "yolov11_obb") in TRAINER_REGISTRY
    assert ("anomaly_detection", "foundation_anomaly") in TRAINER_REGISTRY
    assert {"yolov8_obb", "foundation_anomaly"} <= IMPLEMENTED
    assert {"yolov8_obb", "foundation_anomaly"} & PLANNED == set()
    assert PLANNED == {"dfine_hbb", "rfdetr_hbb"}


def test_to_obb_lines_polygon_and_rectangle():
    s = Sample("x.jpg", 100, 100, regions=[
        Region("a", "polygon", [(10, 10), (50, 10), (50, 40), (10, 40)])])
    line = to_obb_lines(s, {"a": 0})[0].split()
    assert line[0] == "0" and len(line) == 9                       # cls + 8 코너 좌표
    assert abs(float(line[1]) - 0.1) < 1e-6 and abs(float(line[2]) - 0.1) < 1e-6
    s2 = Sample("x.jpg", 100, 100, regions=[Region("a", "rectangle", [(20, 30), (60, 80)])])
    l2 = to_obb_lines(s2, {"a": 0})[0].split()                     # rect 2점 → 4 코너 확장
    assert len(l2) == 9
    assert abs(float(l2[1]) - 0.2) < 1e-6 and abs(float(l2[8]) - 0.8) < 1e-6


def test_obb_preset_builds_and_contract(tmp_path):
    cfg = build_config_from_preset("obb_default", root=str(tmp_path), names=["a", "b"])
    assert cfg.task == "obbdetection" and cfg.resolved_arch == "yolov8_obb"
    assert build_trainer(cfg).num_classes == 2
    out = [1, 7, 8400]                                             # [1, 4+NC+1, A], NC=2
    m = build_manifest(cfg, out)
    my = build_model_yaml(cfg, weights="m.onnx")
    VscExporter(cfg).assert_consistency(m, my, out)
    assert my["model"]["type"] == "yolov8_obb"
    assert my["postprocess"]["type"] == "yolov8_obb_decode"


def test_ad_preset_builds_and_contract(tmp_path):
    cfg = build_config_from_preset("ad_default", root=str(tmp_path), names=["defect"])
    assert cfg.task == "anomaly_detection" and cfg.resolved_arch == "foundation_anomaly"
    assert build_trainer(cfg).num_classes == 1
    assert model_type_of("foundation_anomaly") == "foundation_anomaly"
    out = [1, 1, 256, 256]                                         # heatmap
    m = build_manifest(cfg, out)
    assert m["outputs"]["output"]["value_range"] == [0.0, 1.0]
    assert m["task"]["thresholds"] == 0.5                          # scalar(글로벌)
    my = build_model_yaml(cfg, weights="m.onnx")
    VscExporter(cfg).assert_consistency(m, my, out)
    assert my["postprocess"]["type"] == "foundation_anomaly_decode"
    assert my["postprocess"]["anomaly_sigmoid"] is False


# ── 리뷰 하네스 확정 결함 회귀 ──
def test_to_obb_skips_degenerate_and_offscreen():
    from visionsuitetrain.data.writers.yolo_obb import to_obb_lines
    off = Sample("x.jpg", 100, 100, regions=[
        Region("a", "polygon", [(-50, -50), (-40, -50), (-40, -40), (-50, -40)])])
    assert to_obb_lines(off, {"a": 0}) == []           # 완전 화면밖 → clamp 붕괴 skip
    line = Sample("x.jpg", 100, 100, regions=[
        Region("a", "polygon", [(10, 10), (20, 10), (30, 10), (40, 10)])])
    assert to_obb_lines(line, {"a": 0}) == []           # 일직선(면적 0) skip


def test_to_obb_polygon_gt4_uses_minarearect():
    from visionsuitetrain.data.writers.yolo_obb import to_obb_lines
    s = Sample("x.jpg", 100, 100, regions=[Region("a", "polygon",
               [(10, 10), (30, 12), (50, 10), (50, 40), (30, 42), (10, 40)])])
    assert len(to_obb_lines(s, {"a": 0})[0].split()) == 9   # 6점 → minAreaRect → cls+8좌표


def test_ad_model_yaml_norm_range_and_no_normalize():
    cfg = build_config_from_preset("ad_default", root="./x", names=["defect"])
    my = build_model_yaml(cfg, weights="m.onnx")
    assert my["postprocess"]["anomaly_norm_min"] == 0.0
    assert my["postprocess"]["anomaly_norm_max"] == 1.0
    assert my["preprocess"]["normalize"] is False           # 그래프 /255 bake → VSC 재정규화 금지


def test_cli_rejects_config_and_preset_together(tmp_path):
    from visionsuitetrain.cli import main
    c = tmp_path / "c.yaml"
    c.write_text("task: classification\narch: efficientnet\n"
                 "dataset: {root: ./x, names: [a, b]}\nexport: {input: {w: 64, h: 64}}\n",
                 encoding="utf-8")
    assert main(["check", "-c", str(c), "--preset", "cls_default",
                 "--root", "./x", "--names", "a,b"]) == 2   # 동시 사용 → error exit 2
