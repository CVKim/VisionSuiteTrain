"""preset 시스템 — list/load/deep_merge/build_config_from_preset + CLI 통합 (GPU/heavy-lib 불요)."""
from __future__ import annotations

import pytest

from visionsuitetrain.config.preset import (
    list_presets, load_preset, deep_merge, build_config_from_preset)
from visionsuitetrain.registry import build_trainer
import visionsuitetrain.trainers  # noqa: F401  (registry 자동등록)


def test_bundled_presets_present():
    train = set(list_presets("train"))
    assert {"hbb_speed", "hbb_accuracy", "cls_default", "seg_default"} <= train
    assert list_presets("export") == ["default"]
    assert list_presets("test") == ["default"]


def test_load_preset_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_preset("nope", "train")


def test_deep_merge_recursive_and_nondestructive():
    base = {"a": 1, "train": {"epochs": 10, "lr": 0.1}, "x": {"y": 1}}
    over = {"train": {"epochs": 50}, "x": {"z": 2}, "b": 9}
    m = deep_merge(base, over)
    assert m["train"] == {"epochs": 50, "lr": 0.1}    # 재귀 병합(lr 보존)
    assert m["x"] == {"y": 1, "z": 2}
    assert m["a"] == 1 and m["b"] == 9
    assert base["train"]["epochs"] == 10              # base 비파괴


@pytest.mark.parametrize("preset,task,arch,names", [
    ("hbb_speed", "hbbdetection", "yolov8_hbb", ["a", "b", "c"]),
    ("hbb_accuracy", "hbbdetection", "yolov8_hbb", ["a", "b", "c"]),
    ("cls_default", "classification", "efficientnet", ["ok", "ng"]),
    ("seg_default", "segmentation", "deeplab3pp", ["background", "crack", "dust"]),
])
def test_build_config_from_preset_injects_dataset(preset, task, arch, names, tmp_path):
    cfg = build_config_from_preset(preset, root=str(tmp_path / "data"),
                                   names=names, out_dir=str(tmp_path / "run"))
    assert cfg.task == task and cfg.resolved_arch == arch
    assert cfg.dataset.root == str(tmp_path / "data")
    assert list(cfg.dataset.names) == names
    assert cfg.run.out_dir == str(tmp_path / "run")
    assert build_trainer(cfg).num_classes == len(names)   # 어댑터 인스턴스화까지


def test_preset_overrides_applied(tmp_path):
    cfg = build_config_from_preset("cls_default", root=str(tmp_path), names=["ok", "ng"],
                                   overrides={"train": {"epochs": 7}, "run": {"name": "exp9"}})
    assert cfg.train.epochs == 7
    assert cfg.run.name == "exp9"


def test_hbb_accuracy_carries_patch_and_border():
    cfg = build_config_from_preset("hbb_accuracy", root="./x", names=["a", "b"])
    pc = cfg.export.preprocess_carry
    assert pc["patch"]["overlap"] == 0.5
    assert pc["border_suppression"] == 0.2


def test_cli_presets_and_check_via_preset(capsys, tmp_path):
    from visionsuitetrain.cli import main
    assert main(["presets"]) == 0
    assert "hbb_speed" in capsys.readouterr().out
    rc = main(["check", "--preset", "cls_default", "--root", str(tmp_path), "--names", "ok,ng"])
    assert rc == 0
    assert "adapter=ok" in capsys.readouterr().out


def test_cli_preset_requires_root_and_names():
    from visionsuitetrain.cli import main
    assert main(["check", "--preset", "cls_default"]) == 2   # --root/--names 없음 → 에러 exit 2
