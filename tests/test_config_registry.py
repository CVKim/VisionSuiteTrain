"""config 스키마 + (task,arch) registry 매핑 + arch↔task 정합 검증 (GPU/heavy-lib 불요)."""
from __future__ import annotations

from pathlib import Path

import pytest

from visionsuitetrain.config.schema import load_train_config
from visionsuitetrain.registry import TRAINER_REGISTRY, build_trainer
import visionsuitetrain.trainers  # noqa: F401  (registry 자동등록)

CFG_DIR = Path(__file__).resolve().parents[1] / "configs" / "train"
SAMPLES = ["yolov8_hbb.yaml", "efficientnet.yaml", "deeplab3pp.yaml"]


@pytest.mark.parametrize("fname", SAMPLES)
def test_sample_config_loads_and_maps_to_adapter(fname, tmp_path):
    cfg = load_train_config(CFG_DIR / fname)
    # ${run.name} 치환 확인
    assert "${" not in cfg.run.out_dir
    # 어댑터 등록 존재
    cls = TRAINER_REGISTRY.get((cfg.task, cfg.resolved_arch))
    assert cls is not None, f"{cfg.task}/{cfg.resolved_arch} 어댑터 미등록"
    # 실제 인스턴스화(out_dir 은 tmp 로 격리)
    cfg.run.out_dir = str(tmp_path / "run")
    trainer = build_trainer(cfg)
    assert trainer.task == cfg.task
    assert trainer.num_classes == len(cfg.dataset.names)


def test_three_first_scope_adapters_registered():
    keys = set(TRAINER_REGISTRY)
    assert ("hbbdetection", "yolov8_hbb") in keys
    assert ("classification", "efficientnet") in keys
    assert ("segmentation", "deeplab3pp") in keys


def test_arch_task_mismatch_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "task: classification\narch: yolov8_hbb\n"
        "dataset: { root: ./x, names: [a, b] }\n"
        "export: { input: { w: 224, h: 224 } }\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        load_train_config(bad)


def test_duplicate_names_rejected(tmp_path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "task: classification\narch: efficientnet\n"
        "dataset: { root: ./x, names: [a, a] }\n"
        "export: { input: { w: 224, h: 224 } }\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        load_train_config(bad)


def test_classifier_alias_resolves_to_efficientnet(tmp_path):
    cfg_path = tmp_path / "alias.yaml"
    cfg_path.write_text(
        "task: classification\narch: classifier\n"
        "dataset: { root: ./x, names: [ok, ng] }\n"
        "export: { input: { w: 224, h: 224 } }\n",
        encoding="utf-8")
    cfg = load_train_config(cfg_path)
    assert cfg.resolved_arch == "efficientnet"
    assert TRAINER_REGISTRY.get((cfg.task, cfg.resolved_arch)) is not None
