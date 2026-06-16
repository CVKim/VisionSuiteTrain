"""config 풀 정렬 — 구조화 sub-config(optimizer/scheduler/data_module/data_selection/
threshold_tuning) + 평면 단축키 → 구조 이관(하위호환)."""
from __future__ import annotations

from visionsuitetrain.config.schema import build_train_config
from visionsuitetrain.config.preset import build_config_from_preset


def _base(train: dict) -> dict:
    return {"task": "classification", "arch": "efficientnet",
            "dataset": {"root": "./x", "names": ["a", "b"]},
            "export": {"input": {"w": 64, "h": 64}}, "train": train}


def test_flat_shorthand_migrates_to_structured():
    cfg = build_train_config(_base({
        "epochs": 7, "optimizer": "sgd", "lr": 0.05, "weight_decay": 1.0e-3,
        "workers": 2, "augment": {"mosaic": 0.3, "fliplr": 0.1}}))
    assert cfg.train.epochs == 7
    assert cfg.train.optimizer.type == "sgd"
    assert cfg.train.optimizer.lr == 0.05
    assert cfg.train.optimizer.weight_decay == 1.0e-3
    assert cfg.train.data_module.num_workers == 2
    assert cfg.train.data_module.transforms.mosaic == 0.3
    assert cfg.train.data_module.transforms.fliplr == 0.1


def test_structured_form_loads_directly():
    cfg = build_train_config(_base({
        "optimizer": {"type": "adamw", "lr": 2.5e-4},
        "scheduler": {"type": "cosine", "end_lr": 1.0e-5},
        "data_module": {"num_workers": 4, "transforms": {"mixup": 0.2}},
        "data_selection": {"enable": True, "interval_epoch": 3},
        "threshold_tuning": {"enable": True, "metric_to_maximize": "Recall"}}))
    assert cfg.train.optimizer.lr == 2.5e-4
    assert cfg.train.scheduler.type == "cosine" and cfg.train.scheduler.end_lr == 1.0e-5
    assert cfg.train.data_module.num_workers == 4
    assert cfg.train.data_module.transforms.mixup == 0.2
    assert cfg.train.data_selection.enable and cfg.train.data_selection.interval_epoch == 3
    assert cfg.train.threshold_tuning.metric_to_maximize == "Recall"


def test_defaults_when_train_omitted():
    cfg = build_train_config(_base({}))           # train 생략 → 전 sub-config 기본값
    assert cfg.train.optimizer.type == "adamw"
    assert cfg.train.data_module.num_workers == 8
    assert cfg.train.scheduler.type == "cosine"


def test_presets_use_structured_optimizer():
    cfg = build_config_from_preset("hbb_speed", root="./x", names=["a", "b", "c"])
    assert cfg.train.optimizer.type == "sgd"
    assert cfg.train.optimizer.lr == 1.0e-2
    assert cfg.train.data_module.transforms.mosaic == 0.8
