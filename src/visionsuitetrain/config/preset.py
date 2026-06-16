"""preset 시스템 — named 부분 config(yaml) + 런타임 데이터 주입 → 검증된 TrainConfig.

참조 학습 코어의 presets/{train,test,export} 패턴 클린룸 채택:
  - preset = 모델/하이퍼파라미터/증강/export 의 **sparse override**(데이터셋 제외).
  - dataset(root/names)·출력 경로는 **학습 시점에 주입**(preset 은 데이터 무관 재사용).
  - 머지 엔진 아님 — 부분 dict 를 deep-merge 후 pydantic 기본값이 나머지를 채움.
presets 는 패키지에 동봉(`visionsuitetrain/presets/{train,test,export}/*.yaml`).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import yaml

from .schema import TrainConfig, build_train_config

_PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"   # visionsuitetrain/presets


def presets_dir() -> Path:
    return _PRESETS_DIR


def list_presets(kind: str = "train") -> list[str]:
    d = _PRESETS_DIR / kind
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []


def load_preset(name: str, kind: str = "train") -> dict:
    p = _PRESETS_DIR / kind / f"{name}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"preset '{name}'({kind}) 없음. 가용: {list_presets(kind)}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """override 를 base 에 재귀 병합(override 우선). 양쪽 비파괴."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_config_from_preset(preset: str, *, root: str, names: list[str],
                             out_dir: Optional[str] = None,
                             overrides: Optional[dict] = None) -> TrainConfig:
    """train preset + 런타임 데이터(root/names) 주입 → 검증된 TrainConfig.

    preset 은 dataset 을 포함하지 않는다(데이터 무관 재사용). root/names 는 학습 시점 주입.
    overrides 로 임의 키 덮어쓰기(예: {'train': {'epochs': 50}, 'run': {'name': 'exp1'}}).
    """
    data = load_preset(preset, "train")
    ds = dict(data.get("dataset") or {})        # preset 에 dataset 일부 있으면 보존
    ds.setdefault("format", "labelme")
    ds["root"] = root
    ds["names"] = list(names)
    data["dataset"] = ds
    if out_dir is not None:
        data.setdefault("run", {})["out_dir"] = out_dir
    if overrides:
        data = deep_merge(data, overrides)
    return build_train_config(data)
