"""(task, arch) → BaseTrainer 어댑터 자동등록 registry.

어댑터 모듈이 @register_trainer 로 자동 등록(트리거: trainers/__init__.py import).
VSC 의 TV_REGISTER_MODEL_TYPES 자동등록 패턴과 동형.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config.canonical import canonical_arch

if TYPE_CHECKING:
    from .config.schema import TrainConfig
    from .trainers.base import BaseTrainer

# key = (task, arch) ; value = BaseTrainer 서브클래스
TRAINER_REGISTRY: dict[tuple[str, str], type] = {}


def register_trainer(task: str, *archs: str):
    """어댑터 클래스 데코레이터 — (task, arch) 들로 등록."""
    def deco(cls):
        for a in archs:
            TRAINER_REGISTRY[(task, a)] = cls
        cls._task = task
        cls._archs = tuple(archs)
        return cls
    return deco


def build_trainer(cfg: "TrainConfig") -> "BaseTrainer":
    """config → 해당 어댑터 인스턴스. 미등록이면 가용 목록과 함께 KeyError."""
    for arch in (canonical_arch(cfg.arch), cfg.arch):
        cls = TRAINER_REGISTRY.get((cfg.task, arch))
        if cls is not None:
            return cls(cfg)
    avail = sorted(f"{t}/{a}" for (t, a) in TRAINER_REGISTRY)
    raise KeyError(
        f"(task={cfg.task}, arch={cfg.arch}) 어댑터 없음. 등록된 것: {avail}")


def resolve_devices(gpus: list[int]) -> str | list[int]:
    """gpus 리스트 → 학습 엔진에 넘길 device 지정(빈 리스트면 'cpu')."""
    if not gpus:
        return "cpu"
    return list(gpus)
