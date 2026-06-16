"""BaseTrainer — 모든 어댑터의 공통 인터페이스(ABC).

3-메서드 계약: prepare_data → train → export_to_vsc.
heavy 학습 lib(ultralytics/timm/torchvision)는 각 어댑터가 메서드 내부에서 lazy import
(패키지는 lib 없이도 import 가능 → CI/스키마검증/데이터파이프라인 테스트가 GPU 불요).
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

from ..config.schema import TrainConfig


class BaseTrainer(abc.ABC):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.names = list(cfg.dataset.names)
        self.out_dir = Path(cfg.run.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 메타 ──
    @property
    def task(self) -> str:
        return self.cfg.task

    @property
    def arch(self) -> str:
        return self.cfg.resolved_arch

    @property
    def num_classes(self) -> int:
        return len(self.names)

    # ── 3-메서드 계약 ──
    @abc.abstractmethod
    def prepare_data(self) -> Any:
        """dataset.format(labelme 등) → 어댑터 학습 포맷으로 변환.
        반환: 학습에 넘길 data 핸들(예: ultralytics data.yaml 경로, ImageFolder root 등)."""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, data: Any) -> Path:
        """학습 실행. 반환: best 체크포인트 경로."""
        raise NotImplementedError

    @abc.abstractmethod
    def export_to_vsc(self, ckpt: Path) -> dict[str, Path]:
        """ckpt → VSC 아티팩트. 반환: {'onnx':..., 'model_yaml':..., 'manifest':...} 경로."""
        raise NotImplementedError

    # ── 전체 파이프라인 ──
    def run(self) -> dict[str, Path]:
        data = self.prepare_data()
        ckpt = self.train(data)
        return self.export_to_vsc(ckpt)
