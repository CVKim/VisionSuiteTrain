"""train/test config 스키마 (pydantic v2) + 로더.

arch↔task 정합, ${run.name} 치환, 필수 필드 검증을 로드 시점에 fail-fast.
arch_variant/augment 등 어댑터별 자유 필드는 dict 로 통과시킨다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from .canonical import TASK_TYPES, ARCH_TASK, ARCH_TO_MODEL_TYPE, canonical_arch


class RunCfg(BaseModel):
    name: str = "run"
    seed: int = 42
    out_dir: str = "./runs/${run.name}"


class DatasetCfg(BaseModel):
    format: str = "labelme"             # labelme|coco|yolo|imagefolder|mask_png
    root: str
    names: list[str]                    # 클래스 매핑 단일 진실원(0-base, 순서=라벨 id)
    split: Optional[dict[str, Any]] = None
    cache_to: Optional[str] = None      # labelme→어댑터 중간 캐시 포맷

    @field_validator("names")
    @classmethod
    def _names_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("dataset.names 는 최소 1개 필요")
        if len(set(v)) != len(v):
            raise ValueError(f"dataset.names 중복: {v}")
        return v


class IOName(BaseModel):
    input: str = "data"
    output: str = "output"


class InputCfg(BaseModel):
    w: int
    h: int
    c: int = 3
    dtype: str = "FLOAT"
    value_range: list[float] = [0.0, 255.0]


class ExportCfg(BaseModel):
    to: str = "vsc"
    format: str = "onnx"
    io_names: IOName = Field(default_factory=IOName)
    input: InputCfg
    opset: int = 17
    dynamic_axes: bool = False
    backend: str = "trt"
    fp16: bool = True
    preprocess_carry: dict[str, Any] = Field(default_factory=dict)
    cls_activation: str = "softmax"     # CLS 전용
    seg_background_class: int = 0        # SEG 전용
    seg_mode: str = "multi_channel"     # SEG 전용: multi_channel|one_channel


class TrainCfg(BaseModel):
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    optimizer: str = "adamw"
    lr: float = 1.0e-3
    weight_decay: float = 5.0e-4
    lr_schedule: dict[str, Any] = Field(default_factory=dict)
    amp: bool = True
    gpus: list[int] = Field(default_factory=lambda: [0])
    workers: int = 8
    augment: dict[str, Any] = Field(default_factory=dict)
    early_stop: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    schema_version: int = 1
    run: RunCfg = Field(default_factory=RunCfg)
    task: str
    arch: str
    arch_variant: dict[str, Any] = Field(default_factory=dict)
    dataset: DatasetCfg
    train: TrainCfg = Field(default_factory=TrainCfg)
    export: ExportCfg
    threshold_sweep: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task")
    @classmethod
    def _task_enum(cls, v: str) -> str:
        if v not in TASK_TYPES:
            raise ValueError(f"task '{v}' invalid. one of {sorted(TASK_TYPES)}")
        return v

    @field_validator("arch")
    @classmethod
    def _arch_known(cls, v: str) -> str:
        # 미등록 arch(오타 등)는 로드 시점에 차단(이전엔 fail-open 으로 통과 → build 시 KeyError)
        if canonical_arch(v) not in ARCH_TO_MODEL_TYPE:
            raise ValueError(f"arch '{v}' unknown. one of {sorted(ARCH_TO_MODEL_TYPE)}")
        return v

    @property
    def resolved_arch(self) -> str:
        return canonical_arch(self.arch)


def load_train_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = TrainConfig(**raw)
    # ${run.name} 치환
    cfg.run.out_dir = cfg.run.out_dir.replace("${run.name}", cfg.run.name)
    # arch↔task 정합
    expected = ARCH_TASK.get(cfg.resolved_arch)
    if expected and expected != cfg.task:
        raise ValueError(
            f"arch '{cfg.arch}'(={cfg.resolved_arch}) 는 task '{expected}' 인데 "
            f"config.task='{cfg.task}'")
    # export 입력과 train.imgsz 정합 권고(det/seg) — 어긋나면 경고(어댑터가 최종 처리)
    if cfg.task in ("hbbdetection", "obbdetection", "segmentation"):
        if cfg.export.input.w != cfg.train.imgsz or cfg.export.input.h != cfg.train.imgsz:
            print(f"[config] warn: train.imgsz({cfg.train.imgsz}) != "
                  f"export.input({cfg.export.input.h}x{cfg.export.input.w}) — 학습/배포 해상도 상이")
    # seg one_channel arch ↔ seg_mode 정합(model.type 와 decode 메타 모순 방지)
    if cfg.resolved_arch.endswith("_one_channel") and cfg.export.seg_mode != "one_channel":
        raise ValueError("arch '_one_channel' 인데 export.seg_mode != 'one_channel'")
    if cfg.task == "segmentation" and cfg.export.seg_mode == "one_channel" \
            and not cfg.resolved_arch.endswith("_one_channel"):
        raise ValueError("export.seg_mode 'one_channel' 인데 arch 가 '_one_channel' 아님")
    return cfg
