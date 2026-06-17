"""train/test config 스키마 (pydantic v2) + 로더.

arch↔task 정합, ${run.name} 치환, 필수 필드 검증을 로드 시점에 fail-fast.
arch_variant/augment 등 어댑터별 자유 필드는 dict 로 통과시킨다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

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
    simplify: bool = True        # ultralytics export onnxslim 단순화(일부 env 에서 segfault → false)
    backend: str = "trt"
    fp16: bool = True
    preprocess_carry: dict[str, Any] = Field(default_factory=dict)
    cls_activation: str = "auto"        # CLS: export 가 softmax 베이킹 → VSC auto 가 확률 감지(이중 softmax 회피). softmax 강제는 왜곡.
    seg_background_class: int = 0        # SEG 전용
    seg_mode: str = "multi_channel"     # SEG 전용: multi_channel|one_channel


class OptimizerCfg(BaseModel):
    type: str = "adamw"                 # adamw | sgd
    lr: float = 1.0e-3
    weight_decay: float = 5.0e-4
    momentum: float = 0.937             # sgd 전용
    betas: list[float] = [0.9, 0.999]   # adamw 전용


class SchedulerCfg(BaseModel):
    type: str = "cosine"                # cosine | linear_decay | multistep | none
    end_lr: float = 1.0e-4
    warmup_epochs: int = 0
    milestones: list[int] = Field(default_factory=list)
    gamma: float = 0.1


class TransformsCfg(BaseModel):
    """증강 스펙(ultralytics 호환 키 + 공통). yolo 어댑터가 그대로 전달."""
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0


class PatchCfg(BaseModel):
    enable: bool = False
    height: int = 0
    width: int = 0
    overlap_ratio: float = 0.5
    margin_ratio: float = 0.25
    selector: str = "centric"


class DataModuleCfg(BaseModel):
    num_workers: int = 8
    rois: Optional[list[Any]] = None
    filter_unlabeled: bool = False
    patch: PatchCfg = Field(default_factory=PatchCfg)
    transforms: TransformsCfg = Field(default_factory=TransformsCfg)


class DataSelectionCfg(BaseModel):
    enable: bool = False
    interval_epoch: int = 5
    sampling_ratio: float = 1.0
    score_type: str = "target_uncertainty"
    warmup_ratio: float = 0.1


class ThresholdTuningCfg(BaseModel):
    enable: bool = False
    metric_to_maximize: str = "mAP50"   # mAP50 | Recall | Fbeta ...
    beta: float = 1.0


class TrainCfg(BaseModel):
    """학습 설정(구조화). 평면 단축키(lr/weight_decay/optimizer:str/workers/augment/lr_schedule)는
    model_validator 가 구조화 필드로 자동 이관(하위호환 + sparse preset 편의)."""
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    amp: bool = True
    gpus: list[int] = Field(default_factory=lambda: [0])
    max_grad_norm: float = 10.0
    optimizer: OptimizerCfg = Field(default_factory=OptimizerCfg)
    scheduler: SchedulerCfg = Field(default_factory=SchedulerCfg)
    data_module: DataModuleCfg = Field(default_factory=DataModuleCfg)
    data_selection: DataSelectionCfg = Field(default_factory=DataSelectionCfg)
    threshold_tuning: ThresholdTuningCfg = Field(default_factory=ThresholdTuningCfg)

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        opt = d.get("optimizer")
        if isinstance(opt, str):           # optimizer: "adamw" → {type: adamw}
            d["optimizer"] = {"type": opt}
        if not isinstance(d.get("optimizer"), dict):
            d["optimizer"] = {}
        for k in ("lr", "weight_decay", "momentum"):   # 평면 lr 등 → optimizer.*
            if k in d:
                d["optimizer"].setdefault(k, d.pop(k))
        dm = d.get("data_module")
        if not isinstance(dm, dict):
            dm = {}
        d["data_module"] = dm
        if "workers" in d:                 # workers → data_module.num_workers
            dm.setdefault("num_workers", d.pop("workers"))
        if "augment" in d:                 # augment(dict) → data_module.transforms
            tr = dm.get("transforms")
            if not isinstance(tr, dict):
                tr = {}
            dm["transforms"] = tr
            for k, v in (d.pop("augment") or {}).items():
                tr.setdefault(k, v)
        if "lr_schedule" in d:             # lr_schedule(dict) → scheduler
            sched = d.pop("lr_schedule") or {}
            if isinstance(sched, dict) and sched and not d.get("scheduler"):
                d["scheduler"] = sched
        d.pop("early_stop", None)          # 미사용 폐기
        return d


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


def _finalize(cfg: TrainConfig) -> TrainConfig:
    """raw→TrainConfig 이후 공통 후처리/정합(파일 로드·preset 빌드가 공유)."""
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


def build_train_config(raw: dict) -> TrainConfig:
    """raw dict → 검증된 TrainConfig (preset 머지 결과/프로그램적 구성에 사용)."""
    return _finalize(TrainConfig(**(raw or {})))


def load_train_config(path: str | Path) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return build_train_config(raw)
