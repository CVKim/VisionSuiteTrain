"""vstrain CLI — train / export / validate / check / list / presets.

config 소스 2가지: -c/--config (전체 config yaml) 또는 --preset NAME (+ --root/--names 런타임 주입).
registry 채우기 위해 trainers import(데코레이터 자동등록). heavy lib 은 어댑터 메서드 내부 lazy import.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from . import __version__
from . import trainers  # noqa: F401  (registry 자동등록 트리거)
from .config.preset import build_config_from_preset, list_presets
from .config.schema import TrainConfig, load_train_config
from .registry import TRAINER_REGISTRY, build_trainer


def _resolve_config(args: argparse.Namespace) -> TrainConfig:
    """--config 또는 --preset(+--root/--names) → 검증된 TrainConfig."""
    if getattr(args, "preset", None):
        if not args.root or not args.names:
            raise ValueError("--preset 사용 시 --root 와 --names 필요")
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        return build_config_from_preset(args.preset, root=args.root, names=names,
                                        out_dir=getattr(args, "out", None))
    if getattr(args, "config", None):
        return load_train_config(args.config)
    raise ValueError("-c/--config 또는 --preset 중 하나가 필요합니다")


def _cmd_validate(args: argparse.Namespace) -> int:
    from .data import iter_labelme_dir, validate_samples, summarize
    cfg = _resolve_config(args)
    samples = iter_labelme_dir(cfg.dataset.root)
    issues = validate_samples(samples, list(cfg.dataset.names), cfg.task)
    print(f"[vstrain] samples={len(samples)} {summarize(issues)}")
    return 1 if issues else 0


def _cmd_check(args: argparse.Namespace) -> int:
    from .config.canonical import PLANNED
    cfg = _resolve_config(args)
    trainer_cls = None
    for arch in (cfg.resolved_arch, cfg.arch):
        trainer_cls = TRAINER_REGISTRY.get((cfg.task, arch))
        if trainer_cls:
            break
    adapter = "ok" if trainer_cls else ("planned(2차)" if cfg.resolved_arch in PLANNED else "MISSING")
    print(f"[vstrain] config OK: run={cfg.run.name} task={cfg.task} "
          f"arch={cfg.arch}(={cfg.resolved_arch}) names={len(cfg.dataset.names)} "
          f"adapter={adapter} out={cfg.run.out_dir}")
    return 0 if trainer_cls else 2


def _cmd_list(_args: argparse.Namespace) -> int:
    print("[vstrain] registered (task/arch -> trainer):")
    for (task, arch), cls in sorted(TRAINER_REGISTRY.items()):
        print(f"  {task}/{arch} -> {cls.__name__}")
    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    names = list_presets(args.kind)
    print(f"[vstrain] presets({args.kind}): {', '.join(names) if names else '(none)'}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    out = build_trainer(cfg).run()
    print(f"[vstrain] done. artifacts: { {k: str(v) for k, v in out.items()} }")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    cfg = _resolve_config(args)
    ckpt = Path(args.ckpt)
    if not ckpt.exists():            # 재export 엔트리포인트 — ckpt 조기 검증
        print(f"[vstrain] ckpt not found: {ckpt}", file=sys.stderr)
        return 2
    out = build_trainer(cfg).export_to_vsc(ckpt)
    print(f"[vstrain] exported: { {k: str(v) for k, v in out.items()} }")
    return 0


def _add_cfg_args(p: argparse.ArgumentParser) -> None:
    """config 소스 인자(--config 또는 --preset+런타임 주입) 공통 추가."""
    p.add_argument("-c", "--config", help="전체 config yaml 경로")
    p.add_argument("-p", "--preset", help="train preset 이름 (`vstrain presets` 로 목록)")
    p.add_argument("--root", help="--preset 사용 시 dataset 루트(labelme)")
    p.add_argument("--names", help="--preset 사용 시 클래스명 CSV (0-base 순서)")
    p.add_argument("--out", help="run.out_dir 덮어쓰기")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vstrain",
                                description="VisionSuiteTrain — VSC 추론 컨트랙트 학습 코어")
    p.add_argument("-V", "--version", action="version", version=f"vstrain {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="prepare→train→export 전체 실행")
    _add_cfg_args(pt)
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("export", help="기존 ckpt → VSC 아티팩트만 재export")
    _add_cfg_args(pe)
    pe.add_argument("--ckpt", required=True)
    pe.set_defaults(func=_cmd_export)

    pv = sub.add_parser("validate", help="데이터셋 무결성 검사(학습 불요)")
    _add_cfg_args(pv)
    pv.set_defaults(func=_cmd_validate)

    pc = sub.add_parser("check", help="config 스키마/어댑터 매핑만 확인")
    _add_cfg_args(pc)
    pc.set_defaults(func=_cmd_check)

    pl = sub.add_parser("list", help="등록된 (task/arch) 어댑터 목록")
    pl.set_defaults(func=_cmd_list)

    pp = sub.add_parser("presets", help="동봉 preset 목록")
    pp.add_argument("--kind", default="train", choices=["train", "test", "export"])
    pp.set_defaults(func=_cmd_presets)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError, ValidationError) as e:
        # 예상 가능한 config/usage 오류는 깔끔한 메시지 + 종료코드(스택트레이스 X)
        print(f"[vstrain] error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
