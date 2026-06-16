"""vstrain CLI — train / export / validate / list / version.

registry 채우기 위해 trainers 패키지를 import(데코레이터 자동등록). heavy lib 은
어댑터 메서드 내부 lazy import 라 train/export 실행 시점에만 필요(validate/list/check 는 불요).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from . import trainers  # noqa: F401  (registry 자동등록 트리거)
from .config.schema import load_train_config
from .registry import TRAINER_REGISTRY, build_trainer


def _cmd_validate(args: argparse.Namespace) -> int:
    from .data import iter_labelme_dir, validate_samples, summarize
    cfg = load_train_config(args.config)
    samples = iter_labelme_dir(cfg.dataset.root)
    issues = validate_samples(samples, list(cfg.dataset.names), cfg.task)
    print(f"[vstrain] samples={len(samples)} {summarize(issues)}")
    return 1 if issues else 0


def _cmd_check(args: argparse.Namespace) -> int:
    cfg = load_train_config(args.config)
    trainer_cls = None
    for arch in (cfg.resolved_arch, cfg.arch):
        trainer_cls = TRAINER_REGISTRY.get((cfg.task, arch))
        if trainer_cls:
            break
    print(f"[vstrain] config OK: run={cfg.run.name} task={cfg.task} "
          f"arch={cfg.arch}(={cfg.resolved_arch}) names={len(cfg.dataset.names)} "
          f"adapter={'ok' if trainer_cls else 'MISSING'} out={cfg.run.out_dir}")
    return 0 if trainer_cls else 2


def _cmd_list(_args: argparse.Namespace) -> int:
    print("[vstrain] registered (task/arch -> trainer):")
    for (task, arch), cls in sorted(TRAINER_REGISTRY.items()):
        print(f"  {task}/{arch} -> {cls.__name__}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = load_train_config(args.config)
    out = build_trainer(cfg).run()
    print(f"[vstrain] done. artifacts: { {k: str(v) for k, v in out.items()} }")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    cfg = load_train_config(args.config)
    out = build_trainer(cfg).export_to_vsc(Path(args.ckpt))
    print(f"[vstrain] exported: { {k: str(v) for k, v in out.items()} }")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vstrain",
                                description="VisionSuiteTrain — VSC 추론 컨트랙트 학습 코어")
    p.add_argument("-V", "--version", action="version", version=f"vstrain {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="prepare→train→export 전체 실행")
    pt.add_argument("-c", "--config", required=True)
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("export", help="기존 ckpt → VSC 아티팩트만 재export")
    pe.add_argument("-c", "--config", required=True)
    pe.add_argument("--ckpt", required=True)
    pe.set_defaults(func=_cmd_export)

    pv = sub.add_parser("validate", help="데이터셋 무결성 검사(학습 불요)")
    pv.add_argument("-c", "--config", required=True)
    pv.set_defaults(func=_cmd_validate)

    pc = sub.add_parser("check", help="config 스키마/어댑터 매핑만 확인")
    pc.add_argument("-c", "--config", required=True)
    pc.set_defaults(func=_cmd_check)

    pl = sub.add_parser("list", help="등록된 (task/arch) 어댑터 목록")
    pl.set_defaults(func=_cmd_list)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
