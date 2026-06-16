"""train/val/test 분할 — 비율(ratio) 또는 파일리스트(stem 기준)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

from .ir import Sample


def split_samples(samples: list[Sample],
                  split_cfg: Optional[dict[str, Any]],
                  seed: int = 42) -> dict[str, list[Sample]]:
    if not split_cfg:
        return {"train": list(samples)}

    # (a) 비율 — {ratio: {train,val,test}} 또는 {train: 0.8, ...}
    ratio = split_cfg.get("ratio") if "ratio" in split_cfg else (
        split_cfg if all(isinstance(v, (int, float)) for v in split_cfg.values()) else None)
    if ratio is not None:
        if not isinstance(ratio, dict):
            raise ValueError("split.ratio 는 {train,val,test} dict 여야 함")
        # 재현가능 셔플 — 파일명 정렬 순서를 그대로 자르면 시퀀셜 촬영에서 분포 편향/누수
        items = list(samples)
        random.Random(seed).shuffle(items)
        n = len(items)
        ntr = int(round(n * float(ratio.get("train", 0.8))))
        nva = int(round(n * float(ratio.get("val", 0.1))))
        return {
            "train": items[:ntr],
            "val": items[ntr:ntr + nva],
            "test": items[ntr + nva:],
        }

    # (b) 파일리스트 — split: {train: path.txt, val: ..., test: ...}
    out: dict[str, list[Sample]] = {}
    for split, listfile in split_cfg.items():
        p = Path(str(listfile))
        stems = set()
        if p.exists():
            stems = {Path(line.strip()).stem for line in p.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
        out[split] = [s for s in samples if Path(s.image_path).stem in stems]
    return out
