"""IR → ImageFolder(cls). 이미지 단위 라벨로 root/<split>/<class>/<img> 구성.

라벨 출처 우선순위: image_labels(flags) → 단일 region.class_name. 모호하면 skip.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ..ir import Sample


def label_of(sample: Sample, names: list[str]) -> Optional[str]:
    nameset = set(names)
    for lab in sample.image_labels:
        if lab in nameset:
            return lab
    cls = {r.class_name for r in sample.regions if r.class_name in nameset}
    if len(cls) == 1:
        return next(iter(cls))
    return None


def write_imagefolder(samples_by_split: dict[str, list[Sample]], names: list[str],
                      out_root: str | Path) -> Path:
    out_root = Path(out_root)
    for split, samples in samples_by_split.items():
        for s in samples:
            lab = label_of(s, names)
            if lab is None:
                continue
            d = out_root / split / lab
            d.mkdir(parents=True, exist_ok=True)
            if Path(s.image_path).exists():
                dst = d / Path(s.image_path).name
                if not dst.exists():
                    try:
                        shutil.copy2(s.image_path, dst)
                    except OSError:
                        pass
    return out_root
