"""IR → mask PNG(seg). polygon→cv2.fillPoly(화소=class_id, 배경=background).

겹침은 그리기 순서(나중 region 이 덮음). 배경 0 == VSC seg_background_class.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..ir import Sample


def to_mask(sample: Sample, name2id: dict[str, int], background: int = 0) -> np.ndarray:
    """한 Sample → HxW uint8 class-id 맵."""
    H = max(1, sample.height)
    W = max(1, sample.width)
    mask = np.full((H, W), background, dtype=np.uint8)
    for r in sample.regions:
        if r.class_name not in name2id or len(r.points) < 3:
            continue
        cid = int(name2id[r.class_name])
        pts = np.array([[int(round(x)), int(round(y))] for x, y in r.points], dtype=np.int32)
        cv2.fillPoly(mask, [pts], cid)
    return mask


def write_masks(samples_by_split: dict[str, list[Sample]], names: list[str],
                out_root: str | Path, background: int = 0) -> Path:
    out_root = Path(out_root)
    name2id = {n: i for i, n in enumerate(names)}
    for split, samples in samples_by_split.items():
        m_dir = out_root / "masks" / split
        m_dir.mkdir(parents=True, exist_ok=True)
        for s in samples:
            m = to_mask(s, name2id, background)
            cv2.imwrite(str(m_dir / f"{Path(s.image_path).stem}.png"), m)
    return out_root
