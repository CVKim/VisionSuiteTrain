"""IR → mask PNG(seg). polygon→cv2.fillPoly(화소=class_id, 배경=background).

겹침은 그리기 순서(나중 region 이 덮음). 배경 0 == VSC seg_background_class.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..ir import Sample


def to_mask(sample: Sample, name2id: dict[str, int], background: int = 0) -> np.ndarray:
    """한 Sample → HxW uint8 class-id 맵. shape_type 별 렌더(polygon/rectangle/circle).

    polygon 만 처리하던 과거 동작은 rectangle/circle seg 영역을 조용히 누락시켰음 →
    shape_type 분기로 렌더하고, 지원 못 하는 도형은 경고(silent drop 방지).
    """
    H = max(1, sample.height)
    W = max(1, sample.width)
    mask = np.full((H, W), background, dtype=np.uint8)
    for r in sample.regions:
        if r.class_name not in name2id or not r.points:
            continue
        cid = int(name2id[r.class_name])
        st = r.shape_type
        if st == "rectangle" and len(r.points) == 2:
            (x0, y0), (x1, y1) = r.points[0], r.points[1]
            cv2.rectangle(mask, (int(round(x0)), int(round(y0))),
                          (int(round(x1)), int(round(y1))), cid, thickness=-1)
        elif st == "circle" and len(r.points) == 2:
            (cx, cy), (ex, ey) = r.points[0], r.points[1]
            rad = int(round(((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5))
            cv2.circle(mask, (int(round(cx)), int(round(cy))), rad, cid, thickness=-1)
        elif len(r.points) >= 3:      # polygon
            pts = np.array([[int(round(x)), int(round(y))] for x, y in r.points], dtype=np.int32)
            cv2.fillPoly(mask, [pts], cid)
        else:
            print(f"[mask] skip unsupported region shape={st} pts={len(r.points)} "
                  f"({sample.image_path})")
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
