"""labelme JSON → IR(Sample).

labelme 필드: version, flags(이미지단위 bool=cls 약라벨), shapes[](label, points, shape_type,
group_id), imagePath, imageData(base64 PNG, null 가능), imageHeight, imageWidth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2

from ..ir import Sample, Region


def _read_img_size(p: Path) -> Optional[tuple[int, int]]:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    return w, h


def read_labelme(json_path: str | Path, image_root: Optional[str | Path] = None) -> Sample:
    jp = Path(json_path)
    d = json.loads(jp.read_text(encoding="utf-8"))

    W = int(d.get("imageWidth") or 0)
    H = int(d.get("imageHeight") or 0)
    img_rel = d.get("imagePath") or (jp.stem + ".jpg")
    root = Path(image_root) if image_root else jp.parent
    img_path = root / Path(img_rel).name

    # 실제 이미지 크기로 보강/대조(편집 후 리사이즈된 파일 → 좌표 깨짐 탐지)
    if img_path.exists():
        wh = _read_img_size(img_path)
        if wh:
            if not W or not H:
                W, H = wh
            # (W,H) 불일치는 validate 에서 경고 처리

    regions: list[Region] = []
    for s in d.get("shapes", []) or []:
        pts = [(float(x), float(y)) for x, y in (s.get("points") or [])]
        regions.append(Region(
            class_name=str(s.get("label", "")).strip(),
            shape_type=str(s.get("shape_type", "polygon")),
            points=pts,
            group_id=s.get("group_id"),
        ))

    image_labels = [k for k, v in (d.get("flags") or {}).items() if v]
    return Sample(image_path=str(img_path), width=W, height=H,
                  regions=regions, image_labels=image_labels)


def iter_labelme_dir(root: str | Path) -> list[Sample]:
    """폴더(재귀) 내 모든 *.json → Sample 리스트. 파싱 실패는 skip+경고."""
    out: list[Sample] = []
    for jp in sorted(Path(root).rglob("*.json")):
        try:
            out.append(read_labelme(jp))
        except Exception as e:  # noqa: BLE001
            print(f"[labelme] skip {jp}: {e}")
    return out
