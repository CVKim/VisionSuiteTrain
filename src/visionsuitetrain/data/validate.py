"""데이터 무결성 검증 — class∈names / 좌표 / 이미지-라벨 짝 / id 범위."""
from __future__ import annotations

from pathlib import Path

import cv2

from .ir import Sample


def validate_samples(samples: list[Sample], names: list[str], task: str,
                     check_image_size: bool = True) -> dict:
    """문제 카테고리별 리스트 반환(빈 dict 면 클린). 호출부가 fail/warn 정책 결정."""
    nameset = set(names)
    issues: dict[str, list] = {
        "unknown_class": [], "missing_image": [], "size_mismatch": [],
        "empty": [], "bad_geometry": [],
    }
    for s in samples:
        if not Path(s.image_path).exists():
            issues["missing_image"].append(s.image_path)
        for r in s.regions:
            if r.class_name and r.class_name not in nameset:
                issues["unknown_class"].append((s.image_path, r.class_name))
            if r.shape_type == "polygon" and len(r.points) < 3:
                issues["bad_geometry"].append((s.image_path, "polygon<3pts"))
            if r.shape_type == "rectangle" and len(r.points) != 2:
                issues["bad_geometry"].append((s.image_path, "rect!=2pts"))
        if not s.regions and not s.image_labels:
            issues["empty"].append(s.image_path)
        if check_image_size and Path(s.image_path).exists() and s.width and s.height:
            im = cv2.imread(s.image_path)
            if im is not None:
                h, w = im.shape[:2]
                if (w, h) != (s.width, s.height):
                    issues["size_mismatch"].append((s.image_path, (s.width, s.height), (w, h)))
    return {k: v for k, v in issues.items() if v}


def summarize(issues: dict) -> str:
    if not issues:
        return "[validate] clean"
    return "[validate] " + ", ".join(f"{k}={len(v)}" for k, v in issues.items())
