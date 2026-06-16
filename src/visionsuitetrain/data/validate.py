"""데이터 무결성 검증 — class∈names / 라벨 누락 / 좌표 기하 / 이미지 크기 / task별 빈 샘플."""
from __future__ import annotations

from pathlib import Path

import cv2

from .ir import Sample

# shape_type → 기대 점 개수(이 외는 bad_geometry). polygon 은 ≥3.
_EXPECT_PTS = {"rectangle": 2, "circle": 2, "line": 2, "point": 1}


def validate_samples(samples: list[Sample], names: list[str], task: str,
                     check_image_size: bool = True) -> dict:
    """문제 카테고리별 리스트 반환(빈 dict 면 클린). 호출부가 fail/warn 정책 결정."""
    nameset = set(names)
    issues: dict[str, list] = {
        "unknown_class": [], "missing_label": [], "missing_image": [],
        "size_mismatch": [], "empty": [], "bad_geometry": [],
    }
    for s in samples:
        if not Path(s.image_path).exists():
            issues["missing_image"].append(s.image_path)
        for r in s.regions:
            if not r.class_name:
                issues["missing_label"].append(s.image_path)
            elif r.class_name not in nameset:
                issues["unknown_class"].append((s.image_path, r.class_name))
            if r.shape_type == "polygon":
                if len(r.points) < 3:
                    issues["bad_geometry"].append((s.image_path, "polygon<3pts"))
            elif r.shape_type in _EXPECT_PTS and len(r.points) != _EXPECT_PTS[r.shape_type]:
                issues["bad_geometry"].append(
                    (s.image_path, f"{r.shape_type}!={_EXPECT_PTS[r.shape_type]}pts"))
        # task 별 '빈 샘플' 기준: cls 는 이미지라벨/region, det·seg 는 region 필요
        empty = (not s.image_labels and not s.regions) if task == "classification" \
            else (not s.regions)
        if empty:
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
