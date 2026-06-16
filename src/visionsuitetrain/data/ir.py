"""중간표현(IR) — 모든 입력 포맷(labelme/coco/yolo)을 단일 형태로 정규화.

Reader(format→IR) 와 Writer(IR+task→출력) 를 직교 분리하는 허브.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Region:
    class_name: str
    shape_type: str                       # polygon|rectangle|circle|line|point|mask
    points: list[tuple[float, float]]      # 픽셀 좌표(원점 좌상단, 비정규화)
    group_id: Optional[int] = None         # 인스턴스 묶음(없으면 None)

    def aabb(self) -> tuple[float, float, float, float]:
        """축정렬 bbox (x, y, w, h) 픽셀.

        labelme circle 은 points=[center, edge] 2점(반지름=두 점 거리)이라
        min/max 로는 외접 사각형이 안 됨 → circle 은 (cx-r, cy-r, 2r, 2r) 로 계산.
        """
        if self.shape_type == "circle" and len(self.points) >= 2:
            (cx, cy), (ex, ey) = self.points[0], self.points[1]
            r = ((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5
            return cx - r, cy - r, 2 * r, 2 * r
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        return x0, y0, x1 - x0, y1 - y0


@dataclass
class Sample:
    image_path: str
    width: int
    height: int
    regions: list[Region] = field(default_factory=list)
    image_labels: list[str] = field(default_factory=list)   # 이미지 단위 라벨(classification)

    def bbox_regions(self) -> list[tuple[str, float, float, float, float]]:
        """검출용 — region 마다 (class_name, x, y, w, h) 픽셀 bbox."""
        out = []
        for r in self.regions:
            if not r.points:
                continue
            x, y, w, h = r.aabb()
            out.append((r.class_name, x, y, w, h))
        return out
