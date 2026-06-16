"""IR → YOLO-OBB txt(obb det) + data.yaml (ultralytics OBB 호환).

라벨 변환(to_obb_lines)은 순수 함수 → 단위테스트 대상.
ultralytics OBB 포맷: 'cls x1 y1 x2 y2 x3 y3 x4 y4' (4 코너, 0~1 정규화).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from ..ir import Sample


def _corners(r) -> list[tuple[float, float]] | None:
    """Region → 4 코너점(픽셀). rectangle(2점)=축정렬 코너, polygon(≥4점)=앞 4점."""
    pts = r.points
    if r.shape_type == "rectangle" and len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if len(pts) >= 4:
        return list(pts[:4])
    return None


def to_obb_lines(sample: Sample, name2id: dict[str, int]) -> list[str]:
    """한 Sample → YOLO-OBB 라인(0~1 정규화 코너 8값). 미지 클래스/부적합 도형은 skip."""
    W = max(1, sample.width)
    H = max(1, sample.height)
    lines: list[str] = []
    for r in sample.regions:
        if r.class_name not in name2id:
            continue
        corners = _corners(r)
        if corners is None:
            continue
        coords = []
        for x, y in corners:
            coords.append(min(max(x / W, 0.0), 1.0))
            coords.append(min(max(y / H, 0.0), 1.0))
        lines.append(f"{name2id[r.class_name]} " + " ".join(f"{c:.6f}" for c in coords))
    return lines


def write_yolo_obb(samples_by_split: dict[str, list[Sample]], names: list[str],
                   out_root: str | Path, link_images: bool = True) -> Path:
    out_root = Path(out_root)
    name2id = {n: i for i, n in enumerate(names)}
    for split, samples in samples_by_split.items():
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for s in samples:
            stem = Path(s.image_path).stem
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(to_obb_lines(s, name2id)), encoding="utf-8")
            if link_images and Path(s.image_path).exists():
                dst = img_dir / Path(s.image_path).name
                if not dst.exists():
                    try:
                        shutil.copy2(s.image_path, dst)
                    except OSError:
                        pass
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
        "nc": len(names),
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_yaml
