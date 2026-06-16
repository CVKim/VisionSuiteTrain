"""IR → YOLO-txt(det) + data.yaml (ultralytics 호환).

라벨 변환(to_yolo_lines)은 순수 함수 → 단위테스트 대상.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from ..ir import Sample


def to_yolo_lines(sample: Sample, name2id: dict[str, int]) -> list[str]:
    """한 Sample → YOLO txt 라인 'cls cx cy w h'(0~1 정규화). 미지 클래스는 skip."""
    W = max(1, sample.width)
    H = max(1, sample.height)
    lines: list[str] = []
    for cls, x, y, w, h in sample.bbox_regions():
        if cls not in name2id:
            continue
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        ww, hh = w / W, h / H
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        ww, hh = min(max(ww, 0.0), 1.0), min(max(hh, 0.0), 1.0)
        lines.append(f"{name2id[cls]} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
    return lines


def write_yolo(samples_by_split: dict[str, list[Sample]], names: list[str],
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
                "\n".join(to_yolo_lines(s, name2id)), encoding="utf-8")
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
