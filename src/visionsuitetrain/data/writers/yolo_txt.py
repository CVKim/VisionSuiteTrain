"""IR → YOLO-txt(det) + data.yaml (ultralytics 호환).

라벨 변환(to_yolo_lines)은 순수 함수 → 단위테스트 대상.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from ..ir import Sample


def to_yolo_lines(sample: Sample, name2id: dict[str, int]) -> list[str]:
    """한 Sample → YOLO txt 라인 'cls cx cy w h'(0~1 정규화). 미지 클래스는 skip.

    부분적으로 화면 밖인 박스는 [0,W]x[0,H] 픽셀공간에서 먼저 clip 한 뒤 중심/크기를
    재계산한다(cx,cy 와 w,h 를 독립 클램프하면 위치가 평행이동하는 버그 방지).
    """
    W = max(1, sample.width)
    H = max(1, sample.height)
    lines: list[str] = []
    for cls, x, y, w, h in sample.bbox_regions():
        if cls not in name2id:
            continue
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(W), x + w), min(float(H), y + h)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:        # 완전히 화면 밖 → skip
            continue
        cx = (x0 + bw / 2) / W
        cy = (y0 + bh / 2) / H
        lines.append(f"{name2id[cls]} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
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
