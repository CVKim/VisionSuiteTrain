"""labelme JSON → IR(Sample) 파싱 (이미지 파일 불요: imageWidth/Height 사용)."""
from __future__ import annotations

import json
from pathlib import Path

from visionsuitetrain.data import read_labelme, iter_labelme_dir


def _write_labelme(p: Path, *, with_flags: bool = False) -> None:
    doc = {
        "version": "5.0.0",
        "flags": {"ok": True, "ng": False} if with_flags else {},
        "shapes": [
            {"label": "crack", "shape_type": "polygon",
             "points": [[10, 10], [40, 10], [40, 40], [10, 40]], "group_id": None},
            {"label": "dent", "shape_type": "rectangle",
             "points": [[50, 50], [70, 80]], "group_id": 1},
        ],
        "imagePath": "img0.jpg",
        "imageData": None,
        "imageHeight": 128,
        "imageWidth": 256,
    }
    p.write_text(json.dumps(doc), encoding="utf-8")


def test_read_labelme_basic(tmp_path):
    jp = tmp_path / "img0.json"
    _write_labelme(jp)
    s = read_labelme(jp)
    assert s.width == 256 and s.height == 128
    assert len(s.regions) == 2
    assert s.regions[0].class_name == "crack"
    assert s.regions[0].shape_type == "polygon"
    assert s.regions[1].group_id == 1
    # aabb on rectangle 코너
    x, y, w, h = s.regions[1].aabb()
    assert (x, y, w, h) == (50, 50, 20, 30)


def test_read_labelme_image_flags(tmp_path):
    jp = tmp_path / "img1.json"
    _write_labelme(jp, with_flags=True)
    s = read_labelme(jp)
    assert s.image_labels == ["ok"]   # True 인 플래그만


def test_iter_labelme_dir_recursive(tmp_path):
    (tmp_path / "a").mkdir()
    _write_labelme(tmp_path / "a" / "x.json")
    _write_labelme(tmp_path / "y.json")
    samples = iter_labelme_dir(tmp_path)
    assert len(samples) == 2
