"""writers 순수-코어 함수 검증: yolo 정규화 / mask fillPoly / imagefolder 라벨링."""
from __future__ import annotations

import numpy as np

from visionsuitetrain.data.ir import Sample, Region
from visionsuitetrain.data.writers.yolo_txt import to_yolo_lines
from visionsuitetrain.data.writers.mask_png import to_mask
from visionsuitetrain.data.writers.imagefolder import label_of


def test_to_yolo_lines_normalization():
    s = Sample(image_path="x.jpg", width=200, height=100, regions=[
        Region("scratch", "rectangle", [(50, 25), (150, 75)]),   # 중심(100,50), wh(100,50)
    ])
    lines = to_yolo_lines(s, {"scratch": 0, "dent": 1})
    assert len(lines) == 1
    cid, cx, cy, w, h = lines[0].split()
    assert cid == "0"
    assert abs(float(cx) - 0.5) < 1e-6     # 100/200
    assert abs(float(cy) - 0.5) < 1e-6     # 50/100
    assert abs(float(w) - 0.5) < 1e-6      # 100/200
    assert abs(float(h) - 0.5) < 1e-6      # 50/100


def test_to_yolo_lines_skips_unknown_class():
    s = Sample(image_path="x.jpg", width=100, height=100, regions=[
        Region("ghost", "rectangle", [(0, 0), (10, 10)]),
    ])
    assert to_yolo_lines(s, {"scratch": 0}) == []


def test_to_mask_fillpoly_class_id():
    s = Sample(image_path="x.jpg", width=50, height=50, regions=[
        Region("crack", "polygon", [(10, 10), (40, 10), (40, 40), (10, 40)]),
    ])
    # names: background=0, crack=1
    m = to_mask(s, {"background": 0, "crack": 1}, background=0)
    assert m.shape == (50, 50)
    assert m.dtype == np.uint8
    assert m[25, 25] == 1                   # 폴리곤 내부 = crack id
    assert m[0, 0] == 0                      # 배경


def test_label_of_priority_and_ambiguity():
    names = ["ok", "ng"]
    # 이미지단위 플래그 우선
    s1 = Sample("x.jpg", 10, 10, image_labels=["ok"])
    assert label_of(s1, names) == "ok"
    # 단일 region.class_name
    s2 = Sample("x.jpg", 10, 10, regions=[Region("ng", "polygon", [(0, 0), (1, 1), (2, 2)])])
    assert label_of(s2, names) == "ng"
    # 모호(두 클래스 region) → None
    s3 = Sample("x.jpg", 10, 10, regions=[
        Region("ok", "polygon", [(0, 0), (1, 1), (2, 2)]),
        Region("ng", "polygon", [(3, 3), (4, 4), (5, 5)]),
    ])
    assert label_of(s3, names) is None
