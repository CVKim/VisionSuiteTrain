"""OCR 인식 데이터 — labels.txt(상대이미지경로<TAB>텍스트) → [(image_path, text)].

표준 OCR-rec 포맷(크롭 단어 이미지 + 전사). dataset.names = charset(문자 리스트).
"""
from __future__ import annotations

from pathlib import Path


def read_ocr_labels(root: str | Path, labels_file: str = "labels.txt") -> list[tuple[str, str]]:
    root = Path(root)
    p = root / labels_file
    rows: list[tuple[str, str]] = []
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        name, text = line.split("\t", 1)
        if not name.strip() or not text:
            continue
        img = root / name.strip()
        if img.exists():
            rows.append((str(img), text))
    return rows
