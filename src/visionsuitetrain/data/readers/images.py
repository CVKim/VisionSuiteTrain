"""이미지 폴더 → 이미지 경로 리스트 (재귀 glob). 비지도 AD / 이미지폴더 입력용.

라벨 불필요한 경로(이상탐지 정상셋, classification imagefolder 등)에서 이미지를 직접 수집.
"""
from __future__ import annotations

from pathlib import Path

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: str | Path, recursive: bool = True) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(str(p) for p in it if p.is_file() and p.suffix.lower() in _EXTS)


def mvtec_normal_dir(root: str | Path) -> Path:
    """MVTec-AD 레이아웃(<category>/train/good)이면 정상셋 디렉터리, 아니면 root 자체."""
    good = Path(root) / "train" / "good"
    return good if good.is_dir() else Path(root)
