"""I/O, config, and logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


SUPPORTED_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def get_logger(name: str = "anomaly_inspector", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    return paths


def load_gray(path: str | Path) -> np.ndarray:
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"failed to read image: {path}")
    return img


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Unicode-path-safe replacement for cv2.imread.

    cv2.imread on Windows decodes the path through the active ANSI codepage,
    which mangles non-ASCII characters (Korean, Japanese, etc.) and silently
    fails. Reading the bytes via numpy and decoding in-memory dodges the
    issue without changing call sites.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, img: np.ndarray,
                    params: list[int] | None = None) -> bool:
    """Unicode-safe replacement for cv2.imwrite. Encodes via the file
    extension and writes raw bytes, bypassing OpenCV's ANSI path handling."""
    p = Path(path)
    ext = p.suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(p))
    return True


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_reference(path: str | Path, master: np.ndarray, tolerance: np.ndarray,
                   meta: dict[str, Any] | None = None) -> None:
    """Persist master + tolerance map (and a meta dict) as a single npz."""
    ensure_dir(Path(path).parent)
    np.savez_compressed(
        path,
        master=master.astype(np.float32),
        tolerance=tolerance.astype(np.float32),
        meta=np.array([yaml.safe_dump(meta or {})], dtype=object),
    )


def load_reference(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    master = data["master"].astype(np.float32)
    tolerance = data["tolerance"].astype(np.float32)
    meta_raw = data["meta"][0] if "meta" in data.files else "{}"
    meta = yaml.safe_load(meta_raw) or {}
    return master, tolerance, meta


def stack_images(images: Iterable[np.ndarray]) -> np.ndarray:
    arr = list(images)
    if not arr:
        raise ValueError("empty image list")
    shapes = {a.shape for a in arr}
    if len(shapes) != 1:
        raise ValueError(f"shape mismatch in stack: {shapes}")
    return np.stack(arr, axis=0)
