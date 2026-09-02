"""Image loading and saving with Pillow.
TR: Kanal, dtype ve renk biçimi sınırları burada normalize edilir.
EN: Channel, dtype, and color-format boundaries are normalized here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_INPUTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as uint8/uint16 HxW or uint8 HxWx3 RGB array."""
    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_INPUTS:
        raise ValueError(f"Unsupported image extension: {source.suffix}")
    with Image.open(source) as image:
        if image.mode == "I;16":
            return np.asarray(image, dtype=np.uint16).copy()
        if image.mode in {"1", "L", "I", "F"}:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save a uint8 image, creating its parent directory when needed."""
    array = np.asarray(image)
    if array.dtype not in (np.uint8, np.uint16):
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)):
        raise ValueError("Expected HxW grayscale or HxWx3/HxWx4 image")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A non-contiguous NumPy view can otherwise produce an invalid/empty
    # image in some Pillow versions.  Keep all writes on a contiguous copy.
    Image.fromarray(np.ascontiguousarray(array)).save(destination)


def to_float(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def from_float(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)
