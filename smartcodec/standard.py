"""Standard Pillow JPEG/JPEG2000 file I/O kept separate from experimental SWC.
TR: Standart dosya akışı deneysel SWC container'ından bilinçli olarak ayrıdır.
EN: Standard file I/O is deliberately separate from the experimental SWC container.
"""

from __future__ import annotations

import json
import io
from pathlib import Path

import numpy as np
from PIL import Image


def available_standard_codecs() -> dict[str, bool]:
    result = {"jpeg": True, "jpeg2000": False}
    try:
        Image.init()
        if "JPEG2000" in Image.SAVE and "JPEG2000" in Image.OPEN:
            probe = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
            buffer = io.BytesIO()
            probe.save(buffer, format="JPEG2000", quality_mode="rates", quality_layers=[4.0])
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                decoded.load()
            result["jpeg2000"] = True
    except Exception:
        pass
    return result


def encode_standard(image: np.ndarray, destination: str | Path, *, codec: str = "jpeg",
                    quality: int = 75, rate: float = 4.0) -> dict:
    codec = codec.lower()
    if codec not in {"jpeg", "jpeg2000"}:
        raise ValueError("codec must be jpeg or jpeg2000")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_array = np.asarray(image)
    if source_array.ndim not in (2, 3) or (source_array.ndim == 3 and source_array.shape[2] not in (3, 4)):
        raise ValueError("Standart codec gri tonlamali veya RGB bir görüntü bekliyor")
    # JPEG is an 8-bit format.  Normalise high-bit-depth inputs explicitly so
    # the saved file is a real JPEG instead of relying on Pillow's mode error.
    array = source_array[..., :3] if source_array.ndim == 3 else source_array
    if array.dtype == np.uint16:
        array = np.rint(array.astype(np.float32) / 257.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    image_object = Image.fromarray(array)
    try:
        if codec == "jpeg":
            image_object.save(output, format="JPEG", quality=int(np.clip(quality, 1, 100)))
        else:
            image_object.save(output, format="JPEG2000", quality_mode="rates", quality_layers=[float(rate)])
    except (OSError, ValueError) as exc:
        if codec == "jpeg2000":
            raise RuntimeError("JPEG2000 is unavailable; install a Pillow build with OpenJPEG support") from exc
        raise
    file_size = output.stat().st_size
    return {"codec": codec, "output": str(output), "file_size": file_size,
            "original_bytes": int(source_array.nbytes),
            "compression_ratio": float(source_array.nbytes / max(1, file_size)),
            "bits_per_pixel": float(file_size * 8 / max(1, array.shape[0] * array.shape[1])),
            "encoder": "Pillow/OpenJPEG" if codec == "jpeg2000" else "Pillow",
            "quality": quality if codec == "jpeg" else None, "rate": rate if codec == "jpeg2000" else None}


def decode_standard(source: str | Path) -> np.ndarray:
    with Image.open(source) as image:
        return np.asarray(image.convert("L" if image.mode in {"1", "L", "I", "I;16", "F"} else "RGB")).copy()


def standard_roundtrip(source: str | Path, destination: str | Path, *, codec: str = "jpeg", **kwargs) -> dict:
    decoded = decode_standard(source)
    info = encode_standard(decoded, destination, codec=codec, **kwargs)
    info["decoded_shape"] = list(decode_standard(destination).shape)
    return info
