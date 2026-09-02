"""Simple block-DCT baseline used to explain and compare JPEG-style coding.
TR: 8x8 DCT eğitimsel JPEG-benzeri karşılaştırma primitive'idir.
EN: The 8x8 DCT is an educational JPEG-like comparison primitive.
"""

from __future__ import annotations

import numpy as np

from .image_io import from_float, to_float
from .transforms import block_dct, block_idct


JPEG_LUMA_QUANTIZATION = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], dtype=np.float32,
)


def jpeg_like_roundtrip(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """Run a transparent JPEG-style DCT/quantization/IDCT baseline."""
    quality = int(np.clip(quality, 1, 100))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    quantization = np.floor((JPEG_LUMA_QUANTIZATION * scale + 50) / 100)
    quantization = np.clip(quantization, 1, 255)
    array = to_float(image)
    channels = 1 if array.ndim == 2 else array.shape[2]
    planes = [array] if channels == 1 else [array[..., index] for index in range(channels)]
    reconstructed = []
    for plane in planes:
        coefficients, padded_shape = block_dct(plane)
        quantized = np.rint(coefficients / quantization.repeat(padded_shape[0] // 8, 0).repeat(padded_shape[1] // 8, 1))
        restored = quantized * quantization.repeat(padded_shape[0] // 8, 0).repeat(padded_shape[1] // 8, 1)
        reconstructed.append(block_idct(restored, plane.shape))
    result = reconstructed[0] if channels == 1 else np.stack(reconstructed, axis=-1)
    return from_float(result)
