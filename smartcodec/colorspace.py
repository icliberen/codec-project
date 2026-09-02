"""Full-range RGB/YCbCr conversion used by the lossy color path.
TR: Renk dönüşümü yalnızca kayıplı renk akışında kullanılır.
EN: Color conversion is used only by the lossy color path.
"""

from __future__ import annotations

import numpy as np


def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3")
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    return np.stack(
        [
            0.299000 * red + 0.587000 * green + 0.114000 * blue,
            -0.168736 * red - 0.331264 * green + 0.500000 * blue + 128.0,
            0.500000 * red - 0.418688 * green - 0.081312 * blue + 128.0,
        ],
        axis=-1,
    )


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("YCbCr image must have shape HxWx3")
    y, cb, cr = array[..., 0], array[..., 1] - 128.0, array[..., 2] - 128.0
    return np.stack(
        [
            y + 1.402000 * cr,
            y - 0.344136 * cb - 0.714136 * cr,
            y + 1.772000 * cb,
        ],
        axis=-1,
    )
