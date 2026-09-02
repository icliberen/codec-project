"""DWT and JPEG-like block DCT transforms.
TR: Dönüşüm katmanı codec'lerden bağımsız matematiksel primitive'ler sağlar.
EN: The transform layer provides codec-independent mathematical primitives.
"""

from __future__ import annotations

import numpy as np
import pywt


WAVELET_ALIASES = {
    "haar": "haar",
    "db1": "db1",
    "db4": "db4",
    "daubechies4": "db4",
    "db8": "db8",
    "daubechies8": "db8",
    "db12": "db12",
    "daubechies12": "db12",
    # The qmf option is the real PyWavelets biorthogonal PR wavelet backend.
    "qmf": "bior2.2",
    "4-band-pr-qmf": "bior2.2",
    "bior2.2": "bior2.2",
}
WAVELET_BOUNDARY_MODE = "periodization"
PRQMF4_DESCRIPTION = (
    "PyWavelets bior2.2 separable four-subband PR-QMF wavelet path; "
    "legacy 4x4 SWC payloads remain decodable."
)


def canonical_wavelet(name: str) -> str:
    key = name.strip().lower()
    if key not in WAVELET_ALIASES:
        supported = ", ".join(sorted(WAVELET_ALIASES))
        raise ValueError(f"Unknown wavelet {name!r}. Supported names: {supported}")
    return WAVELET_ALIASES[key]


def dwt2(image: np.ndarray, wavelet: str, level: int) -> list:
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 1:
        raise ValueError("DWT expects a non-empty two-dimensional plane")
    if level < 1:
        raise ValueError("DWT level must be at least 1")
    name = canonical_wavelet(wavelet)
    allowed = pywt.dwt_max_level(min(image.shape), pywt.Wavelet(name).dec_len)
    if level > allowed:
        raise ValueError(f"Level {level} is too high for shape {image.shape}; maximum is {allowed}")
    return pywt.wavedec2(np.asarray(image, dtype=np.float32), name, mode=WAVELET_BOUNDARY_MODE, level=level)


def idwt2(coefficients: list, wavelet: str, shape: tuple[int, int]) -> np.ndarray:
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError("DWT output shape must be positive (height, width)")
    reconstructed = pywt.waverec2(coefficients, canonical_wavelet(wavelet), mode=WAVELET_BOUNDARY_MODE)
    return np.asarray(reconstructed[: shape[0], : shape[1]], dtype=np.float32)


def _dct_matrix() -> np.ndarray:
    matrix = np.empty((8, 8), dtype=np.float32)
    for u in range(8):
        alpha = (1 / 8) ** 0.5 if u == 0 else (2 / 8) ** 0.5
        for x in range(8):
            matrix[u, x] = alpha * np.cos((2 * x + 1) * u * np.pi / 16)
    return matrix


DCT_MATRIX = _dct_matrix()


def block_dct(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply an 8x8 orthonormal DCT, padding the input by edge replication."""
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape
    padded_shape = ((height + 7) // 8 * 8, (width + 7) // 8 * 8)
    padded = np.pad(array, ((0, padded_shape[0] - height), (0, padded_shape[1] - width)), mode="edge")
    output = np.empty_like(padded)
    for row in range(0, padded.shape[0], 8):
        for column in range(0, padded.shape[1], 8):
            block = padded[row : row + 8, column : column + 8] - 128.0
            output[row : row + 8, column : column + 8] = DCT_MATRIX @ block @ DCT_MATRIX.T
    return output, padded_shape


def block_idct(coefficients: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(coefficients, dtype=np.float32)
    output = np.empty_like(array)
    for row in range(0, array.shape[0], 8):
        for column in range(0, array.shape[1], 8):
            block = array[row : row + 8, column : column + 8]
            output[row : row + 8, column : column + 8] = DCT_MATRIX.T @ block @ DCT_MATRIX + 128.0
    return output[: shape[0], : shape[1]]


def _prqmf4_matrix() -> np.ndarray:
    matrix = np.empty((4, 4), dtype=np.float32)
    for band in range(4):
        alpha = 0.5 if band == 0 else (0.5 * 2**0.5)
        for sample in range(4):
            matrix[band, sample] = alpha * np.cos((2 * sample + 1) * band * np.pi / 8)
    return matrix


PRQMF4_MATRIX = _prqmf4_matrix()


def block_prqmf4(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Legacy 4x4 PR-QMF analysis bank for old SWC payload decoding.

    New PR-QMF4 files use :func:`dwt2` with ``bior2.2`` instead. This helper is
    retained only to decode containers created by the previous implementation.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("PR-QMF4 expects a non-empty two-dimensional plane")
    height, width = array.shape
    padded_shape = ((height + 3) // 4 * 4, (width + 3) // 4 * 4)
    padded = np.pad(array, ((0, padded_shape[0] - height), (0, padded_shape[1] - width)), mode="edge")
    output = np.empty_like(padded)
    for row in range(0, padded.shape[0], 4):
        for column in range(0, padded.shape[1], 4):
            block = padded[row : row + 4, column : column + 4]
            output[row : row + 4, column : column + 4] = PRQMF4_MATRIX @ block @ PRQMF4_MATRIX.T
    return output, padded_shape


def block_iprqmf4(coefficients: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of the legacy block PR-QMF path; new files use ``bior2.2``."""
    array = np.asarray(coefficients, dtype=np.float32)
    output = np.empty_like(array)
    for row in range(0, array.shape[0], 4):
        for column in range(0, array.shape[1], 4):
            block = array[row : row + 4, column : column + 4]
            output[row : row + 4, column : column + 4] = PRQMF4_MATRIX.T @ block @ PRQMF4_MATRIX
    return output[: shape[0], : shape[1]]
