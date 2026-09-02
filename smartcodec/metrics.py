"""Image quality and compression metrics.
TR: Metrikler encode/decode sonuçlarını ortak ve tekrar üretilebilir ölçekte karşılaştırır.
EN: Metrics compare encode/decode results on a shared reproducible scale.
"""

from __future__ import annotations

import math

import numpy as np


def mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    return float(np.mean(difference * difference))


def psnr(reference: np.ndarray, candidate: np.ndarray, data_range: float = 255.0) -> float:
    error = mse(reference, candidate)
    return float("inf") if error == 0 else float(10.0 * math.log10((data_range**2) / error))


def _luma(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        return array
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def ssim(reference: np.ndarray, candidate: np.ndarray, data_range: float = 255.0) -> float:
    """Global SSIM, deliberately dependency-free and suitable for comparisons."""
    x = _luma(reference)
    y = _luma(candidate)
    if x.shape != y.shape:
        raise ValueError("Images must have the same shape")
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    covariance = np.mean((x - mean_x) * (y - mean_y))
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    return float(numerator / denominator) if denominator else 1.0


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes <= 0:
        return float("inf")
    return float(original_bytes / compressed_bytes)


def bits_per_pixel(compressed_bytes: int, shape: tuple[int, ...]) -> float:
    pixels = int(shape[0] * shape[1])
    return float(compressed_bytes * 8 / pixels) if pixels else 0.0


def masked_psnr(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    mask_array = np.asarray(mask, dtype=bool)
    if reference.ndim == 3:
        if mask_array.shape != reference.shape[:2]:
            raise ValueError("Mask must match the image height and width")
        reference = np.asarray(reference)[mask_array]
        candidate = np.asarray(candidate)[mask_array]
    else:
        reference = np.asarray(reference)[mask_array]
        candidate = np.asarray(candidate)[mask_array]
    if not np.any(mask_array):
        return float("nan")
    return psnr(reference, candidate)


def region_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return ROI and background quality metrics for semantic compression."""
    mask_array = np.asarray(mask, dtype=bool)
    if reference.shape[:2] != mask_array.shape or candidate.shape[:2] != mask_array.shape:
        raise ValueError("Mask must match both image height and width")

    def select(array: np.ndarray, selected: np.ndarray) -> np.ndarray:
        return np.asarray(array)[selected]

    inverse = ~mask_array
    roi_reference, roi_candidate = select(reference, mask_array), select(candidate, mask_array)
    background_reference, background_candidate = select(reference, inverse), select(candidate, inverse)

    def values(ref: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
        if ref.size == 0:
            return float("nan"), float("nan"), float("nan")
        return mse(ref, pred), psnr(ref, pred), ssim(ref, pred)

    roi_mse, roi_psnr, roi_ssim = values(roi_reference, roi_candidate)
    bg_mse, bg_psnr, bg_ssim = values(background_reference, background_candidate)
    return {
        "roi_mse": roi_mse,
        "roi_psnr": roi_psnr,
        "roi_ssim": roi_ssim,
        "background_mse": bg_mse,
        "background_psnr": bg_psnr,
        "background_ssim": bg_ssim,
    }
