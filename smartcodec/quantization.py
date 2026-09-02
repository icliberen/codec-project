"""Scalar quantization helpers, including an optional ROI quality map.
TR: Step ve ROI kalite haritası katsayı nicemlemesini belirler.
EN: Step and ROI quality maps control coefficient quantization.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(mask.shape) == tuple(shape):
        return np.asarray(mask, dtype=np.float32)
    image = Image.fromarray(np.asarray(mask, dtype=np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.BOX), dtype=np.float32)


def coefficient_steps(
    base_step: float,
    shape: tuple[int, int],
    roi_mask: np.ndarray | None = None,
    roi_strength: float = 0.65,
    level: int = 0,
    detail: bool = False,
    step_multiplier: float = 1.0,
) -> np.ndarray:
    if base_step <= 0:
        raise ValueError("Quantization step must be positive")
    if step_multiplier <= 0:
        raise ValueError("Step multiplier must be positive")
    steps = np.full(shape, float(base_step) * float(step_multiplier), dtype=np.float32)
    if detail:
        steps *= 1.0 + 0.08 * level
    if roi_mask is not None:
        strength = float(np.clip(roi_strength, 0.0, 0.95))
        local = np.clip(resize_mask(roi_mask, shape), 0.0, 1.0)
        steps *= 1.0 - strength * local
        steps = np.maximum(steps, base_step * 0.05)
    return steps


def quantize(coefficients: np.ndarray, steps: np.ndarray) -> np.ndarray:
    return np.rint(np.asarray(coefficients, dtype=np.float32) / steps).astype(np.int32)


def dequantize(values: np.ndarray, steps: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * steps
