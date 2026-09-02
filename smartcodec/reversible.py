"""Integer reversible 5/3 lifting transform used by lossless mode.
TR: Tamsayı lifting ters dönüşümde bit-exact sonuç sağlar. / EN: Integer lifting enables bit-exact inversion.
"""

from __future__ import annotations

import numpy as np


def forward_53_1d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.int64).reshape(-1)
    even = signal[::2].copy()
    odd = signal[1::2].copy()
    if odd.size:
        index = np.arange(odd.size)
        right = even[np.minimum(index + 1, even.size - 1)]
        detail = odd - ((even[: odd.size] + right) // 2)
    else:
        detail = np.empty(0, dtype=np.int64)
    if detail.size:
        index = np.arange(even.size)
        left = detail[np.maximum(index - 1, 0)]
        right = detail[np.minimum(index, detail.size - 1)]
        approximation = even + ((left + right + 2) // 4)
    else:
        approximation = even.copy()
    return approximation, detail


def inverse_53_1d(approximation: np.ndarray, detail: np.ndarray) -> np.ndarray:
    approx = np.asarray(approximation, dtype=np.int64).reshape(-1)
    high = np.asarray(detail, dtype=np.int64).reshape(-1)
    if high.size:
        index = np.arange(approx.size)
        left = high[np.maximum(index - 1, 0)]
        right = high[np.minimum(index, high.size - 1)]
        even = approx - ((left + right + 2) // 4)
        odd = high + ((even[: high.size] + even[np.minimum(np.arange(high.size) + 1, even.size - 1)]) // 2)
    else:
        even = approx
        odd = np.empty(0, dtype=np.int64)
    output = np.empty(even.size + odd.size, dtype=np.int64)
    output[::2] = even
    output[1::2] = odd
    return output


def forward_53_2d(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    array = np.asarray(image, dtype=np.int64)
    height, width = array.shape
    low_width, high_width = (width + 1) // 2, width // 2
    row_low = np.empty((height, low_width), dtype=np.int64)
    row_high = np.empty((height, high_width), dtype=np.int64)
    for row in range(height):
        low, high = forward_53_1d(array[row])
        row_low[row, : low.size] = low
        row_high[row, : high.size] = high

    low_height, high_height = (height + 1) // 2, height // 2
    approximation = np.empty((low_height, low_width), dtype=np.int64)
    horizontal = np.empty((high_height, low_width), dtype=np.int64)
    vertical = np.empty((low_height, high_width), dtype=np.int64)
    diagonal = np.empty((high_height, high_width), dtype=np.int64)
    for column in range(low_width):
        low, high = forward_53_1d(row_low[:, column])
        approximation[:, column] = low
        horizontal[:, column] = high
    for column in range(high_width):
        low, high = forward_53_1d(row_high[:, column])
        vertical[:, column] = low
        diagonal[:, column] = high
    return approximation, (horizontal, vertical, diagonal)


def inverse_53_2d(
    approximation: np.ndarray,
    details: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    horizontal, vertical, diagonal = details
    low_height, low_width = approximation.shape
    high_height, high_width = horizontal.shape[0], vertical.shape[1]
    height, width = low_height + high_height, low_width + high_width
    row_low = np.empty((height, low_width), dtype=np.int64)
    row_high = np.empty((height, high_width), dtype=np.int64)
    for column in range(low_width):
        row_low[:, column] = inverse_53_1d(approximation[:, column], horizontal[:, column])
    for column in range(high_width):
        row_high[:, column] = inverse_53_1d(vertical[:, column], diagonal[:, column])
    output = np.empty((height, width), dtype=np.int64)
    for row in range(height):
        output[row] = inverse_53_1d(row_low[row], row_high[row])
    return output


def max_level_53(shape: tuple[int, int]) -> int:
    height, width = shape
    level = 0
    while min(height, width) >= 2:
        height, width = (height + 1) // 2, (width + 1) // 2
        level += 1
    return max(1, level)


def forward_53(image: np.ndarray, level: int) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    maximum = max_level_53(tuple(image.shape))
    if level < 1 or level > maximum:
        raise ValueError(f"5/3 level must be between 1 and {maximum}")
    current = np.asarray(image, dtype=np.int64)
    details = []
    for _ in range(level):
        current, detail = forward_53_2d(current)
        details.append(detail)
    return current, details


def inverse_53(
    approximation: np.ndarray,
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    shape: tuple[int, int],
) -> np.ndarray:
    current = np.asarray(approximation, dtype=np.int64)
    for detail in reversed(details):
        current = inverse_53_2d(current, detail)
    return current[: shape[0], : shape[1]]
