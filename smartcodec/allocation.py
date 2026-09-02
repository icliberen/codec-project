"""Deterministic subband bit allocation strategies.

The ``dp`` strategy is an exact multiple-choice dynamic program for the
declared finite candidate model.  Its rate axis is integerized in one-bit
units, so the optimality claim is explicit and reproducible rather than an
unqualified claim about the continuous codec design space.
"""

from __future__ import annotations

import numpy as np


DEFAULT_MULTIPLIERS = (0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
RATE_UNIT_BITS = 1.0


def _entropy_bits(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0.0
    _, counts = np.unique(flat, return_counts=True)
    probabilities = counts / flat.size
    return float(np.sum(-counts * np.log2(probabilities)))


def entropy_bits(values: np.ndarray) -> float:
    """Return the deterministic zero-order entropy estimate used by the allocator."""
    return _entropy_bits(np.asarray(values))


def choose_step_multiplier(coefficients: np.ndarray, base_step: float, level: int, kind: str) -> float:
    """Choose a subband multiplier by minimizing distortion + rate cost.

    This is intentionally transparent rather than a claim of global optimality:
    it greedily searches a small candidate set independently for each subband.
    """
    if kind == "approx":
        candidates = (0.65, 0.8, 1.0, 1.25)
    else:
        candidates = (0.75, 1.0, 1.25, 1.5, 1.9)
    data = np.asarray(coefficients, dtype=np.float32)
    variance = float(np.var(data)) + 1e-9
    # Larger base steps imply a stronger rate penalty for keeping detail.
    rate_lambda = max(0.01, min(2.0, base_step / 32.0)) * variance * 0.02
    best_multiplier, best_cost = 1.0, float("inf")
    for multiplier in candidates:
        actual_step = base_step * multiplier * (1.0 + 0.06 * max(0, level - 1))
        quantized = np.rint(data / actual_step)
        reconstructed = quantized * actual_step
        distortion = float(np.mean((data - reconstructed) ** 2))
        bits = _entropy_bits(quantized.astype(np.int32))
        cost = distortion + rate_lambda * bits
        if cost < best_cost:
            best_multiplier, best_cost = multiplier, cost
    return float(best_multiplier)


def rate_distortion_candidates(coefficients: np.ndarray, base_step: float, *, level: int = 0,
                               kind: str = "detail", multipliers: tuple[float, ...] = DEFAULT_MULTIPLIERS) -> list[dict]:
    """Measure deterministic candidate rate/distortion points for one band."""
    if base_step <= 0:
        raise ValueError("base_step must be positive")
    data = np.asarray(coefficients, dtype=np.float32)
    result = []
    for multiplier in multipliers:
        if multiplier <= 0:
            raise ValueError("multipliers must be positive")
        actual_step = float(base_step * multiplier)
        values = np.rint(data / actual_step).astype(np.int32)
        restored = values.astype(np.float32) * actual_step
        result.append({
            "multiplier": float(multiplier),
            "step": actual_step,
            "bits": _entropy_bits(values),
            "distortion": float(np.mean((data - restored) ** 2)) if data.size else 0.0,
            "level": int(level),
            "kind": kind,
        })
    return result


def allocate_subbands(bands: list[np.ndarray], base_step: float, *, target_bpp: float | None = None,
                      target_psnr: float | None = None, pixels: int | None = None,
                      method: str = "greedy", roi_weights: list[float] | None = None) -> dict:
    """Select one quantizer candidate per band.

    A target BPP is interpreted as the coefficient budget divided by image
    pixels; a target PSNR uses a caller-supplied 8-bit MSE approximation.
    ``method="dp"`` is globally optimal for the finite candidate tables and
    the integerized entropy-bit objective described in ``optimality`` below.
    """
    if not bands:
        return {"method": method, "choices": [], "estimated_bits": 0.0, "estimated_distortion": 0.0}
    if base_step <= 0:
        raise ValueError("base_step must be positive")
    if target_bpp is not None and target_bpp <= 0:
        raise ValueError("target_bpp must be positive")
    if target_psnr is not None and target_psnr <= 0:
        raise ValueError("target_psnr must be positive")
    if method not in {"greedy", "lagrangian", "dp"}:
        raise ValueError("method must be greedy, lagrangian or dp")
    weights = roi_weights or [1.0] * len(bands)
    if len(weights) != len(bands):
        raise ValueError("roi_weights must match bands")
    tables = [rate_distortion_candidates(band, base_step, kind="band") for band in bands]
    total_pixels = max(1, int(pixels or sum(np.asarray(band).size for band in bands)))
    budget = float(target_bpp * total_pixels) if target_bpp is not None else None
    weighted_tables = [
        [{**item, "weighted_distortion": item["distortion"] * float(weight)} for item in table]
        for table, weight in zip(tables, weights)
    ]

    def score(choice: list[dict]) -> tuple[float, float, float]:
        bits = sum(item["bits"] for item in choice)
        distortion = sum(item["weighted_distortion"] for item in choice)
        if budget is not None:
            return (abs(bits - budget), distortion, bits)
        if target_psnr is not None:
            target_mse = 255.0 ** 2 / (10 ** (target_psnr / 10.0))
            return (max(0.0, distortion - target_mse), bits, distortion)
        return (distortion + 0.02 * bits, distortion, bits)

    search_states = 0
    candidate_combinations = 1
    for table in weighted_tables:
        candidate_combinations *= len(table)
    if method == "dp":
        # TR: Exact multiple-choice DP invarianti: k bant işlendiğinde states[r],
        #     toplam integerize rate r için tüm seçimlerin en düşük distortion'ını tutar.
        # EN: Exact DP invariant: after k bands, states[r] stores the minimum
        #     distortion among all choices with integerized total rate r.
        # TR: Budama yoktur; bu nedenle son seçim sonlu modelin global optimumudur.
        # EN: There is no pruning; therefore the final choice is globally optimal
        #     for the declared finite model.
        exact_tables = [
            [{**item, "rate_units": int(round(float(item["bits"]) / RATE_UNIT_BITS))} for item in table]
            for table in weighted_tables
        ]
        budget_units = None if budget is None else int(round(float(budget) / RATE_UNIT_BITS))
        states: dict[int, tuple[float, list[dict], float]] = {0: (0.0, [], 0.0)}
        for table in exact_tables:
            next_states: dict[int, tuple[float, list[dict], float]] = {}
            for previous_bits, (previous_distortion, previous_choice, exact_bits) in states.items():
                for item in table:
                    new_exact_bits = exact_bits + float(item["bits"])
                    key = previous_bits + int(item["rate_units"])
                    new_distortion = previous_distortion + float(item["weighted_distortion"])
                    current = next_states.get(key)
                    if current is None or (new_distortion, new_exact_bits) < (current[0], current[2]):
                        next_states[key] = (new_distortion, previous_choice + [item], new_exact_bits)
            states = next_states
        search_states = len(states)
        def exact_score(state: tuple[int, tuple[float, list[dict], float]]) -> tuple[float, float, int]:
            rate_units, (distortion, _, _) = state
            if budget_units is not None:
                return (abs(rate_units - budget_units), distortion, rate_units)
            if target_psnr is not None:
                target_mse = 255.0 ** 2 / (10 ** (target_psnr / 10.0))
                return (max(0.0, distortion - target_mse), rate_units, distortion)
            return (distortion + 0.02 * rate_units, distortion, rate_units)
        selected_units, selected_state = min(states.items(), key=lambda item: exact_score(item))
        combinations = [selected_state[1]]
        selected_exact_bits = selected_state[2]
    elif method == "lagrangian":
        # Search a fixed lambda grid and choose the point nearest the requested
        # budget. The lambda grid is explicit in the result for reproducibility.
        lambdas = np.geomspace(1e-4, 10.0, 32)
        combinations = []
        for rate_lambda in lambdas:
            choice = [min(table, key=lambda item: item["weighted_distortion"] + float(rate_lambda) * item["bits"])
                      for table in weighted_tables]
            combinations.append(choice)
        search_states = len(combinations)
    else:
        combinations = [[min(table, key=lambda item: item["weighted_distortion"] + 0.02 * item["bits"])
                          for table in weighted_tables]]
        search_states = len(combinations)
    selected = min(combinations, key=score)
    selected_bits = float(sum(item["bits"] for item in selected))
    selected_distortion = float(sum(item["weighted_distortion"] for item in selected))
    selected_rate_units = int(round(selected_bits / RATE_UNIT_BITS))
    if method == "dp":
        # Keep the exact DP choice; the legacy floating-point score can differ
        # only in a tie between integerized states and must not override it.
        selected_bits = float(selected_exact_bits)
        selected_rate_units = int(selected_units)
        selected_distortion = float(selected_state[0])
    return {
        "method": method,
        "target_bpp": target_bpp,
        "target_psnr": target_psnr,
        "estimated_bits": selected_bits,
        "estimated_bpp": selected_bits / max(1, total_pixels),
        "target_bpp_error": None if budget is None else selected_bits / max(1, total_pixels) - float(target_bpp),
        "estimated_distortion": selected_distortion,
        "search_states": int(search_states),
        "estimated_rate_units": selected_rate_units,
        "rate_unit_bits": RATE_UNIT_BITS,
        "candidate_combinations": int(candidate_combinations),
        "optimality": {
            "global": method == "dp",
            "solver": "exact-multiple-choice-dp" if method == "dp" else method,
            "scope": "finite per-band candidate tables",
            "objective": "lexicographic rate/distortion objective on integerized entropy bits",
            "rate_unit_bits": RATE_UNIT_BITS,
            "state_pruning": False if method == "dp" else None,
        },
        "per_band": [
            {key: value for key, value in item.items() if key in {"kind", "level", "multiplier", "step", "bits", "rate_units", "distortion", "weighted_distortion"}}
            for item in selected
        ],
        "choices": selected,
    }
