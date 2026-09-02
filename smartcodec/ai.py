"""Lightweight learned wavelet-parameter estimator.

This module intentionally avoids a heavyweight ML runtime. It extracts image
complexity features and learns class prototypes for candidate codec settings.
It is a reproducible baseline for the document's "wavelet parameter
estimation" research direction, not a hallucination-prone generative decoder.

TR: Bu dosya özellik çıkarımı ve model/fallback kararını ayrı tutar.
EN: This file keeps feature extraction and model/fallback decisions separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


DEFAULT_CANDIDATES = (
    {"wavelet": "haar", "step": 24.0, "quantizer": "uniform"},
    {"wavelet": "haar", "step": 12.0, "quantizer": "scalar"},
    {"wavelet": "db4", "step": 8.0, "quantizer": "scalar"},
    {"wavelet": "db8", "step": 6.0, "quantizer": "scalar"},
)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    mask: np.ndarray | None = None

    def as_dict(self) -> dict:
        item = {"label": self.label, "confidence": float(self.confidence), "box": list(self.box)}
        if self.mask is not None:
            item["mask_shape"] = list(self.mask.shape)
            item["segmentation"] = True
            item["mask_fraction"] = float(np.mean(np.asarray(self.mask) > 0.5))
        else:
            item["segmentation"] = False
        return item


class SemanticAnalyzer(Protocol):
    """Common interface for fallback and optional semantic models."""

    name: str

    def analyze(self, image: np.ndarray) -> tuple[list[Detection], dict]: ...


class FallbackSemanticAnalyzer:
    name = "deterministic-fallback"

    def analyze(self, image: np.ndarray) -> tuple[list[Detection], dict]:
        array = np.asarray(image, dtype=np.float32)
        gray = array if array.ndim == 2 else 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
        dx = np.diff(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
        dy = np.diff(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
        edge_density = float((np.abs(dx).mean() + np.abs(dy).mean()) / 2.0 / 255.0)
        normalized = gray / max(float(np.max(gray)), 1.0)
        flatness = float(1.0 - np.clip(np.std(normalized), 0.0, 1.0))
        features = {
            "texture": float(np.std(normalized)),
            "edge_density": edge_density,
            "flatness": flatness,
            "face_count": 0,
            "object_count": 0,
            "analyzer": self.name,
        }
        return [], features


class YOLOSemanticAnalyzer:
    """Lazy Ultralytics adapter; it never downloads a model implicitly."""

    def __init__(self, model_path: str | Path, confidence: float = 0.25, device: str | None = None,
                 classes: set[str] | None = None) -> None:
        self.model_path = Path(model_path)
        self.confidence = float(confidence)
        self.device = device
        self.classes = classes
        self.name = "ultralytics-yolo"
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found; download/install it explicitly: {self.model_path}")

    def analyze(self, image: np.ndarray) -> tuple[list[Detection], dict]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install requirements-optional.txt for the YOLO adapter") from exc
        model = YOLO(str(self.model_path))
        kwargs = {"source": np.asarray(image), "conf": self.confidence, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        source = np.asarray(image)
        result = model.predict(**kwargs)[0]
        detections: list[Detection] = []
        names = result.names if isinstance(result.names, dict) else {}
        masks = getattr(result, "masks", None)
        mask_data = None if masks is None or masks.data is None else masks.data.cpu().numpy()
        for index, (box, cls, confidence) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy())):
            label = names.get(int(cls), str(int(cls)))
            if self.classes and label not in self.classes:
                continue
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            mask = None
            if mask_data is not None and index < len(mask_data):
                height, width = source.shape[:2]
                mask_image = Image.fromarray(np.asarray(mask_data[index], dtype=np.float32), mode="F")
                mask = np.asarray(
                    mask_image.resize((width, height), resample=Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ).clip(0.0, 1.0)
            detections.append(Detection(label, float(confidence), (x0, y0, max(0, x1 - x0), max(0, y1 - y0)), mask))
        features = {"analyzer": self.name, "object_count": len(detections), "model": str(self.model_path),
                    "model_sha256": hashlib.sha256(self.model_path.read_bytes()).hexdigest(),
                    "confidence_threshold": self.confidence, "model_task": getattr(model, "task", None),
                    "segmentation_enabled": mask_data is not None,
                    "segmentation_count": sum(item.mask is not None for item in detections)}
        return detections, features


def image_features(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        array = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    array /= max(float(np.max(array)), 1.0)
    dx = np.diff(array, axis=1)
    dy = np.diff(array, axis=0)
    gradients = np.concatenate((np.abs(dx).reshape(-1), np.abs(dy).reshape(-1)))
    histogram, _ = np.histogram(array, bins=32, range=(0, 1), density=False)
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))) / 5.0
    return np.array(
        [
            1.0,
            float(np.mean(array)),
            float(np.std(array)),
            float(np.mean(gradients)),
            float(np.percentile(gradients, 90)),
            entropy,
        ],
        dtype=np.float32,
    )


def _proxy_label(features: np.ndarray, candidate_count: int) -> int:
    complexity = float(0.55 * features[3] + 0.3 * features[4] + 0.15 * features[5])
    if candidate_count <= 1:
        return 0
    normalized = int(np.clip(complexity * candidate_count * 2.4, 0, candidate_count - 1))
    return normalized


class WaveletParameterEstimator:
    """Prototype classifier trained from representative images."""

    def __init__(self, candidates: tuple[dict, ...] = DEFAULT_CANDIDATES) -> None:
        self.candidates = tuple(candidates)
        self.prototypes: dict[int, np.ndarray] = {}

    def fit(self, images: list[np.ndarray], labels: list[int] | None = None) -> "WaveletParameterEstimator":
        if labels is not None and len(labels) != len(images):
            raise ValueError("labels must have the same length as images")
        buckets: dict[int, list[np.ndarray]] = {}
        for index, image in enumerate(images):
            features = image_features(image)
            label = int(labels[index]) if labels is not None else _proxy_label(features, len(self.candidates))
            if label < 0 or label >= len(self.candidates):
                raise ValueError("estimator label is outside the candidate range")
            buckets.setdefault(label, []).append(features)
        self.prototypes = {label: np.mean(values, axis=0) for label, values in buckets.items()}
        return self

    def predict(self, image: np.ndarray) -> tuple[dict, float]:
        features = image_features(image)
        if not self.prototypes:
            label = _proxy_label(features, len(self.candidates))
            return dict(self.candidates[label]), 0.5
        distances = {label: float(np.linalg.norm(features - prototype)) for label, prototype in self.prototypes.items()}
        label = min(distances, key=distances.get)
        ordered = sorted(distances.values())
        # A single learned prototype cannot establish class separation.  Do
        # not report a misleading 1.0 confidence for the small-data baseline.
        confidence = 0.5 if len(ordered) < 2 else float(np.clip(1.0 - ordered[0] / max(ordered[1], 1e-9), 0, 1))
        return dict(self.candidates[label]), confidence

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "candidates": list(self.candidates),
                   "prototypes": {str(key): value.tolist() for key, value in self.prototypes.items()},
                   "model_type": "prototype-baseline"}
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "WaveletParameterEstimator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        estimator = cls(tuple(payload.get("candidates", DEFAULT_CANDIDATES)))
        estimator.prototypes = {int(key): np.asarray(value, dtype=np.float32) for key, value in payload.get("prototypes", {}).items()}
        return estimator


def estimate_parameters(image: np.ndarray) -> tuple[dict, float]:
    """Return a deterministic adaptive recommendation and confidence."""
    return WaveletParameterEstimator().predict(image)


class RestorationStrategy(Protocol):
    name: str

    def restore(self, image: np.ndarray) -> np.ndarray: ...


class ResidualDetailRestoration:
    name = "deterministic-residual-baseline"

    def __init__(self, strength: float = 0.18) -> None:
        self.strength = strength

    def restore(self, image: np.ndarray) -> np.ndarray:
        return detail_reconstruction(image, self.strength)


class TorchRestorationAdapter:
    """Optional PyTorch/JIT restoration adapter with explicit model loading."""

    name = "pytorch-model"

    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Restoration model not found: {self.model_path}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch separately to use the restoration adapter") from exc
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        # PyTorch currently emits a Python 3.14 deprecation warning for the
        # TorchScript loader even though the shipped models are intentionally
        # TorchScript-compatible. Keep this narrow and visible to other
        # deprecations while preserving backward compatibility with .pt files.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()
        self.sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()

    def restore(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        work = array if array.ndim == 3 else array[..., None]
        # The shipped restorers are RGB models.  Replicate grayscale input so
        # that the same real model also works for monochrome images.
        model_work = np.repeat(work, 3, axis=2) if work.shape[2] == 1 else work
        tensor = self._torch.from_numpy(model_work.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            output = self.model(tensor)
        result = output.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 255.0
        if array.ndim == 2:
            result = result[..., 0]
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)


_RESTORER_MODEL_TYPES = {"pytorch-smartcodec-artifact-restorer", "pytorch-residual-denoiser"}


def _restoration_model_roots(workspace: str | Path | None = None) -> list[Path]:
    """Return model directories for source-tree and portable-app layouts."""
    roots: list[Path] = []
    if workspace is not None:
        supplied = Path(workspace)
        roots.extend((supplied / "outputs" / "models", supplied / "_internal" / "outputs" / "models",
                      supplied / "models", supplied))
    roots.extend((Path.cwd() / "outputs" / "models", Path(__file__).resolve().parents[1] / "outputs" / "models"))
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _read_restoration_report(model_path: Path) -> dict:
    report_path = model_path.with_suffix(model_path.suffix + ".report.json")
    if not report_path.is_file():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _restoration_model_score(model_path: Path, report: dict) -> tuple:
    """Rank measured codec-aware models ahead of unmeasured/weak candidates."""
    model_type = str(report.get("model_type", ""))
    history = report.get("history") if isinstance(report.get("history"), list) else []
    validation_values = [
        float(item["validation_mse"])
        for item in history
        if isinstance(item, dict) and isinstance(item.get("validation_mse"), (int, float))
        and math.isfinite(float(item["validation_mse"]))
    ]
    final_mse = min(validation_values) if validation_values else float("inf")
    baseline = report.get("baseline_validation_mse")
    improvement_ratio = 0.0
    if isinstance(baseline, (int, float)) and float(baseline) > 0 and math.isfinite(float(baseline)) and math.isfinite(final_mse):
        improvement_ratio = max(0.0, (float(baseline) - final_mse) / float(baseline))
    artifact_bonus = 1 if model_type == "pytorch-smartcodec-artifact-restorer" else 0
    report_bonus = 1 if report else 0
    train_count = int(report.get("train_count", 0) or 0)
    validation_count = int(report.get("validation_count", 0) or 0)
    preferred_name = 1 if model_path.name == "div2k_codec_restorer_full.pt" else 0
    return (artifact_bonus, preferred_name, improvement_ratio, report_bonus, train_count, validation_count, -final_mse)


def select_best_restoration_model(model_path: str | Path | None = None,
                                  workspace: str | Path | None = None) -> tuple[Path | None, dict]:
    """Select an explicit model or the best measured model available locally.

    ``None``, ``"auto"`` and ``"best"`` scan adjacent model reports.  The
    selection is deterministic and prefers codec-aware models with measured
    validation improvement; it never downloads weights implicitly.
    """
    requested = str(model_path).strip() if model_path is not None else ""
    if requested and requested.lower() not in {"auto", "best"}:
        selected = Path(requested).expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"Restoration model not found: {selected}")
        report = _read_restoration_report(selected)
        return selected.resolve(), {"selection": "explicit", "model": str(selected.resolve()),
                                    "model_type": report.get("model_type", "unknown"),
                                    "report": str(selected.with_suffix(selected.suffix + ".report.json")) if report else None}

    candidates: list[tuple[tuple, Path, dict]] = []
    seen: set[str] = set()
    for root in _restoration_model_roots(workspace):
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob("*.pt")):
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            report = _read_restoration_report(candidate)
            if report and report.get("model_type") not in _RESTORER_MODEL_TYPES:
                continue
            candidates.append((_restoration_model_score(candidate, report), candidate, report))
    if not candidates:
        return None, {"selection": "auto", "model": None, "reason": "no-local-restoration-model"}
    _, selected, report = max(candidates, key=lambda item: item[0])
    return selected.resolve(), {
        "selection": "auto-best-available", "model": str(selected.resolve()),
        "model_type": report.get("model_type", "unknown"),
        "dataset": report.get("dataset"), "train_count": report.get("train_count"),
        "validation_count": report.get("validation_count"),
        "report": str(selected.with_suffix(selected.suffix + ".report.json")) if report else None,
    }


def restore_decoded(image: np.ndarray, *, mode: str = "lossy", strategy: RestorationStrategy | None = None) -> np.ndarray:
    """Apply restoration only to lossy output and make the policy explicit."""
    if mode == "lossless":
        raise ValueError("Restoration is forbidden for lossless output")
    return (strategy or ResidualDetailRestoration()).restore(image)


def detail_reconstruction(image: np.ndarray, strength: float = 0.18) -> np.ndarray:
    """A conservative, optional residual-detail postprocessor.

    It is intentionally disabled for lossless medical/fingerprint mode. The
    operation is a transparent baseline for comparing decoder-side detail
    recovery; it does not invent an unverified diagnostic result.
    """
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        work = array[..., None]
    else:
        work = array
    padded = np.pad(work, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    blur = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] + 4 * padded[1:-1, 1:-1]
    ) / 8.0
    residual = work - blur
    reconstructed = np.clip(work + float(np.clip(strength, 0, 0.5)) * residual, 0, 255)
    result = reconstructed[..., 0] if array.ndim == 2 else reconstructed
    return np.rint(result).astype(np.uint8)
