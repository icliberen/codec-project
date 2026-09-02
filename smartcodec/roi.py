"""ROI mask utilities and optional semantic detection adapter.
TR: ROI kaynakları maske, yüz algılama veya YOLO olabilir.
EN: ROI sources may be masks, face detection, or YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class ROIRegion:
    """Serializable semantic ROI description used by CLI, GUI and reports."""

    x: int
    y: int
    width: int
    height: int
    label: str = "manual"
    confidence: float = 1.0
    importance: float = 1.0
    quality_weight: float = 1.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def as_dict(self) -> dict:
        return asdict(self)


def boxes_to_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Create a float mask from (x, y, width, height) boxes."""
    height, width = shape
    mask = Image.new("F", (width, height), 0.0)
    draw = ImageDraw.Draw(mask)
    for x, y, box_width, box_height in boxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, x0 + int(box_width)), min(height, y0 + int(box_height))
        if x1 > x0 and y1 > y0:
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=1.0)
    return np.asarray(mask, dtype=np.float32)


def regions_to_mask(shape: tuple[int, int], regions: list[ROIRegion | dict | tuple[int, int, int, int]]) -> np.ndarray:
    normalized = []
    for region in regions:
        if isinstance(region, ROIRegion):
            normalized.append(region.box)
        elif isinstance(region, dict):
            normalized.append((region["x"], region["y"], region["width"], region["height"]))
        else:
            normalized.append(tuple(region))
    return boxes_to_mask(shape, normalized)


def feather_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Create a soft ROI transition without requiring OpenCV."""
    array = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if radius <= 0:
        return array
    padded = np.pad(array, radius, mode="edge")
    windows = []
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            windows.append(padded[dy : dy + array.shape[0], dx : dx + array.shape[1]])
    return np.mean(windows, axis=0).astype(np.float32)


def regions_metadata(regions: list[ROIRegion | dict]) -> list[dict]:
    output = []
    for region in regions:
        if isinstance(region, ROIRegion):
            output.append(region.as_dict())
        elif isinstance(region, dict):
            output.append(dict(region))
        else:
            x, y, width, height = region
            output.append({"x": int(x), "y": int(y), "width": int(width), "height": int(height), "label": "manual"})
    return output


def load_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L").resize((shape[1], shape[0])), dtype=np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.asarray(mask) * 255, 0, 255).astype(np.uint8)).save(destination)


def _yolo_model_search_paths(model_name: str | Path) -> list[Path]:
    requested_text = str(model_name).strip()
    automatic = requested_text.lower() in {"", "auto", "best"}
    if automatic:
        # Prefer the installed higher-quality instance-segmentation model, then
        # fall back through lighter segmentation and detection weights.
        filenames = ["yolo11m-seg.pt", "yolo11s-seg.pt", "yolo11n-seg.pt", "yolo11n.pt"]
    else:
        requested = Path(requested_text).expanduser()
        if requested.is_absolute() or len(requested.parts) > 1:
            return [requested]
        filenames = [requested.name]
    roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[1],
        Path(sys.executable).resolve().parent,
        Path(sys.executable).resolve().parent / "_internal",
        Path(getattr(sys, "_MEIPASS", Path.cwd())),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for filename in filenames:
            candidate = root / filename
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
    return unique


def resolve_yolo_model_path(model_name: str | Path = "auto") -> Path | None:
    """Find a user-specified or bundled YOLO weight without downloading it."""
    for candidate in _yolo_model_search_paths(model_name):
        if candidate.is_file():
            return candidate.resolve()
    return None


def detect_with_yolo(image_path: str | Path, model_name: str = "auto", confidence: float = 0.25,
                     device: str | None = None, classes: set[str] | None = None) -> list[tuple[int, int, int, int]]:
    return [item["box"] for item in analyze_scene(image_path, model_name, confidence, device, classes)]


def analyze_scene(image_path: str | Path, model_name: str = "auto", confidence: float = 0.25,
                 device: str | None = None, classes: set[str] | None = None) -> list[dict]:
    """Return object boxes, class labels and confidence for scene understanding.

    The adapter is lazy so the core codec stays lightweight when YOLO is not used.
    """
    detections, _ = analyze_scene_with_metadata(image_path, model_name, confidence, device, classes)
    return detections


def _resize_detection_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a YOLO mask to source-image coordinates."""
    array = np.asarray(mask, dtype=np.float32)
    if tuple(array.shape) == tuple(shape):
        return np.clip(array, 0.0, 1.0)
    height, width = shape
    resized = Image.fromarray(np.clip(array, 0.0, 1.0), mode="F").resize(
        (width, height), resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32).clip(0.0, 1.0)


def detections_to_mask(shape: tuple[int, int], detections: list[dict]) -> np.ndarray:
    """Build one pixel ROI from segmentation masks, falling back to boxes."""
    mask = np.zeros(tuple(shape), dtype=np.float32)
    for item in detections:
        local_mask = item.get("mask") if isinstance(item, dict) else None
        if local_mask is not None:
            mask = np.maximum(mask, _resize_detection_mask(local_mask, shape))
        elif isinstance(item, dict) and item.get("box") is not None:
            mask = np.maximum(mask, boxes_to_mask(shape, [tuple(item["box"])]))
    return mask


def serializable_detections(detections: list[dict]) -> list[dict]:
    """Drop binary mask arrays before placing detections in SWC JSON metadata."""
    output = []
    for item in detections:
        clean = {key: value for key, value in item.items() if key != "mask"}
        local_mask = item.get("mask")
        clean["segmentation"] = local_mask is not None
        if local_mask is not None:
            array = np.asarray(local_mask)
            clean["mask_shape"] = list(array.shape)
            clean["mask_fraction"] = float(np.mean(array > 0.5))
        output.append(clean)
    return output


def analyze_scene_with_metadata(image_path: str | Path, model_name: str = "auto", confidence: float = 0.25,
                                device: str | None = None, classes: set[str] | None = None) -> tuple[list[dict], dict]:
    """Run YOLO and return detections together with auditable model metadata."""
    model_path = resolve_yolo_model_path(model_name)
    if model_path is None:
        raise FileNotFoundError(
            f"YOLO ağırlığı bulunamadı: {model_name}. Otomatik indirme kapalı; "
            "yolo11n.pt dosyasını uygulama klasörüne koyun veya --yolo-model ile açık yol verin."
        )
    try:
        from ultralytics import YOLO, __version__ as ultralytics_version
    except ImportError as exc:
        raise RuntimeError(
            "YOLO desteği için requirements-optional.txt içindeki ultralytics paketini kurun."
        ) from exc
    model = YOLO(str(model_path))
    kwargs = {"source": str(image_path), "conf": float(confidence), "verbose": False}
    if device:
        kwargs["device"] = device
    result = model.predict(**kwargs)[0]
    with Image.open(image_path) as source_image:
        source_shape = (source_image.height, source_image.width)
    boxes = []
    coordinates = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.zeros(len(coordinates))
    confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(coordinates))
    masks = getattr(result, "masks", None)
    mask_data = None if masks is None or masks.data is None else masks.data.cpu().numpy()
    for index, (xyxy, class_id, detection_confidence) in enumerate(zip(coordinates, class_ids, confidences)):
        x0, y0, x1, y1 = [int(round(value)) for value in xyxy]
        names = result.names
        label = names.get(int(class_id), str(int(class_id))) if isinstance(names, dict) else str(int(class_id))
        if classes and label not in classes:
            continue
        item = {
            "box": (x0, y0, max(0, x1 - x0), max(0, y1 - y0)),
            "label": label,
            "confidence": float(detection_confidence),
        }
        if mask_data is not None and index < len(mask_data):
            item["mask"] = _resize_detection_mask(mask_data[index], source_shape)
        boxes.append(item)
    resolved_model_path = Path(getattr(model, "ckpt_path", model_path))
    metadata = {
        "analyzer": "ultralytics-yolo",
        "model": str(resolved_model_path),
        "model_sha256": hashlib.sha256(resolved_model_path.read_bytes()).hexdigest() if resolved_model_path.is_file() else None,
        "ultralytics_version": str(ultralytics_version),
        "confidence_threshold": float(confidence),
        "device": device or "auto",
        "object_count": len(boxes),
        "model_task": getattr(model, "task", None),
        "segmentation_enabled": mask_data is not None,
        "segmentation_count": sum("mask" in item for item in boxes),
    }
    return boxes, metadata


def detect_faces(image_path: str | Path) -> list[tuple[int, int, int, int]]:
    """Detect teleconference-style face ROIs using OpenCV's bundled cascade."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Yüz ROI için opencv-python paketini kurun.") from exc
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Görüntü açılamadı: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [(int(x), int(y), int(width), int(height)) for x, y, width, height in detections]
