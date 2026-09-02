"""Optional dependency/model diagnostics for CLI and GUI surfaces.
TR: Eksik opsiyonel bağımlılıklar ana codec'i bozmayacak şekilde raporlanır.
EN: Missing optional dependencies are reported without breaking the core codec.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .roi import resolve_yolo_model_path


def dependency_status(model_path: str | Path | None = None) -> dict[str, dict]:
    status = {}
    for name, package in (("Pillow", "PIL"), ("PyWavelets", "pywt"), ("OpenCV", "cv2"),
                          ("YOLO", "ultralytics"), ("PyTorch", "torch")):
        available = importlib.util.find_spec(package) is not None
        status[name] = {"available": available, "optional": name in {"OpenCV", "YOLO", "PyTorch"}}
    if model_path is not None:
        path = resolve_yolo_model_path(model_path)
        status["model"] = {"available": path is not None, "path": str(path or Path(model_path))}
    return status


def format_dependency_status(model_path: str | Path | None = None) -> str:
    values = dependency_status(model_path)
    return ", ".join(f"{name}: {'ok' if item['available'] else 'missing'}" for name, item in values.items())
