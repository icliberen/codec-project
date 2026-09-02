"""Shared GUI dependencies / Ortak GUI bağımlılıkları."""

from __future__ import annotations

import os
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

DND_FILES = None

from .ai import TorchRestorationAdapter, detail_reconstruction, estimate_parameters
from .benchmark import generate_dataset, generate_samples, run_benchmark, run_normalized_benchmark, run_roi_comparison_benchmark
from .codec import decode_array, decode_file, encode_file
from .diagnostics import format_dependency_status
from .image_io import load_image, save_image
from .metrics import mse, psnr, region_metrics, ssim
from .roi import analyze_scene, boxes_to_mask, detect_faces, detections_to_mask, load_mask
from .standard import decode_standard, encode_standard
from .transport import send_udp_file, simulate_file
from .video import compress_video, decompress_video, package_video, simulate_video_transport

# TR: Bu modül yalnızca ortak isimleri toplar; iş mantığı mixin dosyalarında yaşar.
# EN: This module only centralizes shared names; business logic lives in mixin files.
