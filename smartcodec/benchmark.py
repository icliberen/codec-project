"""Reproducible sample generation and rate-distortion benchmarks.

TR: Deneyler manifest üzerinden tekrar üretilebilir sonuç, grafik ve durum raporu üretir.
EN: Experiments produce reproducible results, plots, and status reports from manifests.
"""

from __future__ import annotations

import csv
import io
import math
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .codec import decode_array, encode_file
from .dataset import manifest_for_dataset, scan_dataset, write_manifest
from .image_io import load_image, save_image
from .metrics import bits_per_pixel, compression_ratio, mse, psnr, region_metrics, ssim
from .roi import load_mask


def generate_samples(destination: str | Path, size: tuple[int, int] = (256, 256)) -> list[Path]:
    """Create deterministic synthetic and hybrid images for experiments."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    gradient = np.stack(
        [xx / max(1, width - 1) * 255, yy / max(1, height - 1) * 255, (xx + yy) / (width + height - 2) * 255],
        axis=-1,
    ).astype(np.uint8)
    checker = (((xx // 16 + yy // 16) % 2) * 255).astype(np.uint8)
    circles = np.zeros((height, width, 3), dtype=np.uint8)
    circles[:] = (30, 45, 90)
    image = Image.fromarray(circles)
    draw = ImageDraw.Draw(image)
    for radius, color in ((100, (240, 80, 50)), (70, (60, 220, 120)), (35, (245, 220, 70))):
        draw.ellipse((width // 2 - radius, height // 2 - radius, width // 2 + radius, height // 2 + radius), fill=color)
    mixed = gradient.copy()
    mixed[height // 4 : height // 2, width // 4 : width // 2] = 255 - mixed[height // 4 : height // 2, width // 4 : width // 2]
    images = {
        "synthetic_gradient.png": gradient,
        "synthetic_checkerboard.png": checker,
        "synthetic_circles.png": np.asarray(image),
        "hybrid_gradient_graphics.png": mixed,
    }
    paths = []
    for name, array in images.items():
        path = output / name
        save_image(array, path)
        paths.append(path)
    return paths


def generate_dataset(destination: str | Path, size: tuple[int, int] = (256, 256)) -> list[Path]:
    """Generate a categorized, deterministic benchmark corpus.

    Natural and biomedical samples are deliberately labelled synthetic surrogates;
    real scientific conclusions still require domain datasets supplied by the user.
    """
    root = Path(destination)
    width, height = size
    generated: list[Path] = []
    generated.extend(generate_samples(root / "synthetic", size))

    yy, xx = np.mgrid[0:height, 0:width]
    rng = np.random.default_rng(2026)
    smooth_noise = rng.normal(0, 1, (height, width)).astype(np.float32)
    for _ in range(5):
        smooth_noise = (smooth_noise + np.roll(smooth_noise, 1, 0) + np.roll(smooth_noise, -1, 0) + np.roll(smooth_noise, 1, 1) + np.roll(smooth_noise, -1, 1)) / 5
    natural_like = np.stack(
        [
            np.clip(75 + 70 * yy / max(1, height - 1) + 28 * smooth_noise, 0, 255),
            np.clip(100 + 80 * yy / max(1, height - 1) + 22 * smooth_noise, 0, 255),
            np.clip(145 - 75 * yy / max(1, height - 1) + 18 * smooth_noise, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    natural_path = root / "natural" / "natural_like_landscape_surrogate.png"
    save_image(natural_like, natural_path)
    generated.append(natural_path)

    hybrid = natural_like.copy()
    hybrid[height // 4 : height // 2, width // 4 : width // 2] = 255 - hybrid[height // 4 : height // 2, width // 4 : width // 2]
    hybrid_path = root / "hybrid" / "natural_synthetic_hybrid.png"
    save_image(hybrid, hybrid_path)
    generated.append(hybrid_path)

    fingerprint = ((np.sin(xx * 0.18 + 12 * np.sin(yy * 0.025)) + 1) * 127.5).astype(np.uint8)
    fingerprint_path = root / "fingerprint" / "fingerprint_surrogate.png"
    save_image(fingerprint, fingerprint_path)
    generated.append(fingerprint_path)

    biomedical = np.full((height, width), 18, dtype=np.float32)
    for center_x, center_y, radius_x, radius_y, intensity in (
        (width * 0.35, height * 0.45, width * 0.18, height * 0.24, 150),
        (width * 0.62, height * 0.58, width * 0.12, height * 0.16, 220),
    ):
        field = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
        biomedical += np.where(field <= 1, intensity * (1 - np.clip(field, 0, 1)), 0)
    biomedical = np.clip(biomedical + rng.normal(0, 4, biomedical.shape), 0, 255).astype(np.uint8)
    biomedical_path = root / "biomedical" / "biomedical_phantom_surrogate.png"
    save_image(biomedical, biomedical_path)
    generated.append(biomedical_path)

    letters = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(letters)
    draw.text((width // 8, height // 3), "DSP 2026", fill="black")
    draw.rectangle((width // 2, height // 2, width * 3 // 4, height * 3 // 4), outline="red", width=5)
    letters_path = root / "letters_logos" / "letters_and_logo_surrogate.png"
    save_image(np.asarray(letters), letters_path)
    generated.append(letters_path)
    # Keep generation provenance next to the images.  The categories that are
    # visually natural/biomedical are still explicitly marked as surrogates.
    write_manifest(
        scan_dataset(root, source="surrogate"), root / "dataset_manifest.json",
        root=str(root.resolve()), generator="smartcodec.generate_dataset", seed=2026,
        image_size=list(size), surrogate=True,
    )
    return generated


def _jpeg_bytes(image: np.ndarray, quality: int) -> tuple[bytes, np.ndarray]:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        result = np.asarray(decoded.convert("L" if image.ndim == 2 else "RGB"), dtype=np.uint8).copy()
    return buffer.getvalue(), result


def _jpeg2000_bytes(image: np.ndarray, rate: float) -> tuple[bytes, np.ndarray]:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG2000", quality_mode="rates", quality_layers=[rate])
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        result = np.asarray(decoded.convert("L" if image.ndim == 2 else "RGB"), dtype=np.uint8).copy()
    return buffer.getvalue(), result


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record_result(original: np.ndarray, reconstructed: np.ndarray, info: dict, source: Path,
                   category: str, codec: str, step: float, encode_seconds: float,
                   decode_seconds: float, *, status: str = "ok", error: str | None = None,
                   roi_mask: np.ndarray | None = None, peak_memory_bytes: int | None = None,
                   extra: dict | None = None) -> dict:
    row = {
        "category": category,
        "image": source.name,
        "path": str(source),
        "source": "surrogate" if "surrogate" in source.name.lower() else "external",
        "codec": codec,
        "step": step,
        "original_shape": list(original.shape),
        "channels": 1 if original.ndim == 2 else original.shape[2],
        "dtype": str(original.dtype),
        "original_bytes": int(original.nbytes),
        "file_bytes": int(info.get("file_size", info.get("encoded_bytes", 0))),
        "compression_ratio": compression_ratio(original.nbytes, int(info.get("file_size", info.get("encoded_bytes", 0)))),
        "bits_per_pixel": bits_per_pixel(int(info.get("file_size", info.get("encoded_bytes", 0))), original.shape),
        "mse": mse(original, reconstructed),
        "psnr": psnr(original, reconstructed),
        "ssim": ssim(original, reconstructed),
        "roi_mse": None,
        "roi_psnr": None,
        "roi_ssim": None,
        "background_mse": None,
        "background_psnr": None,
        "background_ssim": None,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "status": status,
        "error": error,
    }
    if extra:
        row.update(extra)
    if roi_mask is not None and status not in {"failed", "unsupported"}:
        row.update(region_metrics(original, reconstructed, roi_mask))
    return row


def _external_codec_bytes(image: np.ndarray, codec: str, setting: float) -> tuple[bytes, np.ndarray]:
    if codec == "jpeg":
        return _jpeg_bytes(image, int(setting))
    if codec == "jpeg2000":
        return _jpeg2000_bytes(image, setting)
    raise ValueError(codec)


def _write_statistics(rows: list[dict], destination: Path) -> None:
    numeric = ("bits_per_pixel", "compression_ratio", "mse", "psnr", "ssim", "encode_seconds", "decode_seconds", "peak_memory_bytes")
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((str(row["category"]), str(row["codec"])), []).append(row)
    summary = []
    for (category, codec), values in sorted(groups.items()):
        item = {"category": category, "codec": codec, "count": len(values)}
        for field in numeric:
            numbers = [float(row[field]) for row in values
                       if _safe_float(row.get(field)) is not None and math.isfinite(float(row[field]))]
            item[f"{field}_mean"] = statistics.fmean(numbers) if numbers else None
            item[f"{field}_median"] = statistics.median(numbers) if numbers else None
            item[f"{field}_std"] = statistics.pstdev(numbers) if len(numbers) > 1 else 0.0 if numbers else None
        summary.append(item)
    (destination / "statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary:
        with (destination / "statistics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    (destination / "status_summary.json").write_text(
        json.dumps({"total": len(rows), "by_status": status_counts}, indent=2), encoding="utf-8"
    )


def _finalize_benchmark(rows: list[dict], destination: Path, manifest: dict,
                        image_records: list[dict], experiment_config: dict) -> list[dict]:
    source_labels = {str(Path(record["path"]).resolve()): record.get("source", "external") for record in image_records}
    for row in rows:
        row["source"] = source_labels.get(str(Path(row["path"]).resolve()), row.get("source", "external"))
    if rows:
        with (destination / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (destination / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        _write_statistics(rows, destination)
        (destination / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (destination / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2), encoding="utf-8")
        _plot_results(rows, destination / "rate_distortion.png")
        for category in sorted({str(row.get("category", "uncategorized")) for row in rows}):
            safe_category = "".join(character if character.isalnum() or character in "-_" else "_" for character in category)
            _plot_results([row for row in rows if str(row.get("category")) == category],
                          destination / f"rate_distortion_{safe_category}.png", title=f"Rate-distortion: {category}")
    return rows


def run_benchmark(input_dir: str | Path, output_dir: str | Path, *, step_values: list[float] | None = None,
                  wavelets: list[str] | None = None, level: int = 2, dataset_root: str | Path | None = None,
                  seed: int = 2026, codecs: list[str] | None = None, include_jpeg2000: bool = True,
                  roi_mask_path: str | Path | None = None, ai_enabled: bool = False,
                  restoration: bool = False, quantizer: str = "uniform", allocation_method: str = "greedy",
                  split: str | None = None, limit: int | None = None) -> list[dict]:
    if quantizer not in {"uniform", "scalar"}:
        raise ValueError("benchmark quantizer must be uniform or scalar")
    if allocation_method not in {"greedy", "lagrangian", "dp"}:
        raise ValueError("benchmark allocation_method must be greedy, lagrangian or dp")
    source_dir = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    steps = step_values or [4.0, 8.0, 16.0, 32.0]
    selected_wavelets = wavelets or ["haar", "db4"]
    rows: list[dict] = []
    root = Path(dataset_root) if dataset_root is not None else source_dir
    manifest = manifest_for_dataset(root, source="surrogate" if dataset_root is None else "external")
    rows: list[dict] = []
    image_records = [record for record in manifest["records"] if record.get("readable") and
                     (split is None or record.get("split") == split)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("benchmark limit must be positive")
        image_records = image_records[:int(limit)]
    selected_codecs = set(codecs or ["dwt", "dct", "prqmf4", "jpeg", "jpeg2000"])
    for record in image_records:
        source = Path(record["path"])
        original = load_image(source)
        category = record.get("category", "uncategorized")
        active_roi = load_mask(roi_mask_path, original.shape[:2]) if roi_mask_path else None
        for wavelet in selected_wavelets:
            if "dwt" not in selected_codecs:
                break
            for step in steps:
                encoded_path = destination / f"{source.stem}_dwt_{wavelet}_{step:g}.swc"
                started = time.perf_counter()
                tracemalloc.start()
                try:
                    info = encode_file(source, encoded_path, mode="lossy", wavelet=wavelet, level=level, step=step,
                                       quantizer=quantizer, allocation_method=allocation_method,
                                       roi_mask=active_roi, ai_enabled=ai_enabled, restoration=restoration)
                    encode_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    reconstructed = decode_array(encoded_path)
                    decode_seconds = time.perf_counter() - started
                    _, peak_memory = tracemalloc.get_traced_memory()
                    rows.append(_record_result(original, reconstructed, info, source, category, f"dwt-{wavelet}", step,
                                                encode_seconds, decode_seconds, roi_mask=active_roi,
                                                peak_memory_bytes=int(peak_memory),
                                                extra={"level": level, "quantizer": quantizer, "allocation_method": allocation_method,
                                                       "ai_enabled": ai_enabled, "restoration": restoration}))
                except Exception as exc:
                    _, peak_memory = tracemalloc.get_traced_memory()
                    rows.append(_record_result(original, original, {"file_size": 0}, source, category, f"dwt-{wavelet}", step,
                                                time.perf_counter() - started, 0.0, status="failed",
                                                error=f"{type(exc).__name__}: {exc}", peak_memory_bytes=int(peak_memory),
                                                extra={"quantizer": quantizer, "allocation_method": allocation_method}))
                finally:
                    tracemalloc.stop()
        for codec in ("dct", "prqmf4"):
            if codec not in selected_codecs:
                continue
            for step in steps:
                encoded_path = destination / f"{source.stem}_{codec}_{step:g}.swc"
                started = time.perf_counter()
                tracemalloc.start()
                try:
                    info = encode_file(source, encoded_path, mode="lossy", codec=codec, step=step,
                                       quantizer=quantizer, allocation_method=allocation_method,
                                       roi_mask=active_roi, ai_enabled=ai_enabled, restoration=restoration)
                    encode_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    reconstructed = decode_array(encoded_path)
                    decode_seconds = time.perf_counter() - started
                    _, peak_memory = tracemalloc.get_traced_memory()
                    rows.append(_record_result(original, reconstructed, info, source, category, codec, step,
                                                encode_seconds, decode_seconds, roi_mask=active_roi,
                                                peak_memory_bytes=int(peak_memory),
                                                extra={"quantizer": quantizer, "allocation_method": allocation_method,
                                                       "ai_enabled": ai_enabled, "restoration": restoration}))
                except Exception as exc:
                    _, peak_memory = tracemalloc.get_traced_memory()
                    rows.append(_record_result(original, original, {"file_size": 0}, source, category, codec, step,
                                                time.perf_counter() - started, 0.0, status="failed",
                                                error=f"{type(exc).__name__}: {exc}", peak_memory_bytes=int(peak_memory),
                                                extra={"quantizer": quantizer, "allocation_method": allocation_method}))
                finally:
                    tracemalloc.stop()
        if "jpeg" in selected_codecs:
            for quality in (30, 60, 90):
                started = time.perf_counter()
                tracemalloc.start()
                try:
                    encoded, reconstructed = _external_codec_bytes(original, "jpeg", quality)
                    _, peak_memory = tracemalloc.get_traced_memory()
                    rows.append(_record_result(original, reconstructed, {"encoded_bytes": len(encoded)}, source, category, "jpeg", quality,
                                                time.perf_counter() - started, 0.0, peak_memory_bytes=int(peak_memory),
                                                extra={"encoder": "Pillow", "quantizer": quantizer,
                                                       "allocation_method": allocation_method}))
                except Exception as exc:
                    _, peak_memory = tracemalloc.get_traced_memory()
                    rows.append(_record_result(original, original, {"encoded_bytes": 0}, source, category, "jpeg", quality,
                                                time.perf_counter() - started, 0.0, status="failed",
                                                error=f"{type(exc).__name__}: {exc}", peak_memory_bytes=int(peak_memory),
                                                extra={"encoder": "Pillow", "quantizer": quantizer,
                                                       "allocation_method": allocation_method}))
                finally:
                    tracemalloc.stop()
        if include_jpeg2000 and "jpeg2000" in selected_codecs:
            for rate in (1, 4, 16):
                started = time.perf_counter()
                tracemalloc.start()
                try:
                    encoded, reconstructed = _external_codec_bytes(original, "jpeg2000", rate)
                    _, peak_memory = tracemalloc.get_traced_memory()
                    rows.append(_record_result(original, reconstructed, {"encoded_bytes": len(encoded)}, source, category, "jpeg2000", rate,
                                                time.perf_counter() - started, 0.0, peak_memory_bytes=int(peak_memory),
                                                extra={"encoder": "Pillow/OpenJPEG", "quantizer": quantizer,
                                                       "allocation_method": allocation_method}))
                except Exception as exc:
                    _, peak_memory = tracemalloc.get_traced_memory()
                    rows.append(_record_result(original, original, {"encoded_bytes": 0}, source, category, "jpeg2000", rate,
                                                time.perf_counter() - started, 0.0, status="unsupported",
                                                error=f"{type(exc).__name__}: {exc}", peak_memory_bytes=int(peak_memory),
                                                extra={"encoder": "Pillow/OpenJPEG", "quantizer": quantizer,
                                                       "allocation_method": allocation_method}))
                finally:
                    tracemalloc.stop()
    return _finalize_benchmark(rows, destination, manifest, image_records, {
        "mode": "grid",
        "seed": seed, "input_dir": str(source_dir.resolve()), "dataset_root": str(root.resolve()),
        "steps": steps, "wavelets": selected_wavelets, "level": level,
        "codecs": sorted(selected_codecs), "surrogate": dataset_root is None,
        "roi_mask": str(roi_mask_path) if roi_mask_path else None,
        "ai_enabled": ai_enabled, "restoration": restoration, "quantizer": quantizer,
        "allocation_method": allocation_method,
        "split": split, "limit": limit,
        "measurement": {"peak_memory": "tracemalloc peak over encode/decode operation"},
    })


def run_normalized_benchmark(input_dir: str | Path, output_dir: str | Path, *, target_psnr: float | None = None,
                             target_bpp: float | None = None,
                             step_values: list[float] | None = None, wavelets: list[str] | None = None, level: int = 2,
                             dataset_root: str | Path | None = None, seed: int = 2026,
                             codecs: list[str] | None = None, include_jpeg2000: bool = True,
                             roi_mask_path: str | Path | None = None, split: str | None = None,
                             quantizer: str = "uniform", allocation_method: str = "greedy",
                             limit: int | None = None, adaptive_step_search: bool = True) -> list[dict]:
    """Compare codecs at a shared target PSNR or BPP.

    Target BPP is approximate because each encoded file includes a fixed header and
    metadata payload. Results therefore expose both the achieved BPP and its error.

    TR: Hedef BPP'de SWC step adayları gerektiğinde geometrik olarak genişletilir.
    EN: For target BPP, SWC step candidates expand geometrically when needed.
    """
    if quantizer not in {"uniform", "scalar"}:
        raise ValueError("benchmark quantizer must be uniform or scalar")
    if allocation_method not in {"greedy", "lagrangian", "dp"}:
        raise ValueError("benchmark allocation_method must be greedy, lagrangian or dp")
    if target_psnr is not None and target_bpp is not None:
        raise ValueError("target_psnr and target_bpp are mutually exclusive")
    if target_psnr is None and target_bpp is None:
        target_psnr = 35.0
    if target_psnr is not None and target_psnr <= 0:
        raise ValueError("target_psnr must be positive")
    if target_bpp is not None and target_bpp <= 0:
        raise ValueError("target_bpp must be positive")
    normalization = "target-psnr" if target_psnr is not None else "target-bpp"
    bpp_tolerance = max(0.1, float(target_bpp) * 0.10) if target_bpp is not None else None

    def target_fields(*, achieved_psnr: float | None = None, achieved_bpp: float | None = None) -> dict:
        fields = {"normalization": normalization}
        if target_psnr is not None:
            fields["target_psnr"] = target_psnr
            if achieved_psnr is not None:
                fields["achieved_psnr"] = achieved_psnr
        else:
            fields["target_bpp"] = target_bpp
            if achieved_bpp is not None:
                fields["achieved_bpp"] = achieved_bpp
                fields["bpp_error"] = achieved_bpp - float(target_bpp)
        return fields

    def select_measurement(measurements: list[dict]) -> dict:
        if target_psnr is not None:
            acceptable = [item for item in measurements if item["psnr"] >= target_psnr]
            if acceptable:
                return min(acceptable, key=lambda item: (item["size"], -item["psnr"]))
            return max(measurements, key=lambda item: (item["psnr"], -item["size"]))
        return min(measurements, key=lambda item: (abs(item["bpp"] - float(target_bpp)), -item["psnr"]))

    def measurement_status(item: dict) -> str:
        if target_psnr is not None:
            return "ok" if item["psnr"] >= target_psnr else "below-target"
        return "ok" if abs(item["bpp"] - float(target_bpp)) <= float(bpp_tolerance) else "off-target-bpp"

    source_dir = Path(input_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    root = Path(dataset_root) if dataset_root is not None else source_dir
    manifest = manifest_for_dataset(root, source="surrogate" if dataset_root is None else "external")
    image_records = [record for record in manifest["records"] if record.get("readable") and
                     (split is None or record.get("split") == split)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("benchmark limit must be positive")
        image_records = image_records[:int(limit)]
    selected_codecs = set(codecs or ["dwt", "dct", "prqmf4", "jpeg", "jpeg2000"])
    selected_wavelets = wavelets or ["haar", "db4"]
    candidate_steps = step_values or [4.0, 8.0, 16.0, 32.0]
    adaptive_enabled = bool(adaptive_step_search and target_bpp is not None)
    adaptive_max_iterations = 8
    adaptive_min_step = 0.125
    adaptive_max_step = 4096.0
    rows: list[dict] = []
    for record in image_records:
        source = Path(record["path"])
        original = load_image(source)
        category = record.get("category", "uncategorized")
        active_roi = load_mask(roi_mask_path, original.shape[:2]) if roi_mask_path else None

        def run_swc(codec: str, label: str, wavelet: str | None = None) -> None:
            measurements: list[dict] = []
            measured_steps: set[float] = set()
            adaptive_steps: list[float] = []

            def measure_step(candidate_step: float) -> None:
                candidate_step = float(candidate_step)
                step_key = round(candidate_step, 12)
                if step_key in measured_steps:
                    return
                measured_steps.add(step_key)
                encoded_path = destination / f"{source.stem}_{label.replace('-', '_')}_{candidate_step:g}.swc"
                started = time.perf_counter()
                tracemalloc.start()
                try:
                    info = encode_file(
                        source, encoded_path, mode="lossy", codec=codec, wavelet=wavelet or "haar", level=level,
                        step=candidate_step, quantizer=quantizer, allocation_method=allocation_method,
                        roi_mask=active_roi,
                    )
                    encode_seconds = time.perf_counter() - started
                    decode_started = time.perf_counter()
                    reconstructed = decode_array(encoded_path)
                    decode_seconds = time.perf_counter() - decode_started
                    _, peak_memory = tracemalloc.get_traced_memory()
                    measurements.append({
                        "step": candidate_step, "info": info, "reconstructed": reconstructed,
                        "psnr": psnr(original, reconstructed), "encode_seconds": encode_seconds,
                        "decode_seconds": decode_seconds, "peak": int(peak_memory),
                        "size": int(info.get("file_size", 0)),
                        "bpp": bits_per_pixel(int(info.get("file_size", 0)), original.shape),
                    })
                except Exception:
                    pass
                finally:
                    tracemalloc.stop()

            for candidate_step in candidate_steps:
                measure_step(candidate_step)

            # TR: Sabit grid başlangıç noktasıdır; HR görüntülerde aday hedefi
            #     bracket etmiyorsa step geometrik olarak büyütülür/küçültülür.
            # EN: The fixed grid is a starting point; for HR images, expand or
            #     contract the step geometrically until the target is bracketed.
            if adaptive_enabled:
                for _ in range(adaptive_max_iterations):
                    if not measurements:
                        break
                    measured_bpps = [item["bpp"] for item in measurements]
                    target = float(target_bpp)
                    if min(measured_bpps) <= target <= max(measured_bpps):
                        break
                    if min(measured_bpps) > target:
                        next_step = max(item["step"] for item in measurements) * 2.0
                    else:
                        next_step = min(item["step"] for item in measurements) / 2.0
                    if next_step <= adaptive_min_step:
                        next_step = adaptive_min_step
                    if next_step >= adaptive_max_step:
                        next_step = adaptive_max_step
                    if round(next_step, 12) in measured_steps:
                        break
                    adaptive_steps.append(float(next_step))
                    measure_step(next_step)

            if not measurements:
                rows.append(_record_result(
                    original, original, {"file_size": 0}, source, category, label, float(candidate_steps[0]),
                    0.0, 0.0, status="failed", error="No SWC candidate setting succeeded",
                    extra={**target_fields(), "wavelet": wavelet, "quantizer": quantizer,
                           "allocation_method": allocation_method,
                           "adaptive_step_search": adaptive_enabled, "adaptive_steps": adaptive_steps},
                ))
                return
            selected = select_measurement(measurements)
            actual_psnr = selected["psnr"]
            rows.append(_record_result(
                original, selected["reconstructed"], selected["info"], source, category, label,
                float(selected["step"]), selected["encode_seconds"], selected["decode_seconds"],
                status=measurement_status(selected), roi_mask=active_roi,
                peak_memory_bytes=selected["peak"], extra={
                    **target_fields(achieved_psnr=actual_psnr, achieved_bpp=selected["bpp"]),
                    "wavelet": wavelet, "codec": codec, "quantizer": quantizer,
                    "allocation_method": allocation_method,
                    "adaptive_step_search": adaptive_enabled, "adaptive_steps": adaptive_steps,
                    "candidate_count": len(measurements),
                },
            ))

        if "dwt" in selected_codecs:
            for wavelet in selected_wavelets:
                run_swc("dwt", f"dwt-{wavelet}", wavelet)
        for codec in ("dct", "prqmf4"):
            if codec in selected_codecs:
                run_swc(codec, codec)

        def run_external(codec: str, candidates: list[float]) -> None:
            measurements = []
            for setting in candidates:
                started = time.perf_counter()
                tracemalloc.start()
                try:
                    encoded, reconstructed = _external_codec_bytes(original, codec, setting)
                    _, peak_memory = tracemalloc.get_traced_memory()
                    measurements.append({
                        "setting": setting, "encoded": encoded, "reconstructed": reconstructed,
                        "psnr": psnr(original, reconstructed), "seconds": time.perf_counter() - started,
                        "peak": int(peak_memory), "size": len(encoded),
                        "bpp": bits_per_pixel(len(encoded), original.shape),
                    })
                except Exception:
                    pass
                finally:
                    tracemalloc.stop()
            if not measurements:
                rows.append(_record_result(
                    original, original, {"encoded_bytes": 0}, source, category, codec, 0.0, 0.0, 0.0,
                    status="unsupported", error="No candidate setting succeeded",
                    extra={**target_fields(), "quantizer": quantizer, "allocation_method": allocation_method},
                ))
                return
            selected = select_measurement(measurements)
            rows.append(_record_result(
                original, selected["reconstructed"], {"encoded_bytes": len(selected["encoded"])}, source, category,
                codec, float(selected["setting"]), selected["seconds"], 0.0, status=measurement_status(selected),
                peak_memory_bytes=selected["peak"], extra={
                    **target_fields(achieved_psnr=selected["psnr"], achieved_bpp=selected["bpp"]),
                    "setting_name": "quality" if codec == "jpeg" else "rate",
                    "quantizer": quantizer, "allocation_method": allocation_method,
                },
            ))

        if "jpeg" in selected_codecs:
            run_external("jpeg", [10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
        if include_jpeg2000 and "jpeg2000" in selected_codecs:
            run_external("jpeg2000", [0.25, 0.5, 1, 2, 4, 8, 16, 32])
    return _finalize_benchmark(rows, destination, manifest, image_records, {
        "mode": f"normalized-{normalization}", "target_psnr": target_psnr, "target_bpp": target_bpp,
        "bpp_tolerance": bpp_tolerance, "seed": seed,
        "input_dir": str(source_dir.resolve()), "dataset_root": str(root.resolve()),
        "steps": candidate_steps, "wavelets": selected_wavelets, "level": level, "codecs": sorted(selected_codecs),
        "quantizer": quantizer, "allocation_method": allocation_method,
        "surrogate": dataset_root is None, "roi_mask": str(roi_mask_path) if roi_mask_path else None,
        "split": split, "limit": limit,
        "adaptive_step_search": adaptive_enabled,
        "adaptive_step_search_config": {
            "max_iterations": adaptive_max_iterations,
            "min_step": adaptive_min_step,
            "max_step": adaptive_max_step,
            "strategy": "geometric bracket expansion/contraction for SWC candidates",
        },
        "candidate_grids": {"jpeg_quality": [10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
                             "jpeg2000_rate": [0.25, 0.5, 1, 2, 4, 8, 16, 32]},
        "measurement": {"peak_memory": "tracemalloc peak over candidate encode/decode operation"},
    })


def run_roi_comparison_benchmark(input_dir: str | Path, output_dir: str | Path, *, roi_mask_path: str | Path,
                                 target_bpp: float, step_values: list[float] | None = None,
                                 wavelets: list[str] | None = None, level: int = 2,
                                 dataset_root: str | Path | None = None, seed: int = 2026,
                                 codecs: list[str] | None = None, include_jpeg2000: bool = True,
                                 quantizer: str = "uniform", allocation_method: str = "greedy",
                                 split: str | None = None, limit: int | None = None,
                                 adaptive_step_search: bool = True) -> list[dict]:
    """Run matched target-BPP experiments with ROI disabled and enabled.

    The two branches use the same input manifest, candidate grid, target and
    codec configuration.  Achieved BPP remains explicit because SWC headers
    and the embedded ROI mask have real size; the output therefore measures
    the practical, approximate same-rate comparison rather than hiding header
    overhead.
    """
    if not roi_mask_path:
        raise ValueError("roi_mask_path is required for ROI comparison")
    if target_bpp <= 0:
        raise ValueError("target_bpp must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    common = {
        "target_bpp": float(target_bpp), "step_values": step_values, "wavelets": wavelets,
        "level": level, "dataset_root": dataset_root, "seed": seed, "codecs": codecs,
        "include_jpeg2000": include_jpeg2000, "quantizer": quantizer,
        "allocation_method": allocation_method, "split": split, "limit": limit,
        "adaptive_step_search": adaptive_step_search,
    }
    without_rows = run_normalized_benchmark(
        input_dir, destination / "without_roi", roi_mask_path=None, **common,
    )
    with_rows = run_normalized_benchmark(
        input_dir, destination / "with_roi", roi_mask_path=roi_mask_path, **common,
    )
    rows: list[dict] = []
    for enabled, branch_rows in ((False, without_rows), (True, with_rows)):
        for row in branch_rows:
            item = dict(row)
            item["roi_enabled"] = enabled
            item["roi_mode"] = "with_roi" if enabled else "without_roi"
            rows.append(item)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (destination / "roi_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (destination / "roi_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row.get("status") not in {"failed", "unsupported"}:
            grouped.setdefault((str(row["roi_mode"]), str(row["codec"])), []).append(row)
    summary = []
    for (roi_mode, codec), values in sorted(grouped.items()):
        def mean(field: str) -> float | None:
            numbers = [float(value[field]) for value in values
                       if _safe_float(value.get(field)) is not None and math.isfinite(float(value[field]))]
            return statistics.fmean(numbers) if numbers else None
        summary.append({
            "roi_mode": roi_mode, "codec": codec, "count": len(values),
            "bits_per_pixel_mean": mean("bits_per_pixel"), "psnr_mean": mean("psnr"),
            "ssim_mean": mean("ssim"), "roi_psnr_mean": mean("roi_psnr"),
            "background_psnr_mean": mean("background_psnr"),
        })
    (destination / "roi_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "roi_comparison_config.json").write_text(json.dumps({
        "mode": "matched-roi-target-bpp", "target_bpp": float(target_bpp),
        "roi_mask": str(Path(roi_mask_path).resolve()), "input_dir": str(Path(input_dir).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()) if dataset_root is not None else None,
        "steps": step_values or [4.0, 8.0, 16.0, 32.0], "wavelets": wavelets or ["haar", "db4"],
        "codecs": codecs or ["dwt", "dct", "prqmf4", "jpeg", "jpeg2000"],
        "quantizer": quantizer, "allocation_method": allocation_method,
        "split": split, "limit": limit,
    }, indent=2), encoding="utf-8")
    if rows:
        plot_rows = []
        for row in rows:
            if row.get("status") in {"failed", "unsupported"}:
                continue
            item = dict(row)
            item["codec"] = f"{row['codec']} ({row['roi_mode']})"
            plot_rows.append(item)
        _plot_results(plot_rows, destination / "roi_rate_distortion.png",
                      title="Matched target-BPP ROI comparison", include_non_ok=True)
    return rows


def _plot_results(rows: list[dict], path: Path, *, title: str = "Rate-distortion comparison",
                  include_non_ok: bool = False) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axis = plt.subplots(figsize=(8, 5))
    valid_rows = [row for row in rows if (include_non_ok or row.get("status") == "ok")
                  and math.isfinite(float(row.get("bits_per_pixel", 0)))
                  and math.isfinite(float(row.get("psnr", 0)))]
    if not valid_rows:
        return
    groups = sorted({row["codec"] for row in valid_rows})
    for codec in groups:
        selected = [row for row in valid_rows if row["codec"] == codec]
        selected.sort(key=lambda row: row["bits_per_pixel"])
        axis.plot([row["bits_per_pixel"] for row in selected], [row["psnr"] for row in selected], marker="o", label=codec)
    axis.set_xlabel("Bits per pixel")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
