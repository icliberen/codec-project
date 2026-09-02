"""SWC video layer with frame metadata and an optional deterministic GOP.

TR: I/P frame, motion metadata, ROI tracking ve partial transport akışlarını yönetir.
EN: Manages I/P frames, motion metadata, ROI tracking, and partial transport flows.
"""

from __future__ import annotations

import json
import csv
import posixpath
import time
import zlib
import zipfile
from pathlib import Path

import numpy as np

from .codec import decode_array, encode_array
from .metrics import bits_per_pixel, psnr, ssim
from .roi import load_mask
from .transport import simulate_transport


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video desteği için opencv-python paketini kurun: python -m pip install -r requirements-video.txt") from exc
    return cv2


def _motion_metadata(previous: np.ndarray | None, current: np.ndarray, enabled: bool,
                    method: str = "translation", block_size: int = 16,
                    search_radius: int = 8) -> dict:
    """Return informational motion vectors without changing deterministic decoding.

    TR: Motion metadata kaliteyi zorunlu olarak değiştirmez; predictor ayrı seçenektir.
    EN: Motion metadata does not force a quality change; prediction is separately enabled.
    """
    if not enabled or previous is None or method == "none":
        return {"type": "none"}
    if method not in {"translation", "block", "optical-flow"}:
        raise ValueError("motion method must be translation, block, optical-flow or none")
    try:
        cv2 = _cv2()
        old = previous if previous.ndim == 2 else cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        new = current if current.ndim == 2 else cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        if method == "translation":
            shift, response = cv2.phaseCorrelate(old.astype(np.float32), new.astype(np.float32))
            return {"type": "translation", "dx": float(shift[0]), "dy": float(shift[1]), "response": float(response)}
        if method == "optical-flow":
            points = cv2.goodFeaturesToTrack(old, maxCorners=256, qualityLevel=0.01, minDistance=7, blockSize=7)
            if points is None:
                return {"type": "optical-flow", "vector_count": 0, "reason": "no-trackable-features"}
            next_points, status, errors = cv2.calcOpticalFlowPyrLK(old, new, points, None)
            if next_points is None or status is None:
                return {"type": "optical-flow", "vector_count": 0, "reason": "flow-unavailable"}
            valid = status.reshape(-1).astype(bool)
            deltas = next_points.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
            if not len(deltas):
                return {"type": "optical-flow", "vector_count": 0, "reason": "no-valid-vectors"}
            vectors = [
                {"dx": float(delta[0]), "dy": float(delta[1])}
                for delta in deltas[:256]
            ]
            return {
                "type": "optical-flow", "vector_count": int(len(deltas)),
                "mean_dx": float(np.mean(deltas[:, 0])), "mean_dy": float(np.mean(deltas[:, 1])),
                "median_dx": float(np.median(deltas[:, 0])), "median_dy": float(np.median(deltas[:, 1])),
                "vectors": vectors,
            }
        height, width = new.shape[:2]
        vectors = []
        block_size = max(4, int(block_size))
        search_radius = max(0, int(search_radius))
        for top in range(0, height, block_size):
            for left in range(0, width, block_size):
                block = new[top : min(top + block_size, height), left : min(left + block_size, width)]
                if block.shape[0] < 4 or block.shape[1] < 4:
                    continue
                y0 = max(0, top - search_radius)
                x0 = max(0, left - search_radius)
                search = old[y0 : min(height, top + block.shape[0] + search_radius),
                             x0 : min(width, left + block.shape[1] + search_radius)]
                if search.shape[0] < block.shape[0] or search.shape[1] < block.shape[1]:
                    continue
                scores = cv2.matchTemplate(search, block, cv2.TM_SQDIFF_NORMED)
                min_score, _, min_location, _ = cv2.minMaxLoc(scores)
                vectors.append({
                    "x": int(left), "y": int(top),
                    "dx": int(x0 + min_location[0] - left), "dy": int(y0 + min_location[1] - top),
                    "error": float(min_score),
                })
        if not vectors:
            return {"type": "block", "vector_count": 0, "reason": "no-valid-blocks"}
        return {
            "type": "block", "block_size": block_size, "search_radius": search_radius,
            "vector_count": len(vectors), "mean_dx": float(np.mean([item["dx"] for item in vectors])),
            "mean_dy": float(np.mean([item["dy"] for item in vectors])), "vectors": vectors,
        }
    except Exception:
        return {"type": method, "vector_count": 0, "reason": "motion-estimation-unavailable"}


def _motion_predictor(previous: np.ndarray, motion: dict | None) -> np.ndarray:
    """Warp a reconstructed reference frame using stored motion metadata."""
    if previous is None or not motion or motion.get("type") in {None, "none"}:
        return previous
    cv2 = _cv2()
    reference = np.asarray(previous)
    height, width = reference.shape[:2]
    motion_type = str(motion.get("type"))
    if motion_type == "block":
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        for vector in motion.get("vectors", []):
            left = max(0, min(width, int(vector.get("x", 0))))
            top = max(0, min(height, int(vector.get("y", 0))))
            block_size = max(4, int(motion.get("block_size", 16)))
            right = min(width, left + block_size)
            bottom = min(height, top + block_size)
            if right <= left or bottom <= top:
                continue
            grid_x[top:bottom, left:right] += float(vector.get("dx", 0.0))
            grid_y[top:bottom, left:right] += float(vector.get("dy", 0.0))
        return cv2.remap(reference, grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dx = float(motion.get("dx", motion.get("mean_dx", 0.0)))
    dy = float(motion.get("dy", motion.get("mean_dy", 0.0)))
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(reference, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _video_roi_mask(spec, shape: tuple[int, int]) -> np.ndarray | None:
    """Normalize a video ROI path/array to the current frame dimensions."""
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        return load_mask(spec, shape)
    mask = np.asarray(spec, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != tuple(shape):
        raise ValueError("Video ROI mask must match the first frame dimensions")
    return np.clip(mask, 0.0, 1.0)


def _track_video_roi(previous_mask: np.ndarray | None, motion: dict | None) -> tuple[np.ndarray | None, str]:
    """Track an ROI with the same motion model as the P-frame predictor.

    A failed/empty motion estimate deliberately falls back to the last valid
    mask instead of silently dropping foreground quality protection.
    """
    if previous_mask is None:
        return None, "none"
    if not motion or motion.get("type") in {None, "none"}:
        return previous_mask, "fallback-static"
    try:
        tracked = np.clip(_motion_predictor(previous_mask, motion), 0.0, 1.0).astype(np.float32)
        if not np.isfinite(tracked).all() or float(np.max(tracked)) <= 1e-6:
            return previous_mask, "fallback-static"
        return tracked, "tracked"
    except Exception:
        return previous_mask, "fallback-static"


def compress_video(input_path: str | Path, output_dir: str | Path, **codec_options) -> dict:
    """Encode video as SWC I/P frames.

    ``gop_size=1`` is the backwards-compatible I-frame-only pipeline.  For a
    larger GOP, P-frames store an offset residual against the reconstructed
    previous frame, so decoder error propagation is explicit and measurable.

    TR: GOP büyüdükçe P-frame residual'ları önceki decode edilmiş kareye bağlanır.
    EN: With a larger GOP, P-frame residuals depend on the previously decoded frame.
    """
    cv2 = _cv2()
    source = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    gop_size = int(codec_options.pop("gop_size", 1))
    keyframe_interval = int(codec_options.pop("keyframe_interval", gop_size))
    keyframe_interval = max(1, keyframe_interval)
    motion_estimation = bool(codec_options.pop("motion_estimation", False))
    motion_method = str(codec_options.pop("motion_method", "translation" if motion_estimation else "none"))
    motion_compensation = bool(codec_options.pop("motion_compensation", False))
    roi_spec = codec_options.pop("roi_mask", codec_options.pop("roi_mask_path", None))
    roi_tracking = bool(codec_options.pop("roi_tracking", False))
    roi_mask: np.ndarray | None = None
    roi_enabled = roi_spec is not None
    if motion_method != "none":
        motion_estimation = True
    if gop_size < 1:
        raise ValueError("gop_size must be at least 1")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Video açılamadı: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[dict] = []
    previous_reconstructed: np.ndarray | None = None
    total_bytes = 0
    total_psnr = []
    total_ssim = []
    index = 0
    started_total = time.perf_counter()
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            motion = _motion_metadata(previous_reconstructed, rgb, motion_estimation, motion_method)
            if index == 0 and roi_spec is not None:
                roi_mask = _video_roi_mask(roi_spec, rgb.shape[:2])
                roi_source = "provided"
            elif roi_tracking and roi_mask is not None:
                roi_mask, roi_source = _track_video_roi(roi_mask, motion)
            elif roi_mask is not None:
                roi_source = "fallback-static"
            else:
                roi_source = "none"
            frame_type = "I" if previous_reconstructed is None or index % keyframe_interval == 0 or gop_size == 1 else "P"
            reference_index = None if frame_type == "I" else index - 1
            encoded_image = rgb
            predictor = previous_reconstructed
            if frame_type == "P" and motion_compensation and previous_reconstructed is not None:
                predictor = _motion_predictor(previous_reconstructed, motion)
            if frame_type == "P":
                encoded_image = np.clip(rgb.astype(np.int16) - predictor.astype(np.int16) + 128, 0, 255).astype(np.uint8)
            frame_path = destination / f"frame_{index:06d}.swc"
            encode_started = time.perf_counter()
            info = encode_array(
                encoded_image, frame_path, **codec_options,
                roi_mask=roi_mask,
                metadata={"video_frame_type": frame_type, "reference_frame": reference_index,
                          "residual_offset": 128, "motion_compensation": motion_compensation,
                          "roi_source": roi_source, "roi_tracking": roi_tracking},
            )
            encode_seconds = time.perf_counter() - encode_started
            decode_started = time.perf_counter()
            decoded_payload = decode_array(frame_path)
            if frame_type == "P":
                reconstructed = np.clip(predictor.astype(np.int16) + decoded_payload.astype(np.int16) - 128, 0, 255).astype(np.uint8)
            else:
                reconstructed = decoded_payload
            decode_seconds = time.perf_counter() - decode_started
            total_bytes += int(info["file_size"])
            total_psnr.append(psnr(rgb, reconstructed))
            total_ssim.append(ssim(rgb, reconstructed))
            frame_bytes = frame_path.read_bytes()
            frames.append({
                "index": index,
                "timestamp": index / fps,
                "file": frame_path.name,
                "frame_type": frame_type,
                "reference_frame": reference_index,
                "dependency": [] if reference_index is None else [reference_index],
                "motion": motion,
                "motion_compensation": motion_compensation,
                "roi_enabled": roi_mask is not None,
                "roi_source": roi_source,
                "roi_fraction": None if roi_mask is None else float(np.mean(roi_mask > 0.05)),
                "file_size": info["file_size"],
                "payload_checksum": int(zlib.crc32(frame_bytes) & 0xFFFFFFFF),
                "encode_seconds": encode_seconds,
                "decode_seconds": decode_seconds,
                "psnr": total_psnr[-1],
                "ssim": total_ssim[-1],
            })
            previous_reconstructed = reconstructed
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError("Videoda okunabilir kare bulunamadı")
    manifest = {
        "version": 2,
        "format": "smartcodec-video",
        "transport_mode": "simulated-only",
        "source": source.name,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": len(frames),
        "color_space": "RGB",
        "codec_options": codec_options,
        "gop_size": gop_size,
        "keyframe_interval": keyframe_interval,
        "motion_estimation": motion_estimation,
        "motion_method": motion_method,
        "motion_compensation": motion_compensation,
        "roi_enabled": roi_enabled,
        "roi_tracking": roi_tracking,
        "total_bytes": total_bytes,
        "average_bpp": float(total_bytes * 8 / max(1, width * height * len(frames))),
        "average_psnr": float(np.mean(total_psnr)),
        "average_ssim": float(np.mean(total_ssim)),
        "encode_seconds": time.perf_counter() - started_total,
        "frames": frames,
    }
    manifest_path = destination / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "frame_count": len(frames), "fps": fps, "width": width,
            "height": height, "gop_size": gop_size, "average_bpp": manifest["average_bpp"],
            "average_psnr": manifest["average_psnr"]}


def decompress_video(manifest_path: str | Path, output_path: str | Path) -> dict:
    cv2 = _cv2()
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid video manifest: {manifest_file}") from exc
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(manifest["fps"]),
                             (int(manifest["width"]), int(manifest["height"])))
    if not writer.isOpened():
        raise ValueError(f"Video çıktı dosyası açılamadı: {output_path}")
    previous: np.ndarray | None = None
    dropped = 0
    decoded_count = 0
    started = time.perf_counter()
    try:
        for frame in manifest["frames"]:
            try:
                frame_path = manifest_file.parent / frame["file"]
                if not frame_path.exists():
                    raise ValueError("frame payload missing")
                raw = frame_path.read_bytes()
                expected_checksum = frame.get("payload_checksum")
                if expected_checksum is not None and int(expected_checksum) != int(zlib.crc32(raw) & 0xFFFFFFFF):
                    raise ValueError("frame checksum mismatch")
                image = decode_array(frame_path)
                if frame.get("frame_type", "I") == "P":
                    if previous is None:
                        raise ValueError("P-frame has no valid reference")
                    offset = int(frame.get("residual_offset", 128))
                    predictor = previous
                    if manifest.get("motion_compensation", False) or frame.get("motion_compensation", False):
                        predictor = _motion_predictor(previous, frame.get("motion"))
                    image = np.clip(predictor.astype(np.int16) + image.astype(np.int16) - offset, 0, 255).astype(np.uint8)
                if image.ndim == 2:
                    image = np.repeat(image[..., None], 3, axis=2)
                previous = image
                decoded_count += 1
            except Exception:
                dropped += 1
                if previous is None:
                    previous = np.zeros((int(manifest["height"]), int(manifest["width"]), 3), dtype=np.uint8)
                image = previous
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return {"output": str(output_path), "frame_count": len(manifest["frames"]), "decoded_frames": decoded_count,
            "dropped_frame_count": dropped, "decode_seconds": time.perf_counter() - started,
            "gop_size": int(manifest.get("gop_size", 1))}


def simulate_video_transport(manifest_path: str | Path, output_dir: str | Path, *,
                             loss_rate: float = 0.0, mtu: int = 1200, latency_ms: float = 0.0,
                             jitter_ms: float = 0.0, bandwidth_mbps: float = 0.0,
                             burst_loss: int = 0, retransmission: bool = False,
                             fec: bool = False, seed: int = 2026) -> dict:
    """Transport each SWC frame independently and write a decodable partial manifest.

    ZIP bundle transport remains available for atomic delivery. This frame-level
    path is the loss-tolerant video mode: segmented SWC frames that survive the
    network are kept, missing frames remain absent so ``decompress_video`` uses
    its documented GOP fallback, and every frame gets its own transport report.
    """
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid video manifest: {manifest_file}") from exc
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Video manifest has no frames")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_frames = []
    frame_reports = []
    partial_count = 0
    dropped_count = 0
    total_packets = 0
    total_received = 0
    total_missing = 0
    quality_values = []
    for index, original_frame in enumerate(frames):
        if not isinstance(original_frame, dict) or not original_frame.get("file"):
            raise ValueError("Video manifest contains an invalid frame entry")
        frame_name = _safe_video_bundle_member(original_frame["file"])
        source_frame = manifest_file.parent / frame_name
        frame = dict(original_frame)
        target_frame = destination / frame_name
        target_frame.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "frame_index": int(index),
            "frame": frame_name,
            "source": str(source_frame),
            "output": None,
        }
        if not source_frame.is_file():
            frame["transport_status"] = "missing-source"
            frame["transport_partial"] = True
            dropped_count += 1
            frame_reports.append(report | {"partial_decode": True, "reason": "missing-source"})
            output_frames.append(frame)
            continue
        raw = source_frame.read_bytes()
        restored, transport_report = simulate_transport(
            raw, mtu=mtu, loss_rate=loss_rate, latency_ms=latency_ms, jitter_ms=jitter_ms,
            bandwidth_mbps=bandwidth_mbps, burst_loss=burst_loss,
            retransmission=retransmission, fec=fec, seed=int(seed) + index,
        )
        report.update({
            "packets": int(transport_report.get("packets", 0)),
            "received": int(transport_report.get("received", 0)),
            "missing_packet_count": len(transport_report.get("missing_packet_ids", [])),
            "partial_decode": bool(transport_report.get("partial_decode", False)),
            "partial_decode_mode": transport_report.get("partial_decode_mode", "none"),
            "zero_filled_segment_count": int(transport_report.get("zero_filled_segment_count", 0)),
            "quality_degradation": float(transport_report.get("quality_degradation", 0.0)),
        })
        total_packets += report["packets"]
        total_received += report["received"]
        total_missing += report["missing_packet_count"]
        quality_values.append(report["quality_degradation"])
        if restored is None:
            frame["transport_status"] = "dropped"
            frame["transport_partial"] = True
            dropped_count += 1
        else:
            target_frame.write_bytes(restored)
            try:
                decode_array(target_frame)
            except Exception:
                target_frame.unlink(missing_ok=True)
                restored = None
                frame["transport_status"] = "dropped-undecodable"
                frame["transport_partial"] = True
                dropped_count += 1
            else:
                frame["transport_status"] = "partial" if transport_report.get("partial_decode") else "complete"
                frame["transport_partial"] = bool(transport_report.get("partial_decode"))
                frame["payload_checksum"] = int(zlib.crc32(restored) & 0xFFFFFFFF)
                frame["file_size"] = len(restored)
                report["output"] = str(target_frame)
                if transport_report.get("partial_decode"):
                    partial_count += 1
        output_frames.append(frame)
        frame_reports.append(report)
    output_manifest = dict(manifest)
    output_manifest["frames"] = output_frames
    output_manifest["transport_mode"] = "frame-level-simulation"
    output_manifest["simulation"] = True
    output_manifest["hardware_test"] = False
    output_manifest["transport_config"] = {
        "loss_rate": float(loss_rate), "mtu": int(mtu), "latency_ms": float(latency_ms),
        "jitter_ms": float(jitter_ms), "bandwidth_mbps": float(bandwidth_mbps),
        "burst_loss": int(burst_loss), "retransmission": bool(retransmission),
        "fec": bool(fec), "seed": int(seed),
    }
    output_manifest["transport_summary"] = {
        "frame_count": len(frames), "partial_frame_count": partial_count,
        "dropped_frame_count": dropped_count, "packets": total_packets,
        "received": total_received, "missing_packet_count": total_missing,
        "mean_quality_degradation": float(np.mean(quality_values)) if quality_values else 0.0,
    }
    output_manifest_path = destination / "video_manifest.json"
    output_manifest_path.write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    report_path = destination / "video_transport_report.json"
    report_payload = {
        "manifest": str(manifest_file), "output_manifest": str(output_manifest_path),
        "frames": frame_reports, "summary": output_manifest["transport_summary"],
        "simulation": True, "hardware_test": False,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return {"manifest": str(output_manifest_path), "report": str(report_path),
            **output_manifest["transport_summary"]}


def run_video_benchmark(input_path: str | Path, output_dir: str | Path, *,
                        gop_sizes: list[int] | tuple[int, ...] = (1, 4, 8),
                        mode: str = "lossy", codec: str = "dwt", wavelet: str = "haar",
                        level: int = 2, step: float = 12.0,
                        motion_method: str = "translation", motion_compensation: bool = False,
                        roi_mask_path: str | Path | None = None, roi_tracking: bool = False,
                        transport_segments: bool = False, transport_tiles: bool = False,
                        transport_tile_size: int = 64) -> dict:
    """Compare I-frame-only and GOP video settings on one identical input."""
    values = sorted({int(value) for value in gop_sizes})
    if not values or any(value < 1 for value in values):
        raise ValueError("video benchmark gop sizes must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for gop_size in values:
        run_dir = destination / f"gop_{gop_size}"
        try:
            info = compress_video(
                input_path, run_dir, mode=mode, codec=codec, wavelet=wavelet,
                level=level, step=step, gop_size=gop_size, keyframe_interval=gop_size,
                motion_estimation=gop_size > 1 and motion_method != "none",
                motion_method=motion_method if gop_size > 1 else "none",
                motion_compensation=bool(motion_compensation and gop_size > 1),
                roi_mask_path=roi_mask_path, roi_tracking=bool(roi_tracking and gop_size > 1),
                transport_segments=bool(transport_segments), transport_tiles=bool(transport_tiles),
                transport_tile_size=int(transport_tile_size),
            )
            manifest_path = Path(info["manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            decoded = decompress_video(manifest_path, run_dir / "benchmark_decoded.mp4")
            frame_rows = manifest.get("frames", [])
            rows.append({
                "gop_size": gop_size,
                "frame_count": int(manifest["frame_count"]),
                "fps": float(manifest["fps"]),
                "total_bytes": int(manifest["total_bytes"]),
                "average_bpp": float(manifest["average_bpp"]),
                "average_psnr": float(manifest["average_psnr"]),
                "average_ssim": float(manifest["average_ssim"]),
                "encode_seconds": float(manifest["encode_seconds"]),
                "decode_seconds": float(decoded["decode_seconds"]),
                "decoded_frames": int(decoded["decoded_frames"]),
                "dropped_frame_count": int(decoded["dropped_frame_count"]),
                "i_frame_count": sum(frame.get("frame_type") == "I" for frame in frame_rows),
                "p_frame_count": sum(frame.get("frame_type") == "P" for frame in frame_rows),
                "status": "ok",
                "manifest": str(manifest_path),
                "output_dir": str(run_dir),
            })
        except Exception as exc:
            rows.append({"gop_size": gop_size, "status": "failed", "error": str(exc),
                         "manifest": None, "output_dir": str(run_dir)})
    config = {
        "input": str(Path(input_path).resolve()), "output_dir": str(destination.resolve()),
        "gop_sizes": values, "mode": mode, "codec": codec, "wavelet": wavelet,
        "level": int(level), "step": float(step), "motion_method": motion_method,
        "motion_compensation": bool(motion_compensation), "roi_mask": str(roi_mask_path) if roi_mask_path else None,
        "roi_tracking": bool(roi_tracking), "simulation": True, "hardware_test": False,
        "transport_segments": bool(transport_segments),
        "transport_tiles": bool(transport_tiles),
        "transport_tile_size": int(transport_tile_size),
    }
    (destination / "video_benchmark_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (destination / "video_benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (destination / "video_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    plot_path = destination / "video_gop_comparison.png"
    plot_error = None
    try:
        import matplotlib.pyplot as plt
        valid = [row for row in rows if row.get("status") == "ok"]
        if valid:
            figure, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot([row["gop_size"] for row in valid], [row["total_bytes"] for row in valid], marker="o")
            axes[0].set_title("Video size")
            axes[0].set_xlabel("GOP size")
            axes[0].set_ylabel("SWC bytes")
            axes[1].plot([row["gop_size"] for row in valid], [row["average_psnr"] for row in valid], marker="o", label="PSNR")
            axes[1].set_title("Video quality")
            axes[1].set_xlabel("GOP size")
            axes[1].set_ylabel("Average PSNR (dB)")
            axes[1].grid(True, alpha=0.25)
            figure.tight_layout()
            figure.savefig(plot_path, dpi=150)
            plt.close(figure)
        else:
            plot_path = None
    except Exception as exc:
        plot_error = str(exc)
        plot_path = None
    summary = {
        "rows": len(rows), "successful_rows": sum(row.get("status") == "ok" for row in rows),
        "failed_rows": sum(row.get("status") != "ok" for row in rows),
        "plot": str(plot_path) if plot_path else None, "plot_error": plot_error,
        "simulation": True, "hardware_test": False,
    }
    (destination / "video_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"output_dir": str(destination), "rows": len(rows), "summary": summary,
            "csv": str(destination / "video_benchmark.csv"),
            "json": str(destination / "video_benchmark.json"), "plot": summary["plot"]}


def _safe_video_bundle_member(name: str) -> str:
    """Validate a ZIP member before it is written/extracted."""
    normalized = posixpath.normpath(str(name).replace("\\", "/"))
    if normalized in {"", "."} or normalized.startswith("/") or normalized.startswith("../") or "/../" in normalized:
        raise ValueError(f"Unsafe video bundle member: {name}")
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ValueError(f"Unsafe video bundle member: {name}")
    return normalized


def package_video(manifest_path: str | Path, bundle_path: str | Path) -> dict:
    """Pack a video manifest and every referenced SWC frame into one safe ZIP."""
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid video manifest: {manifest_file}") from exc
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Video manifest has no frames")
    output = Path(bundle_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    members = {"video_manifest.json"}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("video_manifest.json", json.dumps(manifest, indent=2))
        for frame in frames:
            if not isinstance(frame, dict) or not frame.get("file"):
                raise ValueError("Video manifest contains an invalid frame entry")
            frame_name = _safe_video_bundle_member(frame["file"])
            if frame_name in members:
                raise ValueError(f"Duplicate video bundle member: {frame_name}")
            frame_file = manifest_file.parent / frame_name
            if not frame_file.is_file():
                raise FileNotFoundError(f"Video frame payload not found: {frame_file}")
            archive.write(frame_file, arcname=frame_name)
            members.add(frame_name)
    return {"bundle": str(output), "manifest": str(manifest_file), "frame_count": len(frames),
            "member_count": len(members), "file_size": output.stat().st_size}


def unpack_video(bundle_path: str | Path, output_dir: str | Path) -> dict:
    """Safely unpack a video ZIP and return the extracted manifest path."""
    bundle = Path(bundle_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(bundle, "r") as archive:
            names = [_safe_video_bundle_member(name) for name in archive.namelist()]
            if "video_manifest.json" not in names:
                raise ValueError("Video bundle has no video_manifest.json")
            if len(names) != len(set(names)):
                raise ValueError("Video bundle contains duplicate members")
            for original_name, safe_name in zip(archive.namelist(), names):
                target = (destination / safe_name).resolve()
                root = destination.resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"Unsafe video bundle extraction path: {original_name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(original_name))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid video bundle: {bundle}") from exc
    manifest_file = destination / "video_manifest.json"
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Extracted video bundle manifest is invalid") from exc
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Extracted video bundle manifest has no frames")
    frame_count = 0
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Extracted video bundle contains an invalid frame entry")
        frame_name = _safe_video_bundle_member(frame.get("file", ""))
        if not (destination / frame_name).is_file():
            raise ValueError(f"Video bundle is missing frame payload: {frame_name}")
        frame_count += 1
    return {"bundle": str(bundle), "output_dir": str(destination), "manifest": str(manifest_file),
            "frame_count": frame_count, "member_count": len(names)}
