"""TR: SWC container, codec adayları ve decode akışı / EN: SWC container, codec candidates, and decode pipeline."""

from __future__ import annotations

import base64
import json
import struct
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pywt

from .allocation import allocate_subbands, choose_step_multiplier, entropy_bits
from .colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from .dct import JPEG_LUMA_QUANTIZATION
from .entropy import HuffmanEncoded, decode_coefficients, encode_coefficients
from .image_io import from_float, load_image, save_image, to_float
from .metrics import bits_per_pixel, psnr
from .quantization import coefficient_steps, dequantize, quantize, resize_mask
from .reversible import forward_53, inverse_53, max_level_53
from .roi import feather_mask, load_mask, regions_metadata, regions_to_mask
from .standard import decode_standard, encode_standard
from .transforms import (
    block_dct,
    block_idct,
    block_iprqmf4,
    block_prqmf4,
    canonical_wavelet,
    dwt2,
    idwt2,
    PRQMF4_DESCRIPTION,
    WAVELET_BOUNDARY_MODE,
)


MAGIC = b"SWC1"
STANDARD_JPEG_SUFFIXES = {".jpg", ".jpeg"}
STANDARD_JPEG2000_SUFFIXES = {".jp2", ".j2k", ".j2c", ".jpf"}
VERSION = 2
SUPPORTED_VERSIONS = {1, VERSION}
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024
MAX_PIXELS = 200_000_000


def _write_container(header: dict, payload: bytes, destination: str | Path) -> None:
    header = dict(header)
    header.setdefault("payload_crc32", int(zlib.crc32(payload) & 0xFFFFFFFF))
    header.setdefault("container", "SWC")
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack(">I", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(payload)


def _read_container(source: str | Path) -> tuple[dict, bytes]:
    try:
        size = Path(source).stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot inspect SWC file: {source}") from exc
    if size < 8:
        raise ValueError("Truncated SWC container")
    if size > MAX_PAYLOAD_BYTES + MAX_HEADER_BYTES + 8:
        raise ValueError("SWC file is larger than the safety limit")
    with Path(source).open("rb") as stream:
        if stream.read(4) != MAGIC:
            raise ValueError("Not a Smart Wavelet Codec file")
        raw_length = stream.read(4)
        if len(raw_length) != 4:
            raise ValueError("Truncated SWC header")
        header_length = struct.unpack(">I", raw_length)[0]
        if header_length > MAX_HEADER_BYTES:
            raise ValueError("Unreasonably large SWC header")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("Truncated SWC header payload")
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid SWC JSON header") from exc
        if not isinstance(header, dict):
            raise ValueError("SWC header must be an object")
        payload = stream.read()
    checksum = header.get("payload_crc32")
    if checksum is not None and int(checksum) != int(zlib.crc32(payload) & 0xFFFFFFFF):
        raise ValueError("SWC payload checksum mismatch")
    _validate_header(header, len(payload))
    return header, payload


def _validate_header(header: dict, payload_length: int | None = None) -> None:
    # TR: Decoder'a ulaşmadan önce sınırlar ve checksum girdileri doğrulanır.
    # EN: Validate bounds and checksum inputs before any decoder touches the payload.
    version = header.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported SWC version: {version}")
    shape = header.get("shape")
    if not isinstance(shape, list) or len(shape) not in (2, 3) or any(int(value) <= 0 for value in shape):
        raise ValueError("SWC header has an invalid image shape")
    if int(np.prod(shape[:2])) > MAX_PIXELS:
        raise ValueError("SWC dimensions exceed the safety limit")
    channels = int(header.get("channels", 0))
    if channels not in (1, 3) or (len(shape) == 3 and int(shape[2]) != channels):
        raise ValueError("SWC header has an invalid channel count")
    channel_offsets = header.get("channel_offsets")
    if channel_offsets is not None:
        if (not isinstance(channel_offsets, list) or len(channel_offsets) != channels
                or any(not np.isfinite(float(value)) for value in channel_offsets)):
            raise ValueError("SWC header has invalid channel offsets")
    if header.get("mode") not in {"lossy", "lossless"}:
        raise ValueError("SWC header has an invalid mode")
    if payload_length is not None and payload_length > MAX_PAYLOAD_BYTES:
        raise ValueError("SWC payload exceeds the safety limit")
    if header.get("mode") == "lossy" and float(header.get("step", 0)) <= 0:
        raise ValueError("SWC header has an invalid quantization step")
    if "coefficient_count" in header:
        coefficient_count = int(header["coefficient_count"])
        if coefficient_count < 0 or coefficient_count > MAX_PIXELS * channels * 8:
            raise ValueError("SWC header has an invalid coefficient count")
    if "frequencies" in header and len(header["frequencies"]) != 256:
        raise ValueError("SWC Huffman frequency table must contain 256 entries")
    if "bands" in header:
        if not isinstance(header["bands"], list) or len(header["bands"]) > channels * 64:
            raise ValueError("SWC header has an invalid band table")
    if header.get("payload_layout") == "band-segmented-v1":
        bands = header.get("bands", [])
        if int(header.get("transport_segment_count", len(bands))) != len(bands):
            raise ValueError("SWC segmented payload count does not match its band table")
        for entry in bands:
            if int(entry.get("transport_offset", -1)) < 0 or int(entry.get("transport_size", -1)) < 0:
                raise ValueError("SWC segmented payload has an invalid range")
            regions = entry.get("transport_regions")
            tiles = entry.get("transport_tiles")
            pieces = tiles or regions
            if pieces:
                if tiles and not isinstance(tiles, list):
                    raise ValueError("SWC segmented tile table must be a list")
                expected_count = 0
                for piece in pieces:
                    if len(piece.get("transport_frequencies", [])) != 256:
                        raise ValueError("SWC segmented payload has an invalid frequency table")
                    piece_count = int(piece.get("transport_count", -1))
                    if piece_count < 0:
                        raise ValueError("SWC segmented payload has an invalid piece count")
                    expected_count += piece_count
                    if tiles:
                        region = piece.get("coefficient_region")
                        if not isinstance(region, list) or len(region) != 4:
                            raise ValueError("SWC segmented tile has no valid coefficient region")
                        row_start, row_end, col_start, col_end = (int(value) for value in region)
                        shape = tuple(int(value) for value in entry.get("shape", []))
                        if len(shape) != 2 or not (0 <= row_start < row_end <= shape[0] and
                                                   0 <= col_start < col_end <= shape[1]):
                            raise ValueError("SWC segmented tile region is outside its band")
                        if piece_count != (row_end - row_start) * (col_end - col_start):
                            raise ValueError("SWC segmented tile count does not match its region")
                if tiles and expected_count != int(entry.get("count", expected_count)):
                    raise ValueError("SWC segmented tile table does not cover its band")
            elif len(entry.get("transport_frequencies", [])) != 256:
                raise ValueError("SWC segmented payload has an invalid frequency table")


def _mask_from_header(header: dict) -> np.ndarray | None:
    encoded = header.get("roi_mask")
    if not encoded:
        return None
    try:
        raw = zlib.decompress(base64.b64decode(encoded, validate=True))
    except Exception as exc:
        raise ValueError("Invalid compressed ROI mask in SWC header") from exc
    shape = tuple(header["shape"][:2])
    if len(raw) != int(np.prod(shape)):
        raise ValueError("ROI mask size does not match SWC image shape")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0


def _coeff_band_entries(coefficients: list, level: int) -> list[tuple[str, int, np.ndarray]]:
    entries: list[tuple[str, int, np.ndarray]] = [("approx", level, coefficients[0])]
    for index, detail_tuple in enumerate(coefficients[1:], start=1):
        detail_level = level - index + 1
        for name, array in zip(("horizontal", "vertical", "diagonal"), detail_tuple):
            entries.append((name, detail_level, array))
    return entries


def _coefficients_from_entries(entries: list[dict], values: list[int], header: dict) -> list[list]:
    channel_count = int(header["channels"])
    wavelet_level = int(header["level"])
    grouped: list[dict[tuple[str, int], np.ndarray]] = [dict() for _ in range(channel_count)]
    mask = _mask_from_header(header)
    position = 0
    for entry in entries:
        count = int(entry["count"])
        raw_values = np.asarray(values[position : position + count], dtype=np.int32).reshape(tuple(entry["shape"]))
        position += count
        steps = coefficient_steps(
            float(header["step"]),
            raw_values.shape,
            roi_mask=mask,
            roi_strength=float(header.get("roi_strength", 0.65)),
            level=int(entry["level"]),
            detail=header.get("quantizer") == "scalar" and entry["kind"] != "approx",
            step_multiplier=float(entry.get("step_multiplier", 1.0)),
        )
        grouped[int(entry["channel"])][(entry["kind"], int(entry["level"]))] = dequantize(raw_values, steps)

    output: list[list] = []
    for channel in range(channel_count):
        channel_bands = grouped[channel]
        approx = channel_bands[("approx", wavelet_level)]
        details = []
        for detail_level in range(wavelet_level, 0, -1):
            details.append(
                (
                    channel_bands[("horizontal", detail_level)],
                    channel_bands[("vertical", detail_level)],
                    channel_bands[("diagonal", detail_level)],
                )
            )
        output.append([approx, *details])
    return output


def _tile_edges(source_size: int, coefficient_size: int, tile_size: int) -> list[int]:
    """Map source-image tile boundaries onto one coefficient-grid dimension."""
    if source_size <= 0 or coefficient_size <= 0 or tile_size <= 0:
        raise ValueError("transport tile dimensions must be positive")
    edges = [0]
    for source_edge in range(tile_size, source_size, tile_size):
        mapped = int(np.ceil(source_edge * coefficient_size / source_size))
        if mapped > edges[-1] and mapped < coefficient_size:
            edges.append(mapped)
    if edges[-1] != coefficient_size:
        edges.append(coefficient_size)
    return edges


def _encode_band_segments(band_values: list[list[int]], entries: list[dict],
                          roi_mask: np.ndarray | None = None,
                          tile_size: int | None = None,
                          source_shape: tuple[int, int] | None = None) -> bytes:
    """Encode bands as semantic streams, optionally split into spatial tiles."""
    if len(band_values) != len(entries):
        raise ValueError("Band values and entries must have the same length")
    if tile_size is not None:
        if source_shape is None or len(source_shape) != 2:
            raise ValueError("source_shape is required for spatial transport tiles")
        tile_size = int(tile_size)
        if tile_size < 8:
            raise ValueError("transport tile size must be at least 8 pixels")
    payload = bytearray()
    for values, entry in zip(band_values, entries):
        values_array = np.asarray(values, dtype=np.int32)
        if tile_size is not None:
            band_array = values_array.reshape(tuple(entry["shape"]))
            row_edges = _tile_edges(int(source_shape[0]), band_array.shape[0], tile_size)
            col_edges = _tile_edges(int(source_shape[1]), band_array.shape[1], tile_size)
            tiles = []
            band_start = len(payload)
            for tile_y, (row_start, row_end) in enumerate(zip(row_edges[:-1], row_edges[1:])):
                for tile_x, (col_start, col_end) in enumerate(zip(col_edges[:-1], col_edges[1:])):
                    tile_values = band_array[row_start:row_end, col_start:col_end].reshape(-1).tolist()
                    encoded = encode_coefficients([int(value) for value in tile_values])
                    offset = len(payload)
                    payload.extend(encoded.data)
                    source_region = [
                        int(round(row_start * source_shape[0] / band_array.shape[0])),
                        int(round(row_end * source_shape[0] / band_array.shape[0])),
                        int(round(col_start * source_shape[1] / band_array.shape[1])),
                        int(round(col_end * source_shape[1] / band_array.shape[1])),
                    ]
                    tile_roi = False
                    if roi_mask is not None:
                        y0, y1, x0, x1 = source_region
                        tile_roi = bool(np.any(np.asarray(roi_mask)[y0:y1, x0:x1] > 0.05))
                    tiles.append({
                        "tile_y": int(tile_y),
                        "tile_x": int(tile_x),
                        "coefficient_region": [int(row_start), int(row_end), int(col_start), int(col_end)],
                        "source_region": source_region,
                        "transport_offset": int(offset),
                        "transport_size": int(len(encoded.data)),
                        "transport_bit_length": int(encoded.bit_length),
                        "transport_frequencies": list(encoded.frequencies),
                        "transport_count": int(len(tile_values)),
                        "transport_priority": 0 if tile_roi or entry.get("kind") == "approx" else 1,
                    })
            entry.update({
                "transport_offset": int(band_start),
                "transport_size": int(len(payload) - band_start),
                "transport_priority": min(tile["transport_priority"] for tile in tiles),
                "transport_tiles": tiles,
            })
            continue
        region_masks: list[tuple[str, np.ndarray]] = []
        if roi_mask is not None:
            coefficient_mask = resize_mask(roi_mask, tuple(entry["shape"])).reshape(-1) > 0.05
            if np.any(coefficient_mask) and np.any(~coefficient_mask):
                region_masks = [("roi", coefficient_mask), ("background", ~coefficient_mask)]
        if not region_masks:
            region_masks = [("all", np.ones(values_array.size, dtype=bool))]
        regions = []
        band_start = len(payload)
        for region_name, region_mask in region_masks:
            region_values = values_array[region_mask].tolist()
            encoded = encode_coefficients([int(value) for value in region_values])
            offset = len(payload)
            payload.extend(encoded.data)
            priority = 0 if region_name == "roi" or entry.get("kind") == "approx" else 1
            regions.append({
                "region": region_name,
                "transport_offset": int(offset),
                "transport_size": int(len(encoded.data)),
                "transport_bit_length": int(encoded.bit_length),
                "transport_frequencies": list(encoded.frequencies),
                "transport_count": int(len(region_values)),
                "transport_priority": priority,
            })
        entry.update({
            "transport_offset": int(band_start),
            "transport_size": int(len(payload) - band_start),
            "transport_priority": min(region["transport_priority"] for region in regions),
        })
        if roi_mask is not None and len(regions) > 1:
            entry["transport_regions"] = regions
        else:
            entry.update({
                "transport_bit_length": regions[0]["transport_bit_length"],
                "transport_frequencies": regions[0]["transport_frequencies"],
            })
    return bytes(payload)


def _decode_band_segments(header: dict, payload: bytes) -> list[int]:
    """Decode independently packed bands, ROI streams, or spatial tiles."""
    values: list[int] = []
    for entry in header.get("bands", []):
        offset = int(entry.get("transport_offset", -1))
        size = int(entry.get("transport_size", -1))
        if offset < 0 or size < 0 or offset + size > len(payload):
            raise ValueError("Invalid segmented SWC payload range")
        regions = entry.get("transport_regions")
        tiles = entry.get("transport_tiles")
        if tiles:
            reconstructed = np.zeros(int(entry["count"]), dtype=np.int32).reshape(tuple(entry["shape"]))
            for tile in tiles:
                tile_offset = int(tile.get("transport_offset", -1))
                tile_size = int(tile.get("transport_size", -1))
                region = tile.get("coefficient_region", [])
                if tile_offset < 0 or tile_size < 0 or tile_offset + tile_size > len(payload) or len(region) != 4:
                    raise ValueError("Invalid segmented SWC tile payload range")
                row_start, row_end, col_start, col_end = (int(value) for value in region)
                if not (0 <= row_start < row_end <= reconstructed.shape[0] and
                        0 <= col_start < col_end <= reconstructed.shape[1]):
                    raise ValueError("Invalid segmented SWC tile coefficient region")
                encoded = HuffmanEncoded(
                    payload[tile_offset : tile_offset + tile_size],
                    int(tile.get("transport_bit_length", 0)),
                    list(tile.get("transport_frequencies", [])),
                )
                tile_count = (row_end - row_start) * (col_end - col_start)
                decoded = decode_coefficients(encoded, int(tile.get("transport_count", tile_count)))
                if len(decoded) != tile_count:
                    raise ValueError("Segmented SWC tile count mismatch")
                reconstructed[row_start:row_end, col_start:col_end] = np.asarray(decoded, dtype=np.int32).reshape(
                    (row_end - row_start, col_end - col_start)
                )
            values.extend(reconstructed.reshape(-1).tolist())
        elif regions:
            source_mask = _mask_from_header(header)
            if source_mask is None:
                raise ValueError("Segmented ROI payload has no ROI mask")
            coefficient_mask = resize_mask(source_mask, tuple(entry["shape"])).reshape(-1) > 0.05
            reconstructed = np.zeros(int(entry["count"]), dtype=np.int32)
            for region in regions:
                region_offset = int(region.get("transport_offset", -1))
                region_size = int(region.get("transport_size", -1))
                region_name = str(region.get("region"))
                if region_offset < 0 or region_size < 0 or region_offset + region_size > len(payload):
                    raise ValueError("Invalid segmented SWC ROI payload range")
                region_mask = coefficient_mask if region_name == "roi" else ~coefficient_mask
                encoded = HuffmanEncoded(
                    payload[region_offset : region_offset + region_size],
                    int(region.get("transport_bit_length", 0)),
                    list(region.get("transport_frequencies", [])),
                )
                decoded = decode_coefficients(encoded, int(region.get("transport_count", np.count_nonzero(region_mask))))
                positions = np.flatnonzero(region_mask)
                if len(decoded) != len(positions):
                    raise ValueError("Segmented SWC ROI region count mismatch")
                reconstructed[positions] = np.asarray(decoded, dtype=np.int32)
            values.extend(reconstructed.tolist())
        else:
            encoded = HuffmanEncoded(
                payload[offset : offset + size],
                int(entry.get("transport_bit_length", 0)),
                list(entry.get("transport_frequencies", [])),
            )
            values.extend(decode_coefficients(encoded, int(entry["count"])))
    if len(values) != int(header.get("coefficient_count", len(values))):
        raise ValueError("Segmented SWC coefficient count mismatch")
    return values


def _encode_lossy(image: np.ndarray, wavelet: str, level: int, step: float, quantizer: str,
                  roi_mask: np.ndarray | None, roi_strength: float,
                  colorspace: str, allocation_method: str = "greedy",
                  transport_segments: bool = False, transport_tiles: bool = False,
                  transport_tile_size: int = 64,
                  compact_header: bool = False) -> tuple[dict, bytes]:
    array = to_float(image)
    channels = 1 if array.ndim == 2 else array.shape[2]
    source_shape = tuple(array.shape)
    if channels == 3 and colorspace == "ycbcr":
        array = rgb_to_ycbcr(array)
    elif colorspace not in {"rgb", "ycbcr"}:
        raise ValueError("colorspace must be rgb or ycbcr")
    # Preserve the per-channel DC component explicitly and transform only the
    # zero-mean residual.  At very low rates a coarse approximation-band step
    # can otherwise round the Y/Cb/Cr DC values to a different level, creating
    # a severe whole-image brightness or colour cast.  The few offset values
    # are included in the container size used by target-BPP search.
    if channels == 1:
        channel_offsets = np.asarray([np.mean(array, dtype=np.float64)], dtype=np.float32)
        array = array - channel_offsets[0]
    else:
        channel_offsets = np.mean(array, axis=(0, 1), dtype=np.float64).astype(np.float32)
        array = array - channel_offsets.reshape((1, 1, channels))
    channel_arrays = [array] if channels == 1 else [array[..., index] for index in range(channels)]
    requested_wavelet = wavelet
    wavelet = canonical_wavelet(wavelet)
    maximum = pywt.dwt_max_level(min(array.shape[:2]), pywt.Wavelet(wavelet).dec_len)
    if maximum < 1:
        raise ValueError("Image is too small for the selected wavelet")
    level = int(level)
    if level < 1 or level > maximum:
        raise ValueError(f"Level {level} is invalid for shape {array.shape[:2]}; maximum is {maximum}")

    entries: list[dict] = []
    all_values: list[int] = []
    band_values: list[list[int]] = []
    allocation_bands: list[dict] = []
    allocation_reports: list[dict] = []
    # TR: Her kanal bağımsız işlendiği için allocation DP'si kanal başına küçük kalır.
    # EN: Channels are independent, so the allocation DP remains small per channel.
    for channel, plane in enumerate(channel_arrays):
        coefficients = dwt2(plane, wavelet, level)
        band_entries = _coeff_band_entries(coefficients, level)
        allocated_choices = None
        if quantizer == "scalar" and allocation_method in {"lagrangian", "dp"}:
            allocation = allocate_subbands([item[2] for item in band_entries], step, method=allocation_method,
                                           pixels=plane.shape[0] * plane.shape[1])
            allocated_choices = allocation["choices"]
            allocation_reports.append({
                "channel": int(channel),
                "method": allocation["method"],
                "search_states": int(allocation["search_states"]),
                "candidate_combinations": int(allocation["candidate_combinations"]),
                "optimality": allocation["optimality"],
            })
        for band_index, (kind, band_level, coefficient_array) in enumerate(band_entries):
            step_multiplier = 1.0
            if quantizer == "scalar":
                if allocated_choices is not None:
                    step_multiplier = float(allocated_choices[band_index]["multiplier"])
                else:
                    step_multiplier = choose_step_multiplier(coefficient_array, step, band_level, kind)
            steps = coefficient_steps(
                step,
                coefficient_array.shape,
                roi_mask=roi_mask,
                roi_strength=roi_strength,
                level=band_level,
                detail=quantizer == "scalar" and kind != "approx",
                step_multiplier=step_multiplier,
            )
            quantized = quantize(coefficient_array, steps)
            entries.append({
                "channel": channel,
                "kind": kind,
                "level": band_level,
                "shape": list(quantized.shape),
                "count": int(quantized.size),
                "step_multiplier": step_multiplier,
            })
            if not compact_header:
                allocation_bands.append({
                    "channel": int(channel), "kind": kind, "level": int(band_level),
                    "count": int(quantized.size), "step_multiplier": float(step_multiplier),
                    "estimated_entropy_bits": float(entropy_bits(quantized)),
                    "coefficient_variance": float(np.var(coefficient_array)),
                })
            values = quantized.reshape(-1).tolist()
            band_values.append(values)
            all_values.extend(values)

    encoded = None if transport_segments else encode_coefficients(all_values)
    payload = _encode_band_segments(
        band_values, entries, roi_mask,
        tile_size=transport_tile_size if transport_tiles else None,
        source_shape=source_shape[:2],
    ) if transport_segments else encoded.data
    header = {
        "version": VERSION,
        "mode": "lossy",
        "codec": "dwt",
        "shape": list(source_shape),
        "dtype": "uint8",
        "channels": channels,
        "wavelet": wavelet,
        "wavelet_requested": requested_wavelet,
        "boundary_mode": WAVELET_BOUNDARY_MODE,
        "level": level,
        "step": float(step),
        "quantizer": quantizer,
        "colorspace": colorspace if channels == 3 else "gray",
        "signal_centering": "per-channel-mean-v1",
        "channel_offsets": [round(float(value), 6) for value in channel_offsets],
        "bit_allocation": f"{allocation_method}-subband" if quantizer == "scalar" else "uniform",
        "allocation_method": allocation_method,
        "roi_strength": float(roi_strength),
        "bands": entries,
        "coefficient_count": len(all_values),
        "bit_length": 0 if encoded is None else encoded.bit_length,
        "frequencies": [0] * 256 if encoded is None else encoded.frequencies,
    }
    if not compact_header:
        header["allocation_bands"] = allocation_bands
        header["allocation_reports"] = allocation_reports
    if transport_segments:
        header.update({"payload_layout": "band-segmented-v1",
                       "transport_segment_count": len(entries),
                       "transport_piece_count": int(sum(
                           len(entry.get("transport_tiles", entry.get("transport_regions", [entry])))
                           for entry in entries
                       )),
                       "transport_priority_scope": (
                           "header+roi+spatial-tiles" if transport_tiles and roi_mask is not None else
                           "header+spatial-tiles" if transport_tiles else
                           "header+roi+approximation-bands" if roi_mask is not None else
                           "header+approximation-bands"
                       ),
                       "transport_layout": "spatial-tiled-v1" if transport_tiles else "band-v1",
                       "transport_tile_size": int(transport_tile_size) if transport_tiles else None})
    if roi_mask is not None:
        mask_bytes = np.clip(roi_mask * 255, 0, 255).astype(np.uint8).tobytes()
        header["roi_mask"] = base64.b64encode(zlib.compress(mask_bytes, level=9)).decode("ascii")
    return header, payload


def _encode_lossless_dwt53(image: np.ndarray, requested_level: int,
                            transport_segments: bool = False, transport_tiles: bool = False,
                            transport_tile_size: int = 64) -> tuple[dict, bytes]:
    array = np.asarray(image)
    if array.dtype not in (np.uint8, np.uint16):
        raise ValueError("Lossless mode supports uint8 and uint16 images")
    channels = 1 if array.ndim == 2 else array.shape[2]
    planes = [array] if channels == 1 else [array[..., index] for index in range(channels)]
    maximum = max_level_53(array.shape[:2])
    level = int(requested_level)
    if level < 1 or level > maximum:
        raise ValueError(f"5/3 level {level} is invalid; maximum is {maximum}")
    entries: list[dict] = []
    all_values: list[int] = []
    band_values: list[list[int]] = []
    for channel, plane in enumerate(planes):
        approximation, details = forward_53(plane, level)
        bands = [("approx", level, approximation)]
        for index, detail_tuple in enumerate(reversed(details), start=1):
            detail_level = level - index + 1
            bands.extend(zip(("horizontal", "vertical", "diagonal"), (detail_level,) * 3, detail_tuple))
        for kind, band_level, band in bands:
            values = np.asarray(band, dtype=np.int64)
            entries.append({
                "channel": channel,
                "kind": kind,
                "level": int(band_level),
                "shape": list(values.shape),
                "count": int(values.size),
            })
            flattened = values.reshape(-1).tolist()
            band_values.append(flattened)
            all_values.extend(flattened)
    encoded = None if transport_segments else encode_coefficients(all_values)
    payload = _encode_band_segments(
        band_values, entries, tile_size=transport_tile_size if transport_tiles else None,
        source_shape=tuple(array.shape[:2]),
    ) if transport_segments else encoded.data
    header = {
        "version": VERSION,
        "mode": "lossless",
        "codec": "dwt53",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "channels": channels,
        "wavelet": "reversible-5/3",
        "quantization_applied": False,
        "ai_enabled": False,
        "restoration": False,
        "level": level,
        "bands": entries,
        "coefficient_count": len(all_values),
        "bit_length": 0 if encoded is None else encoded.bit_length,
        "frequencies": [0] * 256 if encoded is None else encoded.frequencies,
    }
    if transport_segments:
        header.update({"payload_layout": "band-segmented-v1",
                       "transport_segment_count": len(entries),
                       "transport_piece_count": int(sum(
                           len(entry.get("transport_tiles", [entry])) for entry in entries
                       )),
                       "transport_priority_scope": "header+spatial-tiles" if transport_tiles else "header+approximation-bands",
                       "transport_layout": "spatial-tiled-v1" if transport_tiles else "band-v1",
                       "transport_tile_size": int(transport_tile_size) if transport_tiles else None})
    return header, payload


def _encode_lossless_raw_zlib(image: np.ndarray) -> tuple[dict, bytes]:
    """Legacy lossless path kept so older SWC files remain decodable."""
    array = np.asarray(image)
    if array.dtype not in (np.uint8, np.uint16):
        raise ValueError("Lossless mode supports uint8 and uint16 images")
    raw = array.tobytes(order="C")
    payload = zlib.compress(raw, level=9)
    header = {
        "version": VERSION,
        "mode": "lossless",
        "codec": "raw-zlib",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "channels": 1 if array.ndim == 2 else array.shape[2],
        "original_bytes": len(raw),
    }
    return header, payload


def _roi_factor(shape: tuple[int, int], roi_mask: np.ndarray | None, roi_strength: float) -> np.ndarray:
    if roi_mask is None:
        return np.ones(shape, dtype=np.float32)
    return np.maximum(0.05, 1.0 - float(np.clip(roi_strength, 0.0, 0.95)) * resize_mask(roi_mask, shape))


def _dct_steps(shape: tuple[int, int], step: float, roi_mask: np.ndarray | None = None, roi_strength: float = 0.65) -> np.ndarray:
    if step <= 0:
        raise ValueError("DCT quantization step must be positive")
    repeats = ((shape[0] + 7) // 8, (shape[1] + 7) // 8)
    base = np.tile(JPEG_LUMA_QUANTIZATION * (float(step) / 8.0), repeats)[: shape[0], : shape[1]]
    return base * _roi_factor(shape, roi_mask, roi_strength)


def _encode_dct(image: np.ndarray, step: float, colorspace: str, roi_mask: np.ndarray | None = None,
                roi_strength: float = 0.65, transport_segments: bool = False,
                transport_tiles: bool = False, transport_tile_size: int = 64) -> tuple[dict, bytes]:
    array = to_float(image)
    channels = 1 if array.ndim == 2 else array.shape[2]
    if channels == 3 and colorspace == "ycbcr":
        array = rgb_to_ycbcr(array)
    elif colorspace not in {"rgb", "ycbcr"}:
        raise ValueError("colorspace must be rgb or ycbcr")
    planes = [array] if channels == 1 else [array[..., index] for index in range(channels)]
    entries: list[dict] = []
    all_values: list[int] = []
    band_values: list[list[int]] = []
    for channel, plane in enumerate(planes):
        coefficients, padded_shape = block_dct(plane)
        steps = _dct_steps(padded_shape, step, roi_mask, roi_strength)
        quantized = np.rint(coefficients / steps).astype(np.int32)
        entries.append({
            "channel": channel,
            "kind": "dct-plane",
            "shape": list(quantized.shape),
            "count": int(quantized.size),
            "padded_shape": list(padded_shape),
        })
        values = quantized.reshape(-1).tolist()
        band_values.append(values)
        all_values.extend(values)
    encoded = None if transport_segments else encode_coefficients(all_values)
    payload = _encode_band_segments(
        band_values, entries, roi_mask,
        tile_size=transport_tile_size if transport_tiles else None,
        source_shape=tuple(array.shape[:2]),
    ) if transport_segments else encoded.data
    header = {
        "version": VERSION,
        "mode": "lossy",
        "codec": "dct",
        "shape": list(image.shape),
        "dtype": "uint8",
        "channels": channels,
        "colorspace": colorspace if channels == 3 else "gray",
        "step": float(step),
        "roi_strength": float(roi_strength),
        "bands": entries,
        "coefficient_count": len(all_values),
        "bit_length": 0 if encoded is None else encoded.bit_length,
        "frequencies": [0] * 256 if encoded is None else encoded.frequencies,
    }
    if transport_segments:
        header.update({"payload_layout": "band-segmented-v1",
                       "transport_segment_count": len(entries),
                       "transport_piece_count": int(sum(
                           len(entry.get("transport_tiles", entry.get("transport_regions", [entry])))
                           for entry in entries
                       )),
                       "transport_priority_scope": (
                           "header+roi+spatial-tiles" if transport_tiles and roi_mask is not None else
                           "header+spatial-tiles" if transport_tiles else
                           "header+roi+band-segments" if roi_mask is not None else
                           "header+band-segments"
                       ),
                       "transport_layout": "spatial-tiled-v1" if transport_tiles else "band-v1",
                       "transport_tile_size": int(transport_tile_size) if transport_tiles else None})
    if roi_mask is not None:
        mask_bytes = np.clip(roi_mask * 255, 0, 255).astype(np.uint8).tobytes()
        header["roi_mask"] = base64.b64encode(zlib.compress(mask_bytes, level=9)).decode("ascii")
    return header, payload


def _prqmf4_steps(shape: tuple[int, int], step: float, roi_mask: np.ndarray | None = None,
                  roi_strength: float = 0.65) -> np.ndarray:
    if step <= 0:
        raise ValueError("PR-QMF quantization step must be positive")
    base = np.array(
        [[0.65, 0.85, 1.0, 1.1], [0.85, 1.0, 1.1, 1.2], [1.0, 1.1, 1.2, 1.3], [1.1, 1.2, 1.3, 1.4]],
        dtype=np.float32,
    )
    steps = np.tile(base * float(step), ((shape[0] + 3) // 4, (shape[1] + 3) // 4))[: shape[0], : shape[1]]
    return steps * _roi_factor(shape, roi_mask, roi_strength)


def _encode_prqmf4(image: np.ndarray, step: float, colorspace: str, roi_mask: np.ndarray | None = None,
                   roi_strength: float = 0.65, transport_segments: bool = False,
                   transport_tiles: bool = False, transport_tile_size: int = 64,
                   level: int = 2, quantizer: str = "uniform",
                   allocation_method: str = "greedy",
                   compact_header: bool = False) -> tuple[dict, bytes]:
    """Encode PR-QMF4 through the real PyWavelets ``bior2.2`` path.

    The old block-matrix payload is still readable by the decoder for backwards
    compatibility, but new files use the same four-subband 2-D wavelet layout
    as the ``qmf`` wavelet option.
    """
    header, payload = _encode_lossy(
        image,
        "qmf",
        level,
        step,
        quantizer,
        roi_mask,
        roi_strength,
        colorspace,
        allocation_method,
        transport_segments,
        transport_tiles,
        int(transport_tile_size),
        compact_header,
    )
    header["codec"] = "prqmf4"
    header["codec_description"] = PRQMF4_DESCRIPTION
    header["qmf_backend"] = "bior2.2"
    return header, payload


def _search_lossy_target(build, original: np.ndarray, target_bpp: float | None,
                         target_psnr: float | None, prepare_header) -> tuple[float, float, float | None, dict, bytes]:
    """Find the quantization step for one rate or quality target.

    The former UI-supplied Step value only searched a fixed range around 12.
    That made low BPP targets unreachable for some colour DWT images and made
    PSNR results jump between a coarse set of values.  Bracket the requested
    target first, then refine the boundary geometrically.  The search uses the
    complete prepared SWC header, so its BPP estimate matches the file that is
    ultimately written.
    """
    if (target_bpp is None) == (target_psnr is None):
        raise ValueError("Exactly one of target_bpp or target_psnr is required")

    minimum_step = 0.01
    maximum_step = 65536.0
    cache: dict[float, tuple[float, float, float | None, dict, bytes]] = {}

    def evaluate(candidate_step: float) -> tuple[float, float, float | None, dict, bytes]:
        candidate_step = float(np.clip(candidate_step, minimum_step, maximum_step))
        key = round(candidate_step, 9)
        if key in cache:
            return cache[key]
        candidate_header, candidate_payload = build(candidate_step)
        quality = None
        if target_psnr is not None:
            candidate_image = _decode_payload(candidate_header, candidate_payload)
            quality = psnr(original, candidate_image)
        prepared_header = prepare_header(candidate_header, candidate_payload, quality)
        estimated_size = _container_size(prepared_header, candidate_payload)
        rate = bits_per_pixel(estimated_size, original.shape)
        result = (candidate_step, rate, quality, prepared_header, candidate_payload)
        cache[key] = result
        return result

    start = evaluate(12.0)
    lower = upper = start

    if target_bpp is not None:
        # BPP normally falls as Step grows. Expand until the target is bracketed.
        if start[1] > target_bpp:
            while upper[0] < maximum_step and upper[1] > target_bpp:
                lower = upper
                upper = evaluate(min(maximum_step, upper[0] * 2.0))
                if upper[0] == lower[0]:
                    break
        else:
            while lower[0] > minimum_step and lower[1] < target_bpp:
                upper = lower
                lower = evaluate(max(minimum_step, lower[0] / 2.0))
                if lower[0] == upper[0]:
                    break
    else:
        assert target_psnr is not None
        # PSNR normally falls as Step grows. Find the largest Step that still
        # satisfies the requested quality before refining that boundary.
        if float(start[2]) >= target_psnr:
            while upper[0] < maximum_step and float(upper[2]) >= target_psnr:
                lower = upper
                upper = evaluate(min(maximum_step, upper[0] * 2.0))
                if upper[0] == lower[0]:
                    break
        else:
            while lower[0] > minimum_step and float(lower[2]) < target_psnr:
                upper = lower
                lower = evaluate(max(minimum_step, lower[0] / 2.0))
                if lower[0] == upper[0]:
                    break

    # Geometric bisection is appropriate because useful quantization steps span
    # several orders of magnitude. Sixteen rounds give a much finer result than
    # the old 25-point global grid without making ordinary searches slower.
    if lower[0] < upper[0]:
        for _ in range(16):
            middle = evaluate(float(np.sqrt(lower[0] * upper[0])))
            if target_bpp is not None:
                if middle[1] > target_bpp:
                    lower = middle
                else:
                    upper = middle
            elif float(middle[2]) >= float(target_psnr):
                lower = middle
            else:
                upper = middle

    scored = list(cache.values())
    if target_bpp is not None:
        return min(scored, key=lambda item: (abs(item[1] - target_bpp), item[0]))
    acceptable = [item for item in scored if float(item[2]) >= float(target_psnr)]
    if acceptable:
        return min(acceptable, key=lambda item: (item[1], abs(float(item[2]) - float(target_psnr))))
    return max(scored, key=lambda item: (float(item[2]), -item[1]))


def encode_array(image: np.ndarray, destination: str | Path, *, mode: str = "lossy",
                 wavelet: str = "haar", level: int = 2, step: float = 12.0,
                 quantizer: str = "uniform", roi_mask: np.ndarray | None = None,
                 roi_strength: float = 0.65, colorspace: str = "ycbcr",
                 codec: str = "dwt", target_bpp: float | None = None,
                 target_psnr: float | None = None, allocation_method: str = "greedy",
                 compact_header: bool = False,
                 metadata: dict | None = None, ai_enabled: bool = False,
                 restoration: bool = False, roi_regions: list | None = None,
                 roi_feather: int = 0, transport_segments: bool = False,
                 transport_tiles: bool = False, transport_tile_size: int = 64) -> dict:
    """Encode an array and return its header plus file statistics."""
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] != 3):
        raise ValueError("Expected grayscale or RGB image")
    if mode not in {"lossy", "lossless"}:
        raise ValueError("mode must be lossy or lossless")
    if quantizer not in {"uniform", "scalar"}:
        raise ValueError("quantizer must be uniform or scalar")
    if allocation_method not in {"greedy", "lagrangian", "dp"}:
        raise ValueError("allocation_method must be greedy, lagrangian or dp")
    if target_bpp is not None and target_psnr is not None:
        raise ValueError("target_bpp and target_psnr are mutually exclusive")
    if transport_tiles and not transport_segments:
        transport_segments = True
    if transport_tiles and int(transport_tile_size) < 8:
        raise ValueError("transport_tile_size must be at least 8")
    if roi_regions is not None:
        generated_mask = regions_to_mask(array.shape[:2], roi_regions)
        roi_mask = generated_mask if roi_mask is None else np.maximum(roi_mask, generated_mask)
        if metadata is None:
            metadata = {}
        metadata = dict(metadata)
        metadata["roi_regions"] = regions_metadata(roi_regions)
    if roi_mask is not None:
        roi_mask = np.asarray(roi_mask, dtype=np.float32)
        if roi_mask.shape != array.shape[:2]:
            raise ValueError("ROI mask must match image height and width")
        if roi_feather:
            roi_mask = feather_mask(roi_mask, int(roi_feather))
    if mode == "lossless":
        if roi_mask is not None or ai_enabled or restoration:
            raise ValueError("ROI, AI and restoration are disabled in lossless mode")
        if target_bpp is not None or target_psnr is not None:
            raise ValueError("Target rate/quality search is only available in lossy mode")
        if codec != "dwt":
            raise ValueError("Lossless mode uses the reversible 5/3 DWT codec")
        header, payload = _encode_lossless_dwt53(
            array, level, transport_segments=transport_segments,
            transport_tiles=transport_tiles, transport_tile_size=int(transport_tile_size),
        )
    else:
        if step <= 0:
            raise ValueError("Quantization step must be positive")
        if target_bpp is not None and target_bpp <= 0:
            raise ValueError("target_bpp must be positive")
        if target_psnr is not None and target_psnr <= 0:
            raise ValueError("target_psnr must be positive")
        if array.dtype == np.uint16:
            array = np.rint(array.astype(np.float32) / 257.0).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
        original_lossy = array.copy()
        def build(candidate_step: float) -> tuple[dict, bytes]:
            if codec == "dwt":
                return _encode_lossy(array, wavelet, level, candidate_step, quantizer, roi_mask, roi_strength,
                                     colorspace, allocation_method, transport_segments,
                                     transport_tiles, int(transport_tile_size),
                                     compact_header=compact_header or target_bpp is not None or target_psnr is not None)
            if codec == "dct":
                return _encode_dct(
                    array, candidate_step, colorspace, roi_mask, roi_strength, transport_segments,
                    transport_tiles, int(transport_tile_size),
                )
            if codec == "prqmf4":
                return _encode_prqmf4(
                    array, candidate_step, colorspace, roi_mask, roi_strength, transport_segments,
                    transport_tiles, int(transport_tile_size), level=level, quantizer=quantizer,
                    allocation_method=allocation_method,
                    compact_header=compact_header or target_bpp is not None or target_psnr is not None,
                )
            raise ValueError("codec must be dwt, dct or prqmf4")
        if target_bpp is not None or target_psnr is not None:
            def prepare_header(candidate_header: dict, candidate_payload: bytes,
                               achieved_psnr: float | None) -> dict:
                prepared = dict(candidate_header)
                prepared["target_bpp"] = target_bpp
                prepared["target_psnr"] = target_psnr
                prepared["achieved_psnr"] = achieved_psnr
                prepared["allocation_method"] = allocation_method
                prepared["source_dtype"] = str(np.asarray(image).dtype)
                prepared["quantization_applied"] = True
                prepared["ai_enabled"] = bool(ai_enabled)
                prepared["restoration"] = bool(restoration)
                if metadata:
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata must be a dictionary")
                    prepared["metadata"] = metadata
                if roi_mask is not None:
                    prepared["roi_source"] = (metadata or {}).get("roi_source", "mask-or-regions")
                prepared.setdefault("created_by", "smartcodec")
                prepared["payload_crc32"] = int(zlib.crc32(candidate_payload) & 0xFFFFFFFF)
                return prepared

            step, _, achieved_psnr, header, payload = _search_lossy_target(
                build, original_lossy, target_bpp, target_psnr, prepare_header,
            )
        else:
            header, payload = build(step)
        header["source_dtype"] = str(np.asarray(image).dtype)
        header["quantization_applied"] = True
        header["ai_enabled"] = bool(ai_enabled)
        header["restoration"] = bool(restoration)
    if metadata:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        header["metadata"] = metadata
    if roi_mask is not None:
        header["roi_source"] = (metadata or {}).get("roi_source", "mask-or-regions")
    header.setdefault("created_by", "smartcodec")
    header["payload_crc32"] = int(zlib.crc32(payload) & 0xFFFFFFFF)
    _write_container(header, payload, destination)
    file_size = Path(destination).stat().st_size
    result = {
        **header,
        "file_size": file_size,
        "compression_ratio": np.asarray(image).nbytes / file_size if file_size else float("inf"),
        "bits_per_pixel": file_size * 8 / (array.shape[0] * array.shape[1]),
    }
    if target_bpp is not None:
        result["target_error"] = result["bits_per_pixel"] - float(target_bpp)
    elif target_psnr is not None:
        result["target_error"] = float(result.get("achieved_psnr", 0.0)) - float(target_psnr)
    return result


def encode_file(source: str | Path, destination: str | Path, **kwargs) -> dict:
    image = load_image(source)
    codec = str(kwargs.get("codec", "dwt")).lower()
    standard_quality = int(kwargs.pop("standard_quality", kwargs.pop("quality", 75)))
    standard_rate = float(kwargs.pop("standard_rate", kwargs.pop("rate", 4.0)))
    if codec in {"jpeg", "jpeg2000"}:
        if kwargs.get("mode", "lossy") != "lossy":
            raise ValueError("Standard JPEG/JPEG2000 output is lossy; use SWC lossless mode instead")
        if any(kwargs.get(name) is not None for name in ("roi_mask", "roi_mask_path", "target_bpp", "target_psnr", "metadata", "roi_regions")):
            raise ValueError("Standard JPEG/JPEG2000 output does not support ROI, target search or SWC metadata")
        if kwargs.get("ai_enabled") or kwargs.get("restoration") or kwargs.get("transport_segments") or kwargs.get("transport_tiles"):
            raise ValueError("Standard JPEG/JPEG2000 output does not support AI restoration or transport options")
        suffix = Path(destination).suffix.lower()
        valid_suffixes = STANDARD_JPEG_SUFFIXES if codec == "jpeg" else STANDARD_JPEG2000_SUFFIXES
        expected = ".jpg" if codec == "jpeg" else ".jp2"
        if suffix not in valid_suffixes:
            raise ValueError(f"{codec} output must use a standard {expected} file extension")
        return encode_standard(
            image,
            destination,
            codec=codec,
            quality=standard_quality,
            rate=standard_rate,
        )
    if kwargs.get("roi_mask_path"):
        kwargs["roi_mask"] = load_mask(kwargs.pop("roi_mask_path"), image.shape[:2])
    return encode_array(image, destination, **kwargs)


def _container_size(header: dict, payload: bytes) -> int:
    header_copy = dict(header)
    header_copy.setdefault("payload_crc32", int(zlib.crc32(payload) & 0xFFFFFFFF))
    return 8 + len(json.dumps(header_copy, separators=(",", ":"), sort_keys=True).encode("utf-8")) + len(payload)


def _decode_coefficient_values(header: dict, payload: bytes) -> list[int]:
    if header.get("payload_layout") == "band-segmented-v1":
        return _decode_band_segments(header, payload)
    encoded = HuffmanEncoded(payload, int(header["bit_length"]), list(header["frequencies"]))
    return decode_coefficients(encoded, int(header["coefficient_count"]))


def _decode_dwt53(header: dict, payload: bytes) -> np.ndarray:
    values = _decode_coefficient_values(header, payload)
    channels = int(header["channels"])
    grouped: list[dict[tuple[str, int], np.ndarray]] = [dict() for _ in range(channels)]
    position = 0
    for entry in header["bands"]:
        count = int(entry["count"])
        shape = tuple(entry["shape"])
        grouped[int(entry["channel"])][(entry["kind"], int(entry["level"]))] = np.asarray(
            values[position : position + count], dtype=np.int64
        ).reshape(shape)
        position += count
    planes = []
    level = int(header["level"])
    shape = tuple(header["shape"][:2])
    for channel in range(channels):
        approx = grouped[channel][("approx", level)]
        details = []
        # inverse_53 expects details in the same fine-to-coarse order as forward_53.
        for detail_level in range(1, level + 1):
            details.append(
                (
                    grouped[channel][("horizontal", detail_level)],
                    grouped[channel][("vertical", detail_level)],
                    grouped[channel][("diagonal", detail_level)],
                )
            )
        planes.append(inverse_53(approx, details, shape))
    reconstructed = planes[0] if channels == 1 else np.stack(planes, axis=-1)
    dtype = np.dtype(header.get("dtype", "uint8"))
    return np.asarray(reconstructed, dtype=dtype)


def _decode_dct(header: dict, payload: bytes) -> np.ndarray:
    values = _decode_coefficient_values(header, payload)
    channels = int(header["channels"])
    planes = []
    position = 0
    roi_mask = _mask_from_header(header)
    for entry in header["bands"]:
        count = int(entry["count"])
        padded_shape = tuple(entry["padded_shape"])
        quantized = np.asarray(values[position : position + count], dtype=np.int32).reshape(padded_shape)
        position += count
        coefficients = quantized * _dct_steps(padded_shape, float(header["step"]), roi_mask, float(header.get("roi_strength", 0.65)))
        planes.append(block_idct(coefficients, tuple(header["shape"][:2])))
    reconstructed = planes[0] if channels == 1 else np.stack(planes, axis=-1)
    if channels == 3 and header.get("colorspace") == "ycbcr":
        reconstructed = ycbcr_to_rgb(reconstructed)
    return from_float(reconstructed)


def _restore_channel_offsets(reconstructed: np.ndarray, header: dict) -> np.ndarray:
    """Restore DC values explicitly preserved by new low-rate DWT files.

    Older SWC files have no ``channel_offsets`` member and therefore retain
    their original decode path byte-for-byte.
    """
    offsets = header.get("channel_offsets")
    if offsets is None:
        return reconstructed
    array = np.asarray(reconstructed, dtype=np.float32)
    values = np.asarray(offsets, dtype=np.float32)
    if array.ndim == 2:
        return array + values[0]
    return array + values.reshape((1, 1, array.shape[2]))


def _decode_prqmf4(header: dict, payload: bytes) -> np.ndarray:
    if header.get("qmf_backend") == "bior2.2":
        values = _decode_coefficient_values(header, payload)
        coefficients = _coefficients_from_entries(header["bands"], values, header)
        channels = int(header["channels"])
        shape = tuple(header["shape"][:2])
        planes = [idwt2(channel_coefficients, header["wavelet"], shape) for channel_coefficients in coefficients]
        reconstructed = planes[0] if channels == 1 else np.stack(planes, axis=-1)
        reconstructed = _restore_channel_offsets(reconstructed, header)
        if channels == 3 and header.get("colorspace") == "ycbcr":
            reconstructed = ycbcr_to_rgb(reconstructed)
        return from_float(reconstructed)

    # Legacy PR-QMF4 files used the educational 4x4 block matrix. Keep this
    # branch so existing SWC artifacts remain decodable after the backend change.
    values = _decode_coefficient_values(header, payload)
    channels = int(header["channels"])
    planes = []
    position = 0
    roi_mask = _mask_from_header(header)
    for entry in header["bands"]:
        count = int(entry["count"])
        padded_shape = tuple(entry["padded_shape"])
        quantized = np.asarray(values[position : position + count], dtype=np.int32).reshape(padded_shape)
        position += count
        coefficients = quantized * _prqmf4_steps(padded_shape, float(header["step"]), roi_mask, float(header.get("roi_strength", 0.65)))
        planes.append(block_iprqmf4(coefficients, tuple(header["shape"][:2])))
    reconstructed = planes[0] if channels == 1 else np.stack(planes, axis=-1)
    if channels == 3 and header.get("colorspace") == "ycbcr":
        reconstructed = ycbcr_to_rgb(reconstructed)
    return from_float(reconstructed)


def _decode_payload(header: dict, payload: bytes) -> np.ndarray:
    # TR: Header'daki codec türü, doğru ters dönüşüm ve renk uzayı yolunu seçer.
    # EN: The header codec selects the matching inverse transform and color path.
    _validate_header(header, len(payload))
    shape = tuple(header["shape"])
    if header["mode"] == "lossless" and header.get("codec") == "raw-zlib":
        raw = zlib.decompress(payload)
        expected = int(np.prod(shape))
        dtype = np.dtype(header.get("dtype", "uint8"))
        if len(raw) != expected * dtype.itemsize:
            raise ValueError("Lossless payload byte size does not match its dtype")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if header["mode"] == "lossless" and header.get("codec") == "dwt53":
        return _decode_dwt53(header, payload)
    if header.get("codec") == "dct":
        return _decode_dct(header, payload)
    if header.get("codec") == "prqmf4":
        return _decode_prqmf4(header, payload)
    values = _decode_coefficient_values(header, payload)
    coefficients = _coefficients_from_entries(header["bands"], values, header)
    channels = int(header["channels"])
    planes = [idwt2(channel_coefficients, header["wavelet"], shape[:2]) for channel_coefficients in coefficients]
    reconstructed = planes[0] if channels == 1 else np.stack(planes, axis=-1)
    reconstructed = _restore_channel_offsets(reconstructed, header)
    if channels == 3 and header.get("colorspace") == "ycbcr":
        reconstructed = ycbcr_to_rgb(reconstructed)
    return from_float(reconstructed)


def decode_array(source: str | Path) -> np.ndarray:
    header, payload = _read_container(source)
    return _decode_payload(header, payload)


def decode_bytes(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode an SWC container already held in memory.

    Transport simulation uses this path for quality measurements without
    creating an intermediate file for every packet-loss condition.
    """
    raw = bytes(data)
    if len(raw) < 8 or raw[:4] != MAGIC:
        raise ValueError("Not a Smart Wavelet Codec byte container")
    header_length = struct.unpack(">I", raw[4:8])[0]
    if header_length > MAX_HEADER_BYTES or 8 + header_length > len(raw):
        raise ValueError("Truncated or oversized SWC byte header")
    try:
        header = json.loads(raw[8 : 8 + header_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid SWC JSON byte header") from exc
    if not isinstance(header, dict):
        raise ValueError("SWC byte header must be an object")
    payload = raw[8 + header_length :]
    checksum = header.get("payload_crc32")
    if checksum is not None and int(checksum) != int(zlib.crc32(payload) & 0xFFFFFFFF):
        raise ValueError("SWC byte payload checksum mismatch")
    _validate_header(header, len(payload))
    return _decode_payload(header, payload)


def decode_file(source: str | Path, destination: str | Path) -> dict:
    suffix = Path(source).suffix.lower()
    if suffix in STANDARD_JPEG_SUFFIXES or suffix in STANDARD_JPEG2000_SUFFIXES:
        image = decode_standard(source)
        save_image(image, destination)
        return {
            "shape": list(image.shape),
            "output": str(destination),
            "format": "jpeg2000" if suffix in STANDARD_JPEG2000_SUFFIXES else "jpeg",
        }
    image = decode_array(source)
    save_image(image, destination)
    return {"shape": list(image.shape), "output": str(destination)}
