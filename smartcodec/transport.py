"""Priority packet transport, live UDP integration, and reproducible 5G/drone simulation.

TR: Gerçek modem gerektirmeden packet-loss, FEC ve canlı UDP davranışını ölçer.
EN: Measures packet loss, FEC, and live UDP behavior without requiring a modem.
"""

from __future__ import annotations

import json
import random
import socket
import struct
import time
import zlib
import csv
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np

from .codec import decode_bytes
from .entropy import encode_coefficients


@dataclass(frozen=True)
class Packet:
    """TR: Bir veri/parity datagramının metadata ve payload kaydı.
    EN: Metadata and payload record for one data or parity datagram.
    """
    sequence: int
    total: int
    checksum: int
    payload: bytes
    frame_id: int = 0
    priority: int = 1
    dependency: tuple[int, ...] = ()
    timestamp: float = 0.0
    payload_length: int = -1
    is_parity: bool = False
    protected_sequences: tuple[int, ...] = ()

    @property
    def packet_id(self) -> int:
        return self.sequence


# The live adapter deliberately uses a small self-describing UDP envelope so a
# 5G modem, drone radio, or ordinary IP link can carry SWC without requiring a
# third-party transport library.  Simulation remains the default path.
UDP_MAGIC = b"SWUDP1"
UDP_HEADER = struct.Struct("!6s16sIIIBBHIIII")
UDP_PARITY_FLAG = 1


def _swc_header_end(data: bytes) -> int:
    """Return the byte boundary of an SWC header, or zero for other payloads."""
    if len(data) < 8 or data[:4] != b"SWC1":
        return 0
    try:
        header_length = struct.unpack(">I", data[4:8])[0]
    except struct.error:
        return 0
    if header_length <= 0 or header_length > 64 * 1024 * 1024:
        return 0
    return min(len(data), 8 + int(header_length))


def _swc_priority_ranges(data: bytes) -> tuple[list[tuple[int, int, int]], str]:
    """Return byte ranges protected by transport priority and their scope label."""
    header_end = _swc_header_end(data)
    if not header_end:
        return [], "none"
    ranges: list[tuple[int, int, int]] = [(0, header_end, 0)]
    scope = "header"
    try:
        header = json.loads(data[8:header_end].decode("utf-8"))
        if header.get("payload_layout") == "band-segmented-v1":
            scope = str(header.get("transport_priority_scope", "header+approximation-bands"))
            for entry in header.get("bands", []):
                regions = entry.get("transport_regions") or entry.get("transport_tiles")
                candidates = regions if regions else [entry]
                for region in candidates:
                    if int(region.get("transport_priority", entry.get("transport_priority", 1))) <= 0:
                        start = header_end + int(region.get("transport_offset", -1))
                        size = int(region.get("transport_size", -1))
                        if start >= header_end and size > 0:
                            ranges.append((start, min(len(data), start + size), 0))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        pass
    return ranges, scope


def _swc_segment_info(data: bytes) -> tuple[dict, int, list[dict]] | None:
    """Read the semantic segment map from a band-segmented SWC payload."""
    header_end = _swc_header_end(data)
    if not header_end:
        return None
    try:
        header = json.loads(data[8:header_end].decode("utf-8"))
        if header.get("payload_layout") != "band-segmented-v1":
            return None
        segments = []
        for index, entry in enumerate(header.get("bands", [])):
            pieces = entry.get("transport_regions") or entry.get("transport_tiles") or [entry]
            for piece in pieces:
                offset = int(piece.get("transport_offset", -1))
                size = int(piece.get("transport_size", -1))
                start = header_end + offset
                end = start + size
                if offset < 0 or size <= 0 or start < header_end or end > len(data):
                    return None
                segments.append({"index": index, "entry": entry, "piece": piece,
                                 "start": start, "end": end})
        return header, header_end, segments
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def _swc_partial_decode_mode(data: bytes) -> str:
    """Name the decoder fallback represented by a segmented payload."""
    info = _swc_segment_info(data)
    if info is not None and info[0].get("transport_layout") == "spatial-tiled-v1":
        return "tile-zero-fill"
    return "band-zero-fill"


def _partial_segmented_swc(data: bytes, received: list[Packet], mtu: int) -> tuple[bytes | None, int]:
    """Zero-fill missing enhancement bands so a segmented SWC remains decodable."""
    info = _swc_segment_info(data)
    if info is None:
        return None, 0
    header, header_end, segments = info
    intervals = sorted(
        (packet.sequence * mtu, min(len(data), packet.sequence * mtu + len(packet.payload)))
        for packet in received
        if packet.sequence >= 0
    )
    def available(start: int, end: int) -> bool:
        cursor = start
        for left, right in intervals:
            if right <= cursor:
                continue
            if left > cursor:
                return False
            cursor = max(cursor, right)
            if cursor >= end:
                return True
        return cursor >= end

    if not available(0, header_end):
        return None, 0
    payload = bytearray()
    zero_filled = 0
    new_entries = []
    for entry_index, original_entry in enumerate(header.get("bands", [])):
        entry = dict(original_entry)
        entry_segments = [segment for segment in segments if segment["index"] == entry_index]
        new_pieces = []
        band_start = len(payload)
        for segment in entry_segments:
            piece = dict(segment["piece"])
            if available(segment["start"], segment["end"]):
                segment_bytes = data[segment["start"] : segment["end"]]
            else:
                count = int(piece.get("transport_count", entry.get("count", 0)))
                zero_encoded = encode_coefficients([0] * count)
                segment_bytes = zero_encoded.data
                piece.update({
                    "transport_bit_length": int(zero_encoded.bit_length),
                    "transport_frequencies": list(zero_encoded.frequencies),
                })
                zero_filled += 1
            piece["transport_offset"] = len(payload)
            piece["transport_size"] = len(segment_bytes)
            payload.extend(segment_bytes)
            if original_entry.get("transport_regions") or original_entry.get("transport_tiles"):
                new_pieces.append(piece)
            else:
                entry.update({
                    "transport_bit_length": piece.get("transport_bit_length", entry.get("transport_bit_length")),
                    "transport_frequencies": piece.get("transport_frequencies", entry.get("transport_frequencies")),
                })
        entry["transport_offset"] = band_start
        entry["transport_size"] = len(payload) - band_start
        if original_entry.get("transport_regions") or original_entry.get("transport_tiles"):
            if original_entry.get("transport_regions"):
                entry["transport_regions"] = new_pieces
            else:
                entry["transport_tiles"] = new_pieces
            entry["transport_priority"] = min(piece.get("transport_priority", 1) for piece in new_pieces)
            entry.pop("transport_bit_length", None)
            entry.pop("transport_frequencies", None)
        new_entries.append(entry)
    header = dict(header)
    header["bands"] = new_entries
    header["transport_partial_decode"] = True
    header["transport_zero_filled_segments"] = zero_filled
    header["payload_crc32"] = int(zlib.crc32(payload) & 0xFFFFFFFF)
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    output = b"SWC1" + struct.pack(">I", len(header_bytes)) + header_bytes + bytes(payload)
    return output, zero_filled


def _stream_id_bytes(stream_id: str | bytes | None) -> tuple[bytes, str]:
    if stream_id is None:
        value = uuid4()
        return value.bytes, str(value)
    if isinstance(stream_id, bytes):
        if len(stream_id) != 16:
            raise ValueError("stream_id bytes must contain exactly 16 bytes")
        value = UUID(bytes=stream_id)
    else:
        try:
            value = UUID(str(stream_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("stream_id must be a UUID string") from exc
    return value.bytes, str(value)


def _encode_udp_datagram(packet: Packet, stream_id: bytes, stream_size: int, chunk_size: int) -> bytes:
    flags = UDP_PARITY_FLAG if packet.is_parity else 0
    payload = bytes(packet.payload)
    header = UDP_HEADER.pack(
        UDP_MAGIC, stream_id, int(packet.sequence), int(packet.total), int(packet.frame_id),
        int(packet.priority), flags, len(payload), int(packet.checksum),
        int(packet.payload_length if packet.payload_length >= 0 else len(payload)),
        int(stream_size), int(chunk_size),
    )
    return header + payload


def _decode_udp_datagram(data: bytes) -> tuple[str, Packet, int, int]:
    if len(data) < UDP_HEADER.size:
        raise ValueError("UDP datagram is shorter than the SWC header")
    magic, stream_bytes, sequence, total, frame_id, priority, flags, payload_length, checksum, declared_length, stream_size, chunk_size = UDP_HEADER.unpack(
        data[:UDP_HEADER.size]
    )
    if magic != UDP_MAGIC:
        raise ValueError("Invalid SWC UDP magic")
    payload = data[UDP_HEADER.size:]
    if payload_length != len(payload) or declared_length != len(payload):
        raise ValueError(f"UDP payload length mismatch for packet {sequence}")
    actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_checksum != checksum:
        raise ValueError(f"Checksum error in UDP packet {sequence}")
    packet = Packet(
        sequence=int(sequence), total=int(total), checksum=int(checksum), payload=payload,
        frame_id=int(frame_id), priority=int(priority), timestamp=time.time(),
        payload_length=int(declared_length), is_parity=bool(flags & UDP_PARITY_FLAG),
    )
    return str(UUID(bytes=stream_bytes)), packet, int(stream_size), int(chunk_size)


def send_udp(data: bytes, host: str, port: int, *, mtu: int = 1200, frame_id: int = 0,
             priority: int = 1, stream_id: str | bytes | None = None,
             prioritize_header: bool = True, fec: bool = False, hardware_test: bool = False) -> dict:
    """Send one SWC/video payload over a live UDP/IP endpoint.

    This is the hardware-independent integration point for a 5G/drone modem:
    the modem supplies the IP route, while the codec owns packet identity,
    priority, payload length and CRC validation.  It intentionally does not
    claim delivery; the receiver report exposes missing packets.
    """
    if not host:
        raise ValueError("UDP destination host is required")
    if not 1 <= int(port) <= 65535:
        raise ValueError("UDP destination port must be between 1 and 65535")
    if mtu < UDP_HEADER.size + 64:
        raise ValueError(f"MTU must leave at least 64 payload bytes after the UDP header ({UDP_HEADER.size})")
    stream_bytes, stream_label = _stream_id_bytes(stream_id)
    chunk_size = int(mtu) - UDP_HEADER.size
    priority_ranges, priority_scope = _swc_priority_ranges(data) if prioritize_header else ([], "disabled")
    packets = packetize(data, chunk_size, frame_id=frame_id, priority=priority,
                        prioritize_header=prioritize_header, priority_ranges=priority_ranges)
    parity_packet = None
    if fec:
        width = max((len(packet.payload) for packet in packets), default=0)
        parity = bytearray(width)
        for packet in packets:
            for index, value in enumerate(packet.payload):
                parity[index] ^= value
        parity_packet = Packet(
            sequence=len(packets), total=len(packets), checksum=zlib.crc32(parity) & 0xFFFFFFFF,
            payload=bytes(parity), frame_id=frame_id, priority=0, timestamp=float(len(packets)),
            payload_length=width, is_parity=True, protected_sequences=tuple(range(len(packets))),
        )
    transmitted = packets + ([parity_packet] if parity_packet is not None else [])
    started = time.perf_counter()
    wire_bytes = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for packet in transmitted:
            datagram = _encode_udp_datagram(packet, stream_bytes, len(data), chunk_size)
            sock.sendto(datagram, (host, int(port)))
            wire_bytes += len(datagram)
    return {
        "transport_mode": "live-udp",
        "network_backend": "udp",
        "simulation": False,
        "hardware_test": bool(hardware_test),
        "destination": f"{host}:{int(port)}",
        "stream_id": stream_label,
        "frame_id": int(frame_id),
        "packets": len(packets),
        "transmitted_packets": len(transmitted),
        "received": None,
        "payload_bytes": len(data),
        "wire_bytes": wire_bytes,
        "mtu": int(mtu),
        "priority_packet_ids": [packet.sequence for packet in transmitted if packet.priority <= 0],
        "priority_scope": priority_scope,
        "priority_range_count": len(priority_ranges),
        "missing_packet_ids": [],
        "checksum_errors": [],
        "partial_decode": False,
        "quality_degradation": None,
        "fec": bool(fec),
        "fec_parity_sent": parity_packet is not None,
        "elapsed_seconds": time.perf_counter() - started,
    }


def receive_udp(bind_host: str, port: int, *, timeout: float = 5.0,
                expected_stream_id: str | None = None, fec: bool = True,
                hardware_test: bool = False, reference: np.ndarray | None = None,
                roi_mask: np.ndarray | None = None) -> tuple[bytes | None, dict]:
    """Receive a UDP stream, with optional decodable partial segmented-SWC output."""
    if not 0.0 < float(timeout):
        raise ValueError("UDP receive timeout must be positive")
    if not 1 <= int(port) <= 65535:
        raise ValueError("UDP listen port must be between 1 and 65535")
    expected = str(UUID(str(expected_stream_id))) if expected_stream_id else None
    started = time.perf_counter()
    received: dict[int, Packet] = {}
    stream_id = expected
    total = 0
    stream_size = 0
    chunk_size = 0
    checksum_errors: list[str] = []
    parity_packet: Packet | None = None
    fec_recovered = False
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, int(port)))
        sock.settimeout(float(timeout))
        while True:
            try:
                datagram, _address = sock.recvfrom(65535)
            except socket.timeout:
                break
            try:
                incoming_stream, packet, incoming_stream_size, incoming_chunk_size = _decode_udp_datagram(datagram)
            except ValueError as exc:
                checksum_errors.append(str(exc))
                continue
            if stream_id is None:
                stream_id = incoming_stream
            if incoming_stream != stream_id or packet.is_parity:
                if incoming_stream != stream_id:
                    continue
                if packet.is_parity:
                    parity_packet = packet
                    total = max(total, packet.total)
                    stream_size = max(stream_size, incoming_stream_size)
                    chunk_size = max(chunk_size, incoming_chunk_size)
                    if total and len(received) >= max(0, total - 1):
                        break
                    continue
            total = max(total, packet.total)
            stream_size = max(stream_size, incoming_stream_size)
            chunk_size = max(chunk_size, incoming_chunk_size)
            received[packet.sequence] = packet
            if total and len(received) >= total:
                break
    missing = sorted(set(range(total)) - set(received))
    if fec and parity_packet is not None and len(missing) == 1 and chunk_size > 0:
        missing_sequence = missing[0]
        recovered = bytearray(parity_packet.payload)
        for packet in received.values():
            for index, value in enumerate(packet.payload):
                recovered[index] ^= value
        if missing_sequence == total - 1 and stream_size:
            expected_length = max(0, stream_size - chunk_size * (total - 1))
        else:
            expected_length = chunk_size
        recovered_payload = bytes(recovered[:expected_length])
        recovered_packet = Packet(
            sequence=missing_sequence, total=total, checksum=zlib.crc32(recovered_payload) & 0xFFFFFFFF,
            payload=recovered_payload, frame_id=parity_packet.frame_id, priority=1,
            payload_length=expected_length,
        )
        received[missing_sequence] = recovered_packet
        missing = sorted(set(range(total)) - set(received))
        fec_recovered = not missing
    ordered = [received[index] for index in sorted(received)]
    payload = None
    partial_decode_mode = "none"
    zero_filled_segment_count = 0
    if not missing and total:
        payload = reassemble(ordered)
    elif missing and ordered and stream_size and chunk_size:
        available_bytes = bytearray(stream_size)
        for packet in ordered:
            start = packet.sequence * chunk_size
            available_bytes[start : start + len(packet.payload)] = packet.payload
        payload, zero_filled_segment_count = _partial_segmented_swc(bytes(available_bytes), ordered, chunk_size)
        if payload is not None:
            partial_decode_mode = _swc_partial_decode_mode(bytes(available_bytes))
    partial_decode = bool(not total or missing or (checksum_errors and not fec_recovered))
    report = {
        "transport_mode": "live-udp",
        "network_backend": "udp",
        "simulation": False,
        "hardware_test": bool(hardware_test),
        "listen_endpoint": f"{bind_host}:{int(port)}",
        "stream_id": stream_id,
        "frame_id": int(ordered[0].frame_id) if ordered else None,
        "priority_packet_ids": [packet.sequence for packet in ordered if packet.priority <= 0],
        "packets": total,
        "transmitted_packets": None,
        "received": len(received),
        "payload_bytes": len(payload) if payload is not None else sum(len(packet.payload) for packet in ordered),
        "missing_packet_ids": missing,
        "checksum_errors": checksum_errors,
        "partial_decode": partial_decode,
        "quality_degradation": 1.0 if not total else float(len(missing) / max(1, total)),
        "timeout_seconds": float(timeout),
        "fec": bool(fec),
        "fec_recovered": fec_recovered,
        "fec_parity_received": parity_packet is not None,
        "partial_decode_mode": partial_decode_mode,
        "zero_filled_segment_count": int(zero_filled_segment_count),
        "elapsed_seconds": time.perf_counter() - started,
    }
    quality_source = bytes(payload) if payload is not None and not partial_decode else b""
    _add_transport_quality_metrics(report, quality_source, payload, reference, roi_mask)
    return payload, report


def send_udp_file(source: str | Path, host: str, port: int, *, mtu: int = 1200,
                  frame_id: int = 0, fec: bool = False, hardware_test: bool = False) -> dict:
    """Send a file and write a sender-side JSON report next to it."""
    source_path = Path(source)
    data = source_path.read_bytes()
    report = send_udp(data, host, port, mtu=mtu, frame_id=frame_id, fec=fec, hardware_test=hardware_test)
    report.update({"source": str(source_path), "source_bytes": len(data)})
    report_path = source_path.with_name(source_path.stem + "_udp_send_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def receive_udp_file(bind_host: str, port: int, destination: str | Path, *, timeout: float = 5.0,
                     expected_stream_id: str | None = None, fec: bool = True,
                     hardware_test: bool = False, reference_path: str | Path | None = None,
                     roi_mask_path: str | Path | None = None) -> dict:
    """Receive a file over UDP, writing complete or semantically decodable partial SWC output."""
    destination_path = Path(destination)
    reference = None
    roi_mask = None
    if reference_path is not None:
        from .image_io import load_image
        reference = load_image(reference_path)
    if roi_mask_path is not None:
        from .roi import load_mask
        if reference is None:
            raise ValueError("roi_mask_path requires reference_path for UDP quality metrics")
        roi_mask = load_mask(roi_mask_path, reference.shape[:2])
    payload, report = receive_udp(bind_host, port, timeout=timeout, expected_stream_id=expected_stream_id, fec=fec,
                                  hardware_test=hardware_test, reference=reference, roi_mask=roi_mask)
    report["output"] = None
    report["output_partial"] = False
    if payload is not None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(payload)
        report["output"] = str(destination_path)
        report["output_partial"] = bool(report.get("partial_decode"))
    report_path = destination_path.with_name(destination_path.stem + "_udp_receive_report.json")
    report["reference"] = str(reference_path) if reference_path else None
    report["roi_mask"] = str(roi_mask_path) if roi_mask_path else None
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report


def packetize(data: bytes, mtu: int = 1200, *, frame_id: int = 0, priority: int = 1,
              dependency: tuple[int, ...] = (), prioritize_header: bool = False,
              priority_ranges: list[tuple[int, int, int]] | None = None) -> list[Packet]:
    # TR: SWC payload'ı MTU parçalarına bölünür; header/ROI önceliği burada korunur.
    # EN: Split the SWC payload into MTU chunks while preserving header/ROI priority.
    if mtu < 64:
        raise ValueError("MTU must be at least 64 bytes")
    chunks = [data[index : index + mtu] for index in range(0, len(data), mtu)] or [b""]
    total = len(chunks)
    ranges = list(priority_ranges or [])
    if prioritize_header and not ranges:
        ranges, _ = _swc_priority_ranges(data)
    def packet_priority(index: int) -> int:
        start = index * mtu
        end = min(len(data), start + mtu)
        if any(start < range_end and end > range_start and level <= 0
               for range_start, range_end, level in ranges):
            return 0
        return priority
    return [Packet(index, total, zlib.crc32(chunk) & 0xFFFFFFFF, chunk, frame_id,
                   packet_priority(index), dependency,
                   float(index), len(chunk)) for index, chunk in enumerate(chunks)]


def drop_packets(packets: list[Packet], loss_rate: float, seed: int = 2026, *, burst_loss: int = 0,
                 preserve_priority: bool = False) -> list[Packet]:
    rng = random.Random(seed)
    rate = np_clip(loss_rate, 0.0, 1.0)
    output = []
    burst_remaining = 0
    for packet in packets:
        if preserve_priority and packet.priority <= 0:
            output.append(packet)
            continue
        if burst_remaining:
            burst_remaining -= 1
            continue
        if rng.random() < rate:
            burst_remaining = max(0, int(burst_loss) - 1)
            continue
        output.append(packet)
    return output


def np_clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _add_transport_quality_metrics(report: dict, original_data: bytes, restored: bytes | None,
                                   reference: np.ndarray | None, roi_mask: np.ndarray | None) -> None:
    """Attach image quality and ROI/background metrics when the payload is SWC."""
    report["quality_metrics_available"] = False
    report["quality_metric_basis"] = None
    if restored is None:
        return
    try:
        candidate = decode_bytes(restored)
        if reference is not None:
            target = np.asarray(reference)
            metric_basis = "reference-image"
        else:
            target = decode_bytes(original_data)
            metric_basis = "complete-decode"
        if target.shape != candidate.shape:
            raise ValueError("reference and transported decode shapes do not match")
        from .metrics import mse, psnr, ssim, region_metrics

        report.update({
            "overall_mse": mse(target, candidate),
            "overall_psnr": psnr(target, candidate),
            "overall_ssim": ssim(target, candidate),
            "quality_metrics_available": True,
            "quality_metric_basis": metric_basis,
        })
        if roi_mask is not None:
            report.update(region_metrics(target, candidate, np.asarray(roi_mask) > 0.05))
            report["roi_metrics_available"] = True
        else:
            report["roi_metrics_available"] = False
    except Exception as exc:
        report["quality_metrics_error"] = str(exc)


def reassemble(packets: list[Packet]) -> bytes:
    if not packets:
        raise ValueError("No packets received")
    total = packets[0].total
    if any(packet.total != total for packet in packets):
        raise ValueError("Packet total mismatch")
    if len(packets) != total or {packet.sequence for packet in packets} != set(range(total)):
        raise ValueError("Packet loss detected; the stream cannot be reassembled")
    ordered = sorted(packets, key=lambda packet: packet.sequence)
    for packet in ordered:
        if zlib.crc32(packet.payload) & 0xFFFFFFFF != packet.checksum:
            raise ValueError(f"Checksum error in packet {packet.sequence}")
    return b"".join(packet.payload for packet in ordered)


def _received_report(packets: list[Packet], received: list[Packet], *, loss_rate: float, latency_ms: float,
                     jitter_ms: float, bandwidth_mbps: float, retransmission: bool, fec: bool, burst_loss: int) -> dict:
    data_packets = [packet for packet in packets if not packet.is_parity]
    data_received = [packet for packet in received if not packet.is_parity]
    missing = sorted(set(range(data_packets[0].total if data_packets else 0)) -
                     {packet.sequence for packet in data_received})
    checksum_errors = [packet.sequence for packet in received if zlib.crc32(packet.payload) & 0xFFFFFFFF != packet.checksum]
    return {
        "simulation": True,
        "hardware_test": False,
        "loss_rate": float(loss_rate),
        "latency_ms": float(latency_ms),
        "jitter_ms": float(jitter_ms),
        "bandwidth_mbps": float(bandwidth_mbps),
        "burst_loss": int(burst_loss),
        "retransmission": bool(retransmission),
        "fec": bool(fec),
        "packets": len(data_packets),
        "transmitted_packets": len(received),
        "received": len(data_received),
        "missing_packet_ids": missing,
        "checksum_errors": checksum_errors,
        "priority_packet_ids": [packet.sequence for packet in data_received if packet.priority <= 0],
        "partial_decode": bool(missing or checksum_errors),
        "quality_degradation": float((len(missing) + len(checksum_errors)) / max(1, len(data_packets))),
    }


def simulate_transport(data: bytes, *, mtu: int = 1200, loss_rate: float = 0.0, latency_ms: float = 0.0,
                       jitter_ms: float = 0.0, bandwidth_mbps: float = 0.0, burst_loss: int = 0,
                       retransmission: bool = False, fec: bool = False, seed: int = 2026,
                       priorities: bool = True, reference: np.ndarray | None = None,
                       roi_mask: np.ndarray | None = None) -> tuple[bytes | None, dict]:
    """Simulate a payload transfer and return partial data plus a JSON report.

    TR: Bu akış kontrollü ağ deneyidir; gerçek 5G sonucu olarak etiketlenmez.
    EN: This is a controlled network experiment; it is never labeled as real 5G.
    """
    priority_ranges, priority_scope = _swc_priority_ranges(data) if priorities else ([], "disabled")
    packets = packetize(data, mtu, prioritize_header=priorities, priority_ranges=priority_ranges)
    parity_packet = None
    if fec:
        width = max((len(packet.payload) for packet in packets), default=0)
        parity = bytearray(width)
        for packet in packets:
            for index, value in enumerate(packet.payload):
                parity[index] ^= value
        parity_packet = Packet(
            sequence=len(packets), total=len(packets), checksum=zlib.crc32(parity) & 0xFFFFFFFF,
            payload=bytes(parity), frame_id=packets[0].frame_id if packets else 0, priority=0,
            dependency=tuple(packet.sequence for packet in packets), timestamp=float(len(packets)),
            payload_length=width, is_parity=True, protected_sequences=tuple(packet.sequence for packet in packets),
        )
    transmitted = packets + ([parity_packet] if parity_packet is not None else [])
    received = drop_packets(transmitted, loss_rate, seed, burst_loss=burst_loss, preserve_priority=priorities)
    received_data = [packet for packet in received if not packet.is_parity]
    received_parity = next((packet for packet in received if packet.is_parity), None)
    recovered_by_fec = False
    if fec and received_parity is not None:
        expected_parity_checksum = zlib.crc32(received_parity.payload) & 0xFFFFFFFF
        missing_ids = sorted(set(range(len(packets))) - {packet.sequence for packet in received_data})
        if expected_parity_checksum == received_parity.checksum and len(missing_ids) == 1:
            missing = missing_ids[0]
            recovered = bytearray(received_parity.payload)
            for packet in received_data:
                for index, value in enumerate(packet.payload):
                    recovered[index] ^= value
            recovered_payload = bytes(recovered[:packets[missing].payload_length])
            original = packets[missing]
            received_data.append(Packet(original.sequence, original.total, zlib.crc32(recovered_payload) & 0xFFFFFFFF,
                                        recovered_payload, original.frame_id, original.priority, original.dependency,
                                        original.timestamp, original.payload_length))
            recovered_by_fec = True
    if retransmission:
        present = {packet.sequence for packet in received_data}
        received_data.extend(packet for packet in packets if packet.sequence not in present)
    report = _received_report(transmitted, received_data + ([received_parity] if received_parity else []), loss_rate=loss_rate, latency_ms=latency_ms,
                              jitter_ms=jitter_ms, bandwidth_mbps=bandwidth_mbps,
                              retransmission=retransmission, fec=fec, burst_loss=burst_loss)
    report["fec_recovered"] = recovered_by_fec
    report["fec_parity_sent"] = parity_packet is not None
    report["fec_parity_received"] = received_parity is not None
    report["priority_scope"] = priority_scope
    report["priority_range_count"] = len(priority_ranges)
    report["partial_decode_mode"] = "none"
    report["zero_filled_segment_count"] = 0
    try:
        restored = reassemble(received_data)
    except ValueError:
        # Keep contiguous data available for a receiver that supports partial
        # decode; the missing packet IDs remain in the report.
        restored = None
        segmented_restored, zero_filled = _partial_segmented_swc(data, received_data, mtu)
        if segmented_restored is not None:
            restored = segmented_restored
            report["partial_decode_mode"] = _swc_partial_decode_mode(data)
            report["zero_filled_segment_count"] = int(zero_filled)
        available = {packet.sequence: packet for packet in received_data}
        if restored is None and available:
            restored = b"".join(available[index].payload for index in sorted(available))
    if bandwidth_mbps > 0:
        report["transmit_seconds"] = len(data) * 8 / (bandwidth_mbps * 1_000_000)
    else:
        report["transmit_seconds"] = 0.0
    report["simulated_delay_seconds"] = max(0.0, latency_ms / 1000.0) + max(0.0, jitter_ms / 2000.0)
    _add_transport_quality_metrics(report, data, restored, reference, roi_mask)
    return restored, report


def run_transport_sweep(source: str | Path, output_dir: str | Path, *,
                        loss_rates: list[float] | tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4),
                        seeds: list[int] | tuple[int, ...] = (2026,), mtu: int = 1200,
                        latency_ms: float = 0.0, jitter_ms: float = 0.0,
                        bandwidth_mbps: float = 0.0, burst_loss: int = 0,
                        retransmission: bool = False, fec: bool = False,
                        priorities: bool = True, reference_path: str | Path | None = None,
                        roi_mask_path: str | Path | None = None) -> dict:
    """Run a reproducible packet-loss sweep and write tabular/graph outputs."""
    source_path = Path(source)
    data = source_path.read_bytes()
    rates = [float(rate) for rate in loss_rates]
    seed_values = [int(seed) for seed in seeds]
    if not rates or not seed_values:
        raise ValueError("transport sweep needs at least one loss rate and seed")
    if any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError("transport loss rates must be between 0 and 1")
    if burst_loss < 0:
        raise ValueError("burst_loss must be non-negative")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reference = None
    roi_mask = None
    if reference_path is not None:
        from .image_io import load_image
        reference = load_image(reference_path)
    if roi_mask_path is not None:
        from .roi import load_mask
        if reference is None:
            raise ValueError("roi_mask_path requires reference_path for transport quality metrics")
        roi_mask = load_mask(roi_mask_path, reference.shape[:2])
    rows: list[dict] = []
    for rate in rates:
        for seed in seed_values:
            restored, report = simulate_transport(
                data, mtu=mtu, loss_rate=rate, latency_ms=latency_ms, jitter_ms=jitter_ms,
                bandwidth_mbps=bandwidth_mbps, burst_loss=burst_loss,
                retransmission=retransmission, fec=fec, seed=seed, priorities=priorities,
                reference=reference, roi_mask=roi_mask,
            )
            rows.append({
                "loss_rate": rate,
                "seed": seed,
                "mtu": int(mtu),
                "fec": bool(fec),
                "retransmission": bool(retransmission),
                "priorities": bool(priorities),
                "packets": int(report["packets"]),
                "received": int(report["received"]),
                "missing_packet_count": len(report["missing_packet_ids"]),
                "checksum_error_count": len(report["checksum_errors"]),
                "priority_packet_count": len(report.get("priority_packet_ids", [])),
                "priority_scope": report.get("priority_scope", "none"),
                "priority_range_count": int(report.get("priority_range_count", 0)),
                "quality_degradation": float(report["quality_degradation"]),
                "partial_decode": bool(report["partial_decode"]),
                "partial_decode_mode": report.get("partial_decode_mode", "none"),
                "zero_filled_segment_count": int(report.get("zero_filled_segment_count", 0)),
                "fec_recovered": bool(report.get("fec_recovered", False)),
                "restored_bytes": int(len(restored)) if restored is not None else 0,
                "exact_recovery": bool(restored == data),
                "transmit_seconds": float(report.get("transmit_seconds", 0.0)),
                "simulated_delay_seconds": float(report.get("simulated_delay_seconds", 0.0)),
                "quality_metrics_available": bool(report.get("quality_metrics_available", False)),
                "quality_metric_basis": report.get("quality_metric_basis"),
                "overall_psnr": report.get("overall_psnr"),
                "overall_ssim": report.get("overall_ssim"),
                "roi_psnr": report.get("roi_psnr"),
                "roi_ssim": report.get("roi_ssim"),
                "background_psnr": report.get("background_psnr"),
                "background_ssim": report.get("background_ssim"),
            })
    config = {
        "source": str(source_path.resolve()),
        "source_bytes": len(data),
        "loss_rates": rates,
        "seeds": seed_values,
        "mtu": int(mtu),
        "latency_ms": float(latency_ms),
        "jitter_ms": float(jitter_ms),
        "bandwidth_mbps": float(bandwidth_mbps),
        "burst_loss": int(burst_loss),
        "retransmission": bool(retransmission),
        "fec": bool(fec),
        "priorities": bool(priorities),
        "simulation": True,
        "hardware_test": False,
        "reference_path": str(Path(reference_path).resolve()) if reference_path else None,
        "roi_mask_path": str(Path(roi_mask_path).resolve()) if roi_mask_path else None,
    }
    (destination / "transport_sweep_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (destination / "transport_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (destination / "transport_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "rows": len(rows),
        "exact_recovery_count": sum(row["exact_recovery"] for row in rows),
        "partial_decode_count": sum(row["partial_decode"] for row in rows),
        "mean_quality_degradation": float(sum(row["quality_degradation"] for row in rows) / len(rows)),
        "simulation": True,
        "hardware_test": False,
    }
    plot_path = destination / "transport_sweep.png"
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for label, group in _group_transport_rows(rows):
            ordered = sorted(group, key=lambda row: row["loss_rate"])
            plt.plot([row["loss_rate"] for row in ordered],
                     [row["quality_degradation"] for row in ordered], marker="o", label=label)
        plt.xlabel("Packet loss rate")
        plt.ylabel("Quality degradation proxy")
        plt.title("Transport packet-loss sweep")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        summary["plot"] = str(plot_path)
    except Exception as exc:
        summary["plot_error"] = str(exc)
        summary["plot"] = None
    (destination / "transport_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"output_dir": str(destination), "rows": len(rows), "summary": summary,
            "config": str(destination / "transport_sweep_config.json"),
            "csv": str(destination / "transport_sweep.csv"),
            "json": str(destination / "transport_sweep.json"),
            "plot": summary.get("plot")}


def _group_transport_rows(rows: list[dict]):
    groups: dict[str, list[dict]] = {}
    for row in rows:
        label = f"fec={row['fec']}, retransmission={row['retransmission']}, priority={row['priorities']}"
        groups.setdefault(label, []).append(row)
    return groups.items()


def simulate_file(source: str | Path, destination: str | Path, loss_rate: float = 0.0, mtu: int = 1200,
                   seed: int = 2026, *, latency_ms: float = 0.0, jitter_ms: float = 0.0,
                   bandwidth_mbps: float = 0.0, burst_loss: int = 0, retransmission: bool = False,
                   fec: bool = False, reference_path: str | Path | None = None,
                   roi_mask_path: str | Path | None = None) -> dict:
    started = time.perf_counter()
    data = Path(source).read_bytes()
    reference = None
    roi_mask = None
    if reference_path is not None:
        from .image_io import load_image
        reference = load_image(reference_path)
    if roi_mask_path is not None:
        from .roi import load_mask
        if reference is None:
            raise ValueError("roi_mask_path requires reference_path for transport quality metrics")
        roi_mask = load_mask(roi_mask_path, reference.shape[:2])
    restored, report = simulate_transport(data, mtu=mtu, loss_rate=loss_rate, latency_ms=latency_ms,
                                           jitter_ms=jitter_ms, bandwidth_mbps=bandwidth_mbps,
                                           burst_loss=burst_loss, retransmission=retransmission, fec=fec, seed=seed,
                                           reference=reference, roi_mask=roi_mask)
    if restored is not None:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(restored)
        report["output"] = str(destination)
    else:
        report["output"] = None
    report.update({"source": str(source), "source_bytes": len(data), "elapsed_seconds": time.perf_counter() - started,
                   "reference": str(reference_path) if reference_path else None,
                   "roi_mask": str(roi_mask_path) if roi_mask_path else None})
    report_path = Path(destination).with_name("transport_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report
