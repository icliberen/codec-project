"""Deterministic external-dataset discovery and manifest utilities.

The project deliberately does not ship third-party datasets.  This module
turns a user supplied file/folder into an auditable manifest instead, while
keeping unreadable files in the report so a benchmark never silently drops
evidence.

TR: Manifest, dosya güvenliği ve hash doğrulaması burada merkezileştirilir.
EN: Manifest creation, file safety, and hash verification are centralized here.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
CATEGORIES = {"natural", "synthetic", "hybrid", "fingerprint", "biomedical", "letters_logos"}


@dataclass(frozen=True)
class DatasetRecord:
    path: str
    relative_path: str
    category: str
    width: int | None
    height: int | None
    channels: int | None
    dtype: str | None
    bit_depth: int | None
    source: str
    readable: bool
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _paths(root: str | Path, recursive: bool = True) -> tuple[Path, list[Path]]:
    supplied = Path(root)
    if supplied.is_file():
        return supplied.parent, [supplied]
    if not supplied.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {supplied}")
    iterator = supplied.rglob("*") if recursive else supplied.glob("*")
    return supplied, sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _category(path: Path, root: Path) -> str:
    try:
        first = path.relative_to(root).parts[0]
    except (ValueError, IndexError):
        return "uncategorized"
    return first if first in CATEGORIES else "uncategorized"


def _image_properties(path: Path) -> tuple[int, int, int, str, int]:
    with Image.open(path) as image:
        width, height = image.size
        channels = len(image.getbands())
        mode = image.mode
        bit_depth = 16 if mode in {"I;16", "I;16B", "I;16L", "I"} else 8
        dtype = "uint16" if bit_depth > 8 else "uint8"
        return width, height, channels, dtype, bit_depth


def scan_dataset(root: str | Path, *, recursive: bool = True, source: str | None = None) -> list[dict]:
    """Scan one image, a folder, or a recursive folder into manifest rows."""
    base, paths = _paths(root, recursive)
    source_label = source or ("external" if Path(root).exists() else "unknown")
    records: list[dict] = []
    for path in paths:
        relative = path.name if Path(root).is_file() else str(path.relative_to(base))
        try:
            width, height, channels, dtype, bit_depth = _image_properties(path)
            record = DatasetRecord(str(path.resolve()), relative, _category(path, base), width, height,
                                   channels, dtype, bit_depth, source_label, True)
        except Exception as exc:  # keep bad files visible and continue
            record = DatasetRecord(str(path.resolve()), relative, _category(path, base), None, None, None,
                                   None, None, source_label, False, f"{type(exc).__name__}: {exc}")
        records.append(record.as_dict())
    return records


def write_manifest(records: Iterable[dict], destination: str | Path, **metadata) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "records": list(records), **metadata}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("Dataset manifest must contain a records list")
    return payload


def verify_manifest(dataset_root: str | Path, *, check_hash: bool = True) -> dict:
    """Verify manifest paths, image metadata and optional SHA-256 records."""
    supplied = Path(dataset_root)
    manifest_path = supplied / "dataset_manifest.json" if supplied.is_dir() else supplied
    manifest = load_manifest(manifest_path)
    root = manifest_path.parent.resolve()
    records = manifest.get("records", [])
    missing: list[str] = []
    unsafe_paths: list[dict] = []
    unreadable: list[dict] = []
    hash_mismatches: list[dict] = []
    metadata_mismatches: list[dict] = []
    hash_checked = 0
    split_counts: dict[str, int] = {}

    for index, record in enumerate(records):
        split = str(record.get("split", "unspecified"))
        split_counts[split] = split_counts.get(split, 0) + 1
        raw_path = record.get("path")
        if not raw_path:
            missing.append(f"record[{index}]")
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        path = candidate.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            unsafe_paths.append({"path": str(candidate), "reason": "outside-manifest-root"})
            continue
        if candidate.is_symlink():
            unsafe_paths.append({"path": str(candidate), "reason": "symlink-not-allowed"})
            continue
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                image.load()
                actual = {"width": image.width, "height": image.height, "channels": len(image.getbands())}
        except Exception as exc:
            unreadable.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        expected = {key: record.get(key) for key in ("width", "height", "channels")}
        differences = {key: {"expected": expected[key], "actual": actual[key]}
                       for key in expected if expected[key] is not None and expected[key] != actual[key]}
        if differences:
            metadata_mismatches.append({"path": str(path), "differences": differences})
        expected_hash = record.get("sha256")
        if check_hash and expected_hash:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hash_checked += 1
            actual_hash = digest.hexdigest()
            if actual_hash.lower() != str(expected_hash).lower():
                hash_mismatches.append({"path": str(path), "expected": expected_hash, "actual": actual_hash})

    problems = len(missing) + len(unsafe_paths) + len(unreadable) + len(hash_mismatches) + len(metadata_mismatches)
    return {
        "status": "passed" if problems == 0 else "failed",
        "dataset": manifest.get("dataset"), "manifest": str(manifest_path.resolve()),
        "records": len(records), "split_counts": split_counts,
        "missing_count": len(missing), "unsafe_path_count": len(unsafe_paths),
        "unreadable_count": len(unreadable),
        "metadata_mismatch_count": len(metadata_mismatches), "hash_checked": hash_checked,
        "hash_mismatch_count": len(hash_mismatches), "hash_check_enabled": bool(check_hash),
        "missing": missing, "unsafe_paths": unsafe_paths, "unreadable": unreadable,
        "metadata_mismatches": metadata_mismatches, "hash_mismatches": hash_mismatches,
    }


def split_dataset(records: list[dict], *, seed: int = 2026, train: float = 0.8,
                  validation: float = 0.1, test: float = 0.1) -> dict[str, list[dict]]:
    """Split readable records reproducibly without changing the input list."""
    if any(value < 0 for value in (train, validation, test)) or not np.isclose(train + validation + test, 1.0):
        raise ValueError("train, validation and test fractions must be non-negative and sum to 1")
    readable = [dict(row) for row in records if row.get("readable", False)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(readable))
    shuffled = [readable[index] for index in order]
    train_end = int(len(shuffled) * train)
    validation_end = train_end + int(len(shuffled) * validation)
    return {"train": shuffled[:train_end], "validation": shuffled[train_end:validation_end], "test": shuffled[validation_end:]}


def manifest_for_dataset(root: str | Path, destination: str | Path | None = None, *, source: str | None = None,
                         recursive: bool = True) -> dict:
    supplied = Path(root)
    existing = supplied / "dataset_manifest.json" if supplied.is_dir() else None
    if existing is not None and existing.exists() and destination is None:
        payload = load_manifest(existing)
        payload.setdefault("root", str(supplied.resolve()))
        if source is not None:
            payload["source"] = source
        return payload
    records = scan_dataset(root, recursive=recursive, source=source)
    payload = {"version": 1, "root": str(Path(root).resolve()), "records": records}
    if destination is not None:
        write_manifest(records, destination, root=payload["root"], source=source or "external")
    return payload
