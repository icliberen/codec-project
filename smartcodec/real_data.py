"""Download and prepare a real public image dataset for reproducible training.

TR: Arşiv güvenliği, manifest, hash ve split bilgisi tek veri hazırlama katmanındadır.
EN: Archive safety, manifests, hashes, and splits live in one data-preparation layer.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .ai import DEFAULT_CANDIDATES, WaveletParameterEstimator
from .codec import decode_array, encode_array
from .dataset import write_manifest
from .metrics import bits_per_pixel, psnr


CIFAR10_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
KODAK_PAGE = "https://r0k.us/graphics/kodak/index.html"
KODAK_URL_TEMPLATE = "https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992/{index:02d}.png"
DIV2K_URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "validation": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_cifar10(destination: str | Path, *, force: bool = False) -> Path:
    """Download the official Python archive and verify its published MD5."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if force or not archive.exists():
        urllib.request.urlretrieve(CIFAR10_URL, archive)
    checksum = _md5(archive)
    if checksum != CIFAR10_MD5:
        raise ValueError(f"CIFAR-10 MD5 mismatch: expected {CIFAR10_MD5}, got {checksum}")
    return archive


def _load_batch(path: Path) -> tuple[np.ndarray, list[int]]:
    with path.open("rb") as stream:
        payload = pickle.load(stream, encoding="bytes")
    data = np.asarray(payload[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    return np.transpose(data, (0, 2, 3, 1)), [int(value) for value in payload[b"labels"]]


def prepare_cifar10(destination: str | Path, *, train_limit: int | None = None,
                    test_limit: int | None = None, validation_limit: int | None = None,
                    seed: int = 2026) -> dict:
    """Extract CIFAR-10 and preserve a canonical full manifest.

    Optional limits produce ``dataset_manifest.sample.json``; they never
    replace the complete 60,000-record canonical manifest.
    """
    for name, value in (("train_limit", train_limit), ("test_limit", test_limit),
                        ("validation_limit", validation_limit)):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive when provided")
    root = Path(destination)
    archive = download_cifar10(root / "raw")
    extract_root = root / "raw" / "cifar-10-batches-py"
    if not extract_root.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(root / "raw", filter="data")
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    sample_records = []
    for split, batches, limit in (
        ("train", [extract_root / f"data_batch_{index}" for index in range(1, 6)], train_limit),
        ("test", [extract_root / "test_batch"], test_limit),
    ):
        arrays = []
        labels: list[int] = []
        for batch in batches:
            data, batch_labels = _load_batch(batch)
            arrays.append(data)
            labels.extend(batch_labels)
        images = np.concatenate(arrays, axis=0)
        order = rng.permutation(len(images))
        full_split_orders = [(split, order)]
        if split == "train":
            requested_validation = 5000
            validation_count = min(max(0, int(requested_validation)), max(0, len(order) - 1))
            if validation_count:
                full_split_orders = [("train", order[:-validation_count]), ("validation", order[-validation_count:])]
        source_records: dict[int, dict] = {}
        for output_split, split_order in full_split_orders:
            for output_index, source_index in enumerate(split_order):
                label = labels[int(source_index)]
                category = CIFAR10_CLASSES[label]
                path = image_root / output_split / category / f"{output_index:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(images[int(source_index)]).save(path)
                record = {
                    "path": str(path.resolve()), "relative_path": str(path.relative_to(image_root)),
                    "category": category, "split": output_split, "label": label, "width": 32, "height": 32,
                    "channels": 3, "dtype": "uint8", "bit_depth": 8, "source": "CIFAR-10-real",
                    "sha256": _sha256(path), "readable": True, "error": None,
                }
                records.append(record)
                source_records[int(source_index)] = record
        sample_split_orders = []
        if split == "train" and (train_limit is not None or validation_limit is not None):
            sample_order = order[: min(int(train_limit) if train_limit is not None else len(order), len(order))]
            requested_validation = validation_limit if validation_limit is not None else 0
            validation_count = min(max(0, int(requested_validation)), max(0, len(sample_order) - 1))
            sample_split_orders = [("train", sample_order)]
            if validation_count:
                sample_split_orders = [("train", sample_order[:-validation_count]),
                                       ("validation", sample_order[-validation_count:])]
        elif split == "test" and test_limit is not None:
            sample_split_orders = [("test", order[: min(int(test_limit), len(order))])]
        for output_split, split_order in sample_split_orders:
            for sample_index, source_index in enumerate(split_order):
                sample_record = dict(source_records[int(source_index)])
                sample_record["split"] = output_split
                sample_record["sample_index"] = sample_index
                sample_records.append(sample_record)
    manifest = {
        "version": 1, "dataset": "CIFAR-10", "source_url": CIFAR10_URL,
        "archive_md5": CIFAR10_MD5, "seed": seed, "records": records,
        "source_note": "Official CIFAR-10 Python archive; cite Krizhevsky 2009.",
    }
    manifest_path = root / "dataset_manifest.json"
    if train_limit is not None or test_limit is not None or validation_limit is not None:
        sample_path = root / "dataset_manifest.sample.json"
        sample_payload = dict(manifest)
        sample_payload.update({
            "records": sample_records,
            "sample_limits": {"train": train_limit, "validation": validation_limit, "test": test_limit},
            "parent_manifest": str(manifest_path.resolve()),
        })
        sample_path.write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")
        manifest["sample_manifest"] = str(sample_path.resolve())
        manifest["sample_records"] = len(sample_records)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_kodak(destination: str | Path, *, seed: int = 2026) -> dict:
    """Download the 24-image Kodak compression reference suite."""
    root = Path(destination)
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(24)
    records = []
    for position, image_number in enumerate(order):
        split = "train" if position < 16 else "validation" if position < 20 else "test"
        path = image_root / split / f"kodim{int(image_number) + 1:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            urllib.request.urlretrieve(KODAK_URL_TEMPLATE.format(index=int(image_number) + 1), path)
        with Image.open(path) as image:
            width, height = image.size
            channels = len(image.getbands())
        records.append({"path": str(path.resolve()), "relative_path": str(path.relative_to(image_root)),
                        "category": "natural", "split": split, "label": None, "width": width,
                        "height": height, "channels": channels, "dtype": "uint8", "bit_depth": 8,
                        "source": "Kodak-real", "sha256": _sha256(path), "readable": True, "error": None})
    manifest = {"version": 1, "dataset": "Kodak Lossless True Color Image Suite",
                "source_url": KODAK_PAGE, "image_urls": [KODAK_URL_TEMPLATE.format(index=index) for index in range(1, 25)],
                "seed": seed, "records": records,
                "source_note": "Reference suite page describes 24-bit PNGs and its stated usage terms; verify local licensing before redistribution."}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_div2k(destination: str | Path, *, force: bool = False) -> dict[str, Path]:
    """Download the official DIV2K high-resolution train/validation archives.

    The archive SHA-256 values are recorded after download because the upstream
    distribution does not publish a stable checksum alongside the files.
    Individual image SHA-256 values are added to the prepared dataset manifest.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    archives: dict[str, Path] = {}
    archive_records = {}
    for split, url in DIV2K_URLS.items():
        archive = root / f"DIV2K_{'valid' if split == 'validation' else split}_HR.zip"
        if force or not archive.exists():
            urllib.request.urlretrieve(url, archive)
        if not archive.is_file() or archive.stat().st_size == 0:
            raise ValueError(f"DIV2K {split} archive is missing or empty: {archive}")
        with zipfile.ZipFile(archive) as checked:
            if not any(name.lower().endswith(".png") for name in checked.namelist()):
                raise ValueError(f"DIV2K {split} archive contains no PNG images: {archive}")
        archives[split] = archive
        archive_records[split] = {
            "url": url, "path": str(archive.resolve()), "bytes": archive.stat().st_size,
            "sha256": _sha256(archive),
        }
    (root / "archive_manifest.json").write_text(
        json.dumps({"dataset": "DIV2K", "archives": archive_records}, indent=2), encoding="utf-8"
    )
    return archives


def _extract_div2k_archive(archive: Path, output_dir: Path) -> list[Path]:
    """Safely extract only image files from one DIV2K archive."""
    # TR: Arşiv üyeleri hedef kök dışına çıkmadan önce path-traversal açısından kontrol edilir.
    # EN: Archive members are checked for path traversal before leaving the target root.
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as checked:
        for member in checked.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".png"):
                continue
            # DIV2K archives contain one flat numeric filename per image.  Do
            # not trust archive paths: flatten after validating the filename.
            normalized_name = member.filename.replace("\\", "/")
            member_path = PurePosixPath(normalized_name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe DIV2K archive member: {member.filename}")
            # ZIP symlink entries must never be followed during extraction.
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"Symlink DIV2K archive member is not allowed: {member.filename}")
            filename = member_path.name
            if not filename[:-4].isdigit() or Path(filename).suffix.lower() != ".png":
                raise ValueError(f"Unexpected DIV2K image member: {member.filename}")
            target = (output_dir / filename).resolve()
            try:
                target.relative_to(output_dir.resolve())
            except ValueError as exc:
                raise ValueError(f"Unsafe DIV2K archive member: {member.filename}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with checked.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(target)
    return sorted(set(extracted))


def prepare_div2k(destination: str | Path, *, train_limit: int | None = None,
                  validation_limit: int | None = None, seed: int = 2026,
                  force: bool = False) -> dict:
    """Prepare DIV2K HR images as a manifest-backed real dataset.

    By default all 800 train and 100 validation images are represented. Limits
    are optional quick-sample selectors: the canonical ``dataset_manifest.json``
    remains complete, while ``dataset_manifest.sample.json`` contains the
    selected subset. The downloaded archives remain intact for later runs.
    """
    root = Path(destination)
    archives = download_div2k(root / "raw", force=force)
    rng = np.random.default_rng(seed)
    records = []
    sample_records = []
    split_limits = {"train": train_limit, "validation": validation_limit}
    for split, archive in archives.items():
        extracted_root = root / "raw" / f"DIV2K_{'valid' if split == 'validation' else split}_HR"
        paths = sorted(extracted_root.glob("*.png"))
        if not paths or force:
            paths = _extract_div2k_archive(archive, extracted_root)
        if not paths:
            raise ValueError(f"No DIV2K {split} images were extracted from {archive}")
        order = rng.permutation(len(paths))
        limit = split_limits[split]
        if limit is not None and limit < 1:
            raise ValueError("DIV2K train/validation limits must be positive when provided")
        selected_count = len(order) if limit is None else min(int(limit), len(order))
        for output_index, source_index in enumerate(order):
            path = paths[int(source_index)]
            with Image.open(path) as image:
                width, height = image.size
                channels = len(image.getbands())
            record = {
                "path": str(path.resolve()), "relative_path": str(path.resolve().relative_to(root.resolve())),
                "category": "natural", "split": split, "label": None,
                "width": width, "height": height, "channels": channels,
                "dtype": "uint8", "bit_depth": 8, "source": "DIV2K-real",
                "sha256": _sha256(path), "readable": True, "error": None,
                "sample_index": output_index,
            }
            records.append(record)
            if output_index < selected_count:
                sample_records.append(record)
    manifest = {
        "version": 1, "dataset": "DIV2K high-resolution image dataset",
        "source_urls": DIV2K_URLS, "seed": seed, "records": records,
        "archives": {
            split: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for split, path in archives.items()
        },
        "source_note": "DIV2K HR train/validation archives; verify upstream license terms before redistribution.",
    }
    manifest_path = root / "dataset_manifest.json"
    if train_limit is not None or validation_limit is not None:
        sample_path = root / "dataset_manifest.sample.json"
        sample_payload = dict(manifest)
        sample_payload.update({
            "records": sample_records,
            "sample_limits": {"train": train_limit, "validation": validation_limit},
            "parent_manifest": str(manifest_path.resolve()),
        })
        sample_path.write_text(json.dumps(sample_payload, indent=2), encoding="utf-8")
        manifest["sample_manifest"] = str(sample_path.resolve())
        manifest["sample_records"] = len(sample_records)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _best_candidate(image: np.ndarray, candidates: tuple[dict, ...], scratch: Path) -> tuple[int, dict]:
    measurements = []
    for index, candidate in enumerate(candidates):
        path = scratch / f"candidate_{index}.swc"
        info = encode_array(image, path, mode="lossy", codec="dwt", wavelet=candidate["wavelet"],
                            level=1, step=float(candidate["step"]), quantizer=candidate["quantizer"])
        reconstructed = decode_array(path)
        measurements.append({"index": index, "psnr": psnr(image, reconstructed),
                             "bpp": bits_per_pixel(info["file_size"], image.shape)})
    # Explicitly optimize a quality-rate objective; this is a label for the
    # estimator, not a claim that the candidate is globally optimal.
    selected = max(measurements, key=lambda item: item["psnr"] - 0.35 * item["bpp"])
    return int(selected["index"]), {"measurements": measurements, "selected": selected}


def train_real_estimator(dataset_root: str | Path, model_path: str | Path, *, train_limit: int = 200,
                         validation_limit: int = 40, test_limit: int = 40, seed: int = 2026) -> dict:
    """Train the estimator on manifest-backed real images and evaluate choices."""
    root = Path(dataset_root)
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.exists():
        prepare_cifar10(root, train_limit=max(train_limit, validation_limit),
                        validation_limit=validation_limit, test_limit=validation_limit + test_limit, seed=seed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_records = [row for row in manifest["records"] if row.get("split") == "train"][:train_limit]
    explicit_validation = [row for row in manifest["records"] if row.get("split") == "validation"]
    explicit_test = [row for row in manifest["records"] if row.get("split") == "test"]
    validation_records = (explicit_validation[:validation_limit] if explicit_validation else explicit_test[:validation_limit])
    test_records = (explicit_test[:test_limit] if explicit_validation else explicit_test[validation_limit:validation_limit + test_limit])
    if not train_records or not validation_records or not test_records:
        raise ValueError("CIFAR-10 manifest does not contain enough train/validation/test images")
    candidates = tuple(DEFAULT_CANDIDATES)
    train_images = [np.asarray(Image.open(row["path"]).convert("RGB"), dtype=np.uint8) for row in train_records]
    scratch = Path(tempfile.mkdtemp(prefix="smartcodec-cifar-labels-", dir=str(root)))
    try:
        labels = []
        label_measurements = []
        for image in train_images:
            label, details = _best_candidate(image, candidates, scratch)
            labels.append(label)
            label_measurements.append(details)
    finally:
        for path in scratch.glob("*"):
            path.unlink(missing_ok=True)
        scratch.rmdir()
    estimator = WaveletParameterEstimator(candidates).fit(train_images, labels)
    saved = estimator.save(model_path)
    evaluation = {}
    for split_name, records in (("validation", validation_records), ("test", test_records)):
        correct = 0
        selected_scores = []
        eval_scratch = Path(tempfile.mkdtemp(prefix=f"smartcodec-cifar-{split_name}-", dir=str(root)))
        try:
            for row in records:
                image = np.asarray(Image.open(row["path"]).convert("RGB"), dtype=np.uint8)
                predicted, confidence = estimator.predict(image)
                actual_index, details = _best_candidate(image, candidates, eval_scratch)
                actual = candidates[actual_index]
                correct += int(predicted == actual)
                predicted_measurement = next(item for item in details["measurements"] if item["index"] == candidates.index(predicted))
                selected_scores.append({"predicted": predicted, "actual": actual, "confidence": confidence,
                                        "predicted_psnr": predicted_measurement["psnr"], "predicted_bpp": predicted_measurement["bpp"]})
        finally:
            for path in eval_scratch.glob("*"):
                path.unlink(missing_ok=True)
            eval_scratch.rmdir()
        evaluation[split_name] = {"count": len(records), "candidate_accuracy": correct / len(records),
                                  "mean_psnr": float(np.mean([item["predicted_psnr"] for item in selected_scores])),
                                  "mean_bpp": float(np.mean([item["predicted_bpp"] for item in selected_scores])),
                                  "samples": selected_scores}
    dataset_name = manifest.get("dataset", "real-image-dataset")
    report = {"dataset": dataset_name, "model": str(saved), "model_type": "prototype-estimator-trained-on-real-images",
              "candidate_count": len(candidates), "train_count": len(train_records),
              "validation_count": len(validation_records), "test_count": len(test_records), "seed": seed,
              "validation": evaluation["validation"], "test": evaluation["test"],
              "source_url": manifest.get("source_url"), "archive_md5": manifest.get("archive_md5"),
              "source_note": manifest.get("source_note"),
              "record_sha256": {split: [row.get("sha256") for row in records if row.get("sha256")]
                                for split, records in (("train", train_records), ("validation", validation_records), ("test", test_records))}}
    report_path = Path(model_path).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
