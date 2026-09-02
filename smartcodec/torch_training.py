"""Optional PyTorch training for a small decoder-side restoration baseline.

TR: Eğitim ve değerlendirme akışı codec restoration sonuçlarını ayrı raporlar.
EN: Training and evaluation report codec-restoration results separately.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from .codec import decode_array, encode_array, encode_file
from .metrics import bits_per_pixel, mse, psnr, ssim


def _trace_torchscript(torch, model, example):
    """Trace legacy .pt artifacts while silencing Python 3.14 API notices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return torch.jit.trace(model, example)


def _load_records(root: Path, split: str, limit: int | None) -> list[dict]:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    records = [row for row in manifest.get("records", []) if row.get("split") == split and row.get("readable")]
    if not records:
        raise ValueError(f"No readable {split} records in dataset manifest")
    return records if limit is None else records[:limit]


def _make_residual_denoiser(torch, nn):
    class ResidualDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 3, 3, padding=1),
            )

        def forward(self, x):
            return torch.clamp(x + 0.5 * self.body(x), 0.0, 1.0)

    return ResidualDenoiser()


def train_restoration(dataset_root: str | Path, model_path: str | Path, *, train_limit: int = 20000,
                      validation_limit: int = 2000, epochs: int = 2, batch_size: int = 128,
                      noise_std: float = 0.08, seed: int = 2026) -> dict:
    """Train and TorchScript-save a residual denoiser from manifest-backed images."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements-optional.txt for PyTorch training") from exc
    if epochs < 1 or batch_size < 1 or not 0 < noise_std < 1:
        raise ValueError("epochs/batch_size must be positive and noise_std must be in (0, 1)")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    root = Path(dataset_root)
    train_records = _load_records(root, "train", train_limit)
    validation_records = _load_records(root, "validation", validation_limit)

    class RealNoiseDataset(Dataset):
        def __init__(self, records: list[dict], base_seed: int) -> None:
            self.records = records
            self.base_seed = base_seed

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            with Image.open(self.records[index]["path"]) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            clean = torch.from_numpy(array).permute(2, 0, 1)
            generator = torch.Generator().manual_seed(self.base_seed + int(index))
            noisy = torch.clamp(clean + torch.randn(clean.shape, generator=generator) * noise_std, 0.0, 1.0)
            return noisy, clean

    train_loader = DataLoader(RealNoiseDataset(train_records, seed), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(RealNoiseDataset(validation_records, seed + 1_000_000), batch_size=batch_size,
                                   shuffle=False, num_workers=0)
    model = _make_residual_denoiser(torch, nn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for noisy, clean in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(noisy), clean)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for noisy, clean in validation_loader:
                validation_losses.append(float(criterion(model(noisy), clean)))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(train_losses)),
                        "validation_mse": float(np.mean(validation_losses))})

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    scripted = _trace_torchscript(torch, model, example)
    scripted.save(str(destination))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    report = {
        "dataset": manifest.get("dataset"),
        "source_url": manifest.get("source_url") or manifest.get("source_urls"),
        "model": str(destination), "model_sha256": digest,
        "model_type": "pytorch-residual-denoiser", "torch_version": torch.__version__,
        "train_count": len(train_records), "validation_count": len(validation_records),
        "epochs": epochs, "batch_size": batch_size, "noise_std": noise_std, "seed": seed,
        "history": history,
    }
    destination.with_suffix(destination.suffix + ".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_codec_restoration(dataset_root: str | Path, model_path: str | Path, *, train_limit: int = 10000,
                            validation_limit: int = 1000, epochs: int = 2, batch_size: int = 128,
                            codec: str = "dwt", wavelet: str = "db8", level: int = 1,
                            step: float = 6.0, quantizer: str = "scalar", crop_size: int | None = None,
                            crops_per_image: int = 1, seed: int = 2026) -> dict:
    """Train a restoration model on actual Smart Codec encode/decode artifacts.

    TR: Model, gürültü varsayımı yerine gerçek encode/decode çiftlerinden öğrenir.
    EN: The model learns from real encode/decode pairs instead of a noise assumption.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements-optional.txt for PyTorch training") from exc
    if epochs < 1 or batch_size < 1 or step <= 0 or level < 1 or crops_per_image < 1:
        raise ValueError("epochs/batch_size/level/crops_per_image must be positive and step must be positive")
    if crop_size is not None and crop_size < 8:
        raise ValueError("crop_size must be at least 8 when provided")
    if codec not in {"dwt", "dct", "prqmf4"}:
        raise ValueError("codec must be dwt, dct or prqmf4")
    if quantizer not in {"uniform", "scalar"}:
        raise ValueError("quantizer must be uniform or scalar")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    root = Path(dataset_root)
    train_records = _load_records(root, "train", train_limit)
    validation_records = _load_records(root, "validation", validation_limit)

    class CodecArtifactDataset(Dataset):
        def __init__(self, records: list[dict], *, training: bool, base_seed: int,
                     crop_size: int | None, crops_per_image: int) -> None:
            self.records = records
            self.training = training
            self.base_seed = base_seed
            self.crop_size = crop_size
            self.crops_per_image = max(1, crops_per_image if training else 1)
            # For HR training, encode the requested crop rather than a full
            # 2K image. Keep the cache bounded by record count; caching every
            # crop from a large train split can otherwise consume gigabytes.
            self.cache_pairs = len(records) <= 64
            self.pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            self.temp_dir = tempfile.TemporaryDirectory(prefix="smartcodec_codec_train_")
            self.work_path = Path(self.temp_dir.name) / "sample.swc"

        def __len__(self) -> int:
            return len(self.records) * self.crops_per_image

        def _load_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
            record_index = index // self.crops_per_image
            cache_key = int(index) if crop_size is not None else record_index
            if self.cache_pairs and cache_key in self.pairs:
                return self.pairs[cache_key]
            record = self.records[record_index]
            with Image.open(record["path"]) as image:
                clean_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if crop_size is not None:
                height, width = clean_array.shape[:2]
                size = min(crop_size, height, width)
                generator = np.random.default_rng(self.base_seed + int(index))
                if self.training:
                    top = int(generator.integers(0, height - size + 1))
                    left = int(generator.integers(0, width - size + 1))
                else:
                    top = (height - size) // 2
                    left = (width - size) // 2
                clean_array = clean_array[top : top + size, left : left + size]
                if self.training and bool(generator.integers(0, 2)):
                    clean_array = np.flip(clean_array, axis=1).copy()
                encode_array(
                    clean_array, self.work_path, codec=codec, wavelet=wavelet, level=level,
                    step=step, quantizer=quantizer,
                )
            else:
                encode_file(
                    record["path"], self.work_path, codec=codec, wavelet=wavelet, level=level,
                    step=step, quantizer=quantizer,
                )
            corrupted_array = np.asarray(decode_array(self.work_path), dtype=np.uint8)
            pair = (corrupted_array, clean_array)
            if self.cache_pairs:
                self.pairs[cache_key] = pair
            return pair

        def __getitem__(self, index: int):
            corrupted_array, clean_array = self._load_pair(int(index))
            clean = torch.from_numpy(clean_array.astype(np.float32) / 255.0).permute(2, 0, 1)
            corrupted = torch.from_numpy(corrupted_array.astype(np.float32) / 255.0).permute(2, 0, 1)
            return corrupted, clean

        def close(self) -> None:
            self.temp_dir.cleanup()

    train_dataset = CodecArtifactDataset(
        train_records, training=True, base_seed=seed, crop_size=crop_size, crops_per_image=crops_per_image,
    )
    validation_dataset = CodecArtifactDataset(
        validation_records, training=False, base_seed=seed + 1_000_000, crop_size=crop_size, crops_per_image=1,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _make_residual_denoiser(torch, nn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    history = []
    baseline_validation_mse = None
    try:
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for corrupted, clean in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(corrupted), clean)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach()))
            model.eval()
            validation_losses = []
            baseline_losses = []
            with torch.no_grad():
                for corrupted, clean in validation_loader:
                    validation_losses.append(float(criterion(model(corrupted), clean)))
                    baseline_losses.append(float(criterion(corrupted, clean)))
            if baseline_validation_mse is None:
                baseline_validation_mse = float(np.mean(baseline_losses))
            history.append({"epoch": epoch + 1, "train_mse": float(np.mean(train_losses)),
                            "validation_mse": float(np.mean(validation_losses))})
    finally:
        train_dataset.close()
        validation_dataset.close()

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    trace_size = crop_size or 32
    example = torch.zeros((1, 3, trace_size, trace_size), dtype=torch.float32)
    scripted = _trace_torchscript(torch, model, example)
    scripted.save(str(destination))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    report = {
        "dataset": manifest.get("dataset"),
        "source_url": manifest.get("source_url") or manifest.get("source_urls"),
        "model": str(destination), "model_sha256": digest,
        "model_type": "pytorch-smartcodec-artifact-restorer", "torch_version": torch.__version__,
        "train_count": len(train_records), "validation_count": len(validation_records),
        "epochs": epochs, "batch_size": batch_size, "seed": seed,
        "crop_size": crop_size, "crops_per_image": crops_per_image,
        "crop_encoding": "crop-first" if crop_size is not None else "full-image",
        "cache_pairs": bool(train_dataset.cache_pairs),
        "corruption": {"codec": codec, "wavelet": wavelet, "level": level,
                        "step": step, "quantizer": quantizer},
        "baseline_validation_mse": baseline_validation_mse,
        "history": history,
    }
    destination.with_suffix(destination.suffix + ".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_codec_restoration(dataset_root: str | Path, model_path: str | Path,
                               output_path: str | Path, *, split: str = "validation",
                               limit: int = 8, codec: str = "dwt", wavelet: str = "db8",
                               level: int = 1, step: float = 6.0,
                               quantizer: str = "scalar") -> dict:
    """Evaluate a trained restoration model on full manifest-backed images.

    Unlike crop training, this deliberately runs the model on complete images
    so the report measures deployment-shaped inference rather than only crop
    loss. The input SWC artifacts and quality metrics are recorded per sample.
    """
    if limit < 0 or level < 1 or step <= 0:
        raise ValueError("limit must be non-negative, level must be positive and step must be positive")
    if codec not in {"dwt", "dct", "prqmf4"}:
        raise ValueError("codec must be dwt, dct or prqmf4")
    if quantizer not in {"uniform", "scalar"}:
        raise ValueError("quantizer must be uniform or scalar")
    from .ai import TorchRestorationAdapter

    root = Path(dataset_root)
    # `--limit 0` is an explicit all-records mode.  The default remains a
    # small smoke evaluation so accidental full HR evaluations are avoided.
    records = _load_records(root, split, None if limit == 0 else limit)
    adapter = TorchRestorationAdapter(model_path)
    work_dir = tempfile.TemporaryDirectory(prefix="smartcodec_restore_eval_")
    samples = []
    try:
        for index, record in enumerate(records):
            with Image.open(record["path"]) as image:
                clean = np.asarray(image.convert("RGB"), dtype=np.uint8)
            artifact = Path(work_dir.name) / f"sample_{index:04d}.swc"
            info = encode_array(
                clean, artifact, codec=codec, wavelet=wavelet, level=level,
                step=step, quantizer=quantizer,
            )
            decoded = np.asarray(decode_array(artifact), dtype=np.uint8)
            restored = np.asarray(adapter.restore(decoded), dtype=np.uint8)
            decoded_psnr = psnr(clean, decoded)
            restored_psnr = psnr(clean, restored)
            samples.append({
                "path": record["path"], "relative_path": record.get("relative_path"),
                "shape": list(clean.shape), "bpp": bits_per_pixel(info["file_size"], clean.shape),
                "decoded_mse": mse(clean, decoded), "decoded_psnr": decoded_psnr,
                "decoded_ssim": ssim(clean, decoded), "restored_mse": mse(clean, restored),
                "restored_psnr": restored_psnr, "restored_ssim": ssim(clean, restored),
                "psnr_delta": restored_psnr - decoded_psnr,
            })
    finally:
        work_dir.cleanup()
    report = {
        "dataset": json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8")).get("dataset"),
        "split": split, "limit": limit, "sample_count": len(samples),
        "model": str(Path(model_path).resolve()), "model_sha256": adapter.sha256,
        "corruption": {"codec": codec, "wavelet": wavelet, "level": level,
                        "step": step, "quantizer": quantizer},
        "samples": samples,
        "summary": {
            "mean_bpp": float(np.mean([item["bpp"] for item in samples])),
            "mean_decoded_psnr": float(np.mean([item["decoded_psnr"] for item in samples])),
            "mean_restored_psnr": float(np.mean([item["restored_psnr"] for item in samples])),
            "mean_decoded_ssim": float(np.mean([item["decoded_ssim"] for item in samples])),
            "mean_restored_ssim": float(np.mean([item["restored_ssim"] for item in samples])),
            "mean_psnr_delta": float(np.mean([item["psnr_delta"] for item in samples])),
            "improved_count": sum(item["psnr_delta"] > 0 for item in samples),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(destination)
    return report


def _report_statistics(values: list[float]) -> dict[str, float | int]:
    """Return deterministic descriptive statistics for an evaluation column."""
    if not values:
        raise ValueError("Cannot summarize an empty evaluation column")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def compare_restoration_reports(report_paths: list[str | Path], output_path: str | Path) -> dict:
    """Compare one or more full-image restoration reports.

    The first report is treated as the paired baseline.  When later reports
    contain the same ``relative_path`` values, paired restored-PSNR deltas are
    reported as candidate minus baseline.  This makes model comparisons
    reproducible without rerunning expensive high-resolution evaluation.
    """
    if not report_paths:
        raise ValueError("At least one restoration report is required")
    loaded: list[tuple[Path, dict]] = []
    for raw_path in report_paths:
        path = Path(raw_path)
        report = json.loads(path.read_text(encoding="utf-8"))
        samples = report.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Restoration report has no samples: {path}")
        loaded.append((path, report))

    metrics = {
        "bpp": "bpp",
        "decoded_psnr": "decoded_psnr",
        "restored_psnr": "restored_psnr",
        "decoded_ssim": "decoded_ssim",
        "restored_ssim": "restored_ssim",
        "psnr_delta": "psnr_delta",
    }
    comparisons = []
    for path, report in loaded:
        samples = report["samples"]
        summary = {
            metric: _report_statistics([float(sample[field]) for sample in samples])
            for metric, field in metrics.items()
        }
        summary["improved_count"] = int(sum(float(sample["psnr_delta"]) > 0 for sample in samples))
        summary["improvement_rate"] = float(summary["improved_count"] / len(samples))
        comparisons.append({
            "name": path.stem,
            "report": str(path.resolve()),
            "dataset": report.get("dataset"),
            "split": report.get("split"),
            "model": report.get("model"),
            "model_sha256": report.get("model_sha256"),
            "sample_count": len(samples),
            "summary": summary,
        })

    baseline_path, baseline_report = loaded[0]
    baseline_samples = {
        str(sample.get("relative_path") or sample.get("path")): sample
        for sample in baseline_report["samples"]
    }
    paired = []
    for candidate_path, candidate_report in loaded[1:]:
        candidate_samples = {
            str(sample.get("relative_path") or sample.get("path")): sample
            for sample in candidate_report["samples"]
        }
        common = sorted(set(baseline_samples) & set(candidate_samples))
        if not common:
            paired.append({
                "baseline": baseline_path.stem,
                "candidate": candidate_path.stem,
                "common_samples": 0,
                "status": "not-paired",
            })
            continue
        restored_deltas = [
            float(candidate_samples[key]["restored_psnr"]) - float(baseline_samples[key]["restored_psnr"])
            for key in common
        ]
        decoded_deltas = [
            float(candidate_samples[key]["decoded_psnr"]) - float(baseline_samples[key]["decoded_psnr"])
            for key in common
        ]
        paired.append({
            "baseline": baseline_path.stem,
            "candidate": candidate_path.stem,
            "common_samples": len(common),
            "status": "paired",
            "restored_psnr_delta_candidate_minus_baseline": _report_statistics(restored_deltas),
            "decoded_psnr_delta_candidate_minus_baseline": _report_statistics(decoded_deltas),
            "candidate_higher_restored_count": int(sum(value > 0 for value in restored_deltas)),
        })

    result = {
        "report_type": "smartcodec-restoration-comparison-v1",
        "baseline": baseline_path.stem,
        "comparisons": comparisons,
        "paired_comparisons": paired,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["report"] = str(destination)
    return result
