"""Command line interface for the codec and experiments.

TR: CLI yalnızca argümanları doğrular ve iş mantığını ilgili modüle yönlendirir.
EN: The CLI validates arguments and delegates business logic to focused modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import generate_dataset, generate_samples, run_benchmark, run_normalized_benchmark, run_roi_comparison_benchmark
from .codec import decode_array, decode_file, encode_file
from .ai import TorchRestorationAdapter, WaveletParameterEstimator, estimate_parameters, select_best_restoration_model
from .image_io import load_image
from .roi import (
    analyze_scene_with_metadata,
    boxes_to_mask,
    detect_faces,
    detect_with_yolo,
    detections_to_mask,
    load_mask,
    save_mask,
    serializable_detections,
)
from .standard import decode_standard, encode_standard
from .standard import available_standard_codecs
from .diagnostics import dependency_status
from .real_data import prepare_cifar10, prepare_div2k, prepare_kodak, train_real_estimator
from .transport import receive_udp_file, run_transport_sweep, send_udp_file, simulate_file
from .video import compress_video, decompress_video, package_video, run_video_benchmark, simulate_video_transport, unpack_video
from .torch_training import compare_restoration_reports, evaluate_codec_restoration, train_codec_restoration, train_restoration


def _roi_box(value: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI kutusu x,y,w,h biçiminde olmalı") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI kutusu x,y,w,h biçiminde olmalı")
    return values


def _compact_encode_output(value):
    """Avoid dumping repeated Huffman frequency tables in normal CLI output."""
    if isinstance(value, dict):
        compact = {}
        for key, item in value.items():
            if key in {"frequencies", "transport_frequencies"}:
                continue
            if key == "transport_tiles" and isinstance(item, list):
                compact[key] = {
                    "count": len(item),
                    "priority_counts": {
                        str(priority): sum(int(tile.get("transport_priority", 1)) == priority for tile in item)
                        for priority in sorted({int(tile.get("transport_priority", 1)) for tile in item})
                    },
                    "first_tile": _compact_encode_output(item[0]) if item else None,
                    "last_tile": _compact_encode_output(item[-1]) if item else None,
                }
                continue
            compact[key] = _compact_encode_output(item)
        return compact
    if isinstance(value, list):
        return [_compact_encode_output(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart JPEG/JPEG2000-inspired image codec")
    commands = parser.add_subparsers(dest="command", required=True)

    encode = commands.add_parser("encode", help="Encode an image into SWC or standard JPEG/JPEG2000")
    encode.add_argument("input")
    encode.add_argument("output")
    encode.add_argument("--mode", choices=("lossy", "lossless"), default="lossy")
    encode.add_argument("--wavelet", default="haar", choices=("haar", "db4", "db8", "db12", "qmf"))
    encode.add_argument("--codec", choices=("dwt", "dct", "prqmf4", "jpeg", "jpeg2000"), default="dwt")
    encode.add_argument("--level", type=int, default=2)
    encode.add_argument("--step", type=float, default=12.0)
    encode.add_argument("--quantizer", choices=("uniform", "scalar"), default="uniform")
    encode.add_argument("--colorspace", choices=("ycbcr", "rgb"), default="ycbcr")
    encode.add_argument("--quality", type=int, default=75, help="JPEG quality for --codec jpeg")
    encode.add_argument("--rate", type=float, default=4.0, help="JPEG2000 rate for --codec jpeg2000")
    encode.add_argument("--roi", action="append", type=_roi_box, help="ROI: x,y,width,height; repeatable")
    encode.add_argument("--roi-mask", help="Grayscale ROI mask image")
    encode.add_argument("--yolo", action="store_true", help="Detect ROI boxes with optional YOLO adapter")
    encode.add_argument("--yolo-model", default="auto", help="YOLO model path, or auto (prefers local/bundled yolo11m-seg.pt)")
    encode.add_argument("--yolo-confidence", type=float, default=0.25)
    encode.add_argument("--yolo-device")
    encode.add_argument("--yolo-classes", help="Comma-separated YOLO class labels")
    encode.add_argument("--faces", action="store_true", help="Detect face ROIs with OpenCV")
    encode.add_argument("--roi-strength", type=float, default=0.65)
    encode.add_argument("--roi-feather", type=int, default=0)
    encode.add_argument("--target-bpp", type=float)
    encode.add_argument("--target-psnr", type=float)
    encode.add_argument("--allocation", choices=("greedy", "lagrangian", "dp"), default="greedy")
    encode.add_argument("--metadata", help="JSON metadata object stored in the SWC header")
    encode.add_argument("--ai-restoration", action="store_true", help="Record optional lossy restoration intent")
    encode.add_argument("--transport-segments", action="store_true", help="Pack coefficient bands independently for priority transport")
    encode.add_argument("--transport-tiles", action="store_true", help="Split semantic transport into image-space coefficient tiles")
    encode.add_argument("--transport-tile-size", type=int, default=64, help="Image-space transport tile size in pixels")

    decode = commands.add_parser("decode", help="Decode SWC or standard JPEG/JPEG2000 files")
    decode.add_argument("input")
    decode.add_argument("output")

    samples = commands.add_parser("generate-samples", help="Generate deterministic test images")
    samples.add_argument("output_dir")

    dataset = commands.add_parser("generate-dataset", help="Generate categorized benchmark images")
    dataset.add_argument("output_dir")

    verify_dataset = commands.add_parser("verify-dataset", help="Verify a dataset manifest, image metadata and SHA-256 records")
    verify_dataset.add_argument("dataset_root", help="Dataset directory or dataset_manifest.json path")
    verify_dataset.add_argument("--no-hash", action="store_true", help="Skip SHA-256 calculation and only verify files/metadata")

    benchmark = commands.add_parser("benchmark", help="Run rate-distortion experiments")
    benchmark.add_argument("input_dir")
    benchmark.add_argument("output_dir")
    benchmark.add_argument("--level", type=int, default=2)
    benchmark.add_argument("--steps", default="4,8,16,32", help="Comma-separated DWT quantization steps")
    benchmark.add_argument("--wavelets", default="haar,db4", help="Comma-separated wavelets")
    benchmark.add_argument("--dataset-root", help="External dataset root (kept distinct from surrogate input)")
    benchmark.add_argument("--codecs", default="dwt,dct,prqmf4,jpeg,jpeg2000")
    benchmark.add_argument("--no-jpeg2000", action="store_true")
    benchmark.add_argument("--roi-mask")
    benchmark.add_argument("--ai", action="store_true")
    benchmark.add_argument("--restoration", action="store_true")
    benchmark.add_argument("--quantizer", choices=("uniform", "scalar"), default="uniform")
    benchmark.add_argument("--allocation", choices=("greedy", "lagrangian", "dp"), default="greedy")
    benchmark.add_argument("--split", choices=("train", "validation", "test"), help="Use a named manifest split")
    benchmark.add_argument("--limit", type=int, help="Limit the number of manifest images for a quick large-dataset sample")
    benchmark.add_argument("--normalize-psnr", type=float, help="Compare codecs at a shared target PSNR instead of independent quality grids")
    benchmark.add_argument("--normalize-bpp", type=float, help="Compare codecs at a shared target bits-per-pixel rate")
    benchmark.add_argument("--no-adaptive-step-search", action="store_true",
                           help="Keep the fixed SWC step grid for normalized-BPP benchmarks")
    benchmark.add_argument("--compare-roi", action="store_true", help="Run matched target-BPP benchmark with ROI disabled/enabled")

    video_encode = commands.add_parser("video-encode", help="Encode video frames into an SWC manifest")
    video_encode.add_argument("input")
    video_encode.add_argument("output_dir")
    video_encode.add_argument("--mode", choices=("lossy", "lossless"), default="lossy")
    video_encode.add_argument("--codec", choices=("dwt", "dct", "prqmf4"), default="dwt")
    video_encode.add_argument("--wavelet", default="haar")
    video_encode.add_argument("--level", type=int, default=2)
    video_encode.add_argument("--step", type=float, default=12.0)
    video_encode.add_argument("--gop-size", type=int, default=1)
    video_encode.add_argument("--keyframe-interval", type=int)
    video_encode.add_argument("--motion-estimation", action="store_true")
    video_encode.add_argument("--motion-method", choices=("none", "translation", "block", "optical-flow"), default="none")
    video_encode.add_argument("--motion-compensation", action="store_true", help="Use stored motion metadata for P-frame prediction")
    video_encode.add_argument("--roi-mask", help="Optional grayscale ROI mask; tracked across frames when enabled")
    video_encode.add_argument("--roi-strength", type=float, default=0.65)
    video_encode.add_argument("--roi-feather", type=int, default=0)
    video_encode.add_argument("--roi-tracking", action="store_true", help="Warp the ROI with the selected motion estimator")
    video_encode.add_argument("--transport-segments", action="store_true", help="Pack each SWC band for semantic transport priority")
    video_encode.add_argument("--transport-tiles", action="store_true", help="Split each frame payload into image-space transport tiles")
    video_encode.add_argument("--transport-tile-size", type=int, default=64)

    video_decode = commands.add_parser("video-decode", help="Decode an SWC video manifest")
    video_decode.add_argument("manifest")
    video_decode.add_argument("output")

    video_pack = commands.add_parser("video-pack", help="Pack a video manifest and all SWC frames for transport")
    video_pack.add_argument("manifest")
    video_pack.add_argument("bundle")

    video_unpack = commands.add_parser("video-unpack", help="Safely unpack a transported video bundle")
    video_unpack.add_argument("bundle")
    video_unpack.add_argument("output_dir")

    video_transport = commands.add_parser("video-transport", help="Transport SWC video frames independently with partial decode")
    video_transport.add_argument("manifest")
    video_transport.add_argument("output_dir")
    video_transport.add_argument("--loss-rate", type=float, default=0.0)
    video_transport.add_argument("--mtu", type=int, default=1200)
    video_transport.add_argument("--latency-ms", type=float, default=0.0)
    video_transport.add_argument("--jitter-ms", type=float, default=0.0)
    video_transport.add_argument("--bandwidth-mbps", type=float, default=0.0)
    video_transport.add_argument("--burst-loss", type=int, default=0)
    video_transport.add_argument("--retransmission", action="store_true")
    video_transport.add_argument("--fec", action="store_true")
    video_transport.add_argument("--seed", type=int, default=2026)

    video_benchmark = commands.add_parser("video-benchmark", help="Compare I-frame and GOP video settings")
    video_benchmark.add_argument("input")
    video_benchmark.add_argument("output_dir")
    video_benchmark.add_argument("--gop-sizes", default="1,4,8", help="Comma-separated positive GOP sizes")
    video_benchmark.add_argument("--mode", choices=("lossy", "lossless"), default="lossy")
    video_benchmark.add_argument("--codec", choices=("dwt", "dct", "prqmf4"), default="dwt")
    video_benchmark.add_argument("--wavelet", default="haar")
    video_benchmark.add_argument("--level", type=int, default=2)
    video_benchmark.add_argument("--step", type=float, default=12.0)
    video_benchmark.add_argument("--motion-method", choices=("none", "translation", "block", "optical-flow"), default="translation")
    video_benchmark.add_argument("--motion-compensation", action="store_true")
    video_benchmark.add_argument("--roi-mask")
    video_benchmark.add_argument("--roi-tracking", action="store_true")
    video_benchmark.add_argument("--transport-segments", action="store_true")
    video_benchmark.add_argument("--transport-tiles", action="store_true")
    video_benchmark.add_argument("--transport-tile-size", type=int, default=64)

    transport = commands.add_parser("simulate-transport", help="Simulate packet transport for an SWC file")
    transport.add_argument("input")
    transport.add_argument("output")
    transport.add_argument("--loss-rate", type=float, default=0.0)
    transport.add_argument("--mtu", type=int, default=1200)
    transport.add_argument("--latency-ms", type=float, default=0.0)
    transport.add_argument("--jitter-ms", type=float, default=0.0)
    transport.add_argument("--bandwidth-mbps", type=float, default=0.0)
    transport.add_argument("--burst-loss", type=int, default=0)
    transport.add_argument("--retransmission", action="store_true")
    transport.add_argument("--fec", action="store_true")
    transport.add_argument("--seed", type=int, default=2026)
    transport.add_argument("--reference", help="Reference image for decoded quality metrics")
    transport.add_argument("--roi-mask", help="ROI mask for ROI/background transport metrics")

    transport_benchmark = commands.add_parser("transport-benchmark", help="Run reproducible packet-loss sweep with CSV/JSON/PNG outputs")
    transport_benchmark.add_argument("input")
    transport_benchmark.add_argument("output_dir")
    transport_benchmark.add_argument("--loss-rates", default="0,0.01,0.05,0.1,0.2,0.4", help="Comma-separated rates between 0 and 1")
    transport_benchmark.add_argument("--seeds", default="2026", help="Comma-separated deterministic seeds")
    transport_benchmark.add_argument("--mtu", type=int, default=1200)
    transport_benchmark.add_argument("--latency-ms", type=float, default=0.0)
    transport_benchmark.add_argument("--jitter-ms", type=float, default=0.0)
    transport_benchmark.add_argument("--bandwidth-mbps", type=float, default=0.0)
    transport_benchmark.add_argument("--burst-loss", type=int, default=0)
    transport_benchmark.add_argument("--retransmission", action="store_true")
    transport_benchmark.add_argument("--fec", action="store_true")
    transport_benchmark.add_argument("--no-priority", action="store_true", help="Do not preserve SWC header priority")
    transport_benchmark.add_argument("--reference", help="Reference image for decoded quality metrics")
    transport_benchmark.add_argument("--roi-mask", help="ROI mask for ROI/background sweep metrics")

    udp_send = commands.add_parser("udp-send", help="Send an SWC/file payload over a live UDP/IP endpoint")
    udp_send.add_argument("input")
    udp_send.add_argument("host")
    udp_send.add_argument("port", type=int)
    udp_send.add_argument("--mtu", type=int, default=1200)
    udp_send.add_argument("--frame-id", type=int, default=0)
    udp_send.add_argument("--fec", action="store_true", help="Send one XOR parity packet for single-packet recovery")
    udp_send.add_argument("--hardware-test", action="store_true", help="Explicitly label this report as a hardware test")

    udp_receive = commands.add_parser("udp-receive", help="Receive one complete live UDP/IP payload")
    udp_receive.add_argument("bind_host")
    udp_receive.add_argument("port", type=int)
    udp_receive.add_argument("output")
    udp_receive.add_argument("--timeout", type=float, default=5.0)
    udp_receive.add_argument("--stream-id")
    udp_receive.add_argument("--no-fec", action="store_true", help="Do not attempt XOR parity recovery")
    udp_receive.add_argument("--hardware-test", action="store_true", help="Explicitly label this report as a hardware test")
    udp_receive.add_argument("--reference", help="Reference image for decoded quality metrics")
    udp_receive.add_argument("--roi-mask", help="ROI mask for ROI/background UDP metrics")

    standard_encode = commands.add_parser("standard-encode", help="Write a standard JPEG or JPEG2000 file")
    standard_encode.add_argument("input")
    standard_encode.add_argument("output")
    standard_encode.add_argument("--codec", choices=("jpeg", "jpeg2000"), default="jpeg")
    standard_encode.add_argument("--quality", type=int, default=75)
    standard_encode.add_argument("--rate", type=float, default=4.0)

    standard_decode = commands.add_parser("standard-decode", help="Decode a standard JPEG/JPEG2000 file")
    standard_decode.add_argument("input")
    standard_decode.add_argument("output")

    estimate = commands.add_parser("estimate", help="Print deterministic AI/fallback codec parameters")
    estimate.add_argument("input")
    estimate.add_argument("--model", help="Optional saved estimator JSON; otherwise use the deterministic fallback")
    restore = commands.add_parser("restore", help="Apply the best available PyTorch restoration model to a lossy SWC")
    restore.add_argument("input")
    restore.add_argument("output")
    restore.add_argument("--model", default="auto", help="TorchScript model path, or auto (default: best local model)")
    restore.add_argument("--reference", help="Optional original image for decoded/restored MSE, PSNR and SSIM")
    estimator_train = commands.add_parser("train-estimator", help="Fit and save the lightweight estimator baseline")
    estimator_train.add_argument("dataset_root")
    estimator_train.add_argument("output")

    cifar = commands.add_parser("download-cifar10", help="Download and prepare the official CIFAR-10 image dataset")
    cifar.add_argument("output_dir")
    cifar.add_argument("--train-limit", type=int)
    cifar.add_argument("--test-limit", type=int)
    cifar.add_argument("--validation-limit", type=int)
    cifar.add_argument("--force", action="store_true")

    kodak = commands.add_parser("download-kodak", help="Download the Kodak real-image compression reference suite")
    kodak.add_argument("output_dir")
    kodak.add_argument("--seed", type=int, default=2026)

    div2k = commands.add_parser("download-div2k", help="Download and prepare the official DIV2K high-resolution dataset")
    div2k.add_argument("output_dir")
    div2k.add_argument("--train-limit", type=int, help="Optional manifest sample limit for train images")
    div2k.add_argument("--validation-limit", type=int, help="Optional manifest sample limit for validation images")
    div2k.add_argument("--seed", type=int, default=2026)
    div2k.add_argument("--force", action="store_true", help="Redownload archives and refresh extracted files")

    real_train = commands.add_parser("train-real-estimator", help="Train/evaluate the estimator on manifest-backed real images")
    real_train.add_argument("dataset_root")
    real_train.add_argument("model_output")
    real_train.add_argument("--train-limit", type=int, default=200)
    real_train.add_argument("--validation-limit", type=int, default=40)
    real_train.add_argument("--test-limit", type=int, default=40)
    real_train.add_argument("--seed", type=int, default=2026)
    restorer_train = commands.add_parser("train-restorer", help="Train a PyTorch residual restoration model on a real image manifest")
    restorer_train.add_argument("dataset_root")
    restorer_train.add_argument("model_output")
    restorer_train.add_argument("--train-limit", type=int, default=20000)
    restorer_train.add_argument("--validation-limit", type=int, default=2000)
    restorer_train.add_argument("--epochs", type=int, default=2)
    restorer_train.add_argument("--batch-size", type=int, default=128)
    restorer_train.add_argument("--noise-std", type=float, default=0.08)
    restorer_train.add_argument("--seed", type=int, default=2026)
    codec_restorer_train = commands.add_parser("train-codec-restorer", help="Train PyTorch restoration on Smart Codec encode/decode artifacts")
    codec_restorer_train.add_argument("dataset_root")
    codec_restorer_train.add_argument("model_output")
    codec_restorer_train.add_argument("--train-limit", type=int, default=10000)
    codec_restorer_train.add_argument("--validation-limit", type=int, default=1000)
    codec_restorer_train.add_argument("--epochs", type=int, default=2)
    codec_restorer_train.add_argument("--batch-size", type=int, default=128)
    codec_restorer_train.add_argument("--codec", choices=("dwt", "dct", "prqmf4"), default="dwt")
    codec_restorer_train.add_argument("--wavelet", choices=("haar", "db4", "db8", "db12", "qmf"), default="db8")
    codec_restorer_train.add_argument("--level", type=int, default=1)
    codec_restorer_train.add_argument("--step", type=float, default=6.0)
    codec_restorer_train.add_argument("--quantizer", choices=("uniform", "scalar"), default="scalar")
    codec_restorer_train.add_argument("--crop-size", type=int, help="Train on deterministic/random square crops for high-resolution images")
    codec_restorer_train.add_argument("--crops-per-image", type=int, default=1)
    codec_restorer_train.add_argument("--seed", type=int, default=2026)
    restorer_eval = commands.add_parser("evaluate-restorer", help="Evaluate a restoration model on complete manifest images")
    restorer_eval.add_argument("dataset_root")
    restorer_eval.add_argument("model")
    restorer_eval.add_argument("output_report")
    restorer_eval.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    restorer_eval.add_argument("--limit", type=int, default=8,
                               help="Number of records to evaluate; 0 evaluates the complete split")
    restorer_eval.add_argument("--codec", choices=("dwt", "dct", "prqmf4"), default="dwt")
    restorer_eval.add_argument("--wavelet", choices=("haar", "db4", "db8", "db12", "qmf"), default="db8")
    restorer_eval.add_argument("--level", type=int, default=1)
    restorer_eval.add_argument("--step", type=float, default=6.0)
    restorer_eval.add_argument("--quantizer", choices=("uniform", "scalar"), default="scalar")
    restorer_compare = commands.add_parser("compare-restorer", help="Compare full-image restoration JSON reports")
    restorer_compare.add_argument("output_report")
    restorer_compare.add_argument("reports", nargs="+", help="Two or more evaluate-restorer JSON reports")
    commands.add_parser("status", help="Show core/optional dependency and standard codec availability")
    return parser


def main() -> None:
    # TR: Tek giriş noktası komutu parse eder; codec/benchmark kodu burada çoğaltılmaz.
    # EN: The single entry point parses commands without duplicating codec/benchmark logic.
    args = _parser().parse_args()
    if args.command == "encode":
        roi_mask = None
        roi_sources = []
        if args.roi_mask:
            shape = load_image(args.input).shape[:2]
            roi_mask = load_mask(args.roi_mask, shape)
            roi_sources.append("mask-file")
        if args.roi:
            shape = load_image(args.input).shape[:2]
            generated = boxes_to_mask(shape, args.roi)
            roi_mask = generated if roi_mask is None else (roi_mask + generated).clip(0, 1)
            roi_sources.append("manual")
        if args.yolo:
            shape = load_image(args.input).shape[:2]
            classes = {value.strip() for value in args.yolo_classes.split(",") if value.strip()} if args.yolo_classes else None
            detected, yolo_metadata = analyze_scene_with_metadata(args.input, args.yolo_model, args.yolo_confidence,
                                                                  args.yolo_device, classes)
            generated = detections_to_mask(shape, detected)
            roi_mask = generated if roi_mask is None else (roi_mask + generated).clip(0, 1)
            roi_sources.append("yolo")
        if args.faces:
            shape = load_image(args.input).shape[:2]
            detected = detect_faces(args.input)
            generated = boxes_to_mask(shape, detected)
            roi_mask = generated if roi_mask is None else (roi_mask + generated).clip(0, 1)
            roi_sources.append("faces")
        metadata = json.loads(args.metadata) if args.metadata else None
        if roi_sources:
            metadata = dict(metadata or {})
            metadata["roi_source"] = roi_sources
        if args.yolo:
            metadata = dict(metadata or {})
            metadata["semantic_detections"] = serializable_detections(detected)
            metadata["semantic_model"] = yolo_metadata
        info = encode_file(args.input, args.output, mode=args.mode, wavelet=args.wavelet, level=args.level,
                           step=args.step, quantizer=args.quantizer, codec=args.codec, colorspace=args.colorspace, roi_mask=roi_mask,
                           roi_strength=args.roi_strength, target_bpp=args.target_bpp, target_psnr=args.target_psnr,
                           allocation_method=args.allocation, metadata=metadata, restoration=args.ai_restoration,
                           ai_enabled=args.ai_restoration, roi_feather=args.roi_feather,
                           transport_segments=args.transport_segments, transport_tiles=args.transport_tiles,
                           transport_tile_size=args.transport_tile_size, standard_quality=args.quality,
                           standard_rate=args.rate)
        print(json.dumps(_compact_encode_output(info), indent=2, default=str))
    elif args.command == "decode":
        print(json.dumps(decode_file(args.input, args.output), indent=2))
    elif args.command == "generate-samples":
        print(json.dumps([str(path) for path in generate_samples(args.output_dir)], indent=2))
    elif args.command == "generate-dataset":
        print(json.dumps([str(path) for path in generate_dataset(args.output_dir)], indent=2))
    elif args.command == "benchmark":
        steps = [float(value) for value in args.steps.split(",") if value.strip()]
        wavelets = [value.strip() for value in args.wavelets.split(",") if value.strip()]
        selected_codecs = [value.strip() for value in args.codecs.split(",") if value.strip()]
        if args.normalize_psnr is not None and args.normalize_bpp is not None:
            raise ValueError("--normalize-psnr and --normalize-bpp cannot be used together")
        if args.compare_roi:
            if args.normalize_bpp is None or args.normalize_psnr is not None:
                raise ValueError("--compare-roi requires --normalize-bpp and cannot use --normalize-psnr")
            if not args.roi_mask:
                raise ValueError("--compare-roi requires --roi-mask")
            rows = run_roi_comparison_benchmark(
                args.input_dir, args.output_dir, target_bpp=args.normalize_bpp, roi_mask_path=args.roi_mask,
                step_values=steps, wavelets=wavelets, level=args.level, dataset_root=args.dataset_root,
                codecs=selected_codecs, include_jpeg2000=not args.no_jpeg2000,
                quantizer=args.quantizer, allocation_method=args.allocation, split=args.split, limit=args.limit,
                adaptive_step_search=not args.no_adaptive_step_search,
            )
        elif args.normalize_psnr is not None or args.normalize_bpp is not None:
            rows = run_normalized_benchmark(
                args.input_dir, args.output_dir, target_psnr=args.normalize_psnr, target_bpp=args.normalize_bpp,
                step_values=steps, wavelets=wavelets,
                level=args.level, dataset_root=args.dataset_root, codecs=selected_codecs,
                include_jpeg2000=not args.no_jpeg2000, roi_mask_path=args.roi_mask,
                quantizer=args.quantizer, allocation_method=args.allocation, split=args.split, limit=args.limit,
                adaptive_step_search=not args.no_adaptive_step_search,
            )
        else:
            rows = run_benchmark(args.input_dir, args.output_dir, step_values=steps, wavelets=wavelets, level=args.level,
                                 dataset_root=args.dataset_root, codecs=selected_codecs,
                                 include_jpeg2000=not args.no_jpeg2000, roi_mask_path=args.roi_mask,
                                 ai_enabled=args.ai, restoration=args.restoration, quantizer=args.quantizer,
                                 allocation_method=args.allocation, split=args.split, limit=args.limit)
        print(json.dumps({"rows": len(rows), "output_dir": str(Path(args.output_dir).resolve())}, indent=2))
    elif args.command == "video-encode":
        motion_method = args.motion_method
        if args.motion_estimation and motion_method == "none":
            motion_method = "translation"
        info = compress_video(args.input, args.output_dir, mode=args.mode, codec=args.codec, wavelet=args.wavelet,
                              level=args.level, step=args.step, gop_size=args.gop_size,
                              keyframe_interval=args.keyframe_interval or args.gop_size,
                              motion_estimation=args.motion_estimation or motion_method != "none",
                              motion_method=motion_method, motion_compensation=args.motion_compensation,
                              roi_mask_path=args.roi_mask, roi_strength=args.roi_strength,
                              roi_feather=args.roi_feather, roi_tracking=args.roi_tracking,
                               transport_segments=args.transport_segments, transport_tiles=args.transport_tiles,
                               transport_tile_size=args.transport_tile_size)
        print(json.dumps(info, indent=2))
    elif args.command == "video-decode":
        print(json.dumps(decompress_video(args.manifest, args.output), indent=2))
    elif args.command == "video-pack":
        print(json.dumps(package_video(args.manifest, args.bundle), indent=2))
    elif args.command == "video-unpack":
        print(json.dumps(unpack_video(args.bundle, args.output_dir), indent=2))
    elif args.command == "video-transport":
        print(json.dumps(simulate_video_transport(
            args.manifest, args.output_dir, loss_rate=args.loss_rate, mtu=args.mtu,
            latency_ms=args.latency_ms, jitter_ms=args.jitter_ms, bandwidth_mbps=args.bandwidth_mbps,
            burst_loss=args.burst_loss, retransmission=args.retransmission, fec=args.fec, seed=args.seed,
        ), indent=2))
    elif args.command == "video-benchmark":
        try:
            gop_sizes = [int(value.strip()) for value in args.gop_sizes.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("gop-sizes must be comma-separated positive integers") from exc
        print(json.dumps(run_video_benchmark(
            args.input, args.output_dir, gop_sizes=gop_sizes, mode=args.mode, codec=args.codec,
            wavelet=args.wavelet, level=args.level, step=args.step, motion_method=args.motion_method,
            motion_compensation=args.motion_compensation, roi_mask_path=args.roi_mask,
             roi_tracking=args.roi_tracking, transport_segments=args.transport_segments,
             transport_tiles=args.transport_tiles, transport_tile_size=args.transport_tile_size,
        ), indent=2))
    elif args.command == "simulate-transport":
        print(json.dumps(simulate_file(args.input, args.output, args.loss_rate, args.mtu, args.seed,
                                       latency_ms=args.latency_ms, jitter_ms=args.jitter_ms,
                                       bandwidth_mbps=args.bandwidth_mbps, burst_loss=args.burst_loss,
                                       retransmission=args.retransmission, fec=args.fec,
                                       reference_path=args.reference, roi_mask_path=args.roi_mask), indent=2))
    elif args.command == "transport-benchmark":
        try:
            loss_rates = [float(value.strip()) for value in args.loss_rates.split(",") if value.strip()]
            seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("loss-rates and seeds must be comma-separated numbers") from exc
        print(json.dumps(run_transport_sweep(
            args.input, args.output_dir, loss_rates=loss_rates, seeds=seeds, mtu=args.mtu,
            latency_ms=args.latency_ms, jitter_ms=args.jitter_ms, bandwidth_mbps=args.bandwidth_mbps,
            burst_loss=args.burst_loss, retransmission=args.retransmission, fec=args.fec,
            priorities=not args.no_priority, reference_path=args.reference, roi_mask_path=args.roi_mask,
        ), indent=2))
    elif args.command == "udp-send":
        print(json.dumps(send_udp_file(args.input, args.host, args.port, mtu=args.mtu,
                                       frame_id=args.frame_id, fec=args.fec, hardware_test=args.hardware_test), indent=2))
    elif args.command == "udp-receive":
        print(json.dumps(receive_udp_file(args.bind_host, args.port, args.output, timeout=args.timeout,
                                          expected_stream_id=args.stream_id, fec=not args.no_fec,
                                          hardware_test=args.hardware_test, reference_path=args.reference,
                                          roi_mask_path=args.roi_mask), indent=2))
    elif args.command == "standard-encode":
        print(json.dumps(encode_standard(load_image(args.input), args.output, codec=args.codec,
                                         quality=args.quality, rate=args.rate), indent=2))
    elif args.command == "standard-decode":
        from .image_io import save_image
        array = decode_standard(args.input)
        save_image(array, args.output)
        print(json.dumps({"output": args.output, "shape": list(array.shape)}, indent=2))
    elif args.command == "estimate":
        image = load_image(args.input)
        if args.model:
            estimator = WaveletParameterEstimator.load(args.model)
            parameters, confidence = estimator.predict(image)
            model_name = str(Path(args.model).resolve())
        else:
            parameters, confidence = estimate_parameters(image)
            model_name = "deterministic-fallback-baseline"
        print(json.dumps({"parameters": parameters, "confidence": confidence, "model": model_name}, indent=2))
    elif args.command == "restore":
        from .codec import _read_container
        from .image_io import save_image
        from .metrics import mse, psnr, ssim
        header, _ = _read_container(args.input)
        if header.get("mode") == "lossless":
            raise ValueError("PyTorch restoration is forbidden for lossless SWC output")
        selected_model, selection = select_best_restoration_model(args.model)
        if selected_model is None:
            raise RuntimeError("No local restoration model found; train one with train-codec-restorer or pass --model PATH")
        adapter = TorchRestorationAdapter(selected_model)
        decoded = decode_array(args.input)
        restored = adapter.restore(decoded)
        save_image(restored, args.output)
        result = {"input": args.input, "output": args.output, "model": str(selected_model),
                  "model_sha256": adapter.sha256, "device": adapter.device, "model_selection": selection,
                  "mode": header.get("mode"), "metrics": None}
        if args.reference:
            reference = load_image(args.reference)
            result["metrics"] = {
                "decoded_mse": mse(reference, decoded), "decoded_psnr": psnr(reference, decoded),
                "decoded_ssim": ssim(reference, decoded), "restored_mse": mse(reference, restored),
                "restored_psnr": psnr(reference, restored), "restored_ssim": ssim(reference, restored),
            }
            report_path = Path(args.output).with_suffix(".report.json")
            report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["report"] = str(report_path)
        print(json.dumps(result, indent=2))
    elif args.command == "train-estimator":
        from .dataset import scan_dataset
        images = [load_image(row["path"]) for row in scan_dataset(args.dataset_root) if row.get("readable")]
        if not images:
            raise ValueError("No readable images found for estimator training")
        estimator = WaveletParameterEstimator().fit(images)
        print(json.dumps({"output": str(estimator.save(args.output)), "samples": len(images)}, indent=2))
    elif args.command == "verify-dataset":
        from .dataset import verify_manifest
        print(json.dumps(verify_manifest(args.dataset_root, check_hash=not args.no_hash), indent=2))
    elif args.command == "download-cifar10":
        from .real_data import download_cifar10
        if args.force:
            download_cifar10(Path(args.output_dir) / "raw", force=True)
        payload = prepare_cifar10(args.output_dir, train_limit=args.train_limit, test_limit=args.test_limit,
                                  validation_limit=args.validation_limit)
        print(json.dumps({"dataset": "CIFAR-10", "records": len(payload["records"]),
                          "manifest": str(Path(args.output_dir) / "dataset_manifest.json"),
                          "sample_manifest": payload.get("sample_manifest"),
                          "sample_records": payload.get("sample_records")}, indent=2))
    elif args.command == "train-real-estimator":
        report = train_real_estimator(args.dataset_root, args.model_output, train_limit=args.train_limit,
                                      validation_limit=args.validation_limit, test_limit=args.test_limit, seed=args.seed)
        print(json.dumps({"model": report["model"], "validation": report["validation"], "test": report["test"]}, indent=2))
    elif args.command == "download-kodak":
        payload = prepare_kodak(args.output_dir, seed=args.seed)
        print(json.dumps({"dataset": payload["dataset"], "records": len(payload["records"]),
                          "manifest": str(Path(args.output_dir) / "dataset_manifest.json")}, indent=2))
    elif args.command == "download-div2k":
        payload = prepare_div2k(args.output_dir, train_limit=args.train_limit,
                                 validation_limit=args.validation_limit, seed=args.seed, force=args.force)
        print(json.dumps({"dataset": payload["dataset"], "records": len(payload["records"]),
                          "manifest": str(Path(args.output_dir) / "dataset_manifest.json"),
                          "sample_manifest": payload.get("sample_manifest"),
                          "sample_records": payload.get("sample_records"),
                          "archives": payload["archives"]}, indent=2))
    elif args.command == "train-restorer":
        report = train_restoration(args.dataset_root, args.model_output, train_limit=args.train_limit,
                                   validation_limit=args.validation_limit, epochs=args.epochs,
                                   batch_size=args.batch_size, noise_std=args.noise_std, seed=args.seed)
        print(json.dumps(report, indent=2))
    elif args.command == "train-codec-restorer":
        report = train_codec_restoration(
            args.dataset_root, args.model_output, train_limit=args.train_limit,
            validation_limit=args.validation_limit, epochs=args.epochs,
            batch_size=args.batch_size, codec=args.codec, wavelet=args.wavelet,
            level=args.level, step=args.step, quantizer=args.quantizer, crop_size=args.crop_size,
            crops_per_image=args.crops_per_image, seed=args.seed,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "evaluate-restorer":
        report = evaluate_codec_restoration(
            args.dataset_root, args.model, args.output_report, split=args.split,
            limit=args.limit, codec=args.codec, wavelet=args.wavelet, level=args.level,
            step=args.step, quantizer=args.quantizer,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "compare-restorer":
        report = compare_restoration_reports(args.reports, args.output_report)
        print(json.dumps(report, indent=2))
    elif args.command == "status":
        print(json.dumps({"dependencies": dependency_status(), "standard_codecs": available_standard_codecs()}, indent=2))


if __name__ == "__main__":
    main()
