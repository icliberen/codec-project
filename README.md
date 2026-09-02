# Smart Codec

**Explore image compression through visual comparisons and measurable results.**

Smart Codec is an educational desktop application built with Python and PySide6. It lets you experiment with DCT, DWT, JPEG, and JPEG2000 on the same image, inspect compression artifacts, and compare rate–distortion results. The application includes English and Turkish interfaces, White/Blue/Dark themes, and full-screen comparison tools.

> This is an educational and research project. Its custom `.swc` format is not a JPEG or JPEG2000 implementation. Standard `.jpg` and `.jp2` files are produced through a separate Pillow/OpenJPEG path. The application has not been validated for clinical or safety-critical use.

## Run on Windows

1. Download **Smart-Codec-Windows-x64.zip** from [Releases](https://github.com/icliberen/codec-project/releases).
2. Extract the **entire ZIP** into a folder where you have write permission.
3. Open `Smart Codec/Smart Codec.exe`. No Python installation is required.

**Keep the EXE and the `_internal` folder together.** The folder contains the Python, Qt, and AI runtime components. The package is not digitally signed, so Windows may display a publisher warning. You can verify the download's SHA-256 checksum against `SHA256SUMS.txt` in the release.

## Your first experiment

1. Select an image under **Encode / Decode → 1. Files**.
2. Choose **JPEG**, **JPEG2000**, **DCT**, or **DWT**.
3. For lossy DCT/DWT, select either **Target BPP** or **Target PSNR**, not both. JPEG uses a quality setting; JPEG2000 uses a rate setting.
4. Run the encode/decode operation and inspect the Original, Decoded, Restored, Difference, and Compare views.
5. Click the corresponding **Save** button to keep an output file. Processing may use temporary files, but user-selected outputs are saved explicitly. Application preferences are stored separately.

For a presentation, try a 512×512 image with DWT/db4/level 3 and DCT at target rates of 0.2, 0.4, 0.8, and 1.2 BPP. Results depend on image content: always distinguish the requested target from the achieved value.

## Features

| Area | Capabilities |
| --- | --- |
| Compression | Lossy DWT, 8×8 block DCT, and PR-QMF; lossless reversible 5/3 DWT; standard JPEG/JPEG2000 |
| Compare | Slider and side-by-side comparisons, full-screen mode, error maps, histograms, progressive stages, and an 8-bit grayscale preview |
| Transform views | Qt-rendered DWT trees, LL/LH/HL/HH subbands, DCT DC/AC explanations, and block/grid visualizations |
| Wavelet comparison | Compare db4, db8, and db12 results together when the DWT comparison mode is selected |
| PSNR–BPP chart | DCT and DWT results for the same image: **achieved BPP** on the horizontal axis and **PSNR (dB)** on the vertical axis; blue DCT points and orange DWT points |
| AI, ROI and restoration | Prioritize selected objects or regions, use optional YOLO analysis, and explore restoration tools |
| Benchmark | Evaluate multiple methods/settings and export CSV/JSON reports |
| Video / Transport | Educational frame/GOP compression, packetization, packet-loss experiments, and UDP transfer tools |

Video/Transport is not a standard H.264/H.265 encoder or a real 5G network model. YOLO requires optional dependencies and model weights. When no trained restoration model is available, a basic enhancement path may be used; do not interpret it as a learned model's output. The grayscale comparison option is a preview and does not modify the source image.

### Understanding the metrics

- **BPP** = total encoded file size in bits / (width × height). File headers are included. RGB results are not divided by the number of channels.
- **PSNR** is calculated from the MSE between the original and reconstructed images. The peak value is 255 for 8-bit images. Equal PSNR does not guarantee equal perceived quality across different images.
- **SSIM** uses the project's global SSIM implementation; results are not directly interchangeable with sliding-window implementations in other libraries.
- **Compression ratio** is based on raw pixel memory size / encoded file size. The source file's size on disk may be displayed separately; a PNG or JPEG input may already be compressed.
- Very low BPP can cause substantial distortion, especially in color images or complex textures. Some targets may be unreachable because of header overhead and codec limits.
- The GUI selects one size/quality target instead of exposing Step; the quantization step is searched internally. The CLI retains `--step` for compatibility.

## Run from source

The verified Windows configuration is **Python 3.12 x64** with **PySide6 6.8.3**. Use a fresh virtual environment to avoid Qt DLL conflicts; do not copy DLLs between Python installations.

```powershell
git clone https://github.com/icliberen/codec-project.git
cd codec-project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt "PySide6==6.8.3"
.\.venv\Scripts\python.exe start_gui.pyw
```

To enable the optional video and AI features:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video.txt -r requirements-optional.txt
.\.venv\Scripts\python.exe scripts/download_models.py
```

The model download command retrieves two YOLO11 weight files from an official Ultralytics release and verifies their SHA-256 checksums. Model weights are not stored in the source repository. JPEG2000 requires a Pillow build with OpenJPEG support.

### Command-line examples

Run the following commands from the project root using your virtual environment's Python interpreter:

```powershell
python -m smartcodec generate-samples samples
python -m smartcodec encode input.tiff output.swc --codec dwt --wavelet db4 --level 3 --quantizer scalar --target-bpp 0.4
python -m smartcodec encode input.tiff output-dct.swc --codec dct --target-psnr 30
python -m smartcodec decode output.swc decoded.png
python -m smartcodec encode input.tiff output.jpg --codec jpeg --quality 30
python -m smartcodec encode input.tiff output.jp2 --codec jpeg2000 --rate 20
python -m smartcodec --help
```

Lossless mode uses only the reversible DWT path. BPP/PSNR targets, ROI, and restoration are not available in that mode.

## Validation and Windows packaging

```powershell
.\.venv\Scripts\python.exe -m compileall -q smartcodec scripts start_gui.pyw
.\.venv\Scripts\python.exe scripts/verify_release.py
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1 -PythonExe .\.venv\Scripts\python.exe
powershell -ExecutionPolicy Bypass -File .\validate_windows_package.ps1 -MinimalEnvironment -StartupSeconds 30
```

The build command installs pinned dependencies, verifies the official YOLO weights, collects license notices, and creates `dist/Smart Codec`. It updates any existing build output at that location. Windows system font files are not copied into the package.

`scripts/verify_release.py` runs small synthetic-image checks for codec round-trips, mutually exclusive targets, and Qt startup with the supported themes and languages. It is a release smoke check, not a comprehensive quality assessment. The package validator starts the EXE with a restricted PATH, checks that the main window appears, and then closes only the process it started.

The [Windows release workflow](.github/workflows/release.yml) builds the package on its initial publication or when triggered manually through Actions. It also runs when selected packaging files change. If all checks pass, it publishes the ZIP. The version comes from `pyproject.toml`; existing releases are not overwritten.

## Repository layout

```text
smartcodec/                   Codec, CLI, Qt UI, benchmark, AI, and transport
assets/                       SVG icons
scripts/                      Model downloads, license collection, and release checks
start_gui.pyw                 Qt application entry point
build_windows.ps1             Windows packaging
validate_windows_package.ps1  Packaged application startup check
requirements*.txt             Core, optional, and build dependencies
LICENSE                       AGPL-3.0 license
licenses/                     Third-party open-source license texts
THIRD_PARTY_NOTICES.txt        Third-party notices
```

Historical working reports, personal settings, caches, generated outputs, executable/runtime files, and large model weights are excluded from the source repository. Test photographs with unverified redistribution rights are not included. Use your own images or the output of `generate-samples`.

## License

Copyright © 2026 icliberen. Smart Codec source code is released under **GNU AGPL-3.0-only**; see [LICENSE](LICENSE) for the full text. The software is provided without warranty.

Unless covered by a commercial license, Ultralytics code and YOLO weights are subject to [Ultralytics' AGPL-3.0 terms](https://www.ultralytics.com/license). Other dependencies retain their own licenses; see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and the license directory in the Windows package. Windows font files are not redistributed.
