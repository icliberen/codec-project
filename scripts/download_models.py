"""Explicit, checksum-verified download of the optional official YOLO models."""
from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
MODELS = {
    "yolo11m-seg.pt": "eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf",
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    for name, expected in MODELS.items():
        target = ROOT / name
        if target.exists():
            if sha256(target) != expected:
                raise RuntimeError(f"Existing model differs from the official checksum: {target}")
            print(f"Verified {name}")
            continue
        temporary = target.with_suffix(".download")
        try:
            with urllib.request.urlopen(BASE + name, timeout=120) as response, temporary.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if sha256(temporary) != expected:
                raise RuntimeError(f"Checksum mismatch: {name}")
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {name}")


if __name__ == "__main__":
    main()
