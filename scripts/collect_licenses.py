"""Collect installed component versions and verbatim license/notice files."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    destination = ROOT / "third_party_licenses"
    destination.mkdir(exist_ok=True)
    # PySide6 6.8.3 wheels list the LGPL/GPL alternatives in METADATA but
    # include only a commercial-license notice; ship the open-source texts too.
    qt_notices = destination / "Qt-license-texts"
    qt_notices.mkdir(exist_ok=True)
    for license_file in (ROOT / "licenses").glob("*.txt"):
        shutil.copy2(license_file, qt_notices / license_file.name)
    manifest = []
    for distribution in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata.get("Name", "").lower()):
        name = distribution.metadata.get("Name", "unknown")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        component = destination / f"{safe_name}-{distribution.version}"
        copied = []
        for file in distribution.files or []:
            if not re.search(r"(?i)(licen[cs]e|copying|copyright|notice)", str(file)):
                continue
            source = Path(distribution.locate_file(file))
            if not source.is_file() or source.suffix.lower() in {".py", ".pyc", ".pyd", ".dll", ".exe"}:
                continue
            relative = Path(*[part for part in Path(file).parts if part not in {"..", "."}])
            if relative.is_absolute():
                continue
            # Wheel notice directories can exceed Windows' path-length limit.
            # Keep a stable short filename and preserve the original in the manifest.
            key = hashlib.sha256(str(file).encode("utf-8")).hexdigest()[:16]
            safe_file = re.sub(r"[^A-Za-z0-9_.-]", "_", relative.name)[:80]
            target = component / f"{key}-{safe_file}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({"file": target.relative_to(destination).as_posix(), "original": str(file)})
        manifest.append({"name": name, "version": distribution.version,
                         "license": distribution.metadata.get("License-Expression") or distribution.metadata.get("License"),
                         "project_urls": distribution.metadata.get_all("Project-URL") or [],
                         "license_files": copied})
    for candidate in (Path(sys.base_prefix) / "LICENSE.txt", Path(sys.base_prefix) / "LICENSE"):
        if candidate.is_file():
            shutil.copy2(candidate, destination / "Python-LICENSE.txt")
            break
    (destination / "components.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Recorded {len(manifest)} installed distributions and their available notices.")


if __name__ == "__main__":
    main()
