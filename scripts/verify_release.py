"""Small release smoke check; no external photos, models, or network required."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6 import QtWidgets
from smartcodec.codec import decode_array, encode_array
from smartcodec.standard import decode_standard, encode_standard
from smartcodec import qt_gui_common


def main() -> None:
    yy, xx = np.indices((96, 96))
    gray = ((xx * 2 + yy) % 256).astype(np.uint8)
    rgb = np.stack((gray, np.flipud(gray), np.fliplr(gray)), axis=2)
    with tempfile.TemporaryDirectory(prefix="smartcodec-release-check-") as temporary:
        root = Path(temporary)
        for label, source in (("gray8", gray), ("rgb8", rgb), ("gray16", gray.astype(np.uint16) * 257)):
            path = root / f"{label}.swc"
            encode_array(source, path, codec="dwt", mode="lossless", level=2)
            np.testing.assert_array_equal(decode_array(path), source)
        for codec in ("dwt", "dct"):
            for target in ({"target_bpp": 2.0}, {"target_psnr": 30.0}):
                path = root / f"{codec}.swc"
                encode_array(rgb, path, codec=codec, wavelet="db4", level=2, **target)
                result = decode_array(path)
                assert result.shape == rgb.shape and result.dtype == rgb.dtype
            try:
                encode_array(rgb, root / "invalid.swc", codec=codec, target_bpp=1, target_psnr=30)
            except ValueError:
                pass
            else:
                raise AssertionError("Both targets must not be accepted together")
        for codec, suffix in (("jpeg", "jpg"), ("jpeg2000", "jp2")):
            path = root / f"standard.{suffix}"
            encode_standard(rgb, path, codec=codec)
            assert decode_standard(path).shape == rgb.shape
        print("PASS: reversible gray8/rgb8/gray16; DCT/DWT targets; JPEG/JPEG2000")

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        qt_gui_common.configure_application_font(app)
        # Isolate GUI settings and outputs from the user's workspace.
        with patch.object(qt_gui_common, "workspace_root", return_value=root):
            from smartcodec.qt_gui import SmartCodecQtWindow
            window = SmartCodecQtWindow()
            window.show()
            for theme in ("White", "Blue", "Dark"):
                window.theme_combo.setCurrentText(theme)
                window._theme_changed(theme)
                app.processEvents()
                assert window._theme == theme and bool(app.styleSheet())
            for language in ("Türkçe", "English"):
                window.language_combo.setCurrentText(language)
                app.processEvents()
                assert window._language == ("tr" if language == "Türkçe" else "en")
            assert window.windowTitle() == "Smart Codec - Image and Video Compression"
            assert window.tabs.count() == 4
            window.close()
            app.processEvents()
        print("PASS: Qt startup, three themes, two languages, preserved title/tabs")


if __name__ == "__main__":
    main()
