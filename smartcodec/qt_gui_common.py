"""TR: PySide6 ortak yardimcilari ve arka plan is parcacigi altyapisi.
EN: Shared PySide6 helpers and background-worker infrastructure.

TR: Qt arayuzu codec cekirdeginden bagimsiz kalir; bu modul sadece UI ile
    uzun suren islemler arasindaki guvenli kopruyu saglar.
EN: The Qt UI stays independent from the codec core; this module only provides
    the safe bridge between the UI thread and long-running operations.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets


def configure_combo_popup(combo: QtWidgets.QComboBox) -> QtWidgets.QComboBox:
    """Keep the app's short combo menus compact and free of phantom rows.

    Every combo box in Smart Codec contains at most a handful of entries.  On
    Windows, Qt's styled popup can nevertheless reserve one item-height for
    the popup scroll bar's arrow buttons.  Hiding that unnecessary bar on the
    view itself avoids the reserved blank strip more reliably than a QSS
    descendant selector (the popup lives in a separate top-level window).
    """
    view = combo.view()
    view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
    if isinstance(view, QtWidgets.QListView):
        view.setSpacing(0)
        view.setUniformItemSizes(True)
    combo.setMaxVisibleItems(10)
    return combo


def workspace_root() -> Path:
    """TR: Kaynak ve paketli calisma kokunu bulur.
    EN: Locate the source-tree or packaged application workspace root.
    """
    candidates = [Path.cwd(), Path(__file__).resolve().parents[1]]
    if getattr(sys, "frozen", False):
        candidates.insert(0, Path(sys.executable).resolve().parent)
        candidates.insert(1, Path(sys.executable).resolve().parent.parent)
    for candidate in candidates:
        if ((candidate / "outputs").is_dir() or (candidate / "_internal" / "outputs").is_dir()
                or (candidate / "smartcodec").is_dir()):
            return candidate
    return Path.cwd()


def resource_path(relative_path: str) -> Path:
    """TR: Kaynak ve PyInstaller paketindeki asset yolunu bulur.
    EN: Resolve an asset path both in source and PyInstaller layouts.
    """
    bundled_root = Path(getattr(sys, "_MEIPASS", Path.cwd()))
    candidates = [
        bundled_root / relative_path,
        Path(sys.executable).resolve().parent / relative_path,
        Path(__file__).resolve().parents[1] / relative_path,
        workspace_root() / relative_path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def configure_application_font(app: QtWidgets.QApplication) -> None:
    """Load a valid UI font when Qt cannot access the system font database."""
    windows_fonts = Path(os.environ.get("WINDIR", r"C:\\Windows")) / "Fonts"
    candidates = (
        resource_path("fonts/segoeui.ttf"),
        windows_fonts / "segoeui.ttf",
        windows_fonts / "arial.ttf",
    )
    for font_path in candidates:
        if not font_path.is_file():
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_path))
        if font_id < 0:
            continue
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QtGui.QFont(families[0], 9))
            return


class WorkerSignals(QtCore.QObject):
    """TR: Is parcacigi sinyalleri ana Qt thread'ine tasir.
    EN: Worker signals marshal results back to the main Qt thread.
    """

    result = QtCore.Signal(object)
    error = QtCore.Signal(str, str)
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()


class Worker(QtCore.QRunnable):
    """TR: Codec/benchmark/video islemlerini UI'yi kilitlemeden calistirir.
    EN: Run codec, benchmark, and video work without blocking the UI.
    """

    def __init__(self, function) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self.function(self.signals.progress.emit)
        except Exception as exc:  # UI boundary: convert errors to readable text.
            self.signals.error.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class AsyncWidget(QtWidgets.QWidget):
    """TR: Qt sekmeleri icin ortak asenkron isletim tabani.
    EN: Shared asynchronous-operation base for Qt tabs.
    """

    statusChanged = QtCore.Signal(str)
    busyChanged = QtCore.Signal(bool)
    progressChanged = QtCore.Signal(int, int)
    errorOccurred = QtCore.Signal(str, str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pool = QtCore.QThreadPool.globalInstance()
        self._worker: Worker | None = None
        self._busy = False
        self._cancel_requested = False
        self._progress_total = 0

    @property
    def is_busy(self) -> bool:
        return self._busy

    def run_background(self, function, on_success, operation: str, total: int = 0) -> None:
        """TR: Bir fonksiyonu worker thread'inde baslatir.
        EN: Start a function on a worker thread and deliver its result safely.
        """
        if self._busy:
            return
        self._busy = True
        self._cancel_requested = False
        self._progress_total = int(total or 0)
        self.busyChanged.emit(True)
        self.progressChanged.emit(0, self._progress_total)
        self.statusChanged.emit(f"{operation} started…")

        worker = Worker(function)
        self._worker = worker
        worker.signals.progress.connect(self._on_progress)
        worker.signals.result.connect(
            lambda result: self._handle_result(result, on_success),
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        worker.signals.error.connect(self._handle_error)
        worker.signals.finished.connect(self._on_finished)
        self._pool.start(worker)

    @QtCore.Slot(int)
    def _on_progress(self, current: int) -> None:
        self.progressChanged.emit(int(current), self._progress_total)
        if self._progress_total:
            self.statusChanged.emit(f"Working: {current}/{self._progress_total}")

    @QtCore.Slot(object)
    def _handle_result(self, result, callback) -> None:
        if self._cancel_requested:
            self.statusChanged.emit("Operation cancelled after the current step.")
            return
        try:
            callback(result)
        except Exception as exc:
            self._handle_error(f"{type(exc).__name__}: {exc}", traceback.format_exc())

    @QtCore.Slot(str, str)
    def _handle_error(self, message: str, details: str) -> None:
        self.statusChanged.emit(f"Error: {message}")
        self.errorOccurred.emit(message, details)

    @QtCore.Slot()
    def _on_finished(self) -> None:
        self._worker = None
        self._busy = False
        self.busyChanged.emit(False)

    def cancel_current(self) -> None:
        """TR: Yeni is baslatilmasini durdurur; aktif codec adimi guvenle tamamlanir.
        EN: Prevents new work and lets the active codec step finish safely.
        """
        if self._busy:
            self._cancel_requested = True
            self.statusChanged.emit("Cancellation requested; the active step will finish safely.")


def display_image(array: np.ndarray) -> Image.Image:
    """TR: Her dtype/kanal tipini ekranda gosterilebilir RGB PIL goruntusune cevirir.
    EN: Convert any supported dtype/channel layout to a displayable RGB PIL image.
    """
    values = np.asarray(array)
    if values.ndim == 2 or (values.ndim == 3 and values.shape[2] == 1):
        values = values[..., 0] if values.ndim == 3 else values
        return Image.fromarray(_display_uint8(values), mode="L").convert("RGB")
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError("Image must be a 2D array or a 3D array with at least three channels")
    return Image.fromarray(_display_uint8(values[..., :3]), mode="RGB")


def _display_uint8(values: np.ndarray) -> np.ndarray:
    """Convert an arbitrary numeric image to a visible 8-bit preview."""
    values = np.asarray(values)
    if values.dtype == np.uint8:
        return np.ascontiguousarray(values)
    numeric = values.astype(np.float32, copy=False)
    finite = np.isfinite(numeric)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    low = float(np.min(numeric[finite]))
    high = float(np.max(numeric[finite]))
    if high <= low:
        # A flat image is still useful in the preview; keep it mid-grey.
        return np.full(values.shape, 128, dtype=np.uint8)
    scaled = (numeric - low) * (255.0 / (high - low))
    scaled = np.where(finite, scaled, 0.0)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def pil_to_pixmap(image: Image.Image, max_edge: int = 1800) -> QtGui.QPixmap:
    """TR: PIL goruntusunu Qt pixmap'ine cevirir ve buyuk goruntuyu sinirlar.
    EN: Convert a PIL image to a Qt pixmap while bounding huge previews.
    """
    image = image.convert("RGB")
    if max(image.size) > max_edge:
        image = image.copy()
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    data = image.tobytes("raw", "RGB")
    qimage = QtGui.QImage(data, image.width, image.height, image.width * 3, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimage.copy())


def array_to_pixmap(array: np.ndarray, max_edge: int = 1800) -> QtGui.QPixmap:
    """TR: NumPy goruntusunu Qt pixmap'ine cevirir.
    EN: Convert a NumPy image array to a Qt pixmap.
    """
    return pil_to_pixmap(display_image(array), max_edge=max_edge)


def format_file_size(byte_count: int | float | None) -> str:
    """Format a byte count compactly for metric cards and comparison captions."""
    if byte_count is None:
        return "—"
    size = max(0.0, float(byte_count))
    units = ("B", "KiB", "MiB", "GiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if size < 1024.0 or candidate == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size)} B"
    return f"{size:.1f} {unit}"


def browse_file(parent, title: str, file_filter: str, save: bool = False, suffix: str = "") -> str:
    """TR: Ortak ac/kaydet dosya secici.
    EN: Shared open/save file dialog helper.
    """
    if save:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, title, "", file_filter)
    else:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(parent, title, "", file_filter)
    if path and suffix and not Path(path).suffix:
        path += suffix
    return path


def browse_directory(parent, title: str) -> str:
    """TR/EN: Select a directory / Dizin secici."""
    return QtWidgets.QFileDialog.getExistingDirectory(parent, title)
