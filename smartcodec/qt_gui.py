"""TR: Smart Codec'in PySide6 masaustu uygulamasi.
EN: Smart Codec's PySide6 desktop application.

TR: QSplitter ile sol kontrol paneli ve sag onizleme paneli ayrilir; boylece
    Orijinal, Decoded, Difference ve Karsilastir sekmeleri her zaman gorunur.
EN: QSplitter separates the left control panel from the right preview panel,
    keeping Original, Decoded, Difference, and Comparison always visible.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .qt_gui_benchmark import BenchmarkTab
from .qt_gui_codec import CodecTab
from .qt_gui_common import configure_combo_popup, resource_path, workspace_root
from .qt_i18n import apply_language, translate_status
from .qt_gui_theme import apply_theme
from .qt_gui_video import VideoTab


class SmartCodecQtWindow(QtWidgets.QMainWindow):
    """TR: Qt ana pencere, tema ve ortak is durumunu yonetir.
    EN: Qt main window managing themes and shared operation state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.root = workspace_root()
        self.settings_path = self.root / "outputs" / "gui_settings.json"
        self._dark = False
        self._white = False
        self._theme = "Blue"
        self._language = "en"
        self._active_tab = None
        self._last_error = ""
        self._status_message = "Ready. Select an image."
        self.setWindowTitle("Smart Codec - Image and Video Compression")
        self._set_window_icon()
        self.resize(1480, 940)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._load_settings()

    def _set_window_icon(self) -> None:
        """TR: Kamera ikonunu pencere ve gorev cubuguna uygular.
        EN: Apply the camera icon to the window and taskbar entry.
        """
        icon_path = resource_path("assets/smartcodec_camera.svg")
        if icon_path.is_file():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(14, 12, 14, 8)
        main_layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("Smart Codec")
        title.setObjectName("appTitle")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        subtitle = QtWidgets.QLabel("JPEG, JPEG2000 and wavelet-based image/video compression platform")
        subtitle.setObjectName("appSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box, 1)
        header.addWidget(QtWidgets.QLabel("Theme"))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(("White", "Blue", "Dark"))
        configure_combo_popup(self.theme_combo)
        self.theme_combo.setMinimumWidth(96)
        self.theme_combo.currentTextChanged.connect(self._theme_changed)
        self.theme_combo.setToolTip("Dark mode can be more comfortable for image inspection.")
        header.addWidget(self.theme_combo)
        header.addWidget(QtWidgets.QLabel("Language"))
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.setObjectName("languageCombo")
        self.language_combo.addItems(("English", "Türkçe"))
        configure_combo_popup(self.language_combo)
        self.language_combo.setMinimumWidth(100)
        self.language_combo.currentTextChanged.connect(self._language_changed)
        self.language_combo.setToolTip("Choose the application language.")
        header.addWidget(self.language_combo)
        main_layout.addLayout(header)

        self.tabs = QtWidgets.QTabWidget()
        self.codec_tab = CodecTab()
        self.benchmark_tab = BenchmarkTab()
        self.video_tab = VideoTab()
        self.tabs.addTab(self.codec_tab, "Encode / Decode")
        self.tabs.addTab(self.codec_tab.ai_roi_page, "AI, ROI and restoration")
        self.tabs.addTab(self.benchmark_tab, "Benchmark")
        self.tabs.addTab(self.video_tab, "Video / Transport")
        self.codec_tab.workspaceRequested.connect(lambda: self.tabs.setCurrentWidget(self.codec_tab))
        main_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        status = self.statusBar()
        self.status_label = QtWidgets.QLabel("Ready. Select an image.")
        status.addWidget(self.status_label, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(160)
        self.progress.setVisible(False)
        status.addPermanentWidget(self.progress)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_current)
        status.addPermanentWidget(self.cancel_button)
        self.copy_error_button = QtWidgets.QPushButton("Copy error")
        self.copy_error_button.setEnabled(False)
        self.copy_error_button.clicked.connect(self._copy_error)
        status.addPermanentWidget(self.copy_error_button)

        for tab in (self.codec_tab, self.benchmark_tab, self.video_tab):
            tab.statusChanged.connect(self._set_status)
            tab.busyChanged.connect(lambda busy, source=tab: self._busy_changed(source, busy))
            tab.progressChanged.connect(self._progress_changed)
            tab.errorOccurred.connect(self._error_occurred)

    def _set_status(self, message: str) -> None:
        self._status_message = message
        self.status_label.setText(translate_status(message, self._language))

    def _busy_changed(self, source, busy: bool) -> None:
        if busy:
            self._active_tab = source
            self.cancel_button.setEnabled(True)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
        elif source is self._active_tab:
            self._active_tab = None
            self.cancel_button.setEnabled(False)
            self.progress.setVisible(False)

    def _progress_changed(self, current: int, total: int) -> None:
        if not self.progress.isVisible():
            return
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(current)
        else:
            self.progress.setRange(0, 0)

    def _cancel_current(self) -> None:
        if self._active_tab is not None:
            self._active_tab.cancel_current()

    def _error_occurred(self, message: str, details: str) -> None:
        self._last_error = f"{message}\n\n{details}"
        self.copy_error_button.setEnabled(True)

    def _copy_error(self) -> None:
        if self._last_error:
            QtWidgets.QApplication.clipboard().setText(self._last_error)
            self._set_status("Error details copied to the clipboard.")

    def _theme_changed(self, name: str) -> None:
        canonical = {"Beyaz": "White", "Açık": "White", "Mavi": "Blue", "Siyah": "Dark", "Koyu": "Dark"}
        self._theme = canonical.get(name, name)
        self._dark = self._theme == "Dark"
        self._white = self._theme == "White"
        apply_theme(QtWidgets.QApplication.instance(), self._dark, self._white)

    def _language_changed(self, name: str) -> None:
        self._language = "tr" if name == "Türkçe" else "en"
        self.setWindowTitle(
            "Smart Codec - Görüntü ve Video Sıkıştırma"
            if self._language == "tr" else "Smart Codec - Image and Video Compression"
        )
        apply_language(self.centralWidget(), self._language)
        # Refresh dynamic helper text whose content depends on current values.
        self.codec_tab._update_roi_summary()
        self.benchmark_tab._update_simple_fields()
        self.video_tab._update_video_fields()
        self.video_tab._update_transport_fields()
        # The language chooser itself deliberately remains bilingual.
        self.language_combo.setCurrentText(name)
        self._set_status(self._status_message)

    def _load_settings(self) -> None:
        if not self.settings_path.is_file():
            self._theme_changed("White")
            return
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Settings JSON must contain an object")
            if "codec" in payload or "benchmark" in payload or "video" in payload:
                codec_values = payload.get("codec", {})
                benchmark_values = payload.get("benchmark", {})
                video_values = payload.get("video", {})
            else:
                # TR: Eski Tkinter duz JSON ayarlarini kaybetmeden yeni widget adlarina aktar.
                # EN: Map the legacy flat Tkinter settings without losing user preferences.
                codec_values = payload
                benchmark_values = {
                    "input": payload.get("benchmark_input", ""),
                    "output": payload.get("benchmark_output", ""),
                    "steps": payload.get("benchmark_steps", "4,8,16,32"),
                    "wavelets": payload.get("benchmark_wavelets", "haar,db4,db8"),
                    "normalization": payload.get("benchmark_normalization", "grid"),
                    "target": payload.get("benchmark_target", ""),
                    "roi_mask": payload.get("benchmark_roi_mask", ""),
                    "quantizer": payload.get("benchmark_quantizer", "uniform"),
                    "allocation": payload.get("benchmark_allocation", "greedy"),
                }
                video_values = payload
            if codec_values.get("profile") == "Ozel":
                codec_values = dict(codec_values, profile="Özel")
            self.codec_tab.apply_settings(codec_values)
            self.benchmark_tab.apply_settings(benchmark_values)
            self.video_tab.apply_settings(video_values)
            stored_theme = payload.get("qt_theme")
            theme = "Dark" if stored_theme == "dark" else "Blue" if stored_theme == "blue" else "White"
            self.theme_combo.setCurrentText(theme)
            self._theme_changed(theme)
            language = "Türkçe" if payload.get("qt_language") == "tr" else "English"
            self.language_combo.setCurrentText(language)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self._set_status(f"Could not load settings: {exc}")
            self._theme_changed("White")

    def _save_settings(self) -> None:
        payload = {
            "qt_theme": "dark" if self._dark else "white" if self._white else "blue",
            "qt_language": self._language,
            "codec": self.codec_tab.settings_payload(),
            "benchmark": self.benchmark_tab.settings_payload(),
            "video": self.video_tab.settings_payload(),
        }
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self._set_status(f"Could not save settings: {exc}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.codec_tab.cancel_current()
        self.benchmark_tab.cancel_current()
        self.video_tab.cancel_current()
        self._save_settings()
        event.accept()


def main() -> int:
    """TR: PySide6 uygulamasini baslatir. / EN: Start the PySide6 application."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("Smart Wavelet Codec")
    app.setOrganizationName("DSP-Summer2026")
    window = SmartCodecQtWindow()
    # Start maximized so both the controls and the comparison preview are
    # immediately readable on a presentation screen.
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
