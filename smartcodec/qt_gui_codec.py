"""TR: PySide6 goruntu codec sekmesi.
EN: PySide6 image codec tab.

TR: Bu modul Tkinter widget'larini tekrar kullanmaz; ayni codec/AI/ROI
    backend'lerini responsive Qt kontrolleriyle cagirir.
EN: This module does not reuse Tkinter widgets; it calls the same codec,
    AI, and ROI backends through responsive Qt controls.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pywt
from PySide6 import QtCore, QtGui, QtWidgets

from .ai import (
    TorchRestorationAdapter,
    detail_reconstruction,
    select_best_restoration_model,
)
from .codec import decode_array, decode_file, encode_array, encode_file
from .image_io import load_image, save_image
from .metrics import mse, psnr, region_metrics, ssim
from .qt_gui_common import (
    AsyncWidget,
    browse_directory,
    browse_file,
    configure_combo_popup,
    format_file_size,
    workspace_root,
)
from .qt_i18n import USER_ROLE, combo_value, set_combo_value, translate, translate_status
from .qt_gui_preview import (
    GrayscaleComparisonPreview,
    HistogramPreview,
    ImagePreview,
    RateDistortionPreview,
    TransformGridPreview,
    TransformTreePreview,
    WaveletComparisonPreview,
    comparison_image,
    difference_image,
)
from .roi import analyze_scene, boxes_to_mask, detect_faces, detections_to_mask, load_mask
from .standard import decode_standard, encode_standard


IMAGE_FILTER = "Images (*.png *.bmp *.tif *.tiff *.jpg *.jpeg);;All files (*.*)"
SWC_FILTER = "SWC (*.swc);;All files (*.*)"
FIXED_YOLO_MODEL = "yolo11m-seg.pt"
WAVELET_COMPARISON_SELECTION = "db4 / db8 / db12 comparison"
BASE_COMPARISON_MODES = (
    "Slider", "Side-by-side", "Progressive stages", "Transform tree",
    "Block grid", "Error heatmap", "Histogram", "PSNR–BPP graph",
    "8-bit grayscale",
)


class CompareFullscreenDialog(QtWidgets.QDialog):
    """Full-screen comparison window with the five image tabs preserved."""

    def __init__(self, owner: "CodecTab") -> None:
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Smart Codec - Fullscreen comparison")
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        language = str(owner.property("ui_language") or "en")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(translate("Fullscreen comparison", language))
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        top.addWidget(title)
        top.addStretch(1)
        exit_button = QtWidgets.QPushButton(translate("Exit fullscreen", language))
        exit_button.clicked.connect(self.close)
        top.addWidget(exit_button)
        layout.addLayout(top)

        comparison_controls = QtWidgets.QHBoxLayout()
        comparison_controls.addWidget(QtWidgets.QLabel(translate("View", language)))
        self.mode_combo = QtWidgets.QComboBox()
        for mode in owner._available_comparison_modes():
            self.mode_combo.addItem(translate(mode, language))
            self.mode_combo.setItemData(self.mode_combo.count() - 1, mode, USER_ROLE)
        configure_combo_popup(self.mode_combo)
        set_combo_value(self.mode_combo, combo_value(owner.comparison_mode))
        self.mode_combo.setToolTip(translate("Choose how the original and decoded images are displayed.", language))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        comparison_controls.addWidget(self.mode_combo)
        comparison_controls.addSpacing(10)
        self.slider_label = QtWidgets.QLabel(translate("Slider", language))
        comparison_controls.addWidget(self.slider_label)
        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 100)
        self.position_slider.setSingleStep(1)
        self.position_slider.setPageStep(10)
        self.position_slider.setValue(owner.comparison_position.value())
        self.position_slider.setToolTip(translate("Move the slider to reveal the decoded image.", language))
        self.position_slider.valueChanged.connect(self._position_changed)
        comparison_controls.addWidget(self.position_slider, 1)
        self.position_value_label = QtWidgets.QLabel(f"{self.position_slider.value()}%")
        self.position_value_label.setMinimumWidth(44)
        comparison_controls.addWidget(self.position_value_label)
        layout.addLayout(comparison_controls)
        self._set_slider_controls_visible(combo_value(self.mode_combo) == "Slider")

        metrics_strip = QtWidgets.QWidget()
        metrics_strip.setObjectName("metricsStrip")
        metrics_layout = QtWidgets.QHBoxLayout(metrics_strip)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(8)
        self.metric_values: dict[str, QtWidgets.QLabel] = {}
        for metric_title, key in (
            ("PSNR", "psnr"), ("SSIM", "ssim"), ("BPP", "bpp"),
            ("Compression ratio", "ratio"), ("Original size", "original_size"),
            ("Encoded size", "encoded_size"),
        ):
            card = QtWidgets.QFrame()
            card.setObjectName("metricCard")
            card.setProperty("metric_key", key)
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(13, 7, 13, 8)
            card_layout.setSpacing(1)
            heading = QtWidgets.QLabel(translate(metric_title, language))
            heading.setObjectName("metricTitle")
            value_label = QtWidgets.QLabel("—")
            value_label.setObjectName("metricValue")
            card_layout.addWidget(heading)
            card_layout.addWidget(value_label)
            metrics_layout.addWidget(card, 1)
            self.metric_values[key] = value_label
        layout.addWidget(metrics_strip)
        self.sync_metrics({key: label.text() for key, label in owner.metric_values.items()})

        self.tabs = QtWidgets.QTabWidget()
        self.original_preview = ImagePreview(allow_roi=False)
        self.decoded_preview = ImagePreview()
        self.restored_preview = ImagePreview()
        self.difference_preview = ImagePreview()
        self.comparison_preview = ImagePreview()
        self.transform_preview = TransformTreePreview()
        self.transform_grid_preview = TransformGridPreview()
        self.wavelet_comparison_preview = WaveletComparisonPreview()
        self.histogram_preview = HistogramPreview()
        self.rate_distortion_preview = RateDistortionPreview()
        self.grayscale_preview = GrayscaleComparisonPreview()
        self.comparison_stack = QtWidgets.QStackedLayout()
        comparison_holder = QtWidgets.QWidget()
        comparison_holder.setLayout(self.comparison_stack)
        self.comparison_stack.addWidget(self.comparison_preview)
        self.comparison_stack.addWidget(self.transform_preview)
        self.comparison_stack.addWidget(self.transform_grid_preview)
        self.comparison_stack.addWidget(self.wavelet_comparison_preview)
        self.comparison_stack.addWidget(self.histogram_preview)
        self.comparison_stack.addWidget(self.rate_distortion_preview)
        self.comparison_stack.addWidget(self.grayscale_preview)
        self.tabs.addTab(self.original_preview, translate("Original", language))
        self.tabs.addTab(self.decoded_preview, translate("Decoded", language))
        self.tabs.addTab(self.restored_preview, translate("Restored", language))
        self.tabs.addTab(self.difference_preview, translate("Difference", language))
        self.tabs.addTab(comparison_holder, translate("Compare", language))
        self.tabs.setCurrentIndex(4)
        layout.addWidget(self.tabs, 1)

    @QtCore.Slot(int)
    def _mode_changed(self, _index: int) -> None:
        """Drive the main comparison renderer from the full-screen view menu."""
        mode = combo_value(self.mode_combo)
        self._set_slider_controls_visible(mode == "Slider")
        if combo_value(self.owner.comparison_mode) != mode:
            set_combo_value(self.owner.comparison_mode, mode)

    def _set_slider_controls_visible(self, visible: bool) -> None:
        """Show the reveal control only for the view that actually uses it."""
        self.slider_label.setVisible(visible)
        self.position_slider.setVisible(visible)
        self.position_value_label.setVisible(visible)

    @QtCore.Slot(int)
    def _position_changed(self, value: int) -> None:
        """Drive the real comparison position from the full-screen control."""
        self.position_value_label.setText(f"{value}%")
        if self.owner.comparison_position.value() != value:
            self.owner.comparison_position.setValue(value)

    def sync_controls(self, mode: str, value: int) -> None:
        """Mirror main-window controls without recursively emitting changes."""
        with QtCore.QSignalBlocker(self.mode_combo):
            set_combo_value(self.mode_combo, mode)
        with QtCore.QSignalBlocker(self.position_slider):
            self.position_slider.setValue(value)
        self.position_value_label.setText(f"{value}%")
        self._set_slider_controls_visible(mode == "Slider")

    def sync_modes(self, modes: tuple[str, ...], selected: str) -> None:
        """Mirror codec-dependent comparison choices in the full-screen view."""
        language = str(self.owner.property("ui_language") or "en")
        with QtCore.QSignalBlocker(self.mode_combo):
            self.mode_combo.clear()
            for mode in modes:
                self.mode_combo.addItem(translate(mode, language))
                self.mode_combo.setItemData(self.mode_combo.count() - 1, mode, USER_ROLE)
            set_combo_value(self.mode_combo, selected)

    def sync_metrics(self, values: dict[str, str]) -> None:
        """Mirror the main result cards without recomputing codec measurements."""
        for key, label in self.metric_values.items():
            label.setText(str(values.get(key, "—")))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.owner._fullscreen_dialog = None
        super().closeEvent(event)


class CodecTab(AsyncWidget):
    """TR: Goruntu kodlama, ROI, AI restoration ve karsilastirma ekranidir.
    EN: Image encoding, ROI, AI restoration, and comparison screen.
    """

    workspaceRequested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._fields: dict[str, QtWidgets.QWidget] = {}
        self._optional_field_rows: dict[str, tuple[QtWidgets.QWidget, QtWidgets.QLabel | None]] = {}
        # Several codecs reuse the same compact controls in the form.  Keep a
        # separate draft for each codec so, for example, a destructive DCT
        # demonstration step does not silently become the next DWT step.
        self._codec_setting_fields = {
            "jpeg": ("mode", "standard_quality"),
            "jpeg2000": ("mode", "standard_rate"),
            "dwt": ("mode", "wavelet", "level", "step", "quantizer", "colorspace", "quality_target", "target_bpp", "target_psnr"),
            "dct": ("mode", "step", "colorspace", "quality_target", "target_bpp", "target_psnr"),
            "prqmf4": ("mode", "level", "step", "quantizer", "colorspace", "quality_target", "target_bpp", "target_psnr"),
        }
        self._codec_settings_by_method: dict[str, dict] = {
            "jpeg": {"mode": "lossy", "standard_quality": 75},
            "jpeg2000": {"mode": "lossy", "standard_rate": 4.0},
            "dwt": {
                "mode": "lossy", "wavelet": "haar", "level": 2, "step": 12.0,
                "quantizer": "uniform", "colorspace": "ycbcr", "quality_target": "BPP",
                "target_bpp": 1.0, "target_psnr": 30.0,
            },
            "dct": {
                "mode": "lossy", "step": 12.0, "colorspace": "ycbcr", "quality_target": "BPP",
                "target_bpp": 1.0, "target_psnr": 30.0,
            },
            "prqmf4": {
                "mode": "lossy", "level": 2, "step": 12.0, "quantizer": "uniform",
                "colorspace": "ycbcr", "quality_target": "BPP", "target_bpp": 1.0, "target_psnr": 30.0,
            },
        }
        self._active_codec: str | None = None
        self._suspend_codec_setting_sync = False
        self.original_array: np.ndarray | None = None
        self.decoded_array: np.ndarray | None = None
        self.restored_array: np.ndarray | None = None
        self.current_roi_mask: np.ndarray | None = None
        self.semantic_roi_mask: np.ndarray | None = None
        self.last_encode_info: dict | None = None
        self.progressive_stages: list[np.ndarray] = []
        self.progressive_labels: list[str] = []
        self.wavelet_comparison_results: list[dict] = []
        self._wavelet_comparison_signature: tuple | None = None
        self._wavelet_compare_timer = QtCore.QTimer(self)
        self._wavelet_compare_timer.setSingleShot(True)
        self._wavelet_compare_timer.setInterval(250)
        self._wavelet_compare_timer.timeout.connect(self._ensure_wavelet_comparison)
        self.rate_distortion_results: list[dict] = []
        self.rate_distortion_codec = ""
        self._rate_distortion_signature: tuple | None = None
        self._last_encode_options: dict | None = None
        self._rate_distortion_timer = QtCore.QTimer(self)
        self._rate_distortion_timer.setSingleShot(True)
        self._rate_distortion_timer.setInterval(250)
        self._rate_distortion_timer.timeout.connect(self._ensure_rate_distortion)
        self._fullscreen_dialog: CompareFullscreenDialog | None = None
        session_template = str(Path(QtCore.QDir.tempPath()) / "smartcodec_image_XXXXXX")
        self._session_output_dir = QtCore.QTemporaryDir(session_template)
        if not self._session_output_dir.isValid():
            raise RuntimeError("Could not create temporary image workspace")
        self._session_output_dir.setAutoRemove(True)
        self._latest_encoded_path: Path | None = None
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._cleanup_session_output)
        self._settings_path = workspace_root() / "outputs" / "gui_settings.json"
        self._build_ui()

    @QtCore.Slot()
    def _cleanup_session_output(self) -> None:
        """Remove the session workspace on normal widget/application shutdown."""
        if hasattr(self, "_session_output_dir"):
            self._session_output_dir.remove()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._cleanup_session_output()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # TR: Widget construction / EN: Widget construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)
        self._build_files(controls_layout)
        self._build_codec_settings(controls_layout)
        self._build_ai_roi()
        self._build_transport(controls_layout)
        self._build_actions(controls_layout)
        controls_layout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("controlScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        # TR: Ilk acilista dosya yollari ve Sec/Kaydet dugmeleri rahat gorunsun.
        # EN: Give file paths and Select/Save buttons enough room on first launch.
        scroll.setMinimumWidth(650)
        scroll.setMaximumWidth(700)
        # Keep Qt's native vertical scrollbar directly on the settings area.
        # Its themed arrow buttons and draggable thumb are shared by every
        # scrollable surface, rather than duplicating extra controls here.
        self.controls_scroll = scroll

        preview = QtWidgets.QWidget()
        preview.setObjectName("previewPane")
        preview_layout = QtWidgets.QVBoxLayout(preview)
        preview_layout.setContentsMargins(8, 10, 10, 10)
        metrics_strip = QtWidgets.QWidget()
        metrics_strip.setObjectName("metricsStrip")
        metrics_layout = QtWidgets.QHBoxLayout(metrics_strip)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(8)
        self.metric_values: dict[str, QtWidgets.QLabel] = {}
        for title, key, unit in (
            ("PSNR", "psnr", "dB"), ("SSIM", "ssim", ""), ("BPP", "bpp", "bpp"),
            ("Compression ratio", "ratio", "×"), ("Original size", "original_size", ""),
            ("Encoded size", "encoded_size", ""),
        ):
            card = QtWidgets.QFrame()
            card.setObjectName("metricCard")
            card.setProperty("metric_key", key)
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setContentsMargins(13, 9, 13, 10)
            card_layout.setSpacing(2)
            heading = QtWidgets.QLabel(title)
            heading.setObjectName("metricTitle")
            value = QtWidgets.QLabel("—")
            value.setObjectName("metricValue")
            value.setProperty("unit", unit)
            card_layout.addWidget(heading)
            card_layout.addWidget(value)
            metrics_layout.addWidget(card, 1)
            self.metric_values[key] = value
        preview_layout.addWidget(metrics_strip)
        self.metrics_label = QtWidgets.QLabel("No encode/decode operation yet.")
        self.metrics_label.setObjectName("metricsDetails")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setMinimumHeight(48)
        preview_layout.addWidget(self.metrics_label)
        self.preview_tabs = QtWidgets.QTabWidget()
        self.original_preview = ImagePreview(allow_roi=True)
        self.decoded_preview = ImagePreview()
        self.restored_preview = ImagePreview()
        self.difference_preview = ImagePreview()
        self.comparison_preview = ImagePreview()
        self.transform_preview = TransformTreePreview()
        self.transform_grid_preview = TransformGridPreview()
        self.wavelet_comparison_preview = WaveletComparisonPreview()
        self.histogram_preview = HistogramPreview()
        self.rate_distortion_preview = RateDistortionPreview()
        self.grayscale_preview = GrayscaleComparisonPreview()
        self.preview_tabs.addTab(self.original_preview, "Original")
        self.preview_tabs.addTab(self.decoded_preview, "Decoded")
        self.preview_tabs.addTab(self.restored_preview, "Restored")
        self.preview_tabs.addTab(self.difference_preview, "Difference")
        comparison_page = QtWidgets.QWidget()
        comparison_page.setObjectName("comparisonPage")
        comparison_layout = QtWidgets.QVBoxLayout(comparison_page)
        comparison_bar = QtWidgets.QHBoxLayout()
        comparison_bar.addWidget(QtWidgets.QLabel("View"))
        self.comparison_mode = QtWidgets.QComboBox()
        for mode in BASE_COMPARISON_MODES:
            self.comparison_mode.addItem(mode)
            self.comparison_mode.setItemData(self.comparison_mode.count() - 1, mode, USER_ROLE)
        configure_combo_popup(self.comparison_mode)
        comparison_bar.addWidget(self.comparison_mode)
        comparison_bar.addWidget(QtWidgets.QLabel("Compare with"))
        self.comparison_target = QtWidgets.QComboBox()
        self.comparison_target.addItems(("Decoded", "Restored"))
        configure_combo_popup(self.comparison_target)
        comparison_bar.addWidget(self.comparison_target)
        self.comparison_slider_label = QtWidgets.QLabel("Slider")
        comparison_bar.addWidget(self.comparison_slider_label)
        self.comparison_position = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.comparison_position.setRange(0, 100)
        self.comparison_position.setValue(50)
        comparison_bar.addWidget(self.comparison_position, 1)
        self.comparison_value_label = QtWidgets.QLabel("50%")
        self.comparison_value_label.setMinimumWidth(40)
        comparison_bar.addWidget(self.comparison_value_label)
        fullscreen_button = QtWidgets.QPushButton("Fullscreen")
        fullscreen_button.clicked.connect(self._open_compare_fullscreen)
        fullscreen_button.setToolTip("Open the comparison in a full-screen window while keeping the image tabs available.")
        comparison_bar.addWidget(fullscreen_button)
        fit_button = QtWidgets.QPushButton("Fit")
        fit_button.clicked.connect(self._fit_comparison_view)
        fit_button.setToolTip("Resize the comparison image to fit the available area.")
        comparison_bar.addWidget(fit_button)
        comparison_layout.addLayout(comparison_bar)
        self.comparison_stack = QtWidgets.QStackedLayout()
        comparison_holder = QtWidgets.QWidget()
        comparison_holder.setLayout(self.comparison_stack)
        self.comparison_stack.addWidget(self.comparison_preview)
        self.comparison_stack.addWidget(self.transform_preview)
        self.comparison_stack.addWidget(self.transform_grid_preview)
        self.comparison_stack.addWidget(self.wavelet_comparison_preview)
        self.comparison_stack.addWidget(self.histogram_preview)
        self.comparison_stack.addWidget(self.rate_distortion_preview)
        self.comparison_stack.addWidget(self.grayscale_preview)
        comparison_layout.addWidget(comparison_holder, 1)
        self.preview_tabs.addTab(comparison_page, "Compare")
        preview_layout.addWidget(self.preview_tabs, 1)
        splitter.addWidget(scroll)
        splitter.addWidget(preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([660, 1100])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.original_preview.roiSelected.connect(self._roi_selected)
        self.comparison_position.setSingleStep(1)
        self.comparison_position.setPageStep(10)
        self.comparison_position.setToolTip("Move the slider to reveal the decoded image.")
        self.comparison_mode.setToolTip("Choose how the original and decoded images are displayed.")
        self.comparison_target.setToolTip("Choose the image to compare with the original.")
        self.comparison_mode.currentTextChanged.connect(self._comparison_mode_changed)
        self.comparison_target.currentTextChanged.connect(self._refresh_comparison)
        self.comparison_position.valueChanged.connect(self._comparison_position_changed)
        self._fields["wavelet"].currentTextChanged.connect(self._wavelet_selection_changed)
        self._fields["level"].valueChanged.connect(self._transform_setting_changed)
        for field_name in ("quantizer", "colorspace", "mode", "quality_target", "target_bpp", "target_psnr"):
            field = self._fields[field_name]
            if isinstance(field, QtWidgets.QComboBox):
                field.currentTextChanged.connect(self._wavelet_setting_changed)
            else:
                field.valueChanged.connect(self._wavelet_setting_changed)

    def _group(self, parent_layout: QtWidgets.QVBoxLayout, title: str) -> QtWidgets.QFormLayout:
        group = QtWidgets.QGroupBox(title)
        group.setObjectName("controlCard")
        form = QtWidgets.QFormLayout(group)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        parent_layout.addWidget(group)
        return form

    def _comparison_mode_changed(self, _mode: str) -> None:
        """Show the split control only in the comparison mode that uses it."""
        selected_mode = combo_value(self.comparison_mode)
        is_slider_mode = selected_mode == "Slider"
        self.comparison_slider_label.setVisible(is_slider_mode)
        self.comparison_position.setVisible(is_slider_mode)
        self.comparison_value_label.setVisible(is_slider_mode)
        self._refresh_comparison()
        if selected_mode == "Wavelet comparison":
            self._wavelet_compare_timer.start()
        elif selected_mode == "PSNR–BPP graph":
            self._rate_distortion_timer.start()

    def _wavelet_comparison_selected(self) -> bool:
        return (
            self._value("codec_method") == "dwt"
            and self._value("wavelet") == WAVELET_COMPARISON_SELECTION
        )

    def _effective_wavelet(self) -> str:
        """Use db4 for the ordinary output while comparison mode builds all three."""
        selected = str(self._value("wavelet"))
        return "db4" if selected == WAVELET_COMPARISON_SELECTION else selected

    def _available_comparison_modes(self) -> tuple[str, ...]:
        modes = list(BASE_COMPARISON_MODES)
        if self._wavelet_comparison_selected():
            modes.insert(5, "Wavelet comparison")
        return tuple(modes)

    def _update_comparison_mode_options(self) -> None:
        """Expose the three-wavelet renderer only for its explicit DWT mode."""
        if not hasattr(self, "comparison_mode"):
            return
        modes = self._available_comparison_modes()
        current = combo_value(self.comparison_mode)
        selected = current if current in modes else "Slider"
        language = str(self.property("ui_language") or "en")
        with QtCore.QSignalBlocker(self.comparison_mode):
            self.comparison_mode.clear()
            for mode in modes:
                self.comparison_mode.addItem(translate(mode, language))
                self.comparison_mode.setItemData(self.comparison_mode.count() - 1, mode, USER_ROLE)
            set_combo_value(self.comparison_mode, selected)
        dialog = self._fullscreen_dialog
        if dialog is not None:
            dialog.sync_modes(modes, selected)
        if selected != current:
            self._comparison_mode_changed(selected)

    def _wavelet_selection_changed(self, _value=None) -> None:
        self.wavelet_comparison_results = []
        self._wavelet_comparison_signature = None
        if self._wavelet_comparison_selected() and self._value("mode") != "lossy":
            # db4/db8/db12 are lossy DWT choices; lossless mode deliberately
            # uses the fixed reversible 5/3 transform instead.
            self._set_value("mode", "lossy")
        self._update_comparison_mode_options()
        self._transform_setting_changed()

    def _comparison_position_changed(self, value: int) -> None:
        """Refresh the split view without changing the selected comparison mode."""
        self.comparison_value_label.setText(f"{value}%")
        if combo_value(self.comparison_mode) == "Slider":
            self._refresh_comparison()

    def _transform_setting_changed(self, _value=None) -> None:
        """Keep scientific transform views aligned with their visible controls."""
        if combo_value(self.comparison_mode) in {"Transform tree", "Block grid"}:
            self._refresh_comparison()

    def _wavelet_setting_changed(self, _value=None) -> None:
        """Invalidate a cached db4/db8/db12 comparison after a relevant edit."""
        self.wavelet_comparison_results = []
        self._wavelet_comparison_signature = None
        if combo_value(self.comparison_mode) == "Wavelet comparison":
            self.wavelet_comparison_preview.show_message("Wavelet comparison is being processed.")
            self.comparison_stack.setCurrentWidget(self.wavelet_comparison_preview)
            self._wavelet_compare_timer.start()

    def _wavelet_comparison_options(self) -> tuple[tuple, dict]:
        if self.original_array is None:
            raise ValueError("Select or show an image before running the wavelet comparison.")
        if self._value("mode") != "lossy":
            raise ValueError("Wavelet comparison requires lossy mode because lossless mode uses the fixed reversible 5/3 transform.")
        image = np.asarray(self.original_array).copy()
        wavelets = ("db4", "db8", "db12")
        common_maximum = min(
            pywt.dwt_max_level(min(image.shape[:2]), pywt.Wavelet(name).dec_len)
            for name in wavelets
        )
        if common_maximum < 1:
            raise ValueError("The selected image is too small for a db4/db8/db12 comparison.")
        requested_level = int(self._value("level"))
        level = min(requested_level, common_maximum)
        target_bpp, target_psnr = self._selected_quality_targets()
        step = float(self._value("step"))
        quantizer = str(self._value("quantizer"))
        colorspace = str(self._value("colorspace"))
        roi_strength = float(self._value("roi_strength"))
        roi_mask = None if self.current_roi_mask is None else np.asarray(self.current_roi_mask, dtype=np.float32).copy()
        input_path = Path(self._value("input_path").strip())
        original_file_size = input_path.stat().st_size if input_path.is_file() else int(image.nbytes)
        signature = (
            id(self.original_array), tuple(image.shape), level, target_bpp, target_psnr, quantizer, colorspace,
            roi_strength, original_file_size,
            None if roi_mask is None else (roi_mask.shape, float(np.sum(roi_mask))),
        )
        return signature, {
            "image": image,
            "wavelets": wavelets,
            "level": level,
            "step": step,
            "target_bpp": target_bpp,
            "target_psnr": target_psnr,
            "quantizer": quantizer,
            "colorspace": colorspace,
            "roi_mask": roi_mask,
            "roi_strength": roi_strength,
            "original_file_size": original_file_size,
        }

    @staticmethod
    def _build_wavelet_comparison(options: dict, progress) -> list[dict]:
        image = np.asarray(options["image"])
        results: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="smartcodec_wavelets_") as temp_dir:
            for index, wavelet in enumerate(options["wavelets"], start=1):
                encoded_path = Path(temp_dir) / f"{wavelet}.swc"
                info = encode_array(
                    image,
                    encoded_path,
                    mode="lossy",
                    codec="dwt",
                    wavelet=wavelet,
                    level=int(options["level"]),
                    step=float(options["step"]),
                    quantizer=str(options["quantizer"]),
                    colorspace=str(options["colorspace"]),
                    roi_mask=options["roi_mask"],
                    roi_strength=float(options["roi_strength"]),
                    target_bpp=options["target_bpp"],
                    target_psnr=options["target_psnr"],
                )
                reconstructed = decode_array(encoded_path)
                results.append({
                    "wavelet": wavelet,
                    "image": reconstructed,
                    "level": int(options["level"]),
                    "step": float(info.get("step", options["step"])),
                    "mse": mse(image, reconstructed),
                    "psnr": psnr(image, reconstructed),
                    "ssim": ssim(image, reconstructed),
                    "bpp": float(info.get("bits_per_pixel", 0.0)),
                    "ratio": float(info.get("compression_ratio", 0.0)),
                    "file_size": int(info.get("file_size", 0)),
                    "original_file_size": int(options["original_file_size"]),
                })
                progress(index)
        return results

    def _ensure_wavelet_comparison(self) -> None:
        if combo_value(self.comparison_mode) != "Wavelet comparison":
            return
        if self.is_busy:
            self._wavelet_compare_timer.start(400)
            return
        language = str(self.property("ui_language") or "en")
        try:
            signature, options = self._wavelet_comparison_options()
        except ValueError as exc:
            self.wavelet_comparison_preview.show_message(str(exc), language)
            self.comparison_stack.setCurrentWidget(self.wavelet_comparison_preview)
            self._sync_fullscreen_previews()
            return
        if signature == self._wavelet_comparison_signature and self.wavelet_comparison_results:
            self._refresh_comparison()
            return
        self.wavelet_comparison_preview.show_message("Wavelet comparison is being processed.", language)
        self.comparison_stack.setCurrentWidget(self.wavelet_comparison_preview)
        self._sync_fullscreen_previews()

        def work(progress):
            return signature, self._build_wavelet_comparison(options, progress)

        self.run_background(work, self._wavelet_comparison_done, "Wavelet comparison", 3)

    def _wavelet_comparison_done(self, result: tuple[tuple, list[dict]]) -> None:
        signature, comparisons = result
        self._wavelet_comparison_signature = signature
        self.wavelet_comparison_results = comparisons
        if combo_value(self.comparison_mode) == "Wavelet comparison":
            self._refresh_comparison()
        self.statusChanged.emit("Wavelet comparison complete: db4, db8 and db12.")

    def _rate_distortion_options(self) -> tuple[tuple, dict]:
        """Freeze one source and each codec's settings for a shared-rate comparison."""
        if self.original_array is None or self.decoded_array is None or self.last_encode_info is None:
            raise ValueError("Run Encode + Decode to generate the PSNR–BPP graph.")
        if not self._last_encode_options or self._last_encode_options.get("mode") != "lossy":
            raise ValueError("PSNR–BPP graph requires lossy mode.")
        image = np.asarray(self.original_array).copy()
        current_image = np.asarray(self.decoded_array).copy()
        if image.shape != current_image.shape:
            raise ValueError("Run Encode + Decode to generate the PSNR–BPP graph.")
        # Match the codec's lossy input domain, including 16-bit TIFF sources.
        if image.dtype == np.uint16:
            image = np.rint(image.astype(np.float32) / 257.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        info = dict(self.last_encode_info)
        snapshot = dict(self._last_encode_options)
        roi_mask = snapshot.get("roi_mask")
        if roi_mask is not None:
            roi_mask = np.asarray(roi_mask, dtype=np.float32).copy()
        methods = {name: dict(values) for name, values in snapshot["rd_method_settings"].items()}
        anchor = round(max(0.01, float(snapshot.get("target_bpp") or info.get("bits_per_pixel") or 0.8)), 3)
        targets = sorted(set((0.1, 0.2, 0.4, 0.8, 1.2, 2.0, anchor)))
        signature = (
            id(self.original_array), id(self.decoded_array), "dct+dwt", tuple(image.shape),
            tuple(targets), anchor,
            tuple((name, tuple(sorted(values.items()))) for name, values in sorted(methods.items())),
        )
        return signature, {
            "image": image,
            "methods": methods,
            "targets": targets,
            "anchor_bpp": anchor,
            "roi_mask": roi_mask,
            "roi_strength": float(snapshot.get("roi_strength", 0.65)),
        }

    @staticmethod
    def _build_rate_distortion(options: dict, progress) -> list[dict]:
        """Measure DCT and DWT independently at the same requested BPP values.

        Plot achieved file rates, never substitute the requested target. The
        existing target search chooses a separate quantization step per codec.
        Neither the main decoded image nor its saved output is replaced.
        """
        image = np.asarray(options["image"])
        results: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="smartcodec_rate_curve_") as temp_dir:
            for target in options["targets"]:
                for codec in ("dct", "dwt"):
                    settings = options["methods"][codec]
                    encoded_path = Path(temp_dir) / f"{codec}_{len(results)}.swc"
                    info = encode_array(
                        image,
                        encoded_path,
                        mode="lossy",
                        codec=codec,
                        wavelet=str(settings.get("wavelet", "haar")),
                        level=int(settings.get("level", 2)),
                        quantizer=str(settings.get("quantizer", "uniform")),
                        colorspace=str(settings["colorspace"]),
                        target_bpp=float(target),
                        roi_mask=options["roi_mask"],
                        roi_strength=float(options["roi_strength"]),
                        compact_header=True,
                    )
                    reconstructed = decode_array(encoded_path)
                    size = encoded_path.stat().st_size
                    detail = (f"{settings['wavelet']} / L{settings['level']} / {settings['quantizer']}"
                              if codec == "dwt" else "8×8")
                    results.append({
                        "codec": codec,
                        "bpp": size * 8.0 / (image.shape[0] * image.shape[1]),
                        "psnr": float(psnr(image, reconstructed)),
                        "file_size": size,
                        "parameter": f"{detail} / {settings['colorspace'].upper()}",
                        "step": float(info.get("step", 0)),
                        "target_bpp": float(target),
                        "highlighted": target == options["anchor_bpp"],
                    })
                    progress(len(results))
        return results

    def _ensure_rate_distortion(self) -> None:
        if combo_value(self.comparison_mode) != "PSNR–BPP graph":
            return
        if self.is_busy:
            self._rate_distortion_timer.start(400)
            return
        language = str(self.property("ui_language") or "en")
        try:
            signature, options = self._rate_distortion_options()
        except ValueError as exc:
            self.rate_distortion_preview.show_message(str(exc), language)
            self.comparison_stack.setCurrentWidget(self.rate_distortion_preview)
            self._sync_fullscreen_previews()
            return
        if signature == self._rate_distortion_signature and self.rate_distortion_results:
            self._refresh_comparison()
            return
        self.rate_distortion_preview.show_message("PSNR–BPP graph is being processed.", language)
        self.comparison_stack.setCurrentWidget(self.rate_distortion_preview)
        self._sync_fullscreen_previews()

        def work(progress):
            return signature, "DCT / DWT", self._build_rate_distortion(options, progress)

        total = 2 * len(options["targets"])
        self.run_background(work, self._rate_distortion_done, "PSNR–BPP graph", total)

    def _rate_distortion_done(self, result: tuple[tuple, str, list[dict]]) -> None:
        signature, codec, points = result
        # A new image can arrive while a background measurement is finishing.
        # Never attach that old image's points to the newly selected image.
        if signature[:2] != (id(self.original_array), id(self.decoded_array)):
            return
        self._rate_distortion_signature = signature
        self.rate_distortion_codec = codec
        self.rate_distortion_results = points
        if combo_value(self.comparison_mode) == "PSNR–BPP graph":
            self._refresh_comparison()
        self.statusChanged.emit("PSNR–BPP graph complete.")

    def _fit_comparison_view(self) -> None:
        """Fit the active comparison renderer without changing selected mode."""
        mode = combo_value(self.comparison_mode)
        if mode == "Transform tree":
            self.transform_preview.fit_to_view()
        elif mode == "Block grid":
            self.transform_grid_preview.fit_to_view()
        elif mode == "Wavelet comparison":
            self.wavelet_comparison_preview.fit_to_view()
        elif mode == "Histogram":
            self.histogram_preview.fit_to_view()
        elif mode == "PSNR–BPP graph":
            self.rate_distortion_preview.fit_to_view()
        elif mode == "8-bit grayscale":
            self.grayscale_preview.fit_to_view()
        else:
            self.comparison_preview.fit_to_view()

    def _line(self, name: str, form: QtWidgets.QFormLayout, label: str, value: str = "", button_text: str | None = None, callback=None) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value)
        edit.setObjectName(name)
        self._fields[name] = edit
        if button_text and callback:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(edit, 1)
            button = QtWidgets.QPushButton(button_text)
            button.clicked.connect(callback)
            row.addWidget(button)
            form.addRow(label, row)
        else:
            form.addRow(label, edit)
        return edit

    def _combo(self, name: str, form: QtWidgets.QFormLayout, label: str, values: tuple[str, ...], current: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(values)
        configure_combo_popup(combo)
        combo.setCurrentText(current)
        self._fields[name] = combo
        form.addRow(label, combo)
        return combo

    def _spin(self, name: str, form: QtWidgets.QFormLayout, label: str, value: float, minimum: float, maximum: float, integer: bool = False) -> QtWidgets.QAbstractSpinBox:
        if integer:
            widget: QtWidgets.QAbstractSpinBox = QtWidgets.QSpinBox()
            widget.setRange(int(minimum), int(maximum))
            widget.setValue(int(value))
        else:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(float(minimum), float(maximum))
            widget.setDecimals(3)
            widget.setSingleStep(0.5)
            widget.setValue(float(value))
        self._fields[name] = widget
        form.addRow(label, widget)
        return widget

    def _optional_spin(self, name: str, form: QtWidgets.QFormLayout, label: str, unit: str,
                       maximum: float) -> QtWidgets.QDoubleSpinBox:
        """Create an optional numeric target with a visible unit label."""
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(0.0, float(maximum))
        widget.setDecimals(3)
        widget.setSingleStep(0.1)
        widget.setSpecialValueText("")
        # Keep target inputs compact while leaving their unit visually separate.
        widget.setFixedWidth(105)
        widget.setToolTip("Leave empty to disable this automatic target.")
        unit_label = QtWidgets.QLabel(unit)
        unit_label.setObjectName(f"{name}Unit")
        row_container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(row_container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget, 1)
        row.addWidget(unit_label)
        self._fields[name] = widget
        form.addRow(label, row_container)
        self._optional_field_rows[name] = (row_container, form.labelForField(row_container))
        return widget

    def _check(self, name: str, form: QtWidgets.QFormLayout, label: str, value: bool = False) -> QtWidgets.QCheckBox:
        widget = QtWidgets.QCheckBox(label)
        widget.setChecked(value)
        self._fields[name] = widget
        form.addRow(widget)
        return widget

    def _build_files(self, layout: QtWidgets.QVBoxLayout) -> None:
        form = self._group(layout, "1. Files")
        self._line("input_path", form, "Input image", button_text="Browse", callback=self._choose_input)
        encoded_path = self._line("encoded_path", form, "Encoded output", button_text="Save", callback=self._choose_encoded)
        decoded_path = self._line("decoded_path", form, "Decoded image", button_text="Save", callback=self._choose_decoded)
        for field in (encoded_path, decoded_path):
            field.setReadOnly(True)
            field.setPlaceholderText("—")

    def _build_codec_settings(self, layout: QtWidgets.QVBoxLayout) -> None:
        form = self._group(layout, "2. Compression settings")
        codec_method = self._combo("codec_method", form, "Output format", ("jpeg", "jpeg2000", "dwt", "dct", "prqmf4"), "jpeg")
        codec_method.currentTextChanged.connect(self._codec_method_changed)
        codec_method.setToolTip("Choose JPEG or JPEG2000 for a standard image file. DWT/DCT options also expose the transform tree for any loaded image.")
        form.addRow(QtWidgets.QLabel("JPEG/JPEG2000 produces standard files; DWT/DCT/PR-QMF are Smart Codec transform methods."))
        mode = self._combo("mode", form, "Mode", ("lossy", "lossless"), "lossy")
        mode.currentTextChanged.connect(lambda _text: self._mode_changed(str(self._value("mode"))))
        standard_quality = self._spin("standard_quality", form, "JPEG quality (1-100)", 75, 1, 100, integer=True)
        standard_quality.setToolTip("Lower values create a smaller JPEG file with more visible quality loss. Use 30 for a clear presentation demo.")
        standard_rate = self._spin("standard_rate", form, "JPEG2000 rate", 4.0, 0.01, 100.0)
        standard_rate.setToolTip("Higher rates apply stronger JPEG2000 compression. This setting is used only when JPEG2000 is selected.")
        # These controls are meaningful only for the corresponding standard
        # codecs; keep their labels and fields hidden for DWT/DCT/PR-QMF.
        self._standard_codec_fields = (standard_quality, standard_rate)
        self._standard_codec_labels = tuple(form.labelForField(widget) for widget in self._standard_codec_fields)
        self._advanced_codec_group = QtWidgets.QGroupBox("Wavelet-based Smart Codec settings")
        self._advanced_codec_group.setObjectName("controlCard")
        advanced = QtWidgets.QFormLayout(self._advanced_codec_group)
        advanced.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        wavelet = self._combo(
            "wavelet", advanced, "Wavelet",
            ("haar", "db4", "db8", "db12", "qmf", WAVELET_COMPARISON_SELECTION),
            "haar",
        )
        level = self._spin("level", advanced, "DWT level", 2, 1, 8, integer=True)
        # Step remains an internal codec parameter for file/preset compatibility,
        # but target search now selects it automatically and it is no longer a
        # user-facing compression control.
        step = QtWidgets.QDoubleSpinBox()
        step.setRange(0.01, 65536.0)
        step.setValue(12.0)
        self._fields["step"] = step
        quantizer = self._combo("quantizer", advanced, "Quantizer", ("uniform", "scalar"), "uniform")
        self._combo("colorspace", advanced, "Color space", ("ycbcr", "rgb"), "ycbcr")
        quality_target = self._combo("quality_target", advanced, "Quality target", ("BPP", "PSNR"), "BPP")
        quality_target.setToolTip("Choose one automatic compression target. BPP and PSNR cannot be active together.")
        self._quality_target_field = quality_target
        self._quality_target_label = advanced.labelForField(quality_target)
        target_bpp = self._optional_spin("target_bpp", advanced, "Target BPP", "BPP", 1000.0)
        target_bpp.setValue(1.0)
        target_bpp.setToolTip("Set the requested encoded bits per image pixel.")
        target_psnr = self._optional_spin("target_psnr", advanced, "Target PSNR", "dB", 100.0)
        target_psnr.setValue(30.0)
        target_psnr.setToolTip("Set the minimum requested reconstruction quality in decibels.")
        quality_target.currentTextChanged.connect(self._quality_target_changed)
        # DCT has no wavelet family, decomposition level, or wavelet-band
        # quantizer. Keep those rows available for the codecs that actually
        # consume them and remove them completely from the DCT form.
        self._wavelet_codec_fields = {
            "wavelet": wavelet,
            "level": level,
            "quantizer": quantizer,
        }
        self._wavelet_codec_labels = {
            name: advanced.labelForField(widget)
            for name, widget in self._wavelet_codec_fields.items()
        }
        layout.addWidget(self._advanced_codec_group)
        self._codec_method_changed(codec_method.currentText())

    def _build_ai_roi(self) -> None:
        """Build the shared AI/ROI controls as their own top-level page.

        The widgets remain owned by ``CodecTab`` logically: encoding, preset
        compatibility, and settings persistence still read the same entries in
        ``self._fields``.  Only their visual location changes.
        """
        self.ai_roi_page = QtWidgets.QWidget(self)
        self.ai_roi_page.setObjectName("aiRoiPage")
        page_layout = QtWidgets.QVBoxLayout(self.ai_roi_page)
        page_layout.setContentsMargins(28, 24, 28, 24)
        page_layout.setSpacing(12)

        self.ai_roi_group = QtWidgets.QGroupBox("AI, ROI and restoration", self.ai_roi_page)
        self.ai_roi_group.setObjectName("controlCard")
        form = QtWidgets.QFormLayout(self.ai_roi_group)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        introduction = QtWidgets.QLabel("First select an image in Encode / Decode. Then enable restoration or choose one ROI method below.")
        introduction.setWordWrap(True)
        form.addRow(introduction)
        ai_reconstruction = self._check("ai_reconstruction", form, "Use AI restoration (TorchScript)")
        ai_reconstruction.setToolTip("Optional: creates a separate AI-restored image after decoding; it never replaces the actual decoded image.")

        method_label = QtWidgets.QLabel("Protect an important area (optional)")
        method_label.setObjectName("metricTitle")
        form.addRow(method_label)
        roi_buttons = QtWidgets.QGridLayout()
        draw = QtWidgets.QPushButton("Draw an area")
        draw.clicked.connect(self._begin_roi_drawing)
        draw.setToolTip("Draw a rectangle over the Original image to mark an important region.")
        yolo = QtWidgets.QPushButton("Find objects automatically")
        yolo.clicked.connect(self._detect_yolo)
        yolo.setToolTip("Find people, vehicles and other objects automatically and use them as ROI regions.")
        faces = QtWidgets.QPushButton("Find faces")
        faces.clicked.connect(self._detect_faces)
        faces.setToolTip("Detect faces and use them as ROI regions.")
        mask = QtWidgets.QPushButton("Choose a mask image")
        mask.clicked.connect(self._choose_roi_mask)
        mask.setToolTip("Use a black-and-white image to specify the important area.")
        clear = QtWidgets.QPushButton("Clear ROI")
        clear.clicked.connect(self._clear_roi)
        roi_buttons.addWidget(draw, 0, 0)
        roi_buttons.addWidget(yolo, 0, 1)
        roi_buttons.addWidget(faces, 1, 0)
        roi_buttons.addWidget(mask, 1, 1)
        roi_buttons.addWidget(clear, 2, 0, 1, 2)
        form.addRow(roi_buttons)

        self.roi_summary = QtWidgets.QLabel("No ROI selected. The whole image will use the same quality.")
        self.roi_summary.setWordWrap(True)
        self.roi_summary.setObjectName("metricsDetails")
        form.addRow(self.roi_summary)

        self.ai_roi_advanced_toggle = QtWidgets.QToolButton()
        self.ai_roi_advanced_toggle.setText("Advanced ROI settings")
        self.ai_roi_advanced_toggle.setCheckable(True)
        self.ai_roi_advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.ai_roi_advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.ai_roi_advanced_toggle.toggled.connect(self._set_ai_roi_advanced_visible)
        form.addRow(self.ai_roi_advanced_toggle)

        self.ai_roi_advanced_group = QtWidgets.QGroupBox()
        advanced = QtWidgets.QFormLayout(self.ai_roi_advanced_group)
        advanced.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._line("roi_boxes", advanced, "ROI boxes", "")
        self._line("roi_mask_path", advanced, "ROI mask", button_text="Browse", callback=self._choose_roi_mask)
        self._spin("roi_strength", advanced, "ROI strength", 0.65, 0.0, 0.95)
        self._spin("roi_feather", advanced, "Feather", 0, 0, 64, integer=True)
        self.ai_roi_advanced_group.setVisible(False)
        form.addRow(self.ai_roi_advanced_group)
        page_layout.addWidget(self.ai_roi_group)
        page_layout.addStretch(1)

    def _set_ai_roi_advanced_visible(self, visible: bool) -> None:
        self.ai_roi_advanced_group.setVisible(visible)
        arrow = QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        self.ai_roi_advanced_toggle.setArrowType(arrow)

    def _update_roi_summary(self) -> None:
        """Keep the beginner-facing ROI summary in sync with advanced values."""
        if not hasattr(self, "roi_summary"):
            return
        try:
            count = len(self._parse_roi_boxes())
        except (TypeError, ValueError):
            count = 0
        has_mask = bool(self._value("roi_mask_path").strip())
        language = str(self.property("ui_language") or "en")
        if language == "tr":
            if count and has_mask:
                text = f"ROI etkin: {count} önemli alan ve bir maske görüntüsü."
            elif count:
                text = f"ROI etkin: {count} önemli alan."
            elif has_mask:
                text = "ROI etkin: maske görüntüsü seçildi."
            else:
                text = "ROI seçilmedi. Tüm görüntü aynı kaliteyi kullanacak."
        elif count and has_mask:
            text = f"ROI active: {count} area(s) and one mask image."
        elif count:
            text = f"ROI active: {count} important area(s)."
        elif has_mask:
            text = "ROI active: mask image selected."
        else:
            text = "No ROI selected. The whole image will use the same quality."
        self.roi_summary.setText(text)

    def _begin_roi_drawing(self) -> None:
        """Open the Original workspace where rectangle selection is available."""
        self.preview_tabs.setCurrentWidget(self.original_preview)
        self.workspaceRequested.emit()
        self.statusChanged.emit("Drag on the Original tab to draw an ROI.")

    def _build_transport(self, layout: QtWidgets.QVBoxLayout) -> None:
        # Detailed transport controls are kept in the dedicated Video / Transport tab.
        # The image workflow stays focused on encoding and visual comparison.
        return

    def _build_actions(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("4. Actions")
        group.setObjectName("controlCard")
        row = QtWidgets.QGridLayout(group)
        actions = (("Show image", self._preview_input), ("Encode", self._encode), ("Decode", self._decode), ("Encode + Decode", self._encode_decode), ("Batch", self._batch_menu))
        for index, (text, callback) in enumerate(actions):
            button = QtWidgets.QPushButton(text)
            if text == "Encode + Decode":
                button.setProperty("primary", True)
            button.clicked.connect(callback)
            if text == "Encode + Decode":
                button.setToolTip("Run compression and decoding together, then fill every comparison panel. This is the recommended presentation button.")
            row.addWidget(button, index // 3, index % 3)
        layout.addWidget(group)

    # ------------------------------------------------------------------
    # TR: Value helpers / EN: Value helpers
    # ------------------------------------------------------------------
    def _value(self, name: str):
        widget = self._fields[name]
        if isinstance(widget, QtWidgets.QComboBox):
            return combo_value(widget)
        if isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QtWidgets.QDoubleSpinBox):
            if name in {"target_bpp", "target_psnr"} and widget.value() <= 0:
                return ""
            return widget.value()
        if isinstance(widget, QtWidgets.QSpinBox):
            return widget.value()
        return widget.text()

    @staticmethod
    def _encoded_suffix(codec: str) -> str:
        return ".jpg" if codec == "jpeg" else ".jp2" if codec == "jpeg2000" else ".swc"

    def _session_encoded_path(self, codec: str) -> Path:
        """Return an auto-cleaned working path that is never presented as a saved output."""
        return Path(self._session_output_dir.path()) / f"encoded{self._encoded_suffix(codec)}"

    def _store_codec_settings(self, codec: str | None) -> None:
        """Capture the visible draft without mixing it into another codec."""
        if codec not in self._codec_setting_fields:
            return
        state = self._codec_settings_by_method.setdefault(codec, {})
        for name in self._codec_setting_fields[codec]:
            if name in self._fields:
                state[name] = self._value(name)

    def _restore_codec_settings(self, codec: str) -> None:
        """Restore a codec draft atomically, without exposing partial UI state."""
        state = self._codec_settings_by_method.get(codec, {})
        blockers: list[QtCore.QSignalBlocker] = []
        previous_suspend = self._suspend_codec_setting_sync
        self._suspend_codec_setting_sync = True
        try:
            for name in self._codec_setting_fields.get(codec, ()):
                if name in state and name in self._fields:
                    blockers.append(QtCore.QSignalBlocker(self._fields[name]))
                    self._set_value(name, state[name])
        finally:
            # Destroy blockers before restoring normal synchronization.  Qt's
            # blocker destructor restores each widget's prior signal state.
            blockers.clear()
            self._suspend_codec_setting_sync = previous_suspend

    def _selected_quality_targets(self) -> tuple[float | None, float | None]:
        """Return exactly one active lossy target for the codec backend."""
        if (str(self._value("mode")) == "lossless"
                or str(self._value("codec_method")).lower() in {"jpeg", "jpeg2000"}):
            return None, None
        target = str(self._value("quality_target")).upper()
        if target == "PSNR":
            value = float(self._value("target_psnr") or 30.0)
            return None, value
        value = float(self._value("target_bpp") or 1.0)
        return value, None

    def _quality_target_changed(self, _target: str | None = None) -> None:
        """Show one target input and make the other target impossible to submit."""
        if not hasattr(self, "_quality_target_field"):
            return
        codec = str(self._value("codec_method")).lower()
        available = codec not in {"jpeg", "jpeg2000"} and str(self._value("mode")) != "lossless"
        selected = str(self._value("quality_target")).upper()
        if selected == "PSNR" and float(self._fields["target_psnr"].value()) <= 0:
            self._fields["target_psnr"].setValue(30.0)
        elif selected != "PSNR" and float(self._fields["target_bpp"].value()) <= 0:
            self._fields["target_bpp"].setValue(1.0)
        self._quality_target_field.setVisible(available)
        if self._quality_target_label is not None:
            self._quality_target_label.setVisible(available)
        for name in ("target_bpp", "target_psnr"):
            row, label = self._optional_field_rows[name]
            visible = available and ((name == "target_psnr") == (selected == "PSNR"))
            row.setVisible(visible)
            if label is not None:
                label.setVisible(visible)

    def _codec_method_changed(self, codec: str) -> None:
        """Keep the generated file type and visible settings aligned with the codec."""
        codec = str(codec).lower()
        if not self._suspend_codec_setting_sync and codec != self._active_codec:
            self._store_codec_settings(self._active_codec)
            self._restore_codec_settings(codec)
        self._active_codec = codec
        if hasattr(self, "_advanced_codec_group"):
            self._advanced_codec_group.setVisible(codec not in {"jpeg", "jpeg2000"})
        if hasattr(self, "_wavelet_codec_fields"):
            wavelet_visibility = {
                "wavelet": codec == "dwt",
                "level": codec in {"dwt", "prqmf4"},
                "quantizer": codec in {"dwt", "prqmf4"},
            }
            for name, visible in wavelet_visibility.items():
                self._wavelet_codec_fields[name].setVisible(visible)
                label = self._wavelet_codec_labels.get(name)
                if label is not None:
                    label.setVisible(visible)
        if hasattr(self, "_standard_codec_fields"):
            jpeg_visible = codec == "jpeg"
            jpeg2000_visible = codec == "jpeg2000"
            self._standard_codec_fields[0].setVisible(jpeg_visible)
            self._standard_codec_fields[1].setVisible(jpeg2000_visible)
            self._standard_codec_labels[0].setVisible(jpeg_visible)
            self._standard_codec_labels[1].setVisible(jpeg2000_visible)
        self._update_comparison_mode_options()
        self._mode_changed(str(self._value("mode")))
        if hasattr(self, "comparison_mode") and combo_value(self.comparison_mode) in {"Transform tree", "Block grid"}:
            self._refresh_comparison()

    def _set_value(self, name: str, value) -> None:
        widget = self._fields.get(name)
        if widget is None:
            return
        if isinstance(widget, QtWidgets.QComboBox):
            set_combo_value(widget, value)
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            widget.setValue(0.0 if value in {None, ""} else float(value))
        elif isinstance(widget, QtWidgets.QLineEdit):
            widget.setText(str(value))
            widget.setCursorPosition(0)

    def _set_metrics_text(self, text: str) -> None:
        """Keep the source metrics text so it can be relocalized on a language switch."""
        self.metrics_label.setProperty("source_text", text)
        self.metrics_label.setText(translate_status(text, str(self.property("ui_language") or "en")))

    def _set_metric_cards(self, psnr_value: float | None = None, ssim_value: float | None = None,
                          bpp_value: float | None = None, ratio_value: float | None = None,
                          original_size: int | float | None = None,
                          encoded_size: int | float | None = None) -> None:
        """Show key measurement values as stable, scannable result cards."""
        values = {
            "psnr": "—" if psnr_value is None else f"{psnr_value:.2f} dB",
            "ssim": "—" if ssim_value is None else f"{ssim_value:.4f}",
            "bpp": "—" if bpp_value is None else f"{bpp_value:.3f}",
            "ratio": "—" if ratio_value is None else f"{ratio_value:.2f}×",
            "original_size": format_file_size(original_size),
            "encoded_size": format_file_size(encoded_size),
        }
        for key, value in values.items():
            self.metric_values[key].setText(value)
        dialog = self._fullscreen_dialog
        if dialog is not None and dialog.isVisible():
            dialog.sync_metrics(values)

    def settings_payload(self) -> dict:
        """TR: Ayarlari JSON'a yazilabilir sozluge donusturur.
        EN: Convert settings to a JSON-serializable dictionary.
        """
        self._store_codec_settings(self._active_codec)
        payload = {name: self._value(name) for name in self._fields}
        payload["codec_settings_by_method"] = {
            codec: dict(values) for codec, values in self._codec_settings_by_method.items()
        }
        payload["comparison_mode"] = combo_value(self.comparison_mode)
        payload["comparison_target"] = combo_value(self.comparison_target)
        payload["comparison_position"] = self.comparison_position.value() / 100.0
        return payload

    def apply_settings(self, values: dict) -> None:
        saved_codec_settings = values.get("codec_settings_by_method")
        if isinstance(saved_codec_settings, dict):
            for codec, saved_state in saved_codec_settings.items():
                if codec not in self._codec_setting_fields or not isinstance(saved_state, dict):
                    continue
                allowed = self._codec_setting_fields[codec]
                self._codec_settings_by_method[codec].update(
                    {name: value for name, value in saved_state.items() if name in allowed}
                )
                if "quality_target" not in saved_state:
                    try:
                        old_psnr = float(saved_state.get("target_psnr") or 0)
                    except (TypeError, ValueError):
                        old_psnr = 0.0
                    self._codec_settings_by_method[codec]["quality_target"] = "PSNR" if old_psnr > 0 else "BPP"
        self._suspend_codec_setting_sync = True
        try:
            for name, value in values.items():
                if name in self._fields:
                    if name == "profile":
                        value = {
                            "Ozel": "Custom",
                            "Özel": "Custom",
                            "Yuksek kalite": "High quality",
                            "Yüksek kalite": "High quality",
                            "Maksimum sikistirma": "Maximum compression",
                            "Maksimum sıkıştırma": "Maximum compression",
                            "Kayipsiz tibbi": "Lossless medical",
                            "Kayıpsız tıbbi": "Lossless medical",
                            "ROI nesne": "ROI object",
                        }.get(str(value), value)
                    self._set_value(name, value)
        finally:
            self._suspend_codec_setting_sync = False
        if "quality_target" not in values:
            try:
                old_psnr = float(values.get("target_psnr") or 0)
            except (TypeError, ValueError):
                old_psnr = 0.0
            self._set_value("quality_target", "PSNR" if old_psnr > 0 else "BPP")
        self._active_codec = str(self._value("codec_method")).lower()
        self._store_codec_settings(self._active_codec)
        self._quality_target_changed()
        if "comparison_mode" in values:
            mode = {
                "Yan yana": "Side-by-side",
                "Kaydırıcı": "Slider",
                "Aşamalı aşamalar": "Progressive stages",
                "Dönüşüm ağacı": "Transform tree",
                "Blok ızgarası": "Block grid",
                "Dalgacık karşılaştırması": "Wavelet comparison",
                "Hata ısı haritası": "Error heatmap",
            }.get(str(values["comparison_mode"]), str(values["comparison_mode"]))
            set_combo_value(self.comparison_mode, mode)
        if "comparison_target" in values:
            target = {"Çözülmüş": "Decoded", "Restorasyon": "Restored"}.get(
                str(values["comparison_target"]), str(values["comparison_target"])
            )
            set_combo_value(self.comparison_target, target)
        if "comparison_position" in values:
            raw_position = float(values["comparison_position"])
            # Old settings stored a fraction; accept a direct percentage too.
            position = raw_position * 100.0 if 0.0 <= raw_position <= 1.0 else raw_position
            self.comparison_position.setValue(int(max(0, min(100, position))))
        self._update_roi_summary()

    def codec_preset(self) -> dict:
        names = tuple(name for name in self._fields if name not in {"input_path", "encoded_path", "decoded_path"})
        return {name: self._value(name) for name in names}

    # ------------------------------------------------------------------
    # TR: File selection and ROI / EN: File selection and ROI
    # ------------------------------------------------------------------
    def _choose_input(self) -> None:
        path = browse_file(self, "Select image", IMAGE_FILTER)
        if path:
            self._set_value("input_path", path)
            self._set_value("encoded_path", "")
            self._set_value("decoded_path", "")
            self._latest_encoded_path = None
            self._preview_input()

    def _choose_encoded(self) -> None:
        source = self._latest_encoded_path
        if source is None or not source.is_file():
            self._show_error("Run Encode or Encode + Decode before saving the encoded output.")
            return
        codec = self._value("codec_method")
        suffix = source.suffix or self._encoded_suffix(codec)
        path = browse_file(self, "Save encoded output", f"Output (*{suffix});;All files (*.*)", save=True, suffix=suffix)
        if path:
            try:
                shutil.copy2(source, path)
            except OSError as exc:
                self._show_error(str(exc))
                return
            self._set_value("encoded_path", path)
            self.statusChanged.emit(f"Encoded output saved: {path}")

    def _choose_decoded(self) -> None:
        if self.decoded_array is None:
            self._show_error("Run Decode or Encode + Decode before saving the decoded image.")
            return
        path = browse_file(self, "Save decoded image", "PNG (*.png);;TIFF (*.tif *.tiff);;BMP (*.bmp)", save=True, suffix=".png")
        if path:
            try:
                save_image(self.decoded_array, path)
            except OSError as exc:
                self._show_error(str(exc))
                return
            self._set_value("decoded_path", path)
            self.statusChanged.emit(f"Decoded image saved: {path}")

    def _choose_roi_mask(self) -> None:
        path = browse_file(self, "Select ROI mask", IMAGE_FILTER)
        if path:
            self._set_value("roi_mask_path", path)
            self._update_roi_summary()

    def _preview_input(self) -> None:
        path = self._value("input_path").strip()
        if not path or not Path(path).is_file():
            QtWidgets.QMessageBox.information(self, "Image", "Select a valid input image first.")
            return
        try:
            self.original_array = load_image(path)
            self.decoded_array = None
            self.restored_array = None
            self.last_encode_info = None
            self.progressive_stages = []
            self.progressive_labels = []
            self.wavelet_comparison_results = []
            self._wavelet_comparison_signature = None
            self._wavelet_compare_timer.stop()
            self.rate_distortion_results = []
            self.rate_distortion_codec = ""
            self._rate_distortion_signature = None
            self._last_encode_options = None
            self._rate_distortion_timer.stop()
            self._latest_encoded_path = None
            self._set_value("encoded_path", "")
            self._set_value("decoded_path", "")
            self.current_roi_mask = None
            self.semantic_roi_mask = None
            self.original_preview.set_array(self.original_array)
            self.decoded_preview.clear_image()
            self.restored_preview.clear_image()
            self.difference_preview.clear_image()
            self.preview_tabs.setCurrentWidget(self.original_preview)
            self._set_metric_cards()
            self._set_metrics_text("Image loaded. You can now encode or decode.")
            self._refresh_comparison()
            if self._fullscreen_dialog is not None:
                self._fullscreen_dialog.tabs.setCurrentWidget(self._fullscreen_dialog.original_preview)
            self.statusChanged.emit(f"Image loaded: {Path(path).name} - {self.original_array.shape}")
        except Exception as exc:
            self._show_error(str(exc))

    def _parse_roi_boxes(self) -> list[tuple[int, int, int, int]]:
        raw = self._value("roi_boxes").strip()
        if not raw:
            return []
        boxes = []
        for item in raw.split(";"):
            values = tuple(int(part.strip()) for part in item.split(","))
            if len(values) != 4:
                raise ValueError("ROI kutusu x,y,w,h biciminde olmali")
            boxes.append(values)
        return boxes

    def _roi_selected(self, x: int, y: int, width: int, height: int) -> None:
        existing = self._value("roi_boxes").strip()
        item = ",".join(map(str, (x, y, width, height)))
        self._set_value("roi_boxes", f"{existing};{item}" if existing else item)
        self.original_preview.set_roi_boxes(self._parse_roi_boxes())
        self._update_roi_summary()
        self.statusChanged.emit(f"ROI added: {item}")

    def _clear_roi(self) -> None:
        self._set_value("roi_boxes", "")
        self._set_value("roi_mask_path", "")
        self.current_roi_mask = None
        self.semantic_roi_mask = None
        self.original_preview.set_roi_boxes([])
        self._update_roi_summary()
        self.statusChanged.emit("ROI cleared.")

    def _build_roi_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        if self._value("mode") == "lossless":
            return None
        mask = None
        if self.semantic_roi_mask is not None and tuple(self.semantic_roi_mask.shape) == tuple(shape):
            mask = np.asarray(self.semantic_roi_mask, dtype=np.float32).copy()
        mask_path = self._value("roi_mask_path").strip()
        if mask_path:
            mask = load_mask(mask_path, shape)
        boxes = self._parse_roi_boxes()
        if boxes:
            generated = boxes_to_mask(shape, boxes)
            mask = generated if mask is None else np.maximum(mask, generated)
        return mask

    def _detect_yolo(self) -> None:
        path = self._value("input_path").strip()
        if not Path(path).is_file():
            self._show_error("Select an image first.")
            return
        self.run_background(lambda _progress: analyze_scene(path, FIXED_YOLO_MODEL), self._yolo_done, "YOLO ROI")

    def _yolo_done(self, detections: list[dict]) -> None:
        boxes = [item["box"] for item in detections]
        self._set_value("roi_boxes", ";".join(",".join(map(str, box)) for box in boxes))
        if self.original_array is not None:
            self.semantic_roi_mask = detections_to_mask(self.original_array.shape[:2], detections)
            self.current_roi_mask = self._build_roi_mask(self.original_array.shape[:2])
        self.original_preview.set_roi_boxes(boxes)
        self._update_roi_summary()
        labels = ", ".join(f"{item['label']}:{item['confidence']:.2f}" for item in detections[:5])
        segmented = sum("mask" in item for item in detections)
        self.statusChanged.emit(f"YOLO found {len(boxes)} ROI boxes ({segmented} segmentation masks). {labels}")

    def _detect_faces(self) -> None:
        path = self._value("input_path").strip()
        if not Path(path).is_file():
            self._show_error("Select an image first.")
            return
        self.run_background(lambda _progress: detect_faces(path), self._faces_done, "Face ROI")

    def _faces_done(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self._set_value("roi_boxes", ";".join(",".join(map(str, box)) for box in boxes))
        self.original_preview.set_roi_boxes(boxes)
        self._update_roi_summary()
        self.statusChanged.emit(f"Face detection found {len(boxes)} ROI boxes.")

    # ------------------------------------------------------------------
    # TR: Encode/decode operations / EN: Encode/decode operations
    # ------------------------------------------------------------------
    def _encode_options(self) -> dict:
        self.progressive_stages = []
        self.progressive_labels = []
        self.wavelet_comparison_results = []
        self._wavelet_comparison_signature = None
        self.rate_distortion_results = []
        self.rate_distortion_codec = ""
        self._rate_distortion_signature = None
        self._last_encode_options = None
        input_path = self._value("input_path").strip()
        if not Path(input_path).is_file():
            raise ValueError("Select a valid input image")
        codec = self._value("codec_method")
        output_path = str(self._session_encoded_path(codec))
        # A fresh run invalidates previously displayed save destinations.  The
        # encoded artifact itself remains only in the auto-cleaned session
        # directory until the user explicitly presses Save.
        self._set_value("encoded_path", "")
        self._set_value("decoded_path", "")
        self._latest_encoded_path = None
        image = load_image(input_path)
        self.original_array = image
        self.decoded_array = None
        self.restored_array = None
        self.last_encode_info = None
        self.current_roi_mask = self._build_roi_mask(image.shape[:2])
        self.original_preview.set_array(image, self._parse_roi_boxes())
        self.decoded_preview.clear_image()
        self.restored_preview.clear_image()
        self.difference_preview.clear_image()
        self.comparison_preview.clear_image()
        self._set_metrics_text("Waiting for encoding to finish…")
        target_bpp, target_psnr = self._selected_quality_targets()
        options = {
            "input_path": input_path, "output_path": output_path, "image": image,
            "original_file_size": Path(input_path).stat().st_size,
            "mode": self._value("mode"), "codec": codec,
            "wavelet": self._effective_wavelet(), "level": int(self._value("level")),
            "step": float(self._value("step")), "quantizer": self._value("quantizer"),
            "colorspace": self._value("colorspace"), "roi_mask": self.current_roi_mask,
            "roi_strength": float(self._value("roi_strength")),
            "target_bpp": target_bpp,
            "target_psnr": target_psnr,
            "roi_feather": int(self._value("roi_feather")),
            "standard_quality": int(self._value("standard_quality")),
            "standard_rate": float(self._value("standard_rate")),
            "transport_segments": False,
            "transport_tiles": False,
            "transport_tile_size": 64,
        }
        # Keep an immutable-enough snapshot for the PSNR–BPP curve.  Later UI
        # edits must not mix another codec's controls into the completed run.
        self._last_encode_options = dict(options)
        # Snapshot both method drafts now, not when the graph is opened later.
        # A hidden/stale DWT control must never supply the DCT comparison's state.
        method_settings = {}
        for method in ("dct", "dwt"):
            state = dict(self._codec_settings_by_method[method])
            if codec == method:
                state.update({name: options[name] for name in self._codec_setting_fields[method] if name in options})
            if state.get("wavelet") == WAVELET_COMPARISON_SELECTION:
                state["wavelet"] = "db4"
            fields = ("colorspace",) if method == "dct" else ("wavelet", "level", "quantizer", "colorspace")
            method_settings[method] = {name: state[name] for name in fields}
        self._last_encode_options["rd_method_settings"] = method_settings
        self._last_encode_options["image"] = np.asarray(image).copy()
        if self.current_roi_mask is not None:
            self._last_encode_options["roi_mask"] = np.asarray(self.current_roi_mask, dtype=np.float32).copy()
        return options

    def _encode_work(self, options: dict):
        info = encode_file(
            options["input_path"], options["output_path"], mode=options["mode"], wavelet=options["wavelet"],
            level=options["level"], step=options["step"], quantizer=options["quantizer"], codec=options["codec"],
            colorspace=options["colorspace"], roi_mask=options["roi_mask"], roi_strength=options["roi_strength"],
            target_bpp=options["target_bpp"], target_psnr=options["target_psnr"], roi_feather=options["roi_feather"],
            restoration=False, standard_quality=options["standard_quality"], standard_rate=options["standard_rate"],
            transport_segments=options["transport_segments"], transport_tiles=options["transport_tiles"],
            transport_tile_size=options["transport_tile_size"],
        )
        info["output"] = str(options["output_path"])
        info["original_file_size"] = int(options["original_file_size"])
        return info

    def _encode(self) -> None:
        try:
            options = self._encode_options()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self.run_background(lambda _progress: self._encode_work(options), self._encode_done, "Encode")

    def _encode_done(self, info: dict) -> None:
        self.last_encode_info = info
        self._latest_encoded_path = Path(str(info["output"]))
        self._set_metrics_text(
            f"Encoding complete. Press Save to keep the encoded output.\n"
            f"Original size: {format_file_size(info.get('original_file_size'))} | "
            f"Encoded size: {format_file_size(info.get('file_size'))} | "
            f"Compression: {info.get('compression_ratio', 0):.2f}× | BPP: {info.get('bits_per_pixel', 0):.3f}"
        )
        self._set_metric_cards(
            bpp_value=info.get("bits_per_pixel"), ratio_value=info.get("compression_ratio"),
            original_size=info.get("original_file_size"), encoded_size=info.get("file_size"),
        )
        self.statusChanged.emit("Encoding complete. The result is temporary until Save is pressed.")

    def _decode(self) -> None:
        saved_source = Path(self._value("encoded_path").strip()) if self._value("encoded_path").strip() else None
        source = self._latest_encoded_path if self._latest_encoded_path is not None else saved_source
        if source is None or not source.is_file():
            self._show_error("Select a valid SWC/JPEG/JPEG2000 file")
            return

        def work(_progress):
            suffix = source.suffix.lower()
            decoded = decode_standard(source) if suffix in {".jpg", ".jpeg", ".jp2", ".j2k", ".j2c", ".jpf"} else decode_array(source)
            return {"array": decoded, "source": str(source)}

        self.run_background(work, self._decode_done, "Decode")

    def _decode_done(self, info: dict) -> None:
        self.decoded_array = np.asarray(info["array"])
        self.restored_array = None
        self._set_value("decoded_path", "")
        self.progressive_stages = []
        self.progressive_labels = []
        self._update_result_previews()
        self.preview_tabs.setCurrentWidget(self.decoded_preview)
        self.statusChanged.emit("Decoding complete. The image is temporary until Save is pressed.")

    def _build_progressive_stages(self, options: dict, info: dict,
                                  decoded: np.ndarray, restored: np.ndarray | None) -> tuple[list[np.ndarray], list[str]]:
        """Build a four-image quality ladder for the comparison view.

        The two middle images are real codec round-trips with smaller
        quantization steps; the last image is the actual selected result.
        """
        original = np.asarray(options["image"])
        final_candidate = restored if restored is not None else decoded
        if options["mode"] != "lossy":
            return [original, original, decoded, final_candidate], [
                "Original", "Lossless transform", "Decoded", "Final result"
            ]
        final_step = float(info.get("step", options["step"]) or options["step"])
        stage_steps = (max(0.01, final_step * 0.25), max(0.01, final_step * 0.5))
        stages: list[np.ndarray] = [original]
        try:
            with tempfile.TemporaryDirectory(prefix="smartcodec_stages_") as temp_dir:
                for index, stage_step in enumerate(stage_steps, start=1):
                    suffix = self._encoded_suffix(options["codec"])
                    stage_path = str(Path(temp_dir) / f"stage_{index}{suffix}")
                    stage_options = dict(options)
                    stage_options.update({
                        "output_path": stage_path,
                        "step": stage_step,
                        "target_bpp": None,
                        "target_psnr": None,
                    })
                    self._encode_work(stage_options)
                    if options["codec"] in {"jpeg", "jpeg2000"}:
                        stage_image = decode_standard(stage_path)
                    else:
                        stage_image = decode_array(stage_path)
                    stages.append(np.asarray(stage_image))
        except (OSError, RuntimeError, ValueError):
            # The final comparison remains usable even if a ladder stage is
            # unavailable for a particular codec/backend.
            stages = [original, original, decoded]
        stages.append(np.asarray(final_candidate))
        method = str(options["codec"]).upper()
        selected_wavelet = str(options.get("wavelet", "")).strip()
        if method == "DWT" and selected_wavelet:
            final_name = f"Final (DWT / {selected_wavelet})"
        else:
            final_name = f"Final ({method})"
        return stages, ["Original", "Light compression", "Medium compression", final_name]

    def _encode_decode(self) -> None:
        try:
            options = self._encode_options()
        except Exception as exc:
            self._show_error(str(exc))
            return
        apply_ai = bool(self._value("ai_reconstruction")) and options["mode"] == "lossy"

        def work(_progress):
            info = self._encode_work(options)
            if options["codec"] in {"jpeg", "jpeg2000"}:
                decoded = decode_standard(options["output_path"])
            else:
                decoded = decode_array(options["output_path"])
            restored = None
            if apply_ai:
                selected_model, selection = select_best_restoration_model("auto", workspace_root())
                if selected_model is None:
                    # Keep the operation usable on a clean installation, but
                    # report that this was not real AI inference.
                    restored = detail_reconstruction(decoded)
                    info["restoration"] = {
                        "enabled": True, "strategy": "deterministic-residual-baseline",
                        "fallback": True, "fallback_reason": selection.get("reason", "no-model"),
                        "model": None,
                    }
                else:
                    try:
                        adapter = TorchRestorationAdapter(selected_model)
                        restored = adapter.restore(decoded)
                    except (ImportError, RuntimeError) as exc:
                        restored = detail_reconstruction(decoded)
                        info["restoration"] = {
                            "enabled": True, "strategy": "deterministic-residual-baseline",
                            "fallback": True, "fallback_reason": str(exc),
                            "model": str(selected_model), "model_selection": selection,
                        }
                    else:
                        info["restoration"] = {
                            "enabled": True, "strategy": adapter.name, "fallback": False,
                            "model": str(selected_model), "model_sha256": adapter.sha256,
                            "device": adapter.device, "model_selection": selection,
                        }
            else:
                info["restoration"] = {"enabled": False, "strategy": "disabled"}
            # Decoded/restored arrays remain in memory.  Only the explicit Save
            # actions in the Files card create persistent user files.
            progressive = self._build_progressive_stages(options, info, decoded, restored)
            return info, decoded, restored, progressive

        self.run_background(work, self._encode_decode_done, "Encode + Decode")

    def _encode_decode_done(self, result) -> None:
        self.last_encode_info, self.decoded_array, self.restored_array, progressive = result
        self._latest_encoded_path = Path(str(self.last_encode_info["output"]))
        self.progressive_stages, self.progressive_labels = progressive
        self._update_result_previews()
        self.preview_tabs.setCurrentWidget(self.decoded_preview)
        restoration = self.last_encode_info.get("restoration", {}) if self.last_encode_info else {}
        if restoration.get("strategy") == "pytorch-model":
            selected = Path(restoration.get("model", ""))
            self.statusChanged.emit(f"Encoding and comparison complete; bundled restoration model used: {selected.name}")
        elif restoration.get("fallback"):
            self.statusChanged.emit("Encoding and comparison complete; the restoration model was unavailable, so the baseline method was used.")
        else:
            self.statusChanged.emit("Encoding and comparison complete.")

    def _update_result_previews(self) -> None:
        """Refresh all result tabs for both Decode and Encode + Decode paths."""
        if self.original_array is None or self.decoded_array is None:
            return
        self.decoded_preview.set_array(self.decoded_array)
        if self.restored_array is not None:
            self.restored_preview.set_array(self.restored_array)
            comparison_target = self.restored_array
        else:
            self.restored_preview.show_message("AI restoration is disabled. Enable it to create a restored image.")
            comparison_target = self.decoded_array
        self.difference_preview.set_pil(difference_image(self.original_array, comparison_target))
        self._refresh_metrics()
        self._refresh_comparison()

    def _open_compare_fullscreen(self) -> None:
        """Open a full-screen comparison window with Original/Decoded/etc. tabs."""
        if self._fullscreen_dialog is None:
            self._fullscreen_dialog = CompareFullscreenDialog(self)
        self._fullscreen_dialog.showFullScreen()
        self._fullscreen_dialog.raise_()
        self._fullscreen_dialog.activateWindow()
        self._sync_fullscreen_previews()

    def _sync_fullscreen_previews(self) -> None:
        """Mirror the current five preview states into the full-screen dialog."""
        dialog = self._fullscreen_dialog
        if dialog is None or not dialog.isVisible():
            return
        dialog.sync_metrics({key: label.text() for key, label in self.metric_values.items()})
        if self.original_array is not None:
            dialog.original_preview.set_array(self.original_array)
        else:
            dialog.original_preview.show_message("No image")
        if self.decoded_array is not None:
            dialog.decoded_preview.set_array(self.decoded_array)
        else:
            dialog.decoded_preview.show_message("No image")
        if self.restored_array is not None:
            dialog.restored_preview.set_array(self.restored_array)
        else:
            dialog.restored_preview.show_message("AI restoration is disabled. Enable it to create a restored image.")
        if self.original_array is not None and self.decoded_array is not None:
            target = self.restored_array if self.restored_array is not None else self.decoded_array
            dialog.difference_preview.set_pil(difference_image(self.original_array, target))
        else:
            dialog.difference_preview.show_message("No image")
        mode = combo_value(self.comparison_mode)
        language = str(self.property("ui_language") or "en")
        dialog.sync_controls(mode, self.comparison_position.value())
        target_name = combo_value(self.comparison_target)
        candidate = self.restored_array if target_name == "Restored" and self.restored_array is not None else self.decoded_array
        if mode == "8-bit grayscale":
            dialog.grayscale_preview.set_images(
                self.original_array, candidate,
                "Restored" if target_name == "Restored" and self.restored_array is not None else "Decoded",
                language,
            )
            dialog.comparison_stack.setCurrentWidget(dialog.grayscale_preview)
            return
        if self.original_array is None:
            if mode == "Histogram":
                dialog.histogram_preview.show_message("No image")
                dialog.comparison_stack.setCurrentWidget(dialog.histogram_preview)
            elif mode == "PSNR–BPP graph":
                dialog.rate_distortion_preview.show_message(
                    "Run Encode + Decode to generate the PSNR–BPP graph.", language,
                )
                dialog.comparison_stack.setCurrentWidget(dialog.rate_distortion_preview)
            elif mode == "Block grid":
                dialog.transform_grid_preview.show_message("Select or show an image to inspect its transform grid.")
                dialog.comparison_stack.setCurrentWidget(dialog.transform_grid_preview)
            elif mode == "Wavelet comparison":
                dialog.wavelet_comparison_preview.show_message(
                    "Select or show an image before running the wavelet comparison.", language,
                )
                dialog.comparison_stack.setCurrentWidget(dialog.wavelet_comparison_preview)
            else:
                dialog.comparison_preview.show_message("No image")
                dialog.comparison_stack.setCurrentWidget(dialog.comparison_preview)
            return
        if mode == "Transform tree":
            codec = self._value("codec_method")
            visual_codec = "dct" if codec in {"jpeg", "dct"} else "dwt"
            dialog.transform_preview.set_transform(
                self.original_array, visual_codec, self._effective_wavelet(), int(self._value("level")),
                language=str(self.property("ui_language") or "en"),
            )
            dialog.comparison_stack.setCurrentWidget(dialog.transform_preview)
        elif mode == "Block grid":
            codec = self._value("codec_method")
            visual_codec = "dct" if codec in {"jpeg", "dct"} else "dwt"
            dialog.transform_grid_preview.set_transform(
                self.original_array, visual_codec, self._effective_wavelet(), int(self._value("level")),
                language=str(self.property("ui_language") or "en"),
            )
            dialog.comparison_stack.setCurrentWidget(dialog.transform_grid_preview)
        elif mode == "Wavelet comparison":
            if self.wavelet_comparison_results:
                dialog.wavelet_comparison_preview.set_results(
                    self.wavelet_comparison_results,
                    language=language,
                )
            else:
                dialog.wavelet_comparison_preview.show_message("Wavelet comparison is being processed.", language)
            dialog.comparison_stack.setCurrentWidget(dialog.wavelet_comparison_preview)
        elif mode == "PSNR–BPP graph":
            if self.rate_distortion_results:
                dialog.rate_distortion_preview.set_results(
                    self.rate_distortion_results,
                    self.rate_distortion_codec,
                    language=language,
                )
            elif self.decoded_array is None or self.last_encode_info is None:
                dialog.rate_distortion_preview.show_message(
                    "Run Encode + Decode to generate the PSNR–BPP graph.", language,
                )
            elif not self._last_encode_options or self._last_encode_options.get("mode") != "lossy":
                dialog.rate_distortion_preview.show_message("PSNR–BPP graph requires lossy mode.", language)
            else:
                dialog.rate_distortion_preview.show_message("PSNR–BPP graph is being processed.", language)
            dialog.comparison_stack.setCurrentWidget(dialog.rate_distortion_preview)
        elif mode == "Histogram":
            if candidate is None:
                dialog.histogram_preview.show_message("No image")
            else:
                dialog.histogram_preview.set_histograms(
                    self.original_array,
                    candidate,
                    "Restored" if target_name == "Restored" else "Decoded",
                    language=str(self.property("ui_language") or "en"),
                )
            dialog.comparison_stack.setCurrentWidget(dialog.histogram_preview)
        elif candidate is not None:
            rendered = comparison_image(
                self.original_array, candidate, mode,
                "Restored" if target_name == "Restored" else "Decoded",
                self.comparison_position.value() / 100.0,
                stages=self.progressive_stages if mode == "Progressive stages" else None,
                stage_labels=self.progressive_labels if mode == "Progressive stages" else None,
                language=str(self.property("ui_language") or "en"),
            )
            dialog.comparison_preview.set_pil(rendered)
            dialog.comparison_stack.setCurrentWidget(dialog.comparison_preview)
        else:
            if mode in {"Slider", "Side-by-side"}:
                # A source-only preview is not a decoded/compressed result.
                dialog.comparison_preview.set_array(self.original_array)
            else:
                dialog.comparison_preview.show_message("Run Encode + Decode to compare this image.")
            dialog.comparison_stack.setCurrentWidget(dialog.comparison_preview)

    # ------------------------------------------------------------------
    # TR: Metrics/comparison / EN: Metrics/comparison
    # ------------------------------------------------------------------
    def _refresh_metrics(self) -> None:
        if self.original_array is None or self.decoded_array is None:
            return
        decoded_psnr = psnr(self.original_array, self.decoded_array)
        decoded_ssim = ssim(self.original_array, self.decoded_array)
        text = f"MSE: {mse(self.original_array, self.decoded_array):.4f} | PSNR: {decoded_psnr:.2f} dB | SSIM: {decoded_ssim:.4f}"
        if self.last_encode_info:
            text += (
                f" | Original size: {format_file_size(self.last_encode_info.get('original_file_size'))}"
                f" | Encoded size: {format_file_size(self.last_encode_info.get('file_size'))}"
            )
        if self.restored_array is not None:
            text += f"\nRestored MSE: {mse(self.original_array, self.restored_array):.4f} | PSNR: {psnr(self.original_array, self.restored_array):.2f} dB | SSIM: {ssim(self.original_array, self.restored_array):.4f}"
        if self.last_encode_info:
            text += f"\nCompression: {self.last_encode_info.get('compression_ratio', 0):.2f}× | BPP: {self.last_encode_info.get('bits_per_pixel', 0):.3f}"
            target_bpp = self.last_encode_info.get("target_bpp")
            target_psnr = self.last_encode_info.get("target_psnr")
            if target_bpp is not None:
                achieved_bpp = float(self.last_encode_info.get("bits_per_pixel", 0.0))
                text += (
                    f"\nTarget BPP: {float(target_bpp):.3f} | Achieved BPP: {achieved_bpp:.3f}"
                    f" | Target error: {achieved_bpp - float(target_bpp):+.3f} BPP"
                )
            elif target_psnr is not None:
                text += (
                    f"\nTarget PSNR: {float(target_psnr):.2f} dB | Achieved PSNR: {decoded_psnr:.2f} dB"
                    f" | Target error: {decoded_psnr - float(target_psnr):+.2f} dB"
                )
        if self.current_roi_mask is not None:
            regions = region_metrics(self.original_array, self.decoded_array, self.current_roi_mask)
            text += f" | ROI PSNR: {regions['roi_psnr']:.2f} dB | Background PSNR: {regions['background_psnr']:.2f} dB"
        self._set_metric_cards(
            decoded_psnr,
            decoded_ssim,
            self.last_encode_info.get("bits_per_pixel") if self.last_encode_info else None,
            self.last_encode_info.get("compression_ratio") if self.last_encode_info else None,
            self.last_encode_info.get("original_file_size") if self.last_encode_info else None,
            self.last_encode_info.get("file_size") if self.last_encode_info else None,
        )
        self._set_metrics_text(text)

    def _refresh_comparison(self) -> None:
        target = combo_value(self.comparison_target)
        wants_restored = target == "Restored"
        candidate = self.restored_array if wants_restored and self.restored_array is not None else self.decoded_array
        mode = combo_value(self.comparison_mode)
        # Transform tree is useful immediately after loading a JPEG/JPEG2000
        # image, before any encode/decode operation has been run.
        if mode == "8-bit grayscale":
            self.grayscale_preview.set_images(
                self.original_array, candidate,
                "Restored" if wants_restored and self.restored_array is not None else "Decoded",
                str(self.property("ui_language") or "en"),
            )
            self.comparison_stack.setCurrentWidget(self.grayscale_preview)
            self._sync_fullscreen_previews()
            return
        if mode == "Transform tree":
            if self.original_array is None:
                self.comparison_preview.show_message("Select or show an image to inspect its transform tree.")
                self.comparison_stack.setCurrentWidget(self.comparison_preview)
                self._sync_fullscreen_previews()
                return
            codec = self._value("codec_method")
            visual_codec = "dct" if codec in {"jpeg", "dct"} else "dwt"
            self.transform_preview.set_transform(
                self.original_array,
                visual_codec,
                self._effective_wavelet(),
                int(self._value("level")),
                language=str(self.property("ui_language") or "en"),
            )
            self.comparison_stack.setCurrentWidget(self.transform_preview)
            self._sync_fullscreen_previews()
            return
        if mode == "Block grid":
            if self.original_array is None:
                self.transform_grid_preview.show_message("Select or show an image to inspect its transform grid.")
            else:
                codec = self._value("codec_method")
                visual_codec = "dct" if codec in {"jpeg", "dct"} else "dwt"
                self.transform_grid_preview.set_transform(
                    self.original_array,
                    visual_codec,
                    self._effective_wavelet(),
                    int(self._value("level")),
                    language=str(self.property("ui_language") or "en"),
                )
            self.comparison_stack.setCurrentWidget(self.transform_grid_preview)
            self._sync_fullscreen_previews()
            return
        if mode == "Wavelet comparison":
            language = str(self.property("ui_language") or "en")
            if self.original_array is None:
                self.wavelet_comparison_preview.show_message(
                    "Select or show an image before running the wavelet comparison.", language,
                )
            elif self._value("mode") != "lossy":
                self.wavelet_comparison_preview.show_message(
                    "Wavelet comparison requires lossy mode because lossless mode uses the fixed reversible 5/3 transform.",
                    language,
                )
            elif self.wavelet_comparison_results:
                self.wavelet_comparison_preview.set_results(self.wavelet_comparison_results, language=language)
            else:
                self.wavelet_comparison_preview.show_message("Wavelet comparison is being processed.", language)
                self._wavelet_compare_timer.start()
            self.comparison_stack.setCurrentWidget(self.wavelet_comparison_preview)
            self._sync_fullscreen_previews()
            return
        if mode == "PSNR–BPP graph":
            language = str(self.property("ui_language") or "en")
            if self.original_array is None or self.decoded_array is None or self.last_encode_info is None:
                self.rate_distortion_preview.show_message(
                    "Run Encode + Decode to generate the PSNR–BPP graph.", language,
                )
            elif not self._last_encode_options or self._last_encode_options.get("mode") != "lossy":
                self.rate_distortion_preview.show_message("PSNR–BPP graph requires lossy mode.", language)
            elif self.rate_distortion_results:
                self.rate_distortion_preview.set_results(
                    self.rate_distortion_results,
                    self.rate_distortion_codec,
                    language=language,
                )
            else:
                self.rate_distortion_preview.show_message("PSNR–BPP graph is being processed.", language)
                self._rate_distortion_timer.start()
            self.comparison_stack.setCurrentWidget(self.rate_distortion_preview)
            self._sync_fullscreen_previews()
            return
        if mode == "Histogram":
            if self.original_array is None or candidate is None:
                self.histogram_preview.show_message("No image")
            else:
                self.histogram_preview.set_histograms(
                    self.original_array,
                    candidate,
                    "Restored" if wants_restored else "Decoded",
                    language=str(self.property("ui_language") or "en"),
                )
            self.comparison_stack.setCurrentWidget(self.histogram_preview)
            self._sync_fullscreen_previews()
            return
        if self.original_array is None or candidate is None:
            if self.original_array is not None and mode in {"Slider", "Side-by-side"}:
                self.comparison_preview.set_array(self.original_array)
            elif self.original_array is not None:
                self.comparison_preview.show_message("Run Encode + Decode to compare this image.")
            else:
                self.comparison_preview.clear_image()
            self.comparison_stack.setCurrentWidget(self.comparison_preview)
            self._sync_fullscreen_previews()
            return
        if mode == "Progressive stages" and not self.progressive_stages:
            self.comparison_preview.show_message("Run Encode + Decode to generate progressive stages.")
            self.comparison_stack.setCurrentWidget(self.comparison_preview)
            self._sync_fullscreen_previews()
            return
        image = comparison_image(
            self.original_array,
            candidate,
            mode,
            "Restored" if wants_restored else "Decoded",
            self.comparison_position.value() / 100.0,
            stages=self.progressive_stages if mode == "Progressive stages" else None,
            stage_labels=self.progressive_labels if mode == "Progressive stages" else None,
            language=str(self.property("ui_language") or "en"),
        )
        self.comparison_preview.set_pil(image)
        self.comparison_stack.setCurrentWidget(self.comparison_preview)
        self._sync_fullscreen_previews()

    # ------------------------------------------------------------------
    # TR: Presets and batch / EN: Presets and batch
    # ------------------------------------------------------------------
    def _save_preset(self) -> None:
        path = browse_file(self, "Save codec preset", "JSON (*.json)", save=True, suffix=".json")
        if path:
            try:
                Path(path).write_text(json.dumps(self.codec_preset(), indent=2, ensure_ascii=False), encoding="utf-8")
                self.statusChanged.emit(f"Preset saved: {path}")
            except OSError as exc:
                self._show_error(str(exc))

    def _load_preset(self) -> None:
        path = browse_file(self, "Load codec preset", "JSON (*.json)")
        if path:
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Preset JSON must contain an object")
                self.apply_settings(payload)
                self.statusChanged.emit(f"Preset loaded: {path}")
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                self._show_error(str(exc))

    def _batch_menu(self) -> None:
        answer = QtWidgets.QMessageBox.question(self, "Batch codec", "Choose Yes to encode an image folder to SWC, or No to decode an SWC folder to images.")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._batch_encode()
        elif answer == QtWidgets.QMessageBox.StandardButton.No:
            self._batch_decode()

    def _batch_encode(self) -> None:
        input_dir = browse_directory(self, "Select input image folder")
        output_dir = browse_directory(self, "Select SWC output folder")
        if not input_dir or not output_dir:
            return
        sources = sorted(path for path in Path(input_dir).rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"})
        if not sources:
            self._show_error("No supported images found in the input folder")
            return
        if self._value("codec_method") in {"jpeg", "jpeg2000"}:
            self._show_error("Standart JPEG/JPEG2000 icin tekli Encode akisini kullanin")
            return
        try:
            target_bpp, target_psnr = self._selected_quality_targets()
            options = {
                "mode": self._value("mode"), "codec": self._value("codec_method"),
                "wavelet": self._effective_wavelet(), "level": int(self._value("level")),
                "step": float(self._value("step")), "quantizer": self._value("quantizer"),
                "colorspace": self._value("colorspace"), "roi_strength": float(self._value("roi_strength")),
                "target_bpp": target_bpp,
                "target_psnr": target_psnr,
                "roi_feather": int(self._value("roi_feather")),
                "standard_quality": int(self._value("standard_quality")),
                "standard_rate": float(self._value("standard_rate")),
                "transport_segments": bool(self._value("transport_segments")),
                "transport_tiles": bool(self._value("transport_tiles")),
                "transport_tile_size": int(self._value("transport_tile_size")),
            }
        except (TypeError, ValueError) as exc:
            self._show_error(str(exc))
            return
        roi_boxes = self._parse_roi_boxes()
        roi_mask_path = self._value("roi_mask_path").strip()
        semantic_mask = None if self.semantic_roi_mask is None else np.asarray(self.semantic_roi_mask, dtype=np.float32).copy()

        def work(report_progress):
            count = 0
            for source in sources:
                if self._cancel_requested:
                    break
                image = load_image(source)
                mask = None
                if semantic_mask is not None and tuple(semantic_mask.shape) == tuple(image.shape[:2]):
                    mask = semantic_mask.copy()
                if roi_mask_path:
                    mask = load_mask(roi_mask_path, image.shape[:2])
                if roi_boxes:
                    generated = boxes_to_mask(image.shape[:2], roi_boxes)
                    mask = generated if mask is None else np.maximum(mask, generated)
                item = dict(options, input_path=str(source), output_path=str(Path(output_dir) / f"{source.stem}.swc"), image=image, roi_mask=mask)
                self._encode_work(item)
                count += 1
                report_progress(count)
            return count

        self.run_background(work, lambda count: self.statusChanged.emit(f"Batch encoding complete: {count} files"), "Batch encode", len(sources))

    def _batch_decode(self) -> None:
        input_dir = browse_directory(self, "Select SWC input folder")
        output_dir = browse_directory(self, "Select image output folder")
        if not input_dir or not output_dir:
            return
        sources = sorted(path for path in Path(input_dir).rglob("*.swc") if path.is_file())
        if not sources:
            self._show_error("No SWC files found in the input folder")
            return

        def work(report_progress):
            count = 0
            for source in sources:
                if self._cancel_requested:
                    break
                decode_file(source, Path(output_dir) / f"{source.stem}.png")
                count += 1
                report_progress(count)
            return count

        self.run_background(work, lambda count: self.statusChanged.emit(f"Batch decoding complete: {count} files"), "Batch decode", len(sources))

    # ------------------------------------------------------------------
    # TR: Profiles, mode and drag/drop / EN: Profiles, mode and drag/drop
    # ------------------------------------------------------------------
    def _apply_profile(self, profile: str) -> None:
        profiles = {
            "High quality": ("dwt", "lossy", "db4", 4.0, "scalar"),
            "Maximum compression": ("dwt", "lossy", "haar", 28.0, "scalar"),
            "Lossless medical": ("dwt", "lossless", "haar", 1.0, "uniform"),
            "ROI object": ("dwt", "lossy", "db4", 16.0, "scalar"),
        }
        if profile in profiles:
            codec, mode, wavelet, step, quantizer = profiles[profile]
            self._set_value("codec_method", codec)
            self._set_value("mode", mode)
            self._set_value("wavelet", wavelet)
            self._set_value("step", step)
            self._set_value("quantizer", quantizer)

    def _mode_changed(self, mode: str) -> None:
        self._quality_target_changed()
        if mode == "lossless":
            self._set_value("ai_reconstruction", False)
            self.statusChanged.emit("Lossless mode uses the reversible integer 5/3 wavelet.")
        else:
            self.statusChanged.emit("Lossy mode supports ROI prioritization and detail restoration.")

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if not paths:
            return
        path = Path(paths[0])
        if path.suffix.lower() == ".swc":
            self._set_value("encoded_path", str(path))
            self._set_value("decoded_path", str(path.with_name(path.stem + "_decoded.png")))
        else:
            self._set_value("input_path", str(path))
            self._set_value("encoded_path", "")
            self._set_value("decoded_path", "")
            self._latest_encoded_path = None
            self._preview_input()
        event.acceptProposedAction()

    def _show_error(self, message: str) -> None:
        self.statusChanged.emit(f"Error: {message}")
        QtWidgets.QMessageBox.critical(self, "Operation failed", message)
