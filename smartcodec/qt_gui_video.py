"""TR: PySide6 video ve transport sekmesi.
EN: PySide6 video and transport tab.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from .qt_gui_common import AsyncWidget, browse_directory, browse_file, configure_combo_popup, workspace_root
from .qt_i18n import combo_value, set_combo_value
from .transport import send_udp_file, simulate_file
from .video import compress_video, decompress_video, package_video, simulate_video_transport


VIDEO_FILTER = "Video (*.mp4 *.avi *.mov *.mkv *.m4v);;All files (*.*)"


class VideoTab(AsyncWidget):
    """TR: Video frame/GOP kodlama ve paket aktarim simulasyonu.
    EN: Video frame/GOP encoding and packet-transport simulation.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = workspace_root()
        self.fields: dict[str, QtWidgets.QWidget] = {}
        self._build_ui(root)

    def _build_ui(self, root: Path) -> None:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        introduction = QtWidgets.QLabel("Simple workflow: choose a video, encode its frames, then create the decoded MP4. Transport testing is optional.")
        introduction.setWordWrap(True)
        layout.addWidget(introduction)
        self._build_paths(layout, root)
        self._build_codec(layout)
        self._build_decode(layout)
        self._build_transport(layout, root)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _group(self, layout: QtWidgets.QVBoxLayout, title: str) -> QtWidgets.QFormLayout:
        group = QtWidgets.QGroupBox(title)
        group.setObjectName("controlCard")
        form = QtWidgets.QFormLayout(group)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.addWidget(group)
        return form

    def _line(self, form: QtWidgets.QFormLayout, name: str, label: str, value: str = "", button: str | None = None, callback=None) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value)
        self.fields[name] = edit
        if button:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(edit, 1)
            action = QtWidgets.QPushButton(button)
            action.clicked.connect(callback)
            row.addWidget(action)
            form.addRow(label, row)
        else:
            form.addRow(label, edit)
        return edit

    def _combo(self, form: QtWidgets.QFormLayout, name: str, label: str, values: tuple[str, ...], current: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItems(values)
        configure_combo_popup(combo)
        combo.setCurrentText(current)
        self.fields[name] = combo
        form.addRow(label, combo)
        return combo

    def _spin(self, form: QtWidgets.QFormLayout, name: str, label: str, value: float, minimum: float, maximum: float, integer: bool = False):
        if integer:
            widget = QtWidgets.QSpinBox()
            widget.setRange(int(minimum), int(maximum))
            widget.setValue(int(value))
        else:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(float(minimum), float(maximum))
            widget.setDecimals(3)
            widget.setValue(float(value))
        self.fields[name] = widget
        form.addRow(label, widget)
        return widget

    def _check(self, form: QtWidgets.QFormLayout, name: str, label: str, value: bool = False) -> QtWidgets.QCheckBox:
        widget = QtWidgets.QCheckBox(label)
        widget.setChecked(value)
        self.fields[name] = widget
        form.addRow(widget)
        return widget

    def _build_paths(self, layout: QtWidgets.QVBoxLayout, root: Path) -> None:
        form = self._group(layout, "Video files")
        self._line(form, "video_input", "Input video", button="Browse", callback=self._choose_video)
        self._line(form, "video_output_dir", "Frame output folder", str(root / "outputs" / "video_encoded"), button="Browse", callback=self._choose_output_dir)

    def _build_codec(self, layout: QtWidgets.QVBoxLayout) -> None:
        form = self._group(layout, "Video codec and motion")
        self._combo(form, "video_mode", "Mode", ("lossy", "lossless"), "lossy")
        codec = self._combo(form, "video_codec", "Compression method", ("dwt", "dct", "prqmf4"), "dwt")
        strength = self._spin(form, "video_step", "Compression strength", 12.0, 0.01, 1000.0)
        strength.setToolTip("Higher values create smaller files with more visible quality loss.")
        explanation = QtWidgets.QLabel("Higher compression strength usually means a smaller file and more visible distortion.")
        explanation.setWordWrap(True)
        form.addRow(explanation)
        encode = QtWidgets.QPushButton("Encode video frames")
        encode.clicked.connect(self._encode_video)
        encode.setProperty("primary", True)
        form.addRow(encode)

        self.video_advanced_toggle = QtWidgets.QToolButton()
        self.video_advanced_toggle.setText("Advanced video settings")
        self.video_advanced_toggle.setCheckable(True)
        self.video_advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.video_advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.video_advanced_toggle.toggled.connect(self._set_video_advanced_visible)
        form.addRow(self.video_advanced_toggle)

        self.video_advanced_group = QtWidgets.QGroupBox()
        advanced = QtWidgets.QFormLayout(self.video_advanced_group)
        advanced.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        wavelet = self._combo(advanced, "video_wavelet", "Wavelet", ("haar", "db4", "db8", "db12"), "haar")
        self.video_wavelet_label = advanced.labelForField(wavelet)
        self._spin(advanced, "video_gop", "GOP", 1, 1, 60, integer=True)
        self._spin(advanced, "video_keyframe", "Keyframe", 1, 1, 60, integer=True)
        self._combo(advanced, "video_motion_method", "Motion", ("none", "translation", "block", "optical-flow"), "none")
        self._line(advanced, "video_roi_mask", "Video ROI mask", button="Browse", callback=self._choose_video_roi)
        self._check(advanced, "video_motion_compensation", "Motion compensation")
        self._check(advanced, "video_roi_tracking", "ROI motion tracking")
        self.video_advanced_group.setVisible(False)
        form.addRow(self.video_advanced_group)
        codec.currentIndexChanged.connect(self._update_video_fields)
        self._update_video_fields()

    def _build_decode(self, layout: QtWidgets.QVBoxLayout) -> None:
        form = self._group(layout, "Decoded video")
        note = QtWidgets.QLabel("After encoding, the manifest is filled automatically. You can also select an existing manifest.")
        note.setWordWrap(True)
        form.addRow(note)
        self._line(form, "video_manifest", "Manifest", button="Browse", callback=self._choose_manifest)
        self._line(form, "video_decoded", "Decoded video", str(workspace_root() / "outputs" / "decoded_video.mp4"), button="Save", callback=self._choose_decoded)
        decode = QtWidgets.QPushButton("Create decoded MP4")
        decode.clicked.connect(self._decode_video)
        decode.setProperty("primary", True)
        form.addRow(decode)

    def _set_video_advanced_visible(self, visible: bool) -> None:
        self.video_advanced_group.setVisible(visible)
        arrow = QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        self.video_advanced_toggle.setArrowType(arrow)

    def _update_video_fields(self) -> None:
        visible = self._value("video_codec") == "dwt"
        self.fields["video_wavelet"].setVisible(visible)
        if self.video_wavelet_label is not None:
            self.video_wavelet_label.setVisible(visible)

    def _build_transport(self, layout: QtWidgets.QVBoxLayout, root: Path) -> None:
        form = self._group(layout, "Transport simulation")
        note = QtWidgets.QLabel("This test shows what happens when parts of the compressed file are lost during transmission.")
        note.setWordWrap(True)
        form.addRow(note)
        loss = self._spin(form, "transport_loss", "Packet loss (0.10 = 10%)", 0.0, 0.0, 0.5)
        loss.setSingleStep(0.05)
        run = QtWidgets.QPushButton("Run transport")
        run.clicked.connect(self._simulate_transport)
        run.setProperty("primary", True)
        form.addRow(run)

        self.transport_advanced_toggle = QtWidgets.QToolButton()
        self.transport_advanced_toggle.setText("Advanced transport settings")
        self.transport_advanced_toggle.setCheckable(True)
        self.transport_advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.transport_advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.transport_advanced_toggle.toggled.connect(self._set_transport_advanced_visible)
        form.addRow(self.transport_advanced_toggle)

        self.transport_advanced_group = QtWidgets.QGroupBox()
        advanced = QtWidgets.QFormLayout(self.transport_advanced_group)
        advanced.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._check(advanced, "video_transport_segments", "Semantic transport bands")
        self._check(advanced, "video_transport_tiles", "Spatial tiles")
        framewise = self._check(advanced, "video_transport_framewise", "Frame-level partial transport")
        self._spin(advanced, "video_transport_tile_size", "Tile px", 64, 8, 2048, integer=True)
        self._line(advanced, "transport_output", "Transport output", str(root / "outputs" / "transport_received.swc"))
        backend = self._combo(advanced, "transport_backend", "Backend", ("simulation", "live-udp"), "simulation")
        host = self._line(advanced, "transport_host", "UDP target host", "127.0.0.1")
        port = self._spin(advanced, "transport_port", "UDP port", 5000, 1, 65535, integer=True)
        fec = self._check(advanced, "transport_fec", "Live UDP XOR-FEC")
        self.transport_live_widgets = (host, advanced.labelForField(host), port, advanced.labelForField(port), fec)
        self.transport_framewise = framewise
        self.transport_advanced_group.setVisible(False)
        form.addRow(self.transport_advanced_group)
        backend.currentIndexChanged.connect(self._update_transport_fields)
        self._update_transport_fields()

    def _set_transport_advanced_visible(self, visible: bool) -> None:
        self.transport_advanced_group.setVisible(visible)
        arrow = QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        self.transport_advanced_toggle.setArrowType(arrow)

    def _update_transport_fields(self) -> None:
        live = self._value("transport_backend") == "live-udp"
        for widget in self.transport_live_widgets:
            if widget is not None:
                widget.setVisible(live)
        self.transport_framewise.setEnabled(not live)

    def _value(self, name: str):
        widget = self.fields[name]
        if isinstance(widget, QtWidgets.QComboBox):
            return combo_value(widget)
        if isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return widget.value()
        return widget.text()

    def _set(self, name: str, value) -> None:
        widget = self.fields[name]
        if isinstance(widget, QtWidgets.QComboBox):
            set_combo_value(widget, value)
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            widget.setValue(value)
        else:
            widget.setText(str(value))
            widget.setCursorPosition(0)

    def settings_payload(self) -> dict:
        return {name: self._value(name) for name in self.fields}

    def apply_settings(self, values: dict) -> None:
        for name, value in values.items():
            if name in self.fields:
                self._set(name, value)
        self._update_video_fields()
        self._update_transport_fields()

    def _choose_video(self) -> None:
        path = browse_file(self, "Select video", VIDEO_FILTER)
        if path:
            self._set("video_input", path)
            self._set("video_output_dir", str(Path(path).with_name(Path(path).stem + "_swc_frames")))

    def _choose_output_dir(self) -> None:
        path = browse_directory(self, "Select frame output folder")
        if path:
            self._set("video_output_dir", path)

    def _choose_manifest(self) -> None:
        path = browse_file(self, "Select video manifest", "JSON (*.json)")
        if path:
            self._set("video_manifest", path)

    def _choose_decoded(self) -> None:
        path = browse_file(self, "Save decoded video", "MP4 (*.mp4)", save=True, suffix=".mp4")
        if path:
            self._set("video_decoded", path)

    def _choose_video_roi(self) -> None:
        path = browse_file(self, "Select video ROI mask", "Images (*.png *.bmp *.tif *.tiff)")
        if path:
            self._set("video_roi_mask", path)

    def _encode_video(self) -> None:
        input_path = self._value("video_input").strip()
        if not Path(input_path).is_file():
            self._error("Select a valid video")
            return
        options = {
            "mode": self._value("video_mode"), "codec": self._value("video_codec"), "wavelet": self._value("video_wavelet"),
            "level": 2, "step": float(self._value("video_step")), "gop_size": int(self._value("video_gop")),
            "keyframe_interval": int(self._value("video_keyframe")), "motion_estimation": self._value("video_motion_method") != "none",
            "motion_method": self._value("video_motion_method"), "motion_compensation": bool(self._value("video_motion_compensation")),
            "roi_mask_path": self._value("video_roi_mask").strip() or None, "roi_tracking": bool(self._value("video_roi_tracking")),
            "transport_segments": bool(self._value("video_transport_segments")), "transport_tiles": bool(self._value("video_transport_tiles")),
            "transport_tile_size": int(self._value("video_transport_tile_size")),
        }
        output_dir = self._value("video_output_dir").strip()
        self.run_background(lambda _progress: compress_video(input_path, output_dir, **options), self._video_done, "Video encode")

    def _video_done(self, info: dict) -> None:
        self._set("video_manifest", info["manifest"])
        self.statusChanged.emit(f"Video encoded as {info['frame_count']} frames: {info['manifest']}")

    def _decode_video(self) -> None:
        manifest = self._value("video_manifest").strip()
        output = self._value("video_decoded").strip()
        if not Path(manifest).is_file():
            self._error("Select a valid video manifest")
            return
        self.run_background(lambda _progress: decompress_video(manifest, output), lambda info: self.statusChanged.emit(f"Video created: {info['output']}"), "Video decode")

    def _simulate_transport(self) -> None:
        source = self._value("video_manifest").strip()
        destination = self._value("transport_output").strip()
        if not Path(source).is_file():
            self._error("Select an SWC file or manifest first")
            return
        is_manifest = Path(source).suffix.lower() == ".json"
        if is_manifest and self._value("video_transport_framewise"):
            if self._value("transport_backend") == "live-udp":
                self._error("Frame-level partial transport is only available with the simulation backend")
                return
            framewise_destination = Path(destination)
            if framewise_destination.suffix:
                framewise_destination = framewise_destination.with_suffix("")
            framewise_destination = framewise_destination.with_name(framewise_destination.name + "_frames")
            self.run_background(lambda _progress: simulate_video_transport(source, framewise_destination, loss_rate=float(self._value("transport_loss")), fec=bool(self._value("transport_fec"))), lambda info: self.statusChanged.emit(f"Frame transport: {info['partial_frame_count']} partial, {info['dropped_frame_count']} dropped"), "Frame transport")
            return
        if is_manifest:
            try:
                bundle = Path(destination)
                if bundle.suffix.lower() != ".zip":
                    bundle = bundle.with_suffix(".zip")
                package_video(source, bundle)
                source = str(bundle)
                destination = str(bundle.with_name(f"{bundle.stem}_received{bundle.suffix}"))
            except Exception as exc:
                self._error(f"Could not prepare video bundle: {exc}")
                return
        if self._value("transport_backend") == "live-udp":
            host, port, fec = self._value("transport_host").strip(), int(self._value("transport_port")), bool(self._value("transport_fec"))
            self.run_background(lambda _progress: send_udp_file(source, host, port, fec=fec), lambda info: self.statusChanged.emit(f"Live UDP sent: {info['transmitted_packets']} packets"), "Live UDP")
            return
        self.run_background(lambda _progress: simulate_file(source, destination, loss_rate=float(self._value("transport_loss"))), lambda info: self.statusChanged.emit(f"Packet simulation: {info['received']}/{info['packets']} packets received."), "Packet simulation")

    def _error(self, message: str) -> None:
        self.statusChanged.emit(f"Error: {message}")
        QtWidgets.QMessageBox.critical(self, "Video/transport failed", message)
