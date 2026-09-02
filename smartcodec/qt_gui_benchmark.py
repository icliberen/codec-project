"""TR: PySide6 benchmark sekmesi.
EN: PySide6 benchmark tab.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

from .benchmark import generate_dataset, generate_samples, run_benchmark, run_normalized_benchmark, run_roi_comparison_benchmark
from .qt_gui_common import AsyncWidget, browse_directory, browse_file, configure_combo_popup, workspace_root
from .qt_i18n import USER_ROLE, combo_value, set_combo_value, translate


class BenchmarkTab(AsyncWidget):
    """TR: Rate-distortion, normalize ve ROI benchmark ekranidir.
    EN: Rate-distortion, normalized, and ROI benchmark screen.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        root = workspace_root()
        self.fields: dict[str, QtWidgets.QWidget] = {}
        self.field_labels: dict[str, QtWidgets.QLabel] = {}
        self.field_rows: dict[str, QtWidgets.QWidget] = {}
        self.plot_path: Path | None = None
        self._build_ui(root)

    def _build_ui(self, root: Path) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)

        introduction = QtWidgets.QLabel("Choose an image folder and what you want to compare. Recommended settings are ready for first use.")
        introduction.setWordWrap(True)
        outer.addWidget(introduction)

        form_group = QtWidgets.QGroupBox("Benchmark settings")
        form_group.setObjectName("controlCard")
        form = QtWidgets.QFormLayout(form_group)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._line(form, "input", "Input folder", str(root / "outputs" / "samples"), self._choose_input)
        self._line(form, "output", "Output folder", str(root / "outputs" / "gui_benchmark"), self._choose_output)

        purpose = QtWidgets.QComboBox()
        for text, value in (
            ("Quick overall comparison", "grid"),
            ("Compare at the same quality (PSNR)", "target-psnr"),
            ("Compare at the same file size (BPP)", "target-bpp"),
            ("Compare ROI on and off", "roi-compare"),
        ):
            purpose.addItem(text)
            purpose.setItemData(purpose.count() - 1, value, USER_ROLE)
        configure_combo_popup(purpose)
        self.fields["normalization"] = purpose
        purpose_label = QtWidgets.QLabel("What do you want to compare?")
        self.field_labels["normalization"] = purpose_label
        form.addRow(purpose_label, purpose)

        self._line(form, "target", "Target value")
        self._line(form, "roi_mask", "ROI mask", button=self._choose_roi)
        outer.addWidget(form_group)

        buttons = QtWidgets.QHBoxLayout()
        for text, callback in (("Create example images", self._generate_samples), ("Run comparison", self._benchmark), ("Open result chart", self._show_plot)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            if text == "Run comparison":
                button.setProperty("primary", True)
            buttons.addWidget(button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setText("Advanced benchmark settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.toggled.connect(self._set_advanced_visible)
        outer.addWidget(self.advanced_toggle)

        self.advanced_group = QtWidgets.QGroupBox()
        advanced = QtWidgets.QFormLayout(self.advanced_group)
        advanced.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._line(advanced, "steps", "Step values", "4,8,16,32")
        self._line(advanced, "wavelets", "Wavelets", "haar,db4,db8")
        self._combo(advanced, "quantizer", "Quantizer", ("uniform", "scalar"), "uniform")
        self._combo(advanced, "allocation", "Allocation", ("greedy", "lagrangian", "dp"), "greedy")
        dataset_button = QtWidgets.QPushButton("Generate category dataset")
        dataset_button.clicked.connect(self._generate_dataset)
        advanced.addRow(dataset_button)
        self.advanced_group.setVisible(False)
        outer.addWidget(self.advanced_group)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(80)
        self.log.setVisible(False)
        outer.addWidget(self.log)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(("Category", "Image", "Codec", "Step", "BPP", "PSNR", "SSIM", "Status"))
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        outer.addWidget(self.table, 1)
        purpose.currentIndexChanged.connect(self._update_simple_fields)
        self._update_simple_fields()

    def _line(self, form: QtWidgets.QFormLayout, name: str, label: str, value: str = "", button=None) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value)
        self.fields[name] = edit
        label_widget = QtWidgets.QLabel(label)
        self.field_labels[name] = label_widget
        if button:
            row_widget = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(edit, 1)
            browse = QtWidgets.QPushButton("Browse")
            browse.clicked.connect(button)
            row.addWidget(browse)
            form.addRow(label_widget, row_widget)
            self.field_rows[name] = row_widget
        else:
            form.addRow(label_widget, edit)
            self.field_rows[name] = edit
        return edit

    def _combo(self, form: QtWidgets.QFormLayout, name: str, label: str, values: tuple[str, ...], current: str) -> None:
        combo = QtWidgets.QComboBox()
        combo.addItems(values)
        configure_combo_popup(combo)
        combo.setCurrentText(current)
        self.fields[name] = combo
        form.addRow(label, combo)

    def _set_advanced_visible(self, visible: bool) -> None:
        self.advanced_group.setVisible(visible)
        arrow = QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow
        self.advanced_toggle.setArrowType(arrow)

    def _set_simple_row_visible(self, name: str, visible: bool) -> None:
        self.field_labels[name].setVisible(visible)
        self.field_rows[name].setVisible(visible)

    def _update_simple_fields(self) -> None:
        mode = self._value("normalization")
        previous_mode = getattr(self, "_last_simple_mode", None)
        self._last_simple_mode = mode
        needs_target = mode != "grid"
        needs_roi = mode == "roi-compare"
        self._set_simple_row_visible("target", needs_target)
        self._set_simple_row_visible("roi_mask", needs_roi)
        if not needs_target:
            return
        source = "Target PSNR (dB)" if mode == "target-psnr" else "Target BPP"
        language = str(self.property("ui_language") or "en")
        self.field_labels["target"].setText(translate(source, language))
        default = "30" if mode == "target-psnr" else "1.0"
        self.fields["target"].setPlaceholderText(f"Example: {default}")
        if mode != previous_mode:
            self.fields["target"].setText(default)

    def _value(self, name: str) -> str:
        widget = self.fields[name]
        return combo_value(widget) if isinstance(widget, QtWidgets.QComboBox) else widget.text()

    def settings_payload(self) -> dict:
        return {name: self._value(name) for name in self.fields}

    def apply_settings(self, values: dict) -> None:
        for name, value in values.items():
            if name not in self.fields:
                continue
            widget = self.fields[name]
            if isinstance(widget, QtWidgets.QComboBox):
                set_combo_value(widget, value)
            else:
                widget.setText(str(value))
                widget.setCursorPosition(0)
        self._update_simple_fields()

    def _choose_input(self) -> None:
        path = browse_directory(self, "Select benchmark input folder")
        if path:
            self.fields["input"].setText(path)

    def _choose_output(self) -> None:
        path = browse_directory(self, "Select benchmark output folder")
        if path:
            self.fields["output"].setText(path)

    def _choose_roi(self) -> None:
        path = browse_file(self, "Select benchmark ROI mask", "Images (*.png *.bmp *.tif *.tiff)")
        if path:
            self.fields["roi_mask"].setText(path)

    def _generate_samples(self) -> None:
        output = Path(self._value("input"))
        self.run_background(lambda _progress: generate_samples(output), lambda paths: self._log(f"{len(paths)} sample images generated: {output}"), "Generate samples")

    def _generate_dataset(self) -> None:
        output = Path(self._value("input"))
        self.run_background(lambda _progress: generate_dataset(output), lambda paths: self._log(f"{len(paths)} categorized samples generated: {output}"), "Generate dataset")

    def _benchmark(self) -> None:
        try:
            steps = [float(item.strip()) for item in self._value("steps").split(",") if item.strip()]
            wavelets = [item.strip() for item in self._value("wavelets").split(",") if item.strip()]
            if not steps or not wavelets:
                raise ValueError("En az bir step ve wavelet girin")
            normalization = self._value("normalization")
            target_text = self._value("target").strip()
            target = float(target_text) if target_text else None
            if normalization != "grid" and (target is None or target <= 0):
                raise ValueError("Enter a positive target for normalized benchmarking")
            roi = self._value("roi_mask").strip()
            if normalization == "roi-compare" and not Path(roi).is_file():
                raise ValueError("Select a valid mask for ROI comparison")
        except ValueError as exc:
            self._show_error(str(exc))
            return
        input_dir, output_dir = self._value("input"), self._value("output")
        quantizer, allocation = self._value("quantizer"), self._value("allocation")
        if normalization == "grid":
            work = lambda _progress: run_benchmark(input_dir, output_dir, step_values=steps, wavelets=wavelets, quantizer=quantizer, allocation_method=allocation)
        elif normalization == "target-psnr":
            work = lambda _progress: run_normalized_benchmark(input_dir, output_dir, target_psnr=target, step_values=steps, wavelets=wavelets, quantizer=quantizer, allocation_method=allocation)
        elif normalization == "target-bpp":
            work = lambda _progress: run_normalized_benchmark(input_dir, output_dir, target_bpp=target, step_values=steps, wavelets=wavelets, quantizer=quantizer, allocation_method=allocation)
        else:
            work = lambda _progress: run_roi_comparison_benchmark(input_dir, output_dir, target_bpp=target, roi_mask_path=roi, step_values=steps, wavelets=wavelets, quantizer=quantizer, allocation_method=allocation)
        self.run_background(work, lambda rows: self._benchmark_done(rows, output_dir), "Benchmark")

    def _benchmark_done(self, rows: list[dict], output_dir: str) -> None:
        self.table.setRowCount(0)
        for row in rows:
            values = (row.get("category", ""), row.get("image", ""), row.get("codec", ""), row.get("step", ""), f"{float(row.get('bits_per_pixel', 0)):.3f}", row.get("psnr", ""), row.get("ssim", ""), row.get("status", ""))
            index = self.table.rowCount()
            self.table.insertRow(index)
            for column, value in enumerate(values):
                self.table.setItem(index, column, QtWidgets.QTableWidgetItem(str(value)))
        output = Path(output_dir)
        self.plot_path = output / ("roi_rate_distortion.png" if (output / "roi_rate_distortion.png").is_file() else "rate_distortion.png")
        statuses: dict[str, int] = {}
        for row in rows:
            key = str(row.get("status", "ok"))
            statuses[key] = statuses.get(key, 0) + 1
        self._log(f"Benchmark complete: {len(rows)} rows ({statuses})\nOutput: {output_dir}")

    def _show_plot(self) -> None:
        path = self.plot_path or Path(self._value("output")) / "rate_distortion.png"
        if not path.is_file():
            QtWidgets.QMessageBox.information(self, "Chart", "Run a benchmark first.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Rate-distortion chart")
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel()
        label.setPixmap(QtGui.QPixmap(str(path)).scaled(1100, 700, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(label)
        dialog.resize(1150, 760)
        dialog.exec()

    def _log(self, text: str) -> None:
        self.log.setVisible(True)
        self.log.appendPlainText(text)
        self.statusChanged.emit(text.splitlines()[0])

    def _show_error(self, message: str) -> None:
        self.statusChanged.emit(f"Error: {message}")
        QtWidgets.QMessageBox.critical(self, "Benchmark failed", message)
