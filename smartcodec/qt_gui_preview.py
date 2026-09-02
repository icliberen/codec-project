"""TR: Qt goruntu canvas'i ve karsilastirma cizimleri.
EN: Qt image canvas and comparison rendering.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pywt
from PySide6 import QtCore, QtGui, QtWidgets

from .qt_gui_common import array_to_pixmap, configure_application_font, display_image, format_file_size, pil_to_pixmap
from .qt_i18n import translate


def _ui_font(size: int = 16) -> ImageFont.ImageFont:
    """Use a Unicode-capable Windows font for Turkish labels in raster previews."""
    for path in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _tree_node(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
               fill: str = "#26364d", outline: str = "#73b7ff", text_fill: str = "white") -> None:
    """Draw a compact, presentation-friendly transform-tree node."""
    draw.rounded_rectangle(box, radius=9, fill=fill, outline=outline, width=2)
    x0, y0, x1, y1 = box
    draw.text((x0 + 8, y0 + (y1 - y0) // 2 - 9), text, fill=text_fill, font=_ui_font(15))


def _tree_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#9aa8ba") -> None:
    """Draw an elbow-free connector with a small arrow head."""
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=2)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 7
    points = [
        end,
        (end[0] - size * math.cos(angle - 0.45), end[1] - size * math.sin(angle - 0.45)),
        (end[0] - size * math.cos(angle + 0.45), end[1] - size * math.sin(angle + 0.45)),
    ]
    draw.polygon(points, fill=color)


def _coefficient_tile(values: np.ndarray, size: tuple[int, int] = (180, 112)) -> Image.Image:
    """Render signed transform coefficients as an auto-scaled heat tile."""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return Image.new("RGB", size, "#111827")
    if array.ndim < 2:
        array = array.reshape(1, -1)
    magnitude = np.log1p(np.abs(array))
    finite = magnitude[np.isfinite(magnitude)]
    high = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    high = max(high, 1e-6)
    normalized = np.clip(magnitude / high, 0.0, 1.0)
    # A neutral black-to-white scale is presentation-friendly and preserves
    # coefficient magnitude exactly: black is low energy, white is high.
    intensity = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.repeat(intensity[..., None], 3, axis=2), mode="RGB")
    return image.resize(size, Image.Resampling.BILINEAR)


def _luminance_plane(array: np.ndarray) -> np.ndarray:
    image = np.asarray(display_image(array), dtype=np.float32)
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


class TransformTreePreview(QtWidgets.QGraphicsView):
    """Native Qt transform map with readable, inspectable coefficient nodes.

    The diagram deliberately uses individual graphics items instead of a raster
    screenshot: labels stay sharp at every DPI, subbands have real tooltips,
    and the DWT dependency from LL(n) to the next level is explicit.
    """

    _NODE_SIZE = QtCore.QSizeF(142, 56)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            configure_application_font(app)
        self.setScene(QtWidgets.QGraphicsScene(self))
        # The transform diagram keeps a stable deep canvas in both themes so
        # its scientific colour scale and high-contrast Qt text stay legible.
        self.scene().setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#0d1420")))
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._zoom = 1.0
        self._auto_fit = True
        self.show_message("Select or show an image to inspect its transform tree.")

    def _text(self, value: str, position: QtCore.QPointF, size: int, color: str = "#eaf2ff",
              weight: int = 400) -> QtWidgets.QGraphicsTextItem:
        item = self.scene().addText(value)
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPointSize(size)
        font.setWeight(QtGui.QFont.Weight(weight))
        item.setFont(font)
        item.setDefaultTextColor(QtGui.QColor(color))
        item.setPos(position)
        return item

    def _node(self, label: str, x: float, y: float, *, fill: str, detail: str = "") -> QtCore.QRectF:
        rect = QtCore.QRectF(x, y, self._NODE_SIZE.width(), self._NODE_SIZE.height())
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 10, 10)
        item = self.scene().addPath(path, QtGui.QPen(QtGui.QColor("#79b8ff"), 1.5), QtGui.QBrush(QtGui.QColor(fill)))
        item.setToolTip(detail or label)
        item.setAcceptHoverEvents(True)
        label_item = self._text(label, QtCore.QPointF(x + 12, y + 14), 11, "#f8fbff", 600)
        label_item.setToolTip(detail or label)
        return rect

    def _arrow(self, source: QtCore.QRectF, target: QtCore.QRectF) -> None:
        start = QtCore.QPointF(source.right(), source.center().y())
        end = QtCore.QPointF(target.left(), target.center().y())
        line_pen = QtGui.QPen(QtGui.QColor("#8ea6c4"), 1.8)
        line_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        self.scene().addLine(QtCore.QLineF(start, end), line_pen)
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        head = 8
        polygon = QtGui.QPolygonF([
            end,
            QtCore.QPointF(end.x() - head * math.cos(angle - 0.48), end.y() - head * math.sin(angle - 0.48)),
            QtCore.QPointF(end.x() - head * math.cos(angle + 0.48), end.y() - head * math.sin(angle + 0.48)),
        ])
        self.scene().addPolygon(polygon, QtGui.QPen(QtCore.Qt.PenStyle.NoPen), QtGui.QBrush(QtGui.QColor("#8ea6c4")))

    @staticmethod
    def _coefficient_pixmap(values: np.ndarray, size: tuple[int, int] = (168, 96)) -> QtGui.QPixmap:
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            if array.ndim < 2:
                array = array.reshape(1, -1)
            magnitude = np.log1p(np.abs(array))
            finite = magnitude[np.isfinite(magnitude)]
            high = max(float(np.percentile(finite, 99.0)) if finite.size else 1.0, 1e-6)
            normalized = np.clip(magnitude / high, 0.0, 1.0)
            intensity = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
            rgb = np.repeat(intensity[..., None], 3, axis=2)
        image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        return QtGui.QPixmap.fromImage(image).scaled(size[0], size[1], QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                                      QtCore.Qt.TransformationMode.SmoothTransformation)

    def show_message(self, message: str) -> None:
        self.scene().clear()
        self._text(message, QtCore.QPointF(18, 18), 12, "#91a1b8")
        self.scene().setSceneRect(0, 0, 620, 280)

    def set_transform(self, array: np.ndarray, codec: str = "dwt", wavelet: str = "db4", level: int = 2,
                      language: str = "en") -> None:
        self.scene().clear()
        plane = _luminance_plane(array)
        codec = str(codec).lower()
        title = translate("Transform tree", language)
        leaves: list[tuple[str, np.ndarray, str]] = []

        self._text(title, QtCore.QPointF(24, 16), 16, "#ffffff", 700)
        if codec in {"jpeg", "dct"}:
            from .transforms import block_dct
            coefficients, _padded_shape = block_dct(plane)
            blocks = coefficients.reshape(-1, 8, 8)
            first = blocks[0] if len(blocks) else coefficients
            details = {
                "DC": "DC is the block's average brightness component.",
                "AC": "AC coefficients describe the block's changing detail and texture.",
            }
            self._text("DCT (8×8) • one block, DC and AC energy", QtCore.QPointF(24, 45), 11, "#b9c9df")
            source = self._node(translate("Input image", language), 28, 96, fill="#243044")
            dct = self._node("DCT (8×8)", 222, 96, fill="#1f5f8b")
            block = self._node("8×8 block", 416, 96, fill="#326f62")
            dc = self._node("DC", 610, 62, fill="#8a5a2b", detail=details["DC"])
            ac = self._node("AC", 610, 132, fill="#8a5a2b", detail=details["AC"])
            self._arrow(source, dct)
            self._arrow(dct, block)
            self._arrow(block, dc)
            self._arrow(block, ac)
            leaves = [("DC", np.asarray([[first[0, 0]]]), details["DC"]), ("AC", np.asarray(first), details["AC"])]
            footer = "DC retains average intensity; AC retains spatial detail."
        else:
            wavelet_name = str(wavelet or "db4")
            try:
                wave = pywt.Wavelet(wavelet_name)
            except (ValueError, TypeError):
                wavelet_name, wave = "db4", pywt.Wavelet("db4")
            maximum = pywt.dwt_max_level(min(plane.shape), wave.dec_len)
            selected = max(1, min(int(level or 1), max(1, maximum)))
            self._text(f"DWT ({wavelet_name}) • level {selected} • only LL flows into the next level", QtCore.QPointF(24, 45), 11, "#b9c9df")
            source = self._node(translate("Input image", language), 28, 120, fill="#243044")
            previous = source
            decomposition_input = plane
            descriptions = {
                "LL": "LL is the low-frequency approximation passed to the next DWT level.",
                "LH": "LH emphasizes horizontal detail.",
                "HL": "HL emphasizes vertical detail.",
                "HH": "HH emphasizes diagonal detail.",
            }
            for current_level in range(1, selected + 1):
                # Keep the LL node on the main horizontal path.  The next
                # level is therefore visibly fed by LL only, while detail
                # bands fan out above and below without crossing connectors.
                x = 222 + (current_level - 1) * 330
                level_node = self._node(translate(f"Level {current_level}", language), x, 120, fill="#326f62")
                self._arrow(previous, level_node)
                ll_values, values = pywt.dwt2(decomposition_input, wavelet_name, mode="periodization")
                positions = ((x + 160, 120), (x + 160, 42), (x + 160, 198), (x + 160, 276))
                children: dict[str, QtCore.QRectF] = {}
                for (kind, values_item), (child_x, child_y) in zip((("LL", ll_values), ("LH", values[0]), ("HL", values[1]), ("HH", values[2])), positions):
                    label = f"{kind}{current_level}"
                    node = self._node(label, child_x, child_y, fill="#326f62" if kind == "LL" else "#8a5a2b", detail=descriptions[kind])
                    self._arrow(level_node, node)
                    children[kind] = node
                    leaves.append((label, np.asarray(values_item), descriptions[kind]))
                previous = children["LL"]
                decomposition_input = ll_values
            footer = "LL keeps approximation; LH, HL and HH retain horizontal, vertical and diagonal detail."

        self._text(translate("Coefficient / subband preview", language), QtCore.QPointF(24, 330), 13, "#79b8ff", 600)
        for index, (label, values, tip) in enumerate(leaves):
            column, row = index % 4, index // 4
            x, y = 28 + column * 190, 366 + row * 140
            pixmap = self.scene().addPixmap(self._coefficient_pixmap(values))
            pixmap.setPos(x, y + 26)
            pixmap.setToolTip(tip)
            self._text(label, QtCore.QPointF(x + 3, y), 11, "#e9f3ff", 600).setToolTip(tip)
            frame = self.scene().addRect(x - 3, y + 23, 174, 102, QtGui.QPen(QtGui.QColor("#4f6683"), 1))
            frame.setToolTip(tip)
        footer_y = 392 + ((len(leaves) + 3) // 4) * 140
        self._text(footer, QtCore.QPointF(28, footer_y), 11, "#b9c9df")
        self.scene().setSceneRect(self.scene().itemsBoundingRect().adjusted(-24, -18, 28, 24))
        self._auto_fit = True
        self._zoom = 1.0
        self.fit_to_view()

    def fit_to_view(self) -> None:
        rect = self.scene().sceneRect()
        if not rect.isEmpty():
            self.resetTransform()
            self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(rect.center())

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.scene().items():
            factor = 1.14 if event.angleDelta().y() > 0 else 1 / 1.14
            next_zoom = max(0.4, min(4.0, self._zoom * factor))
            self.scale(next_zoom / self._zoom, next_zoom / self._zoom)
            self._zoom = next_zoom
            self._auto_fit = False
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_to_view()


class TransformGridPreview(QtWidgets.QGraphicsView):
    """Show the transform's real spatial partition as a native Qt overlay.

    DCT/JPEG uses one square for every 8x8 source block.  DWT/JPEG2000 uses
    the coefficient slices returned by PyWavelets, so the LL/LH/HL/HH borders
    match the selected wavelet and decomposition level instead of being a
    decorative grid.
    """

    _SCENE_SIZE = QtCore.QSizeF(1100, 680)
    _PANEL_SIZE = QtCore.QSizeF(472, 432)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.scene().setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#0d1420")))
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._zoom = 1.0
        self._auto_fit = True
        self.show_message("Select or show an image to inspect its transform grid.")

    def _text(self, value: str, x: float, y: float, size: int = 11, color: str = "#eaf2ff",
              weight: int = 400) -> QtWidgets.QGraphicsTextItem:
        item = self.scene().addText(value)
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPointSize(size)
        font.setWeight(QtGui.QFont.Weight(weight))
        item.setFont(font)
        item.setDefaultTextColor(QtGui.QColor(color))
        item.setPos(x, y)
        return item

    @staticmethod
    def _energy_pixmap(values: np.ndarray) -> QtGui.QPixmap:
        data = np.asarray(values, dtype=np.float32)
        magnitude = np.log1p(np.abs(data))
        finite = magnitude[np.isfinite(magnitude)]
        high = max(float(np.percentile(finite, 99.5)) if finite.size else 1.0, 1e-6)
        normalized = np.clip(magnitude / high, 0.0, 1.0)
        intensity = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        rgb = np.repeat(intensity[..., None], 3, axis=2)
        image = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        return QtGui.QPixmap.fromImage(image)

    def _place_pixmap(self, pixmap: QtGui.QPixmap, panel: QtCore.QRectF) -> tuple[QtWidgets.QGraphicsPixmapItem, QtCore.QRectF]:
        scaled = pixmap.scaled(
            int(panel.width()), int(panel.height()),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        x = panel.x() + (panel.width() - scaled.width()) / 2.0
        y = panel.y() + (panel.height() - scaled.height()) / 2.0
        item = self.scene().addPixmap(scaled)
        item.setPos(x, y)
        return item, QtCore.QRectF(x, y, scaled.width(), scaled.height())

    def _frame(self, rect: QtCore.QRectF, tooltip: str) -> QtWidgets.QGraphicsRectItem:
        item = self.scene().addRect(rect, QtGui.QPen(QtGui.QColor("#526984"), 1.4), QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
        item.setToolTip(tooltip)
        return item

    def _block_grid(self, rect: QtCore.QRectF, source_shape: tuple[int, int], block: int = 8) -> None:
        rows, columns = source_shape
        fine = QtGui.QPen(QtGui.QColor(90, 202, 255, 118), 0.72)
        fine.setCosmetic(True)
        major = QtGui.QPen(QtGui.QColor(255, 190, 92, 205), 1.35)
        major.setCosmetic(True)
        for column in range(0, columns + 1, block):
            x = rect.left() + rect.width() * min(column, columns) / max(1, columns)
            pen = major if column % (block * 8) == 0 else fine
            self.scene().addLine(x, rect.top(), x, rect.bottom(), pen)
        for row in range(0, rows + 1, block):
            y = rect.top() + rect.height() * min(row, rows) / max(1, rows)
            pen = major if row % (block * 8) == 0 else fine
            self.scene().addLine(rect.left(), y, rect.right(), y, pen)

    def _dwt_source_grid(self, rect: QtCore.QRectF, levels: int) -> None:
        current = QtCore.QRectF(rect)
        colors = ("#64c8ff", "#ffbd68", "#86e0b2", "#c49aff", "#ff8ca1")
        for index in range(levels):
            pen = QtGui.QPen(QtGui.QColor(colors[index % len(colors)]), 2.0)
            pen.setCosmetic(True)
            mid_x, mid_y = current.center().x(), current.center().y()
            self.scene().addLine(mid_x, current.top(), mid_x, current.bottom(), pen)
            self.scene().addLine(current.left(), mid_y, current.right(), mid_y, pen)
            level_rect = self.scene().addRect(current, pen, QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
            level_rect.setToolTip(f"Level {index + 1}: only the upper-left LL region continues to the next level.")
            current = QtCore.QRectF(current.left(), current.top(), current.width() / 2.0, current.height() / 2.0)

    @staticmethod
    def _slice_rect(slices: tuple[slice, slice], shape: tuple[int, int], target: QtCore.QRectF) -> QtCore.QRectF:
        row_slice, column_slice = slices
        row_start, row_stop = int(row_slice.start or 0), int(row_slice.stop or shape[0])
        column_start, column_stop = int(column_slice.start or 0), int(column_slice.stop or shape[1])
        x = target.left() + target.width() * column_start / max(1, shape[1])
        y = target.top() + target.height() * row_start / max(1, shape[0])
        width = target.width() * (column_stop - column_start) / max(1, shape[1])
        height = target.height() * (row_stop - row_start) / max(1, shape[0])
        return QtCore.QRectF(x, y, width, height)

    def _subband_rect(self, rect: QtCore.QRectF, label: str, color: str, tooltip: str) -> None:
        fill = QtGui.QColor(color)
        fill.setAlpha(24)
        pen = QtGui.QPen(QtGui.QColor(color), 2.0)
        pen.setCosmetic(True)
        item = self.scene().addRect(rect, pen, QtGui.QBrush(fill))
        item.setToolTip(tooltip)
        if rect.width() >= 40 and rect.height() >= 28:
            label_item = self._text(label, rect.left() + 4, rect.top() + 2, 9, "#ffffff", 700)
            label_item.setToolTip(tooltip)

    @staticmethod
    def _band_tooltip(label: str, values: np.ndarray, description: str, language: str) -> str:
        data = np.asarray(values, dtype=np.float32)
        height = int(data.shape[0]) if data.ndim >= 1 else 1
        width = int(data.shape[1]) if data.ndim >= 2 else 1
        mean_magnitude = float(np.mean(np.abs(data))) if data.size else 0.0
        return (
            f"{label} — {description}\n"
            f"{translate('Band size', language)}: {width}×{height}\n"
            f"{translate('Mean |coefficient|', language)}: {mean_magnitude:.2f}"
        )

    def show_message(self, message: str) -> None:
        self.scene().clear()
        language = str(self.property("ui_language") or "en")
        self._text(translate(message, language), 22, 20, 12, "#91a1b8")
        self.scene().setSceneRect(0, 0, 720, 320)

    def set_transform(self, array: np.ndarray, codec: str = "dwt", wavelet: str = "db4", level: int = 2,
                      language: str = "en") -> None:
        self.scene().clear()
        source = np.asarray(display_image(array))
        plane = _luminance_plane(array)
        codec = str(codec).lower()
        left_panel = QtCore.QRectF(36, 116, self._PANEL_SIZE.width(), self._PANEL_SIZE.height())
        right_panel = QtCore.QRectF(592, 116, self._PANEL_SIZE.width(), self._PANEL_SIZE.height())

        self._text(translate("Block grid", language), 34, 16, 17, "#ffffff", 700)
        self._text(translate("Input image", language), left_panel.x(), 78, 12, "#8fd3ff", 600)
        source_item, source_rect = self._place_pixmap(array_to_pixmap(source), left_panel)

        if codec in {"jpeg", "dct"}:
            from .transforms import block_dct
            coefficients, padded_shape = block_dct(plane)
            self._text(translate("DCT coefficient energy", language), right_panel.x(), 78, 12, "#ffbd68", 600)
            self._text(translate("Every square is one real 8×8 transform block.", language), 34, 46, 11, "#b9c9df")
            coefficient_item, coefficient_rect = self._place_pixmap(self._energy_pixmap(coefficients), right_panel)
            source_tip = translate("DCT divides the image into independent 8×8 pixel blocks.", language)
            coefficient_tip = translate("DC is at each block's upper-left; AC detail spreads toward its lower-right.", language)
            source_item.setToolTip(source_tip)
            coefficient_item.setToolTip(coefficient_tip)
            self._frame(source_rect, source_tip)
            self._frame(coefficient_rect, coefficient_tip)
            self._block_grid(source_rect, plane.shape, 8)
            self._block_grid(coefficient_rect, padded_shape, 8)
            cards = (
                ("8×8", translate("Pixel block", language), "#5acaff"),
                ("DC", translate("Average brightness", language), "#ffbd68"),
                ("AC", translate("Edges and texture", language), "#86e0b2"),
            )
        else:
            from .transforms import canonical_wavelet
            try:
                wavelet_name = canonical_wavelet(str(wavelet or "db4"))
                wave = pywt.Wavelet(wavelet_name)
            except (ValueError, TypeError):
                wavelet_name, wave = "db4", pywt.Wavelet("db4")
            maximum = pywt.dwt_max_level(min(plane.shape), wave.dec_len)
            selected = max(1, min(int(level or 1), max(1, maximum)))
            coefficients = pywt.wavedec2(plane, wavelet_name, mode="periodization", level=selected)
            coefficient_array, coefficient_slices = pywt.coeffs_to_array(coefficients)
            self._text(translate("DWT subband mosaic", language), right_panel.x(), 78, 12, "#ffbd68", 600)
            subtitle = translate("Each level splits LL into four frequency regions.", language)
            self._text(f"{subtitle}  •  {wavelet_name}  •  {translate('Level', language)} {selected}", 34, 46, 11, "#b9c9df")
            coefficient_item, coefficient_rect = self._place_pixmap(self._energy_pixmap(coefficient_array), right_panel)
            source_tip = translate("The coloured borders show each recursive DWT split on the image plane.", language)
            coefficient_tip = translate("This is the actual coefficient mosaic produced by the selected DWT settings.", language)
            source_item.setToolTip(source_tip)
            coefficient_item.setToolTip(coefficient_tip)
            self._frame(source_rect, source_tip)
            self._frame(coefficient_rect, coefficient_tip)
            self._dwt_source_grid(source_rect, selected)
            descriptions = {
                "LL": translate("Low-frequency approximation; this region continues to the next level.", language),
                "LH": translate("Horizontal detail coefficients.", language),
                "HL": translate("Vertical detail coefficients.", language),
                "HH": translate("Diagonal detail coefficients.", language),
            }
            palette = {"LL": "#64c8ff", "LH": "#ffbd68", "HL": "#86e0b2", "HH": "#c49aff"}
            ll_rect = self._slice_rect(coefficient_slices[0], coefficient_array.shape, coefficient_rect)
            ll_values = coefficient_array[coefficient_slices[0]]
            ll_tip = self._band_tooltip(f"LL{selected}", ll_values, descriptions["LL"], language)
            self._subband_rect(ll_rect, f"LL{selected}", palette["LL"], ll_tip)
            grouped_values: dict[str, list[np.ndarray]] = {"LL": [ll_values], "LH": [], "HL": [], "HH": []}
            for detail_index, slice_map in enumerate(coefficient_slices[1:]):
                detail_level = selected - detail_index
                for key, kind in (("ad", "LH"), ("da", "HL"), ("dd", "HH")):
                    if key in slice_map:
                        band_rect = self._slice_rect(slice_map[key], coefficient_array.shape, coefficient_rect)
                        band_values = coefficient_array[slice_map[key]]
                        grouped_values[kind].append(band_values)
                        band_label = f"{kind}{detail_level}"
                        band_tip = self._band_tooltip(band_label, band_values, descriptions[kind], language)
                        self._subband_rect(band_rect, band_label, palette[kind], band_tip)
            short_descriptions = {
                "LL": translate("Approximation / next level", language),
                "LH": translate("Horizontal detail", language),
                "HL": translate("Vertical detail", language),
                "HH": translate("Diagonal detail", language),
            }
            for kind, value_groups in grouped_values.items():
                coefficient_count = sum(values.size for values in value_groups)
                magnitude_sum = sum(float(np.sum(np.abs(values))) for values in value_groups)
                mean_magnitude = magnitude_sum / max(1, coefficient_count)
                short_descriptions[kind] += f"\n{translate('Mean |coefficient|', language)}: {mean_magnitude:.2f}"
            cards = tuple((kind, short_descriptions[kind], palette[kind]) for kind in ("LL", "LH", "HL", "HH"))

        card_width = 246 if len(cards) == 4 else 320
        total_width = card_width * len(cards) + 12 * (len(cards) - 1)
        start_x = (self._SCENE_SIZE.width() - total_width) / 2.0
        for index, (key, description, color) in enumerate(cards):
            rect = QtCore.QRectF(start_x + index * (card_width + 12), 576, card_width, 72)
            path = QtGui.QPainterPath()
            path.addRoundedRect(rect, 9, 9)
            fill = QtGui.QColor("#172234")
            item = self.scene().addPath(path, QtGui.QPen(QtGui.QColor(color), 1.3), QtGui.QBrush(fill))
            item.setToolTip(description)
            self._text(key, rect.left() + 12, rect.top() + 7, 11, color, 700).setToolTip(description)
            description_item = self._text(description, rect.left() + 52, rect.top() + 8, 9, "#dce8f8")
            description_item.setTextWidth(rect.width() - 64)
            description_item.setToolTip(description)

        self.scene().setSceneRect(QtCore.QRectF(0, 0, self._SCENE_SIZE.width(), self._SCENE_SIZE.height()))
        self._auto_fit = True
        self._zoom = 1.0
        self.fit_to_view()

    def fit_to_view(self) -> None:
        rect = self.scene().sceneRect()
        if not rect.isEmpty():
            self.resetTransform()
            self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(rect.center())

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.scene().items():
            factor = 1.14 if event.angleDelta().y() > 0 else 1 / 1.14
            next_zoom = max(0.4, min(5.0, self._zoom * factor))
            self.scale(next_zoom / self._zoom, next_zoom / self._zoom)
            self._zoom = next_zoom
            self._auto_fit = False
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_to_view()


class WaveletComparisonPreview(QtWidgets.QGraphicsView):
    """Native Qt side-by-side comparison for real db4/db8/db12 round-trips."""

    _SCENE_RECT = QtCore.QRectF(0, 0, 1200, 700)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            configure_application_font(app)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.scene().setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#0d1420")))
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._zoom = 1.0
        self._auto_fit = True
        self.show_message("Run the wavelet comparison to see db4, db8 and db12 side by side.")

    def _text(self, value: str, x: float, y: float, size: int = 11, color: str = "#eaf2ff",
              weight: int = 400) -> QtWidgets.QGraphicsTextItem:
        item = self.scene().addText(value)
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPointSize(size)
        font.setWeight(QtGui.QFont.Weight(weight))
        item.setFont(font)
        item.setDefaultTextColor(QtGui.QColor(color))
        item.setPos(x, y)
        return item

    def show_message(self, message: str, language: str | None = None) -> None:
        self.scene().clear()
        selected_language = language or str(self.property("ui_language") or "en")
        self._text(translate(message, selected_language), 24, 22, 12, "#91a1b8")
        self.scene().setSceneRect(0, 0, 820, 320)

    def set_results(self, results: list[dict], language: str = "en") -> None:
        self.scene().clear()
        if not results:
            self.show_message("Run the wavelet comparison to see db4, db8 and db12 side by side.", language)
            return
        best_quality = max(results, key=lambda item: float(item.get("psnr", float("-inf"))))["wavelet"]
        smallest_file = min(results, key=lambda item: int(item.get("file_size", 0) or 0))["wavelet"]
        level = int(results[0].get("level", 1))
        step = float(results[0].get("step", 0.0))
        original_size = format_file_size(results[0].get("original_file_size"))
        self._text(translate("Wavelet comparison", language), 28, 16, 17, "#ffffff", 700)
        subtitle = translate("Same image and settings; only the wavelet changes.", language)
        self._text(
            f"{subtitle}  •  {translate('Level', language)} {level}  •  "
            f"{translate('Step', language)} {step:g}  •  {translate('Original size', language)} {original_size}",
            28, 47, 11, "#b9c9df",
        )
        panel_colors = ("#64c8ff", "#ffbd68", "#c49aff")
        for index, result in enumerate(results):
            wavelet = str(result["wavelet"])
            x = 28 + index * 390
            panel = QtCore.QRectF(x, 92, 364, 564)
            path = QtGui.QPainterPath()
            path.addRoundedRect(panel, 12, 12)
            frame = self.scene().addPath(path, QtGui.QPen(QtGui.QColor(panel_colors[index]), 1.5), QtGui.QBrush(QtGui.QColor("#141f30")))
            tooltip = (
                f"{wavelet}\nMSE: {float(result['mse']):.4f}\nPSNR: {float(result['psnr']):.2f} dB\n"
                f"SSIM: {float(result['ssim']):.4f}\nBPP: {float(result['bpp']):.3f}\n"
                f"{translate('Compression ratio', language)}: {float(result['ratio']):.2f}×\n"
                f"{translate('Original size', language)}: {format_file_size(result.get('original_file_size'))}\n"
                f"{translate('Encoded size', language)}: {format_file_size(result.get('file_size'))}"
            )
            frame.setToolTip(tooltip)
            self._text(wavelet, x + 18, 105, 16, panel_colors[index], 700).setToolTip(tooltip)
            badges: list[str] = []
            if wavelet == best_quality:
                badges.append(translate("Best quality", language))
            if wavelet == smallest_file:
                badges.append(translate("Smallest file", language))
            if badges:
                badge = self._text(" • ".join(badges), x + 104, 109, 9, "#9de2bd", 600)
                badge.setToolTip(tooltip)

            image_panel = QtCore.QRectF(x + 12, 148, 340, 348)
            pixmap = array_to_pixmap(np.asarray(result["image"]))
            scaled = pixmap.scaled(
                int(image_panel.width()), int(image_panel.height()),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            image_x = image_panel.x() + (image_panel.width() - scaled.width()) / 2.0
            image_y = image_panel.y() + (image_panel.height() - scaled.height()) / 2.0
            pixmap_item = self.scene().addPixmap(scaled)
            pixmap_item.setPos(image_x, image_y)
            pixmap_item.setToolTip(tooltip)
            self.scene().addRect(
                QtCore.QRectF(image_x, image_y, scaled.width(), scaled.height()),
                QtGui.QPen(QtGui.QColor("#526984"), 1.0),
            ).setToolTip(tooltip)

            metrics = (
                ("MSE", f"{float(result['mse']):.4f}"),
                ("PSNR", f"{float(result['psnr']):.2f} dB"),
                ("SSIM", f"{float(result['ssim']):.4f}"),
                ("BPP", f"{float(result['bpp']):.3f}"),
                (translate("Compression ratio", language), f"{float(result['ratio']):.2f}×"),
                (translate("Encoded size", language), format_file_size(result.get("file_size"))),
            )
            for metric_index, (label, value) in enumerate(metrics):
                column, row = metric_index % 2, metric_index // 2
                metric_x, metric_y = x + 18 + column * 170, 502 + row * 48
                self._text(label, metric_x, metric_y, 8, "#8fa3bd", 600).setToolTip(tooltip)
                self._text(value, metric_x, metric_y + 17, 11, "#f5f9ff", 700).setToolTip(tooltip)

        self.scene().setSceneRect(self._SCENE_RECT)
        self._auto_fit = True
        self._zoom = 1.0
        self.fit_to_view()

    def fit_to_view(self) -> None:
        rect = self.scene().sceneRect()
        if not rect.isEmpty():
            self.resetTransform()
            self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(rect.center())

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.scene().items():
            factor = 1.14 if event.angleDelta().y() > 0 else 1 / 1.14
            next_zoom = max(0.4, min(5.0, self._zoom * factor))
            self.scale(next_zoom / self._zoom, next_zoom / self._zoom)
            self._zoom = next_zoom
            self._auto_fit = False
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_to_view()


def transform_tree_image(array: np.ndarray, codec: str = "dwt", wavelet: str = "db4", level: int = 2,
                         language: str = "en") -> Image.Image:
    """Visualize the actual DCT/DWT analysis path for any loaded image.

    JPEG/JPEG2000 inputs are first decoded by Pillow and then analysed.  This
    intentionally shows the educational transform pipeline rather than
    pretending to expose a vendor-specific bitstream's private coefficients.
    """
    plane = _luminance_plane(array)
    codec = str(codec).lower()
    if codec in {"jpeg", "dct"}:
        title = f"{translate('Transform tree', language)}  •  DCT (8×8)"
        transform_name = "DCT (8×8 blocks)"
        # A block-DCT preview that is faithful to the codec's 8x8 transform.
        from .transforms import block_dct
        coefficients, _padded_shape = block_dct(plane)
        coeff_view = coefficients.reshape(-1, 8, 8)
        tile = _coefficient_tile(coeff_view[0] if len(coeff_view) else coefficients, (300, 210))
        branch_labels = ["DC", "AC horizontal", "AC vertical", "AC diagonal"]
        branch_values = [coeff_view[0][0, 0] if len(coeff_view) else 0.0,
                         np.mean(np.abs(coeff_view[:, 0, 1:])) if len(coeff_view) else 0.0,
                         np.mean(np.abs(coeff_view[:, 1:, 0])) if len(coeff_view) else 0.0,
                         np.mean(np.abs(coeff_view[:, 1:, 1:])) if len(coeff_view) else 0.0]
        transform_caption = translate("JPEG uses 8×8 DCT blocks; DCT view shows coefficient energy.", language)
        nodes = ["Input image", transform_name, "8×8 block", *branch_labels]
        node_fills = ["#1f2937", "#1d4e89", "#365314", "#7c2d12", "#7c2d12", "#7c2d12", "#7c2d12"]
    else:
        wavelet_name = str(wavelet or "db4")
        try:
            wave = pywt.Wavelet(wavelet_name)
        except (ValueError, TypeError):
            wavelet_name, wave = "db4", pywt.Wavelet("db4")
        max_level = pywt.dwt_max_level(min(plane.shape), wave.dec_len)
        selected_level = max(1, min(int(level or 1), max(1, max_level)))
        coeffs = pywt.wavedec2(plane, wavelet_name, mode="periodization", level=selected_level)
        title = f"{translate('Transform tree', language)}  •  DWT ({wavelet_name}, level {selected_level})"
        transform_name = f"DWT ({wavelet_name})"
        transform_caption = (
            translate("JPEG2000 uses wavelet analysis; this view shows LL/LH/HL/HH subbands.", language)
            + " " + translate("Each deeper level decomposes the previous LL approximation.", language)
        )
        branch_labels = []
        branch_values = []
        # The final approximation plus every detail tuple are the real
        # multilevel decomposition returned by PyWavelets.
        branch_labels.append(f"LL{selected_level}")
        branch_values.append(coeffs[0])
        for coeff_level in range(selected_level, 0, -1):
            detail = coeffs[selected_level - coeff_level + 1]
            for label, values in zip((f"LH{coeff_level}", f"HL{coeff_level}", f"HH{coeff_level}"), detail):
                branch_labels.append(label)
                branch_values.append(values)
        nodes = ["Input image", transform_name, f"Level {selected_level}", *branch_labels]
        node_fills = ["#1f2937", "#1d4e89", "#365314"] + ["#7c2d12"] * len(branch_labels)

    canvas = Image.new("RGB", (1320, 900), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 22), title, fill="white", font=_ui_font(18))
    draw.text((28, 50), transform_caption, fill="#b8c5d8", font=_ui_font(14))
    # Left: a real processing/decomposition tree.
    root_box = (30, 105, 220, 155)
    transform_box = (270, 105, 490, 155)
    _tree_node(draw, root_box, translate(nodes[0], language), fill=node_fills[0])
    _tree_node(draw, transform_box, translate(nodes[1], language), fill=node_fills[1])
    _tree_arrow(draw, (root_box[2], 130), (transform_box[0], 130))
    if codec in {"jpeg", "dct"}:
        branch_root = (545, 105, 755, 155)
        _tree_node(draw, branch_root, translate(nodes[2], language), fill=node_fills[2])
        _tree_arrow(draw, (transform_box[2], 130), (branch_root[0], 130))
        branch_count = len(branch_labels)
        cols = 2 if branch_count > 5 else max(1, branch_count)
        rows = int(math.ceil(branch_count / cols))
        for index, label in enumerate(branch_labels):
            col, row = index % cols, index // cols
            x = 805 + col * 235
            y = 94 + row * 76
            box = (x, y, x + 210, y + 50)
            _tree_node(draw, box, translate(label, language), fill=node_fills[min(index + 3, len(node_fills) - 1)])
            _tree_arrow(draw, (branch_root[2], 130), (box[0], box[1] + 25))
    else:
        # Correct multilevel DWT topology: level 1 decomposes the input;
        # only the approximation LL1 is passed into level 2, then LL2, etc.
        previous_approx: tuple[int, int, int, int] | None = None
        child_x = (775, 910, 1045, 1180)
        child_width = 125
        for decomposition_level in range(1, selected_level + 1):
            level_y = 82 + (decomposition_level - 1) * 112
            level_box = (545, level_y, 755, level_y + 50)
            _tree_node(draw, level_box, translate(f"Level {decomposition_level}", language), fill="#365314")
            if decomposition_level == 1:
                _tree_arrow(draw, (transform_box[2], 130), (level_box[0], level_y + 25))
            elif previous_approx is not None:
                _tree_arrow(draw, (previous_approx[2], previous_approx[1] + 25), (level_box[0], level_y + 25))
            child_labels = (
                f"LH{decomposition_level}", f"HL{decomposition_level}",
                f"HH{decomposition_level}", f"LL{decomposition_level}",
            )
            approx_box = None
            for child_index, child_label in enumerate(child_labels):
                box = (child_x[child_index], level_y, child_x[child_index] + child_width, level_y + 50)
                is_approx = child_label.startswith("LL")
                _tree_node(draw, box, translate(child_label, language), fill="#365314" if is_approx else "#7c2d12")
                _tree_arrow(draw, (level_box[2], level_y + 25), (box[0], box[1] + 25))
                if is_approx:
                    approx_box = box
            previous_approx = approx_box

    # Right/bottom: the coefficient evidence associated with every leaf.
    draw.text((30, 205), translate("Coefficient / subband preview", language), fill="#73b7ff", font=_ui_font(15))
    thumb_w, thumb_h = 180, 112
    gap_x, gap_y = 22, 30
    # Keep the evidence tiles below the tree so the connectors and labels are
    # readable even for a four-level decomposition.
    start_x, start_y = 30, 480
    for index, (label, values) in enumerate(zip(branch_labels, branch_values)):
        col, row = index % 6, index // 6
        x = start_x + col * (thumb_w + gap_x)
        y = start_y + row * (thumb_h + 30 + gap_y)
        tile_image = tile if codec in {"jpeg", "dct"} and index == 0 else _coefficient_tile(np.asarray(values), (thumb_w, thumb_h))
        canvas.paste(tile_image, (x, y + 20))
        draw.text((x + 4, y + 3), translate(label, language), fill="#d7e3f4", font=_ui_font(14))
    if codec in {"jpeg", "dct"}:
        draw.text((30, 850), translate("The first tile is one 8×8 coefficient block; other leaves summarize AC energy.", language), fill="#9fb0c6", font=_ui_font(14))
    else:
        draw.text((30, 850), translate("LL keeps the approximation; LH/HL/HH carry horizontal, vertical and diagonal detail.", language), fill="#9fb0c6", font=_ui_font(14))
    return canvas


class _HistogramCurveItem(QtWidgets.QGraphicsPathItem):
    """Interactive histogram curve with a clearer hover state."""

    def __init__(self, path: QtGui.QPainterPath, color: QtGui.QColor, tooltip: str) -> None:
        super().__init__(path)
        self._color = QtGui.QColor(color)
        self.setAcceptHoverEvents(True)
        self.setToolTip(tooltip)
        self.setZValue(4)
        self._apply_pen(False)

    def _apply_pen(self, hovered: bool) -> None:
        color = QtGui.QColor(self._color)
        if hovered:
            color = color.lighter(125)
        pen = QtGui.QPen(color, 4.2 if hovered else 2.7)
        pen.setCosmetic(True)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        self.setPen(pen)

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self._apply_pen(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self._apply_pen(False)
        super().hoverLeaveEvent(event)


class HistogramPreview(QtWidgets.QGraphicsView):
    """Sharp, theme-safe Qt histogram for the comparison screen.

    Axes, labels, fills, curves, legends and summary cards are individual Qt
    graphics items.  Nothing in this view is rendered into a PIL/matplotlib
    image, so text remains crisp in normal and full-screen presentation modes.
    """

    _SCENE_SIZE = QtCore.QSizeF(1040, 650)
    _PLOT_RECT = QtCore.QRectF(88, 112, 884, 390)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.scene().setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#0d1420")))
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self._auto_fit = True
        self.show_message("No image")

    def _text(self, value: str, x: float, y: float, size: int = 10, color: str = "#dce8f7",
              weight: int = 400) -> QtWidgets.QGraphicsTextItem:
        item = self.scene().addText(value)
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPointSize(size)
        font.setWeight(QtGui.QFont.Weight(weight))
        item.setFont(font)
        item.setDefaultTextColor(QtGui.QColor(color))
        item.setPos(x, y)
        return item

    def _summary_card(self, x: float, label: str, mean: float, deviation: float,
                      color: QtGui.QColor, language: str) -> None:
        rect = QtCore.QRectF(x, 535, 300, 82)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 12, 12)
        self.scene().addPath(path, QtGui.QPen(QtGui.QColor("#32445d"), 1), QtGui.QBrush(QtGui.QColor("#162235")))
        marker = QtCore.QRectF(x + 16, 554, 8, 44)
        marker_path = QtGui.QPainterPath()
        marker_path.addRoundedRect(marker, 4, 4)
        self.scene().addPath(marker_path, QtGui.QPen(QtCore.Qt.PenStyle.NoPen), QtGui.QBrush(color))
        self._text(label, x + 38, 544, 11, "#f4f8ff", 600)
        stats = f"{translate('Mean', language)}  {mean:.1f}    {translate('Std dev', language)}  {deviation:.1f}"
        self._text(stats, x + 38, 574, 10, "#aebed2")

    def show_message(self, message: str) -> None:
        self.scene().clear()
        language = str(self.property("ui_language") or "en")
        self._text(translate(message, language), 24, 24, 12, "#91a1b8")
        self.scene().setSceneRect(0, 0, self._SCENE_SIZE.width(), self._SCENE_SIZE.height())
        self._auto_fit = True
        self.fit_to_view()
        QtCore.QTimer.singleShot(0, self.fit_to_view)

    def set_histograms(self, reference: np.ndarray, candidate: np.ndarray, target: str,
                       language: str = "en") -> None:
        self.scene().clear()
        reference_luma = _luminance_plane(reference).ravel()
        candidate_luma = _luminance_plane(candidate).ravel()
        reference_counts, _ = np.histogram(reference_luma, bins=64, range=(0.0, 256.0))
        candidate_counts, _ = np.histogram(candidate_luma, bins=64, range=(0.0, 256.0))
        peak = max(1.0, float(max(reference_counts.max(initial=0), candidate_counts.max(initial=0))))
        reference_values = reference_counts.astype(np.float64) / peak
        candidate_values = candidate_counts.astype(np.float64) / peak

        original_label = translate("Original", language)
        target_label = translate(target, language)
        original_color = QtGui.QColor("#62c2ff")
        target_color = QtGui.QColor("#ff9d66")
        plot = self._PLOT_RECT

        self._text(translate("Luminance histogram", language), 34, 22, 17, "#ffffff", 700)
        self._text(translate("Original and decoded luminance distribution across 64 intensity bins.", language),
                   34, 55, 10, "#9fb1c8")

        # Legend is built from native vector swatches and Qt text.
        legend_x = 700
        for offset, label, color in ((0, original_label, original_color), (138, target_label, target_color)):
            swatch = QtCore.QRectF(legend_x + offset, 54, 24, 5)
            swatch_path = QtGui.QPainterPath()
            swatch_path.addRoundedRect(swatch, 2.5, 2.5)
            self.scene().addPath(swatch_path, QtGui.QPen(QtCore.Qt.PenStyle.NoPen), QtGui.QBrush(color))
            self._text(label, legend_x + offset + 32, 43, 10, "#dce8f7", 600)

        grid_pen = QtGui.QPen(QtGui.QColor("#2a3a50"), 1)
        grid_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        grid_pen.setCosmetic(True)
        axis_pen = QtGui.QPen(QtGui.QColor("#758ba7"), 1.4)
        axis_pen.setCosmetic(True)
        for index in range(5):
            fraction = index / 4.0
            y = plot.bottom() - fraction * plot.height()
            self.scene().addLine(plot.left(), y, plot.right(), y, grid_pen)
            label = self._text(f"{int(fraction * 100)}%", 42, y - 12, 9, "#8fa3bb")
            label.setToolTip(translate("Normalized pixel count", language))
        for value in (0, 64, 128, 192, 255):
            x = plot.left() + (value / 255.0) * plot.width()
            self.scene().addLine(x, plot.top(), x, plot.bottom(), grid_pen)
            self._text(str(value), x - (8 if value < 100 else 13), plot.bottom() + 8, 9, "#8fa3bb")
        self.scene().addLine(plot.left(), plot.bottom(), plot.right(), plot.bottom(), axis_pen)
        self.scene().addLine(plot.left(), plot.top(), plot.left(), plot.bottom(), axis_pen)
        self._text(translate("Intensity", language), plot.center().x() - 34, plot.bottom() + 38, 10, "#b9c9df", 600)
        self._text(translate("Normalized pixel count", language), 24, 82, 9, "#8fa3bb")

        def add_curve(values: np.ndarray, color: QtGui.QColor, label: str,
                      source_values: np.ndarray) -> None:
            points = [
                QtCore.QPointF(
                    plot.left() + index * plot.width() / max(1, len(values) - 1),
                    plot.bottom() - float(value) * plot.height(),
                )
                for index, value in enumerate(values)
            ]
            line_path = QtGui.QPainterPath(points[0])
            for point in points[1:]:
                line_path.lineTo(point)
            fill_path = QtGui.QPainterPath(QtCore.QPointF(plot.left(), plot.bottom()))
            fill_path.lineTo(points[0])
            for point in points[1:]:
                fill_path.lineTo(point)
            fill_path.lineTo(plot.right(), plot.bottom())
            fill_path.closeSubpath()
            fill_color = QtGui.QColor(color)
            fill_color.setAlpha(42)
            self.scene().addPath(fill_path, QtGui.QPen(QtCore.Qt.PenStyle.NoPen), QtGui.QBrush(fill_color)).setZValue(2)
            maximum_bin = int(np.argmax(values)) * 4
            tooltip = (
                f"{label}\n{translate('Peak intensity', language)}: {maximum_bin}\n"
                f"{translate('Mean', language)}: {float(np.mean(source_values)):.1f}\n"
                f"{translate('Std dev', language)}: {float(np.std(source_values)):.1f}"
            )
            self.scene().addItem(_HistogramCurveItem(line_path, color, tooltip))

        add_curve(reference_values, original_color, original_label, reference_luma)
        add_curve(candidate_values, target_color, target_label, candidate_luma)
        self._summary_card(88, original_label, float(np.mean(reference_luma)), float(np.std(reference_luma)),
                           original_color, language)
        self._summary_card(672, target_label, float(np.mean(candidate_luma)), float(np.std(candidate_luma)),
                           target_color, language)
        self.scene().setSceneRect(0, 0, self._SCENE_SIZE.width(), self._SCENE_SIZE.height())
        self._auto_fit = True
        self.fit_to_view()
        # A global theme switch can briefly relayout the viewport after this
        # method returns. Refit once the queued style/layout events settle.
        QtCore.QTimer.singleShot(0, self.fit_to_view)

    def fit_to_view(self) -> None:
        rect = self.scene().sceneRect()
        if not rect.isEmpty() and self.viewport().width() > 2 and self.viewport().height() > 2:
            self.resetTransform()
            self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(rect.center())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_to_view()


class _RateDistortionPointItem(QtWidgets.QGraphicsEllipseItem):
    """Hoverable measured point on the native Qt PSNR–BPP graph."""

    def __init__(self, center: QtCore.QPointF, color: QtGui.QColor, tooltip: str,
                 current: bool = False, ring: bool = False) -> None:
        radius = (10.0 if current else 8.0) if ring else (7.0 if current else 5.0)
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(center)
        self._color = QtGui.QColor(color)
        self._current = current
        self._ring = ring
        self.setAcceptHoverEvents(True)
        self.setToolTip(tooltip)
        self.setZValue(8)
        self._apply_style(False)

    def _apply_style(self, hovered: bool) -> None:
        color = self._color.lighter(125) if hovered else self._color
        width = 3.0 if self._current or hovered else 2.0
        self.setPen(QtGui.QPen(color if self._ring else QtGui.QColor("#ffffff"), width))
        self.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush) if self._ring else QtGui.QBrush(color))

    def hoverEnterEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self._apply_style(True)
        self.setScale(1.18)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        self.setScale(1.0)
        self._apply_style(False)
        super().hoverLeaveEvent(event)


class RateDistortionPreview(QtWidgets.QGraphicsView):
    """Rate-distortion scatter: measured BPP on X, measured PSNR on Y."""

    _SCENE_SIZE = QtCore.QSizeF(1040, 650)
    _PLOT_RECT = QtCore.QRectF(96, 144, 868, 328)
    _CODEC_COLORS = {"dct": "#67c7ff", "dwt": "#ffbd68"}
    _AXIS_COLOR = "#b9c9df"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.scene().setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#0d1420")))
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self._auto_fit = True
        self.show_message("Run Encode + Decode to generate the PSNR–BPP graph.")

    def _text(self, value: str, x: float, y: float, size: int = 10, color: str = "#dce8f7",
              weight: int = 400) -> QtWidgets.QGraphicsTextItem:
        item = self.scene().addText(value)
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setPointSize(size)
        font.setWeight(QtGui.QFont.Weight(weight))
        item.setFont(font)
        item.setDefaultTextColor(QtGui.QColor(color))
        item.setPos(x, y)
        return item

    def show_message(self, message: str, language: str | None = None) -> None:
        self.scene().clear()
        selected_language = language or str(self.property("ui_language") or "en")
        self._text(translate(message, selected_language), 24, 24, 12, "#91a1b8")
        self.scene().setSceneRect(0, 0, self._SCENE_SIZE.width(), self._SCENE_SIZE.height())
        self._auto_fit = True
        self.fit_to_view()

    @staticmethod
    def _format_bpp(value: float) -> str:
        return f"{value:.2f}" if value >= 1.0 else f"{value:.3f}"

    @staticmethod
    def _format_psnr(value: float) -> str:
        return "∞" if np.isposinf(value) else f"{value:.2f}"

    def _summary_card(self, x: float, title: str, point: dict, accent: str,
                      language: str) -> None:
        rect = QtCore.QRectF(x, 535, 390, 82)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 12, 12)
        self.scene().addPath(path, QtGui.QPen(QtGui.QColor("#32445d"), 1), QtGui.QBrush(QtGui.QColor("#162235")))
        self.scene().addRect(QtCore.QRectF(x + 16, 554, 8, 44), QtGui.QPen(QtCore.Qt.PenStyle.NoPen),
                             QtGui.QBrush(QtGui.QColor(accent)))
        self._text(translate(title, language), x + 38, 544, 11, "#f4f8ff", 600)
        self._text(f"{float(point['bpp']):.3f} BPP", x + 38, 570, 10, accent)
        self._text(f"{self._format_psnr(float(point['psnr']))} dB", x + 174, 570, 10, accent)
        self._text(f"{translate('Target BPP', language)}: {float(point.get('target_bpp', 0)):g} • {point.get('parameter', '')}",
                   x + 38, 593, 8, "#aebed2")

    def set_results(self, results: list[dict], codec: str, language: str = "en") -> None:
        self.scene().clear()
        points = [item for item in results if item.get("codec") in self._CODEC_COLORS
                  and np.isfinite(float(item.get("bpp", np.nan)))
                  and float(item["bpp"]) >= 0
                  and (np.isfinite(float(item.get("psnr", np.nan)))
                       or np.isposinf(float(item.get("psnr", np.nan))))]
        if not points:
            self.show_message("Run Encode + Decode to generate the PSNR–BPP graph.", language)
            return
        bpp_max = max(0.1, max(float(item["bpp"]) for item in points) * 1.12)
        finite_qualities = [float(item["psnr"]) for item in points if np.isfinite(float(item["psnr"]))]
        has_exact = any(np.isposinf(float(item["psnr"])) for item in points)
        psnr_min = math.floor((min(finite_qualities, default=25.0) - 1.5) / 5.0) * 5.0
        psnr_max = max(psnr_min + 5.0, math.ceil((max(finite_qualities, default=45.0) + 1.5) / 5.0) * 5.0)
        plot = self._PLOT_RECT

        self._text(translate("PSNR–BPP graph", language), 34, 20, 17, "#ffffff", 700)
        subtitle = translate("Same image, shared BPP targets; dots show achieved BPP and PSNR.", language)
        self._text(subtitle, 34, 54, 10, "#9fb1c8")
        for x, label in ((105, "DCT"), (260, "DWT")):
            color = self._CODEC_COLORS[label.lower()]
            ring = label == "DWT"
            self.scene().addEllipse(x, 94, 10, 10, QtGui.QPen(QtGui.QColor(color), 2),
                                   QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush) if ring else QtGui.QBrush(QtGui.QColor(color)))
            self._text(translate(label, language), x + 18, 85, 10, color, 600)
        self._text(translate("Larger markers: reference BPP target", language), 400, 85, 9, self._AXIS_COLOR)

        grid_pen = QtGui.QPen(QtGui.QColor("#2a3a50"), 1)
        grid_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        grid_pen.setCosmetic(True)
        for index in range(6):
            fraction = index / 5.0
            x = plot.left() + fraction * plot.width()
            y = plot.bottom() - fraction * plot.height()
            self.scene().addLine(x, plot.top(), x, plot.bottom(), grid_pen)
            self.scene().addLine(plot.left(), y, plot.right(), y, grid_pen)
            tick = self._text(self._format_bpp(fraction * bpp_max), 0, plot.bottom() + 7, 9, self._AXIS_COLOR)
            tick.setX(x - tick.boundingRect().width() / 2)
            self._text(f"{psnr_min + fraction * (psnr_max - psnr_min):.0f}", 49, y - 12, 9, self._AXIS_COLOR)
        self.scene().addLine(plot.left(), plot.bottom(), plot.right(), plot.bottom(), QtGui.QPen(QtGui.QColor(self._AXIS_COLOR), 1.4))
        self.scene().addLine(plot.left(), plot.top(), plot.left(), plot.bottom(), QtGui.QPen(QtGui.QColor(self._AXIS_COLOR), 1.4))
        x_label = self._text(translate("Bits per pixel (BPP)", language), 0, plot.bottom() + 34, 10, self._AXIS_COLOR, 600)
        x_label.setX(plot.center().x() - x_label.boundingRect().width() / 2)
        self._text("PSNR (dB)", 10, plot.center().y() + 42, 10, self._AXIS_COLOR, 600).setRotation(-90)
        if has_exact:
            # Infinity is not a finite dB value. Give exact reconstructions a
            # separate row above the numeric scale, rather than clipping them.
            self._text("∞", 49, plot.top() - 37, 10, self._AXIS_COLOR)
            self._text(translate("∞ = exact reconstruction", language), 735, 85, 9, self._AXIS_COLOR)

        for index, item in enumerate(points, start=1):
            current = bool(item.get("highlighted"))
            method = str(item["codec"])
            x = plot.left() + float(item["bpp"]) / bpp_max * plot.width()
            tooltip = (
                f"{method.upper()} • {item.get('parameter', '')}\n"
                f"{translate('Target BPP', language)}: {float(item.get('target_bpp', 0)):g}\n"
                f"BPP: {float(item['bpp']):.4f}\nPSNR: {self._format_psnr(float(item['psnr']))} dB\n"
                f"{translate('Encoded size', language)}: {format_file_size(item.get('file_size'))}"
            )
            if abs(float(item["bpp"]) - float(item.get("target_bpp", item["bpp"]))) > max(0.002, float(item.get("target_bpp", 0)) * 0.02):
                tooltip += "\n" + translate("Target not reached; the plotted BPP is the actual file rate.", language)
            psnr_y = (plot.top() - 24 if np.isposinf(float(item["psnr"])) else
                      plot.bottom() - (float(item["psnr"]) - psnr_min) / (psnr_max - psnr_min) * plot.height())
            color = self._CODEC_COLORS[method]
            marker = _RateDistortionPointItem(QtCore.QPointF(x, psnr_y), QtGui.QColor(color), tooltip, current, ring=method == "dwt")
            marker.setData(0, "rate_distortion")
            marker.setData(1, index)
            marker.setData(2, float(item["bpp"]))
            marker.setData(3, float(item["psnr"]))
            marker.setData(4, method)
            marker.setZValue((10 if current else 8) + (1 if method == "dwt" else 0))
            self.scene().addItem(marker)

        for x, method in ((96, "dct"), (574, "dwt")):
            series = [item for item in points if item["codec"] == method]
            if series:
                selected = next((item for item in series if item.get("highlighted")), series[len(series) // 2])
                self._summary_card(x, method.upper(), selected, self._CODEC_COLORS[method], language)
        self.scene().setSceneRect(0, 0, self._SCENE_SIZE.width(), self._SCENE_SIZE.height())
        self._auto_fit = True
        self.fit_to_view()
        QtCore.QTimer.singleShot(0, self.fit_to_view)

    def fit_to_view(self) -> None:
        rect = self.scene().sceneRect()
        if not rect.isEmpty() and self.viewport().width() > 2 and self.viewport().height() > 2:
            self.resetTransform()
            self.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.centerOn(rect.center())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_to_view()


def grayscale8_preview(array: np.ndarray) -> np.ndarray:
    """Return a display-only 0..255 plane without stretching each image's contrast.

    RGB uses rounded 0.299R + 0.587G + 0.114B. uint16 has a fixed full-range
    mapping, shared by both panels; source/decoded codec arrays are never edited.
    """
    source = np.asarray(array)
    values = source.astype(np.float64)
    if source.dtype == np.uint16:
        values /= 257.0
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[..., 0]
    elif values.ndim == 3 and values.shape[2] >= 3:
        values = values[..., :3] @ np.array([0.299, 0.587, 0.114])
    elif values.ndim != 2:
        raise ValueError("Expected a grayscale or RGB image")
    values = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(np.rint(values), 0, 255).astype(np.uint8))


class GrayscaleComparisonPreview(QtWidgets.QWidget):
    """Two grayscale previews with native, theme-aware Qt captions."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.heading = QtWidgets.QLabel("8-bit grayscale")
        self.heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(self.heading)
        self.description = QtWidgets.QLabel("256 gray levels • 0 = black • 255 = white • 8 bits/pixel")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        panels = QtWidgets.QHBoxLayout()
        panels.setSpacing(12)
        self.previews: list[ImagePreview] = []
        self.captions: list[QtWidgets.QLabel] = []
        self.dimensions: list[QtWidgets.QLabel] = []
        for _ in range(2):
            panel = QtWidgets.QFrame()
            panel.setObjectName("metricCard")
            panel_layout = QtWidgets.QVBoxLayout(panel)
            panel_layout.setContentsMargins(12, 10, 12, 12)
            caption = QtWidgets.QLabel()
            caption.setStyleSheet("font-size: 16px; font-weight: 600;")
            dimension = QtWidgets.QLabel()
            preview = ImagePreview()
            preview.setMinimumSize(120, 120)
            panel_layout.addWidget(caption)
            panel_layout.addWidget(dimension)
            panel_layout.addWidget(preview, 1)
            panels.addWidget(panel, 1)
            self.previews.append(preview)
            self.captions.append(caption)
            self.dimensions.append(dimension)
        layout.addLayout(panels, 1)
        self.notice = QtWidgets.QLabel(
            "Preview only: files and compression are unchanged. Metrics above describe the original encode/decode result."
        )
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.set_images(None, None)

    def set_images(self, reference: np.ndarray | None, candidate: np.ndarray | None,
                   target: str = "Decoded", language: str = "en") -> None:
        self.heading.setText(translate("8-bit grayscale", language))
        self.description.setText(translate("256 gray levels • 0 = black • 255 = white • 8 bits/pixel", language))
        self.notice.setText(translate(
            "Preview only: files and compression are unchanged. Metrics above describe the original encode/decode result.",
            language,
        ))
        for index, (label, array) in enumerate((("Original", reference), (target, candidate))):
            self.captions[index].setText(translate(label, language))
            self.dimensions[index].setToolTip("")
            self.dimensions[index].setProperty("ui_tooltip_source", "")
            if array is None:
                self.dimensions[index].setText("—")
                message = "Run Encode + Decode to compare this image." if reference is not None else "No image"
                self.previews[index].clear_image(translate(message, language))
                self.previews[index].setToolTip("")
                self.previews[index].setProperty("ui_tooltip_source", "")
            else:
                gray = grayscale8_preview(array)
                self.dimensions[index].setText(f"{gray.shape[1]} × {gray.shape[0]} • uint8 • 0–255")
                self.previews[index].set_array(gray)
                tooltip = "Gray = round(0.299R + 0.587G + 0.114B). No automatic contrast stretching."
                self.previews[index].setProperty("ui_tooltip_source", tooltip)
                self.previews[index].setToolTip(translate(tooltip, language))
                if np.asarray(array).dtype == np.uint16:
                    note = "16-bit input: 0–65535 maps to 0–255 for this preview only."
                    self.dimensions[index].setProperty("ui_tooltip_source", note)
                    self.dimensions[index].setToolTip(translate(note, language))
        QtCore.QTimer.singleShot(0, self.fit_to_view)

    def fit_to_view(self) -> None:
        for preview in self.previews:
            preview.fit_to_view()


class ImagePreview(QtWidgets.QGraphicsView):
    """TR: Fit, zoom, pan ve fareyle ROI secimi destekleyen goruntu alani.
    EN: Image area with fit, zoom, pan, and mouse ROI selection support.
    """

    roiSelected = QtCore.Signal(int, int, int, int)

    def __init__(self, allow_roi: bool = False, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.RenderHint.SmoothPixmapTransform | QtGui.QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.allow_roi = allow_roi
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._roi_items: list[QtWidgets.QGraphicsRectItem] = []
        self._source_shape: tuple[int, int] | None = None
        self._pixmap_size = QtCore.QSize()
        self._drag_start: QtCore.QPointF | None = None
        self._zoom = 1.0
        self._auto_fit = True
        self.clear_image()

    @property
    def source_shape(self) -> tuple[int, int] | None:
        return self._source_shape

    def set_array(self, array: np.ndarray | None, roi_boxes: list[tuple[int, int, int, int]] | None = None) -> None:
        if array is None:
            self.clear_image()
            return
        self._source_shape = tuple(np.asarray(array).shape[:2])
        self._set_pixmap(array_to_pixmap(array), roi_boxes or [])

    def set_pil(self, image: Image.Image | None) -> None:
        if image is None:
            self.clear_image()
            return
        self._source_shape = (image.height, image.width)
        self._set_pixmap(pil_to_pixmap(image), [])

    def _set_pixmap(self, pixmap: QtGui.QPixmap, roi_boxes: list[tuple[int, int, int, int]]) -> None:
        self.scene().clear()
        self._roi_items.clear()
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._pixmap_size = pixmap.size()
        self.scene().setSceneRect(QtCore.QRectF(pixmap.rect()))
        self._add_roi_boxes(roi_boxes)
        self._auto_fit = True
        self._zoom = 1.0
        self._fit_scene()

    def _fit_scene(self) -> None:
        """Reset stale transforms before fitting a newly rendered comparison."""
        if not self._pixmap_item:
            return
        scene_rect = self.scene().sceneRect()
        if scene_rect.isEmpty():
            return
        self.resetTransform()
        self.fitInView(scene_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(scene_rect.center())

    def clear_image(self, message: str = "No image") -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            configure_application_font(app)
        self.scene().clear()
        self._pixmap_item = None
        self._roi_items.clear()
        self._source_shape = None
        self._pixmap_size = QtCore.QSize()
        placeholder = self.scene().addText(message)
        placeholder.setFont(QtWidgets.QApplication.font())
        placeholder.setDefaultTextColor(QtGui.QColor("#8f9aaa"))
        placeholder.setPos(12, 12)

    def show_message(self, message: str) -> None:
        """Clear the canvas with an explicit explanation instead of a blank view."""
        self.clear_image(translate(message, str(self.property("ui_language") or "en")))

    def set_roi_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        for item in self._roi_items:
            self.scene().removeItem(item)
        self._roi_items.clear()
        self._add_roi_boxes(boxes)

    def _add_roi_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        if not self._source_shape or self._pixmap_size.isEmpty():
            return
        source_h, source_w = self._source_shape
        scale_x = self._pixmap_size.width() / max(1, source_w)
        scale_y = self._pixmap_size.height() / max(1, source_h)
        pen = QtGui.QPen(QtGui.QColor("#ff4040"), max(2.0, 2.0 / max(self._zoom, 0.1)))
        for x, y, width, height in boxes:
            item = self.scene().addRect(x * scale_x, y * scale_y, width * scale_x, height * scale_y, pen)
            item.setZValue(2)
            self._roi_items.append(item)

    def fit_to_view(self) -> None:
        if self._pixmap_item:
            self._auto_fit = True
            self._zoom = 1.0
            self._fit_scene()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self._pixmap_item:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        requested_factor = 1.15 if delta > 0 else 1 / 1.15
        previous_zoom = self._zoom
        next_zoom = max(0.2, min(8.0, previous_zoom * requested_factor))
        actual_factor = next_zoom / previous_zoom
        self._auto_fit = False
        self._zoom = next_zoom
        # The prior implementation clamped _zoom but continued to scale the
        # view.  Repeated wheel input could therefore pan the image completely
        # out of the visible scene despite the advertised zoom limit.
        if abs(actual_factor - 1.0) > 1e-9:
            self.scale(actual_factor, actual_factor)
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit and self._pixmap_item:
            self._fit_scene()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.allow_roi and event.button() == QtCore.Qt.MouseButton.LeftButton and self._pixmap_item:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self._drag_start = self.mapToScene(event.position().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_start is not None:
            end = self.mapToScene(event.position().toPoint())
            rect = QtCore.QRectF(self._drag_start, end).normalized()
            rect = rect.intersected(self.scene().sceneRect())
            if rect.width() > 2 and rect.height() > 2 and self._source_shape:
                source_h, source_w = self._source_shape
                sx = source_w / max(1.0, self._pixmap_size.width())
                sy = source_h / max(1.0, self._pixmap_size.height())
                self.roiSelected.emit(int(rect.x() * sx), int(rect.y() * sy), max(1, int(rect.width() * sx)), max(1, int(rect.height() * sy)))
            self._drag_start = None
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            event.accept()
            return
        super().mouseReleaseEvent(event)


def comparison_image(reference: np.ndarray, candidate: np.ndarray, mode: str, target: str, position: float,
                     stages: list[np.ndarray] | None = None, stage_labels: list[str] | None = None,
                     language: str = "en") -> Image.Image:
    """TR: Orijinal/decoded karsilastirmasi icin PIL gorseli uretir.
    EN: Build a PIL image for original/decoded comparison modes.
    """
    reference_image = display_image(reference)
    candidate_image = display_image(candidate)
    if candidate_image.size != reference_image.size:
        candidate_image = candidate_image.resize(reference_image.size, Image.Resampling.BILINEAR)
    if mode in {"Side-by-side", "Yan yana"}:
        left, right = reference_image.copy(), candidate_image.copy()
        height = max(left.height, right.height)
        label_height = 52
        canvas = Image.new("RGB", (left.width + right.width + 18, height + label_height), "#202020")
        canvas.paste(left, (0, label_height))
        canvas.paste(right, (left.width + 18, label_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), translate("Original", language), fill="white", font=_ui_font(24))
        draw.text((left.width + 26, 10), translate(target, language), fill="#73b7ff", font=_ui_font(24))
        return canvas
    if mode in {"Progressive stages", "Aşamalı aşamalar"} and stages:
        images = [display_image(item) for item in stages[:4]]
        base_width = min(820, max(image.width for image in images))
        tiles: list[Image.Image] = []
        for image in images:
            scale = base_width / max(1, image.width)
            size = (base_width, max(1, int(image.height * scale)))
            tiles.append(image.resize(size, Image.Resampling.BILINEAR))
        tile_height = max(image.height for image in tiles)
        # A comparison canvas is usually fitted to the preview viewport.  A
        # 24 px Unicode font and a generous caption band keep the four stage
        # labels readable without requiring the presenter to zoom in.
        label_height, gap = 54, 18
        canvas = Image.new("RGB", (base_width * 2 + gap * 3, (tile_height + label_height) * 2 + gap * 3), "#202020")
        draw = ImageDraw.Draw(canvas)
        default_labels = ["Original", "Light compression", "Medium compression", target]
        labels = stage_labels or default_labels
        for index, image in enumerate(tiles):
            column, row = index % 2, index // 2
            x = gap + column * (base_width + gap)
            y = gap + row * (tile_height + label_height + gap)
            canvas.paste(image, (x, y + label_height))
            draw.rectangle((x, y + label_height, x + image.width - 1, y + label_height + image.height - 1), outline="#52647d", width=2)
            caption = translate(labels[index] if index < len(labels) else default_labels[index], language)
            draw.text((x + 10, y + 8), caption, fill="#e7f0ff", font=_ui_font(24))
        return canvas
    if mode in {"Slider", "Kaydırıcı"}:
        split = int(reference_image.width * min(1.0, max(0.0, position)))
        result = reference_image.copy()
        if split < candidate_image.width:
            result.paste(candidate_image.crop((split, 0, candidate_image.width, candidate_image.height)), (split, 0))
        draw = ImageDraw.Draw(result)
        draw.line((split, 0, split, result.height), fill="#00e676", width=3)
        draw.text((8, 8), "Original", fill="white")
        draw.text((max(8, split + 8), 8), target, fill="#00e676")
        return result
    reference_rgb = np.asarray(reference_image, dtype=np.float32)
    candidate_rgb = np.asarray(candidate_image, dtype=np.float32)
    if mode in {"Error heatmap", "Hata ısı haritası"}:
        error = np.mean(np.abs(reference_rgb - candidate_rgb), axis=2)
        intensity = np.clip(error * 5.0, 0, 255).astype(np.uint8)
        heat = np.repeat(intensity[..., None], 3, axis=2)
        return Image.fromarray(heat.astype(np.uint8), mode="RGB")
    # Histogram uses HistogramPreview's native Qt scene; raster comparison
    # rendering deliberately has no chart fallback.
    return candidate_image.copy()


def difference_image(reference: np.ndarray, candidate: np.ndarray) -> Image.Image:
    """Create an auto-scaled difference heatmap suitable for small JPEG errors.

    Raw pixel differences are often 0--3 for a high-quality JPEG and look like
    a nearly black image.  The 99th-percentile scaling makes those real but
    subtle differences visible without changing the underlying metrics.
    """
    reference_image = display_image(reference)
    candidate_image = display_image(candidate)
    if candidate_image.size != reference_image.size:
        candidate_image = candidate_image.resize(reference_image.size, Image.Resampling.BILINEAR)
    delta = np.mean(
        np.abs(np.asarray(reference_image, dtype=np.float32) - np.asarray(candidate_image, dtype=np.float32)),
        axis=2,
    )
    scale = max(float(np.percentile(delta, 99.0)), 1.0)
    intensity = np.clip(delta * (255.0 / scale), 0, 255)
    # Black means equal; brighter gray/white means a larger absolute error.
    heat = np.repeat(intensity.astype(np.uint8)[..., None], 3, axis=2)
    return Image.fromarray(heat.astype(np.uint8), mode="RGB")
