"""TR: Görsel önizleme ve karşılaştırma çizimlerini yönetir. / EN: Manages visual preview and comparison rendering."""

from __future__ import annotations

from .gui_support import *  # noqa: F401,F403 - shared GUI dependencies / ortak GUI bağımlılıkları


class GuiVisualsMixin:
    """TR: Görsel önizleme ve karşılaştırma çizimlerini yönetir. / EN: Manages visual preview and comparison rendering."""

    def _refresh_metrics(self) -> None:
        if self.original_array is None or self.decoded_array is None:
            return
        values = (
            f"MSE: {mse(self.original_array, self.decoded_array):.4f}    "
            f"PSNR: {psnr(self.original_array, self.decoded_array):.2f} dB    "
            f"SSIM: {ssim(self.original_array, self.decoded_array):.4f}"
        )
        if self.restored_array is not None:
            values += (
                f"\nRestored MSE: {mse(self.original_array, self.restored_array):.4f}    "
                f"PSNR: {psnr(self.original_array, self.restored_array):.2f} dB    "
                f"SSIM: {ssim(self.original_array, self.restored_array):.4f}"
            )
        if self.last_encode_info:
            values += (
                f"    |    Sıkıştırma: {self.last_encode_info['compression_ratio']:.2f}x    "
                f"|    BPP: {self.last_encode_info['bits_per_pixel']:.3f}"
            )
        if self.current_roi_mask is not None:
            regions = region_metrics(self.original_array, self.decoded_array, self.current_roi_mask)
            values += (
                f"    |    ROI PSNR: {regions['roi_psnr']:.2f} dB"
                f"    |    Arka plan PSNR: {regions['background_psnr']:.2f} dB"
            )
        self.metrics_text.set(values)

    def _show_image(self, array: np.ndarray, label, which: str) -> None:
        image = Image.fromarray(array)
        if image.mode not in {"L", "RGB"}:
            image = image.convert("L")
        if which == "original" and self.current_roi_mask is not None:
            mask = Image.fromarray(
                np.clip(np.asarray(self.current_roi_mask, dtype=np.float32) * 180.0, 0, 180).astype(np.uint8),
                mode="L",
            ).resize(image.size, Image.Resampling.BILINEAR)
            overlay = Image.new("RGBA", image.size, (255, 40, 40, 0))
            overlay.putalpha(mask)
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        image.thumbnail((500, 430), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        if which == "original":
            self.original_photo = photo
            self.original_display_size = image.size
            self.original_source_shape = tuple(array.shape[:2])
            self.original_canvas.delete("all")
            self.original_canvas.create_image(0, 0, anchor="nw", image=photo, tags="preview")
            self.original_canvas.configure(scrollregion=(0, 0, image.width, image.height))
            self._draw_roi_boxes()
        elif which == "decoded":
            self.decoded_photo = photo
            label.configure(image=photo, text="")
        elif which == "restored":
            self.restored_photo = photo
            label.configure(image=photo, text="")
        elif which == "difference":
            self.difference_photo = photo
            label.configure(image=photo, text="")

    def _show_difference(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        reference_array = np.asarray(reference, dtype=np.float32)
        candidate_array = np.asarray(candidate, dtype=np.float32)
        difference = np.clip(np.abs(reference_array - candidate_array) * 4.0, 0, 255).astype(np.uint8)
        self._show_image(difference, self.difference_label, "difference")

    @staticmethod
    def _display_array(array: np.ndarray) -> Image.Image:
        values = np.asarray(array)
        if values.ndim == 2:
            if values.dtype != np.uint8:
                values = np.clip(values.astype(np.float32) / max(1.0, float(values.max() or 1.0)) * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(values, mode="L").convert("RGB")
        if values.ndim != 3:
            raise ValueError("Görüntü karşılaştırması 2D veya 3D dizi bekler")
        values = values[..., :3]
        if values.dtype != np.uint8:
            maximum = float(np.iinfo(values.dtype).max) if np.issubdtype(values.dtype, np.integer) else max(1.0, float(values.max()))
            values = np.clip(values.astype(np.float32) / maximum * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(values, mode="RGB")

    @staticmethod
    def _fit_display(image: Image.Image, width: int, height: int) -> Image.Image:
        result = image.copy()
        result.thumbnail((max(1, width), max(1, height)), Image.Resampling.LANCZOS)
        return result

    def _comparison_candidate(self) -> np.ndarray | None:
        if self.comparison_target.get() == "Restored" and self.restored_array is not None:
            return self.restored_array
        return self.decoded_array

    def _set_comparison_image(self, image: Image.Image) -> None:
        if not hasattr(self, "comparison_canvas"):
            return
        photo = ImageTk.PhotoImage(image)
        self.comparison_photo = photo
        canvas = self.comparison_canvas
        canvas.delete("all")
        zoom = max(0.5, min(3.0, float(self.comparison_zoom.get())))
        if abs(zoom - 1.0) > 1e-3:
            image = image.resize(
                (max(1, int(round(image.width * zoom))), max(1, int(round(image.height * zoom)))),
                Image.Resampling.LANCZOS,
            )
            photo = ImageTk.PhotoImage(image)
            self.comparison_photo = photo
        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
        left = max(0, (width - image.width) // 2)
        top = max(0, (height - image.height) // 2)
        canvas.create_image(left, top, anchor="nw", image=photo)
        canvas.configure(scrollregion=(0, 0, max(width, left + image.width), max(height, top + image.height)))

    def _fit_comparison(self) -> None:
        self.comparison_zoom.set(1.0)
        self._refresh_comparison()

    def _comparison_pan_start(self, event) -> None:
        self._comparison_pan_anchor = (int(event.x), int(event.y))
        self.comparison_canvas.scan_mark(event.x, event.y)

    def _comparison_pan_move(self, event) -> None:
        if self._comparison_pan_anchor is None:
            return
        self.comparison_canvas.scan_dragto(event.x, event.y, gain=1)

    def _refresh_comparison(self) -> None:
        # TR: Seçilen karşılaştırma modunu görüntü üzerinde yeniden çizer.
        # EN: Redraw the selected comparison mode on the image canvas.
        if not hasattr(self, "comparison_canvas"):
            return
        reference = self.original_array
        candidate = self._comparison_candidate()
        if reference is None or candidate is None:
            self.comparison_canvas.delete("all")
            self.comparison_canvas.create_text(300, 180, text="Karşılaştırma için görüntü ve çözülmüş çıktı gerekir", fill="white")
            return
        reference_image = self._display_array(reference)
        candidate_image = self._display_array(candidate)
        canvas_width = max(640, int(self.comparison_canvas.winfo_width() or 900))
        canvas_height = max(300, int(self.comparison_canvas.winfo_height() or 430))
        mode = self.comparison_mode.get()
        if mode == "Side-by-side":
            left = self._fit_display(reference_image, (canvas_width - 18) // 2, canvas_height - 30)
            right = self._fit_display(candidate_image, (canvas_width - 18) // 2, canvas_height - 30)
            combined = Image.new("RGB", (left.width + right.width + 18, max(left.height, right.height) + 24), "#202020")
            combined.paste(left, (0, 20))
            combined.paste(right, (left.width + 18, 20))
            draw = ImageDraw.Draw(combined)
            draw.text((4, 3), "Original", fill="white")
            draw.text((left.width + 22, 3), self.comparison_target.get(), fill="white")
            self._set_comparison_image(combined)
            return
        if mode == "Slider":
            width = canvas_width - 20
            height = canvas_height - 24
            left = self._fit_display(reference_image, width, height)
            right = self._fit_display(candidate_image, width, height)
            common = (max(left.width, right.width), max(left.height, right.height))
            left = left.resize(common, Image.Resampling.LANCZOS)
            right = right.resize(common, Image.Resampling.LANCZOS)
            split = int(common[0] * min(1.0, max(0.0, float(self.comparison_position.get()))))
            combined = left.copy()
            if split < common[0]:
                combined.paste(right.crop((split, 0, common[0], common[1])), (split, 0))
            draw = ImageDraw.Draw(combined)
            draw.line((split, 0, split, common[1]), fill="#00ff70", width=2)
            draw.text((6, 6), "Original", fill="white")
            draw.text((max(6, split + 6), 6), self.comparison_target.get(), fill="#00ff70")
            self._set_comparison_image(combined)
            return
        reference_rgb = np.asarray(reference_image, dtype=np.float32)
        candidate_rgb = np.asarray(candidate_image.resize(reference_image.size, Image.Resampling.BILINEAR), dtype=np.float32)
        if mode == "Error heatmap":
            error = np.mean(np.abs(reference_rgb - candidate_rgb), axis=2)
            intensity = np.clip(error * 5.0, 0, 255).astype(np.uint8)
            heat = np.repeat(intensity[..., None], 3, axis=2)
            self._set_comparison_image(self._fit_display(Image.fromarray(heat.astype(np.uint8), mode="RGB"), canvas_width - 20, canvas_height - 24))
            return
        # Histogram mode: draw luminance distributions without making matplotlib a GUI dependency.
        histogram = Image.new("RGB", (canvas_width - 20, canvas_height - 24), "white")
        draw = ImageDraw.Draw(histogram)
        margin = 42
        plot_width = histogram.width - margin - 12
        plot_height = histogram.height - margin - 18
        draw.line((margin, 8, margin, 8 + plot_height), fill="#222222")
        draw.line((margin, 8 + plot_height, margin + plot_width, 8 + plot_height), fill="#222222")
        for values, color, label in (
            (reference_rgb, "#2468c5", "Original"),
            (candidate_rgb, "#d85b2a", self.comparison_target.get()),
        ):
            luminance = np.mean(values, axis=2).astype(np.uint8)
            counts, _ = np.histogram(luminance, bins=32, range=(0, 256))
            counts = counts / max(1, counts.max())
            points = []
            for index, value in enumerate(counts):
                x = margin + int(index * plot_width / 31)
                y = 8 + plot_height - int(value * plot_height)
                points.append((x, y))
            draw.line(points, fill=color, width=2)
            draw.text((margin + 8 + (0 if label == "Original" else 100), 12), label, fill=color)
        self._set_comparison_image(histogram)
