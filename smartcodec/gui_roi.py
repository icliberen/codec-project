"""TR: Dosya seçimi, önizleme ve ROI etkileşimlerini yönetir. / EN: Manages file selection, preview, and ROI interactions."""

from __future__ import annotations

from .gui_support import *  # noqa: F401,F403 - shared GUI dependencies / ortak GUI bağımlılıkları


class GuiRoiMixin:
    """TR: Dosya seçimi, önizleme ve ROI etkileşimlerini yönetir. / EN: Manages file selection, preview, and ROI interactions."""

    def _clear_roi(self) -> None:
        self.roi_boxes.set("")
        self.current_roi_mask = None
        self.semantic_roi_mask = None
        if hasattr(self, "original_canvas"):
            if self.original_array is not None:
                self._show_image(self.original_array, self.original_canvas, "original")
            self._draw_roi_boxes()
        self.status.set("ROI temizlendi.")

    def _draw_roi_boxes(self) -> None:
        if not hasattr(self, "original_canvas"):
            return
        self.original_canvas.delete("roi")
        if not self.original_display_size or not self.original_source_shape:
            return
        display_width, display_height = self.original_display_size
        source_height, source_width = self.original_source_shape
        scale_x, scale_y = display_width / max(source_width, 1), display_height / max(source_height, 1)
        try:
            boxes = self._parse_roi_boxes()
        except ValueError:
            return
        for x, y, width, height in boxes:
            self.original_canvas.create_rectangle(
                x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y,
                outline="#ff3030", width=2, tags="roi",
            )

    def _roi_start(self, event) -> None:
        if not self.original_display_size:
            return
        self.roi_start = (event.x, event.y)
        if self.roi_preview_rect:
            self.original_canvas.delete(self.roi_preview_rect)
        self.roi_preview_rect = self.original_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00ff70", width=2)

    def _roi_drag(self, event) -> None:
        if self.roi_start and self.roi_preview_rect:
            self.original_canvas.coords(self.roi_preview_rect, self.roi_start[0], self.roi_start[1], event.x, event.y)

    def _roi_end(self, event) -> None:
        if not self.roi_start or not self.original_display_size or not self.original_source_shape:
            return
        start_x, start_y = self.roi_start
        x0, x1 = sorted((max(0, start_x), max(0, event.x)))
        y0, y1 = sorted((max(0, start_y), max(0, event.y)))
        display_width, display_height = self.original_display_size
        source_height, source_width = self.original_source_shape
        scale_x, scale_y = display_width / max(source_width, 1), display_height / max(source_height, 1)
        box = (int(x0 / scale_x), int(y0 / scale_y), max(1, int((x1 - x0) / scale_x)), max(1, int((y1 - y0) / scale_y)))
        existing = self.roi_boxes.get().strip()
        box_text = ",".join(map(str, box))
        self.roi_boxes.set(existing + ";" + box_text if existing else box_text)
        self.current_roi_mask = self._build_roi_mask(self.original_source_shape)
        self.roi_start = None
        if self.original_array is not None:
            self._show_image(self.original_array, self.original_canvas, "original")
        self._draw_roi_boxes()
        self.status.set(f"ROI eklendi: {box}")

    def _detect_yolo(self) -> None:
        path = self.input_path.get().strip()
        if not path or not Path(path).exists():
            self._show_error(ValueError("Önce bir görüntü seçin"))
            return
        self._run_async(lambda: analyze_scene(path, self.yolo_model.get().strip() or "auto"), self._yolo_done)

    def _yolo_done(self, detections: list[dict]) -> None:
        boxes = [item["box"] for item in detections]
        self.roi_boxes.set(";".join(",".join(map(str, box)) for box in boxes))
        if self.original_source_shape:
            self.semantic_roi_mask = detections_to_mask(self.original_source_shape, detections)
            self.current_roi_mask = self._build_roi_mask(self.original_source_shape)
            if self.original_array is not None:
                self._show_image(self.original_array, self.original_canvas, "original")
            self._draw_roi_boxes()
        labels = ", ".join(f"{item['label']}:{item['confidence']:.2f}" for item in detections[:6])
        segmented = sum("mask" in item for item in detections)
        self.status.set(f"YOLO {len(boxes)} ROI buldu ({segmented} segmentation maskesi). {labels}")

    def _detect_faces(self) -> None:
        path = self.input_path.get().strip()
        if not path or not Path(path).exists():
            self._show_error(ValueError("Önce bir görüntü seçin"))
            return
        self._run_async(lambda: detect_faces(path), self._faces_done)

    def _faces_done(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self.roi_boxes.set(";".join(",".join(map(str, box)) for box in boxes))
        if self.original_source_shape:
            self.current_roi_mask = self._build_roi_mask(self.original_source_shape)
            self._draw_roi_boxes()
        self.status.set(f"Yüz algılama {len(boxes)} ROI buldu.")

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Görüntü seç",
            filetypes=[("Görüntüler", "*.png *.bmp *.tif *.tiff *.jpg *.jpeg"), ("Tüm dosyalar", "*.*")],
        )
        if not path:
            return
        self.input_path.set(path)
        source = Path(path)
        self.encoded_path.set(str(source.with_suffix(".swc")))
        self.decoded_path.set(str(source.with_name(source.stem + "_decoded.png")))
        self._preview_input()

    def _choose_encoded_output(self) -> None:
        codec = self.codec_method.get()
        if codec == "jpeg":
            default_extension, filetypes = ".jpg", [("JPEG", "*.jpg *.jpeg")]
        elif codec == "jpeg2000":
            default_extension, filetypes = ".jp2", [("JPEG2000", "*.jp2 *.j2k")]
        else:
            default_extension, filetypes = ".swc", [("SWC", "*.swc")]
        path = filedialog.asksaveasfilename(title="Kodlanmış dosyayı kaydet", defaultextension=default_extension, filetypes=filetypes)
        if path:

            self.encoded_path.set(path)

    def _choose_decoded_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Çözülmüş görüntüyü kaydet", defaultextension=".png", filetypes=[("PNG", "*.png"), ("TIFF", "*.tiff"), ("BMP", "*.bmp")]
        )
        if path:
            self.decoded_path.set(path)

    def _choose_roi_mask(self) -> None:
        path = filedialog.askopenfilename(title="ROI maskesi seç", filetypes=[("Görüntüler", "*.png *.bmp *.tif *.tiff"), ("Tüm dosyalar", "*.*")])
        if path:
            self.roi_mask_path.set(path)

    def _choose_yolo_model(self) -> None:
        path = filedialog.askopenfilename(title="YOLO modeli seç", filetypes=[("YOLO TorchScript", "*.pt"), ("Tüm dosyalar", "*.*")])
        if path:
            self.yolo_model.set(path)

    def _choose_benchmark_input(self) -> None:
        path = filedialog.askdirectory(title="Benchmark girdi klasörü seç")
        if path:
            self.benchmark_input.set(path)

    def _choose_benchmark_output(self) -> None:
        path = filedialog.askdirectory(title="Benchmark çıktı klasörü seç")
        if path:
            self.benchmark_output.set(path)

    def _choose_benchmark_roi_mask(self) -> None:
        path = filedialog.askopenfilename(
            title="Benchmark ROI maskesi seç",
            filetypes=[("Görüntüler", "*.png *.bmp *.tif *.tiff"), ("Tüm dosyalar", "*.*")],
        )
        if path:
            self.benchmark_roi_mask.set(path)

    def _choose_video_input(self) -> None:
        path = filedialog.askopenfilename(title="Video seç", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("Tüm dosyalar", "*.*")])
        if path:
            self.video_input.set(path)
            self.video_output_dir.set(str(Path(path).with_name(Path(path).stem + "_swc_frames")))

    def _choose_video_output(self) -> None:
        path = filedialog.askdirectory(title="Video frame çıktı klasörü seç")
        if path:
            self.video_output_dir.set(path)

    def _choose_manifest(self) -> None:
        path = filedialog.askopenfilename(title="Video manifest seç", filetypes=[("JSON", "*.json")])
        if path:
            self.video_manifest.set(path)

    def _choose_video_decoded(self) -> None:
        path = filedialog.asksaveasfilename(title="Çözülmüş videoyu kaydet", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if path:
            self.video_decoded.set(path)

    def _choose_video_roi_mask(self) -> None:
        path = filedialog.askopenfilename(
            title="Video ROI maskesi seç",
            filetypes=[("Görüntüler", "*.png *.bmp *.tif *.tiff"), ("Tüm dosyalar", "*.*")],
        )
        if path:
            self.video_roi_mask.set(path)

    def _preview_input(self) -> None:
        path = self.input_path.get().strip()
        if not path:
            messagebox.showinfo("Görüntü seç", "Önce bir girdi görüntüsü seçin.")
            return
        try:
            self.original_array = load_image(path)
            self.current_roi_mask = None
            self.semantic_roi_mask = None
            self.decoded_array = None
            self.restored_array = None
            self._show_image(self.original_array, self.original_canvas, "original")
            self.decoded_label.configure(image="", text="Decoded image yok")
            self.restored_label.configure(image="", text="Restoration uygulanmadi")
            self.difference_label.configure(image="", text="Difference goruntusu yok")
            self._refresh_comparison()
            self.status.set(f"Görüntü yüklendi: {Path(path).name} - {self.original_array.shape}")
        except Exception as exc:  # GUI boundary: show a friendly error.
            self._show_error(exc)

    def _handle_drop(self, event) -> None:
        """Accept one dropped image/SWC path while keeping DnD optional."""
        try:
            paths = [Path(value) for value in self.tk.splitlist(event.data)]
        except (AttributeError, tk.TclError, TypeError):
            return
        paths = [path for path in paths if path.exists()]
        if not paths:
            self.status.set("Bir dosya veya klasor surukleyip birakin.")
            return
        path = paths[0]
        if path.is_dir():
            self.status.set(f"Klasor birakildi: Batch dugmesini kullanin ({path})")
            return
        if path.suffix.lower() == ".swc":
            self.encoded_path.set(str(path))
            self.decoded_path.set(str(path.with_name(path.stem + "_decoded.png")))
        else:
            self.input_path.set(str(path))
            self.encoded_path.set(str(path.with_suffix(".swc")))
            self.decoded_path.set(str(path.with_name(path.stem + "_decoded.png")))
            self._preview_input()
        self.status.set(f"Dosya birakildi: {path.name} ({len(paths)} dosya)")

    def _parse_roi_boxes(self) -> list[tuple[int, int, int, int]]:
        raw = self.roi_boxes.get().strip()
        if not raw:
            return []
        boxes = []
        for item in raw.split(";"):
            values = tuple(int(part.strip()) for part in item.split(","))
            if len(values) != 4:
                raise ValueError("ROI kutusu x,y,w,h biçiminde olmalı")
            boxes.append(values)
        return boxes

    def _build_roi_mask(self, shape: tuple[int, int]) -> np.ndarray | None:
        # TR: Kullanıcı kutularını kaynak görüntü boyutunda piksel maskesine dönüştürür.
        # EN: Convert user-drawn boxes into a pixel mask at the source image resolution.
        if self.mode.get() == "lossless":
            return None
        mask = None
        if self.semantic_roi_mask is not None and tuple(self.semantic_roi_mask.shape) == tuple(shape):
            mask = np.asarray(self.semantic_roi_mask, dtype=np.float32).copy()
        mask_path = self.roi_mask_path.get().strip()
        if mask_path:
            mask = load_mask(mask_path, shape)
        boxes = self._parse_roi_boxes()
        if boxes:
            generated = boxes_to_mask(shape, boxes)
            mask = generated if mask is None else np.maximum(mask, generated)
        return mask

    def _encode_options(self) -> dict:
        input_path = self.input_path.get().strip()
        output_path = self.encoded_path.get().strip()
        if not input_path or not Path(input_path).exists():
            raise ValueError("Geçerli bir girdi görüntüsü seçin")
        if not output_path:
            raise ValueError("Sıkıştırılmış çıktı yolu seçin")
        image = load_image(input_path)
        self.original_array = image
        self._show_image(image, self.original_canvas, "original")
        self.current_roi_mask = self._build_roi_mask(image.shape[:2])
        self._show_image(image, self.original_canvas, "original")
        raw_target_bpp = self.target_bpp.get().strip()
        target_bpp = float(raw_target_bpp) if raw_target_bpp else None
        raw_target_psnr = self.target_psnr.get().strip()
        target_psnr = float(raw_target_psnr) if raw_target_psnr else None
        return {
            "input_path": input_path,
            "output_path": output_path,
            "image": image,
            "mode": self.mode.get(),
            "codec": self.codec_method.get(),
            "wavelet": self.wavelet.get(),
            "level": int(self.level.get()),
            "step": float(self.step.get()),
            "quantizer": self.quantizer.get(),
            "colorspace": self.colorspace.get(),
            "roi_mask": self.current_roi_mask,
            "roi_strength": float(self.roi_strength.get()),
            "target_bpp": target_bpp,
            "target_psnr": target_psnr,
            "roi_feather": int(self.roi_feather.get()),
            "standard_quality": int(self.standard_quality.get()),
            "standard_rate": float(self.standard_rate.get()),
            "transport_segments": bool(self.transport_segments.get()),
            "transport_tiles": bool(self.transport_tiles.get()),
            "transport_tile_size": int(self.transport_tile_size.get()),
        }
