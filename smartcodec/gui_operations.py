"""TR: Encode, decode, video ve batch işlemlerini yönetir. / EN: Manages encode, decode, video, and batch operations."""

from __future__ import annotations

from .gui_support import *  # noqa: F401,F403 - shared GUI dependencies / ortak GUI bağımlılıkları


class GuiOperationsMixin:
    """TR: Encode, decode, video ve batch işlemlerini yönetir. / EN: Manages encode, decode, video, and batch operations."""

    def _encode_video(self) -> None:
        input_path = self.video_input.get().strip()
        if not input_path or not Path(input_path).exists():
            self._show_error(ValueError("Geçerli bir video seçin"))
            return
        options = {
            "mode": self.video_mode.get(),
            "codec": self.video_codec.get(),
            "wavelet": self.video_wavelet.get(),
            "level": int(self.level.get()),
            "step": float(self.video_step.get()),
            "gop_size": int(self.video_gop.get()),
            "keyframe_interval": int(self.video_keyframe.get()),
            "motion_estimation": self.video_motion_method.get() != "none",
            "motion_method": self.video_motion_method.get(),
            "motion_compensation": bool(self.video_motion_compensation.get()),
            "roi_mask_path": self.video_roi_mask.get().strip() or None,
            "roi_tracking": bool(self.video_roi_tracking.get()),
            "transport_segments": bool(self.video_transport_segments.get()),
            "transport_tiles": bool(self.video_transport_tiles.get()),
            "transport_tile_size": int(self.video_transport_tile_size.get()),
        }
        output_dir = self.video_output_dir.get().strip()
        self._run_async(
            lambda: compress_video(input_path, output_dir, **options),
            self._video_encode_done,
        )

    def _video_encode_done(self, info: dict) -> None:
        self.video_manifest.set(info["manifest"])
        self.status.set(f"Video {info['frame_count']} kare olarak kodlandı.")

    def _decode_video(self) -> None:
        manifest = self.video_manifest.get().strip()
        output = self.video_decoded.get().strip()
        if not manifest or not Path(manifest).exists():
            self._show_error(ValueError("Geçerli bir video manifest seçin"))
            return
        self._run_async(lambda: decompress_video(manifest, output), lambda info: self.status.set(f"Video oluşturuldu: {info['output']}"))

    def _simulate_transport(self) -> None:
        source = self.video_manifest.get().strip()
        destination = self.transport_output.get().strip()
        if not source or not Path(source).exists():
            self._show_error(ValueError("Önce bir SWC veya manifest dosyası seçin"))
            return
        is_manifest = Path(source).suffix.lower() == ".json"
        if is_manifest and self.video_transport_framewise.get():
            if self.transport_backend.get() == "live-udp":
                self._show_error(ValueError("Frame-level partial transport şu anda simulation backend'iyle kullanılmalıdır"))
                return
            loss = float(self.transport_loss.get())
            framewise_destination = Path(destination)
            if framewise_destination.suffix:
                framewise_destination = framewise_destination.with_suffix("")
            framewise_destination = framewise_destination.with_name(framewise_destination.name + "_frames")
            self._run_async(
                lambda: simulate_video_transport(
                    source, framewise_destination, loss_rate=loss, fec=bool(self.transport_fec.get()),
                ),
                lambda info: self.status.set(
                    f"Frame-level transport: {info['partial_frame_count']} partial, {info['dropped_frame_count']} dropped frame"
                ),
            )
            return
        if Path(source).suffix.lower() == ".json":
            try:
                bundle_path = Path(destination)
                if bundle_path.suffix.lower() != ".zip":
                    bundle_path = bundle_path.with_suffix(".zip")
                package_video(source, bundle_path)
                source = str(bundle_path)
                destination = str(bundle_path.with_name(f"{bundle_path.stem}_received{bundle_path.suffix}"))
            except Exception as exc:
                self._show_error(ValueError(f"Video bundle hazırlanamadı: {exc}"))
                return
        if self.transport_backend.get() == "live-udp":
            host = self.transport_host.get().strip()
            try:
                port = int(self.transport_port.get())
            except (TypeError, ValueError, tk.TclError):
                self._show_error(ValueError("UDP port sayısal olmalı"))
                return
            fec = bool(self.transport_fec.get())
            self._run_async(
                lambda: send_udp_file(source, host, port, fec=fec),
                lambda info: self.status.set(
                    f"Live UDP gönderildi: {info['transmitted_packets']} paket, hedef {info['destination']}"
                ),

            )
            return
        loss = float(self.transport_loss.get())
        self._run_async(
            lambda: simulate_file(source, destination, loss_rate=loss),
            lambda info: self.status.set(f"Packet simülasyonu: {info['received']}/{info['packets']} paket alındı."),
        )

    def _encode(self) -> None:
        # TR: Kullanıcı seçimlerini codec çekirdeğinin tek bir encode çağrısına dönüştürür.
        # EN: Translate user selections into one call to the codec core.
        try:
            options = self._encode_options()
        except Exception as exc:
            self._show_error(exc)
            return

        def work():
            if options["codec"] in {"jpeg", "jpeg2000"}:
                if options["mode"] == "lossless":
                    raise ValueError("JPEG/JPEG2000 GUI çıktısı kayıplı modda kullanılmalıdır")
                if options["roi_mask"] is not None or options["target_bpp"] is not None or options["target_psnr"] is not None:
                    raise ValueError("Standart JPEG/JPEG2000 akışı ROI/hedef arama yerine ayrı ayar kullanır")
                return encode_standard(
                    options["image"], options["output_path"], codec=options["codec"],
                    quality=options["standard_quality"], rate=options["standard_rate"],
                )
            info = encode_file(
                options["input_path"], options["output_path"], mode=options["mode"], wavelet=options["wavelet"],
                level=options["level"], step=options["step"], quantizer=options["quantizer"],
                codec=options["codec"], colorspace=options["colorspace"], roi_mask=options["roi_mask"], roi_strength=options["roi_strength"],
                target_bpp=options["target_bpp"], target_psnr=options["target_psnr"], roi_feather=options["roi_feather"], restoration=False,
                transport_segments=options["transport_segments"], transport_tiles=options["transport_tiles"],
                transport_tile_size=options["transport_tile_size"],
            )
            return info

        self._run_async(work, self._encode_done)

    def _encode_done(self, info: dict) -> None:
        self.last_encode_info = info
        self.status.set(f"Encode tamamlandı: {self.encoded_path.get()}")
        self.metrics_text.set(
            f"Dosya boyutu: {info['file_size']:,} byte    |    Sıkıştırma oranı: {info['compression_ratio']:.2f}x    |    BPP: {info['bits_per_pixel']:.3f}"
        )

    def _batch_menu(self) -> None:
        choice = messagebox.askyesno(
            "Batch codec",
            "Goruntu klasorunu SWC olarak kodlamak icin Evet, SWC klasorunu goruntuye donusturmek icin Hayir secin.",
        )
        if choice:
            self._batch_encode()
        else:
            self._batch_decode()

    def _batch_encode(self) -> None:
        if self.codec_method.get() in {"jpeg", "jpeg2000"}:
            self._show_error(ValueError("Standart JPEG/JPEG2000 için tekli Encode akışını kullanın; SWC batch'e ait değildir"))
            return
        input_dir = filedialog.askdirectory(title="Batch girdi goruntu klasoru")
        output_dir = filedialog.askdirectory(title="Batch SWC cikti klasoru")
        if not input_dir or not output_dir:
            return
        try:
            roi_boxes = self._parse_roi_boxes()
            target_bpp = float(self.target_bpp.get()) if self.target_bpp.get().strip() else None
            target_psnr = float(self.target_psnr.get()) if self.target_psnr.get().strip() else None
        except (TypeError, ValueError) as exc:
            self._show_error(exc)
            return
        codec_options = {
            "mode": self.mode.get(), "codec": self.codec_method.get(), "wavelet": self.wavelet.get(),
            "level": int(self.level.get()), "step": float(self.step.get()), "quantizer": self.quantizer.get(),
            "colorspace": self.colorspace.get(), "roi_strength": float(self.roi_strength.get()),
            "roi_feather": int(self.roi_feather.get()), "target_bpp": target_bpp, "target_psnr": target_psnr,
            "transport_segments": bool(self.transport_segments.get()),
            "transport_tiles": bool(self.transport_tiles.get()),
            "transport_tile_size": int(self.transport_tile_size.get()),
        }
        mask_path = self.roi_mask_path.get().strip()
        sources = sorted(
            path for path in Path(input_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
        )
        if not sources:
            self._show_error(ValueError("Girdi klasorunde desteklenen goruntu bulunamadi"))
            return

        def work(report_progress):
            destination_root = Path(output_dir)
            count = 0
            for source in sources:
                if self.cancel_event.is_set():
                    break
                image = load_image(source)
                roi_mask = load_mask(mask_path, image.shape[:2]) if mask_path else None
                if roi_boxes:
                    generated = boxes_to_mask(image.shape[:2], roi_boxes)
                    roi_mask = generated if roi_mask is None else np.maximum(roi_mask, generated)
                destination = destination_root / f"{source.stem}.swc"
                encode_file(source, destination, roi_mask=roi_mask, **codec_options)
                count += 1
                report_progress(count)
            return count

        self._run_async(
            work, lambda count: self.status.set(f"Batch encode tamamlandi: {count} dosya"),
            "Batch encode", progress_total=len(sources), with_progress=True,
        )

    def _batch_decode(self) -> None:
        input_dir = filedialog.askdirectory(title="Batch SWC girdi klasoru")
        output_dir = filedialog.askdirectory(title="Batch goruntu cikti klasoru")
        if not input_dir or not output_dir:
            return
        sources = sorted(path for path in Path(input_dir).rglob("*.swc") if path.is_file())
        if not sources:
            self._show_error(ValueError("Girdi klasorunde SWC bulunamadi"))
            return

        def work(report_progress):
            destination_root = Path(output_dir)
            count = 0
            for source in sources:
                if self.cancel_event.is_set():
                    break
                decode_file(source, destination_root / f"{source.stem}.png")
                count += 1
                report_progress(count)
            return count

        self._run_async(
            work, lambda count: self.status.set(f"Batch decode tamamlandi: {count} dosya"),
            "Batch decode", progress_total=len(sources), with_progress=True,
        )

    def _decode(self) -> None:
        source = self.encoded_path.get().strip()
        destination = self.decoded_path.get().strip()
        if not source or not Path(source).exists():
            self._show_error(ValueError("Geçerli bir SWC/JPEG/JPEG2000 dosyası seçin"))
            return
        if not destination:
            self._show_error(ValueError("Çözülmüş görüntü için çıktı yolu seçin"))
            return
        if Path(source).suffix.lower() in {".jpg", ".jpeg", ".jp2", ".j2k", ".jpf"}:
            def decode_standard_file():
                image = decode_standard(source)
                save_image(image, destination)
                return {"shape": list(image.shape), "output": str(destination)}
            self._run_async(decode_standard_file, self._decode_done)
        else:
            self._run_async(lambda: decode_file(source, destination), self._decode_done)

    def _decode_done(self, info: dict) -> None:
        self.decoded_array = load_image(info["output"])
        self.restored_array = None
        self._show_image(self.decoded_array, self.decoded_label, "decoded")
        self.restored_label.configure(image="", text="Restoration uygulanmadi")
        self.difference_label.configure(image="", text="Difference goruntusu yok")
        self._refresh_metrics()
        self._refresh_comparison()
        self.status.set(f"Decode tamamlandı: {info['output']}")

    def _encode_decode(self) -> None:
        try:
            options = self._encode_options()
        except Exception as exc:
            self._show_error(exc)
            return
        destination = self.decoded_path.get().strip()
        if not destination:
            self._show_error(ValueError("Çözülmüş görüntü için çıktı yolu seçin"))
            return
        apply_ai = self.ai_reconstruction.get() and options["mode"] == "lossy"
        model_path = self.restoration_model.get().strip()

        def work():
            if options["codec"] in {"jpeg", "jpeg2000"}:
                if options["mode"] == "lossless":
                    raise ValueError("JPEG/JPEG2000 GUI çıktısı kayıplı modda kullanılmalıdır")
                if options["roi_mask"] is not None or options["target_bpp"] is not None or options["target_psnr"] is not None:
                    raise ValueError("Standart JPEG/JPEG2000 akışı ROI/hedef arama yerine ayrı ayar kullanır")
                info = encode_standard(
                    options["image"], options["output_path"], codec=options["codec"],
                    quality=options["standard_quality"], rate=options["standard_rate"],
                )
                decoded = decode_standard(options["output_path"])
                save_image(decoded, destination)
                return info, decoded, None
            info = encode_file(
                options["input_path"], options["output_path"], mode=options["mode"], wavelet=options["wavelet"],
                level=options["level"], step=options["step"], quantizer=options["quantizer"],
                codec=options["codec"], colorspace=options["colorspace"], roi_mask=options["roi_mask"], roi_strength=options["roi_strength"],

                target_bpp=options["target_bpp"], target_psnr=options["target_psnr"], roi_feather=options["roi_feather"],
                transport_segments=options["transport_segments"], transport_tiles=options["transport_tiles"],
                transport_tile_size=options["transport_tile_size"],
                restoration=apply_ai, ai_enabled=apply_ai,
            )
            decoded = decode_array(options["output_path"])
            restored = None
            if apply_ai:
                if model_path:
                    restored = TorchRestorationAdapter(model_path).restore(decoded)
                else:
                    restored = detail_reconstruction(decoded)
            save_image(restored if restored is not None else decoded, destination)
            return info, decoded, restored

        self._run_async(work, self._encode_decode_done)

    def _encode_decode_done(self, result) -> None:
        info, decoded, restored = result
        self.last_encode_info = info
        self.decoded_array = decoded
        self.restored_array = restored
        self._show_image(decoded, self.decoded_label, "decoded")
        if restored is not None:
            self._show_image(restored, self.restored_label, "restored")
            self._show_difference(decoded, restored)
        else:
            self.restored_label.configure(image="", text="Restoration uygulanmadi")
            self.difference_label.configure(image="", text="Difference goruntusu yok")
        self._refresh_metrics()
        self._refresh_comparison()
        self.status.set("Encode + Decode tamamlandı.")
