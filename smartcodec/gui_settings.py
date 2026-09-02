"""TR: Ayar, preset ve profil yaşam döngüsünü yönetir. / EN: Manages settings, presets, and profile lifecycle."""

from __future__ import annotations

from .gui_support import *  # noqa: F401,F403 - shared GUI dependencies / ortak GUI bağımlılıkları


class GuiSettingsMixin:
    """TR: Ayar, preset ve profil yaşam döngüsünü yönetir. / EN: Manages settings, presets, and profile lifecycle."""

    def _setting_variables(self) -> dict[str, tk.Variable]:
        return {
            "input_path": self.input_path,
            "encoded_path": self.encoded_path,
            "decoded_path": self.decoded_path,
            "mode": self.mode,
            "codec_method": self.codec_method,
            "profile": self.profile,
            "wavelet": self.wavelet,
            "level": self.level,
            "step": self.step,
            "target_bpp": self.target_bpp,
            "target_psnr": self.target_psnr,
            "comparison_mode": self.comparison_mode,
            "comparison_target": self.comparison_target,
            "comparison_position": self.comparison_position,
            "comparison_zoom": self.comparison_zoom,
            "quantizer": self.quantizer,
            "colorspace": self.colorspace,
            "standard_quality": self.standard_quality,
            "standard_rate": self.standard_rate,
            "roi_boxes": self.roi_boxes,
            "roi_mask_path": self.roi_mask_path,
            "roi_strength": self.roi_strength,
            "roi_feather": self.roi_feather,
            "transport_segments": self.transport_segments,
            "transport_tiles": self.transport_tiles,
            "transport_tile_size": self.transport_tile_size,
            "yolo_model": self.yolo_model,
            "restoration_model": self.restoration_model,
            "ai_reconstruction": self.ai_reconstruction,
            "benchmark_input": self.benchmark_input,
            "benchmark_output": self.benchmark_output,
            "benchmark_steps": self.benchmark_steps,
            "benchmark_wavelets": self.benchmark_wavelets,
            "benchmark_normalization": self.benchmark_normalization,
            "benchmark_target": self.benchmark_target,
            "benchmark_roi_mask": self.benchmark_roi_mask,
            "benchmark_quantizer": self.benchmark_quantizer,
            "benchmark_allocation": self.benchmark_allocation,
            "video_input": self.video_input,
            "video_output_dir": self.video_output_dir,
            "video_manifest": self.video_manifest,
            "video_decoded": self.video_decoded,
            "video_mode": self.video_mode,
            "video_codec": self.video_codec,
            "video_step": self.video_step,
            "video_wavelet": self.video_wavelet,
            "video_gop": self.video_gop,
            "video_keyframe": self.video_keyframe,
            "video_motion_method": self.video_motion_method,
            "video_motion_compensation": self.video_motion_compensation,
            "video_roi_mask": self.video_roi_mask,
            "video_roi_tracking": self.video_roi_tracking,
            "video_transport_segments": self.video_transport_segments,
            "video_transport_tiles": self.video_transport_tiles,
            "video_transport_tile_size": self.video_transport_tile_size,
            "video_transport_framewise": self.video_transport_framewise,
            "transport_loss": self.transport_loss,
            "transport_output": self.transport_output,
            "transport_backend": self.transport_backend,
            "transport_host": self.transport_host,
            "transport_port": self.transport_port,
            "transport_fec": self.transport_fec,
        }

    def _codec_preset(self) -> dict:
        names = (
            "mode", "codec_method", "wavelet", "level", "step", "target_bpp", "target_psnr",
            "quantizer", "colorspace", "roi_boxes", "roi_mask_path", "roi_strength", "roi_feather",
            "yolo_model", "restoration_model", "ai_reconstruction", "standard_quality", "standard_rate",
            "transport_segments", "transport_tiles", "transport_tile_size",
        )
        variables = self._setting_variables()
        return {name: variables[name].get() for name in names}

    def _apply_values(self, values: dict) -> None:
        variables = self._setting_variables()
        for name, value in values.items():
            variable = variables.get(name)
            if variable is None:
                continue
            try:
                variable.set(value)
            except (tk.TclError, TypeError, ValueError):
                continue

    def _load_settings(self) -> None:
        if not self.settings_path.is_file():
            return
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._apply_values(payload)
        except (OSError, json.JSONDecodeError):
            self.status.set(f"Ayar dosyasi okunamadi: {self.settings_path}")

    def _save_settings(self) -> None:
        # TR: GUI ayarlarını tekrar açılışta kullanılmak üzere JSON olarak kalıcılaştırır.
        # EN: Persist GUI settings as JSON so they can be restored on the next launch.
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {name: variable.get() for name, variable in self._setting_variables().items()}
            self.settings_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self.status.set(f"Ayarlar kaydedilemedi: {exc}")

    def _choose_restoration_model(self) -> None:
        path = filedialog.askopenfilename(
            title="PyTorch restoration modeli",
            filetypes=(("TorchScript model", "*.pt"), ("Tum dosyalar", "*.*")),
        )
        if path:
            self.restoration_model.set(path)

    def _save_preset_dialog(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Codec presetini kaydet",
            defaultextension=".json",
            filetypes=(("JSON preset", "*.json"), ("Tum dosyalar", "*.*")),
        )
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self._codec_preset(), indent=2, ensure_ascii=False), encoding="utf-8")
            self.status.set(f"Preset kaydedildi: {path}")
        except OSError as exc:
            self._show_error(exc)

    def _load_preset_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Codec preseti yukle",
            filetypes=(("JSON preset", "*.json"), ("Tum dosyalar", "*.*")),
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Preset JSON nesne olmali")
            self._apply_values(payload)
            self.status.set(f"Preset yuklendi: {path}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self._show_error(exc)

    def _on_close(self) -> None:
        self._closing = True
        self.cancel_event.set()
        self._save_settings()
        self.destroy()

    def _profile_changed(self, *_args) -> None:
        profiles = {
            "Yüksek kalite": ("dwt", "lossy", "db4", 4.0, "scalar"),
            "Maksimum sıkıştırma": ("dwt", "lossy", "haar", 28.0, "scalar"),
            "Kayıpsız tıbbi": ("dwt", "lossless", "haar", 1.0, "uniform"),
            "ROI nesne": ("dwt", "lossy", "db4", 16.0, "scalar"),
        }
        selected = profiles.get(self.profile.get())
        if selected:
            method, mode, wavelet, step, quantizer = selected
            self.codec_method.set(method)
            self.mode.set(mode)
            self.wavelet.set(wavelet)
            self.step.set(step)
            self.quantizer.set(quantizer)

    def _mode_changed(self, *_args) -> None:
        if self.mode.get() == "lossless":
            self.ai_reconstruction.set(False)
            self.status.set("Kayıpsız mod: reversible integer 5/3 wavelet kullanılıyor.")
        else:
            self.status.set("Kayıplı mod: ROI ve isteğe bağlı restoration kullanılabilir.")

    def _suggest_parameters(self) -> None:
        if self.original_array is None:
            self._preview_input()
        if self.original_array is None:
            return
        parameters, confidence = estimate_parameters(self.original_array)
        self.codec_method.set("dwt")
        self.wavelet.set(parameters["wavelet"])
        self.step.set(parameters["step"])
        self.quantizer.set(parameters["quantizer"])
        self.status.set(f"AI parametre önerisi: {parameters['wavelet']} / step={parameters['step']} (güven: {confidence:.2f})")
