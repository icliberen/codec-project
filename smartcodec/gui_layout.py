"""TR: GUI sekmelerini, ayar alanlarını ve ortak widget yardımcılarını içerir. / EN: Contains GUI tabs, settings panels, and shared widget helpers."""

from __future__ import annotations

from .gui_support import *  # noqa: F401,F403 - shared GUI dependencies / ortak GUI bağımlılıkları


class GuiLayoutMixin:
    """TR: GUI sekmelerini, ayar alanlarını ve ortak widget yardımcılarını içerir. / EN: Contains GUI tabs, settings panels, and shared widget helpers."""

    def _build_ui(self) -> None:
        # TR: Bu bölüm pencerenin ana sekmelerini ve ortak durum çubuğunu kurar.
        # EN: This section builds the main tabs and the shared status bar.
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        header = ttk.Frame(self, padding=(14, 12, 14, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Smart Wavelet Codec", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(header, text="Görüntü sıkıştırma, çözme ve kalite karşılaştırma arayüzü").pack(anchor="w")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=8)
        codec_tab = ttk.Frame(notebook, padding=10)
        benchmark_tab = ttk.Frame(notebook, padding=10)
        video_tab = ttk.Frame(notebook, padding=10)
        notebook.add(codec_tab, text="Kodla / Çöz")
        notebook.add(benchmark_tab, text="Benchmark")
        notebook.add(video_tab, text="Video")
        self._build_codec_tab(codec_tab)
        self._build_benchmark_tab(benchmark_tab)
        self._build_video_tab(video_tab)

        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", side="bottom")
        status_bar = ttk.Label(status_frame, textvariable=self.status, relief="sunken", anchor="w", padding=5)
        status_bar.pack(fill="x", side="left", expand=True)
        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=150)
        self.progress.pack(side="right", padx=4)
        self.cancel_button = ttk.Button(status_frame, text="İptal", command=self._cancel_current, state="disabled")
        self.cancel_button.pack(side="right", padx=4)
        self.copy_error_button = ttk.Button(status_frame, text="Hatayı kopyala", command=self._copy_last_error, state="disabled")
        self.copy_error_button.pack(side="right", padx=4)

    def _build_codec_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)

        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(5, weight=1)

        file_frame = ttk.LabelFrame(parent, text="Dosyalar", padding=8)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(1, weight=1)
        self._path_row(file_frame, 0, "Girdi görüntüsü", self.input_path, self._choose_input)
        self._path_row(file_frame, 1, "Sıkıştırılmış SWC", self.encoded_path, self._choose_encoded_output, save=True)
        self._path_row(file_frame, 2, "Çözülmüş görüntü", self.decoded_path, self._choose_decoded_output, save=True)

        settings = ttk.LabelFrame(parent, text="Codec ayarları", padding=8)
        settings.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for column in range(6):
            settings.columnconfigure(column, weight=1 if column in (1, 3, 5) else 0)
        self._combo(settings, 0, 0, "Profil", self.profile, ("Özel", "Yüksek kalite", "Maksimum sıkıştırma", "Kayıpsız tıbbi", "ROI nesne"))
        self._combo(settings, 0, 2, "Yöntem", self.codec_method, ("dwt", "dct", "prqmf4", "jpeg", "jpeg2000"))
        self._combo(settings, 0, 4, "Mod", self.mode, ("lossy", "lossless"))
        self._combo(settings, 1, 0, "Wavelet", self.wavelet, ("haar", "db4", "db8", "db12", "qmf"))
        self._spin(settings, 1, 2, "DWT seviyesi", self.level, 1, 8)
        self._combo(settings, 1, 4, "Quantizer", self.quantizer, ("uniform", "scalar"))
        self._entry(settings, 2, 0, "Step", self.step)
        self._combo(settings, 2, 2, "Renk uzayı", self.colorspace, ("ycbcr", "rgb"))
        ttk.Button(settings, text="AI parametre öner", command=self._suggest_parameters).grid(
            row=2, column=4, columnspan=2, sticky="ew", padx=(0, 14), pady=4
        )
        self._entry(settings, 3, 0, "Hedef BPP", self.target_bpp)
        self._entry(settings, 3, 2, "Hedef PSNR", self.target_psnr)
        ttk.Checkbutton(settings, text="Decoder detail restoration", variable=self.ai_reconstruction).grid(
            row=3, column=4, columnspan=2, sticky="w", pady=4
        )
        ttk.Label(settings, text="PyTorch model").grid(row=4, column=0, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(settings, textvariable=self.restoration_model).grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=(0, 14), pady=4
        )
        ttk.Button(settings, text="Model sec", command=self._choose_restoration_model).grid(
            row=4, column=4, sticky="ew", padx=(0, 6), pady=4
        )
        ttk.Button(settings, text="Preset kaydet", command=self._save_preset_dialog).grid(
            row=4, column=5, sticky="ew", pady=4
        )
        ttk.Button(settings, text="Preset yukle", command=self._load_preset_dialog).grid(
            row=5, column=5, sticky="ew", pady=4
        )
        self._spin(settings, 5, 0, "JPEG kalite", self.standard_quality, 1, 100)
        self._entry(settings, 5, 2, "JP2 rate", self.standard_rate)
        ttk.Checkbutton(settings, text="Semantic transport bands", variable=self.transport_segments).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=4
        )
        ttk.Checkbutton(settings, text="Spatial transport tiles", variable=self.transport_tiles).grid(
            row=6, column=2, columnspan=2, sticky="w", pady=4
        )
        self._spin(settings, 6, 4, "Tile px", self.transport_tile_size, 8, 2048)

        roi = ttk.LabelFrame(parent, text="İsteğe bağlı ROI", padding=8)
        roi.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        roi.columnconfigure(1, weight=1)
        ttk.Label(roi, text="Kutular (x,y,w,h; ...)").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(roi, textvariable=self.roi_boxes).grid(row=0, column=1, sticky="ew")
        ttk.Label(roi, text="Örnek: 120,80,200,180").grid(row=0, column=2, sticky="w", padx=8)
        ttk.Button(roi, text="Fareyle çiz", command=lambda: self.status.set("Orijinal görüntü üzerinde fareyle ROI çizin.")).grid(row=0, column=3, padx=4)
        ttk.Button(roi, text="YOLO ROI bul", command=self._detect_yolo).grid(row=0, column=4, padx=4)
        ttk.Button(roi, text="Yüz ROI bul", command=self._detect_faces).grid(row=0, column=5, padx=4)
        ttk.Label(roi, text="ROI maskesi").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        ttk.Entry(roi, textvariable=self.roi_mask_path).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(roi, text="Seç", command=self._choose_roi_mask).grid(row=1, column=2, padx=(8, 0), pady=(6, 0))
        ttk.Label(roi, text="ROI gücü").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(roi, from_=0.0, to=0.95, variable=self.roi_strength, orient="horizontal").grid(
            row=2, column=1, sticky="ew", pady=(6, 0)
        )
        self._spin(roi, 2, 2, "Feather", self.roi_feather, 0, 64)
        ttk.Label(roi, text="YOLO modeli").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(roi, textvariable=self.yolo_model).grid(row=3, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(roi, text="Seç", command=self._choose_yolo_model).grid(row=3, column=2, padx=8, pady=(6, 0))
        ttk.Button(roi, text="ROI temizle", command=self._clear_roi).grid(row=3, column=3, padx=8, pady=(6, 0))

        actions = ttk.Frame(parent)
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        for index in range(6):
            actions.columnconfigure(index, weight=1)
        self._add_action(actions, "Görüntüyü göster", self._preview_input, 0)
        self._add_action(actions, "Encode", self._encode, 1)
        self._add_action(actions, "Decode", self._decode, 2)
        self._add_action(actions, "Encode + Decode", self._encode_decode, 3)
        self._add_action(actions, "Batch", self._batch_menu, 5)
        self._add_action(actions, "Çıktı klasörünü aç", self._open_output_folder, 4)

        ttk.Label(parent, textvariable=self.metrics_text, justify="left", padding=6).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 5)
        )

        preview = ttk.Notebook(parent)
        preview.grid(row=5, column=0, columnspan=2, sticky="nsew")
        original_frame = ttk.LabelFrame(preview, text="Orijinal", padding=6)
        decoded_frame = ttk.LabelFrame(preview, text="Çözülmüş", padding=6)
        restored_frame = ttk.LabelFrame(preview, text="Restored", padding=6)
        difference_frame = ttk.LabelFrame(preview, text="Difference", padding=6)
        preview.add(original_frame, text="Orijinal")
        preview.add(decoded_frame, text="Decoded")
        preview.add(restored_frame, text="Restored")
        preview.add(difference_frame, text="Difference")
        self.original_canvas = tk.Canvas(original_frame, background="#202020", highlightthickness=0)
        self.original_canvas.pack(fill="both", expand=True)
        self.original_canvas.create_text(250, 200, text="Henüz görüntü seçilmedi", fill="white", tags="placeholder")
        self.original_canvas.bind("<ButtonPress-1>", self._roi_start)
        self.original_canvas.bind("<B1-Motion>", self._roi_drag)
        self.original_canvas.bind("<ButtonRelease-1>", self._roi_end)
        self.decoded_label = ttk.Label(decoded_frame, text="Henüz çözülmüş görüntü yok", anchor="center")
        self.decoded_label.pack(fill="both", expand=True)
        self.restored_label = ttk.Label(restored_frame, text="Restoration uygulanmadi", anchor="center")
        self.restored_label.pack(fill="both", expand=True)
        self.difference_label = ttk.Label(difference_frame, text="Difference goruntusu yok", anchor="center")
        self.difference_label.pack(fill="both", expand=True)
        comparison_frame = ttk.Frame(preview, padding=6)
        preview.add(comparison_frame, text="Karşılaştır")
        comparison_controls = ttk.Frame(comparison_frame)
        comparison_controls.pack(fill="x", pady=(0, 5))
        ttk.Label(comparison_controls, text="Görünüm").pack(side="left", padx=(0, 5))
        ttk.Combobox(
            comparison_controls, textvariable=self.comparison_mode,
            values=("Side-by-side", "Slider", "Error heatmap", "Histogram"), state="readonly", width=16,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(comparison_controls, text="Karşılaştırılan").pack(side="left", padx=(0, 5))
        ttk.Combobox(
            comparison_controls, textvariable=self.comparison_target,
            values=("Decoded", "Restored"), state="readonly", width=12,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(comparison_controls, text="Slider").pack(side="left", padx=(0, 5))
        ttk.Scale(
            comparison_controls, from_=0.0, to=1.0, variable=self.comparison_position,
            orient="horizontal", command=lambda _value: self._refresh_comparison(),
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(comparison_controls, text="Zoom").pack(side="left", padx=(12, 5))
        ttk.Scale(
            comparison_controls, from_=0.5, to=3.0, variable=self.comparison_zoom,
            orient="horizontal", length=110, command=lambda _value: self._refresh_comparison(),
        ).pack(side="left", padx=(0, 5))
        ttk.Button(comparison_controls, text="Fit", command=self._fit_comparison).pack(side="left")
        self.comparison_canvas = tk.Canvas(comparison_frame, background="#202020", highlightthickness=0)
        self.comparison_canvas.pack(fill="both", expand=True)
        self.comparison_canvas.bind("<Configure>", lambda _event: self._refresh_comparison())
        self.comparison_canvas.bind("<ButtonPress-1>", self._comparison_pan_start)
        self.comparison_canvas.bind("<B1-Motion>", self._comparison_pan_move)
        self.comparison_mode.trace_add("write", lambda *_args: self._refresh_comparison())
        self.comparison_target.trace_add("write", lambda *_args: self._refresh_comparison())

    def _build_benchmark_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Girdi klasörü").grid(row=0, column=0, sticky="w", pady=5)
        self.benchmark_input = tk.StringVar(value=str(Path.cwd() / "outputs" / "samples"))
        ttk.Entry(parent, textvariable=self.benchmark_input).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Seç", command=self._choose_benchmark_input).grid(row=0, column=2)

        ttk.Label(parent, text="Çıktı klasörü").grid(row=1, column=0, sticky="w", pady=5)
        self.benchmark_output = tk.StringVar(value=str(Path.cwd() / "outputs" / "gui_benchmark"))
        ttk.Entry(parent, textvariable=self.benchmark_output).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Seç", command=self._choose_benchmark_output).grid(row=1, column=2)
        ttk.Label(parent, text="Step değerleri").grid(row=2, column=0, sticky="w", pady=5)
        self.benchmark_steps = tk.StringVar(value="4,8,16,32")
        ttk.Entry(parent, textvariable=self.benchmark_steps).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Label(parent, text="Waveletler").grid(row=3, column=0, sticky="w", pady=5)
        self.benchmark_wavelets = tk.StringVar(value="haar,db4,db8")
        ttk.Entry(parent, textvariable=self.benchmark_wavelets).grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Label(parent, text="Normalize modu").grid(row=4, column=0, sticky="w", pady=5)
        self.benchmark_normalization = tk.StringVar(value="grid")
        ttk.Combobox(parent, textvariable=self.benchmark_normalization,
                     values=("grid", "target-psnr", "target-bpp", "roi-compare"), state="readonly", width=16).grid(
                         row=4, column=1, sticky="w", padx=8
                     )
        ttk.Label(parent, text="Hedef (PSNR dB veya BPP)").grid(row=4, column=2, sticky="w", pady=5)
        self.benchmark_target = tk.StringVar(value="")
        self.benchmark_roi_mask = tk.StringVar()
        ttk.Entry(parent, textvariable=self.benchmark_target, width=14).grid(row=4, column=3, sticky="w", padx=8)
        ttk.Label(parent, text="Quantizer").grid(row=4, column=4, sticky="w", pady=5)
        self.benchmark_quantizer = tk.StringVar(value="uniform")
        ttk.Combobox(parent, textvariable=self.benchmark_quantizer,
                     values=("uniform", "scalar"), state="readonly", width=10).grid(row=4, column=5, sticky="w", padx=8)
        ttk.Label(parent, text="Allocation").grid(row=4, column=6, sticky="w", pady=5)
        self.benchmark_allocation = tk.StringVar(value="greedy")
        ttk.Combobox(parent, textvariable=self.benchmark_allocation,
                     values=("greedy", "lagrangian", "dp"), state="readonly", width=12).grid(row=4, column=7, sticky="w", padx=8)
        ttk.Label(parent, text="ROI maskesi").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.benchmark_roi_mask).grid(row=5, column=1, columnspan=2, sticky="ew", padx=8)
        ttk.Button(parent, text="Seç", command=self._choose_benchmark_roi_mask).grid(row=5, column=3, sticky="w")
        buttons = ttk.Frame(parent)
        buttons.grid(row=6, column=0, columnspan=4, sticky="w", pady=12)
        ttk.Button(buttons, text="Örnek görüntüler üret", command=self._generate_samples).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Kategori veri seti üret", command=self._generate_dataset).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Benchmark başlat", command=self._benchmark).pack(side="left")
        ttk.Button(buttons, text="Grafiği aç", command=self._show_benchmark_plot).pack(side="left", padx=8)
        self.benchmark_log = tk.Text(parent, height=18, wrap="word", state="disabled")
        self.benchmark_log.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(4, 4))
        columns = ("category", "image", "codec", "step", "bpp", "psnr", "ssim", "status")
        self.benchmark_table = ttk.Treeview(parent, columns=columns, show="headings", height=10)
        headings = {"category": "Kategori", "image": "Görüntü", "codec": "Codec", "step": "Step", "bpp": "BPP", "psnr": "PSNR", "ssim": "SSIM", "status": "Durum"}
        for column in columns:
            self.benchmark_table.heading(column, text=headings[column])
            self.benchmark_table.column(column, width=110 if column in ("image", "category") else 80, anchor="center")
        self.benchmark_table.grid(row=8, column=0, columnspan=4, sticky="nsew")
        parent.rowconfigure(8, weight=1)

    def _build_video_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        self.video_input = tk.StringVar()
        self.video_output_dir = tk.StringVar(value=str(Path.cwd() / "outputs" / "video_encoded"))
        self.video_manifest = tk.StringVar()
        self.video_decoded = tk.StringVar(value=str(Path.cwd() / "outputs" / "decoded_video.mp4"))
        self.video_mode = tk.StringVar(value="lossy")
        self.video_codec = tk.StringVar(value="dwt")
        self.video_step = tk.DoubleVar(value=12.0)
        self.video_wavelet = tk.StringVar(value="haar")
        self.video_gop = tk.IntVar(value=1)
        self.video_keyframe = tk.IntVar(value=1)
        self.video_motion_method = tk.StringVar(value="none")
        self.video_motion_compensation = tk.BooleanVar(value=False)
        self.video_roi_mask = tk.StringVar()
        self.video_roi_tracking = tk.BooleanVar(value=False)
        self.video_transport_segments = tk.BooleanVar(value=False)
        self.video_transport_tiles = tk.BooleanVar(value=False)
        self.video_transport_tile_size = tk.IntVar(value=64)
        self.video_transport_framewise = tk.BooleanVar(value=False)
        self.transport_loss = tk.DoubleVar(value=0.0)
        self.transport_output = tk.StringVar(value=str(Path.cwd() / "outputs" / "transport_received.swc"))
        self.transport_backend = tk.StringVar(value="simulation")
        self.transport_host = tk.StringVar(value="127.0.0.1")
        self.transport_port = tk.IntVar(value=5000)
        self.transport_fec = tk.BooleanVar(value=False)

        ttk.Label(parent, text="Video girişi").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.video_input).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Seç", command=self._choose_video_input).grid(row=0, column=2)
        ttk.Label(parent, text="Frame çıktı klasörü").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.video_output_dir).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Seç", command=self._choose_video_output).grid(row=1, column=2)
        ttk.Label(parent, text="Manifest").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.video_manifest).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Seç", command=self._choose_manifest).grid(row=2, column=2)
        ttk.Label(parent, text="Çözülmüş video").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.video_decoded).grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Kaydet", command=self._choose_video_decoded).grid(row=3, column=2)
        self._combo(parent, 4, 0, "Mod", self.video_mode, ("lossy", "lossless"))
        self._combo(parent, 5, 0, "Yöntem", self.video_codec, ("dwt", "dct", "prqmf4"))
        self._combo(parent, 6, 0, "Wavelet", self.video_wavelet, ("haar", "db4", "db8", "db12"))
        self._entry(parent, 7, 0, "Step", self.video_step)
        self._spin(parent, 7, 2, "GOP", self.video_gop, 1, 60)
        self._spin(parent, 8, 0, "Keyframe", self.video_keyframe, 1, 60)
        self._combo(parent, 8, 2, "Motion", self.video_motion_method, ("none", "translation", "block", "optical-flow"))
        ttk.Label(parent, text="Video ROI maskesi").grid(row=9, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.video_roi_mask).grid(row=9, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Seç", command=self._choose_video_roi_mask).grid(row=9, column=2)
        video_buttons = ttk.Frame(parent)
        video_buttons.grid(row=10, column=0, columnspan=3, sticky="w", pady=12)
        ttk.Button(video_buttons, text="Video karelerini kodla", command=self._encode_video).pack(side="left", padx=(0, 8))
        ttk.Button(video_buttons, text="Manifest'i videoya çevir", command=self._decode_video).pack(side="left")
        ttk.Button(video_buttons, text="Transport çalıştır", command=self._simulate_transport).pack(side="left", padx=8)
        ttk.Checkbutton(video_buttons, text="Motion compensation", variable=self.video_motion_compensation).pack(side="left", padx=8)
        ttk.Checkbutton(video_buttons, text="ROI motion tracking", variable=self.video_roi_tracking).pack(side="left", padx=8)
        ttk.Checkbutton(video_buttons, text="Semantic transport bands", variable=self.video_transport_segments).pack(side="left", padx=8)
        ttk.Checkbutton(video_buttons, text="Spatial tiles", variable=self.video_transport_tiles).pack(side="left", padx=8)
        ttk.Checkbutton(video_buttons, text="Frame-level partial", variable=self.video_transport_framewise).pack(side="left", padx=8)
        self._spin(parent, 11, 0, "Tile px", self.video_transport_tile_size, 8, 2048)
        ttk.Label(parent, text="Packet loss").grid(row=12, column=0, sticky="w", pady=5)
        ttk.Scale(parent, from_=0.0, to=0.5, variable=self.transport_loss, orient="horizontal").grid(row=12, column=1, sticky="ew", padx=8)
        ttk.Entry(parent, textvariable=self.transport_output).grid(row=13, column=1, sticky="ew", padx=8)
        ttk.Label(parent, text="Transport çıktı").grid(row=13, column=0, sticky="w", pady=5)
        self._combo(parent, 14, 0, "Backend", self.transport_backend, ("simulation", "live-udp"))
        ttk.Label(parent, text="UDP hedef host").grid(row=15, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.transport_host).grid(row=15, column=1, sticky="ew", padx=8)
        self._spin(parent, 15, 2, "UDP port", self.transport_port, 1, 65535)
        ttk.Checkbutton(parent, text="Live UDP XOR-FEC", variable=self.transport_fec).grid(
            row=16, column=1, sticky="w", padx=8, pady=5
        )
        ttk.Label(
            parent,
            text="Video desteği I-frame veya GOP/P-frame SWC kodlaması kullanır. Paket aktarımı simülasyondur.",
            wraplength=700,
        ).grid(row=11, column=0, columnspan=3, sticky="w", pady=8)

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, callback, save=False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Button(parent, text="Kaydet..." if save else "Seç...", command=callback).grid(row=row, column=2, padx=(8, 0), pady=3)

    def _combo(self, parent: ttk.Frame, row: int, column: int, label: str, variable, values: tuple[str, ...]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=4)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=12).grid(
            row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=4
        )

    def _entry(self, parent: ttk.Frame, row: int, column: int, label: str, variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=4)
        ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=4)

    def _spin(self, parent: ttk.Frame, row: int, column: int, label: str, variable, minimum: int, maximum: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=4)
        ttk.Spinbox(parent, textvariable=variable, from_=minimum, to=maximum, width=10).grid(
            row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=4
        )

    def _add_action(self, parent: ttk.Frame, text: str, callback, column: int) -> None:
        button = ttk.Button(parent, text=text, command=callback)
        button.grid(row=0, column=column, sticky="ew", padx=3)
        self.busy_widgets.append(button)
