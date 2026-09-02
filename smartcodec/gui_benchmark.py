"""TR: GUI benchmark üretimini ve sonuç gösterimini yönetir. / EN: Manages GUI benchmark execution and result display."""

from __future__ import annotations

from .gui_support import *  # noqa: F401,F403 - shared GUI dependencies / ortak GUI bağımlılıkları


class GuiBenchmarkMixin:
    """TR: GUI benchmark üretimini ve sonuç gösterimini yönetir. / EN: Manages GUI benchmark execution and result display."""

    def _generate_samples(self) -> None:
        output = Path(self.benchmark_input.get())
        self._run_async(lambda: generate_samples(output), lambda paths: self._benchmark_log(f"{len(paths)} örnek görüntü üretildi: {output}"))

    def _generate_dataset(self) -> None:
        output = Path(self.benchmark_input.get())
        self._run_async(lambda: generate_dataset(output), lambda paths: self._benchmark_log(f"{len(paths)} kategorili örnek üretildi: {output}"))

    def _benchmark(self) -> None:
        # TR: GUI benchmark ayarlarını doğrulayıp uygun deney akışını başlatır.
        # EN: Validate GUI benchmark settings and start the appropriate experiment pipeline.
        try:
            steps = [float(value.strip()) for value in self.benchmark_steps.get().split(",") if value.strip()]
            wavelets = [value.strip() for value in self.benchmark_wavelets.get().split(",") if value.strip()]
            if not steps or not wavelets:
                raise ValueError("En az bir step ve wavelet girin")
        except Exception as exc:
            self._show_error(exc)
            return

        input_dir = self.benchmark_input.get().strip()
        output_dir = self.benchmark_output.get().strip()
        normalization = self.benchmark_normalization.get()
        target_text = self.benchmark_target.get().strip()
        roi_mask_path = self.benchmark_roi_mask.get().strip()
        quantizer = self.benchmark_quantizer.get()
        allocation_method = self.benchmark_allocation.get()
        if normalization not in {"grid", "target-psnr", "target-bpp", "roi-compare"}:
            self._show_error(ValueError("Geçersiz benchmark normalize modu"))
            return
        if normalization != "grid" and not target_text:
            self._show_error(ValueError("Normalize benchmark için hedef değeri girin"))
            return
        try:
            target = float(target_text) if target_text else None
        except ValueError as exc:
            self._show_error(ValueError("Benchmark hedefi sayısal olmalı"))
            return
        if normalization != "grid" and (target is None or target <= 0):
            self._show_error(ValueError("Benchmark hedefi pozitif olmalı"))
            return
        if normalization == "roi-compare" and (not roi_mask_path or not Path(roi_mask_path).is_file()):
            self._show_error(ValueError("ROI karşılaştırma modu için geçerli bir ROI maskesi seçin"))
            return
        if normalization == "grid":
            work = lambda: run_benchmark(input_dir, output_dir, step_values=steps, wavelets=wavelets,
                                         quantizer=quantizer, allocation_method=allocation_method)
        elif normalization == "target-psnr":
            work = lambda: run_normalized_benchmark(
                input_dir, output_dir, target_psnr=target, step_values=steps, wavelets=wavelets,
                quantizer=quantizer, allocation_method=allocation_method,
            )
        elif normalization == "target-bpp":
            work = lambda: run_normalized_benchmark(
                input_dir, output_dir, target_bpp=target, step_values=steps, wavelets=wavelets,
                quantizer=quantizer, allocation_method=allocation_method,
            )
        else:
            work = lambda: run_roi_comparison_benchmark(
                input_dir, output_dir, target_bpp=target, roi_mask_path=roi_mask_path,
                step_values=steps, wavelets=wavelets, quantizer=quantizer,
                allocation_method=allocation_method,
            )
        self._run_async(
            work,
            lambda rows: self._benchmark_done(rows, output_dir),
        )

    def _benchmark_done(self, rows: list[dict], output_dir: str) -> None:
        for item in self.benchmark_table.get_children():
            self.benchmark_table.delete(item)
        for row in rows:
            self.benchmark_table.insert(
                "", "end",
                values=(row.get("category", ""), row["image"], row["codec"], row["step"],
                        f"{float(row['bits_per_pixel']):.3f}", row["psnr"], row["ssim"], row.get("status", "")),
            )
        regular_plot = Path(output_dir) / "rate_distortion.png"
        roi_plot = Path(output_dir) / "roi_rate_distortion.png"
        self.benchmark_plot_path = roi_plot if roi_plot.is_file() else regular_plot
        statuses = {}
        for row in rows:
            status = str(row.get("status", "ok"))
            statuses[status] = statuses.get(status, 0) + 1
        status_text = ", ".join(f"{key}={value}" for key, value in sorted(statuses.items()))
        self._benchmark_log(f"Benchmark tamamlandı: {len(rows)} deney satırı ({status_text})\nÇıktı: {output_dir}")

    def _show_benchmark_plot(self) -> None:
        path = self.benchmark_plot_path or Path(self.benchmark_output.get()) / "rate_distortion.png"
        if not path.exists():
            messagebox.showinfo("Grafik", "Önce benchmark çalıştırın.")
            return
        window = tk.Toplevel(self)
        window.title("Rate-distortion grafiği")
        with Image.open(path) as image:
            image = image.copy()
        image.thumbnail((900, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label = ttk.Label(window, image=photo)
        label.image = photo
        label.pack(padx=10, pady=10)

    def _benchmark_log(self, text: str) -> None:
        self.benchmark_log.configure(state="normal")
        self.benchmark_log.insert("end", text + "\n")
        self.benchmark_log.see("end")
        self.benchmark_log.configure(state="disabled")
        self.status.set(text.splitlines()[0])

    def _open_output_folder(self) -> None:
        path = Path(self.encoded_path.get().strip() or Path.cwd()).parent
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except AttributeError:
            messagebox.showinfo("Çıktı klasörü", str(path))
