"""TR: Arka plan iş parçacığı ve iptal durumunu yönetir. / EN: Manages background work and cancellation state."""

from __future__ import annotations

from .gui_support import *  # noqa: F401,F403 - shared GUI dependencies / ortak GUI bağımlılıkları


class GuiAsyncMixin:
    """TR: Arka plan iş parçacığı ve iptal durumunu yönetir. / EN: Manages background work and cancellation state."""

    def _run_async(self, work, on_success, operation_name: str = "Islem", *,
        # TR: Uzun işleri Tk ana iş parçacığını kilitlemeden arka planda çalıştırır.
        # EN: Run long operations in the background without blocking Tk's main thread.
                   progress_total: int | None = None, with_progress: bool = False) -> None:
        if self.operation_started_at is not None:
            return
        self.cancel_event.clear()
        self.operation_started_at = time.perf_counter()
        self.operation_name = operation_name
        self.status.set("İşlem yapılıyor...")
        self.status.set(f"{operation_name} yapiliyor... (gecen sure: 0.0 s)")
        if progress_total:
            self.progress.configure(mode="determinate", maximum=progress_total, value=0)
        else:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        self.cancel_button.configure(state="normal")
        for widget in self.busy_widgets:
            widget.configure(state="disabled")

        def runner() -> None:
            try:
                def report_progress(current: int) -> None:
                    self.async_queue.put(("progress", int(current), int(progress_total or 0)))

                result = work(report_progress) if with_progress else work()
            except Exception as exc:  # Pass exceptions to the Tk main thread.
                self.async_queue.put(("error", exc))
            else:
                if self.cancel_event.is_set():
                    self.async_queue.put(("cancelled", None))
                else:
                    self.async_queue.put(("success", on_success, result))

        threading.Thread(target=runner, daemon=True).start()

    def _poll_async(self) -> None:
        try:
            while True:
                item = self.async_queue.get_nowait()
                if item[0] == "error":
                    self._show_error(item[1])
                elif item[0] == "cancelled":
                    self.status.set("Islem iptal edildi; mevcut arka plan adimi tamamlandi.")
                elif item[0] == "progress":
                    current, total = item[1], item[2]
                    if total:
                        self.progress.configure(mode="determinate", maximum=total, value=current)
                        elapsed = time.perf_counter() - (self.operation_started_at or time.perf_counter())
                        eta = elapsed * (total - current) / current if current else 0.0
                        self.status.set(
                            f"{self.operation_name}: {current}/{total} - gecen {elapsed:.1f} s, tahmini kalan {eta:.1f} s"
                        )
                    continue
                else:
                    item[1](item[2])
                self._enable_actions()
        except queue.Empty:
            pass
        if self.operation_started_at is not None and not self.cancel_event.is_set():
            elapsed = time.perf_counter() - self.operation_started_at
            self.status.set(f"{self.operation_name} yapiliyor... (gecen sure: {elapsed:.1f} s)")
        if self.winfo_exists():
            self.after(50, self._poll_async)

    def _enable_actions(self) -> None:
        self.progress.stop()
        self.cancel_button.configure(state="disabled")
        for widget in self.busy_widgets:
            widget.configure(state="normal")
        self.operation_started_at = None
        self.operation_name = ""

    def _cancel_current(self) -> None:
        if self.operation_started_at is None:
            return
        self.cancel_event.set()
        self.status.set("Iptal istendi; mevcut codec adimi sonlandiriliyor...")

    def _show_error(self, error: Exception) -> None:
        self.last_error_text = f"{type(error).__name__}: {error}"
        self.status.set(f"Hata: {error}")
        if hasattr(self, "copy_error_button"):
            self.copy_error_button.configure(state="normal")
        if hasattr(self, "benchmark_log"):
            self._benchmark_log("HATA: " + self.last_error_text)
        messagebox.showerror("İşlem başarısız", str(error))

    def _copy_last_error(self) -> None:
        if not self.last_error_text:
            return
        self.clipboard_clear()
        self.clipboard_append(self.last_error_text)
        self.update()
        self.status.set("Hata ayrıntısı panoya kopyalandı.")
