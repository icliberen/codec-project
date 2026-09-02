"""Tkinter desktop interface for the Smart Wavelet Codec."""

from __future__ import annotations

from .gui_support import *  # noqa: F401,F403 - shared GUI dependencies / ortak GUI bağımlılıkları
from .gui_layout import GuiLayoutMixin
from .gui_settings import GuiSettingsMixin
from .gui_roi import GuiRoiMixin
from .gui_operations import GuiOperationsMixin
from .gui_visuals import GuiVisualsMixin
from .gui_benchmark import GuiBenchmarkMixin
from .gui_async import GuiAsyncMixin


class SmartCodecApp(
    GuiLayoutMixin,
    GuiSettingsMixin,
    GuiRoiMixin,
    GuiOperationsMixin,
    GuiVisualsMixin,
    GuiBenchmarkMixin,
    GuiAsyncMixin,
    tk.Tk,
):
    """TR: GUI kabuğu; özellikler sorumluluk bazlı mixin modüllerindedir.
    EN: GUI shell; features live in responsibility-focused mixin modules.
    """

    def __init__(self, settings_path: str | Path | None = None) -> None:
        super().__init__()
        self.dnd_enabled = False
        self.dnd_files = None
        try:
            from tkinterdnd2 import DND_FILES as dnd_files, TkinterDnD
            TkinterDnD._require(self)
            self.dnd_files = dnd_files
            self.dnd_enabled = True
        except Exception:
            self.dnd_enabled = False
        self.title("Smart Wavelet Codec")
        self.geometry("1120x780")
        self.minsize(940, 650)

        self.input_path = tk.StringVar()
        self.encoded_path = tk.StringVar()
        self.decoded_path = tk.StringVar()
        self.mode = tk.StringVar(value="lossy")
        self.codec_method = tk.StringVar(value="dwt")
        self.profile = tk.StringVar(value="Özel")
        self.wavelet = tk.StringVar(value="haar")
        self.level = tk.IntVar(value=2)
        self.step = tk.DoubleVar(value=12.0)
        self.target_bpp = tk.StringVar(value="")
        self.target_psnr = tk.StringVar(value="")
        self.quantizer = tk.StringVar(value="uniform")
        self.colorspace = tk.StringVar(value="ycbcr")
        self.standard_quality = tk.IntVar(value=75)
        self.standard_rate = tk.DoubleVar(value=4.0)
        self.roi_boxes = tk.StringVar()
        self.roi_mask_path = tk.StringVar()
        self.roi_strength = tk.DoubleVar(value=0.65)
        self.roi_feather = tk.IntVar(value=0)
        self.transport_segments = tk.BooleanVar(value=False)
        self.transport_tiles = tk.BooleanVar(value=False)
        self.transport_tile_size = tk.IntVar(value=64)
        self.yolo_model = tk.StringVar(value="auto")
        self.restoration_model = tk.StringVar()
        self.ai_reconstruction = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Hazır. Bir görüntü seçin.")
        self.metrics_text = tk.StringVar(value="Henüz ölçüm yok.")
        self.original_photo: ImageTk.PhotoImage | None = None
        self.decoded_photo: ImageTk.PhotoImage | None = None
        self.restored_photo: ImageTk.PhotoImage | None = None
        self.difference_photo: ImageTk.PhotoImage | None = None
        self.comparison_photo: ImageTk.PhotoImage | None = None
        self.original_array: np.ndarray | None = None
        self.decoded_array: np.ndarray | None = None
        self.restored_array: np.ndarray | None = None
        self.comparison_mode = tk.StringVar(value="Side-by-side")
        self.comparison_target = tk.StringVar(value="Decoded")
        self.comparison_position = tk.DoubleVar(value=0.5)
        self.comparison_zoom = tk.DoubleVar(value=1.0)
        self._comparison_pan_anchor: tuple[int, int] | None = None
        self.current_roi_mask: np.ndarray | None = None
        self.semantic_roi_mask: np.ndarray | None = None
        self.last_encode_info: dict | None = None
        self.original_display_size: tuple[int, int] | None = None
        self.original_source_shape: tuple[int, int] | None = None
        self.roi_start: tuple[int, int] | None = None
        self.roi_preview_rect: int | None = None
        self.benchmark_plot_path: Path | None = None
        self.busy_widgets: list[tk.Widget] = []
        self.async_queue: queue.Queue = queue.Queue()
        self.settings_path = Path(settings_path) if settings_path else Path.cwd() / "outputs" / "gui_settings.json"
        self.cancel_event = threading.Event()
        self.operation_started_at: float | None = None
        self.operation_name = ""
        self.last_error_text = ""
        self._closing = False

        self._build_ui()
        self._load_settings()
        if self.dnd_enabled and self.dnd_files is not None and hasattr(self, "drop_target_register"):
            self.drop_target_register(self.dnd_files)
            self.dnd_bind("<<Drop>>", self._handle_drop)
        self.mode.trace_add("write", self._mode_changed)
        self.profile.trace_add("write", self._profile_changed)
        self.status.set("Hazır. " + format_dependency_status(self.yolo_model.get()))
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll_async)

def main() -> None:
    app = SmartCodecApp()
    app.mainloop()


if __name__ == "__main__":

    main()
