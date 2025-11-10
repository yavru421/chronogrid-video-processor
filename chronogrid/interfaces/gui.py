#!/usr/bin/env python3
"""
Chronogrid GUI - Graphical User Interface for Video Processing

A modern GUI for batch processing videos into chronogrids with AI analysis.
"""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageTk

# Drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
except ImportError:
    DND_FILES = None  # type: ignore
    TkinterDnD = None  # type: ignore

# Markdown rendering support
try:
    from tkhtmlview import HTMLLabel  # type: ignore
except ImportError:
    HTMLLabel = None  # type: ignore

# Optional dependencies for palette extraction / splash playback
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from chronogrid.core.processing import ensure_ffmpeg, process_video

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGO_VIDEO_PATH = PROJECT_ROOT / "0001-0300.mp4"

ThemePalette = Dict[str, str]

DEFAULT_THEME: ThemePalette = {
    "background": "#05060b",
    "surface": "#0d1220",
    "surface_alt": "#161d2f",
    "canvas": "#1a2135",
    "accent": "#f19c38",
    "accent_alt": "#53c8ff",
    "text": "#f6f8ff",
    "muted_text": "#9fb2d0",
    "border": "#1f283d",
    "focus": "#f6c165",
}


def _clamp_channel(value: float) -> int:
    return int(max(0, min(255, round(value))))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (_clamp_channel(channel) for channel in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix_colors(
    base: tuple[int, int, int],
    target: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    return tuple(
        _clamp_channel(base[index] * (1 - ratio) + target[index] * ratio)
        for index in range(3)
    )


def _luma(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def derive_theme_from_logo(video_path: Path) -> ThemePalette:
    """Derive a UI palette from the Chronogrid logo video."""

    if cv2 is None or np is None or not video_path.exists():
        return DEFAULT_THEME.copy()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return DEFAULT_THEME.copy()

    frames: List[np.ndarray] = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(total_frames // 24, 1) if total_frames else 4
    frame_index = 0

    while len(frames) < 24:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % stride == 0:
            resized = cv2.resize(frame, (96, 54))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame.reshape(-1, 3))
        frame_index += 1

    capture.release()
    if not frames:
        return DEFAULT_THEME.copy()

    data = np.vstack(frames).astype(np.float32)
    cluster_count = min(4, len(data))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        25,
        0.4,
    )
    _, _, centers = cv2.kmeans(
        data,
        cluster_count,
        None,
        criteria,
        6,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers.astype(int)

    brightness = np.dot(centers, [0.2126, 0.7152, 0.0722])
    bg_index = int(np.argmin(brightness))
    bg_color = tuple(int(value) for value in centers[bg_index])
    highlight_index = int(np.argmax(brightness))
    highlight = tuple(int(value) for value in centers[highlight_index])

    distances = np.linalg.norm(centers - centers[bg_index], axis=1)
    accent_index = int(np.argmax(distances))
    accent_color = tuple(int(value) for value in centers[accent_index])
    secondary_index = int(np.argsort(distances)[-2]) if cluster_count > 2 else highlight_index
    accent_alt = tuple(int(value) for value in centers[secondary_index])

    surface = _mix_colors(bg_color, highlight, 0.2)
    surface_alt = _mix_colors(bg_color, highlight, 0.35)
    border = _mix_colors(bg_color, highlight, 0.1)
    text_color = "#f8fbff" if _luma(bg_color) < 155 else "#111217"
    muted_text = _rgb_to_hex(_mix_colors(highlight, bg_color, 0.6))
    focus = _rgb_to_hex(_mix_colors(accent_color, highlight, 0.5))

    return {
        "background": _rgb_to_hex(bg_color),
        "surface": _rgb_to_hex(surface),
        "surface_alt": _rgb_to_hex(surface_alt),
        "canvas": _rgb_to_hex(_mix_colors(bg_color, highlight, 0.25)),
        "accent": _rgb_to_hex(accent_color),
        "accent_alt": _rgb_to_hex(accent_alt),
        "text": text_color,
        "muted_text": muted_text,
        "border": _rgb_to_hex(border),
        "focus": focus,
    }


def apply_chronogrid_theme(root: tk.Tk, palette: ThemePalette) -> None:
    """Apply palette-driven styling to ttk widgets."""

    root.configure(bg=palette["background"])
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    base_font = ("Segoe UI", 10)
    heading_font = ("Segoe UI", 12, "bold")

    style.configure("Chronogrid.TFrame", background=palette["background"])
    style.configure("Chronogrid.Surface.TFrame", background=palette["surface"])
    style.configure(
        "Chronogrid.TLabel",
        background=palette["background"],
        foreground=palette["text"],
        font=base_font,
    )
    style.configure(
        "Chronogrid.Heading.TLabel",
        background=palette["background"],
        foreground=palette["text"],
        font=heading_font,
    )
    style.configure(
        "Chronogrid.TButton",
        background=palette["accent"],
        foreground=palette["background"],
        font=("Segoe UI", 10, "bold"),
        borderwidth=0,
        focuscolor=palette["focus"],
        padding=(14, 7),
        relief="flat",
    )
    style.map(
        "Chronogrid.TButton",
        background=[
            ("pressed", palette["accent_alt"]),
            ("active", palette["accent_alt"]),
            ("disabled", palette["surface_alt"]),
        ],
        foreground=[("disabled", palette["muted_text"])],
    )
    style.configure(
        "Chronogrid.TNotebook",
        background=palette["background"],
        borderwidth=0,
    )
    style.configure(
        "Chronogrid.TNotebook.Tab",
        background=palette["surface"],
        foreground=palette["muted_text"],
        padding=(18, 10),
    )
    style.map(
        "Chronogrid.TNotebook.Tab",
        background=[
            ("selected", palette["surface_alt"]),
            ("active", palette["surface_alt"]),
        ],
        foreground=[("selected", palette["text"])],
    )
    style.configure(
        "Chronogrid.Horizontal.TProgressbar",
        troughcolor=palette["surface"],
        background=palette["accent"],
        bordercolor=palette["border"],
        lightcolor=palette["accent"],
        darkcolor=palette["accent"],
    )
    style.configure(
        "Chronogrid.TRadiobutton",
        background=palette["surface"],
        foreground=palette["text"],
    )


class ChronogridSplashScreen:
    """Splash screen that loops the Chronogrid logo video while the GUI loads."""

    def __init__(
        self,
        parent: tk.Tk,
        palette: ThemePalette,
        video_path: Path = LOGO_VIDEO_PATH,
        duration_ms: int = 3600,
        on_finish: Optional[Callable[[], None]] = None,
    ) -> None:
        self.parent = parent
        self.palette = palette
        self.video_path = video_path
        self.duration_ms = duration_ms
        self.on_finish = on_finish
        self._cap = self._open_capture()
        self._frame_size = (640, 320)
        self._frame_delay = 42
        self._after_token: Optional[str] = None
        self._current_image: Optional[ImageTk.PhotoImage] = None

        self.window = tk.Toplevel(parent)
        self.window.overrideredirect(True)
        self.window.configure(bg=palette["background"])
        self.window.attributes("-topmost", True)

        width, height = 720, 420
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        x_position = int((screen_w - width) / 2)
        y_position = int((screen_h - height) / 2)
        self.window.geometry(f"{width}x{height}+{x_position}+{y_position}")

        self.video_label = tk.Label(
            self.window,
            bg=palette["background"],
            borderwidth=0,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=24, pady=(24, 12))

        self.caption_label = tk.Label(
            self.window,
            text="Initializing visual timeline systems…",
            font=("Segoe UI", 11),
            fg=palette["muted_text"],
            bg=palette["background"],
        )
        self.caption_label.pack(pady=(0, 6))

        self.hint_label = tk.Label(
            self.window,
            text="Tip: Drag multiple clips to compare chronogrids side-by-side.",
            font=("Segoe UI", 9),
            fg=palette["muted_text"],
            bg=palette["background"],
        )
        self.hint_label.pack()

        self.window.bind("<Button-1>", lambda _event: self.close())
        self.window.bind("<Escape>", lambda _event: self.close())

        if self._cap is None:
            self._show_placeholder()
        else:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 24
            self._frame_delay = max(34, int(1000 / fps))
            self._animate()

        self.window.after(self.duration_ms, self.close)

    def _open_capture(self) -> Optional["cv2.VideoCapture"]:
        if cv2 is None or not self.video_path.exists():
            return None
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return capture

    def _show_placeholder(self) -> None:
        self.video_label.configure(
            text="Chronogrid",
            font=("Segoe UI", 28, "bold"),
            fg=self.palette["accent"],
        )

    def _animate(self) -> None:
        if self._cap is None:
            return
        success, frame = self._cap.read()
        if not success:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self._cap.read()
            if not success:
                return

        resized = cv2.resize(frame, self._frame_size)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        self._current_image = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self._current_image)
        self._after_token = self.window.after(self._frame_delay, self._animate)

    def close(self) -> None:
        if not self.window.winfo_exists():
            return
        if self._after_token:
            try:
                self.window.after_cancel(self._after_token)
            except tk.TclError:
                pass
            self._after_token = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.window.destroy()
        if self.on_finish:
            self.parent.after(0, self.on_finish)


class ChronogridControlRoom:
    def __init__(self, root: tk.Tk, palette: ThemePalette) -> None:
        self.root = root
        self.palette = palette
        apply_chronogrid_theme(self.root, palette)

        self.root.title("Chronogrid Control Room - Visual Timeline Analysis")
        self.root.geometry("1280x820")
        self.root.minsize(1024, 720)

        self.file_listbox = tk.Listbox(
            self.root,
            selectmode=tk.MULTIPLE,
            height=10,
            activestyle="none",
            bg=self.palette["surface"],
            fg=self.palette["text"],
            selectbackground=self.palette["accent"],
            selectforeground=self.palette["background"],
            highlightthickness=0,
            relief=tk.FLAT,
            bd=0,
            exportselection=False,
            font=("Segoe UI", 10),
        )
        self.file_listbox.pack(fill=tk.X, padx=14, pady=(10, 6))

        button_frame = ttk.Frame(self.root, style="Chronogrid.TFrame")
        button_frame.pack(fill=tk.X, padx=14, pady=(0, 8))

        ttk.Button(button_frame, text="➕ Add Files", style="Chronogrid.TButton", command=self.add_files).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="➖ Remove Selected", style="Chronogrid.TButton", command=self.remove_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="🗑️ Clear All", style="Chronogrid.TButton", command=self.clear_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(button_frame, text="🚀 Generate Chronogrids", style="Chronogrid.TButton", command=self.generate_chronogrids).pack(side=tk.LEFT, padx=4)

        self.notebook = ttk.Notebook(self.root, style="Chronogrid.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        preview_frame = ttk.Frame(self.notebook, style="Chronogrid.Surface.TFrame")
        self.notebook.add(preview_frame, text="Chronogrid Preview")
        self.preview_canvas = tk.Canvas(
            preview_frame,
            bg=self.palette["canvas"],
            highlightthickness=0,
            bd=0,
            height=420,
        )
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.preview_canvas.create_text(
            320,
            200,
            text="Chronogrid previews will materialize here",
            fill=self.palette["muted_text"],
            font=("Segoe UI", 13),
        )

        analysis_frame = ttk.Frame(self.notebook, style="Chronogrid.Surface.TFrame")
        self.notebook.add(analysis_frame, text="AI Analysis")
        self.analysis_text = tk.Text(
            analysis_frame,
            wrap=tk.WORD,
            font=("JetBrains Mono", 10),
            bg=self.palette["surface"],
            fg=self.palette["text"],
            insertbackground=self.palette["accent"],
            selectbackground=self.palette["accent"],
            selectforeground=self.palette["background"],
            relief=tk.FLAT,
            bd=0,
        )
        analysis_scrollbar = ttk.Scrollbar(analysis_frame, command=self.analysis_text.yview)
        self.analysis_text.config(yscrollcommand=analysis_scrollbar.set)
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        analysis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)

        log_frame = ttk.Frame(self.notebook, style="Chronogrid.Surface.TFrame")
        self.notebook.add(log_frame, text="Processing Log")
        self.log_mode = tk.StringVar(value="terminal")
        toggle_frame = ttk.Frame(log_frame, style="Chronogrid.Surface.TFrame")
        toggle_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        ttk.Radiobutton(
            toggle_frame,
            text="Terminal",
            variable=self.log_mode,
            value="terminal",
            command=self.update_log_view,
            style="Chronogrid.TRadiobutton",
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(
            toggle_frame,
            text="Rendered",
            variable=self.log_mode,
            value="markdown",
            command=self.update_log_view,
            style="Chronogrid.TRadiobutton",
        ).pack(side=tk.LEFT)

        self.log_text = tk.Text(
            log_frame,
            height=14,
            wrap=tk.WORD,
            font=("JetBrains Mono", 10),
            bg=self.palette["surface"],
            fg=self.palette["text"],
            insertbackground=self.palette["accent"],
            selectbackground=self.palette["accent"],
            selectforeground=self.palette["background"],
            relief=tk.FLAT,
            bd=0,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        self.log_text.tag_configure("info", foreground=self.palette["text"])
        self.log_text.tag_configure("success", foreground=self.palette["accent"])
        self.log_text.tag_configure("error", foreground="#ff6b6b")

        self.markdown_label: Optional[Any] = None
        if HTMLLabel:
            self.markdown_label = HTMLLabel(
                log_frame,
                html="",
                background=self.palette["surface"],
                foreground=self.palette["text"],
            )
            self.markdown_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
            self.markdown_label.pack_forget()

        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            mode="determinate",
            style="Chronogrid.Horizontal.TProgressbar",
        )
        self.progress.pack(fill=tk.X, padx=14, pady=(0, 6))

        self.status_label = ttk.Label(
            self.root,
            text="Ready. Drop clips or add files to begin.",
            style="Chronogrid.TLabel",
            anchor=tk.W,
        )
        self.status_label.pack(fill=tk.X, padx=14, pady=(0, 6))

        self.sidebar_video_label = ttk.Label(
            self.root,
            text="Video: None",
            style="Chronogrid.TLabel",
            anchor=tk.W,
        )
        self.sidebar_video_label.pack(anchor=tk.W, padx=14)
        self.sidebar_folder_label = ttk.Label(
            self.root,
            text="Folder: None",
            style="Chronogrid.TLabel",
            anchor=tk.W,
        )
        self.sidebar_folder_label.pack(anchor=tk.W, padx=14, pady=(0, 8))

        if DND_FILES:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, event: Any) -> None:
        paths = self.root.tk.splitlist(event.data)
        added = 0
        for path in paths:
            candidate = Path(path)
            if candidate.is_file():
                if candidate.suffix.lower() in {'.mp4', '.mov', '.m4v'}:
                    self.file_listbox.insert(tk.END, str(candidate))
                    added += 1
            elif candidate.is_dir():
                for file in candidate.rglob('*'):
                    if file.is_file() and file.suffix.lower() in {'.mp4', '.mov', '.m4v'}:
                        self.file_listbox.insert(tk.END, str(file))
                        added += 1
        if added:
            self.status_label.config(text=f"Added {added} item(s). Ready to process.")

    def add_files(self) -> None:
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.mov *.m4v")])
        for file in files:
            self.file_listbox.insert(tk.END, file)
        if files:
            self.status_label.config(text=f"Queued {len(files)} new file(s).")

    def remove_selected(self) -> None:
        selected = self.file_listbox.curselection()
        for index in reversed(selected):
            self.file_listbox.delete(index)
        if selected:
            self.status_label.config(text=f"Removed {len(selected)} item(s).")

    def clear_all(self) -> None:
        self.file_listbox.delete(0, tk.END)
        self.status_label.config(text="Cleared queue. Add clips to continue.")

    def log(self, message: str, tag: str = "info") -> None:
        self.log_text.insert(tk.END, message + "\n", tag)
        self.log_text.see(tk.END)
        self.update_markdown_log()

    def update_markdown_log(self) -> None:
        if self.markdown_label:
            log_content = self.log_text.get("1.0", tk.END)
            html = (
                "<pre style=\"color:{text};background:{bg};font-family:'JetBrains Mono',monospace;\">"
                "{content}</pre>"
            ).format(text=self.palette["text"], bg=self.palette["surface"], content=log_content)
            self.markdown_label.set_html(html)

    def update_log_view(self) -> None:
        mode = self.log_mode.get()
        if mode == "terminal":
            self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
            if self.markdown_label:
                self.markdown_label.pack_forget()
        else:
            self.log_text.pack_forget()
            if self.markdown_label:
                self.markdown_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

    def set_current_video(self, video_path: str) -> None:
        self.sidebar_video_label.config(text=f"Video: {Path(video_path).name}")

    def set_current_folder(self, folder_path: str) -> None:
        self.sidebar_folder_label.config(text=f"Folder: {folder_path}")

    def generate_chronogrids(self) -> None:
        selected = self.file_listbox.curselection()
        if not selected:
            messagebox.showwarning("No Selection", "Please select video files to process.")
            return

        videos = [self.file_listbox.get(index) for index in selected]
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "Processing videos… AI analysis will appear here.\n")
        self.progress['maximum'] = len(videos)
        self.progress['value'] = 0
        self.status_label.config(text=f"Processing {len(videos)} video(s)…")

        threading.Thread(target=self._process_videos, args=(videos,), daemon=True).start()

    def _process_videos(self, videos: List[str]) -> None:
        for index, video in enumerate(videos, start=1):
            self.log(f"Processing {video}…")
            self.set_current_video(video)
            self.set_current_folder(str(Path(video).parent))
            try:
                result = process_video(Path(video))
                self.log(f"✓ Completed {video}", tag="success")

                if result.chronogrid_path.exists():
                    try:
                        self.preview_canvas.update_idletasks()
                        actual_width = max(self.preview_canvas.winfo_width(), 1)
                        actual_height = max(self.preview_canvas.winfo_height(), 1)
                        target_width = max(actual_width, 640)
                        target_height = max(actual_height, 420)
                        with Image.open(result.chronogrid_path) as src_img:
                            display_img = src_img.copy()
                        display_img.thumbnail(
                            (
                                max(target_width - 32, 200),
                                max(target_height - 32, 200),
                            ),
                        )
                        self.preview_img = ImageTk.PhotoImage(display_img)
                        self.preview_canvas.delete("all")
                        self.preview_canvas.create_image(
                            actual_width / 2,
                            actual_height / 2,
                            image=self.preview_img,
                        )
                        self.notebook.select(0)
                    except Exception as exc:
                        self.log(f"✗ Failed to display chronogrid: {exc}", tag="error")

                if result.analysis_text:
                    self.analysis_text.delete(1.0, tk.END)
                    self.analysis_text.insert(
                        tk.END,
                        f"AI Analysis for: {Path(video).name}\n{'=' * 50}\n\n{result.analysis_text}",
                    )
                    self.notebook.select(1)
                    self.log("✓ AI analysis updated", tag="success")

            except Exception as exc:
                self.log(f"✗ Failed {video}: {exc}", tag="error")
            finally:
                self.progress['value'] = index

        self.log("Processing complete.", tag="success")
        self.status_label.config(text="Processing complete. Review your grids and analyses.")
        messagebox.showinfo("Done", "Chronogrid generation completed with AI analysis!")


def main() -> None:
    try:
        ensure_ffmpeg()
    except RuntimeError as exc:
        error_root = tk.Tk()
        error_root.withdraw()
        messagebox.showerror("FFmpeg Missing", str(exc))
        error_root.destroy()
        print(str(exc))
        sys.exit(1)

    root_class = TkinterDnD.Tk if TkinterDnD else tk.Tk  # type: ignore
    root = root_class()
    palette = derive_theme_from_logo(LOGO_VIDEO_PATH)
    root.withdraw()

    def launch_app() -> None:
        ChronogridControlRoom(root, palette)
        root.deiconify()

    splash = ChronogridSplashScreen(
        parent=root,
        palette=palette,
        video_path=LOGO_VIDEO_PATH,
        on_finish=launch_app,
    )
    root.mainloop()


if __name__ == '__main__':
    main()
