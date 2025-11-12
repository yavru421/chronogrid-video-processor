"""PyQt GUI for Chronogrid processing with responsive background execution."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from types import TracebackType

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, QTimer, QPoint, QEvent
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QCloseEvent, QMovie, QKeySequence
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QCloseEvent, QMovie
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QInputDialog,
    QDialog,
    QLineEdit,
    QDialogButtonBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QMenu,
    QScrollArea,
    QFrame,
    QMenuBar,
    QAction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chronogrid.gui")

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
VIDEO_FILE_FILTER = "Video Files (*.mp4 *.mov *.m4v *.avi *.mkv)"
DEFAULT_PROMPT = (
    "You are analyzing a 4x4 chronogrid of 16 frames extracted from a video. "
    "Describe ONLY what is visually present. Identify the primary subject(s), their actions, "
    "tools, animals, and notable objects. Mention the environment, lighting, and any progression "
    "or sequence that can be inferred from frame order. Avoid speculation, metaphors, or emotional "
    "interpretation—stick to concrete facts. End with a concise timeline-style bullet list summarizing "
    "the sequence in order."
)


def _install_exception_hook() -> None:
    """Ensure uncaught exceptions are logged instead of silently swallowed by Qt."""

    import traceback
    def handler(exc_type: type, exc_value: BaseException, exc_tb: TracebackType | None) -> None:
        logger.critical("Uncaught exception", exc_info=True)
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = handler


@dataclass(frozen=True)
class ProcessingOutcome:
    video_path: Path
    chronogrid_path: Path
    analysis_path: Path | None
    analysis_text: str | None


def _write_analysis_file(chronogrid_path: Path, analysis_text: str | None) -> Path | None:
    if not analysis_text:
        return None
    destination = chronogrid_path.with_name(f"{chronogrid_path.stem}_analysis.txt")
    destination.write_text(analysis_text.strip() + "\n", encoding="utf-8")
    return destination


class ZoomableImageLabel(QLabel):
    """A QLabel that supports zooming with mouse wheel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        # Traverse up the widget hierarchy to find the main window
        widget = self.parent()
        main_window = None

        while widget:
            if hasattr(widget, 'zoom_in_history_image'):
                main_window = widget
                break
            widget = widget.parent()

        if main_window:
            if event.angleDelta().y() > 0:
                main_window.zoom_in_history_image()
            else:
                main_window.zoom_out_history_image()
            event.accept()


def _resolve_asset_path(*parts: str) -> Path:
    """Support both PyInstaller bundles and source installs for asset lookup."""

    try:
        base = Path(getattr(sys, "_MEIPASS"))
    except (AttributeError, TypeError):
        base = Path(__file__).resolve().parents[3]
    return base.joinpath(*parts)


def _center_on_primary_screen(app: QApplication, widget: QWidget) -> None:
    screen = app.primaryScreen()
    if not screen:
        return
    geometry = screen.availableGeometry()
    widget.adjustSize()
    widget.move(
        geometry.center().x() - widget.width() // 2,
        geometry.center().y() - widget.height() // 2,
    )


def _estimate_movie_duration(movie: QMovie) -> int:
    frame_count = movie.frameCount()
    if frame_count <= 0:
        return 3500
    frame_delay = movie.nextFrameDelay() or 80
    duration = frame_count * frame_delay
    return max(2000, min(duration, 8000))


def _show_splash_animation(app: QApplication, gif_path: Path) -> None:
    """Display a lightweight animated splash using a QLabel/QMovie combo."""

    if not gif_path.exists():
        return

    movie = QMovie(str(gif_path))
    pixmap = QPixmap(str(gif_path)) if not movie.isValid() else None
    if not movie.isValid() and (pixmap is None or pixmap.isNull()):
        return

    splash = QLabel()
    flags = splash.windowFlags()
    for flag_name in ("SplashScreen", "FramelessWindowHint", "Tool"):
        flag_value = getattr(Qt, flag_name, None)
        if flag_value is not None:
            flags |= flag_value
    stay_on_top = getattr(Qt, "WindowStaysOnTopHint", None)
    if stay_on_top is not None:
        flags |= stay_on_top
    splash.setWindowFlags(flags)
    wa_delete = getattr(Qt, "WA_DeleteOnClose", None)
    if wa_delete is not None:
        splash.setAttribute(wa_delete, True)

    if movie.isValid():
        movie.setCacheMode(QMovie.CacheMode.CacheAll)
        movie.jumpToFrame(0)
        splash.setMovie(movie)
        movie.start()
        duration = _estimate_movie_duration(movie)
    else:
        splash.setPixmap(pixmap)
        duration = 2500

    _center_on_primary_screen(app, splash)
    splash.show()

    def _close() -> None:
        if movie.isValid():
            movie.stop()
        splash.close()
        if hasattr(app, "_chronogrid_splash"):
            delattr(app, "_chronogrid_splash")

    app._chronogrid_splash = splash  # prevent premature GC
    QTimer.singleShot(duration, _close)


class ProcessingThread(QThread):
    """Background worker that runs the Chronogrid pipeline to keep the UI responsive."""

    progress = pyqtSignal(int, str)  # percent complete, current filename
    artifact_ready = pyqtSignal(object)  # ProcessingOutcome
    failed = pyqtSignal(str)
    license_required = pyqtSignal(str)  # upgrade message

    def __init__(
        self,
        video_paths: Sequence[str],
        prompt: str,
        frame_step: int = 30,
        grid_size: int = 4,
        cleanup: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.video_paths = [Path(path) for path in video_paths]
        self.prompt = prompt
        self.frame_step = frame_step
        self.grid_size = grid_size
        self.cleanup = cleanup

    def run(self) -> None:  # noqa: D401 - required override
        from chronogrid.core.processing import process_video
        from chronogrid.core.licensing import FreemiumLimitError, LicenseError

        total = len(self.video_paths)
        if total == 0:
            return

        for index, path in enumerate(self.video_paths, start=1):
            if self.isInterruptionRequested():
                logger.info("Processing interrupted by user request.")
                return

            try:
                result = process_video(
                    path,
                    frame_step=self.frame_step,
                    grid_size=self.grid_size,
                    prompt=self.prompt,
                    cleanup=self.cleanup,
                )
            except FreemiumLimitError as e:
                self.license_required.emit(str(e))
                return
            except LicenseError as e:
                self.license_required.emit(f"License error: {e}")
                return
            except Exception as exc:  # pragma: no cover - surfaced via signal/UI
                logger.exception("Processing failed for %s", path)
                self.failed.emit(f"Processing failed for {path.name}: {exc}")
                return

            analysis_path = _write_analysis_file(result.chronogrid_path, result.analysis_text)
            outcome = ProcessingOutcome(
                video_path=path,
                chronogrid_path=result.chronogrid_path,
                analysis_path=analysis_path,
                analysis_text=result.analysis_text,
            )
            self.artifact_ready.emit(outcome)
            percent = int((index / total) * 100)
            self.progress.emit(percent, path.name)


class ChronogridGUI(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Chronogrid Video Processor")
        self.resize(1400, 800)
        self.setAcceptDrops(True)

        self.video_paths: list[Path] = []
        self.processing_thread: ProcessingThread | None = None
        self._encountered_error = False

        # UI components for processing tab
        self.file_label: QLabel
        self.select_button: QPushButton
        self.prompt_input: QTextEdit
        self.image_label: QLabel
        self.results_text: QTextEdit
        self.progress_bar: QProgressBar
        self.process_button: QPushButton

        # UI components for history tab
        self.history_tree: QTreeWidget
        self.history_preview_image: QLabel
        self.history_preview_text: QTextEdit
        self.history_summary_text: QTextEdit
        self.history_zoom_scroll_area: QScrollArea
        self.history_zoom_factor: float = 1.0
        self.history_original_pixmap: QPixmap = QPixmap()

        self.tabs: QTabWidget

        self._build_menu()
        self._build_ui()

    # ...existing code...

    def load_prompt(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Prompt From File", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read()
                self.prompt_input.setPlainText(prompt_text)
                QMessageBox.information(self, "Prompt Loaded", f"Prompt loaded from {file_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Load Error", f"Failed to load prompt: {exc}")

    def reset_prompt(self) -> None:
        self.prompt_input.setPlainText(DEFAULT_PROMPT)
        QMessageBox.information(self, "Prompt Reset", "Prompt has been reset to the default.")

    def _build_menu(self) -> None:
        """Build the application menu bar."""
        menubar = self.menuBar()

        # License menu
        license_menu = menubar.addMenu("License")

        enter_license_action = QAction("Enter License Key", self)
        enter_license_action.triggered.connect(self.enter_license_key)
        license_menu.addAction(enter_license_action)

        upgrade_action = QAction("Upgrade to Premium", self)
        upgrade_action.triggered.connect(self.show_upgrade_info)
        license_menu.addAction(upgrade_action)

        status_action = QAction("License Status", self)
        status_action.triggered.connect(self.show_license_status)
        license_menu.addAction(status_action)

    def enter_license_key(self) -> None:
        """Show dialog to enter license key."""
        from chronogrid.core.licensing import activate_license, LicenseError

        license_key, ok = QInputDialog.getText(
            self, "Enter License Key", "Enter your Chronogrid Premium license key:"
        )
        if ok and license_key.strip():
            try:
                email = activate_license(license_key.strip())
                QMessageBox.information(
                    self, "License Activated",
                    f"Premium license activated successfully!\n\nEmail: {email}"
                )
            except LicenseError as e:
                QMessageBox.critical(self, "Invalid License", str(e))

    def show_upgrade_info(self) -> None:
        """Show upgrade information dialog."""
        from chronogrid.core.licensing import get_upgrade_message

        message = get_upgrade_message()
        if message:
            QMessageBox.information(self, "Upgrade to Premium", message)
        else:
            QMessageBox.information(
                self, "Already Premium",
                "You have an active Chronogrid Premium license!\n\nEnjoy unlimited AI analysis."
            )

    def show_license_status(self) -> None:
        """Show current license status."""
        from chronogrid.core.licensing import has_valid_license, get_usage_count

        if has_valid_license():
            QMessageBox.information(
                self, "License Status",
                "✓ Premium license is active\n\nYou have unlimited AI analysis."
            )
        else:
            usage = get_usage_count()
            QMessageBox.information(
                self, "License Status",
                f"Free tier active\n\nAI analyses used this month: {usage}/5"
            )

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Processing tab
        self._build_processing_tab()

        # History tab
        self._build_history_tab()

    def _build_processing_tab(self) -> None:
        """Build the video processing tab."""
        processing_widget = QWidget()
        layout = QVBoxLayout(processing_widget)
        layout.setSpacing(12)

        file_layout = QHBoxLayout()
        self.file_label = QLabel("Drop video files here or click 'Select Files'")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("border: 2px dashed #aaa; padding: 20px;")

        self.select_button = QPushButton("Select Files")
        self.select_button.clicked.connect(self.select_files)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.select_button)
        layout.addLayout(file_layout)

        prompt_label = QLabel("Analysis Prompt")
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter an optional custom prompt…")
        self.prompt_input.setPlainText(DEFAULT_PROMPT)
        self.prompt_input.setFixedHeight(110)

        prompt_button_layout = QHBoxLayout()
        self.save_prompt_button = QPushButton("Save Prompt")
        self.save_prompt_button.clicked.connect(self.save_prompt)
        prompt_button_layout.addWidget(self.save_prompt_button)

        self.load_prompt_button = QPushButton("Load Prompt")
        self.load_prompt_button.clicked.connect(self.load_prompt)
        prompt_button_layout.addWidget(self.load_prompt_button)

        self.reset_prompt_button = QPushButton("Reset Prompt")
        self.reset_prompt_button.clicked.connect(self.reset_prompt)
        prompt_button_layout.addWidget(self.reset_prompt_button)

        layout.addWidget(prompt_label)
        layout.addWidget(self.prompt_input)
        layout.addLayout(prompt_button_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image_label = QLabel("Chronogrid will appear here")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumWidth(420)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background: #fafafa;")

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlainText("Analysis results will appear here.")

        splitter.addWidget(self.image_label)
        splitter.addWidget(self.results_text)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        layout.addWidget(self.progress_bar)

        self.process_button = QPushButton("Process Videos")
        self.process_button.clicked.connect(self.process_videos)
        layout.addWidget(self.process_button)

        self.tabs.addTab(processing_widget, "Process Videos")

    def _build_history_tab(self) -> None:
        """Build the history/preview tab."""
        history_widget = QWidget()
        layout = QHBoxLayout(history_widget)

        # Left panel - file tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        tree_label = QLabel("Previous Chronogrid Runs")
        left_layout.addWidget(tree_label)

        self.history_tree = QTreeWidget()
        self.history_tree.setHeaderLabel("Outputs Directory")
        self.history_tree.itemSelectionChanged.connect(self.on_history_selection_changed)
        self.history_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_tree.customContextMenuRequested.connect(self.show_history_context_menu)
        left_layout.addWidget(self.history_tree)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_history)
        left_layout.addWidget(refresh_button)

        layout.addWidget(left_panel, stretch=1)

        # Right panel - preview area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Image preview
        image_group = QGroupBox("Chronogrid Preview")
        image_layout = QVBoxLayout(image_group)

        # Zoom controls
        zoom_layout = QHBoxLayout()
        self.zoom_in_button = QPushButton("Zoom In (+)")
        self.zoom_in_button.clicked.connect(self.zoom_in_history_image)
        self.zoom_in_button.setEnabled(False)  # Initially disabled
        zoom_layout.addWidget(self.zoom_in_button)

        self.zoom_out_button = QPushButton("Zoom Out (-)")
        self.zoom_out_button.clicked.connect(self.zoom_out_history_image)
        self.zoom_out_button.setEnabled(False)  # Initially disabled
        zoom_layout.addWidget(self.zoom_out_button)

        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_history_zoom)
        self.reset_zoom_button.setEnabled(False)  # Initially disabled
        zoom_layout.addWidget(self.reset_zoom_button)

        zoom_layout.addStretch()
        image_layout.addLayout(zoom_layout)

        # Scrollable image area
        self.history_zoom_scroll_area = QScrollArea()
        self.history_zoom_scroll_area.setWidgetResizable(True)
        self.history_zoom_scroll_area.setMinimumHeight(300)
        self.history_zoom_scroll_area.setStyleSheet("border: 1px solid #ccc; background: #fafafa;")

        self.history_preview_image = ZoomableImageLabel("Select a chronogrid to preview")
        self.history_preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_preview_image.setStyleSheet("background: transparent;")
        self.history_zoom_scroll_area.setWidget(self.history_preview_image)

        image_layout.addWidget(self.history_zoom_scroll_area)
        right_layout.addWidget(image_group)

        # Text previews
        text_splitter = QSplitter(Qt.Orientation.Vertical)

        # Analysis text preview
        analysis_group = QGroupBox("Analysis Text")
        analysis_layout = QVBoxLayout(analysis_group)
        self.history_preview_text = QTextEdit()
        self.history_preview_text.setReadOnly(True)
        self.history_preview_text.setPlainText("Analysis text will appear here when you select a chronogrid.")
        analysis_layout.addWidget(self.history_preview_text)
        analysis_widget = QWidget()
        analysis_widget.setLayout(analysis_layout)
        text_splitter.addWidget(analysis_widget)

        # Semantic summary preview
        summary_group = QGroupBox("Semantic Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.history_summary_text = QTextEdit()
        self.history_summary_text.setReadOnly(True)
        self.history_summary_text.setPlainText("Semantic summary will appear here if available.")
        summary_layout.addWidget(self.history_summary_text)
        summary_widget = QWidget()
        summary_widget.setLayout(summary_layout)
        text_splitter.addWidget(summary_widget)

        text_splitter.setStretchFactor(0, 1)
        text_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(text_splitter, stretch=1)

        layout.addWidget(right_panel, stretch=2)

        self.tabs.addTab(history_widget, "Previous Runs")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Load initial history
        self.refresh_history()

    def refresh_history(self) -> None:
        """Refresh the history tree with current outputs directory contents."""
        self.history_tree.clear()

        outputs_dir = Path.cwd() / "outputs"
        if not outputs_dir.exists():
            root_item = QTreeWidgetItem(["No outputs directory found"])
            self.history_tree.addTopLevelItem(root_item)
            return

        # Create root item
        root_item = QTreeWidgetItem([str(outputs_dir)])
        self.history_tree.addTopLevelItem(root_item)

        # Add video directories
        for video_dir in sorted(outputs_dir.iterdir()):
            if video_dir.is_dir():
                video_item = QTreeWidgetItem([video_dir.name])
                root_item.addChild(video_item)

                # Add files in this directory
                for file_path in sorted(video_dir.iterdir()):
                    if file_path.is_file():
                        file_item = QTreeWidgetItem([file_path.name])
                        file_item.setData(0, Qt.ItemDataRole.UserRole, str(file_path))
                        video_item.addChild(file_item)

        root_item.setExpanded(True)
        # Expand first video directory if it exists
        if root_item.childCount() > 0:
            root_item.child(0).setExpanded(True)

    def on_tab_changed(self, index: int) -> None:
        """Handle tab changes to set up context-specific shortcuts."""
        # Clear existing shortcuts
        if hasattr(self, '_zoom_shortcuts'):
            for shortcut in self._zoom_shortcuts:
                shortcut.setEnabled(False)

        # Set up zoom shortcuts for history tab
        if index == 1:  # History tab index
            self._zoom_shortcuts = []

            # Ctrl++ for zoom in
            zoom_in_shortcut = QAction(self)
            zoom_in_shortcut.setShortcut(QKeySequence("Ctrl++"))
            zoom_in_shortcut.triggered.connect(self.zoom_in_history_image)
            self.addAction(zoom_in_shortcut)
            self._zoom_shortcuts.append(zoom_in_shortcut)

            # Ctrl+- for zoom out
            zoom_out_shortcut = QAction(self)
            zoom_out_shortcut.setShortcut(QKeySequence("Ctrl+-"))
            zoom_out_shortcut.triggered.connect(self.zoom_out_history_image)
            self.addAction(zoom_out_shortcut)
            self._zoom_shortcuts.append(zoom_out_shortcut)

            # Ctrl+0 for reset zoom
            reset_zoom_shortcut = QAction(self)
            reset_zoom_shortcut.setShortcut(QKeySequence("Ctrl+0"))
            reset_zoom_shortcut.triggered.connect(self.reset_history_zoom)
            self.addAction(reset_zoom_shortcut)
            self._zoom_shortcuts.append(reset_zoom_shortcut)

    def on_history_selection_changed(self) -> None:
        """Handle selection changes in the history tree."""
        selected_items = self.history_tree.selectedItems()
        if not selected_items:
            return

        selected_item = selected_items[0]
        file_path_str = selected_item.data(0, Qt.ItemDataRole.UserRole)

        if not file_path_str:
            # This is a directory item, clear previews
            self.history_preview_image.setText("Select a chronogrid file to preview")
            self.history_preview_text.setPlainText("Select an analysis file to view text")
            self.history_summary_text.setPlainText("Select a semantic summary file to view")
            return

        file_path = Path(file_path_str)

        if file_path.suffix.lower() == '.jpg' and 'chronogrid' in file_path.name:
            # Show chronogrid image
            pixmap = QPixmap(str(file_path))
            if not pixmap.isNull():
                # Store original pixmap and reset zoom
                self.history_original_pixmap = pixmap
                self.history_zoom_factor = 1.0
                self._update_history_image_zoom()
            else:
                self.history_preview_image.setText("Unable to load image")
                self.history_original_pixmap = QPixmap()

        elif file_path.suffix.lower() == '.txt' and 'analysis' in file_path.name:
            # Show analysis text
            try:
                analysis_text = file_path.read_text(encoding='utf-8')
                self.history_preview_text.setPlainText(analysis_text)
            except Exception as e:
                self.history_preview_text.setPlainText(f"Error reading file: {e}")

        elif file_path.suffix.lower() == '.json' and 'semantic' in file_path.name:
            # Show semantic summary
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)

                # Format the summary for display
                summary_text = f"Video: {summary_data.get('video_name', 'Unknown')}\n"
                summary_text += f"Analysis Mode: {summary_data.get('analysis_mode', 'Unknown')}\n"
                summary_text += f"Timestamp: {summary_data.get('timestamp', 'Unknown')}\n\n"

                summaries = summary_data.get('summaries', {})
                if summaries:
                    summary_text += "SUMMARIES:\n"
                    for key, value in summaries.items():
                        summary_text += f"{key.title()}: {value}\n\n"

                key_insights = summary_data.get('key_insights', [])
                if key_insights:
                    summary_text += "KEY INSIGHTS:\n"
                    for insight in key_insights:
                        summary_text += f"• {insight}\n"
                    summary_text += "\n"

                actions = summary_data.get('actionable_items', [])
                if actions:
                    summary_text += "ACTIONABLE ITEMS:\n"
                    for action in actions:
                        summary_text += f"• {action}\n"
                    summary_text += "\n"

                narrative = summary_data.get('narrative_summary', '')
                if narrative:
                    summary_text += f"NARRATIVE SUMMARY:\n{narrative}"

                self.history_summary_text.setPlainText(summary_text)

            except Exception as e:
                self.history_summary_text.setPlainText(f"Error reading semantic summary: {e}")

    def zoom_in_history_image(self) -> None:
        """Zoom in on the history preview image."""
        if not self.history_original_pixmap.isNull():
            self.history_zoom_factor *= 1.2
            self._update_history_image_zoom()

    def zoom_out_history_image(self) -> None:
        """Zoom out on the history preview image."""
        if not self.history_original_pixmap.isNull():
            self.history_zoom_factor *= 0.8
            # Don't zoom out too much
            if self.history_zoom_factor < 0.1:
                self.history_zoom_factor = 0.1
            self._update_history_image_zoom()

    def reset_history_zoom(self) -> None:
        """Reset zoom to fit the image in the view."""
        self.history_zoom_factor = 1.0
        self._update_history_image_zoom()

    def _update_history_image_zoom(self) -> None:
        """Update the displayed image with current zoom factor."""
        if self.history_original_pixmap.isNull():
            # Disable zoom controls when no image
            self.zoom_in_button.setEnabled(False)
            self.zoom_out_button.setEnabled(False)
            self.reset_zoom_button.setEnabled(False)
            return

        # Enable zoom controls when image is loaded
        self.zoom_in_button.setEnabled(True)
        self.zoom_out_button.setEnabled(True)
        self.reset_zoom_button.setEnabled(True)

        # Calculate new size
        original_size = self.history_original_pixmap.size()
        new_width = int(original_size.width() * self.history_zoom_factor)
        new_height = int(original_size.height() * self.history_zoom_factor)

        # Scale the pixmap
        scaled_pixmap = self.history_original_pixmap.scaled(
            new_width, new_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.history_preview_image.setPixmap(scaled_pixmap)
        self.history_preview_image.resize(new_width, new_height)

    def show_history_context_menu(self, position: QPoint) -> None:
        """Show context menu for history tree items."""
        selected_items = self.history_tree.selectedItems()
        if not selected_items:
            return

        selected_item = selected_items[0]
        file_path_str = selected_item.data(0, Qt.ItemDataRole.UserRole)

        if file_path_str:
            menu = QMenu(self)
            open_action = QAction("Open with Default App", self)
            open_action.triggered.connect(self.open_file_externally)
            menu.addAction(open_action)

            menu.exec_(self.history_tree.mapToGlobal(position))

    def save_prompt(self) -> None:
        prompt_text = self.prompt_input.toPlainText().strip()
        if not prompt_text:
            QMessageBox.warning(self, "No Prompt", "Prompt area is empty.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Prompt As", "prompt.txt", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                QMessageBox.information(self, "Prompt Saved", f"Prompt saved to {file_path}")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", f"Failed to save prompt: {exc}")

    # Drag-and-drop handlers -------------------------------------------------

    # ...existing code...
    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:  # noqa: N802 (Qt naming)
        if a0 is not None:
            mime = a0.mimeData()
            if mime is not None and hasattr(mime, "hasUrls") and mime.hasUrls():
                a0.acceptProposedAction()

    def dropEvent(self, a0: QDropEvent | None) -> None:  # noqa: N802 (Qt naming)
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime is None or not hasattr(mime, "urls"):
            return
        files: list[Path] = []
        for url in mime.urls():
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(path)
        if files:
            self.video_paths = files
            self.update_file_label()
        else:
            QMessageBox.information(self, "Unsupported Files", "No supported video files were dropped.")

    # File selection ---------------------------------------------------------
    def select_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", VIDEO_FILE_FILTER)
        self.video_paths = [Path(file) for file in files]
        self.update_file_label()

    def update_file_label(self) -> None:
        if not self.video_paths:
            self.file_label.setText("Drop video files here or click 'Select Files'")
            return
        preview = ", ".join(path.name for path in self.video_paths[:3])
        if len(self.video_paths) > 3:
            preview += f" … (+{len(self.video_paths) - 3} more)"
        self.file_label.setText(f"{len(self.video_paths)} file(s) selected: {preview}")

    # Processing -------------------------------------------------------------
    def process_videos(self) -> None:
        if not self.video_paths:
            QMessageBox.warning(self, "No Files Selected", "Please select at least one video to process.")
            return
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.information(self, "Processing in Progress", "Please wait for the current batch to finish.")
            return

        prompt = self.prompt_input.toPlainText().strip() or DEFAULT_PROMPT
        self._encountered_error = False
        self._toggle_processing_ui(active=True)

        self.processing_thread = ProcessingThread(
            [str(path) for path in self.video_paths],
            prompt=prompt,
        )
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.artifact_ready.connect(self.on_processing_result)
        self.processing_thread.failed.connect(self.on_processing_error)
        self.processing_thread.license_required.connect(self.on_license_required)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()

    def update_progress(self, value: int, filename: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{filename} – {value}%")

    def on_processing_result(self, outcome: ProcessingOutcome) -> None:
        pixmap = QPixmap(str(outcome.chronogrid_path))
        if pixmap.isNull():
            self.image_label.setText("Unable to load chronogrid image.")
        else:
            self.image_label.setPixmap(
                pixmap.scaled(520, 520, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

        summary_lines = [
            f"Video: {outcome.video_path}",
            f"Chronogrid: {outcome.chronogrid_path}",
        ]
        if outcome.analysis_path:
            summary_lines.append(f"Analysis file: {outcome.analysis_path}")
        summary_lines.append("")
        summary_lines.append("--- AI Analysis ---")
        summary_lines.append(outcome.analysis_text or "No analysis was generated.")
        self.results_text.setPlainText("\n".join(summary_lines))

    def on_processing_error(self, error_message: str) -> None:
        self._encountered_error = True
        self._toggle_processing_ui(active=False)
        QMessageBox.critical(self, "Processing Error", error_message)

    def on_license_required(self, message: str) -> None:
        self._encountered_error = True
        self._toggle_processing_ui(active=False)
        reply = QMessageBox.question(
            self, "Premium Feature Required",
            f"{message}\n\nWould you like to upgrade to Chronogrid Premium now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.show_upgrade_info()

    def on_processing_finished(self) -> None:
        self._toggle_processing_ui(active=False)
        if not self._encountered_error:
            QMessageBox.information(self, "Done", "All videos were processed successfully.")
        if self.processing_thread:
            self.processing_thread.deleteLater()
            self.processing_thread = None

    def _toggle_processing_ui(self, active: bool) -> None:
        self.process_button.setEnabled(not active)
        self.progress_bar.setVisible(active)
        if not active:
            self.progress_bar.reset()
            self.progress_bar.setFormat("")

    # Qt lifecycle -----------------------------------------------------------
    def closeEvent(self, a0: QCloseEvent | None) -> None:  # noqa: N802
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.requestInterruption()
            self.processing_thread.wait(2000)
        super().closeEvent(a0)


def main() -> int:
    _install_exception_hook()

    app = QApplication(sys.argv)
    splash_gif_path = _resolve_asset_path("outputs", "0001-0300.gif")
    _show_splash_animation(app, splash_gif_path)
    window = ChronogridGUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
