"""PyQt GUI for Chronogrid processing with responsive background execution."""

from __future__ import annotations

import logging
import sys
from PyQt5.QtCore import Qt
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap
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
    def handler(exc_type: type, exc_value: BaseException, exc_tb) -> None:
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


class ProcessingThread(QThread):
    """Background worker that runs the Chronogrid pipeline to keep the UI responsive."""

    progress = pyqtSignal(int, str)  # percent complete, current filename
    artifact_ready = pyqtSignal(object)  # ProcessingOutcome
    failed = pyqtSignal(str)

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
        self.resize(1100, 720)
        self.setAcceptDrops(True)

        self.video_paths: list[Path] = []
        self.processing_thread: ProcessingThread | None = None
        self._encountered_error = False

        self.file_label: QLabel
        self.select_button: QPushButton
        self.prompt_input: QTextEdit
        self.image_label: QLabel
        self.results_text: QTextEdit
        self.progress_bar: QProgressBar
        self.process_button: QPushButton

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

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
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
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 (Qt naming)
        if event.mimeData() and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 (Qt naming)
        if not event.mimeData():
            return
        files = []
        for url in event.mimeData().urls():
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
    def closeEvent(self, event) -> None:  # noqa: N802
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.requestInterruption()
            self.processing_thread.wait(2000)
        super().closeEvent(event)


def main() -> int:
    _install_exception_hook()
    app = QApplication(sys.argv)
    window = ChronogridGUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
