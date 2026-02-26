import logging
logging.getLogger().setLevel(logging.CRITICAL)

import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel,
)
from PyQt6.QtCore import Qt

from vision_engine import VisionEngine
from overlay import select_hand_region, ScreenOverlay

logging.getLogger().setLevel(logging.WARNING)


class LauncherWindow(QWidget):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        self.overlay = None

        self.setWindowTitle("Vision Engine")
        self.setFixedSize(280, 240)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Window)

        layout = QVBoxLayout(self)

        self.status = QLabel("Navigate to the game, then click Start.")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedHeight(36)
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn)

        row = QHBoxLayout()
        self.hand_btn = QPushButton("Redraw Hand")
        self.hand_btn.setFixedHeight(32)
        self.hand_btn.setEnabled(False)
        self.hand_btn.clicked.connect(self._on_redraw_hand)
        row.addWidget(self.hand_btn)

        self.ocr_btn = QPushButton("Redraw OCR")
        self.ocr_btn.setFixedHeight(32)
        self.ocr_btn.setEnabled(False)
        self.ocr_btn.clicked.connect(self._on_redraw_ocr)
        row.addWidget(self.ocr_btn)
        layout.addLayout(row)

        self.debug_btn = QPushButton("Show Confidence")
        self.debug_btn.setFixedHeight(32)
        self.debug_btn.setCheckable(True)
        self.debug_btn.setEnabled(False)
        self.debug_btn.clicked.connect(self._on_toggle_debug)
        layout.addWidget(self.debug_btn)

        self.quit_btn = QPushButton("Quit")
        self.quit_btn.setFixedHeight(32)
        self.quit_btn.setStyleSheet("background-color: #cc3333; color: white;")
        self.quit_btn.clicked.connect(self._on_quit)
        layout.addWidget(self.quit_btn)

    def _on_start(self):
        if self.overlay is not None:
            self.overlay.stop()
            self.overlay = None

        self._set_buttons_enabled(False)
        self.status.setText("Draw your hand region...")
        QApplication.processEvents()
        self.hide()

        hand_region = select_hand_region(self.vision)
        cv2.destroyAllWindows()

        # OCR region is optional and can be set later with "Redraw OCR".
        self.overlay = ScreenOverlay(self.vision, hand_region, None)

        self.status.setText("Overlay running. OCR optional (use Redraw OCR).")
        self.start_btn.setText("Restart")
        self.debug_btn.setChecked(False)
        self._set_buttons_enabled(True)
        self.show()

    def _on_redraw_hand(self):
        if self.overlay is None:
            return
        self._set_buttons_enabled(False)
        self.status.setText("Draw your hand region...")
        QApplication.processEvents()
        self.hide()
        self.overlay.redraw_hand()
        self.status.setText("Overlay running.")
        self._set_buttons_enabled(True)
        self.show()

    def _on_redraw_ocr(self):
        if self.overlay is None:
            return
        self._set_buttons_enabled(False)
        self.status.setText("Draw OCR region...")
        QApplication.processEvents()
        self.hide()
        self.overlay.redraw_ocr()
        self.status.setText("Overlay running.")
        self._set_buttons_enabled(True)
        self.show()

    def _on_toggle_debug(self):
        if self.overlay is None:
            return
        on = self.debug_btn.isChecked()
        self.overlay.show_debug = on
        self.debug_btn.setText("Hide Confidence" if on else "Show Confidence")

    def _on_quit(self):
        if self.overlay is not None:
            self.overlay.stop()
        QApplication.quit()

    def _set_buttons_enabled(self, enabled):
        self.start_btn.setEnabled(enabled)
        has_overlay = self.overlay is not None
        self.hand_btn.setEnabled(enabled and has_overlay)
        self.ocr_btn.setEnabled(enabled and has_overlay)
        self.debug_btn.setEnabled(enabled and has_overlay)


def main():
    app = QApplication(sys.argv)
    vision = VisionEngine()
    launcher = LauncherWindow(vision)
    launcher.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
