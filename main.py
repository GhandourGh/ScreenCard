# Suppress noisy hashlib/blake2 errors
import logging
_root = logging.getLogger()
_root.setLevel(logging.CRITICAL)

import sys
import cv2
from PyQt6.QtWidgets import QApplication
from vision_engine import VisionEngine
from overlay import select_hand_region, select_ocr_region, ScreenOverlay

_root.setLevel(logging.WARNING)


def main():
    # QApplication must exist before any QPixmap (card_renderer uses them)
    app = QApplication(sys.argv)

    vision = VisionEngine()

    print("Select your hand region, then the 'Your turn!' OCR box.")
    print("  Escape = quit")

    hand_region = select_hand_region(vision)
    ocr_region = select_ocr_region(vision)
    cv2.destroyAllWindows()

    overlay = ScreenOverlay(vision, hand_region, ocr_region)
    print("Overlay running — game is playable underneath.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
