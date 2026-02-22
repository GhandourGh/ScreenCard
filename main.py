# Suppress noisy hashlib/blake2 errors from dependencies (Python built without full OpenSSL)
import logging
_root = logging.getLogger()
_root.setLevel(logging.CRITICAL)

import cv2
from vision_engine import VisionEngine
from overlay import select_hand_region, select_ocr_region, run_detection_loop

_root.setLevel(logging.WARNING)


def main():
    vision = VisionEngine()
    print("Select your hand region, then the 'Your turn!' OCR box, then detection starts.")
    print("  'r' = re-select hand | 'o' = re-select OCR box | 'q' = quit")

    hand_region = select_hand_region(vision)
    ocr_region = select_ocr_region(vision)

    try:
        run_detection_loop(vision, hand_region, ocr_region)
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
