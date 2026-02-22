import cv2
import numpy as np
from ultralytics import YOLO
import Quartz
import config


class VisionEngine:
    def __init__(self):
        self.card_model = YOLO(config.CARD_MODEL_PATH)
        self._exclude_window_id = 0  # 0 = no exclusion

        print(f"[DEBUG] Card model loaded: {config.CARD_MODEL_PATH}")
        print(f"[DEBUG] Classes ({len(self.card_model.names)}): {list(self.card_model.names.values())}")

        # Warmup
        dummy = np.zeros((config.INFERENCE_SIZE, config.INFERENCE_SIZE, 3), dtype=np.uint8)
        self.card_model(dummy, verbose=False)

    def set_exclude_window(self, window_id):
        """Set the CGWindowID to exclude from screen capture (our overlay)."""
        self._exclude_window_id = window_id
        print(f"[DEBUG] Excluding window ID {window_id} from capture")

    def capture_screen(self):
        """Capture primary display using Quartz, excluding our overlay window."""
        cg_image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionOnScreenBelowWindow,
            self._exclude_window_id,
            Quartz.kCGWindowImageDefault,
        )
        if cg_image is None:
            # Fallback: capture everything (no exclusion)
            cg_image = Quartz.CGWindowListCreateImage(
                Quartz.CGRectNull,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault,
            )
        if cg_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        width = Quartz.CGImageGetWidth(cg_image)
        height = Quartz.CGImageGetHeight(cg_image)
        bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
        provider = Quartz.CGImageGetDataProvider(cg_image)
        data = Quartz.CGDataProviderCopyData(provider)

        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, bytes_per_row // 4, 4)
        # Quartz gives BGRA; crop to actual width (bytes_per_row may have padding)
        arr = arr[:, :width, :]
        frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return frame

    def detect_cards(self, frame):
        results = self.card_model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_NMS_THRESHOLD,
            imgsz=config.INFERENCE_SIZE,
            verbose=False,
        )
        return results[0].boxes
