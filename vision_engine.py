import cv2
import numpy as np
import mss
from ultralytics import YOLO
import config


class VisionEngine:
    def __init__(self):
        self.card_model = YOLO(config.CARD_MODEL_PATH)
        self.sct = mss.mss()

        print(f"[DEBUG] Card model loaded: {config.CARD_MODEL_PATH}")
        print(f"[DEBUG] Classes ({len(self.card_model.names)}): {list(self.card_model.names.values())}")

        # Warmup
        dummy = np.zeros((config.INFERENCE_SIZE, config.INFERENCE_SIZE, 3), dtype=np.uint8)
        self.card_model(dummy, verbose=False)

    def capture_screen(self):
        monitor = self.sct.monitors[1]
        screenshot = np.array(self.sct.grab(monitor))
        return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    def detect_cards(self, frame):
        results = self.card_model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_NMS_THRESHOLD,
            imgsz=config.INFERENCE_SIZE,
            verbose=False,
        )
        return results[0].boxes
