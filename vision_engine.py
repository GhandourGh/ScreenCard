import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO
import Quartz
import config


class VisionEngine:
    def __init__(self):
        self.card_model = YOLO(config.CARD_MODEL_PATH)
        self._exclude_window_id = 0

        print(f"[VisionEngine] Loaded {config.CARD_MODEL_PATH} "
              f"({len(self.card_model.names)} classes)")

        dummy = np.zeros((config.INFERENCE_SIZE, config.INFERENCE_SIZE, 3), dtype=np.uint8)
        self.card_model(dummy, verbose=False)

        self._clahe = (
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            if config.PREPROCESS_CLAHE else None
        )
        self._sharpen_kernel = np.array(
            [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32
        ) if config.PREPROCESS_SHARPEN else None

    def set_exclude_window(self, window_id):
        self._exclude_window_id = window_id

    # ── Screen capture ───────────────────────────────────────────

    def capture_screen(self):
        cg_image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionOnScreenBelowWindow,
            self._exclude_window_id,
            Quartz.kCGWindowImageDefault,
        )
        if cg_image is None:
            cg_image = Quartz.CGWindowListCreateImage(
                Quartz.CGRectNull,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault,
            )
        if cg_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        w = Quartz.CGImageGetWidth(cg_image)
        h = Quartz.CGImageGetHeight(cg_image)
        bpr = Quartz.CGImageGetBytesPerRow(cg_image)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(cg_image))

        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, bpr // 4, 4)
        return cv2.cvtColor(arr[:, :w, :], cv2.COLOR_BGRA2BGR)

    # ── Preprocessing ────────────────────────────────────────────

    def preprocess(self, frame):
        out = frame
        if self._clahe is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        if self._sharpen_kernel is not None:
            out = cv2.filter2D(out, -1, self._sharpen_kernel)
        return out

    # ── Detection ────────────────────────────────────────────────

    def detect_cards(self, frame):
        processed = self.preprocess(frame)
        h, w = processed.shape[:2]

        scale = 1.0
        if config.DETECTION_UPSCALE_ENABLED and h > 0 and h < config.DETECTION_UPSCALE_MIN_HEIGHT:
            scale = min(config.DETECTION_UPSCALE_MIN_HEIGHT / float(h), config.DETECTION_UPSCALE_MAX_FACTOR)
            if scale > 1.01:
                processed = cv2.resize(
                    processed,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_CUBIC,
                )

        ph, pw = processed.shape[:2]
        if pw / max(ph, 1) > config.TILE_ASPECT_THRESHOLD:
            boxes = self._detect_tiled(processed)
        else:
            boxes = self._run_yolo(processed)

        # Map boxes back to original crop coordinates when upscaled.
        if scale > 1.01 and len(boxes) > 0:
            xyxy = boxes.xyxy.clone()
            xyxy /= scale
            return _MergedBoxes(xyxy, boxes.conf.squeeze(1), boxes.cls.squeeze(1))
        return boxes

    def _run_yolo(self, frame):
        results = self.card_model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_NMS_THRESHOLD,
            imgsz=config.INFERENCE_SIZE,
            verbose=False,
        )
        boxes = results[0].boxes
        if len(boxes) == 0:
            return _MergedBoxes(
                torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,))
            )
        return _MergedBoxes(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu())

    def _detect_tiled(self, frame):
        h, w = frame.shape[:2]
        tile_w = min(int(h * config.TILE_ASPECT_TARGET), w)
        stride = max(1, int(tile_w * (1.0 - config.TILE_OVERLAP)))

        tiles, offsets = [], []
        x = 0
        while x < w:
            x_end = min(x + tile_w, w)
            if x_end - x < tile_w * 0.5 and x > 0:
                x = max(0, x_end - tile_w)
                x_end = min(x + tile_w, w)
            tiles.append(frame[:, x:x_end])
            offsets.append(x)
            if x_end >= w:
                break
            x += stride

        results_list = self.card_model(
            tiles,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_NMS_THRESHOLD,
            imgsz=config.INFERENCE_SIZE,
            verbose=False,
        )

        all_xyxy, all_conf, all_cls = [], [], []
        for result, x_off in zip(results_list, offsets):
            boxes = result.boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().clone()
                xyxy[:, 0] += x_off
                xyxy[:, 2] += x_off
                all_xyxy.append(xyxy)
                all_conf.append(boxes.conf.cpu())
                all_cls.append(boxes.cls.cpu())

        if not all_xyxy:
            return _MergedBoxes(
                torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,))
            )

        xyxy = torch.cat(all_xyxy)
        conf = torch.cat(all_conf)
        cls = torch.cat(all_cls)
        keep = batched_nms(xyxy, conf, cls.long(), iou_threshold=0.5)
        return _MergedBoxes(xyxy[keep], conf[keep], cls[keep])


# ── Lightweight box wrappers (match ultralytics Boxes interface) ─

class _MergedBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf.unsqueeze(1) if conf.dim() == 1 else conf
        self.cls = cls.unsqueeze(1) if cls.dim() == 1 else cls

    def __len__(self):
        return self.xyxy.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield _SingleBox(self.xyxy[i], self.conf[i], self.cls[i])


class _SingleBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy.unsqueeze(0) if xyxy.dim() == 1 else xyxy
        self.conf = conf.unsqueeze(0) if conf.dim() == 0 else conf
        self.cls = cls.unsqueeze(0) if cls.dim() == 0 else cls
