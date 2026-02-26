import ctypes
import threading
from collections import Counter, deque

import cv2
import numpy as np
import objc
import Quartz
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics

from hand_display import HandDisplay
from card_renderer import CARD_H
from config import (
    OCR_INTERVAL_FRAMES,
    SHOW_TURN_BANNER_ALWAYS,
    ROI_PAD_FRACTION,
    OVERLAY_REFRESH_MS,
    SMART_HAND_CROP,
    SMART_HAND_MIN_AREA_FRAC,
    SMART_HAND_V_MIN,
    SMART_HAND_S_MAX,
    SMART_HAND_PAD,
    MIN_BOX_WIDTH_FRAC,
    MIN_BOX_HEIGHT_FRAC,
    MAX_BOX_WIDTH_FRAC,
    MAX_BOX_HEIGHT_FRAC,
    CONSENSUS_WINDOW,
    CONSENSUS_MIN_HITS,
    CONSENSUS_MIN_CONF,
)
from turn_detector import is_my_turn

# ── Region selection (cv2, runs before overlay exists) ───────────

def _select_region(vision_engine, title, optional=False):
    while True:
        if optional:
            print(f"[SELECT] Draw a rectangle for '{title}', then press ENTER or SPACE (or press c to skip).")
        else:
            print(f"[SELECT] Draw a rectangle for '{title}', then press ENTER or SPACE.")
        frame = vision_engine.capture_screen()
        h, w = frame.shape[:2]
        disp_w, disp_h = w // 2, h // 2
        display = cv2.resize(frame, (disp_w, disp_h))
        roi = cv2.selectROI(title, display, showCrosshair=False)
        cv2.destroyWindow(title)
        rx, ry, rw, rh = roi
        if rw <= 0 or rh <= 0:
            if optional:
                print(f"[SELECT] Skipped '{title}'.")
                return None
            print("[SELECT] Empty region, please try again.")
            continue
        sx, sy = w / disp_w, h / disp_h
        region = (int(rx * sx), int(ry * sy), int((rx + rw) * sx), int((ry + rh) * sy))
        print(f"[SELECT] {title}: {region[:2]} -> {region[2:]}  [frame {w}x{h}]")
        return region


def select_hand_region(vision_engine):
    return _select_region(vision_engine, "Select Your Hand Region")


def select_ocr_region(vision_engine):
    return _select_region(vision_engine, "Select 'Your turn!' region (OCR)", optional=True)


# ── Confidence debug colours ─────────────────────────────────────

_CONF_COLORS = [
    (0.50, QColor(255, 60, 60)),
    (0.70, QColor(255, 180, 0)),
    (1.01, QColor(60, 255, 60)),
]

def _conf_color(conf):
    for threshold, color in _CONF_COLORS:
        if conf < threshold:
            return color
    return _CONF_COLORS[-1][1]


# ── Overlay ──────────────────────────────────────────────────────

CARD_STRIP_TOP = 20
CARD_STRIP_H = CARD_H


class ScreenOverlay(QWidget):
    def __init__(self, vision_engine, hand_region, ocr_region):
        super().__init__()
        self.vision = vision_engine
        self.hand_region = hand_region
        self.ocr_region = ocr_region
        self.hand_display = HandDisplay()
        self.show_debug = False

        self._lock = threading.Lock()
        self._latest_boxes = []
        self._roi_origin = (0, 0)
        self._last_crop_shape = (0, 0)
        self._inferring = False

        self._frame_count = 0
        self._my_turn = False
        self._ocr_running = False
        self._stopped = False
        self._prev_labels = []
        self._label_history = deque(maxlen=CONSENSUS_WINDOW)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")

        self.setGeometry(QApplication.primaryScreen().geometry())
        self.show()
        self._setup_window_exclusion()

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_screen)
        self._timer.start(OVERLAY_REFRESH_MS)

    # ── macOS window config ──────────────────────────────────────

    def _setup_window_exclusion(self):
        try:
            ns_view = objc.objc_object(c_void_p=ctypes.c_void_p(int(self.winId())))
            ns_window = ns_view.window()
            self.vision.set_exclude_window(ns_window.windowNumber())
            ns_window.setOpaque_(False)
            ns_window.setBackgroundColor_(objc.lookUpClass("NSColor").clearColor())
            ns_window.setLevel_(1000)
            ns_window.setCollectionBehavior_((1 << 0) | (1 << 8))
            ns_window.setHidesOnDeactivate_(False)
            ns_window.setCanHide_(False)
        except Exception as e:
            print(f"[WARN] Could not configure NSWindow: {e}")

    # ── Public controls ──────────────────────────────────────────

    def stop(self):
        self._stopped = True
        self._timer.stop()
        self.hide()

    def redraw_hand(self):
        self._timer.stop()
        self.hide()
        self.hand_region = select_hand_region(self.vision)
        self.hand_display = HandDisplay()
        self._prev_labels = []
        self._label_history.clear()
        cv2.destroyAllWindows()
        self.show()
        self._timer.start(OVERLAY_REFRESH_MS)

    def redraw_ocr(self):
        self._timer.stop()
        self.hide()
        self.ocr_region = select_ocr_region(self.vision)
        cv2.destroyAllWindows()
        self.show()
        self._timer.start(OVERLAY_REFRESH_MS)

    # ── Background workers ───────────────────────────────────────

    def _hand_crop_for_inference(self, frame):
        x1, y1, x2, y2 = self.hand_region
        fh, fw = frame.shape[:2]
        if x2 <= x1 or y2 <= y1:
            return None

        # Base padded crop from user's selected hand region
        pad_x = int((x2 - x1) * ROI_PAD_FRACTION)
        pad_y = int((y2 - y1) * ROI_PAD_FRACTION)
        px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        px2, py2 = min(fw, x2 + pad_x), min(fh, y2 + pad_y)
        hand = frame[py1:py2, px1:px2]
        if hand.size == 0:
            return None

        if not SMART_HAND_CROP:
            return hand.copy(), (px1, py1)

        # Tighten crop to card-like (white, low saturation) pixels.
        hsv = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, SMART_HAND_V_MIN), (179, SMART_HAND_S_MAX, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return hand.copy(), (px1, py1)

        min_area = hand.shape[0] * hand.shape[1] * SMART_HAND_MIN_AREA_FRAC
        keep = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                keep.append(cv2.boundingRect(c))

        if not keep:
            return hand.copy(), (px1, py1)

        ux1 = min(r[0] for r in keep)
        uy1 = min(r[1] for r in keep)
        ux2 = max(r[0] + r[2] for r in keep)
        uy2 = max(r[1] + r[3] for r in keep)

        ux1 = max(0, ux1 - SMART_HAND_PAD)
        uy1 = max(0, uy1 - SMART_HAND_PAD)
        ux2 = min(hand.shape[1], ux2 + SMART_HAND_PAD)
        uy2 = min(hand.shape[0], uy2 + SMART_HAND_PAD)

        if ux2 <= ux1 or uy2 <= uy1:
            return hand.copy(), (px1, py1)

        smart = hand[uy1:uy2, ux1:ux2]
        if smart.size == 0:
            return hand.copy(), (px1, py1)

        return smart.copy(), (px1 + ux1, py1 + uy1)

    def _run_inference(self, hand_crop, roi_origin):
        try:
            boxes = self.vision.detect_cards(hand_crop)
        except Exception:
            boxes = []
        if self._stopped:
            return
        with self._lock:
            self._latest_boxes = boxes
            self._roi_origin = roi_origin
            self._last_crop_shape = hand_crop.shape[:2]
            self._inferring = False

    def _filter_boxes(self, boxes, crop_shape):
        ch, cw = crop_shape
        if ch <= 0 or cw <= 0:
            return []

        min_w = max(8.0, cw * MIN_BOX_WIDTH_FRAC)
        min_h = max(14.0, ch * MIN_BOX_HEIGHT_FRAC)
        max_w = cw * MAX_BOX_WIDTH_FRAC
        max_h = ch * MAX_BOX_HEIGHT_FRAC

        best_by_label = {}
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONSENSUS_MIN_CONF:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            bw = x2 - x1
            bh = y2 - y1
            if bw < min_w or bh < min_h or bw > max_w or bh > max_h:
                continue

            label = self.vision.card_model.names[int(box.cls[0])]
            prev = best_by_label.get(label)
            if prev is None or conf > prev[0]:
                best_by_label[label] = (conf, box)

        return [v[1] for v in best_by_label.values()]

    def _run_ocr(self, frame, roi_xyxy):
        try:
            result = is_my_turn(frame, roi_xyxy)
        except Exception:
            result = False
        if self._stopped:
            return
        if result and not self._my_turn:
            print("[OCR] Your turn detected")
        self._my_turn = result
        self._ocr_running = False

    # ── Main loop ────────────────────────────────────────────────

    def _update_screen(self):
        if self._stopped:
            return

        frame = self.vision.capture_screen()

        # Launch card detection in background thread
        if not self._inferring:
            crop = self._hand_crop_for_inference(frame)
            if crop is not None:
                hand, origin = crop
                self._inferring = True
                threading.Thread(
                    target=self._run_inference,
                    args=(hand, origin),
                    daemon=True,
                ).start()

        # Read latest detections
        with self._lock:
            boxes = self._latest_boxes
            crop_shape = self._last_crop_shape

        filtered_boxes = self._filter_boxes(boxes, crop_shape)
        labels = [self.vision.card_model.names[int(b.cls[0])] for b in filtered_boxes]
        self._label_history.append(labels)

        counts = Counter()
        for frame_labels in self._label_history:
            counts.update(set(frame_labels))
        stable_labels = sorted([label for label, hits in counts.items() if hits >= CONSENSUS_MIN_HITS])
        self.hand_display.update(stable_labels)

        # Launch OCR in background thread
        self._frame_count += 1
        if (
            self.ocr_region is not None
            and self._frame_count >= OCR_INTERVAL_FRAMES
            and not self._ocr_running
        ):
            self._frame_count = 0
            self._ocr_running = True
            threading.Thread(
                target=self._run_ocr,
                args=(frame.copy(), self.ocr_region),
                daemon=True,
            ).start()

        # Log hand changes
        cur = stable_labels
        if cur != self._prev_labels:
            if cur:
                print(f"[HAND] {', '.join(cur)}")
            self._prev_labels = cur

        self.update()

    # ── Painting ─────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.fillRect(self.rect(), Qt.GlobalColor.transparent)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        s = self.devicePixelRatioF()

        # Region outlines
        x1, y1, x2, y2 = self.hand_region
        p.setPen(QPen(QColor(255, 255, 0), 1))
        p.drawRect(int(x1/s), int(y1/s), int((x2-x1)/s), int((y2-y1)/s))

        if self.ocr_region is not None:
            ox1, oy1, ox2, oy2 = self.ocr_region
            p.setPen(QPen(QColor(255, 0, 255), 1))
            p.drawRect(int(ox1/s), int(oy1/s), int((ox2-ox1)/s), int((oy2-oy1)/s))

        # Debug confidence boxes
        if self.show_debug:
            self._paint_debug_boxes(p, s)

        # Card strip
        self.hand_display.paint(p, int(self.width()))

        # Turn banner
        if self._my_turn or SHOW_TURN_BANNER_ALWAYS:
            self._paint_turn_banner(p, int(self.width()))

        p.end()

    def _paint_debug_boxes(self, p, scale):
        with self._lock:
            boxes = self._latest_boxes
            roi_ox, roi_oy = self._roi_origin
            crop_shape = self._last_crop_shape

        boxes = self._filter_boxes(boxes, crop_shape)
        if not boxes:
            return

        font = QFont("Arial", 10, QFont.Weight.Bold)
        p.setFont(font)
        fm = QFontMetrics(font)

        for box in boxes:
            conf = float(box.conf[0])
            label = self.vision.card_model.names[int(box.cls[0])]
            text = f"{label} {int(conf * 100)}%"
            color = _conf_color(conf)

            bx1 = (float(box.xyxy[0][0]) + roi_ox) / scale
            by1 = (float(box.xyxy[0][1]) + roi_oy) / scale
            bx2 = (float(box.xyxy[0][2]) + roi_ox) / scale
            by2 = (float(box.xyxy[0][3]) + roi_oy) / scale

            p.setPen(QPen(color, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(bx1, by1, bx2 - bx1, by2 - by1))

            text_w = fm.horizontalAdvance(text) + 6
            text_h = fm.height() + 2
            tag_y = by1 - text_h if by1 - text_h >= 0 else by1

            p.save()
            p.setOpacity(0.85)
            p.fillRect(QRectF(bx1, tag_y, text_w, text_h), color)
            p.restore()

            p.setPen(QColor(0, 0, 0))
            p.drawText(QRectF(bx1 + 3, tag_y, text_w, text_h),
                       Qt.AlignmentFlag.AlignVCenter, text)

    def _paint_turn_banner(self, p, screen_w):
        bar_w = min(220, screen_w - 40)
        bar_h = CARD_STRIP_H + 4
        by = CARD_STRIP_TOP - 2
        bx = screen_w - bar_w - 24

        p.save()
        p.setOpacity(1.0)
        p.fillRect(bx, by, bar_w, bar_h, QColor(0, 220, 255))
        p.restore()

        p.setPen(QPen(QColor(0, 255, 255), 2))
        p.drawRect(bx, by, bar_w, bar_h)

        p.setPen(QColor(0, 0, 0))
        p.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        p.drawText(bx, by, bar_w, bar_h, Qt.AlignmentFlag.AlignCenter, "YOUR TURN!")
