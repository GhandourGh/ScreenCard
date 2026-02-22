import ctypes
import threading
import cv2
import objc
import Quartz
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from hand_display import HandDisplay
from card_renderer import CARD_H
from config import OCR_INTERVAL_FRAMES, SHOW_TURN_BANNER_ALWAYS
from turn_detector import is_my_turn


# --- Region selection (cv2, runs before overlay exists) ---

def select_hand_region(vision_engine):
    print("[SELECT] Draw a rectangle around your hand, then press ENTER or SPACE.")
    frame = vision_engine.capture_screen()
    h, w = frame.shape[:2]
    display = cv2.resize(frame, (w // 2, h // 2))
    roi = cv2.selectROI("Select Your Hand Region", display, showCrosshair=False)
    cv2.destroyWindow("Select Your Hand Region")
    rx, ry, rw, rh = roi
    x1, y1 = rx * 2, ry * 2
    x2, y2 = (rx + rw) * 2, (ry + rh) * 2
    print(f"[SELECT] Hand region set: ({x1}, {y1}) -> ({x2}, {y2})")
    return (x1, y1, x2, y2)


def select_ocr_region(vision_engine):
    print("[SELECT] Draw a box around the 'Your turn!' text, then press ENTER or SPACE.")
    frame = vision_engine.capture_screen()
    h, w = frame.shape[:2]
    display = cv2.resize(frame, (w // 2, h // 2))
    roi = cv2.selectROI("Select 'Your turn!' region (OCR)", display, showCrosshair=False)
    cv2.destroyWindow("Select 'Your turn!' region (OCR)")
    rx, ry, rw, rh = roi
    x1, y1 = rx * 2, ry * 2
    x2, y2 = (rx + rw) * 2, (ry + rh) * 2
    print(f"[SELECT] OCR region set: ({x1}, {y1}) -> ({x2}, {y2})")
    return (x1, y1, x2, y2)


# --- PyQt6 Overlay ---

CARD_STRIP_TOP = 20
CARD_STRIP_H = CARD_H


class ScreenOverlay(QWidget):
    def __init__(self, vision_engine, hand_region, ocr_region):
        super().__init__()
        self.vision = vision_engine
        self.hand_region = hand_region
        self.ocr_region = ocr_region

        self.hand_display = HandDisplay()
        self.prev_labels = []

        # Detection state (shared with inference thread)
        self._lock = threading.Lock()
        self._latest_boxes = []
        self._inferring = False

        # OCR state
        self._frame_count = 0
        self._my_turn = False
        self._last_frame = None

        # Hotkey state
        self._pending_action = None
        self._tap = None

        # Window setup: transparent, click-through, always-on-top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")

        # Size to full screen
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.show()

        print(f"[DEBUG] Overlay geometry: {self.geometry().width()}x{self.geometry().height()}")
        print(f"[DEBUG] DPI ratio: {self.devicePixelRatioF()}")
        print(f"[DEBUG] WA_TranslucentBackground: {self.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)}")

        # Get CGWindowID and pass to vision engine
        self._setup_window_exclusion()

        # Global hotkey listener
        self._start_hotkey_listener()

        # Timer drives the loop
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_screen)
        self._timer.start(30)

        print("[DEBUG] Overlay started — r=re-draw hand, o=re-draw OCR, q=quit")

    def _setup_window_exclusion(self):
        """Get this window's CGWindowID, configure always-on-top, exclude from capture."""
        try:
            win_id = int(self.winId())
            ns_view = objc.objc_object(c_void_p=ctypes.c_void_p(win_id))
            ns_window = ns_view.window()
            window_number = ns_window.windowNumber()
            self.vision.set_exclude_window(window_number)

            # Force transparency
            ns_window.setOpaque_(False)
            ns_window.setBackgroundColor_(
                objc.lookUpClass("NSColor").clearColor()
            )

            # Force always-on-top above ALL windows (including fullscreen apps)
            ns_window.setLevel_(1000)
            ns_window.setCollectionBehavior_(
                (1 << 0)   # NSWindowCollectionBehaviorCanJoinAllSpaces
                | (1 << 8) # NSWindowCollectionBehaviorStationary
            )
            # Prevent hiding when user clicks another app
            ns_window.setHidesOnDeactivate_(False)
            ns_window.setCanHide_(False)
            print(f"[DEBUG] Window: level=1000, hidesOnDeactivate=False")
        except Exception as e:
            print(f"[WARN] Could not configure NSWindow: {e}")

    def _start_hotkey_listener(self):
        """Listen for global keypresses (r, o, q) via Quartz event tap."""
        def callback(proxy, event_type, event, refcon):
            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                # r=15, o=31, q=12 on macOS
                if keycode == 15:  # r
                    self._pending_action = "hand"
                elif keycode == 31:  # o
                    self._pending_action = "ocr"
                elif keycode == 12:  # q
                    self._pending_action = "quit"
            return event

        mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            callback,
            None,
        )
        if self._tap:
            source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetMain(), source, Quartz.kCFRunLoopCommonModes
            )
            print("[DEBUG] Global hotkeys active: r=hand, o=OCR, q=quit")
        else:
            print("[WARN] Could not create event tap — run with Accessibility permissions")

    def _run_inference(self, hand_crop):
        boxes = self.vision.detect_cards(hand_crop)
        with self._lock:
            self._latest_boxes = boxes
            self._inferring = False

    def _update_screen(self):
        # Handle pending hotkey actions
        if self._pending_action == "hand":
            self._pending_action = None
            self._timer.stop()
            self.hide()
            self.hand_region = select_hand_region(self.vision)
            self.prev_labels = []
            self.show()
            self._timer.start(30)
            return
        elif self._pending_action == "ocr":
            self._pending_action = None
            self._timer.stop()
            self.hide()
            self.ocr_region = select_ocr_region(self.vision)
            self.show()
            self._timer.start(30)
            return
        elif self._pending_action == "quit":
            self._pending_action = None
            QApplication.quit()
            return

        frame = self.vision.capture_screen()
        self._last_frame = frame

        x1, y1, x2, y2 = self.hand_region

        if not self._inferring:
            hand = frame[y1:y2, x1:x2].copy()
            self._inferring = True
            threading.Thread(target=self._run_inference, args=(hand,), daemon=True).start()

        with self._lock:
            boxes = self._latest_boxes

        raw_labels = []
        for box in boxes:
            label = self.vision.card_model.names[int(box.cls[0])]
            raw_labels.append(label)

        self.hand_display.update(raw_labels)

        self._frame_count += 1
        if self._frame_count >= OCR_INTERVAL_FRAMES:
            self._frame_count = 0
            ox1, oy1, ox2, oy2 = self.ocr_region
            prev_turn = self._my_turn
            self._my_turn = is_my_turn(frame, roi_xyxy=(ox1, oy1, ox2, oy2))
            if self._my_turn and not prev_turn:
                print("[OCR] Your turn detected — showing banner")

        cur_labels = [f"{self.vision.card_model.names[int(b.cls[0])]} {int(float(b.conf[0]) * 100)}%" for b in boxes]
        if sorted(cur_labels) != sorted(self.prev_labels):
            if cur_labels:
                print(f"[MY HAND] {', '.join(cur_labels)}")
            self.prev_labels = cur_labels

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        scale = self.devicePixelRatioF()

        # --- Hand region outline (yellow) ---
        x1, y1, x2, y2 = self.hand_region
        painter.setPen(QPen(QColor(255, 255, 0), 1))
        painter.drawRect(
            int(x1 / scale), int(y1 / scale),
            int((x2 - x1) / scale), int((y2 - y1) / scale),
        )

        # --- OCR region outline (magenta) ---
        ox1, oy1, ox2, oy2 = self.ocr_region
        painter.setPen(QPen(QColor(255, 0, 255), 1))
        painter.drawRect(
            int(ox1 / scale), int(oy1 / scale),
            int((ox2 - ox1) / scale), int((oy2 - oy1) / scale),
        )

        # --- Card hand strip at top (and YOUR TURN! in same overlay strip) ---
        screen_w = int(self.width())
        self.hand_display.paint(painter, screen_w)

        # YOUR TURN! banner: use same scale as outlines so it appears in correct place on HiDPI
        if self._my_turn or SHOW_TURN_BANNER_ALWAYS:
            self._paint_turn_banner(painter, screen_w)

        painter.end()

    def _paint_turn_banner(self, painter, screen_w):
        bar_w = min(220, screen_w - 40)
        bar_h = CARD_STRIP_H + 4
        by1 = CARD_STRIP_TOP - 2
        bx1 = screen_w - bar_w - 24

        painter.save()
        painter.setOpacity(1.0)
        painter.fillRect(bx1, by1, bar_w, bar_h, QColor(0, 220, 255))
        painter.restore()

        painter.setPen(QPen(QColor(0, 255, 255), 2))
        painter.drawRect(bx1, by1, bar_w, bar_h)

        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        painter.drawText(bx1, by1, bar_w, bar_h, Qt.AlignmentFlag.AlignCenter, "YOUR TURN!")
