import cv2
import threading
from hand_display import HandDisplay
from card_renderer import CARD_H
from config import SUIT_BGR, OCR_INTERVAL_FRAMES
from turn_detector import is_my_turn

# Layout: card strip top and height (match hand_display)
CARD_STRIP_TOP = 20
CARD_STRIP_H = CARD_H
LEGEND_PAD = 8
LEGEND_FONT = cv2.FONT_HERSHEY_SIMPLEX
LEGEND_SCALE = 0.5
LEGEND_THICK = 1


def _draw_my_turn_overlay(frame):
    """Draw 'YOUR TURN!' banner in a row below the card strip so it never overlaps cards."""
    h, w = frame.shape[:2]
    bar_h = 44
    bar_w = min(280, w - 40)
    y1 = CARD_STRIP_TOP + CARD_STRIP_H + 10
    y2 = y1 + bar_h
    x1 = (w - bar_w) // 2
    x2 = x1 + bar_w
    if y2 > h - 5:
        return
    overlay = frame[y1:y2, x1:x2].copy()
    yellow = (0, 220, 255)
    cv2.rectangle(overlay, (0, 0), (bar_w, bar_h), yellow, -1)
    cv2.addWeighted(overlay, 0.55, frame[y1:y2, x1:x2], 0.45, 0, frame[y1:y2, x1:x2])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    text = "YOUR TURN!"
    (tw, th), _ = cv2.getTextSize(text, LEGEND_FONT, 0.9, 2)
    tx = x1 + (bar_w - tw) // 2
    ty = y1 + (bar_h + th) // 2
    cv2.putText(frame, text, (tx, ty), LEGEND_FONT, 0.9, (0, 0, 0), 2, cv2.LINE_AA)


def _draw_legend(frame):
    """Draw instructions for regions and keys (top-left, below card strip, so always visible)."""
    h, w = frame.shape[:2]
    lines = [
        "Yellow = hand region | Magenta = Your turn! box",
        "r = re-draw hand | o = re-draw OCR box | q = quit",
    ]
    max_w = 0
    line_heights = []
    for line in lines:
        (lw, lh), _ = cv2.getTextSize(line, LEGEND_FONT, LEGEND_SCALE, LEGEND_THICK)
        line_heights.append(lh)
        max_w = max(max_w, lw)
    line_h = max(line_heights) + 4
    panel_h = len(lines) * line_h + LEGEND_PAD * 2
    panel_w = max_w + LEGEND_PAD * 2
    # Place below card strip, top-left of that row
    py1 = CARD_STRIP_TOP + CARD_STRIP_H + 10
    px1 = LEGEND_PAD
    px2, py2 = px1 + panel_w, py1 + panel_h
    if py2 > h or px2 > w:
        return
    overlay = frame[py1:py2, px1:px2].copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.85, frame[py1:py2, px1:px2], 0.15, 0, frame[py1:py2, px1:px2])
    cv2.rectangle(frame, (px1, py1), (px2, py2), (120, 120, 120), 1)
    for i, line in enumerate(lines):
        y = py1 + LEGEND_PAD + (i + 1) * line_h - 4
        cv2.putText(frame, line, (px1 + LEGEND_PAD, y), LEGEND_FONT, LEGEND_SCALE, (220, 220, 220), LEGEND_THICK, cv2.LINE_AA)


def select_hand_region(vision_engine):
    """Capture a frame and let the user draw a rectangle around their hand."""
    print("[SELECT] Draw a rectangle around your hand, then press ENTER or SPACE.")
    frame = vision_engine.capture_screen()
    h, w = frame.shape[:2]
    display = cv2.resize(frame, (w // 2, h // 2))

    roi = cv2.selectROI("Select Your Hand Region", display, showCrosshair=False)
    cv2.destroyWindow("Select Your Hand Region")

    # roi = (x, y, w, h) in half-res coords — scale back to full res
    rx, ry, rw, rh = roi
    x1, y1 = rx * 2, ry * 2
    x2, y2 = (rx + rw) * 2, (ry + rh) * 2

    print(f"[SELECT] Hand region set: ({x1}, {y1}) → ({x2}, {y2})")
    return (x1, y1, x2, y2)


def select_ocr_region(vision_engine):
    """Capture a frame and let the user draw a rectangle around the 'Your turn!' area."""
    print("[SELECT] Draw a box around the 'Your turn!' text, then press ENTER or SPACE.")
    frame = vision_engine.capture_screen()
    h, w = frame.shape[:2]
    display = cv2.resize(frame, (w // 2, h // 2))

    roi = cv2.selectROI("Select 'Your turn!' region (OCR)", display, showCrosshair=False)
    cv2.destroyWindow("Select 'Your turn!' region (OCR)")

    rx, ry, rw, rh = roi
    x1, y1 = rx * 2, ry * 2
    x2, y2 = (rx + rw) * 2, (ry + rh) * 2

    print(f"[SELECT] OCR region set: ({x1}, {y1}) → ({x2}, {y2})")
    return (x1, y1, x2, y2)


def run_detection_loop(vision_engine, hand_region, ocr_region):
    """Detect cards only within the hand region. Inference runs in background thread."""
    print("[DEBUG] Detection running — 'q' quit | 'r' re-select hand | 'o' re-select OCR box.")

    x1, y1, x2, y2 = hand_region
    ox1, oy1, ox2, oy2 = ocr_region
    prev_labels = []
    hand_display = HandDisplay()

    # OCR "Your turn!" state — run every N frames, smooth display
    frame_count = 0
    my_turn = False

    # Shared state between display loop and inference thread
    lock = threading.Lock()
    latest_boxes = []
    inferring = False

    def run_inference(hand_crop):
        nonlocal latest_boxes, inferring
        boxes = vision_engine.detect_cards(hand_crop)
        with lock:
            latest_boxes = boxes
            inferring = False

    while True:
        frame = vision_engine.capture_screen()

        # Kick off inference in background if not already running
        if not inferring:
            hand = frame[y1:y2, x1:x2].copy()
            inferring = True
            threading.Thread(target=run_inference, args=(hand,), daemon=True).start()

        # Use latest available results (may be from previous frame)
        with lock:
            boxes = latest_boxes

        cur_labels = []
        raw_labels = []

        for box in boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = vision_engine.card_model.names[int(box.cls[0])]
            cur_labels.append(f"{label} {int(conf * 100)}%")
            raw_labels.append(label)

            # Color by suit
            suit = label[-1]
            color = SUIT_BGR.get(suit, (128, 128, 128))

            cv2.rectangle(frame, (x1 + bx1, y1 + by1), (x1 + bx2, y1 + by2), color, 2)
            cv2.putText(frame, f"{label} {int(conf * 100)}%",
                        (x1 + bx1, y1 + by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Hand region outline (yellow); OCR region outline (magenta)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (255, 0, 255), 1)

        # Animated card overlay at top
        hand_display.draw_hand(frame, raw_labels)

        # OCR: detect "Your turn!" every N frames (use drawn box)
        frame_count += 1
        if frame_count >= OCR_INTERVAL_FRAMES:
            frame_count = 0
            my_turn = is_my_turn(frame, roi_xyxy=(ox1, oy1, ox2, oy2))
        if my_turn:
            _draw_my_turn_overlay(frame)

        _draw_legend(frame)

        # Print when hand changes
        if sorted(cur_labels) != sorted(prev_labels):
            if cur_labels:
                print(f"[MY HAND] {', '.join(cur_labels)}")
            prev_labels = cur_labels

        # Display
        h, w = frame.shape[:2]
        display = cv2.resize(frame, (w // 2, h // 2))
        cv2.imshow("Card Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.destroyAllWindows()
            hand_region = select_hand_region(vision_engine)
            x1, y1, x2, y2 = hand_region
            prev_labels = []
        elif key == ord('o'):
            cv2.destroyAllWindows()
            ocr_region = select_ocr_region(vision_engine)
            ox1, oy1, ox2, oy2 = ocr_region

    cv2.destroyAllWindows()
