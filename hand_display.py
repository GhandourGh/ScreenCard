import cv2
import numpy as np
from card_renderer import render_card, CARD_W, CARD_H
from config import CARD_VALUES, CARD_STICKY_FRAMES

FADE_SPEED = 0.15
CARD_GAP = 10
TOP_MARGIN = 20


def _sort_key(label):
    """Sort by suit order: Spades, Hearts, Clubs, Diamonds; then by rank."""
    suit = label[-1]
    value = label[:-1]
    suit_order = {"S": 0, "H": 1, "C": 2, "D": 3}
    return (suit_order.get(suit, 9), CARD_VALUES.get(value, 0))


class HandDisplay:
    def __init__(self):
        self.card_opacities = {}
        self.card_miss_count = {}  # frames a card was not detected (for sticky removal)

    def draw_hand(self, frame, detected_labels):
        """Update opacities and composite card row at top-center of frame."""
        detected_set = set(detected_labels)

        # Fade in detected cards; reset miss count when seen
        for label in detected_set:
            if label not in self.card_opacities:
                self.card_opacities[label] = 0.0
            self.card_opacities[label] = min(1.0, self.card_opacities[label] + FADE_SPEED)
            self.card_miss_count[label] = 0

        # Only fade out / remove after card has been missed for STICKY_FRAMES (reduces flicker, keeps briefly missed cards)
        to_remove = []
        for label in self.card_opacities:
            if label not in detected_set:
                self.card_miss_count[label] = self.card_miss_count.get(label, 0) + 1
                if self.card_miss_count[label] >= CARD_STICKY_FRAMES:
                    self.card_opacities[label] -= FADE_SPEED
                    if self.card_opacities[label] <= 0:
                        to_remove.append(label)

        for label in to_remove:
            del self.card_opacities[label]
            self.card_miss_count.pop(label, None)

        if not self.card_opacities:
            return

        # Sort by suit then rank
        sorted_labels = sorted(self.card_opacities.keys(), key=_sort_key)

        # Calculate row position (centered horizontally)
        total_w = len(sorted_labels) * CARD_W + (len(sorted_labels) - 1) * CARD_GAP
        frame_h, frame_w = frame.shape[:2]
        start_x = max(0, (frame_w - total_w) // 2)
        y = TOP_MARGIN

        # Draw semi-transparent dark background bar
        bar_pad = 10
        bar_x1 = max(0, start_x - bar_pad)
        bar_y1 = max(0, y - bar_pad)
        bar_x2 = min(frame_w, start_x + total_w + bar_pad)
        bar_y2 = min(frame_h, y + CARD_H + bar_pad)

        overlay = frame[bar_y1:bar_y2, bar_x1:bar_x2].copy()
        dark = np.zeros_like(overlay)
        cv2.addWeighted(overlay, 0.35, dark, 0.65, 0, overlay)
        frame[bar_y1:bar_y2, bar_x1:bar_x2] = overlay

        # Composite each card
        for i, label in enumerate(sorted_labels):
            opacity = self.card_opacities[label]
            card_img = render_card(label)  # BGRA
            cx = start_x + i * (CARD_W + CARD_GAP)

            if cx < 0 or cx + CARD_W > frame_w or y + CARD_H > frame_h:
                continue

            bgr = card_img[:, :, :3]
            alpha = (card_img[:, :, 3].astype(np.float32) / 255.0) * opacity

            roi = frame[y:y + CARD_H, cx:cx + CARD_W]
            for c in range(3):
                roi[:, :, c] = (alpha * bgr[:, :, c] + (1.0 - alpha) * roi[:, :, c]).astype(np.uint8)
            frame[y:y + CARD_H, cx:cx + CARD_W] = roi
