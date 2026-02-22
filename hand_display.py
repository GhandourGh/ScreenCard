from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from card_renderer import render_card_pixmap, CARD_W, CARD_H
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
        self.card_miss_count = {}

    def update(self, detected_labels):
        """Update fade state. Call from main thread before paint."""
        detected_set = set(detected_labels)

        for label in detected_set:
            if label not in self.card_opacities:
                self.card_opacities[label] = 0.0
            self.card_opacities[label] = min(1.0, self.card_opacities[label] + FADE_SPEED)
            self.card_miss_count[label] = 0

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

    def paint(self, painter, frame_width):
        """Draw card strip at top-center using QPainter. Call inside paintEvent."""
        if not self.card_opacities:
            return

        sorted_labels = sorted(self.card_opacities.keys(), key=_sort_key)
        total_w = len(sorted_labels) * CARD_W + (len(sorted_labels) - 1) * CARD_GAP
        start_x = max(0, (frame_width - total_w) // 2)
        y = TOP_MARGIN

        # Dark background bar
        bar_pad = 10
        painter.save()
        painter.setOpacity(0.7)
        painter.fillRect(
            start_x - bar_pad, y - bar_pad,
            total_w + bar_pad * 2, CARD_H + bar_pad * 2,
            QColor(20, 20, 30),
        )
        painter.restore()

        # Draw each card with its opacity
        for i, label in enumerate(sorted_labels):
            opacity = self.card_opacities[label]
            pixmap = render_card_pixmap(label)
            cx = start_x + i * (CARD_W + CARD_GAP)

            painter.save()
            painter.setOpacity(opacity)
            painter.drawPixmap(cx, y, pixmap)
            painter.restore()
