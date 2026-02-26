from PyQt6.QtGui import QColor
from card_renderer import render_card_pixmap, CARD_W, CARD_H
from config import CARD_VALUES, CARD_STICKY_FRAMES

FADE_SPEED = 1.0
CARD_GAP = 10
TOP_MARGIN = 20

_SUIT_ORDER = {"S": 0, "H": 1, "C": 2, "D": 3}


def _sort_key(label):
    return (_SUIT_ORDER.get(label[-1], 9), CARD_VALUES.get(label[:-1], 0))


class HandDisplay:
    def __init__(self):
        self.card_opacities = {}
        self.card_miss_count = {}

    def update(self, detected_labels):
        detected = set(detected_labels)

        for label in detected:
            if label not in self.card_opacities:
                self.card_opacities[label] = 0.0
            self.card_opacities[label] = min(1.0, self.card_opacities[label] + FADE_SPEED)
            self.card_miss_count[label] = 0

        to_remove = []
        for label in self.card_opacities:
            if label not in detected:
                self.card_miss_count[label] = self.card_miss_count.get(label, 0) + 1
                if self.card_miss_count[label] >= CARD_STICKY_FRAMES:
                    self.card_opacities[label] -= FADE_SPEED
                    if self.card_opacities[label] <= 0:
                        to_remove.append(label)

        for label in to_remove:
            del self.card_opacities[label]
            self.card_miss_count.pop(label, None)

    def paint(self, painter, frame_width):
        if not self.card_opacities:
            return

        sorted_labels = sorted(self.card_opacities, key=_sort_key)
        n = len(sorted_labels)
        total_w = n * CARD_W + (n - 1) * CARD_GAP
        start_x = max(0, (frame_width - total_w) // 2)
        y = TOP_MARGIN

        pad = 10
        painter.save()
        painter.setOpacity(0.7)
        painter.fillRect(
            start_x - pad, y - pad,
            total_w + pad * 2, CARD_H + pad * 2,
            QColor(20, 20, 30),
        )
        painter.restore()

        for i, label in enumerate(sorted_labels):
            painter.save()
            painter.setOpacity(self.card_opacities[label])
            painter.drawPixmap(start_x + i * (CARD_W + CARD_GAP), y,
                               render_card_pixmap(label))
            painter.restore()
