import os
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import Qt
import config

CARD_W = 80
CARD_H = 120

_pixmap_cache = {}

_RANK_TO_NAME = {
    "A": "ace", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
    "J": "jack", "Q": "queen", "K": "king",
}
_SUIT_TO_NAME = {"C": "clubs", "D": "diamonds", "H": "hearts", "S": "spades"}
_SUIT_QCOLOR = {
    "C": QColor(46, 125, 80),
    "S": QColor(21, 101, 192),
    "H": QColor(198, 80, 80),
    "D": QColor(230, 101, 0),
}


def _label_to_filename(label):
    rank_name = _RANK_TO_NAME.get(label[:-1], label[:-1])
    suit_name = _SUIT_TO_NAME.get(label[-1], label[-1])
    return f"{rank_name}_of_{suit_name}.png"


def _make_fallback_pixmap(label):
    color = _SUIT_QCOLOR.get(label[-1], QColor(128, 128, 128))
    pm = QPixmap(CARD_W, CARD_H)
    pm.fill(QColor(30, 26, 40))
    p = QPainter(pm)
    p.setPen(color)
    p.drawRect(1, 1, CARD_W - 3, CARD_H - 3)
    p.setFont(QFont("Arial", 14, QFont.Weight.Bold))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, label)
    p.end()
    return pm


def render_card_pixmap(label):
    if label in _pixmap_cache:
        return _pixmap_cache[label]

    path = os.path.join(config.CARDS_DIR, _label_to_filename(label))
    if os.path.isfile(path):
        pm = QPixmap(path)
        if not pm.isNull():
            pm = pm.scaled(
                CARD_W, CARD_H,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            _pixmap_cache[label] = pm
            return pm

    pm = _make_fallback_pixmap(label)
    _pixmap_cache[label] = pm
    return pm
