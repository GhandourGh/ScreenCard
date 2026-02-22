"""
Load card images from CARDS_DIR as QPixmap for QPainter overlay rendering.
Falls back to drawing a simple card with text if PNG is missing.
"""
import os
import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import Qt
import config

# Card dimensions for overlay (logical pixels)
CARD_W = 80
CARD_H = 120

# Cache: label → QPixmap
_pixmap_cache = {}

_RANK_TO_NAME = {
    "A": "ace", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
    "J": "jack", "Q": "queen", "K": "king",
}
_SUIT_TO_NAME = {"C": "clubs", "D": "diamonds", "H": "hearts", "S": "spades"}

# Suit QColors (matching config.SUIT_BGR but as RGB for Qt)
_SUIT_QCOLOR = {
    "C": QColor(46, 125, 80),
    "S": QColor(21, 101, 192),
    "H": QColor(198, 80, 80),
    "D": QColor(230, 101, 0),
}


def _label_to_filename(label):
    suit = label[-1]
    rank = label[:-1]
    rank_name = _RANK_TO_NAME.get(rank, rank)
    suit_name = _SUIT_TO_NAME.get(suit, suit)
    return f"{rank_name}_of_{suit_name}.png"


def _make_fallback_pixmap(label):
    """Draw a simple card when PNG is missing."""
    suit = label[-1]
    color = _SUIT_QCOLOR.get(suit, QColor(128, 128, 128))

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
    """Return a QPixmap of the card. Loads from CARDS_DIR, cached."""
    if label in _pixmap_cache:
        return _pixmap_cache[label]

    fname = _label_to_filename(label)
    path = os.path.join(config.CARDS_DIR, fname)

    if os.path.isfile(path):
        pm = QPixmap(path)
        if not pm.isNull():
            pm = pm.scaled(CARD_W, CARD_H, Qt.AspectRatioMode.IgnoreAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            _pixmap_cache[label] = pm
            return pm

    pm = _make_fallback_pixmap(label)
    _pixmap_cache[label] = pm
    return pm


# Keep numpy render_card for backwards compatibility (used nowhere now, but safe to keep)
_np_cache = {}


def render_card(label):
    """Load card as BGRA numpy image (legacy)."""
    if label in _np_cache:
        return _np_cache[label]

    fname = _label_to_filename(label)
    path = os.path.join(config.CARDS_DIR, fname)
    if not os.path.isfile(path):
        img = np.zeros((CARD_H, CARD_W, 4), dtype=np.uint8)
        img[:, :, :3] = 40
        img[:, :, 3] = 255
        cv2.putText(img, label, (8, CARD_H // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        _np_cache[label] = img
        return img

    img = cv2.imread(path)
    if img is None:
        img = np.zeros((CARD_H, CARD_W, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        _np_cache[label] = img
        return img

    img = cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_LANCZOS4)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    _np_cache[label] = img
    return img
