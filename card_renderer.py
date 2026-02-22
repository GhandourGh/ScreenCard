"""
Load and resize card images from a folder of screenshots (e.g. CardCV assets/cards copy).
Filenames: ace_of_hearts.png, 10_of_clubs.png, king_of_spades.png, etc.
"""
import os
import cv2
import numpy as np
import config

# Card dimensions for overlay
CARD_W = 80
CARD_H = 120

# Cache loaded and resized cards
_cache = {}

# Map model label (e.g. "AS", "10H") to filename (e.g. "ace_of_spades.png", "10_of_hearts.png")
_RANK_TO_NAME = {
    "A": "ace", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
    "J": "jack", "Q": "queen", "K": "king",
}
_SUIT_TO_NAME = {"C": "clubs", "D": "diamonds", "H": "hearts", "S": "spades"}


def _label_to_filename(label):
    """e.g. 'AS' -> 'ace_of_spades.png', '10H' -> '10_of_hearts.png'."""
    suit = label[-1]
    rank = label[:-1]
    rank_name = _RANK_TO_NAME.get(rank, rank)
    suit_name = _SUIT_TO_NAME.get(suit, suit)
    return f"{rank_name}_of_{suit_name}.png"


def render_card(label):
    """Load card screenshot from CARDS_DIR, resize to CARD_W x CARD_H, return BGRA numpy image."""
    if label in _cache:
        return _cache[label]

    fname = _label_to_filename(label)
    path = os.path.join(config.CARDS_DIR, fname)
    if not os.path.isfile(path):
        # Fallback: black card with label text
        img = np.zeros((CARD_H, CARD_W, 4), dtype=np.uint8)
        img[:, :, :3] = 40
        img[:, :, 3] = 255
        cv2.putText(img, label, (8, CARD_H // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        _cache[label] = img
        return img

    img = cv2.imread(path)
    if img is None:
        img = np.zeros((CARD_H, CARD_W, 4), dtype=np.uint8)
        img[:, :, 3] = 255
        _cache[label] = img
        return img

    img = cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_LANCZOS4)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    _cache[label] = img
    return img
