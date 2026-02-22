# config.py
import os

# Detection: lower conf = more cards found (more false positives); higher = stricter
CONFIDENCE_THRESHOLD = 0.30
CARD_MODEL_PATH = "models/playingCards.pt"
# Larger size can help small/distant cards but is slower (e.g. 640, 896, 1024)
INFERENCE_SIZE = 640
# NMS: lower iou = keep more overlapping boxes (helps fanned cards); default ~0.7
IOU_NMS_THRESHOLD = 0.5
# How many frames a card can be "missed" before we remove it from the overlay (reduces flicker)
CARD_STICKY_FRAMES = 20

# Card screenshot folder (e.g. CardCV "cards copy" with 10_of_clubs.png, ace_of_hearts.png, ...)
CARDS_DIR = "/Users/macos/Desktop/CardCV/assets/cards copy"

# BGR colors for suit boxes on overlay (no cardui)
SUIT_BGR = {
    "C": (80, 125, 46),   # clubs - green
    "S": (192, 101, 21),  # spades - blue
    "H": (80, 80, 198),   # hearts - red
    "D": (0, 101, 230),   # diamonds - orange
}

CARD_VALUES = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13,
}

# OCR "Your turn!" detection — ROI as fraction of frame (x, y, width, height), top-left region
OCR_ROI = (0.0, 0.0, 0.28, 0.12)
# Run OCR every N frames to reduce CPU (lower = banner updates faster)
OCR_INTERVAL_FRAMES = 4
# Set True to always show "YOUR TURN!" banner (for testing alignment/visibility)
SHOW_TURN_BANNER_ALWAYS = False
