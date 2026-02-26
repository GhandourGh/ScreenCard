# ── Detection ────────────────────────────────────────────────────
CARD_MODEL_PATH = "models/playingCards.pt"
CONFIDENCE_THRESHOLD = 0.08
INFERENCE_SIZE = 960
IOU_NMS_THRESHOLD = 0.72
CARD_STICKY_FRAMES = 24
OVERLAY_REFRESH_MS = 16

# ── Preprocessing ────────────────────────────────────────────────
PREPROCESS_CLAHE = True
PREPROCESS_SHARPEN = True
ROI_PAD_FRACTION = 0.12
SMART_HAND_CROP = False
SMART_HAND_MIN_AREA_FRAC = 0.004
SMART_HAND_V_MIN = 125
SMART_HAND_S_MAX = 95
SMART_HAND_PAD = 10

# Upscale small hand crops before YOLO to improve small-card recall.
DETECTION_UPSCALE_ENABLED = True
DETECTION_UPSCALE_MIN_HEIGHT = 520
DETECTION_UPSCALE_MAX_FACTOR = 2.5

# ── Runtime post-processing (no retraining required) ─────────────
# Keep only card-corner-like boxes inside the hand crop.
MIN_BOX_WIDTH_FRAC = 0.012
MIN_BOX_HEIGHT_FRAC = 0.04
MAX_BOX_WIDTH_FRAC = 0.98
MAX_BOX_HEIGHT_FRAC = 1.00

# Stabilize the displayed hand across recent frames.
CONSENSUS_WINDOW = 10
CONSENSUS_MIN_HITS = 1
CONSENSUS_MIN_CONF = 0.08

# ── Tiling (for wide hand crops) ────────────────────────────────
TILE_ASPECT_THRESHOLD = 2.0
TILE_ASPECT_TARGET = 1.3
TILE_OVERLAP = 0.65

# ── Card assets ──────────────────────────────────────────────────
CARDS_DIR = "/Users/macos/Desktop/CardCV/assets/cards copy"
CARD_VALUES = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13,
}

# ── OCR ("Your turn!" detection) ────────────────────────────────
OCR_INTERVAL_FRAMES = 4
SHOW_TURN_BANNER_ALWAYS = False
