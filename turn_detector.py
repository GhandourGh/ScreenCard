"""
Detect "Your turn!" on screen using OCR (EasyOCR) on a fixed ROI.
"""
import cv2
import config

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def is_my_turn(frame, roi_xyxy=None):
    """
    Run OCR on a region; return True if "your turn" appears (case-insensitive).
    roi_xyxy: (x1, y1, x2, y2) in pixel coords, or None to use config.OCR_ROI fractions.
    """
    h, w = frame.shape[:2]
    if roi_xyxy is not None:
        x1, y1, x2, y2 = roi_xyxy
    else:
        x, y, rw, rh = config.OCR_ROI
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + rw) * w)
        y2 = int((y + rh) * h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    reader = _get_reader()
    results = reader.readtext(roi)
    combined = " ".join([t[1] for t in results]).lower()
    # Match "your turn", "your turn to lead", "turn to lead", etc.
    return "your turn" in combined or "turn to lead" in combined or "turn to play" in combined
