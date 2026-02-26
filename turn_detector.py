_reader = None

_TURN_PHRASES = ("your turn", "turn to lead", "turn to play")


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def is_my_turn(frame, roi_xyxy):
    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    results = _get_reader().readtext(roi)
    text = " ".join(t[1] for t in results).lower()
    return any(phrase in text for phrase in _TURN_PHRASES)
