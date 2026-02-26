# Vision Engine

Real-time macOS overlay for detecting playing cards from browser games and rendering your detected hand as a HUD strip.

## What this build includes (main points)

- overlay updates and smoother hand refresh.
- High-recall detection tuning for screen cards (small card corners and overlap).
- Optional OCR flow:
  - `Start` now asks for **hand region only**.
  - OCR region can be added later from **Redraw OCR**.
- Safer inference pipeline:
  - ROI padding around hand crop.
  - Optional crop upscaling for small in-game cards.
  - Post-filtering + short temporal consensus to reduce noisy labels.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## How to use

- Click **Start** and draw your hand region.
- Overlay starts immediately.
- Optional: click **Redraw OCR** if you want turn-text OCR.
- Use **Show Confidence** to debug raw detections.

## Core files

- `main.py` - launcher and control panel flow.
- `overlay.py` - region selection, threaded update loop, rendering, post-filtering.
- `vision_engine.py` - screen capture, preprocessing, tiled YOLO inference, coordinate mapping.
- `hand_display.py` - stable HUD card strip drawing.
- `config.py` - all runtime tuning knobs.

## Notes on accuracy

This project now includes runtime improvements for recall, but screen-game detection still depends on how similar the game visuals are to the model's training data.

For best results, **fine-tune the model on screenshots from your exact game/site**. 
