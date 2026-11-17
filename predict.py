"""
Predict a digit from your own image using the trained model.

Usage:
  python predict.py path/to/digit.png
Tip: write one digit, dark ink on a light background (or vice versa), photo or scan.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps


def prepare(path, side):
    """Convert an arbitrary image into the same format the model was trained on."""
    img = ImageOps.grayscale(Image.open(path))
    arr = np.asarray(img, dtype=np.float32)
    if arr.mean() > 127:          # light background -> invert (MNIST is white-on-black)
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)

    bbox = img.point(lambda p: 255 if p > 40 else 0).getbbox()
    if bbox:
        img = img.crop(bbox)       # crop tightly around the digit

    # Fit digit into ~70% of the canvas (like MNIST), keep aspect ratio, center it
    target = max(2, int(side * 0.7))
    img.thumbnail((target, target), Image.LANCZOS)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return np.asarray(canvas, dtype=np.float32).reshape(1, -1) / 255.0


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py path/to/digit.png")
    bundle = joblib.load(Path("outputs") / "digit_model.joblib")
    model, side = bundle["model"], bundle["side"]
    x = prepare(sys.argv[1], side)

    pred = int(model.predict(x)[0])
    print(f"Predicted digit: {pred}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        top = np.argsort(proba)[::-1][:3]
        print("Top guesses:", ", ".join(f"{d} ({proba[d]:.1%})" for d in top))


if __name__ == "__main__":
    main()
