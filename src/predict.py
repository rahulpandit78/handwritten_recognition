"""Predict the digit in an image using the trained model.

Usage:
    python -m src.predict path/to/digit.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from .data import DIGIT_FILL, MODELS_DIR


def preprocess_image(path, image_shape, fill: float = 20 / 28) -> np.ndarray:
    """Turn a photo/drawing of one digit into a (1, n_features) array like the training data.

    Steps: grayscale -> make the digit light on a dark background (as in MNIST)
    -> normalise to 0-1 -> crop to the digit -> resize -> centre on a blank canvas.
    """
    h, w = image_shape
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    # Most drawings are dark ink on light paper; MNIST is the opposite, so invert.
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if np.median(border) > 127:
        arr = 255.0 - arr

    arr -= arr.min()
    if arr.max() == 0:
        raise ValueError("The image looks blank - no digit found.")
    arr /= arr.max()

    ys, xs = np.where(arr > 0.3)
    arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # `fill` is the share of the frame the digit occupies (MNIST: a 20x20 box in 28x28).
    inner = max(1, round(min(h, w) * fill))
    scale = inner / max(arr.shape)
    new_h, new_w = max(1, round(arr.shape[0] * scale)), max(1, round(arr.shape[1] * scale))
    resized = Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.Resampling.LANCZOS
    )

    canvas = np.zeros((h, w), dtype=np.float32)
    top, left = (h - new_h) // 2, (w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = np.asarray(resized, dtype=np.float32) / 255.0
    return canvas.reshape(1, -1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Predict the digit in an image.")
    p.add_argument("image", type=Path, help="path to an image of a single digit")
    p.add_argument("--model", type=Path, default=MODELS_DIR / "digit_classifier.joblib")
    args = p.parse_args(argv)

    if not args.model.exists():
        raise SystemExit(f"Model not found at {args.model}. Train one first: python -m src.train")

    bundle = joblib.load(args.model)
    model = bundle["model"]
    X = preprocess_image(args.image, bundle["image_shape"], DIGIT_FILL.get(bundle["dataset"], 20 / 28))

    print(f"Predicted digit: {int(model.predict(X)[0])}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        for idx in np.argsort(proba)[::-1][:3]:
            print(f"  {model.classes_[idx]}: {proba[idx]:.1%}")


if __name__ == "__main__":
    main()
