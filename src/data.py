"""Dataset loading and preprocessing for digit recognition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42

# Share of the image frame a digit occupies, used when preparing your own images:
# MNIST digits sit in a 20x20 box inside the 28x28 frame; the 8x8 digits fill the frame.
DIGIT_FILL = {"mnist": 20 / 28, "digits": 1.0}


@dataclass
class DigitData:
    X: np.ndarray  # (n_samples, n_features), pixel values scaled to [0, 1]
    y: np.ndarray  # (n_samples,), integer labels 0-9
    image_shape: tuple  # (height, width) of one image
    name: str  # "mnist" or "digits"


def _load_mnist() -> DigitData:
    """Download MNIST (70,000 images of 28x28 pixels) from OpenML and cache it in data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        return_X_y=True,
        data_home=str(DATA_DIR),
    )
    # Preprocessing: scale pixel intensities from 0-255 to 0-1.
    return DigitData(X.astype(np.float32) / 255.0, y.astype(int), (28, 28), "mnist")


def _load_digits() -> DigitData:
    """scikit-learn's bundled 8x8 digits set (1,797 images); works offline."""
    d = load_digits()
    # Pixel values here range from 0-16.
    return DigitData((d.data / 16.0).astype(np.float32), d.target.astype(int), (8, 8), "digits")


def load_dataset(name: str = "auto", sample_size: int = 0) -> DigitData:
    """Load a dataset. `name` is "mnist", "digits" or "auto" (MNIST, falling back to digits).

    `sample_size` > 0 draws a class-balanced random subset of that size.
    """
    if name == "mnist":
        data = _load_mnist()
    elif name == "digits":
        data = _load_digits()
    elif name == "auto":
        try:
            data = _load_mnist()
        except Exception as exc:  # no network, OpenML down, etc.
            print(f"[data] Could not load MNIST ({type(exc).__name__}: {exc}).")
            print("[data] Falling back to scikit-learn's built-in 8x8 digits dataset.")
            data = _load_digits()
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    if sample_size and sample_size < len(data.y):
        X, _, y, _ = train_test_split(
            data.X, data.y, train_size=sample_size, stratify=data.y, random_state=RANDOM_STATE
        )
        data = DigitData(X, y, data.image_shape, data.name)
    return data


def split_data(X, y, test_size: float = 0.2):
    """Stratified train/test split so every digit keeps the same share in both sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
