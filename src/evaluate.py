"""Metrics and plots for evaluating digit classifiers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro-averaged precision, recall and F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def save_classification_report(y_true, y_pred, path: Path) -> str:
    report = classification_report(y_true, y_pred, digits=4)
    Path(path).write_text(report)
    return report


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=np.unique(y_true)).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(df, path: Path, title: str) -> None:
    """Bar chart of accuracy per model. `df` needs 'model' and 'accuracy' columns."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(df["model"], df["accuracy"], color="#4C78A8")
    ax.set_ylim(max(0.0, df["accuracy"].min() - 0.05), 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    for bar, acc in zip(bars, df["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, acc, f"{acc:.4f}", ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_digits(X, y, image_shape, path: Path, per_class: int = 5) -> None:
    """Grid showing a few example images of each digit 0-9."""
    fig, axes = plt.subplots(per_class, 10, figsize=(10, per_class * 1.05))
    for digit in range(10):
        imgs = X[y == digit][:per_class]
        for row in range(per_class):
            ax = axes[row, digit]
            ax.axis("off")
            if row < len(imgs):
                ax.imshow(imgs[row].reshape(image_shape), cmap="gray_r")
            if row == 0:
                ax.set_title(str(digit))
    fig.suptitle("Sample digits")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(X, y_true, y_pred, image_shape, path: Path, n: int = 20) -> int:
    """Gallery of test images the model got wrong. Returns how many were drawn."""
    wrong = np.where(np.asarray(y_true) != np.asarray(y_pred))[0][:n]
    if len(wrong) == 0:
        return 0
    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), wrong):
        ax.imshow(X[idx].reshape(image_shape), cmap="gray_r")
        ax.set_title(f"true {y_true[idx]} / pred {y_pred[idx]}", fontsize=9)
    fig.suptitle("Misclassified test samples")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return len(wrong)
