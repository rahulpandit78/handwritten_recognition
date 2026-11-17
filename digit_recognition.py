"""
Handwritten Digit Recognition using Machine Learning (MNIST + Scikit-learn)

Pipeline:
  1. Load data (MNIST via OpenML, or sklearn's small 8x8 digits for offline testing)
  2. Inspect + preprocess (quality checks, scaling to [0, 1])
  3. Explore (class distribution, sample images)
  4. Compare several classifiers with cross-validation
  5. Tune the best one with GridSearchCV
  6. Train final model on the full training set, evaluate on the held-out test set
  7. Save plots, metrics and the trained model to ./outputs

Usage:
  python digit_recognition.py                  # full MNIST run
  python digit_recognition.py --subset 5000    # faster model comparison/tuning
  python digit_recognition.py --dataset digits # offline 8x8 digits dataset
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import fetch_openml, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42
OUT = Path("outputs")


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data(name: str):
    """Return X_train, X_test, y_train, y_test (DataFrames/Series), image side, max pixel."""
    if name == "mnist":
        print("Loading MNIST (downloads ~15 MB the first time)...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X = pd.DataFrame(mnist.data)              # still a Pandas DataFrame for the quality checks
        y = pd.Series(mnist.target.astype(int))
        # Standard MNIST split: first 60,000 train, last 10,000 test
        X_train, X_test = X.iloc[:60000], X.iloc[60000:]
        y_train, y_test = y.iloc[:60000], y.iloc[60000:]
        return X_train, X_test, y_train, y_test, 28, 255.0

    digits = load_digits(as_frame=True)
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, 8, 16.0


# --------------------------------------------------------------------------- #
# 2. Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(X_train, X_test, y_train, y_test, max_pixel):
    """Quality checks with Pandas, then scale pixels to [0, 1] as NumPy arrays."""
    print("\n=== Data quality checks ===")
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Missing values (train): {int(X_train.isna().sum().sum())}")
    print(f"Pixel range: {X_train.values.min():.0f} to {X_train.values.max():.0f}")
    print(f"Duplicate rows in train: {int(X_train.duplicated().sum())}")

    X_train = (X_train.values / max_pixel).astype(np.float32)
    X_test = (X_test.values / max_pixel).astype(np.float32)
    return X_train, X_test, y_train.values, y_test.values


# --------------------------------------------------------------------------- #
# 3. Exploration plots
# --------------------------------------------------------------------------- #
def plot_class_distribution(y_train, y_test):
    counts = pd.DataFrame(
        {"train": pd.Series(y_train).value_counts().sort_index(),
         "test": pd.Series(y_test).value_counts().sort_index()}
    )
    ax = counts.plot(kind="bar", figsize=(8, 4), width=0.8)
    ax.set_title("Class distribution")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "class_distribution.png", dpi=150)
    plt.close()


def plot_samples(X, y, side, n=20):
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 0.6, 3))
    for ax, img, label in zip(axes.ravel(), X[:n], y[:n]):
        ax.imshow(img.reshape(side, side), cmap="gray_r")
        ax.set_title(str(label), fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample training images")
    plt.tight_layout()
    plt.savefig(OUT / "sample_digits.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 4. Model comparison
# --------------------------------------------------------------------------- #
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", random_state=RANDOM_STATE),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 25]},
    "SVM (RBF)": {"C": [1, 5, 10], "gamma": ["scale", 0.01]},
}


def compare_models(X, y):
    """5-fold-style comparison using CV on a (sub)sample; returns a results DataFrame."""
    print("\n=== Model comparison (cross-validation) ===")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in get_models().items():
        start = time.time()
        accs = []
        for tr, va in cv.split(X, y):
            m = clone(model).fit(X[tr], y[tr])
            accs.append(accuracy_score(y[va], m.predict(X[va])))
        elapsed = time.time() - start
        rows.append({"model": name, "cv_accuracy": np.mean(accs),
                     "cv_std": np.std(accs), "seconds": elapsed})
        print(f"{name:22s} acc={np.mean(accs):.4f} (+/- {np.std(accs):.4f})  [{elapsed:.1f}s]")
    results = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False)
    results.to_csv(OUT / "model_comparison.csv", index=False)

    ax = results.plot(x="model", y="cv_accuracy", kind="bar", legend=False,
                      figsize=(8, 4), yerr=results["cv_std"])
    ax.set_ylim(max(0, results["cv_accuracy"].min() - 0.05), 1.0)
    ax.set_ylabel("CV accuracy")
    ax.set_title("Model comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close()
    return results


# --------------------------------------------------------------------------- #
# 5. Hyperparameter tuning
# --------------------------------------------------------------------------- #
def tune_best(name, X, y):
    print(f"\n=== Tuning best model: {name} ===")
    search = GridSearchCV(
        get_models()[name], PARAM_GRIDS[name], cv=3, scoring="accuracy", n_jobs=-1, verbose=1
    )
    search.fit(X, y)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


# --------------------------------------------------------------------------- #
# 6. Final evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, X_test, y_test, side):
    print("\n=== Final evaluation on held-out test set ===")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(classification_report(y_test, y_pred, digits=4))
    pd.DataFrame(report).T.to_csv(OUT / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix (accuracy {acc:.2%})")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close()

    wrong = np.where(y_pred != y_test)[0][:20]
    if len(wrong):
        fig, axes = plt.subplots(2, 10, figsize=(14, 3.4))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), wrong):
            ax.imshow(X_test[i].reshape(side, side), cmap="gray_r")
            ax.set_title(f"T:{y_test[i]} P:{y_pred[i]}", fontsize=8)
        fig.suptitle("Misclassified examples (T = true, P = predicted)")
        plt.tight_layout()
        plt.savefig(OUT / "misclassified.png", dpi=150)
        plt.close()
    return acc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Handwritten digit recognition")
    parser.add_argument("--dataset", choices=["mnist", "digits"], default="mnist")
    parser.add_argument("--subset", type=int, default=10000,
                        help="Training samples used for comparison/tuning (final model uses all)")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    X_tr_df, X_te_df, y_tr_s, y_te_s, side, max_pixel = load_data(args.dataset)
    X_train, X_test, y_train, y_test = preprocess(X_tr_df, X_te_df, y_tr_s, y_te_s, max_pixel)

    plot_class_distribution(y_train, y_test)
    plot_samples(X_train, y_train, side)

    # Use a stratified subset so comparison + tuning stay fast
    n = min(args.subset, len(X_train))
    if n < len(X_train):
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=n, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_sub, y_sub = X_train, y_train
    print(f"\nUsing {len(X_sub)} samples for model comparison and tuning.")

    results = compare_models(X_sub, y_sub)
    best_name = results.iloc[0]["model"]
    best_model, best_params, best_cv = tune_best(best_name, X_sub, y_sub)

    print(f"\nTraining final {best_name} on all {len(X_train)} training samples...")
    start = time.time()
    final_model = clone(best_model).fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Done in {train_time:.1f}s")

    test_acc = evaluate(final_model, X_test, y_test, side)

    joblib.dump({"model": final_model, "side": side, "dataset": args.dataset},
                OUT / "digit_model.joblib")
    metrics = {
        "dataset": args.dataset,
        "best_model": best_name,
        "best_params": best_params,
        "tuning_cv_accuracy": round(float(best_cv), 4),
        "test_accuracy": round(float(test_acc), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "final_train_seconds": round(train_time, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nAll results saved in ./{OUT}/  ->  test accuracy = {test_acc:.2%}")


if __name__ == "__main__":
    main()
