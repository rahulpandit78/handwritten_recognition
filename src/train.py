"""Train, compare, optionally tune, and save handwritten digit classifiers.

Usage:
    python -m src.train
    python -m src.train --dataset mnist --sample-size 0 --tune
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .data import MODELS_DIR, RANDOM_STATE, RESULTS_DIR, load_dataset, split_data
from .evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_misclassified,
    plot_model_comparison,
    plot_sample_digits,
    save_classification_report,
)


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "SVM (RBF)": SVC(kernel="rbf", C=5, gamma="scale"),
    }


def tune_svm(X, y, max_samples: int = 10000) -> tuple[dict, float]:
    """Grid-search the SVM's C and gamma with 3-fold cross-validation.

    Runs on a subset (max_samples) to keep tuning fast; gamma candidates are
    multiples of sklearn's 'scale' heuristic so the grid suits any dataset.
    """
    if len(y) > max_samples:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=RANDOM_STATE
        )
    gamma_scale = 1.0 / (X.shape[1] * float(X.var()))
    grid = {"C": [1, 5, 10], "gamma": [round(gamma_scale * m, 6) for m in (0.5, 1.0, 2.0)]}
    search = GridSearchCV(SVC(kernel="rbf"), grid, cv=3, n_jobs=-1, verbose=1)
    search.fit(X, y)
    print(f"[tune] best params: {search.best_params_}  (cv accuracy {search.best_score_:.4f})")
    return search.best_params_, float(search.best_score_)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate digit classifiers.")
    p.add_argument("--dataset", choices=["auto", "mnist", "digits"], default="auto")
    p.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="number of samples to use (0 = use the whole dataset)",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="fraction held out for testing")
    p.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="fraction of the training data used to compare models",
    )
    p.add_argument("--tune", action="store_true", help="grid-search SVM hyperparameters")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess
    data = load_dataset(args.dataset, args.sample_size)
    print(
        f"[data] {data.name}: {len(data.y)} images of {data.image_shape[0]}x{data.image_shape[1]} "
        f"pixels, {data.X.shape[1]} features"
    )
    print("[data] class distribution:")
    print(pd.Series(data.y).value_counts().sort_index().to_string())
    plot_sample_digits(data.X, data.y, data.image_shape, args.results_dir / "sample_digits.png")

    # 2. Split: test set is touched once at the very end; a validation split picks the model.
    X_train, X_test, y_train, y_test = split_data(data.X, data.y, args.test_size)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"[split] fit {len(y_fit)} | validation {len(y_val)} | test {len(y_test)}")

    # 3. Models (plus a tuned SVM if requested)
    models = build_models()
    tuning = None
    if args.tune:
        print("[tune] searching SVM hyperparameters...")
        best_params, cv_score = tune_svm(X_fit, y_fit)
        models["SVM (tuned)"] = SVC(kernel="rbf", **best_params)
        tuning = {"best_params": best_params, "cv_accuracy": cv_score}

    # 4. Train each model and compare on the validation set
    rows = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_fit, y_fit)
        train_time = time.time() - start
        metrics = compute_metrics(y_val, model.predict(X_val))
        rows.append({"model": name, **metrics, "train_time_s": round(train_time, 1)})
        print(f"[val] {name:<20} accuracy {metrics['accuracy']:.4f}  ({train_time:.1f}s)")

    comparison = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    comparison.to_csv(args.results_dir / "model_comparison.csv", index=False, float_format="%.4f")
    plot_model_comparison(
        comparison, args.results_dir / "model_comparison.png", "Validation accuracy by model"
    )

    # 5. Refit the winner on all training data and score it once on the test set
    best_name = comparison.loc[0, "model"]
    print(f"[final] best model on validation: {best_name}; refitting on full training set")
    final_model = clone(models[best_name]).fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred)
    print(f"[final] test accuracy {test_metrics['accuracy']:.4f}")
    print(save_classification_report(y_test, y_pred, args.results_dir / "classification_report.txt"))

    plot_confusion_matrix(
        y_test,
        y_pred,
        args.results_dir / "confusion_matrix.png",
        f"{best_name} - test accuracy {test_metrics['accuracy']:.4f}",
    )
    plot_misclassified(
        X_test, y_test, y_pred, data.image_shape, args.results_dir / "misclassified_samples.png"
    )

    summary = {
        "dataset": data.name,
        "n_samples": int(len(data.y)),
        "best_model": best_name,
        "test_metrics": test_metrics,
        "validation_comparison": comparison.round(4).to_dict(orient="records"),
        "tuning": tuning,
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    joblib.dump(
        {"model": final_model, "image_shape": data.image_shape, "dataset": data.name},
        args.models_dir / "digit_classifier.joblib",
    )
    print(f"[done] results in {args.results_dir}, model in {args.models_dir}")


if __name__ == "__main__":
    main()
