# Handwritten Digit Recognition using Machine Learning

A handwritten digit classifier built on the **MNIST dataset** with **Scikit-learn**. The project loads and preprocesses the data, trains and compares several classical ML models, tunes the SVM, and reports accuracy, precision, recall, F1 and a confusion matrix.

**Tech Stack:** Python, NumPy, Pandas, Scikit-learn, Matplotlib

## Project Structure

```
handwritten-digit-recognition/
├── src/
│   ├── data.py        # dataset loading, preprocessing, train/test split
│   ├── evaluate.py    # metrics and plots
│   ├── train.py       # train, compare, tune, save the best model
│   └── predict.py     # predict a digit from your own image
├── models/            # saved model (created after training, git-ignored)
├── results/           # metrics, CSV, plots (created after training)
├── data/              # cached dataset (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/USERNAME/handwritten-digit-recognition.git
cd handwritten-digit-recognition
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Train and evaluate all models** (downloads MNIST on first run):

```bash
python -m src.train
```

Useful options:

| Option | Description |
|---|---|
| `--dataset {auto,mnist,digits}` | `auto` tries MNIST and falls back to scikit-learn's small 8x8 digits set if offline |
| `--sample-size N` | Number of samples to use (default 20000 for speed; `0` = full 70,000 MNIST) |
| `--tune` | Run a grid search over SVM hyperparameters (slower, usually the best accuracy) |
| `--test-size 0.2` | Fraction held out for testing |
| `--val-size 0.15` | Fraction of the training data used to compare models |

Example for the best results:

```bash
python -m src.train --dataset mnist --sample-size 0 --tune
```

**Predict from your own image** (a photo or drawing of one digit):

```bash
python -m src.predict path/to/digit.png
```

## Method

1. **Data:** 70,000 28x28 grayscale MNIST images, flattened to 784 features.
2. **Preprocessing:** pixel values scaled to the 0-1 range; stratified train/test split (80/20).
3. **Models compared:** Logistic Regression, K-Nearest Neighbors, Random Forest, SVM (RBF kernel). They are compared on a validation split carved out of the training data, so the test set is never used to choose a model.
4. **Optimization:** optional `GridSearchCV` over SVM `C` and `gamma` with 3-fold cross-validation.
5. **Final evaluation:** the best model is refit on the full training set and scored once on the held-out test set: accuracy, macro precision/recall/F1, per-class report, confusion matrix, and a gallery of misclassified digits.

## Outputs

After training, `results/` contains:

- `sample_digits.png` - example images of each digit
- `model_comparison.csv` / `model_comparison.png` - validation metrics for each model
- `confusion_matrix.png` - best model's test-set confusion matrix
- `misclassified_samples.png` - test examples the model got wrong
- `classification_report.txt` - per-digit precision/recall/F1 on the test set
- `metrics.json` - summary of the run

The trained model is saved to `models/digit_classifier.joblib` (git-ignored because it can be large).

## Results

Run the training command above and paste your numbers here, for example:

| Model | Validation accuracy |
|---|---|
| Logistic Regression | _fill in_ |
| K-Nearest Neighbors | _fill in_ |
| Random Forest | _fill in_ |
| SVM (RBF, tuned) | _fill in_ |

Final test accuracy of the selected model: _fill in_

## License

MIT
