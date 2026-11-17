# Handwritten Digit Recognition using Machine Learning

Classifies handwritten digits (0-9) from the **MNIST** dataset (70,000 28x28 images) using **Scikit-learn**.

**Tech stack:** Python, NumPy, Pandas, Scikit-learn, Matplotlib

## What it does
1. Loads MNIST (60k train / 10k test, the standard split)
2. Data-quality checks (shape, missing values, pixel range, duplicates) and scales pixels to [0, 1]
3. Exploratory plots: class distribution and sample digits
4. Compares Logistic Regression, KNN, Random Forest and SVM (RBF) with stratified cross-validation
5. Tunes the best model with `GridSearchCV`
6. Trains the final model on all 60k images and evaluates on the untouched 10k test set
7. Saves metrics, classification report, confusion matrix, misclassified examples and the trained model

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python digit_recognition.py                   # full MNIST run (a few minutes)
python digit_recognition.py --subset 5000     # faster comparison/tuning
python digit_recognition.py --dataset digits  # tiny offline 8x8 dataset, for quick testing
```
Results are written to `outputs/`:

| File | Contents |
|---|---|
| `metrics.json` | Best model, best params, CV and test accuracy |
| `model_comparison.csv/.png` | Cross-validated comparison of all models |
| `classification_report.csv` | Per-digit precision / recall / F1 |
| `confusion_matrix.png` | Where the model confuses digits |
| `misclassified.png` | Examples of wrong predictions |
| `class_distribution.png`, `sample_digits.png` | Dataset exploration |
| `digit_model.joblib` | Trained model |

## Try your own handwriting
```bash
python predict.py my_digit.png
```
Photograph or scan one digit on plain paper. The script crops, inverts, centers and resizes it to match MNIST.

## Project structure
```
digit-recognition/
├── digit_recognition.py   # full training + evaluation pipeline
├── predict.py             # predict from your own image
├── requirements.txt
└── README.md
```

## Ideas to extend
- Add a small CNN (Keras/PyTorch) and compare against the SVM
- Add PCA to speed up training
- Wrap `predict.py` in a Streamlit or Flask drawing-canvas demo
