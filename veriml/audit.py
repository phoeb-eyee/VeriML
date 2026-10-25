"""The auditing experiments.

Each function runs one experiment and returns its results as a plain dict so
the caller (``main.py``) can print and serialize them without knowing the
details. Every number is computed from a real model fit — nothing is hardcoded.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# The columns the model may use. `leaky_feature` is the planted leak.
FEATURE_COLUMNS = ["income", "credit_score", "leaky_feature"]
GENUINE_COLUMNS = ["income", "credit_score"]
TARGET = "approved"
LEAK_COLUMN = "leaky_feature"
SOURCE_COLUMN = "source"


def _fit_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Fit a logistic regression and return its metrics and coefficients.

    Metrics are cast to plain Python floats so the results are JSON-serializable.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    # `report["1"]` is the positive class ("approved" == 1).
    positive = report["1"]
    coefficients = {
        feature: float(coef) for feature, coef in zip(X_train.columns, model.coef_[0])
    }

    return {
        "accuracy": float(report["accuracy"]),
        "precision": float(positive["precision"]),
        "recall": float(positive["recall"]),
        "f1": float(positive["f1-score"]),
        "coefficients": coefficients,
    }


def _split_historical(df: pd.DataFrame, seed: int):
    """Return a stratified train/test split of the historical rows."""
    hist = df[df[SOURCE_COLUMN] == "historical"]
    X = hist[FEATURE_COLUMNS]
    y = hist[TARGET]
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def run_baseline(df: pd.DataFrame, seed: int = 42) -> dict:
    """Train on historical data with ALL features (leak included)."""
    X_train, X_test, y_train, y_test = _split_historical(df, seed)
    results = _fit_and_evaluate(X_train, y_train, X_test, y_test)
    return {
        "name": "baseline",
        "description": "Train on historical data using ALL features (leak included).",
        "features": FEATURE_COLUMNS,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **results,
    }


def run_ablation(df: pd.DataFrame, seed: int = 42) -> dict:
    """Same historical split, but WITHOUT the leaky feature."""
    hist = df[df[SOURCE_COLUMN] == "historical"]
    X = hist[GENUINE_COLUMNS]
    y = hist[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    results = _fit_and_evaluate(X_train, y_train, X_test, y_test)
    return {
        "name": "ablation",
        "description": "Same data, but the leaky feature is removed.",
        "features": GENUINE_COLUMNS,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **results,
    }


def run_generalization(df: pd.DataFrame) -> dict:
    """Train on all historical data, test on NEW data (where the leak is gone)."""
    hist = df[df[SOURCE_COLUMN] == "historical"]
    new = df[df[SOURCE_COLUMN] == "new"]
    X_train = hist[FEATURE_COLUMNS]
    y_train = hist[TARGET]
    X_test = new[FEATURE_COLUMNS]
    y_test = new[TARGET]
    results = _fit_and_evaluate(X_train, y_train, X_test, y_test)
    return {
        "name": "generalization",
        "description": "Train on historical data, test on new data where the leak is gone.",
        "features": FEATURE_COLUMNS,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **results,
    }
