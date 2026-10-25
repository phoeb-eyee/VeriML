"""Synthetic dataset generator for VeriML.

This module creates a small, deterministic "loan approval" dataset that
contains a planted *data leak*: a feature that accidentally encodes the answer
(the approval label) for historical records. This lets the auditing engine
demonstrate — reproducibly and safely — how a model can look great while
actually leaning on a misleading pattern.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(
    n_historical: int = 800,
    n_new: int = 400,
    seed: int = 42,
    path: str = "data/dataset.csv",
) -> pd.DataFrame:
    """Generate the synthetic dataset and save it to CSV.

    Parameters
    ----------
    n_historical : int
        Number of "historical" rows, where the leak is present.
    n_new : int
        Number of "new" rows, where the leak has been fixed (random noise).
    seed : int
        Random seed so every run produces the identical dataset.
    path : str
        Where to save the CSV.

    Returns
    -------
    pd.DataFrame
        Columns: ``income``, ``credit_score`` (genuine signal),
        ``leaky_feature`` (the planted leak), ``approved`` (target),
        and ``source`` (metadata: "historical" or "new").
    """
    rng = np.random.default_rng(seed)

    # ---- Genuine features (on a roughly 0-100 scale) ------------------------
    income = rng.normal(loc=55, scale=15, size=n_historical + n_new)
    credit_score = rng.normal(loc=60, scale=20, size=n_historical + n_new)

    # ---- The true, learnable rule ------------------------------------------
    # Higher income and credit score make approval more likely. The small
    # noise term keeps the genuine signal strong but not perfectly learnable.
    log_odds = 0.06 * (income - 55) + 0.05 * (credit_score - 60)
    noise = rng.normal(loc=0, scale=0.4, size=income.size)
    approved = (log_odds + noise > 0).astype(int)

    # ---- The planted leak ---------------------------------------------------
    n = n_historical + n_new
    leaky = np.zeros(n, dtype=int)

    # Historical rows: the leak mostly equals the label (a data bug that copies
    # the answer into a feature column), with ~5% of values flipped.
    leaky[:n_historical] = approved[:n_historical].copy()
    flip_mask = rng.random(n_historical) < 0.05
    leaky[:n_historical] = np.where(
        flip_mask, 1 - leaky[:n_historical], leaky[:n_historical]
    )

    # New rows: the bug is "fixed", so the feature is just random noise and
    # carries no information about the label.
    leaky[n_historical:] = rng.integers(0, 2, size=n_new)

    source = np.array(["historical"] * n_historical + ["new"] * n_new)

    df = pd.DataFrame(
        {
            "income": np.round(income, 1),
            "credit_score": np.round(credit_score, 1),
            "leaky_feature": leaky,
            "approved": approved,
            "source": source,
        }
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
