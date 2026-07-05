"""Shared preprocessing used by both training and serving.

In the source datasets, a zero in `cholesterol` or `resting_bp_s` encodes a
missing measurement, not a true value. Left untreated this leaks a spurious
signal (missing cholesterol correlates strongly with disease). We replace
those zeros with the training-set median and apply the exact same transform
at serve time to avoid train/serve skew.

`oldpeak == 0` is a legitimate clinical value (normal ST depression) and is
deliberately left alone.
"""

import pandas as pd

# Columns where 0 means "missing" rather than a real measurement.
ZERO_AS_MISSING = ["cholesterol", "resting_bp_s"]


def fit_impute_values(df: pd.DataFrame) -> dict:
    """Median of the non-zero values for each missing-coded column (train only)."""
    return {col: float(df.loc[df[col] > 0, col].median()) for col in ZERO_AS_MISSING}


def apply_imputation(df: pd.DataFrame, impute_values: dict) -> pd.DataFrame:
    """Replace zero-coded missing values with the fitted medians."""
    df = df.copy()
    for col, median in impute_values.items():
        df[col] = df[col].mask(df[col] == 0, median)
    return df
