"""Tests for src/features.py (new features + preprocessing)."""
import numpy as np
import pandas as pd

from src import config
from src.data import clean, split_xy
from src.features import FeatureEngineer, build_preprocessor


def test_feature_engineer_adds_new_columns(raw_df):
    X, _ = split_xy(clean(raw_df))
    out = FeatureEngineer().fit_transform(X)
    for column in config.NEW_FEATURES:
        assert column in out.columns
    row = out.iloc[0]
    assert np.isclose(row["hr_reserve_pct"], row["thalach"] / (220 - row["age"]))
    assert row["high_bp"] == float(row["trestbps"] >= 140)


def test_preprocessor_fills_missing_values(raw_df):
    X, _ = split_xy(clean(raw_df))
    X.loc[0:5, ["ca", "thal", "chol"]] = np.nan
    result = build_preprocessor().fit_transform(X)
    assert not np.isnan(result).any()
    assert result.shape[0] == len(X)


def test_numeric_columns_are_scaled(raw_df):
    X, _ = split_xy(clean(raw_df))
    result = build_preprocessor().fit_transform(X)
    number_of_numeric_columns = len(config.NUMERIC) + 2             # 6 original + 2 new numeric features
    assert np.allclose(result[:, :number_of_numeric_columns].mean(axis=0), 0, atol=1e-6)


def test_unknown_category_does_not_crash(raw_df):
    X, _ = split_xy(clean(raw_df))
    preprocessor = build_preprocessor().fit(X)
    new_patient = X.iloc[[0]].copy()
    new_patient["thal"] = 99                                        # a value never seen in training
    assert preprocessor.transform(pd.DataFrame(new_patient)).shape[0] == 1
