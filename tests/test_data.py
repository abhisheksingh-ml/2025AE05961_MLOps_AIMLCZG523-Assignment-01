"""Tests for src/data.py (loading and cleaning)."""
import numpy as np
import pandas as pd

from src import config
from src.data import build_clean_dataset, clean, load_raw, split_xy


def test_question_marks_become_missing_values(raw_file):
    df = load_raw(raw_file)
    assert list(df.columns) == config.ALL_COLUMNS
    assert np.isnan(df.loc[0, "ca"])
    assert np.isnan(df.loc[1, "thal"])


def test_target_is_binary(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert out[config.TARGET].sum() == (raw_df["num"] > 0).sum()      # every num > 0 became 1


def test_duplicates_are_removed(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_zero_cholesterol_becomes_missing(raw_df):
    raw_df.loc[3, "chol"] = 0
    assert clean(raw_df)["chol"].isna().sum() >= 1


def test_clean_dataset_is_saved_as_csv(raw_file, tmp_path):
    out_file = tmp_path / "clean.csv"
    df = build_clean_dataset(raw_file, out_file)
    assert out_file.exists()
    assert df["ca"].isna().sum() == 1 and df["thal"].isna().sum() == 1


def test_split_gives_float_features(raw_df):
    X, y = split_xy(clean(raw_df))
    assert list(X.columns) == config.FEATURES
    assert (X.dtypes == "float64").all()
    assert len(X) == len(y)
