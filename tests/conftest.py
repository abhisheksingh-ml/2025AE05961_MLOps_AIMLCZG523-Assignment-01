"""Shared test data. conftest.py is read automatically by pytest."""
import numpy as np
import pandas as pd
import pytest

from src import config


def make_raw_df(n=120, seed=0):
    """Create n made-up patients in the same format as the raw UCI file."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.integers(94, 200, n).astype(float),
        "chol": rng.integers(126, 564, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.integers(71, 202, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    # the diagnosis depends loosely on some features, so a model has something to learn
    score = (df["cp"] == 4) * 1.5 + df["exang"] + df["oldpeak"] / 2 + df["ca"] / 2 - 1.5
    df["num"] = np.where(score + rng.normal(0, 0.5, n) > 0, rng.integers(1, 5, n), 0)
    return df[config.ALL_COLUMNS]


@pytest.fixture
def raw_df():
    return make_raw_df()


@pytest.fixture
def raw_file(tmp_path, raw_df):
    """A file in the original UCI format, with two '?' (missing) values."""
    df = raw_df.copy().astype(object)
    df.loc[0, "ca"] = "?"
    df.loc[1, "thal"] = "?"
    path = tmp_path / "processed.cleveland.data"
    df.to_csv(path, header=False, index=False)
    return path


@pytest.fixture
def sample_patient():
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
            "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
