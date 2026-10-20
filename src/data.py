"""Load the raw file and clean it."""
import numpy as np
import pandas as pd

from src import config


def load_raw(path=config.RAW_FILE):
    """Read the raw UCI file. The file has no header, and '?' means 'missing'."""
    return pd.read_csv(path, header=None, names=config.ALL_COLUMNS, na_values="?")


def clean(raw):
    """Return a cleaned copy of the raw data (missing values are kept as NaN)."""
    df = raw.copy()
    df = df.apply(pd.to_numeric, errors="coerce")      # anything that is not a number becomes NaN
    df = df.drop_duplicates()                          # remove repeated patients
    df.loc[df["chol"] == 0, "chol"] = np.nan           # cholesterol 0 is impossible -> treat as missing
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan   # blood pressure 0 is impossible -> missing
    df = df.dropna(subset=["num"])
    df[config.TARGET] = (df["num"] > 0).astype(int)    # 0 = healthy, 1..4 = disease  ->  0 / 1
    df = df.drop(columns=["num"])
    return df.reset_index(drop=True)


def build_clean_dataset(raw_path=config.RAW_FILE, out_path=config.CLEAN_FILE):
    """Raw file -> cleaned DataFrame, and save it as data/processed/heart_clean.csv."""
    df = clean(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_clean():
    """Read the cleaned CSV (create it first if it does not exist yet)."""
    if not config.CLEAN_FILE.exists():
        build_clean_dataset()
    return pd.read_csv(config.CLEAN_FILE)


def split_xy(df):
    """Split into inputs X (13 features, as floats) and the answer y (0/1)."""
    X = df[config.FEATURES].astype(float)   # floats everywhere -> the MLflow signature and the API agree
    y = df[config.TARGET]
    return X, y
