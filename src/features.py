"""Feature engineering + preprocessing pipeline (shared by training and the API)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds three domain-inspired columns to the patient data.

    hr_reserve_pct : highest heart rate reached / expected maximum (220 - age)
    high_bp        : 1 if resting blood pressure >= 140 mmHg, else 0
    age_chol_index : age * cholesterol / 1000 (a simple 'combined load' number)
    """

    def fit(self, X, y=None):
        return self                      # nothing to learn from the data

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["hr_reserve_pct"] = X["thalach"] / (220 - X["age"]).replace(0, np.nan)
        X["high_bp"] = (X["trestbps"] >= 140).astype(float)
        X.loc[X["trestbps"].isna(), "high_bp"] = np.nan      # unknown BP -> unknown flag (filled later)
        X["age_chol_index"] = X["age"] * X["chol"] / 1000.0
        return X


def build_preprocessor():
    """Everything that happens to the data before the model sees it."""
    numeric_cols = config.NUMERIC + ["hr_reserve_pct", "age_chol_index"]
    binary_cols = config.BINARY + ["high_bp"]

    numbers = Pipeline([("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    flags = Pipeline([("fill", SimpleImputer(strategy="most_frequent"))])
    categories = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    per_column = ColumnTransformer([
        ("num", numbers, numeric_cols),
        ("bin", flags, binary_cols),
        ("cat", categories, config.CATEGORICAL),
    ])
    return Pipeline([("engineer", FeatureEngineer()), ("columns", per_column)])


def build_model_pipeline(classifier):
    """Full pipeline: preprocessing -> classifier. This is the object we train and save."""
    return Pipeline([("preprocess", build_preprocessor()), ("clf", classifier)])
