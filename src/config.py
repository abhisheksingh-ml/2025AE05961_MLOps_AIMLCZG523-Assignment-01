"""Project-wide settings: file paths, column names and constants."""
from pathlib import Path

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_FILE = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "model.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLRUNS_DIR = ROOT / "mlruns"

# ---- experiment settings ----
RANDOM_STATE = 42     # fixed seed -> same results every run (reproducibility)
TEST_SIZE = 0.2       # 20 % of patients are kept aside to test the final model
CV_FOLDS = 5          # cross-validation folds

# ---- columns ----
ALL_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
TARGET = "target"                                   # 1 = heart disease, 0 = no heart disease

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]   # numbers -> scaled
BINARY = ["sex", "fbs", "exang"]                                     # 0/1 flags -> kept as they are
CATEGORICAL = ["cp", "restecg", "slope", "thal"]                     # categories -> one-hot encoded
NEW_FEATURES = ["hr_reserve_pct", "high_bp", "age_chol_index"]       # created by us in features.py

FEATURES = NUMERIC + BINARY + CATEGORICAL           # the 13 inputs of the model
