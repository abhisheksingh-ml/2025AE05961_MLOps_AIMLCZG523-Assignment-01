"""Tests for model training and prediction."""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data import clean, split_xy
from src.evaluate import compute_metrics
from src.features import build_model_pipeline
from src.predict import predict_records
from src.train import train_one_model


def make_split(raw_df):
    X, y = split_xy(clean(raw_df))
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


def test_pipeline_returns_probabilities(raw_df):
    X_train, X_test, y_train, _ = make_split(raw_df)
    model = build_model_pipeline(LogisticRegression(max_iter=500)).fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_metrics_are_between_0_and_1():
    scores = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    assert set(scores) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in scores.values())


def test_training_function_reports_cv_and_test_scores(raw_df):
    X_train, X_test, y_train, y_test = make_split(raw_df)
    result = train_one_model("logreg", LogisticRegression(max_iter=500), {"clf__C": [0.1, 1.0]},
                             X_train, y_train, X_test, y_test, cv_folds=3)
    assert "cv_roc_auc" in result["cv"]
    assert "test_roc_auc" in result["test"]
    assert result["cv"]["cv_roc_auc"] > 0.5            # better than guessing


def test_saved_model_gives_same_answers(raw_df, tmp_path, sample_patient):
    X_train, X_test, y_train, _ = make_split(raw_df)
    model = build_model_pipeline(LogisticRegression(max_iter=500)).fit(X_train, y_train)
    joblib.dump(model, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    assert (loaded.predict(X_test) == model.predict(X_test)).all()
    result = predict_records([sample_patient], loaded)[0]
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1.0
