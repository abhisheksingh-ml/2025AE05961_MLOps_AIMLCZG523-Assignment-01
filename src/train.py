"""Train, tune and compare models, log everything to MLflow, and save the best model.

Run:  python -m src.train
"""
import json
import shutil
from datetime import datetime, timezone

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from src import config
from src.data import load_clean, split_xy
from src.evaluate import compute_metrics, save_plots
from src.features import build_model_pipeline

matplotlib.use("Agg")

# Models to compare: name -> (classifier, settings to try). "clf__" means "a setting of the classifier step".
MODELS = {
    "logreg": (LogisticRegression(max_iter=2000),
               {"clf__C": [0.01, 0.1, 0.5, 1.0, 5.0], "clf__class_weight": [None, "balanced"]}),
    "rf": (RandomForestClassifier(random_state=config.RANDOM_STATE),
           {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 4, 6],
            "clf__min_samples_leaf": [1, 3, 5]}),
    "gboost": (GradientBoostingClassifier(random_state=config.RANDOM_STATE),
               {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3]}),
}

SCORES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def train_one_model(name, classifier, grid, X_train, y_train, X_test, y_test, cv_folds=config.CV_FOLDS):
    """Tune one model with cross-validation, then score it. Returns a dictionary with everything."""
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    # 1) try every setting in the grid with cross-validation, keep the best
    search = GridSearchCV(build_model_pipeline(classifier), grid, scoring="roc_auc", cv=folds, n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # 2) cross-validated scores of the best model (mean over the folds)
    cv_result = cross_validate(model, X_train, y_train, cv=folds, scoring=SCORES, n_jobs=-1)
    cv_scores = {f"cv_{s}": float(cv_result[f"test_{s}"].mean()) for s in SCORES}

    # 3) scores on the untouched test patients
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_scores = {f"test_{k}": v for k, v in compute_metrics(y_test, y_pred, y_prob).items()}

    print(f"[{name}] best settings: {search.best_params_}")
    print(f"[{name}] CV ROC-AUC {cv_scores['cv_roc_auc']:.3f} | "
          f"test ROC-AUC {test_scores['test_roc_auc']:.3f}")
    return {"name": name, "model": model, "params": search.best_params_, "cv": cv_scores,
            "test": test_scores, "y_pred": y_pred, "y_prob": y_prob,
            "grid_table": pd.DataFrame(search.cv_results_)}


def log_to_mlflow(result, X_train, y_test):
    """Write one MLflow run: settings, scores, plots, tuning table and the model itself."""
    plot_dir = config.REPORTS_DIR / "runs" / result["name"]
    with mlflow.start_run(run_name=result["name"]) as run:
        mlflow.set_tag("model_type", result["name"])
        mlflow.log_params({k.replace("clf__", ""): v for k, v in result["params"].items()})
        mlflow.log_params({"cv_folds": config.CV_FOLDS, "test_size": config.TEST_SIZE})
        mlflow.log_metrics(result["cv"])
        mlflow.log_metrics(result["test"])

        for plot in save_plots(result["model"], y_test, result["y_pred"], result["y_prob"],
                               plot_dir, result["name"]):
            mlflow.log_artifact(str(plot), artifact_path="plots")

        table_file = plot_dir / "grid_search_results.csv"
        result["grid_table"].to_csv(table_file, index=False)
        mlflow.log_artifact(str(table_file), artifact_path="tuning")

        signature = infer_signature(X_train, result["model"].predict_proba(X_train)[:, 1])
        mlflow.sklearn.log_model(result["model"], artifact_path="model", signature=signature,
                                 input_example=X_train.head(3))
        return run.info.run_id


def save_best_model(best, X_train):
    """Save the winning pipeline as joblib (used by the API) and in MLflow format, plus metadata."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], config.MODEL_FILE)

    metadata = {"model_name": best["name"], "best_params": best["params"], **best["cv"], **best["test"],
                "features": config.FEATURES, "mlflow_run_id": best["run_id"],
                "trained_at_utc": datetime.now(timezone.utc).isoformat()}
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str))

    mlflow_dir = config.MODEL_DIR / "mlflow_model"
    shutil.rmtree(mlflow_dir, ignore_errors=True)            # save_model refuses to overwrite a folder
    signature = infer_signature(X_train, best["model"].predict_proba(X_train)[:, 1])
    mlflow.sklearn.save_model(best["model"], str(mlflow_dir), signature=signature,
                              input_example=X_train.head(3))


def main():
    # 1) data: 80 % for training, 20 % for the final test (same share of sick patients in both)
    X, y = split_xy(load_clean())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    print(f"Training patients: {len(X_train)} | test patients: {len(X_test)}")

    # 2) tell MLflow where to store the experiment (folder "mlruns" in the project)
    mlflow.set_tracking_uri(config.MLRUNS_DIR.as_uri())
    mlflow.set_experiment("heart-disease-classification")

    # 3) train every model and log it
    results = []
    for name, (classifier, grid) in MODELS.items():
        result = train_one_model(name, classifier, grid, X_train, y_train, X_test, y_test)
        result["run_id"] = log_to_mlflow(result, X_train, y_test)
        results.append(result)

    # 4) the winner = highest cross-validated ROC-AUC
    best = max(results, key=lambda r: r["cv"]["cv_roc_auc"])
    save_best_model(best, X_train)

    # 5) save a comparison table and the winner's scores (the CI pipeline reads metrics.json)
    summary = pd.DataFrame([{"model": r["name"], **r["cv"], **r["test"]} for r in results]).set_index("model")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    (config.REPORTS_DIR / "metrics.json").write_text(
        json.dumps({"best_model": best["name"], **best["cv"], **best["test"]}, indent=2))
    print("\n===== comparison =====")
    print(summary[["cv_accuracy", "cv_precision", "cv_recall", "cv_roc_auc", "test_roc_auc"]].round(3))
    print("Best model:", best["name"])


if __name__ == "__main__":
    main()
