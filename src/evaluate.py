"""Metrics and evaluation plots."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

matplotlib.use("Agg")


def compute_metrics(y_true, y_pred, y_prob):
    """The scores asked for in the assignment. y_prob = predicted probability of disease."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),        # share of correct answers
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),   # predicted sick: how many are
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),         # really sick: how many found
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),          # ranking quality: 0.5 guessing, 1 perfect
    }


def feature_importance(model):
    """Importance of each input column (tree models) or size of its weight (logistic regression)."""
    classifier = model.named_steps["clf"]
    names = model.named_steps["preprocess"].named_steps["columns"].get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        return None
    return pd.Series(values, index=names).sort_values(ascending=False)


def save_plots(model, y_test, y_pred, y_prob, out_dir, name):
    """Save confusion matrix, ROC curve and feature importance; return the file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues",
                                            display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix - {name}")
    paths.append(out_dir / f"{name}_confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)                      # dashed line = random guessing
    ax.set_title(f"ROC curve - {name}")
    paths.append(out_dir / f"{name}_roc_curve.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    importance = feature_importance(model)
    if importance is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        importance.head(15)[::-1].plot.barh(ax=ax, color="#2a9d8f")
        ax.set_title(f"Top features - {name}")
        paths.append(out_dir / f"{name}_feature_importance.png")
        fig.tight_layout()
        fig.savefig(paths[-1], dpi=120)
        plt.close(fig)
    return paths
