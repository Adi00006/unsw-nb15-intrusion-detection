"""Metrics and plots for model evaluation."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config as cfg


def compute_metrics(y_true, y_pred, y_proba, class_names):
    """Overall + IDS-specific metrics.

    attack_detection_rate = share of real attacks flagged as ANY attack (recall of 'attack').
    false_alarm_rate      = share of normal flows wrongly flagged as an attack.
    """
    n = len(class_names)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    normal_idx = class_names.index(cfg.NORMAL_NAME)
    t_att, p_att = np.asarray(y_true) != normal_idx, np.asarray(y_pred) != normal_idx
    tp, fn = (t_att & p_att).sum(), (t_att & ~p_att).sum()
    fp, tn = (~t_att & p_att).sum(), (~t_att & ~p_att).sum()
    m["attack_detection_rate"] = tp / max(tp + fn, 1)
    m["false_alarm_rate"] = fp / max(fp + tn, 1)

    m["roc_auc"] = np.nan
    if y_proba is not None:
        try:
            if n == 2:
                m["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                p = y_proba / y_proba.sum(axis=1, keepdims=True)
                m["roc_auc"] = roc_auc_score(
                    y_true, p, multi_class="ovr", average="macro", labels=list(range(n))
                )
        except ValueError:
            pass
    return m


def text_report(y_true, y_pred, class_names):
    return classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, digits=4, zero_division=0,
    )


def per_class_f1(y_true, y_pred, class_names):
    return f1_score(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )


def plot_confusion(y_true, y_pred, class_names, title, path):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.9), max(4.5, len(labels) * 0.8)))
    ConfusionMatrixDisplay(np.nan_to_num(cm), display_labels=class_names).plot(
        ax=ax, xticks_rotation=45, values_format=".2f", colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, path):
    cols = ["accuracy", "macro_f1", "attack_detection_rate"]
    ax = summary.set_index("model")[cols].plot.bar(figsize=(9, 5), rot=0, width=0.8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("score")
    ax.set_title("Model comparison on the test set")
    ax.legend(ncol=len(cols), loc="upper center", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def plot_per_class_f1(df: pd.DataFrame, path):
    ax = df.plot.bar(figsize=(11, 5), rot=30, width=0.85)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 by model")
    ax.legend(ncol=len(df.columns), loc="upper center", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)


def plot_roc_binary(curves: dict, path):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (Normal vs Attack)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def roc_points(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    return fpr, tpr, roc_auc_score(y_true, y_proba[:, 1])


def plot_feature_importance(importances: pd.Series, title, path, top=20):
    top_s = importances.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    top_s.plot.barh(ax=ax, color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
