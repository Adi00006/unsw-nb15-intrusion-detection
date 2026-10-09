"""Exploratory data analysis: writes figures and summary tables.

    python -m src.eda --train data/UNSW_NB15_training-set.csv
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .data_loader import load_split


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(cfg.DEFAULT_TRAIN))
    args = ap.parse_args(argv)

    fig_dir, res_dir = cfg.FIG_DIR / "eda", cfg.RESULTS_DIR / "eda"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.train)
    n_inf = int(np.isinf(raw.select_dtypes("number")).sum().sum())
    df = load_split(args.train)
    num_cols = [c for c in df.columns if c not in cfg.CATEGORICAL + [cfg.ATTACK_COL, cfg.LABEL_COL, "id"]]

    # ---- overview tables ----
    overview = pd.DataFrame({
        "rows": [len(df)], "columns": [df.shape[1]], "missing_cells": [int(df.isna().sum().sum())],
        "infinite_cells": [n_inf],
        "duplicate_rows_ignoring_id": [int(df.drop(columns=["id"], errors="ignore").duplicated().sum())],
    })
    overview.to_csv(res_dir / "overview.csv", index=False)
    df[num_cols].describe().T.to_csv(res_dir / "numeric_summary.csv")
    print(overview.to_string(index=False))

    # ---- class distribution ----
    counts = df[cfg.ATTACK_COL].value_counts()
    counts.to_csv(res_dir / "class_counts.csv", header=["count"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts[::-1].plot.barh(ax=ax, color="#2a6f97")
    for i, v in enumerate(counts[::-1]):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set_title("Training-set records per class (note the imbalance)")
    ax.set_xlabel("records")
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # ---- top protocols by class ----
    top_proto = df["proto"].value_counts().head(6).index
    ct = pd.crosstab(df[cfg.ATTACK_COL], df["proto"].where(df["proto"].isin(top_proto), "other"))
    ct = ct.div(ct.sum(axis=1), axis=0)
    ax = ct.plot.bar(stacked=True, figsize=(10, 5), rot=30, colormap="tab20")
    ax.set_ylabel("share of class")
    ax.set_title("Protocol mix by class")
    ax.figure.tight_layout()
    ax.figure.savefig(fig_dir / "protocol_by_class.png", dpi=150)
    plt.close(ax.figure)

    # ---- feature distributions: normal vs attack (log scale) ----
    key = [c for c in ["dur", "sbytes", "dbytes", "rate", "sload", "ct_srv_src"] if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), key):
        for lab, colour, nm in [(0, "#2a6f97", "Normal"), (1, "#c1121f", "Attack")]:
            v = np.log1p(df.loc[df[cfg.LABEL_COL] == lab, c].clip(lower=0).dropna())
            ax.hist(v, bins=50, alpha=0.55, color=colour, label=nm, density=True)
        ax.set_title(f"log(1+{c})")
    axes[0, 0].legend()
    fig.suptitle("Feature distributions: normal vs attack")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_distributions.png", dpi=150)
    plt.close(fig)

    # ---- correlation heatmap of the 25 highest-variance (log) features ----
    logged = np.log1p(df[num_cols].clip(lower=0))
    top = logged.var().sort_values(ascending=False).head(25).index
    corr = logged[top].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top)), top, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top)), top, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation (25 highest-variance features, log scale)")
    fig.tight_layout()
    fig.savefig(fig_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"EDA figures -> {fig_dir}\nEDA tables  -> {res_dir}")


if __name__ == "__main__":
    main()
