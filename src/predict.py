"""Load a saved model and classify new network-flow records.

    python -m src.predict --model models/multiclass/best_model.joblib \
                          --input data/UNSW_NB15_testing-set.csv --output predictions.csv
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from .data_loader import clean_frame


def load_artifact(path):
    return joblib.load(path)


def predict_frame(artifact: dict, df: pd.DataFrame):
    """Return (frame with predictions appended, probability matrix or None)."""
    df = clean_frame(df)
    cols = artifact["feature_columns"]
    X = df.reindex(columns=cols)
    for c in cols:
        if c in cfg.CATEGORICAL:
            X[c] = X[c].fillna("nan").astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    pipe, classes = artifact["pipeline"], artifact["classes"]

    idx = pipe.predict(X)
    out = df.copy()
    out["predicted_class"] = [classes[int(i)] for i in idx]
    out["is_attack"] = out["predicted_class"] != cfg.NORMAL_NAME
    proba = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None
    if proba is not None:
        out["confidence"] = np.round(proba.max(axis=1), 4)
    return out, proba


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(cfg.MODELS_DIR / "multiclass" / "best_model.joblib"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    args = ap.parse_args(argv)

    art = load_artifact(args.model)
    out, _ = predict_frame(art, pd.read_csv(args.input))
    out.to_csv(args.output, index=False)
    print(out["predicted_class"].value_counts().to_string())
    print(f"\nSaved {len(out)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
