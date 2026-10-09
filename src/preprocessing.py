"""Feature preprocessing: imputation, log-scaling, standardisation, one-hot encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as cfg


def signed_log1p(x):
    """log(1+|x|) with the original sign. Tames the heavy tails of bytes/rate/load features."""
    return np.sign(x) * np.log1p(np.abs(x))


def numeric_columns(X: pd.DataFrame):
    return [c for c in X.columns if c not in cfg.CATEGORICAL]


def correlated_columns(X: pd.DataFrame, threshold: float = 0.95):
    """Numeric columns that are >threshold correlated with an earlier column (fit on TRAIN only)."""
    corr = X[numeric_columns(X)].apply(pd.to_numeric, errors="coerce").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if (upper[c] > threshold).any()]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = numeric_columns(X)
    cat = [c for c in cfg.CATEGORICAL if c in X.columns]

    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    cat_enc = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False
    )
    return ColumnTransformer(
        [("num", num_pipe, num), ("cat", cat_enc, cat)],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
