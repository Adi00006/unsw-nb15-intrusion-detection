"""Loading and cleaning the UNSW-NB15 CSV files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, text columns and infinite values (no row removal)."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if cfg.ATTACK_COL in df.columns:
        df[cfg.ATTACK_COL] = df[cfg.ATTACK_COL].astype(str).str.strip()
    for c in cfg.CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df


def load_split(path, dedupe: bool = False) -> pd.DataFrame:
    """Read one CSV. Deduplicate only the TRAINING split (never the test split)."""
    df = clean_frame(pd.read_csv(path))
    if dedupe:
        feats = [c for c in df.columns if c not in cfg.ID_COLS]
        df = df.drop_duplicates(subset=feats).reset_index(drop=True)
    return df


def split_xy(df: pd.DataFrame, task: str):
    """Return (X, y). task='binary' uses `label`; task='multiclass' uses `attack_cat`."""
    target = cfg.LABEL_COL if task == "binary" else cfg.ATTACK_COL
    drop = [c for c in cfg.ID_COLS + [cfg.LABEL_COL, cfg.ATTACK_COL] if c in df.columns]
    return df.drop(columns=drop), df[target]
