"""Smoke tests on tiny synthetic data (fast, no real dataset needed)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.data_loader import clean_frame, split_xy  # noqa: E402
from src.predict import load_artifact, predict_frame  # noqa: E402
from src.preprocessing import build_preprocessor  # noqa: E402
from src.train import main as train_main  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--out", str(d),
         "--n-train", "1500", "--n-test", "600"], check=True)
    return d


def test_clean_frame_removes_inf(synth):
    df = clean_frame(pd.read_csv(synth / "UNSW_NB15_training-set.csv"))
    assert not df.select_dtypes("number").isin([float("inf"), float("-inf")]).any().any()


def test_preprocessor_output_is_finite_and_dense(synth):
    df = clean_frame(pd.read_csv(synth / "UNSW_NB15_training-set.csv"))
    X, _ = split_xy(df, "multiclass")
    Z = build_preprocessor(X).fit_transform(X)
    assert Z.shape[0] == len(X)
    assert pd.DataFrame(Z).notna().all().all()


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_train_and_predict(synth, task):
    tag = f"pytest_{task}"
    train_main(["--train", str(synth / "UNSW_NB15_training-set.csv"),
                "--test", str(synth / "UNSW_NB15_testing-set.csv"),
                "--task", task, "--models", "dt", "rf", "--n-estimators", "30", "--tag", tag])
    art = load_artifact(cfg.MODELS_DIR / f"{task}_{tag}" / "best_model.joblib")
    out, proba = predict_frame(art, pd.read_csv(synth / "UNSW_NB15_testing-set.csv").head(50))
    assert len(out) == 50
    assert set(out["predicted_class"]) <= set(art["classes"])
    assert proba.shape == (50, len(art["classes"]))
