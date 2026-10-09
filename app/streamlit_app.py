"""Streamlit demo: upload network-flow records and get attack classifications.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from src import config as cfg  # noqa: E402
from src.predict import load_artifact, predict_frame  # noqa: E402

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Intelligent Network Intrusion Detection")
st.caption("UNSW-NB15 · machine-learning attack classification")


@st.cache_resource
def get_model(path):
    return load_artifact(path)


RUN_LABELS = {
    "binary_real": "Binary detection — real UNSW-NB15 data",
    "multiclass_real": "Attack-family classification — real UNSW-NB15 data",
    "multiclass_real_smote": "Attack-family classification — real data with SMOTE",
}

IMPORTANT_RUNS = ("binary_real", "multiclass_real", "multiclass_real_smote")

MODEL_LABELS = {
    "best_model.joblib": "Best model for this run",
    "DecisionTree.joblib": "Decision Tree",
    "LogisticRegression.joblib": "Logistic Regression",
    "RandomForest.joblib": "Random Forest",
    "XGBoost.joblib": "XGBoost",
}


# ---------------- sidebar ----------------
st.sidebar.header("Model")
folders = (
    [name for name in IMPORTANT_RUNS if (cfg.MODELS_DIR / name).is_dir()]
    if cfg.MODELS_DIR.exists()
    else []
)
if not folders:
    st.error("No real-data models found. Run the real-data training pipeline first.")
    st.stop()
folder = st.sidebar.selectbox(
    "Task and model run",
    folders,
    format_func=lambda name: RUN_LABELS.get(name, name.replace("_", " ").title()),
)
files = sorted(
    path for path in (cfg.MODELS_DIR / folder).glob("*.joblib")
    if path.name != "MLP.joblib"
)
names = [f.name for f in files]
choice = st.sidebar.selectbox(
    "Model algorithm",
    names,
    index=names.index("best_model.joblib") if "best_model.joblib" in names else 0,
    format_func=lambda name: MODEL_LABELS.get(name, name.removesuffix(".joblib")),
)
artifact = get_model(str(cfg.MODELS_DIR / folder / choice))
st.sidebar.write(f"**Algorithm:** {artifact['model_name']}")
st.sidebar.write(f"**Classes:** {', '.join(artifact['classes'])}")

# ---------------- data input ----------------
st.subheader("1 · Provide network-flow records")
up = st.file_uploader("Upload a CSV with UNSW-NB15 columns", type="csv")
df = None
if up is not None:
    df = pd.read_csv(up)
elif cfg.DEFAULT_TEST.exists() and st.button("Use 1,000 random rows from the test set"):
    st.session_state["sample"] = pd.read_csv(cfg.DEFAULT_TEST).sample(1000, random_state=1)
if df is None:
    df = st.session_state.get("sample")
if df is None:
    st.info("Upload a CSV (e.g. a slice of `UNSW_NB15_testing-set.csv`) to begin.")
    st.stop()

result, proba = predict_frame(artifact, df)

# ---------------- results ----------------
st.subheader("2 · Detection summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Flows analysed", f"{len(result):,}")
c2.metric("Flagged as attack", f"{int(result['is_attack'].sum()):,}")
c3.metric("Attack share", f"{result['is_attack'].mean():.1%}")
if "confidence" in result:
    c4.metric("Mean confidence", f"{result['confidence'].mean():.1%}")

truth = None
if artifact["task"] == "multiclass" and cfg.ATTACK_COL in result:
    truth = result[cfg.ATTACK_COL]
elif artifact["task"] == "binary" and cfg.LABEL_COL in result:
    truth = result[cfg.LABEL_COL].map({0: "Normal", 1: "Attack"})
if truth is not None:
    st.success(
        f"Labels found in the file → accuracy **{accuracy_score(truth, result['predicted_class']):.3f}**, "
        f"macro-F1 **{f1_score(truth, result['predicted_class'], average='macro'):.3f}**"
    )

left, right = st.columns([1, 2])
with left:
    st.write("**Predicted class counts**")
    st.bar_chart(result["predicted_class"].value_counts())
with right:
    st.write("**Flagged flows (most confident first)**")
    flagged = result[result["is_attack"]]
    if "confidence" in flagged:
        flagged = flagged.sort_values("confidence", ascending=False)
    show = [c for c in ["predicted_class", "confidence", "proto", "service", "state", "dur", "sbytes", "dbytes"]
            if c in flagged.columns]
    st.dataframe(flagged[show].head(200), use_container_width=True, height=320)

st.download_button("⬇️ Download all predictions (CSV)", result.to_csv(index=False).encode(),
                   "predictions.csv", "text/csv")

# ---------------- single-flow inspector ----------------
if proba is not None:
    st.subheader("3 · Inspect a single flow")
    i = st.number_input("Row number", 0, len(result) - 1, 0)
    row = result.iloc[int(i)]
    st.write(f"Prediction: **{row['predicted_class']}**" + (f" (confidence {row['confidence']:.1%})" if "confidence" in row else ""))
    st.bar_chart(pd.Series(proba[int(i)], index=artifact["classes"], name="probability"))
