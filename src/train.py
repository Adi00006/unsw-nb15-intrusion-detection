"""Train, evaluate and save intrusion-detection models.

Examples (run from the project root):
    python -m src.train --task multiclass
    python -m src.train --task binary --models rf xgb
    python -m src.train --task multiclass --smote --tune --n-iter 10
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from . import config as cfg
from . import evaluate as ev
from .data_loader import load_split, split_xy
from .models import MODEL_NAMES, PARAM_GRIDS, make_model
from .preprocessing import build_preprocessor, correlated_columns


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--train", default=str(cfg.DEFAULT_TRAIN))
    ap.add_argument("--test", default=str(cfg.DEFAULT_TEST))
    ap.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    ap.add_argument("--models", nargs="+", default=["lr", "dt", "rf", "xgb", "mlp"],
                    choices=list(MODEL_NAMES))
    ap.add_argument("--n-estimators", type=int, default=300, help="trees for RF / XGBoost")
    ap.add_argument("--drop-correlated", action="store_true",
                    help="drop numeric features with |corr| > 0.95 (fit on train)")
    ap.add_argument("--dedupe", action="store_true", help="drop duplicate rows from TRAIN")
    ap.add_argument("--smote", action="store_true", help="oversample rare classes in TRAIN")
    ap.add_argument("--smote-target", type=int, default=10000,
                    help="raise every class below this count up to this count")
    ap.add_argument("--tune", action="store_true", help="RandomizedSearchCV before final fit")
    ap.add_argument("--n-iter", type=int, default=8, help="random-search iterations")
    ap.add_argument("--tune-rows", type=int, default=40000, help="subsample size for tuning")
    ap.add_argument("--tag", default="", help="optional suffix for output folders")
    return ap.parse_args(argv)


def oversample(Xt, y, target, seed):
    """SMOTE rare classes up to `target` rows (never touches classes already above it)."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:  # pragma: no cover
        raise SystemExit("imbalanced-learn is required for --smote: pip install imbalanced-learn") from e
    counts = pd.Series(y).value_counts()
    strat = {int(c): int(target) for c, n in counts.items() if n < target and n > 1}
    if not strat:
        return Xt, y
    k = max(1, min(5, int(counts[list(strat)].min()) - 1))
    return SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=seed).fit_resample(Xt, y)


def fit_kwargs(key, y):
    if key == "xgb":  # XGBoost has no class_weight argument -> use sample weights
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def tune(key, X, y, args):
    grid = PARAM_GRIDS[key]
    pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", make_model(key, len(np.unique(y)), args.n_estimators))])
    if len(X) > args.tune_rows:
        X, _, y, _ = train_test_split(X, y, train_size=args.tune_rows, stratify=y,
                                      random_state=cfg.RANDOM_STATE)
    search = RandomizedSearchCV(pipe, grid, n_iter=args.n_iter, cv=3, scoring="f1_macro",
                                n_jobs=1, random_state=cfg.RANDOM_STATE, error_score="raise")
    search.fit(X, y, **fit_kwargs(key, y))
    print(f"  best CV macro-F1={search.best_score_:.4f}  params={search.best_params_}")
    return {k.replace("clf__", ""): v for k, v in search.best_params_.items()}


def main(argv=None):
    args = parse_args(argv)
    suffix = f"_{args.tag}" if args.tag else ""
    task_dir = cfg.MODELS_DIR / f"{args.task}{suffix}"
    fig_dir = cfg.FIG_DIR / f"{args.task}{suffix}"
    res_dir = cfg.RESULTS_DIR / f"{args.task}{suffix}"
    for d in (task_dir, fig_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    train_df = load_split(args.train, dedupe=args.dedupe)
    test_df = load_split(args.test)
    X_train, y_train = split_xy(train_df, args.task)
    X_test, y_test = split_xy(test_df, args.task)

    dropped = []
    if args.drop_correlated:
        dropped = correlated_columns(X_train)
        X_train, X_test = X_train.drop(columns=dropped), X_test.drop(columns=dropped)
        print(f"Dropped {len(dropped)} highly correlated columns: {dropped}")
    print(f"Train {X_train.shape} | Test {X_test.shape} | task={args.task}")

    le = LabelEncoder().fit(pd.concat([y_train, y_test]).astype(str))
    ytr, yte = le.transform(y_train.astype(str)), le.transform(y_test.astype(str))
    class_names = cfg.BINARY_CLASS_NAMES if args.task == "binary" else [str(c) for c in le.classes_]
    print("Train class counts:", dict(zip(class_names, np.bincount(ytr, minlength=len(class_names)))))

    # ---------- train / evaluate ----------
    rows, per_class, roc_curves, artifacts = [], {}, {}, {}
    for key in args.models:
        name = MODEL_NAMES[key]
        print(f"\n=== {name} ===")
        extra = tune(key, X_train, ytr, args) if args.tune else {}
        clf = make_model(key, len(class_names), args.n_estimators)
        if extra:
            clf.set_params(**extra)
        prep = build_preprocessor(X_train)

        t0 = time.perf_counter()
        if args.smote:
            Xt = prep.fit_transform(X_train)
            Xt, y_fit = oversample(Xt, ytr, args.smote_target, cfg.RANDOM_STATE)
            clf.fit(Xt, y_fit, **({"sample_weight": fit_kwargs(key, y_fit)["clf__sample_weight"]}
                                  if key == "xgb" else {}))
            pipe = Pipeline([("prep", prep), ("clf", clf)])
        else:
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            pipe.fit(X_train, ytr, **fit_kwargs(key, ytr))
        train_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = pipe.predict(X_test)
        infer_ms_per_1k = (time.perf_counter() - t0) / max(len(X_test), 1) * 1000 * 1000
        proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

        m = ev.compute_metrics(yte, pred, proba, class_names)
        m.update({"model": name, "train_seconds": train_s, "inference_ms_per_1000_flows": infer_ms_per_1k})
        rows.append(m)
        per_class[name] = ev.per_class_f1(yte, pred, class_names)
        print("  " + "  ".join(f"{k}={m[k]:.4f}" for k in
                               ["accuracy", "macro_f1", "attack_detection_rate", "false_alarm_rate", "roc_auc"]))

        (res_dir / f"report_{name}.txt").write_text(ev.text_report(yte, pred, class_names))
        ev.plot_confusion(yte, pred, class_names, f"{name} - normalised confusion matrix",
                          fig_dir / f"cm_{name}.png")
        if args.task == "binary" and proba is not None:
            roc_curves[name] = ev.roc_points(yte, proba)

        if hasattr(pipe.named_steps["clf"], "feature_importances_"):
            names = pipe.named_steps["prep"].get_feature_names_out()
            imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names)
            imp.sort_values(ascending=False).to_csv(res_dir / f"feature_importance_{name}.csv", header=["importance"])
            ev.plot_feature_importance(imp, f"{name} - top 20 features", fig_dir / f"importance_{name}.png")

        artifact = {
            "pipeline": pipe, "classes": class_names, "task": args.task,
            "feature_columns": list(X_train.columns), "model_name": name,
        }
        artifacts[name] = artifact
        joblib.dump(artifact, task_dir / f"{name}.joblib")

    # ---------- summaries ----------
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    front = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
             "attack_detection_rate", "false_alarm_rate", "roc_auc", "train_seconds",
             "inference_ms_per_1000_flows"]
    summary = summary[front]
    summary.to_csv(res_dir / "metrics_summary.csv", index=False)
    pc = pd.DataFrame(per_class, index=class_names)
    pc.to_csv(res_dir / "per_class_f1.csv")

    ev.plot_model_comparison(summary, fig_dir / "model_comparison.png")
    ev.plot_per_class_f1(pc, fig_dir / "per_class_f1.png")
    if roc_curves:
        ev.plot_roc_binary(roc_curves, fig_dir / "roc_curves.png")

    best = summary.iloc[0]["model"]
    joblib.dump(artifacts[best], task_dir / "best_model.joblib")
    (res_dir / "run_config.json").write_text(json.dumps(
        {**vars(args), "best_model": best, "dropped_columns": dropped, "classes": class_names}, indent=2))

    print("\n" + summary.round(4).to_string(index=False))
    print(f"\nBest model: {best}")
    print(f"Models  -> {task_dir}\nFigures -> {fig_dir}\nResults -> {res_dir}")
    return summary


if __name__ == "__main__":
    main()
