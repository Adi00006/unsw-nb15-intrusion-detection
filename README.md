# Intelligent Network Intrusion Detection and Attack Classification using Machine Learning

An end-to-end machine-learning project on the **UNSW-NB15** dataset. It detects malicious
network flows (binary: *Normal vs Attack*) and classifies them into one of **9 attack
families** (multiclass), compares five algorithms, and ships a **Streamlit** demo.

```
 CSV flows ─► clean ─► preprocess ─► train 5 models ─► evaluate ─► save best model ─► Streamlit / CLI predict
              (inf,    (impute, log,   LR · DT · RF     (macro-F1,    (.joblib incl.
              dupes)    scale, 1-hot)   XGB · MLP        detection &    preprocessing)
                                                         false-alarm)
```

## 1. Project structure

```
unsw_nb15_ids_project/
├── data/                  # put the two UNSW-NB15 CSVs here (see data/README.md)
├── src/
│   ├── config.py          # paths, column names, constants
│   ├── data_loader.py     # load + clean (inf -> NaN, text normalisation, optional train dedupe)
│   ├── preprocessing.py   # ColumnTransformer: impute -> signed log1p -> standardise; one-hot
│   ├── models.py          # LR, Decision Tree, Random Forest, XGBoost, MLP + search spaces
│   ├── evaluate.py        # metrics + plots (confusion, ROC, comparison, importance)
│   ├── train.py           # main training CLI (SMOTE / tuning / correlated-feature drop)
│   ├── predict.py         # load a saved model and classify a CSV
│   └── eda.py             # exploratory analysis figures and tables
├── app/streamlit_app.py   # interactive demo
├── scripts/make_synthetic_data.py   # fake data with the same schema, for smoke tests only
├── tests/test_pipeline.py
├── docs/PROJECT_REPORT.md # full report text (methodology, discussion) to adapt for submission
├── docs/VIVA_QA.md        # likely viva questions with answers
├── requirements.txt
└── run_all.sh
```

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Get the data

Download the **partitioned** files and place them in `data/`:

- `UNSW_NB15_training-set.csv` (about 175k rows)
- `UNSW_NB15_testing-set.csv` (about 82k rows)

Sources: official page https://research.unsw.edu.au/projects/unsw-nb15-dataset or the Kaggle
mirror https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 (search the name if a link moved).

No dataset yet? Create **synthetic** stand-in files to check that everything runs
(their scores are meaningless):

```bash
python scripts/make_synthetic_data.py --out data/synthetic
python -m src.train --train data/synthetic/UNSW_NB15_training-set.csv \
                    --test  data/synthetic/UNSW_NB15_testing-set.csv --tag smoke
```

## 4. Run the project

Run all commands from the project root.

| Step | Command | What you get |
|---|---|---|
| EDA | `python -m src.eda` | class balance, protocol mix, feature distributions, correlation heatmap, summary tables |
| Binary detection | `python -m src.train --task binary` | Normal vs Attack for 5 models |
| Attack classification | `python -m src.train --task multiclass` | 10-class results for 5 models |
| Imbalance handling | `python -m src.train --task multiclass --smote --tag smote` | SMOTE-oversampled rare classes |
| Tuning | `python -m src.train --task multiclass --models rf xgb --tune --n-iter 10 --tag tuned` | randomised search (3-fold CV, macro-F1) |
| Fewer features | `python -m src.train --drop-correlated --tag nocorr` | drops features with abs(corr) > 0.95 |
| Everything | `bash run_all.sh` | EDA + binary + multiclass + SMOTE runs |
| Everything in Jupyter | Open `complete_project_workflow.ipynb` and run all cells | EDA, training, evaluation, predictions from all five algorithms, and saved charts/reports |
| Demo | `streamlit run app/streamlit_app.py` | web UI |
| Batch predict | `python -m src.predict --input some_flows.csv --output out.csv` | predictions CSV |
| Jupyter: all real-model predictions | Open `all_model_predictions.ipynb` and run all cells | predictions from all five algorithms for binary, multiclass, and SMOTE runs, plus evaluation metrics and charts |
| Tests | `pytest -q` | smoke tests on tiny synthetic data |

Useful flags: `--models lr dt rf xgb mlp` (pick a subset), `--n-estimators 100` (faster),
`--dedupe` (remove duplicate training rows), `--tag NAME` (keep runs side by side).

## 5. Outputs

| Location | Content |
|---|---|
| `models/<task>[_tag]/` | one `.joblib` per model plus `best_model.joblib` (preprocessing + classifier in one pipeline) |
| `reports/results/<task>[_tag]/metrics_summary.csv` | accuracy, macro P/R/F1, weighted F1, **attack detection rate**, **false alarm rate**, ROC-AUC, train time, inference time |
| `reports/results/.../per_class_f1.csv`, `report_<Model>.txt` | per-attack-family precision / recall / F1 |
| `reports/results/.../feature_importance_<Model>.csv` | tree-model importances |
| `reports/figures/<task>[_tag]/` | confusion matrices, model comparison, per-class F1, ROC (binary), feature importance |
| `reports/figures/eda/` | EDA figures |

## 6. Design decisions (and why)

1. **Official train/test split.** The UNSW authors' partition is used so results are comparable with published work. Never tune on the test set.
2. **Fit everything on train only.** Imputer, scaler, encoder, correlation filter and SMOTE see only training data, so no leakage.
3. **Signed log transform.** Bytes, rates and loads span many orders of magnitude; the log keeps MLP and logistic regression stable (trees are unaffected).
4. **Class imbalance.** Worms, Shellcode, Backdoor and Analysis are tiny. The code uses class weights (balanced) and an optional SMOTE mode, and reports **macro-F1** and per-class metrics instead of accuracy alone.
5. **IDS-specific metrics.** *Attack detection rate* (attacks flagged as any attack) and *false alarm rate* (normal flows flagged) matter more to an analyst than accuracy.
6. **Whole pipeline saved.** The `.joblib` holds preprocessing and model, so prediction on raw CSV rows needs no extra code.
7. **Duplicates.** Only the training split can be deduplicated (`--dedupe`); the test split is never altered.

## 7. Results on the real UNSW-NB15 partition

The complete run used 175,341 training rows and 82,332 testing rows. Results below are
from the official partitioned CSV schema; the test split was not used for fitting. These
are benchmark results, not production-performance guarantees. The CSVs in
`data/UNSW_NB15_training-set.csv` and `data/UNSW_NB15_testing-set.csv` must be the
175,341-row training split and 82,332-row testing split, respectively.

### Binary detection

| Model | Accuracy | Macro-F1 | Detection rate | False-alarm rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8681 | 0.8642 | 0.9407 | 0.2209 | 0.9670 |
| Decision Tree | 0.8620 | 0.8563 | 0.9636 | 0.2625 | 0.8523 |
| Random Forest | 0.8926 | 0.8889 | 0.9766 | 0.2103 | 0.9814 |
| XGBoost | **0.8964** | **0.8934** | 0.9661 | 0.1890 | **0.9827** |
| MLP | 0.8681 | 0.8627 | **0.9686** | 0.2551 | 0.9767 |

### Multiclass attack classification

| Model | Accuracy | Macro-F1 | Detection rate | False-alarm rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6195 | 0.3562 | 0.9922 | 0.4243 | 0.9436 |
| Decision Tree | 0.7255 | 0.4888 | 0.9606 | 0.2460 | 0.8265 |
| Random Forest | 0.6919 | 0.4982 | **0.9953** | 0.3750 | 0.9072 |
| XGBoost | 0.6967 | **0.5219** | 0.9897 | 0.3621 | **0.9592** |
| MLP | **0.7674** | 0.4635 | 0.9610 | **0.2075** | 0.9538 |

### Multiclass with SMOTE

| Model | Accuracy | Macro-F1 | Detection rate | False-alarm rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6220 | 0.3622 | 0.9914 | 0.4236 | 0.9440 |
| Decision Tree | 0.7255 | 0.4795 | 0.9616 | 0.2533 | 0.8215 |
| Random Forest | 0.6906 | 0.4814 | **0.9950** | 0.3756 | 0.9370 |
| XGBoost | 0.6982 | **0.5106** | 0.9907 | 0.3743 | **0.9605** |
| MLP | **0.7221** | 0.4452 | 0.9767 | **0.2856** | 0.9495 |

The complete metrics, per-class scores and reports are saved in
`reports/results/binary_real/`, `reports/results/multiclass_real/` and
`reports/results/multiclass_real_smote/`. Corresponding model artifacts and figures are
saved in `models/` and `reports/figures/`. SMOTE did not improve multiclass macro-F1 in
this run; compare per-class results before choosing a model for a specific use case.

## 8. Known limitations

- UNSW-NB15 is a 2015 lab-generated dataset; attack mixes and traffic have changed since. Treat results as benchmark performance, not production performance.
- The official test split has a different class mix from the training split, so a model can lose accuracy there. This is a known property of the dataset, not a bug.
- Flow-level features only; there is no payload inspection or session context.
- The real-data EDA, binary, multiclass and SMOTE runs have completed. The test suite passes (`pytest -q`: 4 passed). Logistic Regression emitted an iteration-limit convergence warning in the multiclass runs; its reported metrics are included as produced, and may improve with further tuning.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for the CSVs | file names must match exactly, or pass `--train` / `--test` |
| `KeyError: attack_cat` | you used a raw `UNSW-NB15_1..4.csv` file; use the partitioned training/testing files |
| Training is slow | `--models rf xgb`, `--n-estimators 100`, or skip `lr`/`mlp` |
| `ModuleNotFoundError: src` | run from the project root with `python -m src.train` |
| XGBoost import error | `pip install xgboost` (or omit it with `--models lr dt rf mlp`) |

## 10. Reference

N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," *Military Communications and Information Systems Conference (MilCIS)*, 2015.
