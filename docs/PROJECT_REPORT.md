# Intelligent Network Intrusion Detection and Attack Classification using Machine Learning

*Project report with results from the real UNSW-NB15 partitioned training and testing sets.*

---

## Abstract

Networks face a steady stream of attacks such as denial of service, exploitation and reconnaissance, and signature-based tools miss new or modified variants. This project builds a machine-learning intrusion detection system on the UNSW-NB15 dataset. It solves two tasks: (1) binary detection of normal versus malicious flows, and (2) classification of malicious flows into nine attack families. Five algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost and a Multi-Layer Perceptron) are compared using accuracy, macro-F1, attack detection rate, false alarm rate and ROC-AUC. Class imbalance is handled through class weighting and SMOTE. The best model is packaged with its preprocessing and exposed through a Streamlit application.

## 1. Introduction

### 1.1 Background
An Intrusion Detection System (IDS) monitors network traffic and raises alerts on malicious activity. *Signature-based* IDSs match known patterns and are precise but blind to novel attacks. *Anomaly/ML-based* IDSs learn what traffic looks like and generalise to unseen variants, at the cost of false alarms and the need for good training data.

### 1.2 Problem statement
Given a network flow described by statistical features (duration, packet and byte counts, protocol, TTL, loads, connection-count features), decide whether it is normal or an attack and, if an attack, which family it belongs to.

### 1.3 Objectives
1. Build a reproducible pipeline for cleaning, encoding and scaling UNSW-NB15 flows.
2. Train and compare five classifiers for binary and multiclass tasks.
3. Address severe class imbalance and evaluate fairly on rare attacks.
4. Report IDS-relevant metrics (detection rate, false alarm rate) alongside standard ones.
5. Deliver a usable application for batch classification of flow records.

## 2. Dataset: UNSW-NB15

Created by the Australian Centre for Cyber Security (UNSW Canberra). Realistic normal traffic and synthetic modern attacks were generated in a cyber-range with the IXIA PerfectStorm tool; raw packets were captured with tcpdump, and flow features were extracted with Argus and Bro-IDS using custom scripts. The full dataset has about 2.54 million records with 49 attributes. The authors also provide a partitioned subset, used here:

| Split | Records |
|---|---|
| Training set | 175,341 |
| Testing set | 82,332 |

Each record has a binary `label` (0 normal, 1 attack) and an `attack_cat` field with ten values.

| Class | Description |
|---|---|
| Normal | Benign traffic |
| Generic | Attacks against block ciphers regardless of the key (hash-collision style) |
| Exploits | Use of known vulnerabilities or bugs to gain control |
| Fuzzers | Feeding random data to crash or probe a service |
| DoS | Denial-of-service attempts |
| Reconnaissance | Information gathering and probing (scans) |
| Analysis | Port scans, spam and web-script probing |
| Backdoor | Bypassing normal authentication for covert access |
| Shellcode | Small payloads used to take control of a process |
| Worms | Self-replicating malware spreading across hosts |

Approximate training-set counts (verify against your copy with `python -m src.eda`): Normal 56,000; Generic 40,000; Exploits 33,393; Fuzzers 18,184; DoS 12,264; Reconnaissance 10,491; Analysis 2,000; Backdoor 1,746; Shellcode 1,133; Worms 130. The imbalance between the largest and smallest class is more than 400 to 1, which is the central difficulty.

**Feature groups:** flow features (`proto`, `service`, `state`, `dur`), basic counts (`spkts`, `dpkts`, `sbytes`, `dbytes`), traffic-rate features (`rate`, `sload`, `dload`, `sinpkt`, `dinpkt`, jitter), TCP features (`swin`, `dwin`, `stcpb`, `dtcpb`, `tcprtt`, `synack`, `ackdat`), content features (`smean`, `dmean`, `trans_depth`, `response_body_len`) and time-window connection-count features (`ct_*`).

## 3. Methodology

### 3.1 Pipeline overview
Load official train/test files, clean, preprocess (fit on train only), train five models, evaluate on the held-out test split, save the best pipeline, and serve it through a CLI and a Streamlit app.

### 3.2 Data cleaning
- Column names lower-cased and trimmed; category text normalised.
- Infinite values converted to missing and then median-imputed.
- Duplicate rows (ignoring `id`) optionally removed from the **training split only**, so the test split stays untouched.
- `id`, `label` and `attack_cat` removed from the input features to prevent target leakage.

### 3.3 Preprocessing
| Step | Applied to | Reason |
|---|---|---|
| Median imputation | numeric | robust to outliers |
| Signed log(1+x) | numeric | compresses heavy-tailed byte/rate/load features |
| Standardisation | numeric | needed by MLP and logistic regression |
| One-hot encoding (rare categories grouped, unseen categories tolerated) | `proto`, `service`, `state` | `proto` has over 100 values, many rare |
| Optional correlation filter (abs corr > 0.95, fit on train) | numeric | reduces redundancy |

### 3.4 Handling class imbalance
1. **Class weights** (`balanced`) in logistic regression, decision tree and random forest; sample weights in XGBoost.
2. **SMOTE** (optional) raises every class below a target size (default 10,000) using synthetic interpolation on the preprocessed training data only.
3. **Metrics** that expose minority-class failure: macro-F1 and per-class recall.

### 3.5 Models
| Model | Role |
|---|---|
| Logistic Regression | linear baseline |
| Decision Tree | interpretable baseline |
| Random Forest | bagged trees, strong tabular baseline, gives feature importance |
| XGBoost | gradient-boosted trees, typically top performer on tabular data |
| MLP (128-64) | non-linear neural baseline with early stopping |

Hyper-parameters can be tuned with randomised search (3-fold stratified CV, macro-F1) on a stratified subsample, then the best setting is refit on the full training set.

### 3.6 Evaluation metrics
- **Accuracy**: share of correct predictions (misleading under imbalance).
- **Precision, recall, F1** per class; **macro-F1** averages classes equally.
- **Attack detection rate (DR)**: attacks flagged as any attack divided by all attacks.
- **False alarm rate (FAR)**: normal flows flagged as attacks divided by all normal flows.
- **ROC-AUC** (one-vs-rest macro for multiclass).
- **Training time and inference time per 1,000 flows**, relevant for real-time deployment.

## 4. System design

```
Raw flow CSV ─► clean_frame ─► ColumnTransformer ─► classifier ─► class + confidence
                                (inside one saved sklearn Pipeline)
```
The saved artifact stores the pipeline, class names, task type and the expected feature columns, so new data is aligned automatically. The Streamlit app lets a user upload flows, view detection KPIs and class counts, inspect single flows with class probabilities, and download predictions.

## 5. Implementation summary
Python 3, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, Streamlit. Code lives in `src/` (config, loading, preprocessing, models, evaluation, training, prediction, EDA). Random seed fixed at 42 for reproducibility.

## 6. Experiments to run

| ID | Experiment | Status | Output |
|---|---|---|---|
| E1 | EDA | Complete | `reports/results/eda/`, `reports/figures/eda/` |
| E2 | Binary baseline, five models | Complete | `reports/results/binary_real/`, `reports/figures/binary_real/` |
| E3 | Multiclass baseline, five models | Complete | `reports/results/multiclass_real/`, `reports/figures/multiclass_real/` |
| E4 | Multiclass with SMOTE | Complete | `reports/results/multiclass_real_smote/`, `reports/figures/multiclass_real_smote/` |
| E5 | Tuned RF and XGBoost | Not run; optional follow-up | `python -m src.train --task multiclass --models rf xgb --tune --tag tuned` |
| E6 | Reduced features | Not run; optional follow-up | `python -m src.train --drop-correlated --tag nocorr` |

## 7. Results and discussion

The runs use the correctly assigned partition sizes: 175,341 training records and
82,332 test records. Cleaning found no missing or infinite cells in this copy of the
dataset. The training split contains 67,601 duplicate rows when comparing all columns
except `id`; they were retained because `--dedupe` was not enabled. Test records were
not deduplicated or used in fitting.

### 7.1 Binary detection

| Model | Accuracy | Macro-F1 | Detection rate | False-alarm rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8681 | 0.8642 | 0.9407 | 0.2209 | 0.9670 |
| Decision Tree | 0.8620 | 0.8563 | 0.9636 | 0.2625 | 0.8523 |
| Random Forest | 0.8926 | 0.8889 | **0.9766** | 0.2103 | 0.9814 |
| XGBoost | **0.8964** | **0.8934** | 0.9661 | **0.1890** | **0.9827** |
| MLP | 0.8681 | 0.8627 | 0.9686 | 0.2551 | 0.9767 |

XGBoost gives the strongest overall binary accuracy, macro-F1, ROC-AUC and lowest
false-alarm rate among these runs. Random Forest has the highest attack detection rate,
illustrating the trade-off between catching attacks and avoiding false alarms.

### 7.2 Multiclass classification

| Model | Accuracy | Macro-F1 | Detection rate | False-alarm rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6195 | 0.3562 | 0.9922 | 0.4243 | 0.9436 |
| Decision Tree | 0.7255 | 0.4888 | 0.9606 | 0.2460 | 0.8265 |
| Random Forest | 0.6919 | 0.4982 | **0.9953** | 0.3750 | 0.9072 |
| XGBoost | 0.6967 | **0.5219** | 0.9897 | 0.3621 | **0.9592** |
| MLP | **0.7674** | 0.4635 | 0.9610 | **0.2075** | 0.9538 |

Accuracy and macro-F1 rank models differently: MLP has the highest accuracy, while
XGBoost has the highest macro-F1. Macro-F1 weights all attack families equally, so it
better reflects performance on rare classes than accuracy alone. XGBoost's per-class
F1 is high for Generic (0.985) and Exploits (0.700), but low for Analysis (0.063),
Backdoor (0.111), and Shellcode (0.409). Worms has only 130 training rows and its
XGBoost F1 is 0.697, so this estimate is particularly sensitive to the small class
sample. These results show that strong overall scores do not mean uniformly reliable
classification across all attack families.

### 7.3 Multiclass with SMOTE

| Model | Accuracy | Macro-F1 | Detection rate | False-alarm rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6220 | 0.3622 | 0.9914 | 0.4236 | 0.9440 |
| Decision Tree | 0.7255 | 0.4795 | 0.9616 | 0.2533 | 0.8215 |
| Random Forest | 0.6906 | 0.4814 | **0.9950** | 0.3756 | 0.9370 |
| XGBoost | 0.6982 | **0.5106** | 0.9907 | 0.3743 | **0.9605** |
| MLP | **0.7221** | 0.4452 | 0.9767 | **0.2856** | 0.9495 |

SMOTE did not improve macro-F1 for any of the five classifiers in this run: the best
score decreased from 0.5219 to 0.5106. It improved XGBoost's DoS F1 from 0.253 to 0.387
but lowered its Shellcode F1 from 0.409 to 0.370 and Worms F1 from 0.697 to 0.552.
Oversampling is therefore not a universal improvement; per-class metrics and the
operational cost of misses versus false alarms should guide model selection.

Logistic Regression emitted a convergence warning at its iteration limit on the
multiclass runs. Its results are reported as measured; increasing its iteration limit
or tuning the solver may change those metrics.

### 7.4 Artifacts and repeatability

Full metrics, classification reports and per-class scores are in the three results
folders listed above. Confusion matrices, comparison plots, per-class F1 plots, and
feature-importance plots are in the corresponding `reports/figures/*_real/` folders.
The binary ROC curves are in `reports/figures/binary_real/roc_curves.png`. The saved
pipelines (including preprocessing) are in `models/binary_real/`,
`models/multiclass_real/`, and `models/multiclass_real_smote/`. The selected multiclass
XGBoost artifact was smoke-tested on 100 held-out rows and returned ten-class
probabilities.

## 8. Limitations
- Dataset age and lab origin; real traffic differs and attacks evolve.
- Flow statistics only; no packet payloads or long-term host behaviour.
- Rare classes have very few examples, so their metrics are noisy.
- Supervised models detect only the attack types present in training; unseen attacks need anomaly or open-set methods.

## 9. Future work
- Cross-dataset validation (CICIDS2017, CSE-CIC-IDS2018) to test generalisation.
- Deep sequence models (1D-CNN, LSTM) and autoencoder-based anomaly detection.
- Explainability with SHAP so analysts see why a flow was flagged.
- Probability calibration and cost-sensitive thresholds.
- Streaming deployment (Kafka or live pcap-to-flow extraction) with drift monitoring.

## 10. Conclusion

On the official UNSW-NB15 partition, XGBoost was the strongest overall binary model
(macro-F1 0.8934, attack detection rate 0.9661, false-alarm rate 0.1890) and had the
highest macro-F1 for multiclass classification (0.5219). MLP achieved higher
multiclass accuracy (0.7674) and lower false-alarm rate (0.2075), but lower macro-F1
(0.4635), while SMOTE reduced the best multiclass macro-F1 to 0.5106. Rare attack
families remain difficult and require careful per-class validation before any
operational deployment.

## References
1. N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems," MilCIS, 2015.
2. N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, 2002.
3. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016.
4. L. Breiman, "Random Forests," *Machine Learning*, 2001.
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 2011.
