# Viva / interview questions and answers

**1. What is an intrusion detection system?**
A system that monitors network or host activity and alerts on malicious behaviour. Signature-based IDSs match known patterns; ML-based IDSs learn traffic behaviour and can catch variants.

**2. Why UNSW-NB15 and not KDD99 or NSL-KDD?**
KDD99 (1998) and NSL-KDD are outdated and contain artefacts. UNSW-NB15 (2015) has modern attack families and 49 features from realistic traffic mixed with synthetic attacks.

**3. What are the tasks in your project?**
Binary detection (Normal vs Attack) and multiclass classification (Normal plus 9 attack families).

**4. What attack families are in the dataset?**
Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode and Worms.

**5. What is the main data challenge?**
Severe class imbalance: Worms has about 130 training rows versus tens of thousands for Normal and Generic.

**6. How did you handle imbalance?**
Class weights, optional SMOTE on the training data only, and evaluation by macro-F1 and per-class recall instead of accuracy.

**7. Why is accuracy not enough?**
A model can score high accuracy by ignoring rare classes. Macro-F1 gives every class equal weight.

**8. What are detection rate and false alarm rate?**
Detection rate is the share of attacks flagged as attacks (recall). False alarm rate is the share of normal flows flagged as attacks. An IDS needs high detection and low false alarms.

**9. What preprocessing did you apply?**
Infinite values to NaN, median imputation, signed log transform, standardisation, and one-hot encoding of `proto`, `service` and `state`.

**10. Why the log transform?**
Byte, rate and load features are heavy-tailed. Log compression stabilises MLP and logistic regression training. Tree models do not need it.

**11. How did you avoid data leakage?**
All fitted steps (imputer, scaler, encoder, correlation filter, SMOTE) are fitted on training data only; `id`, `label` and `attack_cat` are removed from the features; the test split is never tuned on or deduplicated.

**12. Why use the official train/test split?**
It makes results comparable with published work and keeps the test set untouched.

**13. Why SMOTE only on training data?**
Synthetic samples in the test set would give unrealistically optimistic results.

**14. Which model did best and why?**
[Fill in from your results.] Tree ensembles such as Random Forest and XGBoost usually do best on tabular flow features because they handle mixed scales, non-linearity and feature interactions.

**15. Why did rare classes perform poorly?**
Few training examples, and overlap with larger classes such as Exploits and Fuzzers. Check your confusion matrix.

**16. What is a confusion matrix?**
A table of true class versus predicted class. The diagonal shows correct predictions; off-diagonal cells show which classes get mixed up.

**17. What is ROC-AUC?**
The area under the true-positive-rate versus false-positive-rate curve; 1.0 is perfect and 0.5 is random. For multiclass we use one-vs-rest macro averaging.

**18. How does Random Forest work?**
It trains many decision trees on bootstrap samples with random feature subsets and averages their votes, which reduces variance.

**19. How does XGBoost differ?**
It builds trees sequentially, each correcting the previous trees' errors using gradient boosting with regularisation.

**20. What is the bias-variance trade-off?**
Simple models (logistic regression, shallow trees) have high bias; very deep trees have high variance. Ensembles and regularisation balance the two.

**21. How would you deploy this?**
Save the pipeline, wrap it in an API or the Streamlit app, convert live traffic to flow features (for example with Argus or CICFlowMeter-style tools), and monitor drift and false alarms.

**22. What are the limitations?**
Old lab dataset, flow-level features only, tiny rare classes, and supervised models miss unseen attack types.

**23. What would you do next?**
Test on CICIDS2017, add SHAP explanations, try 1D-CNN or LSTM and autoencoder anomaly detection, calibrate probabilities, and choose thresholds from a cost model.

**24. How do you know your result is reliable?**
Fixed random seed, official split, per-class metrics, and (optionally) repeated runs or cross-validation on the training set for variance estimates.
