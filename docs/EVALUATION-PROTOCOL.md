# Credit Score evaluation protocol

Locked before the stage-two run on 21 October 2026. This is a local benchmark of dataset categories for previously unseen customers, not a test of loan repayment or a temporal forecast.

## Population, unit and partitions

The unchanged 25,000-row CSV is identified by SHA-256 in the stage-one manifest. The observation is a customer-month; the independent grouping unit is Customer_ID. Monetary units, original license, label construction and year remain unverified. No row exclusions or label corrections are made during this experiment.

Reuse the exact stage-one GroupShuffleSplit seed-42 indices: 19,989 development observations and 5,011 holdout observations, with no Customer_ID overlap. Do not regenerate a different holdout. Five shuffled StratifiedGroupKFold development folds use seed 42. Every observation is predicted once out of fold, every fold contains all three classes, and customer IDs cannot cross train/validation boundaries. Parsing, imputation, outlier fences and any scaling fit inside each fold's Pipeline.

## Prespecified candidates

All candidates use the existing 18-feature parser and train-only median imputation. The primary six use train-only IQR clipping.

| Candidate | Fixed settings |
|---|---|
| Prior baseline | DummyClassifier strategy prior; majority class for labels, development training frequencies for probabilities |
| Logistic regression | StandardScaler, C=1, max_iter=3000, unweighted classes |
| Decision tree | max_depth=10, min_samples_leaf=10, random_state=42 |
| Random forest | 100 trees, default depth/leaf settings, random_state=42, two jobs |
| Regularized random forest | 300 trees, min_samples_leaf=3, balanced_subsample class weights, random_state=42, two jobs |
| Histogram gradient boosting | 150 iterations, learning_rate=.1, max_leaf_nodes=15, l2_regularization=1, early_stopping=False, random_state=42 |

Two development-only sensitivity runs use the 100-tree forest: one removes IQR clipping; the other adds missingness indicators. They diagnose preprocessing choices and are explicitly excluded from winner selection for this run. Findings can motivate a future benchmark with a new untouched holdout, not post-hoc changes to this experiment.

## Selection and calibration

Primary selection metric: highest mean fold macro-F1 among the six primary candidates. Ties use lower mean fold log loss, then candidate name. Report accuracy, balanced accuracy, weighted-F1, macro-F1, log loss and multiclass Brier score for context. Fold SD is descriptive variation, not a confidence interval. Selection from CV has uncertainty and does not prove the winning algorithm is universally superior.

Freeze the selected candidate, parameters, fold results and protocol/source hashes in selection.json before accessing holdout labels for scoring. Fit one scalar temperature on the selected model's development out-of-fold probabilities, minimizing log loss over T in [exp(-2), exp(2)]. Report its development fit and holdout performance alongside uncalibrated probabilities. OOF calibration is post-selection and may be optimistic; the holdout assesses that fitted transformation. Temperature scaling preserves the argmax category. The serving API continues returning categories without a confidence percentage in this phase.

Fit the selected model and the prior baseline on all development rows. Score these two on the holdout once; do not inspect other candidates on holdout and reselect. Preserve the final reports even if performance or calibration deteriorates. A rerun into the same final-output directory must refuse to overwrite the first final evaluation.

## Outcomes and uncertainty

The primary estimand weights monthly observations equally among previously unseen customers. Also report customer-balanced metrics with each customer contributing total weight one. Record confusion matrix, each class's precision/recall/F1/support, and probability diagnostics.

Use 2,000 seeded cluster-bootstrap replicates of holdout customers, retaining all their monthly observations. Report 95% percentile intervals for accuracy, macro-F1, balanced accuracy and per-class recall. Paired bootstrap differences versus the prior baseline use the same customer resamples. Intervals condition on the fixed trained model and this dataset; they do not include retraining, selection or population-shift uncertainty. No p-value or causal claim is planned.

Calibration diagnostics include multiclass log loss, Brier score defined as the mean sum of squared class-probability errors, and ten fixed-bin top-label reliability statistics. ECE is descriptive and depends on binning; a lower proper scoring-rule loss alone does not establish good calibration.

## Coverage and error slices

The benchmark includes all raw holdout observations through the pipeline. Separately measure how many profiles pass the strict serving contract and learned input ranges after source-format parsing; report coverage and accepted-only metrics, rather than transferring whole-holdout performance unqualified to the app. Do not discard rejected profiles from the primary benchmark.

Report descriptive slices by parsed age (under 25, 25–44, 45+, missing) and number of missing parsed features (zero, 1–2, 3+). These are exploratory diagnostics, not fairness certification or confirmatory subgroup comparisons. No policy change is selected from holdout slices.

## Reproducibility and limits

Retain dataset, split, source, parameter and model hashes; dependency versions; fold row/customer/class counts; OOF predictions privately; final prediction arrays privately; complete aggregate CSV/JSON; tests and static figures with source tables. Raw identifiers and CSV rows stay out of reports and packages. Historical 73.24%/73.56% scores use a different protocol and cannot be treated as paired improvement estimates.

References: [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [pipeline leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), and [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
