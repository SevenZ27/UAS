# Credit Score model card

Updated 21 October 2026. Model `stage2-49c6876730d2` is an educational classifier of the dataset's Good, Poor and Standard categories. It is not a probability-of-default model, lending approval tool or validated creditworthiness assessment.

## Model and intended use

The shared sklearn Pipeline applies the 18-feature source parser, development-fitted median imputation, development-fitted IQR clipping and a RandomForestClassifier. The selected forest has 300 trees, min_samples_leaf=3, class_weight=balanced_subsample, random_state=42 and n_jobs=2. Full parameters and the frozen selection are in [selection.json](evaluation/stage-two/selection.json).

Intended use is a portfolio demonstration of customer-grouped evaluation and consistent local/SageMaker inference hooks. The response returns categories and technical validation warnings. No probability is presented as an individual confidence or credit score. The local application uses the evaluated bundle; a packaged artifact and tests do not establish a running AWS deployment.

## Data and partitions

The private course archive contains 25,000 customer-month observations from 11,254 customers. Customer_ID and other identifiers are excluded from inputs. Development has 19,989 observations / 9,003 customers; holdout has 5,011 observations / 2,251 customers. No customer crosses either outer partition or training/validation within five development folds. All preprocessing fits inside each training fold. The final pipeline fits development only.

Original distribution rights, collection or generation process, currency, year, target construction and representativeness remain unverified. See the [data card](DATA-CARD.md) and [input contract](DATA-CONTRACT.md).

## Evaluation

Six primary candidates were compared by mean development-fold macro-F1 before holdout scoring; two extra preprocessing sensitivity variants were not eligible for selection. The selected forest achieved mean fold macro-F1 0.6553 (SD 0.0082). Fold SD is descriptive, not a confidence interval. Only the selected candidate and prior baseline were evaluated on the reserved holdout.

| Holdout metric | Selected forest | Prior baseline |
| --- | ---: | ---: |
| Accuracy | 68.61% | 52.78% |
| Macro-F1 | 0.6709 | 0.2303 |
| Balanced accuracy | 68.50% | 33.33% |
| Log loss | 0.7024 | 1.0062 |

The 95% customer-cluster bootstrap interval is 67.05–70.28% for accuracy and 0.6542–0.6886 for macro-F1 (2,000 replicates). The paired accuracy advantage over baseline is 15.83 percentage points, interval 13.45–18.23 points. Intervals condition on this fixed model and dataset and exclude training, selection and population-shift uncertainty. Customer-balanced accuracy is 68.58%; the primary metric weights monthly observations equally.

## Coverage and failure modes

- The historical strict serving contract accepted 3,067/5,011 parsed holdout profiles, **61.2% reference-domain coverage**. Accepted-only accuracy was 71.80% and macro-F1 0.6989; these are conditional subset results. The current public [exploration policy](SERVING-POLICY.md) returns predictions with warnings outside those reference ranges. The 61.2% figure is not its acceptance rate. Full-holdout benchmark performance must not be transferred unqualified to arbitrary app inputs.
- Good precision is only 54.84% (actual Good support 909); a predicted Good category is not a reliable loan-safety claim.
- The exploratory age-45+ slice has macro-F1 0.5777 across 819 observations; Poor recall is 30.16% among 126 actual Poor observations. This is a weakness, not a causal or certified fairness finding.
- An OOF-fitted temperature of 0.8913 reduces holdout log loss to 0.6981, Brier to 0.4194 and ten-bin top-label ECE to 0.0125. This post-selection diagnostic is not integrated into the category-only API and does not establish calibration in other populations.
- Customer-disjoint evaluation does not test future months, another bank or an external dataset. The meaning of target labels is not independently validated. Historical row-random scores answer a different question.

## Reproducibility and maintenance

[Protocol](EVALUATION-PROTOCOL.md), [run hashes](evaluation/stage-two/run-protocol.json), [split audit](evaluation/stage-two/split-audit.json), [holdout report](evaluation/stage-two/holdout-results.json), [independent checks](verification/evaluation-stage-two.json), and [full stage-two report](STAGE-TWO.md) document the experiment. Model SHA-256 is `d8f6ce0302a8954f4f15be17ad726bd26068ffacb3b2396bc0867f73a0af20b1`.

Raw CSV, identities, split indices and per-row predictions remain private/local. The active fitted pipeline and metadata are checked in so the app runs without the private training CSV; other binary checkpoints remain local. Only trusted joblib artifacts may be loaded. Active metadata records the model/report hashes and runtime; hashes detect accidental changes but do not authenticate untrusted authors. The previous integration bundle is preserved in a local checkpoint. Further tuning after this holdout has been examined must use a new evaluation design and untouched test data for new confirmatory claims.
