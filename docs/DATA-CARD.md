# Credit Score data card

## Source and intended question

This project continues Steven's Model Deployment course work. The local source is the unchanged `data_C.csv` in the archived UAS/lokal folder. SHA-256: `464a762d352a33ce6f18ae42a1bc729d8da5aa03e50d8736e0ef83d854c22412`.

The benchmark question is whether 18 available numeric/binary features classify a previously unseen customer's monthly observation as Good, Poor or Standard. Those labels are supplied by the dataset. Repayment events, future default, approval decisions, loss given default and a prediction horizon are not supplied, so this benchmark does not measure them.

The original distributor, collection/generation method, license, currency, observation year, population and label-construction rules remain unverified. The schema resembles a widely circulated Credit Score Classification dataset; similarity does not establish provenance. The Kaggle listing could not supply inspectable source metadata during this audit. The 25,000-row subset's selection mechanism is not documented. These gaps limit external interpretation even after a sound internal evaluation.

## Grain and coverage

- 25,000 observations, 29 source columns including an unnamed exported index.
- Customer-month grain, 11,254 unique Customer_ID values, between one and seven observations per customer.
- Months January through August; no verified year or prospective ordering protocol.
- Target counts: Standard 13,282; Poor 7,268; Good 4,450.
- No exact duplicate rows or duplicated Customer_ID/Month pairs were found in the full-data audit.
- Source fields include Name and SSN. They are not model inputs, are not reproduced in reports, and are not packaged for deployment. Their presence is not evidence that these are real identifiable people.

## Features and units

The executable [contract](data-contract.json) and [processing decisions](DATA-CONTRACT.md) define the 18 features and validation. All monetary values retain unspecified source-dataset units. Changed_Credit_Limit retains a generic dataset unit because a percentage/nominal interpretation has not been established. Counts require integers, credit-history age uses months, Age uses years, and Payment_of_Min_Amount accepts a documented binary encoding. A negative limit change or signed due-date difference is not globally replaced with a positive number.

Customer_ID controls splitting only. Name, SSN, exported index, ID and Month are excluded from model inputs. Occupation, Type_of_Loan, Credit_Mix and Payment_Behaviour are also excluded under the current numeric-feature contract. Their possible predictive value or availability at prediction time is not established by their exclusion.

## Quality and transformations

The full-data audit found 3,836 missing Monthly_Inhand_Salary values and 2,240 missing Credit_History_Age values; parsing also exposes sentinel/malformed values. A previous regex changed 150 negative delayed-payment counts into positive values. The new parser retains the sign and marks invalid negative counts as missing. Source-format strings are parsed without changing raw data.

Training-domain invalid values become missing, and the manifest records per-feature counts. Median imputation and IQR fences are learned separately within every development fold. The final model fits those parameters on development data only. No data-dependent transform learns from the holdout. No rows or labels are removed to increase the score.

IQR clipping is a starting policy, not a proven universal repair. Development-only no-clipping and missing-indicator sensitivity runs quantify dependence on that policy. Results should be read alongside missingness slices and strict serving-domain coverage.

## Evaluation boundaries

Use the reserved customer holdout and grouped development folds described in the [protocol](EVALUATION-PROTOCOL.md). The old row-random split placed 79.42% of test observations in customers already present during training. Its historical scores answer a different and less clearly specified generalization question.

The new benchmark is customer-disjoint, not temporal, geographic, bank-specific or external validation. Historical strict-domain coverage and accepted-only results are reported separately. The current [exploration policy](SERVING-POLICY.md) returns warnings rather than rejecting schema-valid values outside training reference ranges; this change adds no new evidence of reliability outside the benchmark. Neither internal accuracy nor cluster-bootstrap intervals resolve the provenance and target-validity gaps.

## Distribution and maintenance

The raw CSV, source identifiers, private split indices and per-row prediction arrays remain local and ignored by Git. Public-facing evidence consists of aggregate tables, figures, hashes and implementation. A future data refresh must be versioned; after the holdout has been examined, it is a known benchmark rather than a fresh test set for further tuning.
