# Data contract and preprocessing decisions

Contract 1.0 uses the 18 feature names in [data-contract.json](data-contract.json). The target classes are Good, Poor and Standard. These are categories from the source dataset, not measured probabilities of default.

## Input boundaries

API requests must include all 18 names. An explicit `null` is accepted for nullable features; Age is required. Unknown keys and positional arrays are rejected. Values are finite numbers; count features and month-based credit history are integers. Boolean values are not treated as numbers. Payment_of_Min_Amount is 0 or 1, or null when unknown.

The general schema enforces nonnegative counts/amounts, Age 0–120, utilization and interest rate 0–100, and binary payment status. Credit-limit change, due-date difference and monthly balance retain their signs. These are prototype data-domain rules, not lending criteria or an assertion that every negative value is meaningful.

The fitted bundle publishes reference ranges based on transformed training observations. Since the 21 October 2026 release, the public UI and packaged inference hooks accept schema-valid values outside those reference ranges and return a prominent warning. They preserve the submitted values and use the original fitted pipeline, including its saved IQR clipping. The response discloses which unfamiliar values were clipped and the values actually used. It does not silently clamp inputs to observed training minima/maxima. See [serving policy](SERVING-POLICY.md).

Initial examples use training medians, rounded to two decimals for non-count features and disclosed as such. Missing nullable values are imputed with the original saved training medians and disclosed. Monetary labels are simplified; their help text explains that the currency is unknown and no rupiah conversion is performed. The historical strict service remains available for reproducing the frozen evaluation's coverage analysis, not as the current public app policy.

## Training source adapter

Training reads the raw CSV, checks required fields, target classes, missing Customer_ID and duplicate Customer_ID/Month records, then reserves groups with GroupShuffleSplit seed 42. Customer_ID is used for splitting only; it is not a model feature. All customer-identifying fields stay outside the model bundle and public reports.

The shared FeatureCleaner selects no columns implicitly: it requires the exact feature set. It accepts plain numeric source strings with leading/trailing underscore decoration; other malformed text becomes missing. It does not remove arbitrary embedded characters or the minus sign. Non-finite values become missing. Values violating schema-domain rules become missing during training, with counts recorded in the manifest.

Credit_History_Age accepts `2 Years and 3 Months` or numeric 27 months. Payment_of_Min_Amount accepts Yes/No or numeric 1/0. Applying the stateless parser again to canonical numeric features preserves them; reprocessing no longer erases the two columns.

## Fitted transformations

1. Parse and validate source values using FeatureCleaner.
2. Fit SimpleImputer medians on training rows only.
3. Fit IQR fences on the imputed training matrix only. Fences respect schema bounds; count fences are integers, and payment stays within 0–1. Negative signs on permitted signed fields are preserved rather than globally replaced with the median.
4. Fit the selected RandomForestClassifier with 300 estimators, min_samples_leaf=3, balanced_subsample class weights, seed 42 and two jobs. The historical stage-one integration builder uses 100 trees; the stage-two evaluated bundle uses the frozen selection in [selection.json](evaluation/stage-two/selection.json).

The entire fitted Pipeline is saved as one artifact. Local predictions and cloud hooks call the same Pipeline.predict; they never fit transformations or calculate statistics from incoming requests. Raw CSV rows are never used as an already processed train.csv input for a second split/parse/training stage.

## Bundle and responses

Metadata records source/data hash, model hash, code hashes, dependency versions, split counts and split-index hash, feature order, imputation medians, supported ranges and default values. Local private split indices are saved separately for later evaluation; they are excluded from the SageMaker package. Loading checks the model hash and sklearn version before deserialization.

Successful responses include `status: ok`, contract version, model version, one permitted label per row and a warnings list. Invalid inputs produce structured field issues. Missing/unknown labels, empty lists, wrong lengths, malformed JSON, incompatible versions, endpoint unavailability and model failures produce technical errors, never a synthetic Poor label.

## Known uncertainties

The original source/distributor, license, currency, period/year and target construction are not established. Reference ranges are derived from training, not population bounds. The source can include repeated monthly observations; future temporal claims require a documented time axis. [Stage-two evaluation](STAGE-TWO.md) reports grouped comparison, calibration diagnostics, preprocessing sensitivity and historical strict-domain coverage (61.2%). That percentage is not the current app acceptance rate. Permitting exploration does not validate model reliability on arbitrary unfamiliar profiles.
