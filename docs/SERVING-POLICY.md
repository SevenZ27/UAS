# Serving policy — 21 October 2026

The public app prioritizes completing an exploratory prediction while making unfamiliar inputs visible. Model weights, preprocessing, labels, benchmark scores and the frozen evaluation are unchanged.

## Three different boundaries

| Boundary | Current behavior |
| --- | --- |
| Request structure and basic schema | Reject malformed/missing keys, non-finite numbers, invalid signs, fractions for counts, invalid binary status and schema violations such as Age above 120. Age is required. |
| Training reference ranges in metadata | Accept otherwise valid values, display the category and warn that reliability for that profile is untested. These observed ranges are not population limits. |
| Saved IQR preprocessing fences | Apply the original fitted pipeline. For an unfamiliar input, disclose whether it was clipped and the actual value used before classification. Do not silently substitute the observed minimum/maximum. |

Nullable fields can remain unknown. The saved training median fills them and the response warns about imputation. The UI preserves submitted values in the downloadable JSON, even if the pipeline transforms them internally.

`credit_score.serving_policy.ExplorationService` calls the original `Pipeline.predict`. In-range requests delegate to the original service and retain identical responses. `app_streamlit.py` and `inference.py` load this wrapper. `credit_score.service.CreditScoreService` retains strict range behavior only to reproduce the frozen benchmark's historical coverage analysis. No transforms are fitted on incoming requests.

A general warning appears beside the category. Feature-specific details are expandable and identify the submitted value, reference range and any IQR clipping. A warning means that suitability is not established; it does not claim that every unfamiliar prediction is certainly wrong. Probabilities are not presented as individual confidence.

The frozen 61.2% strict-domain coverage is not a measurement of current app acceptance. The historical full-holdout score remains a same-dataset benchmark; it is not a new estimate for arbitrary outside-range values. Allowing exploration must not be described as new validation, retraining or loan-decision support.

## Verification

Tests compare the wrapper against the exact original pipeline, preserve in-range response parity, reject schema-invalid requests, identify affected batch rows, and compare fresh-process packaged hooks. Browser checks enter Age 70, obtain a category with warnings, retain the original submitted value in JSON, and separately reject Age 999. The archive, evaluated model hash and frozen source/protocol hashes remain unchanged.
