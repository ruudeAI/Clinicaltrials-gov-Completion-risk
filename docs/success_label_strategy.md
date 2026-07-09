# Success Label Strategy for Version 2

## Goal

Define a defensible `target_success` label for clinical trial success prediction without confusing operational completion with clinical success. This note is a strategy document only. It does not implement the final label-generation code.

## Available Columns

The current repository datasets provide these useful columns for label strategy:

### Trial identity and timing

- `nct_id`
- `start_date`
- `start_year` in the processed dataset

These are necessary for temporal ordering. `start_date` should be the preferred prediction date.

### Drug or intervention

- `intervention_names`
- `intervention_types`

These can support same-drug progression matching. However, they are raw ClinicalTrials.gov strings and may include aliases, combinations, dose descriptions, or inconsistent naming. ChEMBL enrichment from `src/chembl_api.py` should later improve matching by adding `chembl_id` and `preferred_name`.

### Disease indication

- `conditions`
- `search_query_source`
- `therapeutic_area_group` in the processed dataset

These can support same-drug/same-disease progression matching. `conditions` is more trial-specific than `search_query_source`, while `search_query_source` is only the query that found the trial.

### Phase

- `phases`

This is the key column for phase progression labels. Values may include combined phases such as Phase 1/2 or Phase 2/3, so a normalization step is required.

### Outcome/status context

- `overall_status`
- `target_completed` in the processed Version 1 dataset

These can help identify trials that are operationally complete or failed, but they should not directly define clinical success. `COMPLETED` is not the same as efficacy or regulatory success.

### Supporting metadata

- `sponsor_name`
- `sponsor_class`
- `enrollment_count`
- `study_type`
- `allocation`
- `intervention_model`
- `masking`
- `primary_purpose`
- `primary_outcome_measures`
- `brief_summary`
- `eligibility_criteria`

These are useful features later, but should not be used to define the label except for audit/context.

### Missing for current label strategy

The current flattened raw dataset does not include:

- `completion_date`
- `primary_completion_date`
- approval evidence
- posted results or endpoint success
- ChEMBL IDs
- CTOD labels

Because of this, the first label strategy should use phase progression where possible and mark uncertain rows as unknown.

## Proposed Phase Progression Strategy

The baseline `target_success` should be inferred from later phase progression evidence. The core idea: if the same molecule/drug appears in a later phase after the current trial, then the earlier trial likely produced enough evidence to continue development.

### Matching key

Use the best available drug identity:

1. Prefer `chembl_id` after ChEMBL enrichment exists.
2. Otherwise use a normalized `intervention_names` value.
3. If multiple interventions exist, split them and label only when at least one active drug has clear progression evidence.

Use disease indication as an optional stricter key:

- Strict strategy: same drug plus same or similar `conditions`.
- Broad strategy: same drug progression regardless of condition.

Recommended first implementation:

- Produce both diagnostics, but use same-drug plus same/similar condition for the main label when possible.
- Fall back to same-drug-only only when condition matching is too sparse, and document that choice.

### Phase 1

Label Phase 1 trial as success (`target_success = 1`) if:

- the same molecule/drug appears later in a Phase 2, Phase 2/3, Phase 3, or Phase 4 trial, and
- the later trial has a `start_date` after the Phase 1 trial's `start_date`.

Label as failure (`target_success = 0`) only with caution:

- no later progression evidence exists after a sufficient follow-up window, and
- the trial is not too recent to have had a chance to progress.

Otherwise mark unknown.

### Phase 2

Label Phase 2 trial as success (`target_success = 1`) if:

- the same molecule/drug appears later in Phase 3, Phase 2/3, or Phase 4, and
- the later trial starts after the Phase 2 trial.

Label as failure only after a sufficient follow-up window and no later progression evidence. Otherwise mark unknown.

### Phase 3

Phase 3 is more difficult because progression to another trial is not always the right success definition.

Possible success evidence:

- same molecule/drug appears later in Phase 4
- public regulatory approval evidence exists
- CTOD or another allowed external label indicates success

Current recommendation:

- If later Phase 4 exists, label Phase 3 as success.
- If approval evidence is unavailable, do not assume Phase 3 failure only because Phase 4 is absent.
- Mark Phase 3 rows as unknown unless later Phase 4, allowed CTOD label, or later public approval evidence exists.

## Unknown and Excluded Rows

Rows should be excluded from supervised success-label training when:

- phase is missing, unknown, not applicable, or not Phase 1/2/3-like
- `intervention_names` is missing or cannot be normalized
- trial has only placebo/control interventions after filtering
- `start_date` is missing or invalid
- the trial is too recent to evaluate progression
- phase is Phase 3 and no Phase 4 or approval evidence is available
- multiple-drug combination makes molecule attribution ambiguous
- label source conflicts across progression, CTOD, or approval evidence
- no confident same-drug/same-condition matching is available

Keep these rows in a separate unlabeled or prediction-only dataset if useful, but do not train `target_success` on them.

## Avoiding Leakage

Leakage controls are mandatory:

- Use only later trials to create the label, never as input features.
- Do not include `progression_nct_id`, later phase evidence, CTOD outcome, approval status, or label provenance as model features.
- Do not use future sponsor history when creating sponsor features.
- Do not use `overall_status` as a direct success feature if the prediction setting is trial-start prediction.
- Do not use ChEMBL `max_phase` as a feature for historical time-based evaluation unless it is proven to be known at prediction time.
- In time-based validation, train on earlier trials and test on later trials.
- Compute all aggregate features, including sponsor history, using only information available before the trial prediction date.
- Trials with the same `start_date` should not provide progression evidence or sponsor history to each other.

Label creation may look into the future because labels are outcomes, but future-derived label evidence must be stored separately from features.

## Known Limitations

- Phase progression is an imperfect proxy for clinical success.
- A drug may progress because of business strategy, licensing, biomarker subgroup results, or external evidence rather than the specific trial.
- A successful trial may not lead to a later registered trial in ClinicalTrials.gov.
- Disease condition matching is noisy because condition strings are not standardized.
- Drug matching is noisy before ChEMBL IDs are available.
- Combination therapies can make it unclear which molecule caused progression.
- Sponsor changes and asset licensing can hide true progression.
- Phase 3 success usually requires approval or endpoint evidence, neither of which is currently available in the existing flattened dataset.
- Recent trials may be falsely labeled as failures if follow-up windows are too short.
- ClinicalTrials.gov registration practices vary across sponsors, diseases, countries, and time.

## CTOD Labels

CTOD labels should be used only if licensing, access terms, and citation requirements allow it.

Recommended approach if allowed:

- Treat CTOD as an additional documented label source, not a silent replacement.
- Preserve `success_label_source`, such as `phase_progression`, `ctod`, or `approval`.
- Compare a ClinicalTrials.gov-only phase progression baseline against a CTOD-labeled version.
- Clearly document CTOD use in the model card and results report.

If CTOD is not allowed or cannot be redistributed:

- Keep the primary baseline ClinicalTrials.gov-only.
- Do not require CTOD for reproducibility.

## Recommended First Label Policy

For the first implementation:

1. Normalize phase and drug names.
2. Add ChEMBL IDs when available.
3. Create phase progression labels for Phase 1 and Phase 2.
4. Label Phase 3 success only when later Phase 4 evidence exists.
5. Mark Phase 3 rows unknown when approval evidence is unavailable.
6. Exclude missing-date, missing-drug, ambiguous-combination, and too-recent rows.
7. Store label provenance and matching confidence.
8. Train only on rows with confident `target_success` labels.

