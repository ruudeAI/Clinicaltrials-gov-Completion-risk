# V2 Checkpoint After Prompts 1-5

## Scope

This checkpoint reviews the repository after the Version 2 planning, scaffolding, ChEMBL enrichment, sponsor history, and success-label strategy work. No models were trained during this checkpoint.

## Files Created or Modified

New documentation:

- `docs/v2_success_duration_plan.md`
- `docs/success_label_strategy.md`
- `docs/v2_checkpoint_after_prompts_1_to_5.md`
- `reports/v2_model_card.md`
- `reports/v2_limitations.md`

New Version 2/scaffold source files:

- `src/chembl_api.py`
- `src/success_labels.py`
- `src/sponsor_features.py`
- `src/endpoint_features.py`
- `src/train_success_model.py`
- `src/train_duration_model.py`
- `src/explain_features.py`
- `src/v2_sponsor_history.py`

New validation file:

- `tests/test_chembl_api.py`

Existing tracked Version 1 files are not shown as modified by `git status --short`.

## Version 1 Status

No Version 1 completion-risk pipeline files were modified:

- `src/clinicaltrials_api.py`
- `src/preprocess.py`
- `src/train_model.py`
- `src/evaluate.py`
- `src/predict.py`
- `app/streamlit_app.py`
- existing reports, figures, data, and model paths

The Version 1 pipeline remains intact. The new files are not imported by Version 1 modules, and no preprocessing, training, prediction, or dashboard behavior has been connected to V2 yet.

## V2 Isolation

The V2 work is isolated:

- ChEMBL lookup lives only in `src/chembl_api.py`.
- Sponsor history lives only in `src/v2_sponsor_history.py`.
- V2 model-training placeholders are separate from `src/train_model.py`.
- V2 documentation is separate from existing Version 1 model-card/results files.
- The ChEMBL tests use temporary cache files and mocked API responses.

One consistency note: there is both `src/sponsor_features.py` as an early placeholder and `src/v2_sponsor_history.py` as the implemented sponsor-history module. This is not a bug, but later cleanup should choose one naming convention. Recommendation: keep `src/v2_sponsor_history.py` as the active module because it clearly marks the V2 boundary, and either remove or repurpose `src/sponsor_features.py` later.

## ChEMBL Review

`src/chembl_api.py` has safe behavior:

- Uses public ChEMBL molecule search API.
- `lookup_drug(...)` returns exactly the public fields:
  - `original_drug_name`
  - `chembl_id`
  - `preferred_name`
  - `molecule_type`
  - `max_phase`
  - `first_match_confidence`
  - `match_notes`
- Empty input returns `UNKNOWN` without calling the API.
- No-match responses return `UNKNOWN` with clear notes.
- Network/API/JSON/unexpected errors are caught and converted to `UNKNOWN` rows.
- Repeated lookups use a CSV cache.
- The default cache path is only `data/cache/chembl_cache.csv`.
- Validation did not create the real cache file.

Important modeling caveat remains documented: ChEMBL `max_phase` can be future-looking and should not be used in time-aware models until the prediction-time policy is settled.

## Sponsor History Review

`src/v2_sponsor_history.py` is designed to avoid leakage:

- Uses `start_date` as the preferred prediction date, with `start_year` fallback.
- Normalizes sponsor names before grouping.
- Sorts by prediction date.
- Computes features for all rows on a date before updating cumulative sponsor history with that date's rows.
- Therefore, future rows and same-date rows are excluded from each trial's history.
- Adds validation checks that recompute prior sponsor counts and phase counts using only earlier dates.
- Supports optional `target_success` features only if that column exists.
- Uses neutral rate priors for sponsors with no history.

The demo run passed and showed same-date rows do not count each other.

## Success Label Strategy Review

`docs/success_label_strategy.md` clearly defines:

- Available columns that can help create labels.
- Phase progression labels:
  - Phase 1 success if the same molecule/drug appears later in Phase 2 or higher.
  - Phase 2 success if the same molecule/drug appears later in Phase 3 or higher.
  - Phase 3 success only with later Phase 4, allowed CTOD label, or later public approval evidence; otherwise unknown.
- How to avoid leakage.
- Known limitations.
- Unknown/excluded rows.
- CTOD label option if licensing and documentation allow.
- Regulatory approval option for later Phase 3 labeling.

The document correctly avoids treating `overall_status = COMPLETED` as clinical success.

## Syntax, Imports, and Path Checks

Full Python syntax check passed using the repository virtual environment:

```powershell
venv\Scripts\python.exe -m py_compile app\streamlit_app.py src\__init__.py src\clinicaltrials_api.py src\config.py src\preprocess.py src\train_model.py src\evaluate.py src\predict.py src\test_accuracy.py src\chembl_api.py src\endpoint_features.py src\explain_features.py src\sponsor_features.py src\success_labels.py src\train_duration_model.py src\train_success_model.py src\v2_sponsor_history.py tests\__init__.py tests\test_chembl_api.py
```

ChEMBL mocked validation passed:

```powershell
venv\Scripts\python.exe tests\test_chembl_api.py
```

Result:

- 5 tests passed.
- No live network was required.
- No real `data/cache/chembl_cache.csv` was written.

No missing imports, syntax errors, or obvious path issues were found.

## Rename Recommendations

No rename is required immediately.

Recommended later cleanup:

- Keep `src/v2_sponsor_history.py` as the implemented sponsor-history module.
- Either remove, rename, or fold `src/sponsor_features.py` into `src/v2_sponsor_history.py` once V2 architecture is finalized.
- Consider a consistent `v2_*.py` convention for all V2 modules before connecting them to preprocessing or training.

## Recommended Next Step

Next implementation step:

Create a V2 data extraction/preprocessing module that produces a separate processed V2 dataset without touching Version 1. It should:

1. Extract or preserve required date fields, especially `start_date`, `primary_completion_date`, and `completion_date` if available from ClinicalTrials.gov.
2. Normalize phase and intervention names.
3. Add ChEMBL enrichment in a controlled optional step.
4. Add endpoint type features.
5. Add sponsor history features from `src/v2_sponsor_history.py`.
6. Prepare for success-label generation, but keep the first label implementation separate and auditable.

Do not train success or duration models until the V2 dataset, label provenance, and leakage checks are in place.

