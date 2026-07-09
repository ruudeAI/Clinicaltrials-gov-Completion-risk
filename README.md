# ClinicalTrials.gov Trial Risk, Success, and Duration Predictor

This repository is a machine learning portfolio project built on public clinical trial metadata.

It contains two related but separate modeling tracks:

- **Version 1:** predicts clinical trial completion vs. termination/withdrawal/suspension risk.
- **Version 2:** predicts a conservative clinical trial success-proxy probability and trial duration in days.

Version 1 remains intact. Version 2 is implemented as a separate pipeline with its own extraction, preprocessing, enrichment, labeling, modeling, and reporting files.

## Important Disclaimer

This project is for educational and research portfolio purposes only.

It does **not** provide medical, clinical, regulatory, investment, or patient-care advice. The models are trained on public registry metadata and should not be used for healthcare decisions, trial enrollment decisions, regulatory decisions, or financial decisions.

## Version 1: Completion Risk Model

Version 1 predicts whether a drug-related clinical trial is likely to complete or be at risk of non-completion.

Target:

- `1`: `COMPLETED`
- `0`: `TERMINATED`, `WITHDRAWN`, or `SUSPENDED`

Version 1 uses ClinicalTrials.gov metadata such as phase, sponsor class, enrollment, trial design, intervention type, eligibility text, summary text, outcome measures, and country/location counts.

Version 1 is still available through the original scripts and Streamlit dashboard.

## Version 2: Success and Duration Models

Version 2 addresses the newer assignment goal: build a Clinical Trial Success and Duration Prediction Model.

It predicts:

1. A conservative success-proxy label based on phase progression.
2. Trial duration in days using `observed_duration_days = completion_date - start_date`.

The Version 2 success label is **not** confirmed clinical efficacy. It is not regulatory approval. It is an inferred public-data proxy based on whether the same drug/intervention later appears in a more advanced phase.

## Narayanan Feature Coverage

| Required feature area | Version 2 implementation |
|---|---|
| Disease indication | `conditions_normalized`, `condition_count`, `search_query_source` |
| Molecule name | `molecule_name`, `preferred_name`, `chembl_id` |
| Drug modality from ChEMBL | `drug_modality`, derived from ChEMBL `molecule_type` and name/type rules |
| Sponsor history | Prior-only sponsor history features computed by trial start date |
| Phase | `phase_normalized` |
| Trial design | `allocation`, `intervention_model`, `masking`, `primary_purpose`, `study_type` |
| Enrollment size | `enrollment_count` |
| Endpoint type | safety, efficacy, survival, biomarker, and response-rate endpoint flags |
| Top predictive features | Included in `reports/v2_success_results.md` and `reports/v2_duration_results.md` |

## Data Sources

Used:

- [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api)
- [ChEMBL API](https://www.ebi.ac.uk/chembl/)

Not used:

- DrugBank, because of licensing concerns
- paid/private datasets
- CTOD labels, unless added later with explicit licensing documentation
- regulatory approval datasets, not yet integrated

## Current Version 2 Baseline Results

These are the baseline results after running:

```powershell
venv\Scripts\python.exe scripts\run_v2_pipeline.py --max-trials 3000
```

### Dataset from Latest Run
* **Total drug-related trials:** 912
* **Valid duration rows:** 879
* **Success labels (for success classifier):** 92 success / 92 failure (184 total labeled rows)
* **Excluded success labels (unknown phase progression):** 728
* **ChEMBL matched rows:** 697

### Success Classification (Time-Based Split)
* **Best Model:** `LogisticRegression`
* **ROC-AUC:** `0.7401`
* **PR-AUC:** `0.4453`
* **F1-Score:** `0.3871`

### Duration Regression (Time-Based Split)
* **Best Model:** `GradientBoostingRegressor`
* **Mean Absolute Error (MAE):** `488.89 days`
* **Root Mean Squared Error (RMSE):** `639.06 days`
* **Coefficient of Determination ($R^2$):** `0.0162`

*Note:* Time-based validation is used because it is more realistic than random splits (it evaluates on trials that started chronologically later than the training set).

## Version 2 Limitations

- The success label is a conservative phase-progression proxy.
- It is not confirmed clinical efficacy.
- It is not regulatory approval prediction yet.
- Phase progression can reflect business, licensing, or sponsor strategy rather than trial-level efficacy.
- Duration prediction is an estimate from public registry metadata.
- ClinicalTrials.gov dates can be missing, partial, delayed, or revised.
- ChEMBL matching has unknown/no-match cases and should be audited for ambiguous interventions.
- The current success classifier has a limited number of labeled rows because many trials remain unknown under conservative labeling.
- The models are not medical advice, regulatory advice, or investment advice.

## Repository Structure

```text
clinicaltrials-gov-completion-risk-predictor/
|-- README.md
|-- requirements.txt
|-- app/
|   `-- streamlit_app.py                    # Version 1 dashboard
|-- data/
|   |-- raw/
|   |   |-- drug_trials_raw.csv             # Version 1 raw data
|   |   `-- v2_trials_raw.csv               # Version 2 raw data
|   |-- interim/
|   |   |-- v2_trials_with_success_labels.csv
|   |   `-- v2_trials_with_chembl.csv
|   |-- processed/
|   |   |-- drug_trials_processed.csv       # Version 1 processed data
|   |   |-- v2_success_duration_dataset.csv
|   |   `-- v2_modeling_dataset.csv
|   `-- cache/
|       `-- chembl_cache.csv
|-- docs/
|   |-- v2_success_duration_plan.md
|   |-- success_label_strategy.md
|   `-- v2_checkpoint_after_prompts_1_to_5.md
|-- models/
|   |-- drug_trial_completion_model.joblib  # Version 1 model
|   |-- v2_duration_regressor.joblib
|   `-- v2_success_classifier.joblib
|-- reports/
|   |-- results_summary.md                  # Version 1 results
|   |-- model_card.md                       # Version 1 model card
|   |-- limitations.md                      # Version 1 limitations
|   |-- v2_duration_results.md
|   |-- v2_success_results.md
|   |-- v2_model_card.md
|   |-- v2_limitations.md
|   `-- figures/
|-- scripts/
|   `-- run_v2_pipeline.py
|-- src/
|   |-- clinicaltrials_api.py               # Version 1 extraction
|   |-- preprocess.py                       # Version 1 preprocessing
|   |-- train_model.py                      # Version 1 training
|   |-- evaluate.py                         # Version 1 evaluation
|   |-- predict.py                          # Version 1 single-trial prediction
|   |-- v2_clinicaltrials_api.py
|   |-- v2_preprocess.py
|   |-- success_labels.py
|   |-- chembl_api.py
|   |-- v2_enrich_chembl.py
|   |-- v2_sponsor_history.py
|   |-- v2_build_modeling_dataset.py
|   |-- train_duration_model.py
|   `-- train_success_model.py
`-- tests/
    `-- test_chembl_api.py
```

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run Version 1

Run the original completion-risk pipeline:

```powershell
venv\Scripts\python.exe src\clinicaltrials_api.py
venv\Scripts\python.exe src\preprocess.py
venv\Scripts\python.exe src\train_model.py
venv\Scripts\python.exe src\evaluate.py
```

Predict one trial by NCT ID:

```powershell
venv\Scripts\python.exe src\predict.py
```

Launch the dashboard:

```powershell
venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

## How to Run Version 2

Run the full Version 2 pipeline:

```powershell
venv\Scripts\python.exe scripts\run_v2_pipeline.py --max-trials 3000
```

Rerun processing/modeling without downloading new ClinicalTrials.gov data:

```powershell
venv\Scripts\python.exe scripts\run_v2_pipeline.py --skip-fetch
```

Run individual V2 steps:

```powershell
venv\Scripts\python.exe src\v2_clinicaltrials_api.py --max-trials 3000
venv\Scripts\python.exe src\v2_preprocess.py
venv\Scripts\python.exe src\success_labels.py
venv\Scripts\python.exe src\v2_enrich_chembl.py
venv\Scripts\python.exe src\v2_build_modeling_dataset.py
venv\Scripts\python.exe src\train_duration_model.py
venv\Scripts\python.exe src\train_success_model.py
```

## Key Version 2 Reports

- `reports/v2_duration_results.md`
- `reports/v2_success_results.md`
- `docs/v2_success_duration_plan.md`
- `docs/success_label_strategy.md`

## Future Improvements

- Improve ChEMBL matching with synonym-aware and combination-therapy logic.
- Add stricter disease-indication matching for success labels.
- Add public regulatory approval evidence for Phase 3 labels if a license-compatible source is selected.
- Evaluate CTOD labels only if licensing allows and documentation is clear.
- Increase dataset size and improve label coverage.
- Add grouped validation by molecule or drug family.
- Add phase-specific models once there are enough labeled examples.
- Add model calibration for success probabilities.
- Add SHAP or permutation importance once the feature set stabilizes.
- Keep Version 1 completion-risk predictions separate from Version 2 success/duration predictions.

## Responsible Use

This project demonstrates machine learning with public clinical trial metadata. It is not validated for clinical, regulatory, investment, legal, or patient-care use.
