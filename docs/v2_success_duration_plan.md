# Version 2 Plan: Clinical Trial Success and Duration Prediction

## Purpose

Version 2 should extend the current ClinicalTrials.gov drug-trial completion-risk predictor into a separate success and duration modeling track. The existing Version 1 model predicts operational completion versus termination/withdrawal/suspension. Version 2 should not replace that model. It should add two new prediction tasks:

1. Probability of clinical trial success
2. Trial duration in days

The safest architecture is to keep Version 1 intact and add a parallel `v2` pipeline with its own data outputs, model artifacts, reports, and optional dashboard view. This preserves the current completion-risk project while allowing success labels, ChEMBL enrichment, sponsor history, temporal validation, and duration regression to evolve independently.

## Current Repository Observations

The repository is a compact scikit-learn project with a working Version 1 pipeline:

- `src/clinicaltrials_api.py` fetches ClinicalTrials.gov API v2 data, flattens nested study JSON, filters drug interventions, and writes raw/sample CSV files.
- `src/preprocess.py` filters to known completion-risk statuses, creates `target_completed`, engineers Version 1 features, and writes the processed dataset.
- `src/train_model.py` compares several classifiers, tunes the completion-probability threshold, saves the best model, and writes `reports/results_summary.md`.
- `src/evaluate.py` loads the trained model and creates plots for confusion matrix, ROC curve, and top features.
- `src/predict.py` fetches a single NCT ID, prepares features, and predicts completion/risk probability.
- `app/streamlit_app.py` provides the current completion-risk dashboard.
- `reports/limitations.md` and `reports/model_card.md` already document the key caveat that completion is not clinical success.

Version 1 currently stores `start_date`, `primary_completion_date` may be requested in config comments, but the current `flatten_study()` does not preserve final `completion_date` or `primary_completion_date` columns in the flat output. Version 2 duration modeling will need explicit date extraction from the ClinicalTrials.gov status module.

## Files That Can Be Reused

- `src/clinicaltrials_api.py`
  - Reuse `safe_get()`, `fetch_studies()`, pagination/rate-limit patterns, and the general flattening approach.
  - Add Version 2-specific extraction in a new module rather than changing the current `flatten_study()` immediately.

- `src/config.py`
  - Reuse path conventions, API base URL, search query list, random seed, and trial scope constants.
  - Add V2 constants later in a separate section only after implementation begins.

- `src/preprocess.py`
  - Reuse ideas for missing-value handling, feature engineering structure, text combination, and status filtering.
  - Do not reuse `target_completed` as a success label.

- `src/train_model.py`
  - Reuse the scikit-learn `Pipeline` and `ColumnTransformer` pattern.
  - Reuse model comparison/reporting style, but implement separate V2 classification and regression trainers.

- `src/evaluate.py`
  - Reuse plotting patterns and feature-importance extraction.
  - Extend later with V2 metrics, phase-specific evaluation, PR-AUC, and regression plots.

- `src/predict.py`
  - Reuse single-NCT fetch and preparation patterns for a future V2 predictor.
  - Keep current output focused on completion risk until V2 is trained and validated.

- `reports/limitations.md` and `reports/model_card.md`
  - Reuse ethical framing and add separate V2 documentation later.

- `requirements.txt`
  - Existing dependencies are enough for a baseline V2 with ClinicalTrials.gov, ChEMBL REST calls, pandas, scikit-learn, matplotlib, seaborn, and Streamlit.
  - Add optional packages later only if justified.

## Files That Should Not Be Touched Initially

To avoid breaking Version 1, do not modify these during the first V2 implementation steps:

- `src/preprocess.py`
- `src/train_model.py`
- `src/evaluate.py`
- `src/predict.py`
- `app/streamlit_app.py`
- `models/drug_trial_completion_model.joblib`
- `data/processed/drug_trials_processed.csv`
- `reports/results_summary.md`
- Existing figures in `reports/figures/`

If changes are needed later, prefer additive compatibility changes. For example, create `src/v2_*` modules first, then optionally refactor shared helpers after both pipelines are stable.

## New Files and Modules to Create

Recommended V2 file layout:

```text
src/
  v2_config.py
  v2_clinicaltrials_api.py
  v2_chembl_enrichment.py
  v2_label_success.py
  v2_preprocess.py
  v2_features.py
  v2_sponsor_history.py
  v2_endpoint_features.py
  v2_train_success_model.py
  v2_train_duration_model.py
  v2_evaluate.py
  v2_predict.py

data/
  raw/
    v2_trials_raw.csv
  interim/
    v2_trials_with_chembl.csv
    v2_trials_with_labels.csv
  processed/
    v2_success_duration_dataset.csv
  cache/
    chembl_name_cache.json
    chembl_molecule_cache.json

models/
  v2_success_classifier.joblib
  v2_duration_regressor.joblib

reports/
  v2_success_duration_results.md
  v2_model_card.md
  figures/
    v2_success_roc_curve.png
    v2_success_pr_curve.png
    v2_success_top_features.png
    v2_duration_actual_vs_predicted.png
    v2_duration_residuals.png
```

Optional later:

```text
app/
  streamlit_app.py
```

The current dashboard can eventually gain a tab or sidebar mode for Version 2, but this should happen only after offline V2 models and reports are stable.

## Feature Areas

Version 2 should cover the assignment-required feature groups:

- Disease indication: `conditions`, query source, therapeutic area mapping, condition count, and text-derived indication groups.
- Molecule: normalized intervention/drug names, ChEMBL IDs, molecule synonyms, molecule type, max phase if available from ChEMBL.
- Drug modality using ChEMBL: small molecule, antibody, protein, oligonucleotide, cell therapy, gene therapy, vaccine/biologic-like categories where available.
- Sponsor history: historical sponsor trial counts and outcomes computed only from trials earlier than each prediction date.
- Phase: normalized phase, phase groups, and phase-specific models or evaluation.
- Trial design: allocation, intervention model, masking, primary purpose, randomization, blinding, arm/intervention counts.
- Enrollment size: enrollment count, transformed enrollment, enrollment per location, enrollment type.
- Endpoint type: safety, efficacy, survival, biomarker, response-rate flags from primary outcome measures.
- Top predictive features: model feature importance in reports, with SHAP deferred until useful.

## Success Target Definition Options

Clinical trial success is not directly available as a clean field in ClinicalTrials.gov. Version 2 should support multiple label strategies and document which one is active.

### Option A: Phase Progression Label

Definition:

- Phase 1 success: the same drug/disease/sponsor or drug/disease pair later appears in a Phase 2 trial.
- Phase 2 success: the same drug/disease/sponsor or drug/disease pair later appears in a Phase 3 trial.
- Phase 3 success: the same drug/disease pair later appears in Phase 4, has approval evidence if available, or has another documented later-stage success signal.

Benefits:

- Uses public ClinicalTrials.gov data.
- Aligns with CTOD's phase-aware idea that early-phase success means progression to the next phase.
- Avoids claiming that `COMPLETED` equals clinical success.

Challenges:

- Requires entity matching across trials.
- Matching disease names and drug names is noisy.
- Sponsor changes, licensing, mergers, and synonyms can cause false negatives.

Implementation approach:

- Create a trial entity key using ChEMBL ID when available, normalized intervention name otherwise, normalized indication, and phase.
- Sort trials by start date.
- For each trial, search only later trials for progression evidence.
- Store label provenance columns such as `success_label_source`, `progression_nct_id`, and `progression_phase`.

### Option B: CTOD Label If Allowed

Definition:

- Use CTOD labels as an additional data source only if licensing, access terms, and citation requirements allow it.
- Keep CTOD labels separate from ClinicalTrials.gov-derived labels.

Benefits:

- CTOD is designed around phase-aware trial outcome definitions.
- Provides stronger success labels than raw completion status.

Challenges:

- Requires explicit documentation of license/terms.
- May add reproducibility limitations if the data is not fully public or easy to obtain.

Implementation approach:

- Add a `data/external/` path only if allowed.
- Create `src/v2_label_success.py` support for a `--label-source ctod` mode.
- Add report language clearly stating CTOD was used as an additional data source.
- Keep a ClinicalTrials.gov-only baseline for comparison.

### Option C: Regulatory Approval Label If Available Later

Definition:

- Phase 3 success is linked to regulatory approval evidence from a public source available in a later project phase.

Benefits:

- Stronger endpoint for late-stage clinical success.

Challenges:

- Approval databases can be difficult to map to trial NCT IDs and molecules.
- Avoid paid/private sources and avoid DrugBank due to licensing concerns.
- Must prevent leakage by using approval only as a label, not as a feature.

Implementation approach:

- Defer until a public, license-compatible source is selected.
- Add label source provenance.
- Do not mix approval evidence into model features.

## Duration Target Definition

Primary target:

```text
observed_duration_days = completion_date - start_date
```

Requirements:

- Extract `startDateStruct.date` and `completionDateStruct.date` from ClinicalTrials.gov API v2.
- Use actual completion date if available.
- Restrict observed-duration training rows to trials with valid start and completion dates.
- Exclude negative, zero, or implausibly large durations after review.
- Keep `observed_duration_days` as the regression target only, never as a feature.

Planned duration:

```text
planned_duration_days = primary_completion_date - start_date
```

Use this only with caution:

- If the prediction setting is "at trial registration/start", planned primary completion date may be known and may be a legitimate feature.
- If predicting after trial completion or using updated registry data, planned dates may have been revised and can leak outcome information.
- Treat `planned_duration_days` as an optional feature behind a configuration flag and document the prediction-time assumption.

Recommended first version:

- Train duration regression to predict `observed_duration_days`.
- Do not include `completion_date`, `actual_duration`, or actual outcome fields as inputs.
- Include `planned_duration_days` only in an ablation report after confirming it is not derived from future updates.

## ChEMBL Integration

Do not use DrugBank. Use ChEMBL because it is public and appropriate for molecule enrichment.

Planned workflow:

1. Parse `intervention_names` from ClinicalTrials.gov and split pipe-delimited values.
2. Normalize names:
   - lowercase
   - strip dosage, route, punctuation, placebo phrases, and combination syntax where possible
   - keep original names for auditability
3. Query ChEMBL API molecule search endpoint by normalized name.
4. Select candidate match using exact synonym/name match first, then high-confidence fuzzy match if needed.
5. Store ChEMBL identifiers and match metadata:
   - `chembl_id`
   - `pref_name`
   - `molecule_type`
   - `max_phase`
   - `therapeutic_flag`
   - `match_score`
   - `match_method`
6. Derive modality features:
   - `modality_small_molecule`
   - `modality_antibody`
   - `modality_protein`
   - `modality_oligonucleotide`
   - `modality_cell_or_gene_therapy`
   - `modality_vaccine_or_biologic`
   - `modality_unknown`
7. Cache API results locally:
   - `data/cache/chembl_name_cache.json` for name-to-candidate mappings
   - `data/cache/chembl_molecule_cache.json` for ChEMBL molecule details
8. Never fail the pipeline because ChEMBL has no match. Use `UNKNOWN` and continue.

Important cautions:

- ChEMBL `max_phase` may be future-looking relative to a historical trial. For time-aware modeling, either exclude it or only use it in clearly marked non-temporal exploratory models.
- ChEMBL mappings can be wrong for combination therapies, generic names, and biologics.
- Molecule features must include match confidence and unknown indicators.

## Sponsor History Feature Engineering

Sponsor history should be computed in a time-aware way to avoid leakage. For each trial, use only trials with dates earlier than the prediction date, ideally `start_date`.

Required features:

- `sponsor_total_trials`
- `sponsor_completed_trials`
- `sponsor_failed_trials`
- `sponsor_success_rate`
- `sponsor_phase_specific_history`

Recommended expanded features:

- `sponsor_phase_total_trials`
- `sponsor_phase_success_rate`
- `sponsor_indication_total_trials`
- `sponsor_indication_success_rate`
- `sponsor_recent_5yr_trials`
- `sponsor_recent_5yr_success_rate`
- `sponsor_median_duration_days_prior`
- `sponsor_has_prior_trial`

Implementation notes:

- Normalize sponsor names before grouping.
- Use `lead_sponsor` as the first version; collaborators can be added later.
- Sort by `start_date`.
- For each row, compute cumulative history from prior rows only.
- For success labels based on phase progression, sponsor history should use labels available from previous trials only.
- For a new sponsor, impute zero-count features and global priors for rates.

## Endpoint Type Extraction

Endpoint type should be extracted from `primary_outcome_measures` and, later, full outcome descriptions/time frames if added to flattening.

Create binary flags:

- `endpoint_safety`
- `endpoint_efficacy`
- `endpoint_survival`
- `endpoint_biomarker`
- `endpoint_response_rate`

Initial keyword rules:

- Safety endpoint:
  - adverse event
  - adverse events
  - serious adverse event
  - safety
  - toxicity
  - dose limiting toxicity
  - tolerability
  - maximum tolerated dose
  - treatment-emergent adverse event

- Efficacy endpoint:
  - efficacy
  - clinical benefit
  - symptom improvement
  - disease activity
  - change from baseline
  - treatment effect

- Survival endpoint:
  - overall survival
  - progression-free survival
  - event-free survival
  - disease-free survival
  - relapse-free survival
  - survival
  - mortality

- Biomarker endpoint:
  - biomarker
  - pharmacodynamic
  - pharmacokinetic
  - immune response
  - viral load
  - tumor marker
  - lab value
  - cytokine

- Response-rate endpoint:
  - objective response rate
  - overall response rate
  - response rate
  - complete response
  - partial response
  - remission
  - RECIST

Add:

- `endpoint_type_count`
- `primary_endpoint_text_length`
- `has_primary_endpoint`

Use simple keyword extraction first. More advanced NLP can wait until the baseline is stable.

## Model Design

### Success Classification Model

Primary target:

- `target_success`, based on selected label strategy.

Baseline model candidates:

- Logistic Regression for interpretability.
- Random Forest as a robust nonlinear baseline.
- HistGradientBoosting or GradientBoosting for stronger tabular performance.

Feature groups:

- Drug/molecule features
- Disease/indication features
- Protocol/trial design features
- Sponsor history features
- Endpoint type features
- Phase features
- Text TF-IDF features from summaries, eligibility, and outcome measures

Phase-aware design:

- Train one global classifier with phase as a feature.
- Also report phase-specific metrics for Phase 1, Phase 2, and Phase 3.
- If enough data exists, train separate phase-specific models as an experiment.

### Duration Regression Model

Primary target:

- `observed_duration_days`.

Baseline model candidates:

- Ridge regression or ElasticNet for interpretable baseline.
- RandomForestRegressor.
- HistGradientBoostingRegressor.

Recommended transformations:

- Use `log1p(observed_duration_days)` as an optional target transform if duration is skewed.
- Report metrics both in original days and transformed space if applicable.

Feature groups:

- Same non-leaky feature groups as success classifier.
- Exclude success/progression/approval labels and final outcome status.
- Be cautious with planned duration as described above.

## Evaluation Design

Evaluate with both random and time-based splits:

- Random split:
  - Useful for quick comparison with Version 1 style.
  - Can be optimistic due to temporal leakage and duplicate/similar trials.

- Time-based split:
  - Recommended primary evaluation.
  - Sort by `start_date`.
  - Train on earlier trials and test on later trials.
  - Mirrors real prediction better and follows the clinical-trial-forecaster lesson.

- Phase-specific evaluation:
  - Report metrics separately for Phase 1, Phase 2, and Phase 3.
  - Avoid hiding weak phase performance behind global averages.

Classification metrics:

- ROC-AUC
- PR-AUC
- F1
- precision
- recall
- confusion matrix
- calibration curve later if practical

Regression metrics:

- MAE
- RMSE
- R2
- median absolute error
- optional percentage within 90/180/365 days

Reports should include:

- Overall random split metrics
- Overall time split metrics
- Phase-specific metrics
- Model comparison table
- Feature group ablation if time allows

## Explainability

Start with feature importance:

- For linear models, use coefficients after preprocessing.
- For tree models, use `feature_importances_` or permutation importance.
- Report top positive and negative predictors for success when available.
- Report top duration drivers for regression.

Add later if practical:

- SHAP for the best tree-based model.
- Per-trial explanation in the dashboard.

Required output:

- `reports/v2_success_duration_results.md`
- top predictive features table
- top predictive features plot

## Leakage Risks and Controls

Major leakage risks:

- Do not use actual completion outcome as an input feature.
- Do not use `overall_status` as an input for success or duration models.
- Do not use future trial information in sponsor history.
- Do not use future phase progression or approval evidence as features.
- Do not use completion date as a feature when predicting duration.
- Do not use ChEMBL `max_phase` in a historical time-based model unless it is proven to be known at prediction time.
- Do not let multiple rows for the same drug/disease pair leak across train/test without checking grouped evaluation.
- Do not use planned duration if registry dates were updated after the trial started unless the prediction-time assumption allows it.

Controls:

- Add a `prediction_date` column, initially equal to parsed `start_date`.
- Sort all temporal features by `prediction_date`.
- Compute sponsor history using only prior rows.
- For phase progression labels, use later trials only to create labels, never features.
- Keep label provenance separate from features.
- Maintain a feature allowlist for each model.
- Run a leakage audit before training that rejects forbidden columns.

## Step-by-Step Implementation Roadmap

### Step 1: Freeze and Document Version 1 Boundary

- Leave existing Version 1 scripts and model artifacts untouched.
- Create V2 modules and paths.
- Add V2 documentation that completion risk is not success prediction.

Difficulty: low.

### Step 2: Build V2 Data Extraction

- Create `src/v2_clinicaltrials_api.py`.
- Reuse API fetch/pagination logic.
- Extract all Version 1 fields plus:
  - `start_date`
  - `primary_completion_date`
  - `completion_date`
  - detailed outcome measures/time frames if available
  - intervention names and types
  - sponsor and collaborator details
- Save `data/raw/v2_trials_raw.csv`.

Difficulty: medium.

### Step 3: Add Duration Target

- Parse date fields.
- Create `observed_duration_days = completion_date - start_date`.
- Create optional `planned_duration_days = primary_completion_date - start_date`.
- Filter invalid durations.
- Save intermediate target diagnostics.

Difficulty: low to medium.

### Step 4: Add Endpoint Type Features

- Create `src/v2_endpoint_features.py`.
- Implement keyword-based endpoint flags.
- Validate on sample records manually.

Difficulty: low.

### Step 5: Add ChEMBL Enrichment

- Create `src/v2_chembl_enrichment.py`.
- Normalize intervention names.
- Query and cache ChEMBL results.
- Add molecule/modality features.
- Keep unknown and low-confidence matches explicit.

Difficulty: medium to high because name matching is noisy.

### Step 6: Create Phase Progression Success Labels

- Create `src/v2_label_success.py`.
- Normalize drug and disease keys.
- Use ChEMBL IDs when available.
- Create phase-aware progression labels.
- Add label provenance columns.
- Document limitations.

Difficulty: high.

### Step 7: Add Sponsor History Features

- Create `src/v2_sponsor_history.py`.
- Sort by prediction date.
- Compute prior-only sponsor totals, failures, successes, rates, and phase-specific history.
- Add tests or validation checks to ensure no future rows are included.

Difficulty: medium.

### Step 8: Build V2 Preprocessing Dataset

- Create `src/v2_preprocess.py` and `src/v2_features.py`.
- Combine cleaned trial metadata, duration target, success label, ChEMBL features, endpoint flags, sponsor history, and text features.
- Save `data/processed/v2_success_duration_dataset.csv`.

Difficulty: medium.

### Step 9: Train Baseline Success Classifier

- Create `src/v2_train_success_model.py`.
- Train Logistic Regression, Random Forest, and gradient boosting baselines.
- Evaluate random and time-based splits.
- Report phase-specific metrics.

Difficulty: medium.

### Step 10: Train Baseline Duration Regressor

- Create `src/v2_train_duration_model.py`.
- Train Ridge/ElasticNet, RandomForestRegressor, and HistGradientBoostingRegressor.
- Evaluate random and time-based splits.
- Report phase-specific regression metrics.

Difficulty: medium.

### Step 11: V2 Evaluation and Reports

- Create `src/v2_evaluate.py`.
- Generate:
  - ROC curve
  - PR curve
  - success confusion matrix
  - top features
  - actual versus predicted duration plot
  - residual plot
- Write `reports/v2_success_duration_results.md`.
- Write `reports/v2_model_card.md`.

Difficulty: medium.

### Step 12: Optional Dashboard Integration

- Add a Version 2 tab to `app/streamlit_app.py` only after models are validated.
- Clearly separate Version 1 completion risk from Version 2 success/duration.
- Show disclaimers that success is an inferred public-data label, not a clinical or regulatory truth.

Difficulty: medium.

## Safest Order of Implementation

The safest order is:

1. Add V2 files and paths without touching Version 1.
2. Extract missing date fields and build duration target.
3. Add endpoint features.
4. Add time-based evaluation scaffolding.
5. Train duration regression baseline.
6. Add ChEMBL enrichment with caching.
7. Add sponsor history with strict temporal controls.
8. Add phase progression success labels.
9. Train success classifier.
10. Add reports and top feature outputs.
11. Consider dashboard integration.

This order gives quick progress on the duration requirement while delaying the hardest and riskiest pieces: molecule matching, sponsor history leakage controls, and inferred success labels.

## Estimated Difficulty

- Duration regression baseline: medium. The biggest blocker is extracting reliable completion dates.
- Endpoint feature extraction: low to medium. Keyword rules are simple but imperfect.
- ChEMBL molecule/modality enrichment: medium to high. API usage is straightforward, but matching clinical trial intervention names to molecules is noisy.
- Sponsor history: medium. The feature logic is simple, but temporal leakage prevention is critical.
- Phase progression success labels: high. This is the core scientific challenge because success is inferred, not directly observed.
- Phase-specific evaluation: medium. Straightforward technically, but small sample sizes may limit reliability.
- Explainability: low for basic feature importance, medium for SHAP.
- Dashboard integration: medium, and should be last.

## Recommended First Milestone

Build a ClinicalTrials.gov-only V2 baseline:

- Extract dates.
- Create `observed_duration_days`.
- Add endpoint flags.
- Add phase/design/enrollment/disease features.
- Train a duration regressor.
- Create a simple phase-progression success label without ChEMBL first, using normalized intervention names.
- Evaluate with both random and time-based splits.

Then improve with:

- ChEMBL molecule/modality enrichment.
- Better sponsor history.
- Stronger phase progression matching.
- Optional CTOD labels if licensing allows.

## Documentation Requirements

Every V2 report should explicitly state:

- The model predicts inferred trial success probability, not treatment efficacy.
- The success label source is phase progression, CTOD, or regulatory evidence.
- ChEMBL is used for public molecule/modality enrichment.
- DrugBank is not used due to licensing concerns.
- Paid/private datasets are not used.
- Time-based validation is the preferred estimate of future performance.
- Sponsor history is computed using only information available before each trial's prediction date.

