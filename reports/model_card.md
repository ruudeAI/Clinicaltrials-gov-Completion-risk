# Model Card: Clinical Trial Completion Risk Predictor

## Model Overview

| Field | Value |
|-------|-------|
| **Model Name** | Clinical Trial Completion Risk Predictor |
| **Version** | 1.0 (MVP) |
| **Type** | Binary Classification |
| **Algorithm** | Logistic Regression |
| **Framework** | scikit-learn |
| **Language** | Python 3.9+ |

---

## Intended Use

### Primary Use
Educational and research portfolio project demonstrating machine learning
with real-world clinical trial data.

### Intended Users
- Students learning machine learning
- Data scientists building portfolio projects
- Researchers exploring ClinicalTrials.gov data

### Use Cases
- Estimating completion probability for a drug-related clinical trial
- Exploring which trial characteristics are associated with completion risk
- Demonstrating an end-to-end ML pipeline with real API data

---

## NOT Intended Use

This model must **NOT** be used for:

- ❌ Making medical or healthcare decisions
- ❌ Advising patients on clinical trial enrollment
- ❌ Evaluating treatment safety or efficacy
- ❌ Predicting regulatory approval (FDA, EMA, etc.)
- ❌ Making investment or financial decisions
- ❌ Insurance underwriting or coverage decisions
- ❌ Legal proceedings or expert testimony
- ❌ Any purpose requiring clinical validation

---

## Data Source

| Field | Detail |
|-------|--------|
| **Source** | [ClinicalTrials.gov](https://clinicaltrials.gov/) API v2 |
| **Access** | Public REST API, no API key required |
| **Scope** | Drug-related clinical trials across 20 conditions |
| **Statuses Used** | COMPLETED, TERMINATED, WITHDRAWN, SUSPENDED |
| **Statuses Excluded** | RECRUITING, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, UNKNOWN, and other ongoing statuses |

No external datasets (AACT, PubMed, FDA, Kaggle) are used.

---

## Features

### Categorical (12)
- `study_type`, `phases`, `sponsor_class`, `enrollment_type`
- `intervention_types`, `allocation`, `intervention_model`
- `masking`, `primary_purpose`, `sex`, `healthy_volunteers`
- `search_query_source` (broad medical condition query source)

### Numeric (3)
- `enrollment_count`, `collaborator_count`, `location_count`

### Text (1)
- `combined_text` — concatenation of brief_summary, eligibility_criteria,
  and primary_outcome_measures, processed by TF-IDF (500 features)

---

## Target Variable

| Value | Label | Statuses |
|-------|-------|----------|
| 1 | Completed | COMPLETED |
| 0 | At Risk | TERMINATED, WITHDRAWN, SUSPENDED |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Train/test split | 80/20 stratified |
| Random seed | 42 |
| Class weighting | `balanced` |
| Text vectorizer | TF-IDF (500 max features, English stop words) |
| Categorical encoding | OneHotEncoder (unknown = ignore) |
| Missing values (numeric) | Median imputation |
| Missing values (categorical) | "UNKNOWN" constant |
| Max iterations | 1000 |

---

## Metrics

Model performance metrics are generated during training and saved to
`reports/results_summary.md`. Typical metrics include:

- **Accuracy** — Overall prediction correctness
- **Precision** — When the model says "completed", how often is it right?
- **Recall** — Of all actual completed trials, how many were identified?
- **F1-Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Model's ability to distinguish between the two classes

See `reports/results_summary.md` for actual values after training.

---

## Limitations

See [limitations.md](limitations.md) for the full list. Key points:

1. "Completed" ≠ clinically successful
2. "Terminated" ≠ medically dangerous
3. ClinicalTrials.gov metadata can be missing or inconsistent
4. Trained on drug trials only — may not generalize to non-drug interventions (devices, behavioral, etc.)
5. No temporal features (start date, planned duration)
6. Text features use simple TF-IDF, not deep NLP

---

## Ethical Considerations

1. **No medical claims** — This model does not evaluate treatment safety or
   efficacy and must not be presented as doing so.

2. **Bias in training data** — ClinicalTrials.gov is biased toward
   U.S.-registered, English-language, larger-sponsored trials. The model
   inherits these biases.

3. **Transparency** — All code, data sources, features, and limitations are
   publicly documented to enable scrutiny and reproducibility.

4. **Responsible framing** — All user-facing outputs include disclaimers
   clarifying the model's educational purpose and limitations.

---

## Citation

If referencing this project:

```
Clinical Trial Completion Risk Predictor.
Data source: ClinicalTrials.gov (https://clinicaltrials.gov/).
Educational portfolio project — not for clinical or regulatory use.
```
