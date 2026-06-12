# ClinicalTrials.gov Trial Completion Risk Predictor

A machine learning project that uses [ClinicalTrials.gov](https://clinicaltrials.gov/) API v2 data to estimate whether a drug-related clinical trial is likely to **complete** or be **at risk** of termination, withdrawal, or suspension.

> **⚠️ DISCLAIMER**
>
> This project is for **educational and research portfolio purposes only**.
> It does **not** provide medical, clinical, regulatory, investment, or
> patient-care advice. See [Ethical Considerations](#ethical-considerations)
> and [Limitations](reports/limitations.md) for details.

---

## Problem Statement

Clinical trials are expensive, time-consuming, and have a significant failure rate. According to published research, a substantial percentage of registered trials never reach completion — they are terminated, withdrawn, or suspended for various reasons including low enrollment, funding issues, safety concerns, or organizational changes.

**Can we use publicly available trial metadata to estimate which trials are more likely to complete?**

This project builds a binary classification model that answers this question using only structured data from ClinicalTrials.gov.

---

## Data Source

| **Source** | [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) |
| **Scope** | Drug-related clinical trials across 20 conditions (cancer, diabetes, heart disease, covid, etc.) |
| **Access** | Public REST API — no API key required |
| **Format** | JSON responses flattened to CSV |

### Why ClinicalTrials.gov?

- It is the **largest public registry** of clinical trials worldwide
- Data is **free, structured, and machine-readable** via a REST API
- It contains rich metadata: study design, sponsors, phases, enrollment, eligibility, outcomes
- No registration, API key, or data-use agreement is required
- It is maintained by the U.S. National Library of Medicine (NLM)

**This project uses ClinicalTrials.gov exclusively.** No AACT, PubMed, FDA, Kaggle, or commercial datasets are used.

---

## Target Variable

We create a binary label from the trial's `overall_status` field:

| Label | Status | Meaning |
|-------|--------|---------|
| **1** (Completed) | `COMPLETED` | Trial finished as planned |
| **0** (At Risk) | `TERMINATED` | Stopped early (various reasons) |
| | `WITHDRAWN` | Pulled before enrolling participants |
| | `SUSPENDED` | Temporarily halted |

### Excluded Statuses

The following statuses are **excluded from training** because their final outcome is not yet known:

- `RECRUITING` — currently enrolling
- `NOT_YET_RECRUITING` — approved but not started
- `ACTIVE_NOT_RECRUITING` — ongoing, enrollment closed
- `ENROLLING_BY_INVITATION` — by invitation only
- `UNKNOWN` — status not verified recently
- `AVAILABLE`, `NO_LONGER_AVAILABLE`, `TEMPORARILY_NOT_AVAILABLE`
- `APPROVED_FOR_MARKETING`

Training on these would be like grading an exam that hasn't been taken yet.

---

## Features Used

### Categorical Features (12)

| Feature | Description |
|---------|-------------|
| `study_type` | Interventional, observational, etc. |
| `phases` | Phase 1, 2, 3, 4, or combinations |
| `sponsor_class` | Industry, NIH, Other, etc. |
| `enrollment_type` | Actual or estimated enrollment |
| `intervention_types` | Drug, biological, device, procedure, etc. (Filtered to keep only trials with DRUG) |
| `allocation` | Randomized or non-randomized |
| `intervention_model` | Parallel, crossover, single group, etc. |
| `masking` | None, single, double, triple, quadruple |
| `primary_purpose` | Treatment, prevention, diagnostic, etc. |
| `sex` | All, male only, female only |
| `healthy_volunteers` | Whether healthy participants are accepted |
| `search_query_source` | Broad medical condition query source (representing the query that retrieved the trial, not necessarily the official category) |

### Numeric Features (3)

| Feature | Description |
|---------|-------------|
| `enrollment_count` | Number of planned/actual participants |
| `collaborator_count` | Number of collaborating organizations |
| `location_count` | Number of trial sites |

### Text Feature (1)

| Feature | Description |
|---------|-------------|
| `combined_text` | Concatenation of brief_summary + eligibility_criteria + primary_outcome_measures, processed by TF-IDF (500 features, English stop words removed) |

---

## Machine Learning Approach

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Algorithm** | Logistic Regression | Interpretable, good baseline for binary classification |
| **Text Processing** | TF-IDF Vectorizer | Captures word importance without deep learning complexity |
| **Categorical Encoding** | OneHotEncoder | Standard approach for nominal categories |
| **Missing Values** | Median (numeric), "UNKNOWN" (categorical) | Simple, robust imputation |
| **Class Imbalance** | `class_weight='balanced'` | Prevents model from ignoring the minority class |
| **Pipeline** | scikit-learn `Pipeline` + `ColumnTransformer` | Bundles preprocessing + model for consistent predictions |

---

## Project Architecture

```
clinicaltrials-gov-completion-risk-predictor/
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← Git ignore rules
│
├── src/                               ← Source code
│   ├── __init__.py                    ← Package init
│   ├── config.py                      ← Centralized configuration
│   ├── clinicaltrials_api.py          ← Step 1: Fetch data from API
│   ├── preprocess.py                  ← Step 2: Clean & label data
│   ├── train_model.py                 ← Step 3: Train ML model
│   ├── evaluate.py                    ← Step 4: Generate plots
│   └── predict.py                     ← Step 5: Predict by NCT ID
│
├── app/
│   └── streamlit_app.py               ← Interactive web dashboard
│
├── data/
│   ├── raw/                           ← Raw API data (git-ignored: drug_trials_raw.csv)
│   ├── processed/                     ← Cleaned dataset (git-ignored: drug_trials_processed.csv)
│   └── sample/                        ← Small sample for demos (drug_trials_sample.csv)
│
├── models/                            ← Saved models (git-ignored: drug_trial_completion_model.joblib)
├── notebooks/                         ← Jupyter notebooks (optional)
│
├── reports/
│   ├── results_summary.md             ← Model evaluation results
│   ├── limitations.md                 ← Known limitations & biases
│   ├── model_card.md                  ← Model Card documentation
│   └── figures/                       ← Evaluation plots
│
└── tests/                             ← Unit tests (future)
```

---

## How to Run

### Prerequisites

- Python 3.9+
- Internet connection (for API access)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/clinicaltrials-gov-completion-risk-predictor.git
cd clinicaltrials-gov-completion-risk-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1: Collect data from ClinicalTrials.gov API (~1000 trials)
python src/clinicaltrials_api.py

# Step 2: Clean data and create labels
python src/preprocess.py

# Step 3: Train the model
python src/train_model.py

# Step 4: Generate evaluation plots (optional)
python src/evaluate.py

# Step 5: Predict for a specific trial
python src/predict.py

# Step 6: Launch the web dashboard
streamlit run app/streamlit_app.py
```

---

## Example Prediction Output

```
  🔮 Clinical Trial Completion Risk Predictor
  ============================================================

  NCT ID:           NCT04280705
  Brief Title:      Study of XYZ in Advanced Cancer
  Current Status:   COMPLETED
  Phase:            PHASE2
  Sponsor Class:    INDUSTRY

  ┌─────────────────────────────────────┐
  │         MODEL PREDICTION             │
  ├─────────────────────────────────────┤
  │  Completion Probability:    82.3%    │
  │  Risk Probability:          17.7%    │
  └─────────────────────────────────────┘

  💚 The model predicts this trial is LIKELY TO COMPLETE.
```

---

## Model Evaluation & Baselines

### Cancer-only Baseline vs. All-Drug-Trials Model

After updating the scope to cover all drug-related clinical trials, we evaluate model metrics against the original cancer-only baseline:

| Metric | Cancer-only Baseline | All-Drug-Trials Model |
|--------|----------------------|-----------------------|
| **Accuracy** | 74.77% | 68.93% |
| **ROC-AUC** | 76.58% | 68.99% |
| **Completed Correctly Predicted** | 67/86 (77.9%) | 104/142 (73.2%) |
| **At-risk Correctly Predicted** | 13/21 (61.9%) | 18/35 (51.4%) |


Visual evaluation plots are generated by `python src/evaluate.py` in `reports/figures/`.

---

## Limitations

See [reports/limitations.md](reports/limitations.md) for the full list. Key limitations include:

- **"Completed" ≠ "clinically successful"** — a trial can complete but fail to prove efficacy
- **"Terminated" ≠ "medically dangerous"** — trials are terminated for many non-safety reasons
- **Missing metadata** — ClinicalTrials.gov fields are often incomplete or inconsistently filled
- **Drug-related trials only** — the model is trained on drug-related trials and may not generalize to non-drug interventions (devices, behavioral, etc.)
- **No temporal features** — the model does not currently use start date, duration, or timeline data

---

## Future Improvements

- [ ] Expand beyond cancer to all disease areas
- [ ] Add temporal features (planned duration, start year)
- [ ] Try Random Forest, XGBoost, or gradient boosting models
- [ ] Add NLP embeddings (sentence-transformers) for richer text features
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add unit tests
- [ ] Cross-validate with multiple seeds for robustness

---

## Ethical Considerations

This project handles public clinical trial metadata. Important ethical guidelines:

1. **No medical claims.** This model does NOT predict whether a treatment is safe, effective, or clinically successful. It only estimates trial completion risk based on structural metadata.

2. **No patient impact.** This tool must not be used to influence patient enrollment decisions, treatment choices, or clinical care.

3. **No regulatory claims.** This model does not predict FDA approval, EMA authorization, or any regulatory outcome.

4. **No investment advice.** Predictions must not be used for financial decisions about pharmaceutical companies, biotech stocks, or clinical-stage assets.

5. **Bias awareness.** The training data reflects the biases of ClinicalTrials.gov — primarily U.S.-registered trials, English-language, and predominantly larger-sponsored studies. The model may not perform equally well across all geographies, disease areas, or sponsor types.

6. **Transparency.** All data comes from a public source. The model's features, limitations, and performance metrics are fully documented.

---

## GitHub Note

⚠️ **Large raw datasets and trained model files are not uploaded to GitHub.** Only a small sample dataset (`data/sample/`) is included in the repository. To reproduce the full results, run the pipeline from Step 1.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

- [ClinicalTrials.gov](https://clinicaltrials.gov/) for providing free, public clinical trial data
- [scikit-learn](https://scikit-learn.org/) for the machine learning framework
- Built as an educational portfolio project demonstrating ML with real-world healthcare data
