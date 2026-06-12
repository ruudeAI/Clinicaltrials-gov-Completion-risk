# Known Limitations & Biases

This document describes the known limitations, biases, and caveats of the
Clinical Trial Completion Risk Predictor. **Read this before interpreting
any model predictions.**

---

## 1. "Completed" Does Not Mean Clinically Successful

A trial with `overall_status = COMPLETED` means the study **finished its
protocol as planned**. It does NOT mean:

- The treatment was effective
- The drug met its primary endpoint
- The results were statistically significant
- The treatment received regulatory approval

Many completed trials have negative results — the treatment didn't work.
Our model predicts **operational completion**, not **clinical success**.

---

## 2. "Terminated" Does Not Always Mean Medical Failure

A trial with `overall_status = TERMINATED` was stopped before completion.
This can happen for many reasons, including:

- **Low enrollment** — not enough patients volunteered
- **Funding issues** — the sponsor ran out of money
- **Business decisions** — company strategy changed
- **Regulatory holds** — FDA paused the trial for review
- **Safety signals** — adverse events were observed
- **Futility** — interim analysis showed the treatment was unlikely to work
- **Better treatments available** — a competing therapy was approved

Our model cannot distinguish between these reasons. It only predicts whether
a trial is likely to reach the "COMPLETED" status.

---

## 3. ClinicalTrials.gov Metadata Can Be Missing or Inconsistent

ClinicalTrials.gov is a voluntary registry. Data quality varies:

- **Missing fields**: Many trials have incomplete enrollment counts, location
  data, or eligibility criteria
- **Inconsistent coding**: Phases, intervention types, and masking are not
  always filled in consistently
- **Delayed updates**: Some records are not updated promptly when a trial's
  status changes
- **Voluntary reporting**: Not all trials worldwide are registered on
  ClinicalTrials.gov

Our model handles missing values with imputation (median for numbers,
"UNKNOWN" for categories), but missing data still reduces prediction quality.

---

## 4. Drug-Related Trials Only (Across 20 Conditions)

The model is trained exclusively on **drug-related clinical trials** across 20 conditions (cancer, diabetes, heart disease, covid, etc.). It may not generalize well to:

- Medical device clinical trials
- Behavioral or lifestyle intervention studies
- Surgical procedure trials
- Diagnostic or screening trials that do not use drug interventions

Intervention dynamics, patient compliance, and trial termination reasons differ significantly between drug and non-drug studies.


---

## 5. Model Performance Depends on Available Public Data

- The model only sees **public metadata** — it does not have access to
  internal sponsor data, financial health, manufacturing status, or interim
  results
- Structural metadata (phase, enrollment, sponsor type) captures broad
  patterns but cannot account for trial-specific factors
- Model performance metrics (accuracy, AUC, etc.) reflect performance on
  historical data and may not hold for future trials

---

## 6. Predictions Should NOT Be Used For:

| ❌ Do Not Use For | Why |
|-------------------|-----|
| Healthcare decisions | The model does not evaluate treatment safety or efficacy |
| Patient enrollment | Patients should consult their physicians, not an ML model |
| Regulatory submissions | Not validated for regulatory purposes |
| Investment decisions | Trial completion ≠ commercial success |
| Insurance decisions | The model has no clinical validity |
| Legal purposes | Predictions are probabilistic estimates, not facts |

---

## 7. Class Imbalance

In the training data, completed trials outnumber terminated/withdrawn/suspended
trials. While we use `class_weight='balanced'` to mitigate this, the model may
still be biased toward predicting completion for borderline cases.

---

## 8. No Temporal Features

The current model does not use:

- Trial start date
- Planned duration
- Time since last update
- Historical completion rates by year

These features could improve predictions and may be added in future versions.

---

## 9. Text Features Are Simplistic

The `combined_text` feature uses TF-IDF with only 500 terms and English stop
word removal. This captures broad keyword patterns but does not understand:

- Medical terminology nuances
- Semantic meaning of eligibility criteria
- Complex protocol descriptions

More advanced NLP (e.g., BioBERT embeddings) could improve text understanding.

---

## Summary

This model is a **learning tool** that demonstrates how public data can be
used for predictive modeling. It provides **rough probabilistic estimates**
of trial completion risk — nothing more. Always interpret predictions with
appropriate skepticism and domain context.
