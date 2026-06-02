# 📓 Notebooks Guide

This folder is for Jupyter notebooks that walk through the project step by step.
Notebooks are great for **interactive exploration** — you can run cells one at a time,
see outputs inline, and add your own notes.

## How to Create the Notebooks

```bash
# 1. Install Jupyter (if not already installed)
pip install jupyter

# 2. Launch Jupyter from the project root
cd clinicaltrials-gov-completion-risk-predictor
jupyter notebook
```

Then create each notebook in this folder:

---

### 📘 `01_data_collection.ipynb`
**Goal:** Fetch trial data from the ClinicalTrials.gov API.

Suggested cells:
1. Import `src.clinicaltrials_api` and `src.config`
2. Run a small test query (10 trials) to see the API response structure
3. Explore the raw JSON — what fields are available?
4. Run the full data collection (or load pre-saved raw data)
5. Check: how many trials per status did we get?

---

### 📘 `02_data_cleaning.ipynb`
**Goal:** Clean raw data and engineer features for the model.

Suggested cells:
1. Load the raw data from `data/raw/`
2. Explore missing values — which columns have gaps?
3. Walk through each feature being engineered (phase, enrollment, duration, etc.)
4. Run the full preprocessing pipeline
5. Visualize the cleaned dataset: class distribution, feature distributions, correlations

---

### 📘 `03_model_training.ipynb`
**Goal:** Train and compare ML models.

Suggested cells:
1. Load the processed dataset from `data/processed/`
2. Split into train/test sets
3. Train Logistic Regression — explain what it is
4. Train Random Forest — explain what it is
5. Compare cross-validation scores
6. Show feature importances

---

### 📘 `04_prediction_demo.ipynb`
**Goal:** Use the trained model to predict on new trial data.

Suggested cells:
1. Load the saved model from `models/`
2. Create a sample trial input (manually or from the API)
3. Make a prediction — is this trial likely to complete?
4. Show the confidence score and top contributing features
5. Discuss limitations and when NOT to use this model
