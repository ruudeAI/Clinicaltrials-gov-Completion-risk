"""
train_model.py — Trains a machine learning model to predict clinical trial completion.

WHAT THIS FILE DOES:
    1. Loads the cleaned dataset from preprocess.py
    2. Splits data into training (80%) and test (20%) sets
    3. Builds a scikit-learn Pipeline that handles:
       - Categorical features (OneHotEncoder)
       - Numeric features (SimpleImputer)
       - Text features (TF-IDF Vectorizer)
    4. Trains a Logistic Regression classifier
    5. Evaluates the model on held-out test data
    6. Saves the trained model and a results summary

HOW TO RUN:
    python src/train_model.py

INPUT:
    data/processed/cancer_trials_processed.csv

OUTPUT:
    models/clinical_trial_completion_model.joblib  — Trained model pipeline
    reports/results_summary.md                     — Evaluation report

BEGINNER CONCEPTS:

    FEATURES are the input columns the model uses to make predictions.
    Think of them as "clues" — the model looks at a trial's phase, sponsor,
    enrollment size, etc. to guess whether it will complete.

    TARGET is what we're predicting: target_completed (1 = completed, 0 = at risk).

    TRAIN/TEST SPLIT: We hide 20% of our data and never show it to the model
    during training. After training, we test the model on this hidden data
    to see how well it performs on trials it has never seen before.

    PIPELINE: A scikit-learn Pipeline chains preprocessing + model into one
    object. This means when you make a prediction on new data, it automatically
    applies the same preprocessing (encoding, imputation, TF-IDF) before
    predicting. No manual steps needed.

    CLASS_WEIGHT='BALANCED': If 80% of trials are "completed" and only 20% are
    "at risk", the model could just always predict "completed" and be 80%
    accurate — but useless. class_weight='balanced' forces the model to pay
    MORE attention to the minority class (at-risk trials), so it learns to
    identify them properly.

DATA SOURCE:
    This project uses data from ClinicalTrials.gov (https://clinicaltrials.gov/).
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure stdout handles UTF-8 (emojis, etc.) on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

# File paths
PROCESSED_CSV_PATH = os.path.join("data", "processed", "cancer_trials_processed.csv")
MODEL_SAVE_PATH = os.path.join("models", "clinical_trial_completion_model.joblib")
REPORT_SAVE_PATH = os.path.join("reports", "results_summary.md")

# Reproducibility — using the same random seed means you get the same
# train/test split and results every time you run this script.
RANDOM_SEED = 42
TEST_SIZE = 0.2  # 20% of data held out for testing

# ------------------------------------------------------------------
# FEATURE DEFINITIONS
# ------------------------------------------------------------------
# These lists tell the ColumnTransformer which columns to process and how.

# CATEGORICAL FEATURES: Non-numeric columns like "PHASE3", "INDUSTRY", "DRUG".
# These get one-hot encoded (converted to 0/1 columns for each category).
CATEGORICAL_FEATURES = [
    "study_type",           # INTERVENTIONAL, OBSERVATIONAL, etc.
    "phases",               # PHASE1, PHASE2, PHASE3, etc.
    "sponsor_class",        # INDUSTRY, NIH, OTHER, etc.
    "enrollment_type",      # ACTUAL or ESTIMATED
    "intervention_types",   # DRUG, BIOLOGICAL, DEVICE, etc.
    "allocation",           # RANDOMIZED, NON_RANDOMIZED, etc.
    "intervention_model",   # PARALLEL, CROSSOVER, SINGLE_GROUP, etc.
    "masking",              # NONE, SINGLE, DOUBLE, TRIPLE, QUADRUPLE
    "primary_purpose",      # TREATMENT, PREVENTION, DIAGNOSTIC, etc.
    "sex",                  # ALL, FEMALE, MALE
    "healthy_volunteers",   # True / False
]

# NUMERIC FEATURES: Columns with actual numbers.
# Missing values are filled with the median.
NUMERIC_FEATURES = [
    "enrollment_count",     # Number of participants
    "collaborator_count",   # Number of collaborating organizations
    "location_count",       # Number of trial locations
]

# TEXT FEATURE: Free-text column processed by TF-IDF.
# TF-IDF converts text into numbers by measuring how "important" each word
# is to a document relative to the entire collection of documents.
TEXT_FEATURE = "combined_text"

# TARGET: What we're predicting.
TARGET = "target_completed"


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train():
    """
    Main training function. Orchestrates the full ML pipeline:
    load → split → build pipeline → train → evaluate → save.
    """
    print("=" * 60)
    print("  🧠 Model Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load the processed data
    # ------------------------------------------------------------------
    print("\n📥 Step 1: Loading processed data...")

    if not os.path.exists(PROCESSED_CSV_PATH):
        print(f"  ❌ File not found: {PROCESSED_CSV_PATH}")
        print(f"     Run 'python src/preprocess.py' first.")
        return

    df = pd.read_csv(PROCESSED_CSV_PATH)
    print(f"  ✓ Loaded {len(df):,} trials, {df.shape[1]} columns")

    # Quick sanity check: does the target column exist?
    if TARGET not in df.columns:
        print(f"  ❌ Target column '{TARGET}' not found in the data.")
        return

    # ------------------------------------------------------------------
    # Step 2: Prepare features (X) and target (y)
    # ------------------------------------------------------------------
    print("\n📊 Step 2: Preparing features and target...")

    # Make sure all expected feature columns exist in the data
    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TEXT_FEATURE]
    missing_cols = [col for col in all_features if col not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Missing columns (will be created as empty): {missing_cols}")
        for col in missing_cols:
            df[col] = "" if col in CATEGORICAL_FEATURES + [TEXT_FEATURE] else 0

    # X = all the features (input data)
    # y = the target label (what we want to predict)
    X = df[all_features].copy()
    y = df[TARGET].copy()

    print(f"  Features: {len(all_features)} columns")
    print(f"    Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"    Numeric:     {len(NUMERIC_FEATURES)}")
    print(f"    Text:        1 ({TEXT_FEATURE})")
    print(f"  Target: {TARGET}")
    print(f"    Value counts: {y.value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Step 3: Split into training and test sets
    # ------------------------------------------------------------------
    # We hold out 20% of the data for testing.
    # stratify=y ensures both sets have the same ratio of completed
    # vs. at-risk trials. Without this, the test set might accidentally
    # have mostly one class.
    # ------------------------------------------------------------------
    print(f"\n✂️  Step 3: Splitting data ({int((1-TEST_SIZE)*100)}% train, "
          f"{int(TEST_SIZE*100)}% test)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,       # ← Keeps class balance in both sets
    )

    print(f"  Training set: {len(X_train):,} trials")
    print(f"  Test set:     {len(X_test):,} trials")
    print(f"  Train target: {y_train.value_counts().to_dict()}")
    print(f"  Test target:  {y_test.value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Step 4: Build the scikit-learn Pipeline
    # ------------------------------------------------------------------
    # A Pipeline chains multiple steps together:
    #   Step 1 (preprocessor): Transform raw features into model-ready format
    #   Step 2 (classifier):   The actual ML model
    #
    # The ColumnTransformer applies different transformations to different
    # column types:
    #   - Categorical → fill missing with "UNKNOWN", then one-hot encode
    #   - Numeric     → fill missing with the median value
    #   - Text        → convert to TF-IDF word importance scores
    # ------------------------------------------------------------------
    print("\n🔧 Step 4: Building model pipeline...")

    # Preprocessing for CATEGORICAL features
    categorical_transformer = Pipeline(steps=[
        # Fill missing categorical values with "UNKNOWN"
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        # Convert categories to binary 0/1 columns
        # handle_unknown="ignore" means if we see a new category during
        # prediction that wasn't in training, we just ignore it (all 0s)
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Preprocessing for NUMERIC features
    numeric_transformer = Pipeline(steps=[
        # Fill missing numeric values with the median (middle value)
        # Median is better than mean because it's not affected by outliers
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Preprocessing for TEXT feature
    # TF-IDF (Term Frequency - Inverse Document Frequency) converts text
    # into numbers. Words that appear frequently in one document but rarely
    # in others get higher scores — they're more "distinctive".
    text_transformer = TfidfVectorizer(
        max_features=500,       # Keep only the 500 most important words
        stop_words="english",   # Remove common words like "the", "and", "is"
    )

    # Combine all preprocessors into one ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("text", text_transformer, TEXT_FEATURE),
        ],
        remainder="drop",  # Drop any columns not listed above
    )

    # Build the full pipeline: preprocessor → classifier
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",  # ← Handle class imbalance
            random_state=RANDOM_SEED,
            max_iter=1000,            # Allow enough iterations to converge
        )),
    ])

    print("  ✓ Pipeline built:")
    print("    1. ColumnTransformer (categorical + numeric + text)")
    print("    2. LogisticRegression (class_weight='balanced')")

    # ------------------------------------------------------------------
    # Step 5: Train the model
    # ------------------------------------------------------------------
    # This is where the actual learning happens! The model looks at the
    # training data and learns patterns that distinguish completed trials
    # from at-risk trials.
    # ------------------------------------------------------------------
    print("\n🏋️ Step 5: Training the model...")

    # Ensure text column is string type (TfidfVectorizer needs strings)
    X_train[TEXT_FEATURE] = X_train[TEXT_FEATURE].fillna("").astype(str)
    X_test[TEXT_FEATURE] = X_test[TEXT_FEATURE].fillna("").astype(str)

    # Ensure categorical columns are string type
    for col in CATEGORICAL_FEATURES:
        X_train[col] = X_train[col].fillna("UNKNOWN").astype(str)
        X_test[col] = X_test[col].fillna("UNKNOWN").astype(str)

    pipeline.fit(X_train, y_train)
    print("  ✓ Model trained successfully!")

    # ------------------------------------------------------------------
    # Step 6: Evaluate the model
    # ------------------------------------------------------------------
    # We test the model on data it has NEVER seen during training.
    # This tells us how well it will perform on truly new trials.
    #
    # METRICS EXPLAINED:
    #   Accuracy:  % of all predictions that are correct
    #   Precision: When the model says "at risk", how often is it right?
    #   Recall:    Of all actual at-risk trials, how many did we catch?
    #   F1-Score:  Harmonic mean of precision and recall (balanced measure)
    #   ROC-AUC:   How well the model separates the two classes (0.5 = random,
    #              1.0 = perfect)
    # ------------------------------------------------------------------
    print("\n📈 Step 6: Evaluating model on test data...")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of class 1

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │      Model Evaluation Results        │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Accuracy:   {accuracy:>8.4f}               │")
    print(f"  │  Precision:  {precision:>8.4f}               │")
    print(f"  │  Recall:     {recall:>8.4f}               │")
    print(f"  │  F1-Score:   {f1:>8.4f}               │")
    print(f"  │  ROC-AUC:    {roc_auc:>8.4f}               │")
    print(f"  └─────────────────────────────────────┘")

    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  AtRisk  Complete")
    print(f"    Actual AtRisk  [{conf_matrix[0][0]:>5,}]  [{conf_matrix[0][1]:>5,}]")
    print(f"    Actual Compl.  [{conf_matrix[1][0]:>5,}]  [{conf_matrix[1][1]:>5,}]")

    print(f"\n  Full Classification Report:")
    print(class_report)

    # ------------------------------------------------------------------
    # Step 7: Save the trained model pipeline
    # ------------------------------------------------------------------
    print("\n💾 Step 7: Saving model and report...")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"  ✓ Model saved to: {MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Step 8: Save a markdown evaluation report
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(REPORT_SAVE_PATH), exist_ok=True)

    report_md = f"""# Model Evaluation Results

> **Generated by:** `python src/train_model.py`
> **Model:** Logistic Regression with TF-IDF text features
> **Data source:** ClinicalTrials.gov (https://clinicaltrials.gov/)

## ⚠️ Disclaimer

This model is for **educational and research portfolio purposes only**.
It does not predict treatment safety, effectiveness, clinical success,
regulatory approval, or investment value.

## Dataset

| Metric | Value |
|--------|-------|
| Total trials | {len(df):,} |
| Training set | {len(X_train):,} |
| Test set | {len(X_test):,} |
| Completed (class 1) | {int(y.sum()):,} ({y.mean()*100:.1f}%) |
| At Risk (class 0) | {int(len(y) - y.sum()):,} ({(1-y.mean())*100:.1f}%) |

## Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | {accuracy:.4f} |
| **Precision** | {precision:.4f} |
| **Recall** | {recall:.4f} |
| **F1-Score** | {f1:.4f} |
| **ROC-AUC** | {roc_auc:.4f} |

## Confusion Matrix

|  | Predicted: At Risk | Predicted: Completed |
|--|-------------------|---------------------|
| **Actual: At Risk** | {conf_matrix[0][0]:,} | {conf_matrix[0][1]:,} |
| **Actual: Completed** | {conf_matrix[1][0]:,} | {conf_matrix[1][1]:,} |

## Classification Report

```
{class_report}
```

## Features Used

### Categorical ({len(CATEGORICAL_FEATURES)})
{chr(10).join(f'- `{f}`' for f in CATEGORICAL_FEATURES)}

### Numeric ({len(NUMERIC_FEATURES)})
{chr(10).join(f'- `{f}`' for f in NUMERIC_FEATURES)}

### Text (1)
- `{TEXT_FEATURE}` (TF-IDF, max 500 features)

## Model Details

- **Algorithm:** Logistic Regression
- **Class weighting:** balanced (handles class imbalance)
- **Text processing:** TF-IDF with 500 max features, English stop words removed
- **Categorical encoding:** One-hot encoding with unknown category handling
- **Missing value handling:** Median for numeric, "UNKNOWN" for categorical
- **Random seed:** {RANDOM_SEED}
- **Test split:** {int(TEST_SIZE * 100)}%
"""

    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  ✓ Report saved to: {REPORT_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ✅ Training complete!")
    print("=" * 60)
    print(f"  Model: {MODEL_SAVE_PATH}")
    print(f"  Report: {REPORT_SAVE_PATH}")
    print(f"\n  Next step: python src/predict.py")
    print("=" * 60)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    train()
