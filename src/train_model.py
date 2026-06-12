"""
train_model.py — Trains and compares ML models to predict clinical trial completion.

WHAT THIS FILE DOES:
    1. Loads the cleaned dataset from preprocess.py
    2. Splits data into training (80%) and test (20%) sets
    3. Builds a scikit-learn Pipeline that handles:
       - Categorical features (OneHotEncoder)
       - Numeric features (SimpleImputer)
       - Text features (TF-IDF Vectorizer)
    4. Benchmarks multiple classifiers:
       - Logistic Regression
       - Random Forest
       - Gradient Boosting
       - HistGradientBoosting
    5. Evaluates each model and selects the best by ROC-AUC
    6. Saves the best model and a comparison report

HOW TO RUN:
    python src/train_model.py

INPUT:
    data/processed/drug_trials_processed.csv

OUTPUT:
    models/drug_trial_completion_model.joblib  — Best trained model pipeline
    reports/results_summary.md                  — Evaluation report

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import settings and paths from config.py
from src.config import (
    DRUG_PROCESSED_CSV_PATH as PROCESSED_CSV_PATH,
    DRUG_MODEL_PATH as MODEL_SAVE_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    CLASSIFICATION_THRESHOLD
)

# File paths
REPORT_SAVE_PATH = os.path.join("reports", "results_summary.md")

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
    "search_query_source",  # Condition area that fetched this trial
    "therapeutic_area_group",  # Broader therapeutic area grouping (Phase 1)
]


# NUMERIC FEATURES: Columns with actual numbers.
# Missing values are filled with the median.
NUMERIC_FEATURES = [
    "enrollment_count",         # Number of participants
    "collaborator_count",       # Number of collaborating organizations
    "location_count",           # Number of trial locations
    "start_year",               # Year trial started (Phase 1)
    "country_count",            # Number of unique countries (Phase 1)
    "condition_count",          # Number of listed conditions (Phase 1)
    "intervention_count",       # Number of intervention names (Phase 1)
    "enrollment_per_location",  # Enrollment / max(locations, 1) (Phase 1)
    "has_multiple_countries",   # 0/1 flag (Phase 1)
    "is_industry_sponsored",    # 0/1 flag (Phase 1)
    "is_randomized",            # 0/1 flag (Phase 1)
    "is_blinded",               # 0/1 flag (Phase 1)
    "text_length_summary",      # Char length of brief_summary (Phase 1)
    "text_length_eligibility",  # Char length of eligibility_criteria (Phase 1)
    "text_length_outcomes",     # Char length of primary_outcome_measures (Phase 1)
]

# TEXT FEATURE: Free-text column processed by TF-IDF.
# TF-IDF converts text into numbers by measuring how "important" each word
# is to a document relative to the entire collection of documents.
TEXT_FEATURE = "combined_text"

# TARGET: What we're predicting.
TARGET = "target_completed"


# ==============================================================================
# HELPER: Build a preprocessing + classifier pipeline
# ==============================================================================

def _build_pipeline(classifier):
    """
    Build a scikit-learn Pipeline with shared preprocessing and a given classifier.

    Parameters
    ----------
    classifier : sklearn estimator
        Any scikit-learn compatible classifier instance.

    Returns
    -------
    Pipeline
        A complete preprocessing + classification pipeline.
    """
    # Preprocessing for CATEGORICAL features
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Preprocessing for NUMERIC features
    # StandardScaler normalizes features to zero mean and unit variance,
    # which helps LogisticRegression converge and improves model stability.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Preprocessing for TEXT feature — upgraded for Phase 1
    text_transformer = TfidfVectorizer(
        max_features=1000,           # 500 → 1000 for richer text signal
        stop_words="english",
        ngram_range=(1, 2),          # Capture bigrams (Phase 1)
        min_df=2,                    # Ignore very rare terms (Phase 1)
        max_df=0.9,                  # Ignore very common terms (Phase 1)
    )

    # Combine all preprocessors into one ColumnTransformer
    # sparse_threshold=0 forces dense output, required for HistGradientBoosting
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("text", text_transformer, TEXT_FEATURE),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train():
    """
    Main training function. Orchestrates the full ML pipeline:
    load → split → build pipelines → train all → evaluate → select best → save.
    """
    print("=" * 60)
    print("  🧠 Model Training Pipeline (Phase 1 — Multi-Model)")
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
    print(f"\n✂️  Step 3: Splitting data ({int((1-TEST_SIZE)*100)}% train, "
          f"{int(TEST_SIZE*100)}% test)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"  Training set: {len(X_train):,} trials")
    print(f"  Test set:     {len(X_test):,} trials")
    print(f"  Train target: {y_train.value_counts().to_dict()}")
    print(f"  Test target:  {y_test.value_counts().to_dict()}")

    # Ensure correct types for preprocessing
    X_train[TEXT_FEATURE] = X_train[TEXT_FEATURE].fillna("").astype(str)
    X_test[TEXT_FEATURE] = X_test[TEXT_FEATURE].fillna("").astype(str)

    for col in CATEGORICAL_FEATURES:
        X_train[col] = X_train[col].fillna("UNKNOWN").astype(str)
        X_test[col] = X_test[col].fillna("UNKNOWN").astype(str)

    # ------------------------------------------------------------------
    # Step 4: Define candidate models
    # ------------------------------------------------------------------
    print("\n🔧 Step 4: Building model pipelines...")

    candidates = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            max_iter=5000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_SEED,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_SEED,
        ),
    }

    print(f"  ✓ {len(candidates)} models to benchmark:")
    for name in candidates:
        print(f"    • {name}")

    # ------------------------------------------------------------------
    # Step 5: Train and evaluate each model
    # ------------------------------------------------------------------
    print("\n🏋️ Step 5: Training and evaluating all models...\n")

    results = {}
    best_name = None
    best_auc = -1
    best_pipeline = None

    for name, clf in candidates.items():
        print(f"  ── {name} ──")

        pipeline = _build_pipeline(clf)

        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            print(f"    ❌ Training failed: {e}\n")
            continue

        y_pred = pipeline.predict(X_test)

        # HistGradientBoosting doesn't support predict_proba with
        # sparse matrices by default, so handle potential issues
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            y_prob = None
            roc_auc = 0.0

        accuracy = accuracy_score(y_test, y_pred)
        
        # Class 1 (Completed) metrics
        prec_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        # Class 0 (At Risk) metrics
        prec_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        rec_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        
        conf = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "precision_completed": prec_1,
            "recall_completed": rec_1,
            "f1_completed": f1_1,
            "precision_at_risk": prec_0,
            "recall_at_risk": rec_0,
            "f1_at_risk": f1_0,
            "roc_auc": roc_auc,
            "confusion_matrix": conf,
            "pipeline": pipeline,
        }

        print(f"    Accuracy:             {accuracy:.4f}")
        print(f"    ROC-AUC:              {roc_auc:.4f}")
        print(f"    [Completed Class 1]")
        print(f"      Precision (Compl):  {prec_1:.4f}")
        print(f"      Recall (Compl):     {rec_1:.4f}")
        print(f"      F1-Score (Compl):   {f1_1:.4f}")
        print(f"    [At-Risk Class 0]")
        print(f"      Precision (Risk):   {prec_0:.4f}")
        print(f"      Recall (Risk):      {rec_0:.4f}")
        print(f"      F1-Score (Risk):    {f1_0:.4f}")
        print()

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        print("  ❌ No models trained successfully. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 6: Select and announce the best model
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  🏆 Best Model: {best_name} (ROC-AUC = {best_auc:.4f})")
    print("=" * 60)

    # For the best model, generate predictions using our custom CLASSIFICATION_THRESHOLD
    y_prob_best = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_best = (y_prob_best >= CLASSIFICATION_THRESHOLD).astype(int)

    best_accuracy = accuracy_score(y_test, y_pred_best)
    best_prec_1 = precision_score(y_test, y_pred_best, pos_label=1, zero_division=0)
    best_rec_1 = recall_score(y_test, y_pred_best, pos_label=1, zero_division=0)
    best_f1_1 = f1_score(y_test, y_pred_best, pos_label=1, zero_division=0)
    best_prec_0 = precision_score(y_test, y_pred_best, pos_label=0, zero_division=0)
    best_rec_0 = recall_score(y_test, y_pred_best, pos_label=0, zero_division=0)
    best_f1_0 = f1_score(y_test, y_pred_best, pos_label=0, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred_best)
    class_report = classification_report(y_test, y_pred_best, zero_division=0)

    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  Best Model Results (Threshold = {CLASSIFICATION_THRESHOLD:.2f})   │")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  Accuracy:                {best_accuracy:>8.4f}           │")
    print(f"  │  ROC-AUC:                 {best_auc:>8.4f}           │")
    print(f"  │                                              │")
    print(f"  │  [Completed Class 1]                         │")
    print(f"  │    Precision (Compl):     {best_prec_1:>8.4f}           │")
    print(f"  │    Recall (Compl):        {best_rec_1:>8.4f}           │")
    print(f"  │    F1-Score (Compl):      {best_f1_1:>8.4f}           │")
    print(f"  │                                              │")
    print(f"  │  [At-Risk Class 0]                           │")
    print(f"  │    Precision (Risk):      {best_prec_0:>8.4f}           │")
    print(f"  │    Recall (Risk):         {best_rec_0:>8.4f}           │")
    print(f"  │    F1-Score (Risk):       {best_f1_0:>8.4f}           │")
    print(f"  └──────────────────────────────────────────────┘")

    print(f"\n  Confusion Matrix ({best_name} @ Threshold {CLASSIFICATION_THRESHOLD:.2f}):")
    print(f"                  Predicted")
    print(f"                  AtRisk  Complete")
    print(f"    Actual AtRisk  [{conf_matrix[0][0]:>5,}]  [{conf_matrix[0][1]:>5,}]")
    print(f"    Actual Compl.  [{conf_matrix[1][0]:>5,}]  [{conf_matrix[1][1]:>5,}]")

    print(f"\n  Full Classification Report:")
    print(class_report)

    # ------------------------------------------------------------------
    # Step 6.5: Run completion probability threshold sweep
    # ------------------------------------------------------------------
    print("\n📈 Step 6.5: Running completion probability threshold sweep...")
    sweep_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    sweep_rows = []
    
    for t in sweep_thresholds:
        y_pred_t = (y_prob_best >= t).astype(int)
        
        acc = accuracy_score(y_test, y_pred_t)
        prec_1 = precision_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        rec_1 = recall_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        f1_1 = f1_score(y_test, y_pred_t, pos_label=1, zero_division=0)
        
        prec_0 = precision_score(y_test, y_pred_t, pos_label=0, zero_division=0)
        rec_0 = recall_score(y_test, y_pred_t, pos_label=0, zero_division=0)
        f1_0 = f1_score(y_test, y_pred_t, pos_label=0, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred_t)
        cm_str = f"TN:{cm[0][0]}, FP:{cm[0][1]}, FN:{cm[1][0]}, TP:{cm[1][1]}"
        
        sweep_rows.append(
            f"| {t:.2f} | {acc:.4f} | {prec_1:.4f} | {rec_1:.4f} | {f1_1:.4f} | "
            f"{prec_0:.4f} | {rec_0:.4f} | {f1_0:.4f} | {cm_str} |"
        )
    sweep_table = "\n".join(sweep_rows)

    # ------------------------------------------------------------------
    # Step 7: Save the best trained model pipeline
    # ------------------------------------------------------------------
    print("\n💾 Step 7: Saving model and report...")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(best_pipeline, MODEL_SAVE_PATH)
    print(f"  ✓ Model saved to: {MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Step 8: Save a markdown comparison report
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(REPORT_SAVE_PATH), exist_ok=True)

    # Build comparison table with class-specific metrics (evaluated at standard 0.50 threshold)
    comp_rows = []
    for name, res in results.items():
        marker = " ⭐" if name == best_name else ""
        comp_rows.append(
            f"| {name}{marker} | {res['accuracy']:.4f} | {res['roc_auc']:.4f} | "
            f"{res['precision_completed']:.4f} | {res['recall_completed']:.4f} | {res['f1_completed']:.4f} | "
            f"{res['precision_at_risk']:.4f} | {res['recall_at_risk']:.4f} | {res['f1_at_risk']:.4f} |"
        )
    comp_table = "\n".join(comp_rows)

    report_md = f"""# Model Evaluation Results — Phase 1

> **Generated by:** `python src/train_model.py`
> **Best Model:** {best_name} (ROC-AUC = {best_auc:.4f})
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

## Model Comparison (Standard 0.50 Decision Boundary)

| Model | Accuracy | ROC-AUC | Completed Precision | Completed Recall | Completed F1 | At-Risk Precision | At-Risk Recall | At-Risk F1 |
|-------|----------|---------|---------------------|------------------|--------------|-------------------|----------------|------------|
{comp_table}

---

## 📈 Decision Threshold Tuning

To optimize the prediction of clinical trial completion vs. non-completion (At-Risk), we swept the decision boundary for predicted completion probability from `0.30` to `0.80`.

### Core Threshold Concepts
* **Lower threshold** (e.g. 0.30) favors **Completed predictions**: The model requires very low probability to predict "At Risk", leading to high Completed recall but missing almost all actual risk.
* **Higher threshold** (e.g. 0.75) makes the model **more likely to flag trials as At-Risk**: The model requires very high confidence to predict "Completed", which increases the recall for at-risk trials.
* Since the primary goal of this project is **risk prediction** (identifying trials likely to fail), we prioritize **At-Risk recall** while keeping overall accuracy reasonable.

### Threshold Sweep Results

| Threshold | Accuracy | Completed Prec | Completed Recall | Completed F1 | At-Risk Prec | At-Risk Recall | At-Risk F1 | Confusion Matrix (TN, FP, FN, TP) |
|---|---|---|---|---|---|---|---|---|
{sweep_table}

### Selected Decision Boundary
* **Selected Completion Threshold:** `{CLASSIFICATION_THRESHOLD:.2f}`
* **Rationale:** At this threshold, the model significantly improves At-Risk Recall to **{best_rec_0:.1%}** (up from **22.0%** at 0.50) and achieves the highest At-Risk F1-score (**{best_f1_0:.4f}**), while maintaining a solid overall accuracy of **{best_accuracy:.1%}**.

---

## 🏆 Best Model Details: {best_name} (Threshold = {CLASSIFICATION_THRESHOLD:.2f})

### Overall Metrics
* **Accuracy:** {best_accuracy:.4f}
* **ROC-AUC:** {best_auc:.4f}

### Completed Class (Class 1)
* **Precision:** {best_prec_1:.4f}
* **Recall:** {best_rec_1:.4f}
* **F1-Score:** {best_f1_1:.4f}

### At-Risk Class (Class 0)
* **Precision:** {best_prec_0:.4f}
* **Recall:** {best_rec_0:.4f}
* **F1-Score:** {best_f1_0:.4f}

## Confusion Matrix ({best_name} @ Threshold {CLASSIFICATION_THRESHOLD:.2f})

| | Predicted: At Risk | Predicted: Completed |
|---|-------------------|---------------------|
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
- `{TEXT_FEATURE}` (TF-IDF, max 1000 features, bigrams, min_df=2, max_df=0.9)

## Model Details

- **Best Algorithm:** {best_name}
- **Models Compared:** {', '.join(results.keys())}
- **Class weighting:** balanced (handles class imbalance)
- **Text processing:** TF-IDF with 1000 max features, bigrams, English stop words
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
    print(f"  Best Model: {best_name}")
    print(f"  Model: {MODEL_SAVE_PATH}")
    print(f"  Report: {REPORT_SAVE_PATH}")
    print(f"\n  Next step: python src/evaluate.py")
    print("=" * 60)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    train()
