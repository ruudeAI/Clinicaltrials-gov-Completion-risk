"""
evaluate.py — Generate evaluation visualizations for the trained model.

WHAT THIS FILE DOES:
    1. Loads the trained model and processed dataset
    2. Re-creates the same train/test split used during training
    3. Generates evaluation plots:
       - Confusion Matrix
       - ROC Curve
       - Top Feature Coefficients (model interpretability)
    4. Saves all plots to reports/figures/

HOW TO RUN:
    python src/evaluate.py

INPUT:
    models/clinical_trial_completion_model.joblib
    data/processed/cancer_trials_processed.csv

OUTPUT:
    reports/figures/confusion_matrix.png
    reports/figures/roc_curve.png
    reports/figures/top_features.png

NOTE:
    This script uses the SAME random seed and test split as train_model.py,
    so it evaluates on the exact same test data. If you change the processed
    dataset, you must re-train before re-evaluating.

DATA SOURCE:
    This project uses data from ClinicalTrials.gov (https://clinicaltrials.gov/).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

# Ensure stdout handles UTF-8 (emojis, etc.) on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

matplotlib.use("Agg")  # Non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import settings and paths from config.py
from src.config import (
    DRUG_PROCESSED_CSV_PATH as PROCESSED_CSV_PATH,
    DRUG_MODEL_PATH as MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE
)

FIGURES_DIR = os.path.join("reports", "figures")

CATEGORICAL_FEATURES = [
    "study_type", "phases", "sponsor_class", "enrollment_type",
    "intervention_types", "allocation", "intervention_model",
    "masking", "primary_purpose", "sex", "healthy_volunteers",
    "search_query_source",
]
NUMERIC_FEATURES = ["enrollment_count", "collaborator_count", "location_count"]
TEXT_FEATURE = "combined_text"
TARGET = "target_completed"



# ==============================================================================
# MAIN EVALUATION FUNCTION
# ==============================================================================

def evaluate():
    """Generate evaluation visualizations and save to reports/figures/."""

    print("=" * 60)
    print("  📊 Model Evaluation & Visualization")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load model and data
    # ------------------------------------------------------------------
    print("\n📥 Step 1: Loading model and data...")

    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Model not found: {MODEL_PATH}")
        print(f"     Run 'python src/train_model.py' first.")
        return

    if not os.path.exists(PROCESSED_CSV_PATH):
        print(f"  ❌ Data not found: {PROCESSED_CSV_PATH}")
        print(f"     Run 'python src/preprocess.py' first.")
        return

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_CSV_PATH)
    print(f"  ✓ Model loaded from {MODEL_PATH}")
    print(f"  ✓ Data loaded: {len(df):,} trials")

    # ------------------------------------------------------------------
    # Step 2: Recreate the same train/test split
    # ------------------------------------------------------------------
    print("\n✂️  Step 2: Recreating train/test split...")

    all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TEXT_FEATURE]
    for col in all_features:
        if col not in df.columns:
            df[col] = "" if col in CATEGORICAL_FEATURES + [TEXT_FEATURE] else 0

    X = df[all_features].copy()
    y = df[TARGET].copy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Ensure correct types (same as train_model.py)
    X_test[TEXT_FEATURE] = X_test[TEXT_FEATURE].fillna("").astype(str)
    for col in CATEGORICAL_FEATURES:
        X_test[col] = X_test[col].fillna("UNKNOWN").astype(str)

    print(f"  ✓ Test set: {len(X_test):,} trials")

    # ------------------------------------------------------------------
    # Step 3: Generate predictions
    # ------------------------------------------------------------------
    print("\n🔮 Step 3: Generating predictions...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"  ✓ Predictions generated")

    # ------------------------------------------------------------------
    # Step 4: Create output directory
    # ------------------------------------------------------------------
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot 1: Confusion Matrix
    # ------------------------------------------------------------------
    print("\n🎨 Generating plots...")

    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["At Risk (0)", "Completed (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format=",d")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path}")

    # ------------------------------------------------------------------
    # Plot 2: ROC Curve
    # ------------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5,
            label=f"Logistic Regression (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--",
            label="Random Baseline (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {path}")

    # ------------------------------------------------------------------
    # Plot 3: Top Feature Coefficients
    # ------------------------------------------------------------------
    try:
        classifier = model.named_steps["classifier"]
        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        # Get top 20 features by absolute coefficient value
        top_n = min(20, len(coefficients))
        top_indices = np.argsort(np.abs(coefficients))[-top_n:]
        top_names = [str(feature_names[i]) for i in top_indices]
        top_coefs = [coefficients[i] for i in top_indices]

        # Shorten long feature names for readability
        short_names = []
        for name in top_names:
            # Remove transformer prefixes like "cat__", "num__", "text__"
            for prefix in ["cat__", "num__", "text__"]:
                name = name.replace(prefix, "")
            if len(name) > 40:
                name = name[:37] + "..."
            short_names.append(name)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#ef4444" if c < 0 else "#22c55e" for c in top_coefs]
        bars = ax.barh(range(len(short_names)), top_coefs, color=colors, height=0.7)
        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_xlabel("Coefficient Value", fontsize=12)
        ax.set_title("Top 20 Most Influential Features", fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="#64748b", linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#22c55e", label="Increases completion probability"),
            Patch(facecolor="#ef4444", label="Decreases completion probability"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

        fig.tight_layout()
        path = os.path.join(FIGURES_DIR, "top_features.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {path}")

    except Exception as e:
        print(f"  ⚠️  Could not generate feature importance plot: {e}")

    # ------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  ✅ All evaluation plots saved to {FIGURES_DIR}/")
    print("=" * 60)


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    evaluate()
