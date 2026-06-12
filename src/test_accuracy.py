"""
test_accuracy.py — Dynamically tests the accuracy of the model on the held-out test dataset.
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure stdout handles UTF-8 (emojis, etc.) on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import settings and paths from config.py
from src.config import (
    DRUG_PROCESSED_CSV_PATH as PROCESSED_CSV_PATH,
    DRUG_MODEL_PATH as MODEL_PATH
)

def main():
    if not os.path.exists(PROCESSED_CSV_PATH) or not os.path.exists(MODEL_PATH):
        print("❌ Model or processed dataset missing. Please run preprocessing and training first.")
        return

    # Load data and replicate train/test split
    df = pd.read_csv(PROCESSED_CSV_PATH)
    X = df.copy()
    y = df["target_completed"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Re-run prediction on the test set
    X_test_clean = X_test.copy()
    X_test_clean["combined_text"] = X_test_clean["combined_text"].fillna("").astype(str)
    for col in [
        "study_type", "phases", "sponsor_class", "enrollment_type",
        "intervention_types", "allocation", "intervention_model",
        "masking", "primary_purpose", "sex", "healthy_volunteers",
        "search_query_source"
    ]:
        X_test_clean[col] = X_test_clean[col].fillna("UNKNOWN").astype(str)


    y_pred = model.predict(X_test_clean)
    y_prob = model.predict_proba(X_test_clean)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("  🧪 Model Accuracy Evaluation on Test Set")
    print("=" * 60)
    print(f"  Total Test Size: {len(X_test)} trials")
    print(f"  Correct Predictions: {sum(y_pred == y_test)} out of {len(X_test)}")
    print(f"  Overall Accuracy: {acc:.2%}")
    print("=" * 60)

    print("\n📊 Detailed Breakdown by Trial Class:")
    print("-" * 60)
    
    # Class 0: At Risk
    actual_at_risk = sum(y_test == 0)
    correct_at_risk = cm[0, 0]
    print(f"  Actual 'At Risk' (Terminated/Withdrawn/Suspended): {actual_at_risk} trials")
    print(f"  Correctly Caught by Model:                     {correct_at_risk} ({correct_at_risk/actual_at_risk:.2%})")
    print(f"  Missed (False Negatives):                       {cm[0, 1]} ({cm[0, 1]/actual_at_risk:.2%})")
    print("-" * 60)

    # Class 1: Completed
    actual_completed = sum(y_test == 1)
    correct_completed = cm[1, 1]
    print(f"  Actual 'Completed' Trials:                     {actual_completed} trials")
    print(f"  Correctly Predicted by Model:                  {correct_completed} ({correct_completed/actual_completed:.2%})")
    print(f"  Missed (False Positives):                       {cm[1, 0]} ({cm[1, 0]/actual_completed:.2%})")
    print("-" * 60)

    print("\n🔍 Random Samples of Correct Predictions:")
    print("=" * 60)
    
    # Let's show 3 actual correct predictions
    correct_positions = [i for i, (p, a) in enumerate(zip(y_pred, y_test)) if p == a]
    
    count = 0
    for pos in correct_positions:
        if count >= 3:
            break
        row = X_test.iloc[pos]
        row_df = pd.DataFrame([row])
        prob = model.predict_proba(row_df)[0][1]
        
        pred_label = "COMPLETED" if y_pred[pos] == 1 else "AT RISK"
        true_label = "COMPLETED" if y_test.iloc[pos] == 1 else "AT RISK"
        
        print(f"NCT ID:         {row['nct_id']}")
        print(f"Title:          {row['brief_title'][:65]}...")
        print(f"Actual Outcome: {true_label} ({row['overall_status']})")
        print(f"Model Estimate: {pred_label} ({prob:.1%} completion probability)")
        print(f"Result:         ✅ CORRECT")
        print("-" * 60)
        count += 1

if __name__ == "__main__":
    main()

