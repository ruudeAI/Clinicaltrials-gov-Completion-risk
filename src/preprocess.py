"""
preprocess.py — Cleans raw ClinicalTrials.gov data and creates the ML target label.

WHAT THIS FILE DOES:
    1. Loads the raw CSV created by clinicaltrials_api.py
    2. Keeps only trials with a KNOWN final outcome (Completed, Terminated, etc.)
    3. Removes ongoing/uncertain trials (we can't train on trials whose outcome is unknown)
    4. Creates a binary target label for the ML model
    5. Combines text columns for future NLP feature use
    6. Saves a clean, ML-ready CSV

HOW TO RUN:
    python src/preprocess.py

INPUT:
    data/raw/cancer_trials_raw.csv

OUTPUT:
    data/processed/cancer_trials_processed.csv

BEGINNER CONCEPTS:
    - TARGET LABEL: The thing we're trying to predict. Here it's a 0 or 1:
        1 = trial COMPLETED (success)
        0 = trial was TERMINATED, WITHDRAWN, or SUSPENDED (at risk)

    - WHY EXCLUDE ONGOING TRIALS? Trials that are still "Recruiting" or
      "Active" haven't finished yet — we don't know their final outcome.
      Training on them would be like grading a test that hasn't been taken.

    - WHY COMBINE TEXT COLUMNS? The brief_summary, eligibility_criteria, and
      primary_outcome_measures each contain useful information. Combining them
      into one column makes it easier to feed into a text-processing model
      (TF-IDF) later. Think of it as giving the model one big "description"
      of the trial instead of three separate ones.

DATA SOURCE:
    This project uses data from ClinicalTrials.gov (https://clinicaltrials.gov/).
"""

import os
import sys
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure stdout handles UTF-8 (emojis, etc.) on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Import settings and paths from config.py
from src.config import (
    DRUG_RAW_CSV_PATH as RAW_CSV_PATH,
    DRUG_PROCESSED_CSV_PATH as PROCESSED_CSV_PATH
)


# These are the ONLY statuses we keep for training.
# They represent trials with a KNOWN final outcome.
KEEP_STATUSES = ["COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"]

# These statuses are EXCLUDED because the trial's final outcome is unknown.
# We list them here for documentation — they are removed by keeping only
# the statuses in KEEP_STATUSES above.
EXCLUDED_STATUSES = [
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "UNKNOWN",
    "AVAILABLE",
    "NO_LONGER_AVAILABLE",
    "TEMPORARILY_NOT_AVAILABLE",
    "APPROVED_FOR_MARKETING",
]


# ==============================================================================
# MAIN PREPROCESSING FUNCTION
# ==============================================================================

def preprocess():
    """
    Run the full data cleaning pipeline.

    Steps:
        1. Load raw CSV
        2. Filter to known-outcome statuses
        3. Create binary target label
        4. Remove duplicate trials
        5. Convert columns to proper types
        6. Create combined text column
        7. Fill missing values
        8. Print summary statistics
        9. Save cleaned data
    """
    print("=" * 60)
    print("  🔧 Data Preprocessing Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load raw data
    # ------------------------------------------------------------------
    print("\n📥 Step 1: Loading raw data...")

    if not os.path.exists(RAW_CSV_PATH):
        print(f"  ❌ File not found: {RAW_CSV_PATH}")
        print(f"     Run 'python src/clinicaltrials_api.py' first to collect data.")
        return None

    df = pd.read_csv(RAW_CSV_PATH)
    original_count = len(df)
    print(f"  ✓ Loaded {original_count:,} rows, {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # Step 2: Keep only trials with known final outcomes
    # ------------------------------------------------------------------
    print("\n🔍 Step 2: Filtering to known-outcome statuses...")

    # Show what statuses exist in the data before filtering
    print(f"\n  Status distribution BEFORE filtering:")
    status_counts = df["overall_status"].value_counts()
    for status, count in status_counts.items():
        marker = "  ✓ KEEP" if status in KEEP_STATUSES else "  ✗ DROP"
        print(f"    {status:<35s} {count:>5,}  {marker}")

    # Keep only rows where overall_status is one of our target statuses
    df = df[df["overall_status"].isin(KEEP_STATUSES)].copy()
    print(f"\n  ✓ Kept {len(df):,} trials with known outcomes "
          f"(dropped {original_count - len(df):,})")

    if len(df) == 0:
        print("  ❌ No trials remaining after filtering. Exiting.")
        return None

    # ------------------------------------------------------------------
    # Step 3: Create binary target label
    # ------------------------------------------------------------------
    # This is what our ML model will try to predict.
    #
    # target_completed = 1 → Trial COMPLETED (the "positive" outcome)
    # target_completed = 0 → Trial was TERMINATED, WITHDRAWN, or SUSPENDED
    #                        (the "at risk" / negative outcome)
    #
    # We chose 1 = completed because it's intuitive:
    #   "What's the probability this trial will complete?"
    # ------------------------------------------------------------------
    print("\n🎯 Step 3: Creating target label...")

    df["target_completed"] = (df["overall_status"] == "COMPLETED").astype(int)

    completed_count = df["target_completed"].sum()
    at_risk_count = len(df) - completed_count
    print(f"  target_completed = 1 (Completed):   {completed_count:,} trials")
    print(f"  target_completed = 0 (At Risk):      {at_risk_count:,} trials")

    ratio = completed_count / at_risk_count if at_risk_count > 0 else float("inf")
    print(f"  Class ratio (completed : at-risk):   {ratio:.2f} : 1")

    if ratio > 3:
        print("  ⚠️  Classes are imbalanced — we'll use class_weight='balanced' in training.")

    # ------------------------------------------------------------------
    # Step 4: Remove duplicate trials
    # ------------------------------------------------------------------
    print("\n🧹 Step 4: Removing duplicate NCT IDs...")

    before_dedup = len(df)
    df = df.drop_duplicates(subset="nct_id", keep="first")
    dupes_removed = before_dedup - len(df)

    if dupes_removed > 0:
        print(f"  ✓ Removed {dupes_removed} duplicate trials")
    else:
        print(f"  ✓ No duplicates found")

    # ------------------------------------------------------------------
    # Step 5: Convert columns to proper numeric types
    # ------------------------------------------------------------------
    print("\n🔢 Step 5: Converting numeric columns...")

    # enrollment_count might have non-numeric values or be stored as string
    df["enrollment_count"] = pd.to_numeric(df["enrollment_count"], errors="coerce")
    print(f"  ✓ enrollment_count → numeric "
          f"({df['enrollment_count'].notna().sum():,} valid values)")

    # location_count should already be numeric, but let's make sure
    df["location_count"] = pd.to_numeric(df["location_count"], errors="coerce")
    print(f"  ✓ location_count → numeric "
          f"({df['location_count'].notna().sum():,} valid values)")

    # collaborator_count too
    df["collaborator_count"] = pd.to_numeric(
        df["collaborator_count"], errors="coerce"
    )
    print(f"  ✓ collaborator_count → numeric "
          f"({df['collaborator_count'].notna().sum():,} valid values)")

    # ------------------------------------------------------------------
    # Step 6: Create combined text column
    # ------------------------------------------------------------------
    # We combine three text fields into one for the TF-IDF vectorizer.
    # This gives the model a richer text signal about each trial.
    # ------------------------------------------------------------------
    print("\n📝 Step 6: Creating combined_text column...")

    # Fill missing text with empty strings first (so we can concatenate)
    text_columns = ["brief_summary", "eligibility_criteria", "primary_outcome_measures"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("")
        else:
            df[col] = ""
            print(f"  ⚠️  Column '{col}' not found — filled with empty strings")

    # Combine into a single text column, separated by spaces
    df["combined_text"] = (
        df["brief_summary"].astype(str) + " " +
        df["eligibility_criteria"].astype(str) + " " +
        df["primary_outcome_measures"].astype(str)
    ).str.strip()

    # Show some stats about the text
    avg_len = df["combined_text"].str.len().mean()
    print(f"  ✓ combined_text created (avg length: {avg_len:,.0f} characters)")

    # ------------------------------------------------------------------
    # Step 7: Fill remaining missing text columns
    # ------------------------------------------------------------------
    print("\n🩹 Step 7: Filling missing values in text columns...")

    other_text_cols = [
        "brief_title", "official_title", "conditions",
        "intervention_types", "intervention_names", "countries",
    ]
    for col in other_text_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col] = df[col].fillna("")
                print(f"  ✓ {col}: filled {missing} missing values with ''")

    # ------------------------------------------------------------------
    # Step 8: Print summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  📊 Preprocessing Summary")
    print("=" * 60)

    print(f"\n  Original row count:  {original_count:,}")
    print(f"  Processed row count: {len(df):,}")
    print(f"  Columns:             {df.shape[1]}")

    print(f"\n  Target distribution (target_completed):")
    target_counts = df["target_completed"].value_counts().sort_index()
    for val, count in target_counts.items():
        label = "Completed" if val == 1 else "At Risk"
        pct = count / len(df) * 100
        print(f"    {val} ({label:>10s}): {count:>5,}  ({pct:5.1f}%)")

    # Missing value summary for important fields
    important_fields = [
        "enrollment_count", "location_count", "phases", "sponsor_class",
        "study_type", "allocation", "masking", "primary_purpose",
        "combined_text",
    ]
    print(f"\n  Missing values in key fields:")
    for col in important_fields:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = missing / len(df) * 100
            status = "⚠️" if pct > 20 else "✓"
            print(f"    {status} {col:<30s} {missing:>5,} missing ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # Step 9: Save cleaned data
    # ------------------------------------------------------------------
    print(f"\n💾 Saving processed data...")

    os.makedirs(os.path.dirname(PROCESSED_CSV_PATH), exist_ok=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False)

    size_mb = os.path.getsize(PROCESSED_CSV_PATH) / (1024 * 1024)
    print(f"  ✓ Saved to: {PROCESSED_CSV_PATH} ({size_mb:.2f} MB)")

    print("\n" + "=" * 60)
    print("  ✅ Preprocessing complete!")
    print(f"  Next step: python src/train_model.py")
    print("=" * 60)

    return df


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    preprocess()
