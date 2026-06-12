"""
predict.py — Predict completion risk for a clinical trial by its NCT ID.

WHAT THIS FILE DOES:
    1. Takes an NCT ID (like "NCT04280705") as input
    2. Fetches the trial's live data from the ClinicalTrials.gov API
    3. Processes it the same way as our training data
    4. Runs it through the trained ML model
    5. Prints the predicted completion and risk probabilities

HOW TO RUN:
    python src/predict.py

    Or import and use in your own code:
        from src.predict import predict_nct_id
        predict_nct_id("NCT04280705")

INPUT:
    - NCT ID (provided by the user)
    - models/clinical_trial_completion_model.joblib (trained model)

OUTPUT:
    - Printed prediction with completion/risk probabilities

BEGINNER CONCEPTS:
    - The model outputs a PROBABILITY between 0 and 1.
      Completion probability = 0.85 means the model thinks there's an
      85% chance this trial will complete.
    - Risk probability is simply 1 - completion probability.
    - We fetch LIVE data from the API, so this works on any trial,
      even ones the model has never seen.

DATA SOURCE:
    This project uses data from ClinicalTrials.gov (https://clinicaltrials.gov/).

⚠️ DISCLAIMER:
    This model predicts trial completion/termination risk based on public
    ClinicalTrials.gov metadata. It does NOT predict treatment safety,
    effectiveness, clinical success, regulatory approval, or investment value.
"""

import os
import sys
import requests
import pandas as pd
import joblib

# Ensure stdout handles UTF-8 (emojis, etc.) on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Make sure we can import from the src/ package regardless of how the
# script is invoked (python src/predict.py OR python -m src.predict)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.clinicaltrials_api import flatten_study


from src.config import (
    DRUG_MODEL_PATH as MODEL_PATH,
    THERAPEUTIC_AREA_MAP,
    INDUSTRY_KEYWORDS,
    CLASSIFICATION_THRESHOLD,
)
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"



# ==============================================================================
# FUNCTION 1: Fetch a single trial by NCT ID
# ==============================================================================

def fetch_study_by_nct_id(nct_id):
    """
    Fetch a single clinical trial record from the ClinicalTrials.gov API.

    Uses the single-study endpoint:
        https://clinicaltrials.gov/api/v2/studies/{nct_id}

    Parameters
    ----------
    nct_id : str
        The NCT identifier (e.g., "NCT04280705").

    Returns
    -------
    dict or None
        The raw study record (nested dictionary), or None if the request fails.
    """
    url = f"{API_BASE_URL}/{nct_id}"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"  ❌ Trial '{nct_id}' not found on ClinicalTrials.gov.")
        else:
            print(f"  ❌ HTTP error fetching '{nct_id}': {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Network error fetching '{nct_id}': {e}")
        return None


# ==============================================================================
# FUNCTION 2: Prepare trial data for prediction
# ==============================================================================

def prepare_for_prediction(study_dict):
    """
    Convert a raw API study record into a one-row DataFrame
    that the trained model can process.

    This reuses flatten_study() from clinicaltrials_api.py so the data
    is processed exactly the same way as during training, then applies
    the same Phase 1 feature engineering as preprocess.py.

    Parameters
    ----------
    study_dict : dict
        Raw study record from the API.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with all the columns the model expects.
    """
    # Flatten the nested JSON into a simple dictionary
    flat = flatten_study(study_dict)

    # For single NCT ID prediction, set search_query_source to "user_input"
    if not flat.get("search_query_source") or flat["search_query_source"] == "user_input":
        flat["search_query_source"] = "user_input"

    # Convert to a one-row DataFrame
    df = pd.DataFrame([flat])

    # Create combined_text exactly the same way as in preprocess.py
    text_cols = ["brief_summary", "eligibility_criteria", "primary_outcome_measures"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["combined_text"] = (
        df["brief_summary"] + " " +
        df["eligibility_criteria"] + " " +
        df["primary_outcome_measures"]
    ).str.strip()

    # ------------------------------------------------------------------
    # Phase 1 Feature Engineering (must match preprocess.py exactly)
    # ------------------------------------------------------------------

    # start_year
    if "start_date" in df.columns:
        df["start_date_parsed"] = pd.to_datetime(df["start_date"], errors="coerce")
        df["start_year"] = df["start_date_parsed"].dt.year
        df.drop(columns=["start_date_parsed"], inplace=True)
    else:
        df["start_year"] = None
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")

    # Ensure numeric types
    df["enrollment_count"] = pd.to_numeric(df["enrollment_count"], errors="coerce")
    df["location_count"] = pd.to_numeric(df["location_count"], errors="coerce")
    df["collaborator_count"] = pd.to_numeric(df["collaborator_count"], errors="coerce")

    # country_count
    df["country_count"] = df["countries"].fillna("").apply(
        lambda x: len([c for c in str(x).split("|") if c.strip()]) if x else 0
    )

    # condition_count
    df["condition_count"] = df["conditions"].fillna("").apply(
        lambda x: len([c for c in str(x).split("|") if c.strip()]) if x else 0
    )

    # intervention_count
    df["intervention_count"] = df["intervention_names"].fillna("").apply(
        lambda x: len([n for n in str(x).split("|") if n.strip()]) if x else 0
    )

    # enrollment_per_location
    df["enrollment_per_location"] = (
        df["enrollment_count"] / df["location_count"].clip(lower=1)
    )
    df["enrollment_per_location"] = df["enrollment_per_location"].fillna(0)

    # has_multiple_countries
    df["has_multiple_countries"] = (df["country_count"] > 1).astype(int)

    # is_industry_sponsored
    def _check_industry(name):
        if pd.isna(name) or not name:
            return 0
        name_lower = str(name).lower()
        return int(any(kw in name_lower for kw in INDUSTRY_KEYWORDS))

    df["is_industry_sponsored"] = df["sponsor_name"].apply(_check_industry)

    # is_randomized
    df["is_randomized"] = (
        df["allocation"].fillna("").str.upper().str.contains("RANDOM")
    ).astype(int)

    # is_blinded
    df["is_blinded"] = (
        df["masking"].fillna("").str.upper().apply(
            lambda x: 0 if x in ("", "NONE") else 1
        )
    ).astype(int)

    # therapeutic_area_group
    df["therapeutic_area_group"] = (
        df["search_query_source"]
        .fillna("unknown")
        .str.lower()
        .map(THERAPEUTIC_AREA_MAP)
        .fillna("other")
    )

    # text length features
    df["text_length_summary"] = df["brief_summary"].fillna("").astype(str).str.len()
    df["text_length_eligibility"] = df["eligibility_criteria"].fillna("").astype(str).str.len()
    df["text_length_outcomes"] = df["primary_outcome_measures"].fillna("").astype(str).str.len()

    return df, flat


# ==============================================================================
# FUNCTION 3: Main prediction function
# ==============================================================================

def predict_nct_id(nct_id):
    """
    Predict whether a clinical trial is likely to complete or be at risk.

    Steps:
        1. Load the saved model
        2. Fetch the trial from ClinicalTrials.gov
        3. Flatten and prepare the data
        4. Run the prediction
        5. Print the results

    Parameters
    ----------
    nct_id : str
        The NCT identifier (e.g., "NCT04280705").
    """
    print("=" * 60)
    print("  🔮 Clinical Trial Completion Risk Predictor")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load the trained model
    # ------------------------------------------------------------------
    print(f"\n  Loading model from: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Model file not found: {MODEL_PATH}")
        print(f"     Run 'python src/train_model.py' first to train a model.")
        return

    model = joblib.load(MODEL_PATH)
    print(f"  ✓ Model loaded successfully")

    # ------------------------------------------------------------------
    # Step 2: Fetch the trial from the API
    # ------------------------------------------------------------------
    print(f"\n  Fetching trial: {nct_id}")

    study = fetch_study_by_nct_id(nct_id)
    if study is None:
        return

    print(f"  ✓ Trial data fetched")

    # ------------------------------------------------------------------
    # Step 3: Prepare data for prediction
    # ------------------------------------------------------------------
    df, flat = prepare_for_prediction(study)

    # ------------------------------------------------------------------
    # Step 4: Make the prediction
    # ------------------------------------------------------------------
    try:
        completion_prob = model.predict_proba(df)[0][1]  # Probability of class 1
        risk_prob = 1.0 - completion_prob                # Probability of class 0
    except Exception as e:
        print(f"\n  ❌ Prediction failed: {e}")
        print(f"     This may happen if the trial data has unexpected values.")
        return

    # ------------------------------------------------------------------
    # Step 5: Display results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  📊 Prediction Results")
    print("=" * 60)

    print(f"\n  NCT ID:           {flat.get('nct_id', 'N/A')}")
    print(f"  Brief Title:      {flat.get('brief_title', 'N/A')}")
    print(f"  Current Status:   {flat.get('overall_status', 'N/A')}")
    print(f"  Study Type:       {flat.get('study_type', 'N/A')}")
    print(f"  Phase:            {flat.get('phases', 'N/A')}")
    print(f"  Sponsor:          {flat.get('sponsor_name', 'N/A')}")
    print(f"  Sponsor Class:    {flat.get('sponsor_class', 'N/A')}")
    print(f"  Enrollment:       {flat.get('enrollment_count', 'N/A')}")
    print(f"  Locations:        {flat.get('location_count', 'N/A')}")

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │         MODEL PREDICTION             │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Completion Probability: {completion_prob:>8.1%}    │")
    print(f"  │  Risk Probability:       {risk_prob:>8.1%}    │")
    print(f"  │  Decision Threshold:     {CLASSIFICATION_THRESHOLD:>8.1%}    │")
    print(f"  └─────────────────────────────────────┘")

    if completion_prob >= CLASSIFICATION_THRESHOLD:
        print(f"\n  💚 The model predicts this trial is LIKELY TO COMPLETE (Low Risk).")
    else:
        print(f"\n  🔴 The model predicts this trial is AT RISK of non-completion.")

    # ------------------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------------------
    print(f"\n  {'─' * 56}")
    print(f"  ⚠️  DISCLAIMER: This model predicts trial completion/")
    print(f"  termination risk based on public ClinicalTrials.gov")
    print(f"  metadata. It does NOT predict treatment safety,")
    print(f"  effectiveness, clinical success, regulatory approval,")
    print(f"  or investment value.")
    print(f"  {'─' * 56}")

    return {
        "nct_id": flat.get("nct_id"),
        "brief_title": flat.get("brief_title"),
        "overall_status": flat.get("overall_status"),
        "completion_probability": round(completion_prob, 4),
        "risk_probability": round(risk_prob, 4),
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================
# Run this to test a prediction with a sample NCT ID.
# NCT04280705 is a cancer-related trial — a good test case.

if __name__ == "__main__":
    predict_nct_id("NCT04280705")
