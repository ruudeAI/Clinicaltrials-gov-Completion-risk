"""
streamlit_app.py — Interactive web dashboard for clinical trial completion prediction.

WHAT THIS FILE DOES:
    Provides a simple web interface where a user can:
    1. Enter an NCT ID for any clinical trial
    2. Click "Predict" to fetch live data from ClinicalTrials.gov
    3. See the trial's metadata and predicted completion/risk probabilities

HOW TO RUN:
    streamlit run app/streamlit_app.py

REQUIRES:
    - Trained model at models/clinical_trial_completion_model.joblib
    - Internet connection (to fetch trial data from the API)

DATA SOURCE:
    This project uses data from ClinicalTrials.gov (https://clinicaltrials.gov/).
"""

import os
import sys
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Fix imports: ensure we can import from src/ regardless of how Streamlit
# runs this file. We add the project root to Python's module search path.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.clinicaltrials_api import flatten_study
from src.predict import fetch_study_by_nct_id, prepare_for_prediction


# ==============================================================================
# CONSTANTS
# ==============================================================================

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "clinical_trial_completion_model.joblib")


# ==============================================================================
# PAGE CONFIG & STYLING
# ==============================================================================

st.set_page_config(
    page_title="Clinical Trial Completion Risk Predictor",
    page_icon="🏥",
    layout="centered",
)

# Custom CSS for a cleaner, more polished look
st.markdown("""
<style>
    /* Main container */
    .block-container { max-width: 800px; padding-top: 2rem; }

    /* Header styling */
    h1 { color: #1e293b; }
    h2 { color: #334155; margin-top: 1.5rem; }

    /* Info cards */
    .trial-info {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }

    /* Metric highlights */
    [data-testid="stMetricValue"] { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# APP LAYOUT
# ==============================================================================

def main():
    """Main Streamlit app function."""

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("🏥 Clinical Trial Completion Risk Predictor")

    st.markdown(
        "This tool uses a machine learning model trained on historical "
        "[ClinicalTrials.gov](https://clinicaltrials.gov/) cancer trial metadata "
        "to estimate completion/termination risk for a trial."
    )

    # ------------------------------------------------------------------
    # Disclaimer (always visible)
    # ------------------------------------------------------------------
    st.warning(
        "⚠️ **Disclaimer:** This model is for **educational and research purposes only**. "
        "It does not predict treatment safety, effectiveness, clinical success, "
        "regulatory approval, or investment value."
    )

    # ------------------------------------------------------------------
    # Check if model exists
    # ------------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"🚫 **Model file not found.**\n\n"
            f"Expected path: `{MODEL_PATH}`\n\n"
            f"Please run the training pipeline first:\n"
            f"```\n"
            f"python src/clinicaltrials_api.py\n"
            f"python src/preprocess.py\n"
            f"python src/train_model.py\n"
            f"```"
        )
        return

    # ------------------------------------------------------------------
    # Input Section
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Enter a Trial")

    col_input, col_btn = st.columns([3, 1])
    with col_input:
        nct_id = st.text_input(
            "NCT ID",
            placeholder="e.g., NCT04280705",
            label_visibility="collapsed",
            help="Enter the NCT identifier from ClinicalTrials.gov",
        )
    with col_btn:
        predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

    # Example NCT IDs for quick testing
    st.caption(
        "💡 Try these: `NCT04280705` · `NCT02693535` · `NCT03891108`"
    )

    # ------------------------------------------------------------------
    # Prediction Logic
    # ------------------------------------------------------------------
    if predict_clicked:
        # Validate input
        nct_id = nct_id.strip().upper()
        if not nct_id:
            st.error("Please enter an NCT ID.")
            return
        if not nct_id.startswith("NCT"):
            st.error("NCT IDs should start with 'NCT' (e.g., NCT04280705).")
            return

        # Fetch trial data from the API
        with st.spinner(f"Fetching trial data for {nct_id}..."):
            study = fetch_study_by_nct_id(nct_id)

        if study is None:
            st.error(
                f"❌ Could not fetch trial **{nct_id}**.\n\n"
                f"Please check that the NCT ID is valid and you have an internet connection."
            )
            return

        # Prepare data and predict
        try:
            df, flat = prepare_for_prediction(study)
            model = joblib.load(MODEL_PATH)
            completion_prob = model.predict_proba(df)[0][1]
            risk_prob = 1.0 - completion_prob
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            return

        # ------------------------------------------------------------------
        # Display Results
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Trial Information")

        # Format helper for missing/blank values
        def fmt_val(val):
            if val is None or str(val).strip() == "" or str(val).strip().upper() in ["NONE", "NAN", "N/A"]:
                return "Not available"
            return str(val)

        # Nicely format countries using commas instead of pipes
        countries_raw = flat.get("countries")
        if countries_raw:
            countries_formatted = str(countries_raw).replace("|", ", ")
        else:
            countries_formatted = None

        nct_id_display = fmt_val(flat.get("nct_id"))
        status_display = fmt_val(flat.get("overall_status"))
        study_type_display = fmt_val(flat.get("study_type"))
        phase_display = fmt_val(flat.get("phases"))
        sponsor_class_display = fmt_val(flat.get("sponsor_class"))
        enrollment_display = fmt_val(flat.get("enrollment_count"))
        location_display = fmt_val(flat.get("location_count"))
        countries_display = fmt_val(countries_formatted)
        title_display = fmt_val(flat.get("brief_title"))

        # Trial metadata in two clean columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**NCT ID:** `{nct_id_display}`")
            st.markdown(f"**Current Status:** {status_display}")
            st.markdown(f"**Study Type:** {study_type_display}")
            st.markdown(f"**Phase:** {phase_display}")
        with col2:
            st.markdown(f"**Sponsor Class:** {sponsor_class_display}")
            st.markdown(f"**Enrollment:** {enrollment_display}")
            st.markdown(f"**Locations:** {location_display}")
            st.markdown(f"**Countries:** {countries_display}")

        # Title (full width, wraps cleanly)
        st.markdown(f"**Title:** {title_display}")

        # ------------------------------------------------------------------
        # Prediction Section
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Prediction")

        # Determine logic for known vs ongoing vs caution statuses
        status_upper = str(flat.get("overall_status", "")).upper()
        known_statuses = ["COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"]
        ongoing_statuses = ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]

        if status_upper in known_statuses:
            st.markdown("### **Known Outcome**")
            st.markdown(f"**Actual Status:** `{status_display}`")
            st.markdown("#### *Retrospective Model Estimate*")
            st.caption(
                "This trial already has a known outcome. The probabilities below show what the "
                "model estimated from the trial metadata, not a future prediction."
            )
        elif status_upper in ongoing_statuses:
            st.markdown("### **Prediction for Ongoing Trial**")
            st.caption(
                "This trial does not have a final outcome yet. The probabilities below estimate "
                "completion/risk based on historical ClinicalTrials.gov patterns."
            )
        else:
            st.markdown("### **Status requires caution**")
            st.caption(
                "This trial status may not represent a final outcome. The probabilities below should "
                "be interpreted with caution."
            )

        # Show two large metric cards
        completion_pct = f"{completion_prob:.1%}"
        risk_pct = f"{risk_prob:.1%}"

        st.markdown("<br>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(
                label="Completion Probability",
                value=completion_pct,
                help="Estimated probability that this trial will complete as planned.",
            )
        with col_m2:
            st.metric(
                label="Risk Probability",
                value=risk_pct,
                help="Estimated probability of termination, withdrawal, or suspension.",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk level indicator (professional, clean, clear spacing)
        if completion_prob >= 0.7:
            st.success("**Low Risk** — The model predicts this trial is likely to complete.")
        elif completion_prob >= 0.4:
            st.info("**Moderate Risk** — The model sees some risk factors for this trial.")
        else:
            st.error("**High Risk** — The model predicts elevated risk of non-completion.")

        # Concise explanation box under the prediction (modern, elegant style)
        st.markdown("""
<div style="background-color: rgba(248, 250, 252, 0.05); border: 1px solid rgba(226, 232, 240, 0.2); border-radius: 0.5rem; padding: 1rem; margin-top: 1.5rem; font-size: 0.9rem;">
    This estimate is based on public ClinicalTrials.gov metadata such as phase, sponsor class, enrollment, study type, locations, intervention type, eligibility criteria, trial summary, and outcome measures.
</div>
""", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Footer and Disclaimer at the very bottom of the page
    # ------------------------------------------------------------------
    st.markdown("---")

    # Visible Disclaimer at the bottom in a clean warning box
    st.warning(
        "This project is for educational and research purposes only. "
        "It does not predict treatment safety, treatment effectiveness, clinical success, "
        "regulatory approval, patient-care decisions, or investment value."
    )

    st.caption(
        "**Data source:** [ClinicalTrials.gov](https://clinicaltrials.gov/) · "
        "**Purpose:** Educational & research portfolio · "
        "**Model:** Logistic Regression with TF-IDF"
    )



# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
