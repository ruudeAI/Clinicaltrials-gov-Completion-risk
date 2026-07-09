"""
Streamlit dashboard for the ClinicalTrials.gov modeling project.

Version 1 remains the original completion-risk dashboard.
Version 2 adds a separate success-proxy and duration dashboard using the
already-trained V2 baseline models.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASSIFICATION_THRESHOLD, DRUG_MODEL_PATH as V1_MODEL_PATH
from src.chembl_api import load_cache as load_chembl_cache
from src.chembl_api import lookup_drug
from src.predict import fetch_study_by_nct_id, prepare_for_prediction
from src.v2_enrich_chembl import derive_drug_modality


V2_SUCCESS_MODEL_PATH = PROJECT_ROOT / "models" / "v2_success_classifier.joblib"
V2_DURATION_MODEL_PATH = PROJECT_ROOT / "models" / "v2_duration_regressor.joblib"
V2_MANUAL_SUCCESS_MODEL_PATH = PROJECT_ROOT / "models" / "v2_manual_success_classifier.joblib"
V2_MANUAL_DURATION_MODEL_PATH = PROJECT_ROOT / "models" / "v2_manual_duration_regressor.joblib"
V2_MODELING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v2_modeling_dataset.csv"
V2_SUCCESS_REPORT_PATH = PROJECT_ROOT / "reports" / "v2_success_results.md"
V2_DURATION_REPORT_PATH = PROJECT_ROOT / "reports" / "v2_duration_results.md"

SPONSOR_HISTORY_COLUMNS = [
    "sponsor_prior_trials",
    "sponsor_prior_completed_trials",
    "sponsor_prior_failed_trials",
    "sponsor_prior_completion_rate",
    "sponsor_prior_phase_trials",
    "sponsor_prior_phase_completion_rate",
    "sponsor_prior_avg_duration_days",
    "sponsor_has_prior_history",
]

ENDPOINT_COLUMNS = [
    "endpoint_safety",
    "endpoint_efficacy",
    "endpoint_survival",
    "endpoint_biomarker",
    "endpoint_response_rate",
]

MODALITY_TO_FALLBACK_MOLECULE_TYPE = {
    "small_molecule": "Small molecule",
    "biologic_antibody": "Antibody",
    "biologic_protein": "Protein",
    "oligonucleotide": "Oligonucleotide",
    "vaccine": "Vaccine",
    "cell_or_gene_therapy": "Cell or gene therapy",
    "unknown": "unknown",
}

TEXT_COLUMNS = [
    "brief_summary",
    "eligibility_criteria",
    "primary_outcome_measures",
    "primary_outcome_timeframes",
    "secondary_outcome_measures",
]


st.set_page_config(
    page_title="ClinicalTrials.gov Predictors",
    page_icon=":bar_chart:",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container { max-width: 1180px; padding-top: 2rem; }
    h1 { color: #1e293b; }
    h2, h3 { color: #334155; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_v1_model():
    """Load the Version 1 completion-risk model."""
    return joblib.load(V1_MODEL_PATH)


@st.cache_resource
def load_v2_models():
    """Load trained Version 2 success and duration models."""
    return {
        "success": joblib.load(V2_SUCCESS_MODEL_PATH),
        "duration": joblib.load(V2_DURATION_MODEL_PATH),
    }


@st.cache_resource
def load_v2_manual_models():
    """Load manual-friendly Version 2 models."""
    return {
        "success": joblib.load(V2_MANUAL_SUCCESS_MODEL_PATH),
        "duration": joblib.load(V2_MANUAL_DURATION_MODEL_PATH),
    }


@st.cache_data
def load_v2_dataset() -> pd.DataFrame:
    """Load the final V2 modeling dataset."""
    df = pd.read_csv(V2_MODELING_DATA_PATH)
    if "prediction_date" in df.columns:
        df["_prediction_date_for_sort"] = pd.to_datetime(
            df["prediction_date"],
            errors="coerce",
            format="mixed",
        )
    else:
        df["_prediction_date_for_sort"] = pd.NaT
    return df


def fmt_val(value) -> str:
    """Format a scalar value for display."""
    if value is None:
        return "Not available"
    if isinstance(value, float) and np.isnan(value):
        return "Not available"
    text = str(value).strip()
    if text == "" or text.upper() in {"NAN", "NA", "N/A", "NONE", "UNKNOWN"}:
        return "Not available"
    return text


def normalize_text(value) -> str:
    """Normalize simple free-text values for matching and model input."""
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    return " ".join(text.split()) if text else "unknown"


def as_int_flag(value) -> int:
    """Convert a flag-like value to 0/1 for display."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return int(numeric)


def is_unknown(value) -> bool:
    """Return True when a value is missing or semantically unknown."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().upper() in {"", "UNKNOWN", "NAN", "NA", "N/A", "NONE"}


def cache_lookup_by_chembl_id(chembl_id: str) -> Dict[str, str]:
    """Find a ChEMBL cache row by ChEMBL ID without changing cache contents."""
    cleaned_id = str(chembl_id or "").strip().upper()
    if not cleaned_id:
        return {}

    cache = load_chembl_cache()
    if cache.empty or "chembl_id" not in cache.columns:
        return {}

    matches = cache[cache["chembl_id"].fillna("").str.upper() == cleaned_id]
    if matches.empty:
        return {}

    row = matches.iloc[-1].to_dict()
    return {
        "molecule_name": row.get("original_drug_name", "unknown"),
        "chembl_id": row.get("chembl_id", cleaned_id),
        "preferred_name": row.get("preferred_name", "UNKNOWN"),
        "molecule_type": row.get("molecule_type", "UNKNOWN"),
        "first_match_confidence": row.get("first_match_confidence", "cache_chembl_id_match"),
        "match_notes": row.get("match_notes", "Matched from local ChEMBL cache by ChEMBL ID."),
    }


def enrich_manual_molecule_fields(
    molecule_name: str,
    chembl_id: str,
    selected_drug_modality: str,
) -> Tuple[Dict[str, str], str]:
    """
    Populate manual-query molecule fields from ChEMBL/cache when possible.

    If ChEMBL is unavailable or returns no usable molecule type, the selected
    modality becomes the explicit fallback so manual rows do not silently lose
    modality information.
    """
    clean_name = str(molecule_name or "").strip()
    clean_chembl_id = str(chembl_id or "").strip().upper()
    result: Dict[str, str] = {}
    source_note = "fallback/default"

    if clean_name:
        lookup = lookup_drug(clean_name, retry_cached_failures=False)
        if not is_unknown(lookup.get("chembl_id")):
            result = {
                "molecule_name": clean_name,
                "chembl_id": lookup.get("chembl_id", clean_chembl_id or "UNKNOWN"),
                "preferred_name": lookup.get("preferred_name", clean_name.upper()),
                "molecule_type": lookup.get("molecule_type", "UNKNOWN"),
                "first_match_confidence": lookup.get("first_match_confidence", "chembl_lookup"),
                "match_notes": lookup.get("match_notes", "Matched using ChEMBL lookup."),
            }
            source_note = "ChEMBL lookup"

    if not result and clean_chembl_id:
        result = cache_lookup_by_chembl_id(clean_chembl_id)
        if result:
            source_note = "ChEMBL lookup"

    if not result:
        result = {
            "molecule_name": clean_name or "unknown",
            "chembl_id": clean_chembl_id or "unknown",
            "preferred_name": clean_name.upper() if clean_name else "unknown",
            "molecule_type": "unknown",
            "first_match_confidence": "manual_entry",
            "match_notes": "Manual molecule fields; ChEMBL match was unavailable or not found.",
        }
        source_note = "user input" if clean_name or clean_chembl_id else "fallback/default"

    if clean_chembl_id and is_unknown(result.get("chembl_id")):
        result["chembl_id"] = clean_chembl_id
    if clean_name and is_unknown(result.get("molecule_name")):
        result["molecule_name"] = clean_name
    if clean_name and is_unknown(result.get("preferred_name")):
        result["preferred_name"] = clean_name.upper()

    if is_unknown(result.get("molecule_type")) and selected_drug_modality != "unknown":
        result["molecule_type"] = MODALITY_TO_FALLBACK_MOLECULE_TYPE.get(
            selected_drug_modality,
            "unknown",
        )
        source_note = f"{source_note} with selected modality fallback"

    derived_modality = derive_drug_modality(
        result.get("molecule_name"),
        result.get("molecule_type"),
        result.get("chembl_id"),
    )
    if derived_modality == "unknown" and selected_drug_modality != "unknown":
        result["drug_modality"] = selected_drug_modality
    else:
        result["drug_modality"] = derived_modality

    return result, source_note


def existing_sponsor_options(v2_df: pd.DataFrame) -> List[str]:
    """Return sorted existing sponsor names for the manual query selector."""
    if "lead_sponsor_normalized" not in v2_df.columns:
        return []
    sponsors = (
        v2_df["lead_sponsor_normalized"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    sponsors = sponsors[
        ~sponsors.str.upper().isin({"", "UNKNOWN", "NAN", "NA", "N/A", "NONE"})
    ]
    return sorted(sponsors.unique())


def model_feature_names(model) -> List[str]:
    """Return the input columns expected by a trained sklearn pipeline."""
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return []
    return [str(name) for name in names]


def training_numeric_default(df: pd.DataFrame, column: str):
    """Median-based numeric default for model feature alignment."""
    if column not in df.columns:
        return 0
    numeric = pd.to_numeric(df[column], errors="coerce")
    median = numeric.median()
    if pd.isna(median):
        return 0
    return median


def training_text_default(df: pd.DataFrame, column: str):
    """Categorical/text default for model feature alignment."""
    if column not in df.columns:
        return "unknown"
    mode = df[column].dropna().astype(str)
    if mode.empty:
        return "unknown"
    return mode.mode().iloc[0] if not mode.mode().empty else "unknown"


def build_text_features(row: pd.Series) -> pd.Series:
    """Build the combined protocol text fields used by the saved V2 pipelines."""
    text = " ".join(fmt_val(row.get(col)) for col in TEXT_COLUMNS)
    row["combined_success_text"] = text
    row["combined_protocol_text"] = text
    return row


def align_to_model_features(
    row: pd.Series,
    model,
    training_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a one-row dataframe with the exact columns expected by a model.

    Missing numeric fields are filled from training medians. Missing categorical
    or text fields are filled with "unknown" or a training mode when available.
    """
    row = build_text_features(row.copy())
    expected = model_feature_names(model)
    missing_before_fill = [col for col in expected if col not in row.index]
    aligned: Dict[str, object] = {}

    for col in expected:
        if col in row.index and not pd.isna(row[col]):
            aligned[col] = row[col]
        elif col in ENDPOINT_COLUMNS:
            aligned[col] = 0
        elif col in {"endpoint_type_count", "has_primary_endpoint"}:
            aligned[col] = 0
        elif col.endswith("_count") or col.startswith("sponsor_prior_") or col in {
            "prediction_year",
            "enrollment_count",
            "primary_endpoint_text_length",
            "sponsor_has_prior_history",
        }:
            aligned[col] = training_numeric_default(training_df, col)
        elif col in {"combined_success_text", "combined_protocol_text"}:
            aligned[col] = ""
        else:
            aligned[col] = training_text_default(training_df, col)

    return pd.DataFrame([aligned], columns=expected), missing_before_fill


def predict_v2(row: pd.Series, training_df: pd.DataFrame) -> Tuple[float, float, List[str]]:
    """Predict V2 success probability and duration from one trial-like row."""
    models = load_v2_models()

    success_input, success_missing = align_to_model_features(
        row,
        models["success"],
        training_df,
    )
    duration_input, duration_missing = align_to_model_features(
        row,
        models["duration"],
        training_df,
    )

    success_prob = float(models["success"].predict_proba(success_input)[0][1])
    duration_days = float(models["duration"].predict(duration_input)[0])
    return success_prob, duration_days, sorted(set(success_missing + duration_missing))


def predict_v2_manual(row: pd.Series, training_df: pd.DataFrame) -> Tuple[float, float, List[str]]:
    """Predict V2 manual-query success probability and duration."""
    models = load_v2_manual_models()

    success_input, success_missing = align_to_model_features(
        row,
        models["success"],
        training_df,
    )
    duration_input, duration_missing = align_to_model_features(
        row,
        models["duration"],
        training_df,
    )

    success_prob = float(models["success"].predict_proba(success_input)[0][1])
    duration_days = float(models["duration"].predict(duration_input)[0])
    return success_prob, duration_days, sorted(set(success_missing + duration_missing))


def token_set(value) -> set:
    """Tokenize short text for lightweight similar-trial search."""
    text = normalize_text(value)
    return {token for token in text.replace("|", " ").replace(";", " ").split() if len(token) > 2}


def jaccard_similarity(left, right) -> float:
    """Return simple token-set Jaccard similarity."""
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens.union(right_tokens))


def exact_match_score(left, right) -> float:
    """Return 1.0 for normalized exact match, otherwise 0.0."""
    return 1.0 if normalize_text(left) == normalize_text(right) else 0.0


def numeric_closeness(left, right) -> float:
    """Return a bounded similarity score for numeric fields."""
    left_value = pd.to_numeric(left, errors="coerce")
    right_value = pd.to_numeric(right, errors="coerce")
    if pd.isna(left_value) or pd.isna(right_value):
        return 0.0
    denominator = max(abs(float(left_value)), abs(float(right_value)), 1.0)
    return max(0.0, 1.0 - abs(float(left_value) - float(right_value)) / denominator)


def find_similar_trials(query_row: pd.Series, v2_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Find similar historical V2 trials for Manual Query context.

    Similarity uses manual-available fields only: phase, condition text,
    modality, sponsor class, enrollment, endpoint flags, and design fields.
    """
    if v2_df.empty:
        return pd.DataFrame()

    candidates = v2_df.copy()
    scores = (
        2.0 * candidates["conditions_normalized"].apply(lambda value: jaccard_similarity(query_row.get("conditions_normalized"), value))
        + 1.5 * candidates["phase_normalized"].apply(lambda value: exact_match_score(query_row.get("phase_normalized"), value))
        + 1.2 * candidates["drug_modality"].apply(lambda value: exact_match_score(query_row.get("drug_modality"), value))
        + 0.8 * candidates["sponsor_class"].apply(lambda value: exact_match_score(query_row.get("sponsor_class"), value))
        + 0.7 * candidates["study_type"].apply(lambda value: exact_match_score(query_row.get("study_type"), value))
        + 0.6 * candidates["allocation"].apply(lambda value: exact_match_score(query_row.get("allocation"), value))
        + 0.6 * candidates["intervention_model"].apply(lambda value: exact_match_score(query_row.get("intervention_model"), value))
        + 0.6 * candidates["masking"].apply(lambda value: exact_match_score(query_row.get("masking"), value))
        + 0.6 * candidates["primary_purpose"].apply(lambda value: exact_match_score(query_row.get("primary_purpose"), value))
        + 0.8 * candidates["enrollment_count"].apply(lambda value: numeric_closeness(query_row.get("enrollment_count"), value))
    )

    for col in ENDPOINT_COLUMNS:
        if col in candidates.columns:
            scores += 0.25 * candidates[col].apply(lambda value: exact_match_score(query_row.get(col, 0), value))

    candidates["_similarity_score"] = scores
    display_cols = [
        "nct_id",
        "brief_title",
        "molecule_name",
        "phase_normalized",
        "lead_sponsor_normalized",
        "observed_duration_days",
        "target_success",
        "_similarity_score",
    ]
    for col in display_cols:
        if col not in candidates.columns:
            candidates[col] = pd.NA
    similar = candidates.sort_values("_similarity_score", ascending=False).head(top_n)
    return similar[display_cols].rename(
        columns={
            "nct_id": "NCT ID",
            "brief_title": "Title",
            "molecule_name": "Molecule",
            "phase_normalized": "Phase",
            "lead_sponsor_normalized": "Sponsor",
            "observed_duration_days": "Observed Duration Days",
            "target_success": "Target Success",
            "_similarity_score": "Similarity Score",
        }
    )


def parse_top_features_from_report(report_path: Path, heading: str, limit: int = 10) -> pd.DataFrame:
    """Parse a simple markdown top-feature table from a V2 report."""
    if not report_path.exists():
        return pd.DataFrame(columns=["feature", "value", "importance_type"])

    lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_section = False
    rows = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ") and in_section:
            break
        if stripped == heading:
            in_section = True
            continue
        if not in_section or not stripped.startswith("|"):
            continue
        if stripped.startswith("| Feature") or stripped.startswith("|---"):
            continue

        content = stripped.strip("|")
        parts = [part.strip() for part in content.rsplit("|", 2)]
        if len(parts) != 3:
            continue
        feature, value, importance_type = parts
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric_value):
            continue
        rows.append(
            {
                "feature": feature,
                "value": float(numeric_value),
                "importance_type": importance_type,
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def success_single_row_contributions(
    row: pd.Series,
    training_df: pd.DataFrame,
    success_model=None,
    top_n: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate LogisticRegression feature contributions for one success prediction."""
    if success_model is None:
        success_model = load_v2_models()["success"]
    classifier = success_model.named_steps.get("classifier")
    preprocessor = success_model.named_steps.get("preprocessor")
    if classifier is None or preprocessor is None or not hasattr(classifier, "coef_"):
        return pd.DataFrame(), pd.DataFrame()

    model_input, _ = align_to_model_features(row, success_model, training_df)
    transformed = preprocessor.transform(model_input)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    values = np.asarray(transformed).ravel()
    coefficients = np.asarray(classifier.coef_).ravel()
    feature_names = preprocessor.get_feature_names_out()
    contributions = values * coefficients

    table = pd.DataFrame(
        {
            "feature": [
                str(name).replace("num__", "").replace("cat__", "").replace("text__", "")
                for name in feature_names
            ],
            "contribution": contributions,
        }
    )
    table = table[table["contribution"].abs() > 1e-9].copy()
    if table.empty:
        return pd.DataFrame(), pd.DataFrame()

    positive = table.sort_values("contribution", ascending=False).head(top_n)
    negative = table.sort_values("contribution", ascending=True).head(top_n)
    return positive, negative


def show_global_feature_fallback(title: str, report_path: Path, heading: str) -> None:
    """Show global top features from a report, when available."""
    features = parse_top_features_from_report(report_path, heading)
    if features.empty:
        st.caption(f"{title}: no feature table was available.")
        return
    st.markdown(f"**{title}**")
    st.dataframe(features, use_container_width=True, hide_index=True)


def show_v2_prediction_explanation(row: pd.Series, training_df: pd.DataFrame, success_model=None) -> None:
    """Render V2 prediction explanation content for lookup and manual modes."""
    st.divider()
    st.subheader("Why this prediction?")
    st.warning("These feature contributions are model explanations, not causal findings.")

    positive, negative = success_single_row_contributions(row, training_df, success_model=success_model)
    if not positive.empty or not negative.empty:
        st.markdown("**Success model single-prediction contributions**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Features pushing probability higher")
            st.dataframe(positive, use_container_width=True, hide_index=True)
        with col2:
            st.caption("Features pushing probability lower")
            st.dataframe(negative, use_container_width=True, hide_index=True)
    else:
        show_global_feature_fallback(
            "Global top success-driving features",
            V2_SUCCESS_REPORT_PATH,
            "## Top Success-Driving Features",
        )

    show_global_feature_fallback(
        "Global top duration-driving features",
        V2_DURATION_REPORT_PATH,
        "## Top Duration-Driving Features",
    )


def v2_files_available() -> bool:
    """Check whether V2 app dependencies exist."""
    missing = [
        path
        for path in [
            V2_SUCCESS_MODEL_PATH,
            V2_DURATION_MODEL_PATH,
            V2_MANUAL_SUCCESS_MODEL_PATH,
            V2_MANUAL_DURATION_MODEL_PATH,
            V2_MODELING_DATA_PATH,
        ]
        if not path.exists()
    ]
    if missing:
        st.error(
            "Version 2 files are missing:\n\n"
            + "\n".join(f"- `{path}`" for path in missing)
            + "\n\nRun `venv\\Scripts\\python.exe scripts\\run_v2_pipeline.py --max-trials 3000` first."
        )
        return False
    return True


def show_v1_dashboard() -> None:
    """Render the original Version 1 completion-risk dashboard."""
    st.header("Version 1: Completion Risk")
    st.markdown(
        "This section estimates completion vs. non-completion risk using the original "
        "Version 1 model trained on ClinicalTrials.gov trial metadata."
    )
    st.warning(
        "This model is for educational and research purposes only. It does not predict "
        "treatment safety, treatment effectiveness, clinical success, regulatory approval, "
        "or investment value."
    )

    if not os.path.exists(V1_MODEL_PATH):
        st.error(
            f"Version 1 model file not found: `{V1_MODEL_PATH}`\n\n"
            "Run the Version 1 training pipeline first."
        )
        return

    st.subheader("Enter a Trial")
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        nct_id = st.text_input(
            "NCT ID",
            placeholder="e.g., NCT04280705",
            help="Enter the NCT identifier from ClinicalTrials.gov.",
            key="v1_nct_id",
        )
    with col_btn:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True, key="v1_predict")

    st.caption("Try these: `NCT04280705`, `NCT02693535`, `NCT03891108`")

    if not predict_clicked:
        return

    nct_id = nct_id.strip().upper()
    if not nct_id:
        st.error("Please enter an NCT ID.")
        return
    if not nct_id.startswith("NCT"):
        st.error("NCT IDs should start with `NCT`, for example `NCT04280705`.")
        return

    with st.spinner(f"Fetching trial data for {nct_id}..."):
        study = fetch_study_by_nct_id(nct_id)

    if study is None:
        st.error(f"Could not fetch trial `{nct_id}`. Check the ID and internet connection.")
        return

    try:
        df, flat = prepare_for_prediction(study)
        model = load_v1_model()
        completion_prob = float(model.predict_proba(df)[0][1])
        risk_prob = 1.0 - completion_prob
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    countries_raw = flat.get("countries")
    countries_formatted = str(countries_raw).replace("|", ", ") if countries_raw else None

    st.divider()
    st.subheader("Trial Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**NCT ID:** `{fmt_val(flat.get('nct_id'))}`")
        st.markdown(f"**Current Status:** {fmt_val(flat.get('overall_status'))}")
        st.markdown(f"**Study Type:** {fmt_val(flat.get('study_type'))}")
        st.markdown(f"**Phase:** {fmt_val(flat.get('phases'))}")
    with col2:
        st.markdown(f"**Sponsor Class:** {fmt_val(flat.get('sponsor_class'))}")
        st.markdown(f"**Enrollment:** {fmt_val(flat.get('enrollment_count'))}")
        st.markdown(f"**Locations:** {fmt_val(flat.get('location_count'))}")
        st.markdown(f"**Countries:** {fmt_val(countries_formatted)}")

    st.markdown(f"**Title:** {fmt_val(flat.get('brief_title'))}")

    st.divider()
    st.subheader("Prediction")
    status_upper = str(flat.get("overall_status", "")).upper()
    known_statuses = {"COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"}
    ongoing_statuses = {
        "RECRUITING",
        "NOT_YET_RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "ENROLLING_BY_INVITATION",
    }

    if status_upper in known_statuses:
        st.caption(
            "This trial already has a known outcome. The probabilities below are a "
            "retrospective model estimate from trial metadata."
        )
    elif status_upper in ongoing_statuses:
        st.caption(
            "This trial does not have a final outcome yet. The probabilities below estimate "
            "completion/risk based on historical ClinicalTrials.gov patterns."
        )
    else:
        st.caption("This trial status may not represent a final outcome. Interpret with caution.")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Completion Probability", f"{completion_prob:.1%}")
    with col_m2:
        st.metric("Risk Probability", f"{risk_prob:.1%}")

    st.markdown(
        f"**Tuned classification boundary:** completion probability >= "
        f"`{CLASSIFICATION_THRESHOLD:.1%}` is predicted as low risk."
    )
    if completion_prob >= CLASSIFICATION_THRESHOLD:
        st.success("Low risk: the model predicts this trial is likely to complete.")
    else:
        st.error("At risk: the model predicts elevated risk of non-completion.")


def display_trial_details(row: pd.Series) -> None:
    """Show V2 trial details for an existing dataset row."""
    st.subheader("Trial Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**NCT ID:** `{fmt_val(row.get('nct_id'))}`")
        st.markdown(f"**Brief title:** {fmt_val(row.get('brief_title'))}")
        st.markdown(f"**Disease indication / conditions:** {fmt_val(row.get('conditions_normalized'))}")
        st.markdown(f"**Molecule name:** {fmt_val(row.get('molecule_name'))}")
        st.markdown(f"**ChEMBL ID:** {fmt_val(row.get('chembl_id'))}")
        st.markdown(f"**Preferred molecule name:** {fmt_val(row.get('preferred_name'))}")
        st.markdown(f"**Drug modality:** {fmt_val(row.get('drug_modality'))}")
    with col2:
        st.markdown(f"**Lead sponsor:** {fmt_val(row.get('lead_sponsor_normalized'))}")
        st.markdown(f"**Phase:** {fmt_val(row.get('phase_normalized'))}")
        st.markdown(f"**Study type:** {fmt_val(row.get('study_type'))}")
        st.markdown(f"**Allocation:** {fmt_val(row.get('allocation'))}")
        st.markdown(f"**Intervention model:** {fmt_val(row.get('intervention_model'))}")
        st.markdown(f"**Masking:** {fmt_val(row.get('masking'))}")
        st.markdown(f"**Primary purpose:** {fmt_val(row.get('primary_purpose'))}")
        st.markdown(f"**Enrollment count:** {fmt_val(row.get('enrollment_count'))}")

    endpoint_summary = {
        "Safety": as_int_flag(row.get("endpoint_safety", 0)),
        "Efficacy": as_int_flag(row.get("endpoint_efficacy", 0)),
        "Survival": as_int_flag(row.get("endpoint_survival", 0)),
        "Biomarker": as_int_flag(row.get("endpoint_biomarker", 0)),
        "Response rate": as_int_flag(row.get("endpoint_response_rate", 0)),
    }
    st.markdown("**Endpoint type flags:** " + ", ".join(f"{k}: {v}" for k, v in endpoint_summary.items()))

    st.markdown("**Sponsor history summary:**")
    sponsor_cols = [col for col in SPONSOR_HISTORY_COLUMNS if col in row.index]
    st.dataframe(pd.DataFrame([row[sponsor_cols]]), use_container_width=True, hide_index=True)


def show_existing_trial_lookup(v2_df: pd.DataFrame) -> None:
    """Render V2 existing-trial lookup mode."""
    st.subheader("Existing Trial Lookup")
    if "nct_id" not in v2_df.columns or v2_df.empty:
        st.error("The V2 modeling dataset does not contain NCT IDs.")
        return

    nct_ids = sorted(v2_df["nct_id"].dropna().astype(str).unique())
    selected_nct = st.selectbox(
        "Search/select NCT ID from the V2 modeling dataset",
        options=nct_ids,
        index=0,
        key="v2_lookup_nct",
    )

    row = v2_df[v2_df["nct_id"].astype(str) == selected_nct].iloc[0].copy()
    if "prediction_year" not in row.index:
        date_value = pd.to_datetime(row.get("prediction_date"), errors="coerce", format="mixed")
        row["prediction_year"] = date_value.year if not pd.isna(date_value) else pd.Timestamp.today().year

    display_trial_details(row)

    try:
        success_prob, duration_days, filled_columns = predict_v2(row, v2_df)
    except Exception as exc:
        expected_success = model_feature_names(load_v2_models()["success"])
        expected_duration = model_feature_names(load_v2_models()["duration"])
        available = set(row.index)
        missing = sorted((set(expected_success) | set(expected_duration)) - available)
        st.error(
            "Version 2 prediction failed. Some expected feature columns may be missing.\n\n"
            f"Missing before fallback alignment: `{missing}`\n\n"
            f"Error: {exc}"
        )
        return

    st.divider()
    st.subheader("V2 Predictions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Success Probability", f"{success_prob:.1%}")
    with col2:
        st.metric("Predicted Duration", f"{duration_days:,.0f} days")
    with col3:
        actual_duration = row.get("observed_duration_days")
        st.metric("Actual Observed Duration", fmt_val(actual_duration))
    with col4:
        target = row.get("target_success")
        st.metric("Retrospective Label", fmt_val(target))

    if filled_columns:
        st.caption(
            "Some derived model fields were rebuilt or default-filled for prediction: "
            + ", ".join(filled_columns)
        )
    show_v2_prediction_explanation(row, v2_df)


def sponsor_history_for_manual(v2_df: pd.DataFrame, sponsor_name: str) -> Tuple[Dict[str, object], str]:
    """Use the most recent known sponsor-history row, or conservative defaults."""
    normalized = normalize_text(sponsor_name)
    global_completion_rate = training_numeric_default(v2_df, "sponsor_prior_completion_rate")
    global_phase_rate = training_numeric_default(v2_df, "sponsor_prior_phase_completion_rate")
    global_duration = training_numeric_default(v2_df, "sponsor_prior_avg_duration_days")

    defaults = {
        "sponsor_prior_trials": 0,
        "sponsor_prior_completed_trials": 0,
        "sponsor_prior_failed_trials": 0,
        "sponsor_prior_completion_rate": global_completion_rate,
        "sponsor_prior_phase_trials": 0,
        "sponsor_prior_phase_completion_rate": global_phase_rate,
        "sponsor_prior_avg_duration_days": global_duration,
        "sponsor_has_prior_history": 0,
    }

    if "lead_sponsor_normalized" not in v2_df.columns or not sponsor_name.strip():
        return defaults, "Sponsor not found in V2 history; neutral defaults used."

    matches = v2_df[v2_df["lead_sponsor_normalized"].astype(str).str.lower().str.strip() == normalized].copy()
    if matches.empty:
        return defaults, "Sponsor not found in V2 history; neutral defaults used."

    matches = matches.sort_values("_prediction_date_for_sort", kind="mergesort")
    latest = matches.iloc[-1]
    history = defaults.copy()
    for col in SPONSOR_HISTORY_COLUMNS:
        if col in latest.index and not pd.isna(latest[col]):
            history[col] = latest[col]
    history["sponsor_has_prior_history"] = 1
    return history, f"Sponsor history matched from V2 dataset: {fmt_val(latest.get('lead_sponsor_normalized'))}."


def show_manual_trial_query(v2_df: pd.DataFrame) -> None:
    """Render V2 manual query mode."""
    st.subheader("Manual Trial Query")
    st.warning("Manual predictions are less reliable when fields are missing or estimated.")
    st.info(
        "Manual Query uses a separate manual-friendly model and similar historical trials "
        "for context. It is not clinical, regulatory, or investment advice."
    )
    sponsor_options = existing_sponsor_options(v2_df)
    sponsor_select_options = ["Custom sponsor"] + sponsor_options

    with st.form("v2_manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            condition_text = st.text_input("Disease indication / condition text", value="")
            molecule_name = st.text_input("Molecule name", value="")
            chembl_id = st.text_input("Optional ChEMBL ID", value="")
            drug_modality = st.selectbox(
                "Drug modality",
                [
                    "small_molecule",
                    "biologic_antibody",
                    "biologic_protein",
                    "oligonucleotide",
                    "vaccine",
                    "cell_or_gene_therapy",
                    "unknown",
                ],
            )
            selected_sponsor = st.selectbox(
                "Lead sponsor from V2 history",
                sponsor_select_options,
                help="Select an existing sponsor to reuse its most recent prior-history features.",
            )
            custom_sponsor = st.text_input(
                "Custom sponsor text",
                value="",
                help="Used when `Custom sponsor` is selected, or to test a sponsor not in V2 history.",
            )
            sponsor_class = st.selectbox("Sponsor class", ["INDUSTRY", "OTHER", "NIH", "FED", "UNKNOWN"])
            phase = st.selectbox(
                "Phase",
                [
                    "EARLY_PHASE1",
                    "PHASE1",
                    "PHASE1_PHASE2",
                    "PHASE2",
                    "PHASE2_PHASE3",
                    "PHASE3",
                    "PHASE4",
                    "UNKNOWN",
                ],
            )
        with col2:
            study_type = st.selectbox("Study type", ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS", "UNKNOWN"])
            allocation = st.selectbox("Allocation", ["RANDOMIZED", "NON_RANDOMIZED", "NA", "UNKNOWN"])
            intervention_model = st.selectbox(
                "Intervention model",
                ["PARALLEL", "SINGLE_GROUP", "CROSSOVER", "FACTORIAL", "SEQUENTIAL", "UNKNOWN"],
            )
            masking = st.selectbox("Masking", ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE", "UNKNOWN"])
            primary_purpose = st.selectbox(
                "Primary purpose",
                ["TREATMENT", "PREVENTION", "DIAGNOSTIC", "SUPPORTIVE_CARE", "SCREENING", "BASIC_SCIENCE", "UNKNOWN"],
            )
            enrollment_count = st.number_input("Enrollment count", min_value=0, value=100, step=10)
            enrollment_type = st.selectbox("Enrollment type", ["ACTUAL", "ESTIMATED", "UNKNOWN"])

        count_col1, count_col2, count_col3 = st.columns(3)
        with count_col1:
            country_count = st.number_input("Country count", min_value=0, value=1, step=1)
        with count_col2:
            location_count = st.number_input("Location count", min_value=0, value=1, step=1)
        with count_col3:
            collaborator_count = st.number_input("Collaborator count", min_value=0, value=0, step=1)

        st.markdown("Endpoint type")
        ep1, ep2, ep3, ep4, ep5 = st.columns(5)
        endpoint_safety = ep1.checkbox("Safety")
        endpoint_efficacy = ep2.checkbox("Efficacy")
        endpoint_survival = ep3.checkbox("Survival")
        endpoint_biomarker = ep4.checkbox("Biomarker")
        endpoint_response = ep5.checkbox("Response-rate")

        submitted = st.form_submit_button("Predict V2 Success and Duration", type="primary")

    if not submitted:
        return

    endpoint_values = {
        "endpoint_safety": int(endpoint_safety),
        "endpoint_efficacy": int(endpoint_efficacy),
        "endpoint_survival": int(endpoint_survival),
        "endpoint_biomarker": int(endpoint_biomarker),
        "endpoint_response_rate": int(endpoint_response),
    }
    endpoint_type_count = sum(endpoint_values.values())
    lead_sponsor = custom_sponsor.strip() if selected_sponsor == "Custom sponsor" else selected_sponsor
    sponsor_history, sponsor_history_note = sponsor_history_for_manual(v2_df, lead_sponsor)
    molecule_fields, molecule_source_note = enrich_manual_molecule_fields(
        molecule_name=molecule_name,
        chembl_id=chembl_id,
        selected_drug_modality=drug_modality,
    )

    manual_row = pd.Series(
        {
            "prediction_year": pd.Timestamp.today().year,
            "conditions_normalized": normalize_text(condition_text),
            "condition_count": max(1, len([part for part in condition_text.replace("|", ";").split(";") if part.strip()])),
            "search_query_source": "manual_query",
            "molecule_name": molecule_fields["molecule_name"],
            "chembl_id": molecule_fields["chembl_id"],
            "preferred_name": molecule_fields["preferred_name"],
            "molecule_type": molecule_fields["molecule_type"],
            "drug_modality": molecule_fields["drug_modality"],
            "first_match_confidence": molecule_fields["first_match_confidence"],
            "lead_sponsor_normalized": normalize_text(lead_sponsor),
            "sponsor_class": sponsor_class,
            "phase_normalized": phase,
            "study_type": study_type,
            "allocation": allocation,
            "intervention_model": intervention_model,
            "masking": masking,
            "primary_purpose": primary_purpose,
            "enrollment_count": enrollment_count,
            "enrollment_type": enrollment_type,
            "drug_intervention_count": 1 if molecule_name.strip() else 0,
            "collaborator_count": collaborator_count,
            "location_count": location_count,
            "country_count": country_count,
            "countries": "manual_unknown",
            "primary_endpoint_text_length": 0,
            "endpoint_type_count": endpoint_type_count,
            "has_primary_endpoint": int(endpoint_type_count > 0),
            "brief_summary": condition_text,
            "eligibility_criteria": "",
            "primary_outcome_measures": "",
            "primary_outcome_timeframes": "",
            "secondary_outcome_measures": "",
            **endpoint_values,
            **sponsor_history,
        }
    )

    try:
        success_prob, duration_days, filled_columns = predict_v2_manual(manual_row, v2_df)
    except Exception as exc:
        expected_success = model_feature_names(load_v2_manual_models()["success"])
        expected_duration = model_feature_names(load_v2_manual_models()["duration"])
        available = set(manual_row.index)
        missing = sorted((set(expected_success) | set(expected_duration)) - available)
        st.error(
            "Manual V2 prediction failed because model features did not align.\n\n"
            f"Missing before fallback alignment: `{missing}`\n\n"
            f"Error: {exc}"
        )
        return

    st.divider()
    st.subheader("Manual Query Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Success Probability", f"{success_prob:.1%}")
    with col2:
        st.metric("Predicted Duration", f"{duration_days:,.0f} days")

    st.info(f"Molecule fields source: {molecule_source_note}.")
    if sponsor_history["sponsor_has_prior_history"]:
        st.success(sponsor_history_note)
    else:
        st.warning(sponsor_history_note)

    sponsor_summary_cols = [
        "sponsor_prior_trials",
        "sponsor_prior_completed_trials",
        "sponsor_prior_failed_trials",
        "sponsor_prior_completion_rate",
        "sponsor_has_prior_history",
    ]
    st.markdown("**Sponsor-history summary used for prediction:**")
    st.dataframe(
        pd.DataFrame([{col: sponsor_history.get(col, "unknown") for col in sponsor_summary_cols}]),
        use_container_width=True,
        hide_index=True,
    )
    if filled_columns:
        st.caption("Default-filled model fields: " + ", ".join(filled_columns))

    debug_columns = [
        "molecule_name",
        "chembl_id",
        "preferred_name",
        "molecule_type",
        "drug_modality",
        "phase_normalized",
        *SPONSOR_HISTORY_COLUMNS,
        *ENDPOINT_COLUMNS,
        "endpoint_type_count",
        "has_primary_endpoint",
    ]
    debug_columns = [col for col in debug_columns if col in manual_row.index]
    with st.expander("Final manual model row fields"):
        st.dataframe(pd.DataFrame([manual_row[debug_columns]]), use_container_width=True, hide_index=True)
        st.caption(f"ChEMBL/molecule notes: {molecule_fields.get('match_notes', 'Not available')}")

    similar_trials = find_similar_trials(manual_row, v2_df)
    st.divider()
    st.subheader("Similar Historical Trials")
    st.caption(
        "Similarity uses manual-available fields: phase, disease text, drug modality, "
        "sponsor class, enrollment, endpoint flags, and trial design."
    )
    st.dataframe(similar_trials, use_container_width=True, hide_index=True)

    show_v2_prediction_explanation(
        manual_row,
        v2_df,
        success_model=load_v2_manual_models()["success"],
    )


def show_v2_dashboard() -> None:
    """Render the Version 2 success and duration dashboard."""
    st.header("Version 2: Clinical Trial Success + Duration Prediction")
    st.markdown(
        "This section uses the separate V2 pipeline to estimate a conservative "
        "success-proxy probability and trial duration in days."
    )
    st.warning(
        "Success prediction uses a conservative phase-progression proxy label. "
        "Duration prediction is an estimate from public registry metadata. "
        "This is not medical, regulatory, or investment advice."
    )

    if not v2_files_available():
        return

    try:
        v2_df = load_v2_dataset()
        load_v2_models()
    except Exception as exc:
        st.error(f"Could not load Version 2 dataset/models: {exc}")
        return

    mode = st.radio(
        "Version 2 mode",
        ["Existing Trial Lookup", "Manual Trial Query"],
        horizontal=True,
    )

    if mode == "Existing Trial Lookup":
        show_existing_trial_lookup(v2_df)
    else:
        show_manual_trial_query(v2_df)


def main() -> None:
    """Main Streamlit entry point."""
    st.title("ClinicalTrials.gov Trial Prediction Dashboard")
    st.markdown(
        "Version 1 estimates completion risk. Version 2 estimates a conservative "
        "success-proxy probability and trial duration."
    )

    v1_tab, v2_tab = st.tabs(["Version 1: Completion Risk", "Version 2: Success + Duration"])
    with v1_tab:
        show_v1_dashboard()
    with v2_tab:
        show_v2_dashboard()

    st.divider()
    st.caption(
        "Data source: ClinicalTrials.gov. V2 molecule enrichment uses ChEMBL. "
        "Educational and research portfolio use only."
    )


if __name__ == "__main__":
    main()
