"""
v2_build_modeling_dataset.py - Build the final V2 modeling dataset.

Purpose
-------
Combine V2 duration targets, conservative success labels, ChEMBL molecule and
modality fields, endpoint features, sponsor history, and non-leaky trial
metadata into one modeling-ready CSV.

Inputs
------
- `data/interim/v2_trials_with_chembl.csv`

Outputs
-------
- `data/processed/v2_modeling_dataset.csv`

Important
---------
This script prepares data only. It does not train duration or success models,
does not call ChEMBL, and does not modify Version 1 files.
"""

import os
import sys
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.v2_sponsor_history import add_sponsor_history_features


INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "v2_trials_with_chembl.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "v2_modeling_dataset.csv")

DURATION_TARGET = "observed_duration_days"
SUCCESS_TARGET = "target_success"

LEAKAGE_COLUMNS = {
    "observed_duration_days",
    "planned_duration_days",
    "completion_date",
    "completion_date_parsed",
    "primary_completion_date",
    "primary_completion_date_parsed",
    "overall_status",
    "target_success",
    "success_label_source",
    "success_label_notes",
    "progression_nct_id",
    "progression_phase",
    "progression_start_date",
    "max_phase",
    "last_update_submit_date",
    "last_update_submit_date_parsed",
}

SPONSOR_HISTORY_FEATURES = [
    "sponsor_prior_trials",
    "sponsor_prior_completed_trials",
    "sponsor_prior_failed_trials",
    "sponsor_prior_completion_rate",
    "sponsor_prior_phase_trials",
    "sponsor_prior_phase_completion_rate",
    "sponsor_prior_avg_duration_days",
    "sponsor_has_prior_history",
    "sponsor_prior_successes",
    "sponsor_prior_failures",
    "sponsor_prior_success_rate",
]

V2_MODEL_FEATURE_ALLOWLIST = [
    # Disease indication
    "conditions_normalized",
    "condition_count",
    "search_query_source",
    # Molecule and modality. `max_phase` is intentionally excluded as leakage-prone.
    "molecule_name",
    "chembl_id",
    "preferred_name",
    "molecule_type",
    "drug_modality",
    "first_match_confidence",
    # Sponsor identity and prior-only history
    "lead_sponsor_normalized",
    "sponsor_class",
    *SPONSOR_HISTORY_FEATURES,
    # Phase and trial design
    "phase_normalized",
    "study_type",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
    "enrollment_count",
    "enrollment_type",
    # Trial scale and geography
    "drug_intervention_count",
    "collaborator_count",
    "location_count",
    "country_count",
    "countries",
    # Endpoint features
    "endpoint_safety",
    "endpoint_efficacy",
    "endpoint_survival",
    "endpoint_biomarker",
    "endpoint_response_rate",
    "endpoint_type_count",
    "has_primary_endpoint",
    "primary_endpoint_text_length",
    # Optional baseline text fields
    "brief_summary",
    "eligibility_criteria",
    "primary_outcome_measures",
    "primary_outcome_timeframes",
    "secondary_outcome_measures",
]

NARAYANAN_KEY_COLUMNS = [
    "conditions_normalized",
    "molecule_name",
    "chembl_id",
    "preferred_name",
    "molecule_type",
    "drug_modality",
    "lead_sponsor_normalized",
    "phase_normalized",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
    "enrollment_count",
    "endpoint_safety",
    "endpoint_efficacy",
    "endpoint_survival",
    "endpoint_biomarker",
    "endpoint_response_rate",
]


def validate_feature_allowlist() -> None:
    """Ensure the model feature allowlist does not contain known leakage fields."""
    overlap = sorted(set(V2_MODEL_FEATURE_ALLOWLIST).intersection(LEAKAGE_COLUMNS))
    if overlap:
        raise ValueError(f"Leakage columns in V2 model feature allowlist: {overlap}")


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize key missing values for diagnostics and downstream modeling."""
    df = df.copy()
    for col in V2_MODEL_FEATURE_ALLOWLIST:
        if col not in df.columns:
            df[col] = pd.NA

    text_like_cols = [
        "conditions_normalized",
        "molecule_name",
        "chembl_id",
        "preferred_name",
        "molecule_type",
        "drug_modality",
        "lead_sponsor_normalized",
        "sponsor_class",
        "phase_normalized",
        "study_type",
        "allocation",
        "intervention_model",
        "masking",
        "primary_purpose",
        "enrollment_type",
        "countries",
        "brief_summary",
        "eligibility_criteria",
        "primary_outcome_measures",
        "primary_outcome_timeframes",
        "secondary_outcome_measures",
    ]
    for col in text_like_cols:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    numeric_cols = [
        "enrollment_count",
        "condition_count",
        "drug_intervention_count",
        "collaborator_count",
        "location_count",
        "country_count",
        "endpoint_safety",
        "endpoint_efficacy",
        "endpoint_survival",
        "endpoint_biomarker",
        "endpoint_response_rate",
        "endpoint_type_count",
        "has_primary_endpoint",
        "primary_endpoint_text_length",
        *SPONSOR_HISTORY_FEATURES,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def build_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    """Create final modeling dataset diagnostics."""
    duration = pd.to_numeric(df[DURATION_TARGET], errors="coerce")
    success = pd.to_numeric(df.get(SUCCESS_TARGET), errors="coerce")
    matched = df["chembl_id"].fillna("UNKNOWN").ne("UNKNOWN")

    missing_key_counts = {}
    for col in NARAYANAN_KEY_COLUMNS:
        if col not in df.columns:
            missing_key_counts[col] = "MISSING_COLUMN"
        else:
            missing_key_counts[col] = int(
                df[col].isna().sum()
                + df[col].astype(str).str.strip().isin(["", "UNKNOWN", "unknown"]).sum()
            )

    sponsor_coverage = {}
    for col in SPONSOR_HISTORY_FEATURES:
        if col in df.columns:
            sponsor_coverage[col] = int(df[col].notna().sum())
        else:
            sponsor_coverage[col] = 0

    return {
        "total_rows": len(df),
        "valid_duration_rows": int((duration.notna() & (duration > 0)).sum()),
        "success_count": int((success == 1).sum()),
        "failure_count": int((success == 0).sum()),
        "unknown_success_count": int(success.isna().sum()),
        "chembl_matched_count": int(matched.sum()),
        "drug_modality_counts": df["drug_modality"].fillna("unknown").value_counts(),
        "sponsor_history_coverage": sponsor_coverage,
        "missing_key_counts": missing_key_counts,
    }


def print_diagnostics(stats: Dict[str, object]) -> None:
    """Print final V2 modeling dataset diagnostics."""
    print("\nV2 modeling dataset diagnostics")
    print("-" * 36)
    print(f"Total rows: {stats['total_rows']:,}")
    print(f"Rows with valid duration target: {stats['valid_duration_rows']:,}")
    print(f"Success count: {stats['success_count']:,}")
    print(f"Failure count: {stats['failure_count']:,}")
    print(f"Unknown success count: {stats['unknown_success_count']:,}")
    print(f"ChEMBL matched count: {stats['chembl_matched_count']:,}")

    print("\nDrug modality counts:")
    print(stats["drug_modality_counts"].to_string())

    print("\nSponsor history feature coverage:")
    for col, count in stats["sponsor_history_coverage"].items():
        print(f"  {col}: {count:,}")

    print("\nMissing/unknown values in key Narayanan columns:")
    for col, count in stats["missing_key_counts"].items():
        print(f"  {col}: {count}")


def build_modeling_dataset(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """Build and save the final V2 modeling dataset."""
    validate_feature_allowlist()

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Run src/v2_enrich_chembl.py first."
        )

    df = pd.read_csv(input_path)

    # Sponsor history is computed with prior-date logic inside
    # add_sponsor_history_features; future and same-day trials are not included.
    df = add_sponsor_history_features(df, validate=True)
    df = normalize_missing_values(df)

    df["v2_model_feature_allowlist"] = "|".join(V2_MODEL_FEATURE_ALLOWLIST)
    df["v2_leakage_excluded_columns"] = "|".join(sorted(LEAKAGE_COLUMNS))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved V2 modeling dataset to: {output_path}")
    print_diagnostics(build_diagnostics(df))
    return df


def main() -> pd.DataFrame:
    """CLI entry point."""
    return build_modeling_dataset()


if __name__ == "__main__":
    main()

