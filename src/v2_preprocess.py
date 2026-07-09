"""
v2_preprocess.py - Initial Version 2 preprocessing for duration modeling.

Purpose
-------
Create an initial processed V2 dataset focused on date parsing, duration target
creation, endpoint type extraction, and basic non-leaky feature preparation.

Inputs
------
- `data/raw/v2_trials_raw.csv`

Outputs
-------
- `data/processed/v2_success_duration_dataset.csv`

Leakage Note
------------
`completion_date` and `observed_duration_days` are allowed here because they
are needed to create the duration regression target. They must not be included
in future model input features. Similarly, `overall_status` is preserved for
auditing but excluded from the feature allowlist.
"""

import os
import re
import sys
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


V2_RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "v2_trials_raw.csv")
V2_PROCESSED_CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "v2_success_duration_dataset.csv",
)

DATE_COLUMNS = [
    "start_date",
    "primary_completion_date",
    "completion_date",
    "study_first_submit_date",
    "last_update_submit_date",
]

FORBIDDEN_MODEL_INPUT_COLUMNS = [
    "overall_status",
    "completion_date",
    "completion_date_parsed",
    "observed_duration_days",
]

V2_FEATURE_ALLOWLIST = [
    "prediction_date",
    "phase_normalized",
    "study_type",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
    "enrollment_count",
    "enrollment_type",
    "conditions_normalized",
    "condition_count",
    "search_query_source",
    "intervention_names_normalized",
    "drug_intervention_names",
    "drug_intervention_count",
    "lead_sponsor_normalized",
    "sponsor_class",
    "collaborator_count",
    "location_count",
    "country_count",
    "countries",
    "planned_duration_days",
    "endpoint_safety",
    "endpoint_efficacy",
    "endpoint_survival",
    "endpoint_biomarker",
    "endpoint_response_rate",
    "endpoint_type_count",
    "has_primary_endpoint",
    "primary_endpoint_text_length",
    "brief_summary",
    "eligibility_criteria",
    "primary_outcome_measures",
    "primary_outcome_timeframes",
    "secondary_outcome_measures",
]

ENDPOINT_KEYWORDS: Dict[str, List[str]] = {
    "endpoint_safety": [
        "adverse event",
        "serious adverse event",
        "safety",
        "toxicity",
        "dose limiting toxicity",
        "tolerability",
        "maximum tolerated dose",
        "treatment emergent adverse event",
    ],
    "endpoint_efficacy": [
        "efficacy",
        "clinical benefit",
        "symptom improvement",
        "disease activity",
        "change from baseline",
        "treatment effect",
    ],
    "endpoint_survival": [
        "overall survival",
        "progression free survival",
        "progression-free survival",
        "event free survival",
        "event-free survival",
        "disease free survival",
        "disease-free survival",
        "relapse free survival",
        "relapse-free survival",
        "survival",
        "mortality",
    ],
    "endpoint_biomarker": [
        "biomarker",
        "pharmacodynamic",
        "pharmacokinetic",
        "immune response",
        "viral load",
        "tumor marker",
        "lab value",
        "cytokine",
    ],
    "endpoint_response_rate": [
        "objective response rate",
        "overall response rate",
        "response rate",
        "complete response",
        "partial response",
        "remission",
        "recist",
    ],
}


def parse_date_series(series: pd.Series) -> pd.Series:
    """Parse raw ClinicalTrials.gov date strings, allowing partial dates."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def normalize_text_value(value: object) -> str:
    """Normalize free-text/categorical values for simple grouping features."""
    if value is None or pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().lower()
    if not text:
        return "UNKNOWN"
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_pipe_list(value: object) -> str:
    """Normalize pipe-delimited list strings while preserving delimiters."""
    if value is None or pd.isna(value):
        return "UNKNOWN"
    parts = [
        normalize_text_value(part)
        for part in str(value).split("|")
        if normalize_text_value(part) != "UNKNOWN"
    ]
    return "|".join(parts) if parts else "UNKNOWN"


def normalize_phase(value: object) -> str:
    """Normalize phase values into stable V2 phase groups."""
    if value is None or pd.isna(value):
        return "UNKNOWN"
    phase = str(value).strip().upper().replace(" ", "").replace("_", "")
    if not phase:
        return "UNKNOWN"
    if "PHASE1" in phase and "PHASE2" in phase:
        return "PHASE1_PHASE2"
    if "PHASE2" in phase and "PHASE3" in phase:
        return "PHASE2_PHASE3"
    if "EARLYPHASE1" in phase:
        return "EARLY_PHASE1"
    if "PHASE1" in phase:
        return "PHASE1"
    if "PHASE2" in phase:
        return "PHASE2"
    if "PHASE3" in phase:
        return "PHASE3"
    if "PHASE4" in phase:
        return "PHASE4"
    if phase in {"NA", "NOTAPPLICABLE"}:
        return "NA"
    return phase


def count_pipe_values(value: object) -> int:
    """Count non-empty values in a pipe-delimited string."""
    if value is None or pd.isna(value):
        return 0
    return len([part for part in str(value).split("|") if part.strip()])


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce count/numeric columns in place and fill missing count values."""
    for col in columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def endpoint_text(row: pd.Series) -> str:
    """Combine protocol endpoint fields for endpoint-type extraction."""
    fields = [
        "primary_outcome_measures",
        "primary_outcome_timeframes",
        "secondary_outcome_measures",
    ]
    values = []
    for field in fields:
        value = row.get(field, "")
        if pd.notna(value):
            values.append(str(value))
    return " ".join(values).lower().replace("|", " ")


def add_endpoint_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple keyword-based endpoint flags.

    These features use protocol outcome text only. They do not inspect posted
    result values, endpoint success/failure, p-values, or other outcome evidence.
    """
    texts = df.apply(endpoint_text, axis=1)

    for feature_name, keywords in ENDPOINT_KEYWORDS.items():
        pattern = "|".join(re.escape(keyword) for keyword in keywords)
        df[feature_name] = texts.str.contains(pattern, case=False, regex=True).astype(int)

    endpoint_flag_cols = list(ENDPOINT_KEYWORDS.keys())
    df["endpoint_type_count"] = df[endpoint_flag_cols].sum(axis=1)
    primary_text = df.get("primary_outcome_measures", pd.Series("", index=df.index))
    primary_text = primary_text.fillna("").astype(str)
    df["has_primary_endpoint"] = (primary_text.str.strip() != "").astype(int)
    df["primary_endpoint_text_length"] = primary_text.str.len()
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and create duration target candidates."""
    for col in DATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
        df[f"{col}_parsed"] = parse_date_series(df[col])

    df["prediction_date"] = df["start_date_parsed"]
    df["observed_duration_days"] = (
        df["completion_date_parsed"] - df["start_date_parsed"]
    ).dt.days
    df["planned_duration_days"] = (
        df["primary_completion_date_parsed"] - df["start_date_parsed"]
    ).dt.days
    return df


def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic normalized fields and count features."""
    df["phase_normalized"] = df.get("phases", pd.Series(pd.NA, index=df.index)).apply(
        normalize_phase
    )
    df["intervention_names_normalized"] = df.get(
        "intervention_names",
        pd.Series(pd.NA, index=df.index),
    ).apply(normalize_pipe_list)
    df["lead_sponsor_normalized"] = df.get(
        "lead_sponsor",
        pd.Series(pd.NA, index=df.index),
    ).apply(normalize_text_value)
    df["conditions_normalized"] = df.get(
        "conditions",
        pd.Series(pd.NA, index=df.index),
    ).apply(normalize_pipe_list)

    numeric_count_cols = [
        "condition_count",
        "drug_intervention_count",
        "collaborator_count",
        "location_count",
        "country_count",
        "enrollment_count",
    ]
    df = coerce_numeric_columns(df, numeric_count_cols)

    if "conditions" in df.columns:
        missing_condition_count = df["condition_count"] == 0
        if missing_condition_count.any():
            df.loc[missing_condition_count, "condition_count"] = df.loc[
                missing_condition_count,
                "conditions",
            ].apply(count_pipe_values).astype(int)

    if "drug_intervention_names" in df.columns:
        missing_drug_count = df["drug_intervention_count"] == 0
        if missing_drug_count.any():
            df.loc[missing_drug_count, "drug_intervention_count"] = df.loc[
                missing_drug_count,
                "drug_intervention_names",
            ].apply(count_pipe_values).astype(int)

    return df


def validate_feature_allowlist() -> None:
    """Fail fast if an obvious leakage column is added to the feature allowlist."""
    leakage_cols = set(FORBIDDEN_MODEL_INPUT_COLUMNS).intersection(V2_FEATURE_ALLOWLIST)
    if leakage_cols:
        raise ValueError(f"Leakage columns found in V2 feature allowlist: {sorted(leakage_cols)}")


def diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    """Build preprocessing diagnostics for dates and durations."""
    observed = pd.to_numeric(df["observed_duration_days"], errors="coerce")
    planned = pd.to_numeric(df["planned_duration_days"], errors="coerce")
    valid_observed = observed.notna() & (observed > 0)
    valid_planned = planned.notna() & (planned > 0)

    return {
        "total_rows": len(df),
        "rows_with_valid_observed_duration_days": int(valid_observed.sum()),
        "rows_with_valid_planned_duration_days": int(valid_planned.sum()),
        "observed_duration_min": observed[valid_observed].min(),
        "observed_duration_median": observed[valid_observed].median(),
        "observed_duration_max": observed[valid_observed].max(),
        "missing_date_counts": {
            col: int(df[f"{col}_parsed"].isna().sum()) for col in DATE_COLUMNS
        },
        "invalid_observed_duration_count": int((observed.notna() & (observed <= 0)).sum()),
        "invalid_planned_duration_count": int((planned.notna() & (planned <= 0)).sum()),
    }


def print_diagnostics(stats: Dict[str, object]) -> None:
    """Print concise preprocessing diagnostics."""
    print("\nV2 preprocessing diagnostics")
    print("-" * 32)
    print(f"Total rows: {stats['total_rows']:,}")
    print(
        "Rows with valid observed_duration_days: "
        f"{stats['rows_with_valid_observed_duration_days']:,}"
    )
    print(
        "Rows with valid planned_duration_days: "
        f"{stats['rows_with_valid_planned_duration_days']:,}"
    )
    print(
        "Observed duration min/median/max: "
        f"{stats['observed_duration_min']} / "
        f"{stats['observed_duration_median']} / "
        f"{stats['observed_duration_max']}"
    )
    print("Missing date counts:")
    for col, count in stats["missing_date_counts"].items():
        print(f"  {col}: {count:,}")
    print(
        "Invalid observed duration count: "
        f"{stats['invalid_observed_duration_count']:,}"
    )
    print(
        "Invalid planned duration count: "
        f"{stats['invalid_planned_duration_count']:,}"
    )


def preprocess_v2(
    input_path: str = V2_RAW_CSV_PATH,
    output_path: str = V2_PROCESSED_CSV_PATH,
) -> pd.DataFrame:
    """Run the initial V2 preprocessing pipeline."""
    validate_feature_allowlist()

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"V2 raw file not found: {input_path}. Run src/v2_clinicaltrials_api.py first."
        )

    df = pd.read_csv(input_path)
    df = add_date_features(df)
    df = add_normalized_features(df)
    df = add_endpoint_features(df)

    stats = diagnostics(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved V2 processed dataset to: {output_path}")
    print_diagnostics(stats)
    print("\nV2 feature allowlist for future models:")
    for feature in V2_FEATURE_ALLOWLIST:
        print(f"  - {feature}")

    return df


def main() -> pd.DataFrame:
    """CLI entry point for V2 preprocessing."""
    return preprocess_v2()


if __name__ == "__main__":
    main()
