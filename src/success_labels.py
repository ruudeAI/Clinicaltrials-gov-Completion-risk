"""
success_labels.py - Initial Version 2 phase-progression success labels.

Purpose
-------
Create a conservative, auditable `target_success` proxy label for Version 2
clinical trial success prediction.

This is not the same as Version 1 completion-risk labeling. A completed trial
is not necessarily clinically successful. This first V2 label uses public
ClinicalTrials.gov phase progression as a proxy:

- Phase 1 success: same normalized drug appears later in Phase 2 or higher.
- Phase 2 success: same normalized drug appears later in Phase 3 or higher.
- Phase 3 success: unknown unless later Phase 4 evidence exists.

Inputs
------
- `data/processed/v2_success_duration_dataset.csv`

Outputs
-------
- `data/interim/v2_trials_with_success_labels.csv`

Leakage Warning
---------------
Later trial progression evidence is used only to create outcome labels and
label provenance columns. It must never be used as an input feature during model
training. Phase 3 approval-based labeling is deferred.

Limitations
-----------
- This is a proxy label, not a direct clinical endpoint or regulatory outcome.
- No-progression does not always mean failure.
- Drug string matching is imperfect until ChEMBL IDs are added.
- Completion is not success.
- Ongoing/recruiting trials generally remain unknown unless progression is
  already visible in later public trial records.
"""

import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


V2_PROCESSED_CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "v2_success_duration_dataset.csv",
)
V2_SUCCESS_LABELS_CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "interim",
    "v2_trials_with_success_labels.csv",
)

FAILED_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
ONGOING_OR_UNCERTAIN_STATUSES = {
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "UNKNOWN",
}


def normalize_phase(value: object) -> str:
    """Normalize phase strings into comparable phase groups."""
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
    if phase in {"NA", "N/A", "NOTAPPLICABLE"}:
        return "NA"
    return phase


def phase_rank(phase: object) -> Optional[int]:
    """
    Convert normalized phase to a rough numeric rank.

    Combined phases use the later phase rank because progression from Phase 1/2
    should require evidence beyond Phase 2.
    """
    normalized = normalize_phase(phase)
    ranks = {
        "EARLY_PHASE1": 1,
        "PHASE1": 1,
        "PHASE1_PHASE2": 2,
        "PHASE2": 2,
        "PHASE2_PHASE3": 3,
        "PHASE3": 3,
        "PHASE4": 4,
    }
    return ranks.get(normalized)


def normalize_status(value: object) -> str:
    """Normalize ClinicalTrials.gov overall_status values."""
    if value is None or pd.isna(value):
        return "UNKNOWN"
    status = str(value).strip().upper().replace(" ", "_")
    return status or "UNKNOWN"


def _normalize_drug_token(value: object) -> str:
    """Normalize one drug/intervention token for string matching."""
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip().lower()
    token = re.sub(r"\([^)]*\)", " ", token)
    token = re.sub(r"\b(placebo|matching placebo|vehicle)\b", " ", token)
    token = re.sub(r"[^a-z0-9\- ]+", " ", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token


def split_intervention_names(value: object) -> List[str]:
    """
    Split normalized intervention strings into candidate drug tokens.

    ClinicalTrials.gov names are inconsistent. This conservative splitter
    handles pipe-delimited lists and common combination separators.
    """
    if value is None or pd.isna(value):
        return []

    raw = str(value).strip()
    if not raw:
        return []

    pieces: List[str] = []
    for pipe_piece in raw.split("|"):
        pieces.extend(re.split(r"\s+\+\s+|,|;|\s+ and \s+|/", pipe_piece, flags=re.I))

    tokens = []
    for piece in pieces:
        token = _normalize_drug_token(piece)
        if token and len(token) > 2:
            tokens.append(token)

    return sorted(set(tokens))


def split_conditions(value: object) -> Set[str]:
    """Split normalized condition strings into simple condition tokens."""
    if value is None or pd.isna(value):
        return set()
    conditions = set()
    for piece in str(value).split("|"):
        normalized = re.sub(r"[^a-z0-9 ]+", " ", piece.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized:
            conditions.add(normalized)
    return conditions


def _required_progression_rank(current_rank: Optional[int]) -> Optional[int]:
    """Return the minimum later phase rank needed for a success label."""
    if current_rank == 1:
        return 2
    if current_rank == 2:
        return 3
    if current_rank == 3:
        return 4
    return None


def _condition_matches(
    current_conditions: Set[str],
    candidate_conditions: Set[str],
    strict_condition_match: bool,
) -> bool:
    """Return whether condition matching passes for the selected mode."""
    if not strict_condition_match:
        return True
    if not current_conditions or not candidate_conditions:
        return False
    return bool(current_conditions.intersection(candidate_conditions))


def find_progression(
    current_row: pd.Series,
    later_rows: pd.DataFrame,
    strict_condition_match: bool = False,
) -> Optional[Dict[str, object]]:
    """
    Search later trials for phase progression evidence.

    Matching defaults to drug-level progression. Condition matching can be made
    stricter later, but defaulting to drug-level progression keeps the first
    implementation less sparse and easier to audit.
    """
    current_rank = phase_rank(current_row.get("phase_for_label"))
    required_rank = _required_progression_rank(current_rank)
    if required_rank is None:
        return None

    current_drugs = set(current_row.get("drug_tokens", set()))
    if not current_drugs:
        return None

    current_conditions = current_row.get("condition_tokens", set())

    for _, candidate in later_rows.iterrows():
        candidate_rank = phase_rank(candidate.get("phase_for_label"))
        if candidate_rank is None or candidate_rank < required_rank:
            continue

        candidate_drugs = set(candidate.get("drug_tokens", set()))
        if not current_drugs.intersection(candidate_drugs):
            continue

        candidate_conditions = candidate.get("condition_tokens", set())
        if not _condition_matches(
            current_conditions=current_conditions,
            candidate_conditions=candidate_conditions,
            strict_condition_match=strict_condition_match,
        ):
            continue

        matched_drugs = sorted(current_drugs.intersection(candidate_drugs))
        return {
            "progression_nct_id": candidate.get("nct_id"),
            "progression_phase": candidate.get("phase_for_label"),
            "progression_start_date": candidate.get("label_date"),
            "matched_drugs": matched_drugs,
        }

    return None


def _unknown_label(notes: str) -> Dict[str, object]:
    """Build an unknown label result."""
    return {
        "target_success": pd.NA,
        "success_label_source": "unknown",
        "success_label_notes": notes,
        "progression_nct_id": pd.NA,
        "progression_phase": pd.NA,
        "progression_start_date": pd.NA,
    }


def _success_label(progression: Dict[str, object]) -> Dict[str, object]:
    """Build a success label result from later progression evidence."""
    matched = ", ".join(progression.get("matched_drugs", []))
    return {
        "target_success": 1,
        "success_label_source": "phase_progression",
        "success_label_notes": f"Later phase progression found for: {matched}",
        "progression_nct_id": progression.get("progression_nct_id"),
        "progression_phase": progression.get("progression_phase"),
        "progression_start_date": progression.get("progression_start_date"),
    }


def _failure_label() -> Dict[str, object]:
    """Build a conservative failure label for failed trials with no progression."""
    return {
        "target_success": 0,
        "success_label_source": "failed_status_no_later_progression",
        "success_label_notes": (
            "Trial status is terminated/withdrawn/suspended and no later phase "
            "progression evidence was found in this dataset."
        ),
        "progression_nct_id": pd.NA,
        "progression_phase": pd.NA,
        "progression_start_date": pd.NA,
    }


def create_success_labels(
    df: pd.DataFrame,
    strict_condition_match: bool = False,
) -> pd.DataFrame:
    """Create initial phase-progression-based `target_success` labels."""
    working = df.copy()

    phase_source = "phase_normalized" if "phase_normalized" in working.columns else "phases"
    drug_source = (
        "intervention_names_normalized"
        if "intervention_names_normalized" in working.columns
        else "intervention_names"
    )
    condition_source = (
        "conditions_normalized" if "conditions_normalized" in working.columns else "conditions"
    )
    date_source = "prediction_date" if "prediction_date" in working.columns else "start_date_parsed"
    if date_source not in working.columns:
        date_source = "start_date"

    working["phase_for_label"] = working[phase_source].apply(normalize_phase)
    working["status_for_label"] = working.get(
        "overall_status",
        pd.Series("UNKNOWN", index=working.index),
    ).apply(normalize_status)
    working["label_date"] = pd.to_datetime(working[date_source], errors="coerce", format="mixed")
    working["drug_tokens"] = working[drug_source].apply(split_intervention_names)
    working["condition_tokens"] = working[condition_source].apply(split_conditions)
    working["phase_rank_for_label"] = working["phase_for_label"].apply(phase_rank)

    working = working.sort_values(["label_date", "nct_id"], kind="mergesort").reset_index(drop=True)

    label_rows: List[Dict[str, object]] = []
    for idx, row in working.iterrows():
        if pd.isna(row["label_date"]):
            label_rows.append(_unknown_label("Missing or invalid prediction/start date."))
            continue

        if row["phase_rank_for_label"] not in {1, 2, 3}:
            label_rows.append(_unknown_label("Phase is not eligible for initial Phase 1/2/3 progression labeling."))
            continue

        if not row["drug_tokens"]:
            label_rows.append(_unknown_label("No usable intervention/drug name for progression matching."))
            continue

        later_rows = working[working["label_date"] > row["label_date"]]
        progression = find_progression(
            current_row=row,
            later_rows=later_rows,
            strict_condition_match=strict_condition_match,
        )

        if progression is not None:
            label_rows.append(_success_label(progression))
            continue

        status = row["status_for_label"]
        if status in FAILED_STATUSES:
            label_rows.append(_failure_label())
        elif status in ONGOING_OR_UNCERTAIN_STATUSES:
            label_rows.append(_unknown_label("Ongoing or uncertain trial status and no later progression evidence found."))
        elif row["phase_rank_for_label"] == 3:
            label_rows.append(_unknown_label("Phase 3 approval labeling is deferred; no later Phase 4 evidence found."))
        else:
            label_rows.append(_unknown_label("No later progression evidence found; not labeled as failure without failed status."))

    labels = pd.DataFrame(label_rows)
    output = working.drop(
        columns=[
            "phase_for_label",
            "status_for_label",
            "label_date",
            "drug_tokens",
            "condition_tokens",
            "phase_rank_for_label",
        ],
        errors="ignore",
    )
    return pd.concat([output, labels], axis=1)


def print_diagnostics(df: pd.DataFrame) -> None:
    """Print label diagnostics and examples."""
    success_count = int((df["target_success"] == 1).sum())
    failure_count = int((df["target_success"] == 0).sum())
    unknown_count = int(df["target_success"].isna().sum())

    print("\nV2 success label diagnostics")
    print("-" * 32)
    print(f"Total rows: {len(df):,}")
    print(f"Success count: {success_count:,}")
    print(f"Failure count: {failure_count:,}")
    print(f"Unknown count: {unknown_count:,}")

    print("\nCounts by phase and label:")
    phase_counts = (
        df.assign(target_success_display=df["target_success"].fillna("UNKNOWN"))
        .groupby(["phase_normalized", "target_success_display"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    print(phase_counts.to_string(index=False))

    example_cols = [
        "nct_id",
        "phase_normalized",
        "intervention_names_normalized",
        "overall_status",
        "target_success",
        "success_label_source",
        "progression_nct_id",
        "progression_phase",
    ]

    print("\nExamples of success labels:")
    successes = df[df["target_success"] == 1][example_cols].head(5)
    print(successes.to_string(index=False) if not successes.empty else "None")

    print("\nExamples of unknown labels:")
    unknowns = df[df["target_success"].isna()][example_cols + ["success_label_notes"]].head(5)
    print(unknowns.to_string(index=False) if not unknowns.empty else "None")


def label_v2_success(
    input_path: str = V2_PROCESSED_CSV_PATH,
    output_path: str = V2_SUCCESS_LABELS_CSV_PATH,
    strict_condition_match: bool = False,
) -> pd.DataFrame:
    """Load processed V2 data, create success labels, and save interim output."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Processed V2 dataset not found: {input_path}. Run src/v2_preprocess.py first."
        )

    df = pd.read_csv(input_path)
    labeled = create_success_labels(df, strict_condition_match=strict_condition_match)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labeled.to_csv(output_path, index=False)

    print(f"Saved V2 success labels to: {output_path}")
    print_diagnostics(labeled)
    return labeled


def main() -> pd.DataFrame:
    """CLI entry point for initial V2 success-label generation."""
    return label_v2_success()


if __name__ == "__main__":
    main()
