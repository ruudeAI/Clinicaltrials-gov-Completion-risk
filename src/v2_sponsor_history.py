"""
v2_sponsor_history.py - Time-aware sponsor history features for Version 2.

Purpose
-------
Create sponsor history features for success and duration models without using
future trials. These features are intentionally standalone and are not connected
to Version 1 preprocessing or any training scripts yet.

Inputs
------
A pandas DataFrame with sponsor/date columns such as:
- `lead_sponsor` or `sponsor_name`
- `start_date` or `start_year`
- `phase` or `phases`
- `overall_status` if available
- `observed_duration_days` if available
- `target_success` if available later

Outputs
-------
The input DataFrame plus:
- `sponsor_prior_trials`
- `sponsor_prior_completed_trials`
- `sponsor_prior_failed_trials`
- `sponsor_prior_completion_rate`
- `sponsor_prior_phase_trials`
- `sponsor_prior_phase_completion_rate`
- `sponsor_prior_avg_duration_days`
- `sponsor_has_prior_history`

If `target_success` exists, the output also includes:
- `sponsor_prior_successes`
- `sponsor_prior_failures`
- `sponsor_prior_success_rate`

Data Leakage Warnings
---------------------
Sponsor history must be computed only from trials that started before the
current trial's prediction date. Trials with the same start date are also
excluded from each other's history because their ordering is not knowable at
prediction time. Never compute sponsor aggregates over the full dataset before
splitting; that would leak future outcomes into earlier trials.
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd


COMPLETED_STATUSES = {"COMPLETED"}
FAILED_STATUSES = {"TERMINATED", "WITHDRAWN", "SUSPENDED"}
DEFAULT_RATE_PRIOR = 0.5
DEFAULT_DURATION_PRIOR_DAYS = 0.0


@dataclass
class SponsorStats:
    """Cumulative prior-only sponsor statistics."""

    trials: int = 0
    completed: int = 0
    failed: int = 0
    duration_sum: float = 0.0
    duration_count: int = 0
    successes: int = 0
    success_failures: int = 0


def normalize_sponsor_name(value: object) -> str:
    """
    Normalize sponsor names for grouping.

    This keeps the logic conservative. It standardizes case, punctuation, and
    whitespace, but does not attempt corporate entity resolution.
    """
    if value is None or pd.isna(value):
        return "unknown_sponsor"

    name = str(value).strip().lower()
    if not name:
        return "unknown_sponsor"

    name = name.replace("&", " and ")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    name = re.sub(r"\b(inc|incorporated|corp|corporation|ltd|llc|plc|gmbh|sa)\b", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "unknown_sponsor"


def normalize_phase(value: object) -> str:
    """Normalize phase values for phase-specific sponsor history."""
    if value is None or pd.isna(value):
        return "UNKNOWN"

    phase = str(value).strip().upper()
    if not phase:
        return "UNKNOWN"

    phase = phase.replace(" ", "").replace("_", "")
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
    return phase


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first available column from a candidate list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _prediction_dates(df: pd.DataFrame) -> pd.Series:
    """Create prediction dates, preferring `start_date` over `start_year`."""
    if "start_date" in df.columns:
        dates = pd.to_datetime(df["start_date"], errors="coerce")
    elif "start_year" in df.columns:
        years = pd.to_numeric(df["start_year"], errors="coerce")
        dates = pd.to_datetime(years.astype("Int64").astype(str) + "-01-01", errors="coerce")
    else:
        raise ValueError("Expected `start_date` or `start_year` for sponsor history.")

    if dates.isna().all():
        raise ValueError("No valid prediction dates could be parsed.")

    return dates.fillna(pd.Timestamp.max)


def _status_flags(status: object) -> tuple[int, int]:
    """Return completed and failed flags from an overall_status value."""
    if status is None or pd.isna(status):
        return 0, 0
    normalized = str(status).strip().upper()
    return int(normalized in COMPLETED_STATUSES), int(normalized in FAILED_STATUSES)


def _success_flags(value: object) -> tuple[int, int]:
    """Return success and failure flags from an optional target_success value."""
    if value is None or pd.isna(value):
        return 0, 0
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return 0, 0
    return int(numeric == 1), int(numeric == 0)


def _rate(numerator: int, denominator: int, prior: float) -> float:
    """Return a rate with a neutral prior when no history exists."""
    if denominator <= 0:
        return float(prior)
    return float(numerator / denominator)


def _avg_duration(stats: SponsorStats, prior_days: float) -> float:
    """Return prior average duration with a neutral value when unavailable."""
    if stats.duration_count <= 0:
        return float(prior_days)
    return float(stats.duration_sum / stats.duration_count)


def _feature_row(
    sponsor_stats: SponsorStats,
    phase_stats: SponsorStats,
    include_success_features: bool,
    rate_prior: float,
    duration_prior_days: float,
) -> Dict[str, float]:
    """Build one row of sponsor history features from prior-only stats."""
    row: Dict[str, float] = {
        "sponsor_prior_trials": sponsor_stats.trials,
        "sponsor_prior_completed_trials": sponsor_stats.completed,
        "sponsor_prior_failed_trials": sponsor_stats.failed,
        "sponsor_prior_completion_rate": _rate(
            sponsor_stats.completed,
            sponsor_stats.completed + sponsor_stats.failed,
            rate_prior,
        ),
        "sponsor_prior_phase_trials": phase_stats.trials,
        "sponsor_prior_phase_completion_rate": _rate(
            phase_stats.completed,
            phase_stats.completed + phase_stats.failed,
            rate_prior,
        ),
        "sponsor_prior_avg_duration_days": _avg_duration(
            sponsor_stats,
            duration_prior_days,
        ),
        "sponsor_has_prior_history": int(sponsor_stats.trials > 0),
    }

    if include_success_features:
        row.update(
            {
                "sponsor_prior_successes": sponsor_stats.successes,
                "sponsor_prior_failures": sponsor_stats.success_failures,
                "sponsor_prior_success_rate": _rate(
                    sponsor_stats.successes,
                    sponsor_stats.successes + sponsor_stats.success_failures,
                    rate_prior,
                ),
            }
        )

    return row


def _update_stats(
    stats: SponsorStats,
    completed: int,
    failed: int,
    duration_days: object,
    success: int = 0,
    success_failure: int = 0,
) -> None:
    """Update cumulative sponsor stats after a date group has been featurized."""
    stats.trials += 1
    stats.completed += completed
    stats.failed += failed
    stats.successes += success
    stats.success_failures += success_failure

    duration = pd.to_numeric(pd.Series([duration_days]), errors="coerce").iloc[0]
    if pd.notna(duration):
        stats.duration_sum += float(duration)
        stats.duration_count += 1


def add_sponsor_history_features(
    df: pd.DataFrame,
    rate_prior: float = DEFAULT_RATE_PRIOR,
    duration_prior_days: float = DEFAULT_DURATION_PRIOR_DAYS,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Add leakage-safe sponsor history features to a DataFrame.

    The function processes trials by prediction date. It creates features for
    every trial on a given date before updating sponsor histories with that
    date's outcomes, which prevents same-day rows from using each other.
    """
    sponsor_col = _pick_column(df, ["lead_sponsor", "sponsor_name"])
    if sponsor_col is None:
        raise ValueError("Expected `lead_sponsor` or `sponsor_name` column.")

    phase_col = _pick_column(df, ["phase", "phases"])
    status_col = "overall_status" if "overall_status" in df.columns else None
    duration_col = "observed_duration_days" if "observed_duration_days" in df.columns else None
    has_success = "target_success" in df.columns

    working = df.copy()
    working["_original_index"] = working.index
    working["_prediction_date"] = _prediction_dates(working)
    working["_normalized_sponsor"] = working[sponsor_col].apply(normalize_sponsor_name)
    working["_normalized_phase"] = (
        working[phase_col].apply(normalize_phase) if phase_col else "UNKNOWN"
    )
    working = working.sort_values(
        ["_prediction_date", "_original_index"],
        kind="mergesort",
    )

    sponsor_history: defaultdict[str, SponsorStats] = defaultdict(SponsorStats)
    sponsor_phase_history: defaultdict[tuple[str, str], SponsorStats] = defaultdict(SponsorStats)
    feature_rows = []

    # Future trials cannot be used because they would not be known when making
    # a prediction for the current trial. We therefore compute features for all
    # rows at a date first, then add those rows to cumulative history.
    for prediction_date, date_group in working.groupby("_prediction_date", sort=True):
        del prediction_date

        for _, row in date_group.iterrows():
            sponsor = row["_normalized_sponsor"]
            phase = row["_normalized_phase"]
            sponsor_stats = sponsor_history[sponsor]
            phase_stats = sponsor_phase_history[(sponsor, phase)]
            features = _feature_row(
                sponsor_stats=sponsor_stats,
                phase_stats=phase_stats,
                include_success_features=has_success,
                rate_prior=rate_prior,
                duration_prior_days=duration_prior_days,
            )
            features["_original_index"] = row["_original_index"]
            feature_rows.append(features)

        for _, row in date_group.iterrows():
            sponsor = row["_normalized_sponsor"]
            phase = row["_normalized_phase"]
            status = row[status_col] if status_col else None
            duration = row[duration_col] if duration_col else None
            success_value = row["target_success"] if has_success else None
            completed, failed = _status_flags(status)
            success, success_failure = _success_flags(success_value)

            _update_stats(
                sponsor_history[sponsor],
                completed=completed,
                failed=failed,
                duration_days=duration,
                success=success,
                success_failure=success_failure,
            )
            _update_stats(
                sponsor_phase_history[(sponsor, phase)],
                completed=completed,
                failed=failed,
                duration_days=duration,
                success=success,
                success_failure=success_failure,
            )

    features_df = pd.DataFrame(feature_rows).set_index("_original_index")
    output = df.copy()
    for col in features_df.columns:
        output[col] = features_df[col].reindex(output.index)

    if validate:
        validate_no_future_sponsor_history(df, output, rate_prior=rate_prior)

    return output


def validate_no_future_sponsor_history(
    original_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    rate_prior: float = DEFAULT_RATE_PRIOR,
) -> None:
    """
    Validate key sponsor history counts with a simple prior-date recomputation.

    This check intentionally uses a direct, slower calculation. It confirms
    that `sponsor_prior_trials` and `sponsor_prior_phase_trials` include only
    rows with earlier prediction dates, never same-date or future rows.
    """
    sponsor_col = _pick_column(original_df, ["lead_sponsor", "sponsor_name"])
    if sponsor_col is None:
        raise ValueError("Expected `lead_sponsor` or `sponsor_name` column.")

    phase_col = _pick_column(original_df, ["phase", "phases"])
    dates = _prediction_dates(original_df)
    sponsors = original_df[sponsor_col].apply(normalize_sponsor_name)
    phases = original_df[phase_col].apply(normalize_phase) if phase_col else pd.Series("UNKNOWN", index=original_df.index)

    for idx in original_df.index:
        earlier = dates < dates.loc[idx]
        same_sponsor = sponsors == sponsors.loc[idx]
        same_phase = phases == phases.loc[idx]

        expected_prior_trials = int((earlier & same_sponsor).sum())
        expected_phase_trials = int((earlier & same_sponsor & same_phase).sum())

        actual_prior_trials = int(enriched_df.loc[idx, "sponsor_prior_trials"])
        actual_phase_trials = int(enriched_df.loc[idx, "sponsor_prior_phase_trials"])

        if actual_prior_trials != expected_prior_trials:
            raise AssertionError(
                f"Sponsor history leakage check failed at index {idx}: "
                f"expected {expected_prior_trials} prior sponsor trials, got "
                f"{actual_prior_trials}."
            )
        if actual_phase_trials != expected_phase_trials:
            raise AssertionError(
                f"Sponsor phase history leakage check failed at index {idx}: "
                f"expected {expected_phase_trials} prior phase trials, got "
                f"{actual_phase_trials}."
            )

    rate_cols = [
        "sponsor_prior_completion_rate",
        "sponsor_prior_phase_completion_rate",
    ]
    if "sponsor_prior_success_rate" in enriched_df.columns:
        rate_cols.append("sponsor_prior_success_rate")

    for col in rate_cols:
        invalid = ~enriched_df[col].between(0.0, 1.0)
        if invalid.any():
            bad_indices = list(enriched_df.index[invalid])
            raise AssertionError(f"Rate column `{col}` outside [0, 1] at {bad_indices}.")

    no_history = enriched_df["sponsor_prior_trials"] == 0
    if no_history.any():
        non_prior = enriched_df.loc[no_history, "sponsor_prior_completion_rate"]
        if not (non_prior == float(rate_prior)).all():
            raise AssertionError("No-history rows must use the configured rate prior.")


def demo_sponsor_history() -> pd.DataFrame:
    """Run a tiny in-memory demo of leakage-safe sponsor history features."""
    sample = pd.DataFrame(
        [
            {
                "nct_id": "NCT001",
                "sponsor_name": "Example Pharma Inc.",
                "start_date": "2020-01-01",
                "phase": "PHASE2",
                "overall_status": "COMPLETED",
                "observed_duration_days": 300,
                "target_success": 1,
            },
            {
                "nct_id": "NCT002",
                "sponsor_name": "Example Pharma",
                "start_date": "2021-01-01",
                "phase": "PHASE2",
                "overall_status": "TERMINATED",
                "observed_duration_days": 120,
                "target_success": 0,
            },
            {
                "nct_id": "NCT003",
                "sponsor_name": "Example Pharma LLC",
                "start_date": "2021-01-01",
                "phase": "PHASE3",
                "overall_status": "COMPLETED",
                "observed_duration_days": 500,
                "target_success": 1,
            },
            {
                "nct_id": "NCT004",
                "sponsor_name": "New Sponsor",
                "start_date": "2022-01-01",
                "phase": "PHASE1",
                "overall_status": "COMPLETED",
                "observed_duration_days": 80,
                "target_success": 1,
            },
        ]
    )
    enriched = add_sponsor_history_features(sample)
    display_cols = [
        "nct_id",
        "sponsor_name",
        "start_date",
        "phase",
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
    print(enriched[display_cols].to_string(index=False))
    return enriched


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    demo_sponsor_history()
