"""
v2_clinicaltrials_api.py - ClinicalTrials.gov API v2 extraction for Version 2.

Purpose
-------
Fetch and flatten ClinicalTrials.gov API v2 study records for the Version 2
clinical trial success and duration pipeline.

This module is intentionally separate from `src/clinicaltrials_api.py` so the
existing Version 1 completion-risk pipeline remains unchanged.

Output
------
- `data/raw/v2_trials_raw.csv`

Leakage Note
------------
This extractor preserves fields such as `overall_status`, `completion_date`,
and `primary_completion_date` because they are needed for label and duration
construction. Later modeling code must treat them carefully:

- `completion_date` may be used to create the duration target.
- `overall_status`, `completion_date`, and actual outcome evidence must not be
  used as model input features for prospective prediction.
- `primary_completion_date` may be leakage-prone if it was revised after trial
  start; use only under a documented prediction-time assumption.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.config import API_BASE_URL, API_RATE_LIMIT_SECONDS, SEARCH_QUERIES


V2_RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "v2_trials_raw.csv")
V2_MAX_TRIALS = 300
V2_PAGE_SIZE = 100


def safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely read a nested dictionary path.

    ClinicalTrials.gov API v2 study records are deeply nested. This helper
    avoids KeyError/TypeError when optional modules or fields are missing.
    """
    current: Any = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


def join_values(values: Optional[Iterable[Any]], separator: str = "|") -> Optional[str]:
    """Join non-empty values into a pipe-delimited string."""
    if not values:
        return None
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return separator.join(cleaned) if cleaned else None


def get_date_value(module: Dict[str, Any], field_name: str) -> Optional[str]:
    """
    Extract raw date string from a ClinicalTrials.gov date struct.

    Example API path:
    `protocolSection.statusModule.startDateStruct.date`
    """
    date_struct = module.get(field_name, {}) or {}
    if isinstance(date_struct, dict):
        return date_struct.get("date")
    return None


def fetch_studies(
    query_term: str,
    page_size: int = 100,
    max_records: int = 300,
) -> List[Dict[str, Any]]:
    """
    Fetch study records from ClinicalTrials.gov API v2 with pagination.

    Parameters
    ----------
    query_term:
        ClinicalTrials.gov search term.
    page_size:
        Number of records per request. ClinicalTrials.gov supports up to 1000.
    max_records:
        Maximum records to collect for this query.
    """
    all_studies: List[Dict[str, Any]] = []
    page_token = None

    while len(all_studies) < max_records:
        remaining = max_records - len(all_studies)
        params = {
            "query.term": query_term,
            "pageSize": min(page_size, remaining),
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            response = requests.get(API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"Warning: ClinicalTrials.gov request failed for '{query_term}': {exc}")
            break

        payload = response.json()
        studies = payload.get("studies", []) or []
        if not studies:
            break

        all_studies.extend(studies)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

        time.sleep(API_RATE_LIMIT_SECONDS)

    return all_studies[:max_records]


def flatten_study(study: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten one ClinicalTrials.gov API v2 study record for V2 modeling.

    Comments below identify the API module where each field is sourced.
    """
    protocol = study.get("protocolSection", {}) or {}

    # protocolSection.identificationModule
    identification_module = safe_get(protocol, "identificationModule", default={}) or {}
    nct_id = identification_module.get("nctId")
    brief_title = identification_module.get("briefTitle")

    # protocolSection.statusModule
    status_module = safe_get(protocol, "statusModule", default={}) or {}
    overall_status = status_module.get("overallStatus")
    start_date = get_date_value(status_module, "startDateStruct")
    primary_completion_date = get_date_value(status_module, "primaryCompletionDateStruct")
    completion_date = get_date_value(status_module, "completionDateStruct")
    study_first_submit_date = status_module.get("studyFirstSubmitDate")
    last_update_submit_date = status_module.get("lastUpdateSubmitDate")

    # protocolSection.designModule
    design_module = safe_get(protocol, "designModule", default={}) or {}
    phases = join_values(design_module.get("phases", []) or [])
    study_type = design_module.get("studyType")
    enrollment_info = design_module.get("enrollmentInfo", {}) or {}
    enrollment_count = enrollment_info.get("count")
    enrollment_type = enrollment_info.get("type")

    # protocolSection.designModule.designInfo
    design_info = design_module.get("designInfo", {}) or {}
    allocation = design_info.get("allocation")
    intervention_model = design_info.get("interventionModel")
    masking = safe_get(design_info, "maskingInfo", "masking")
    primary_purpose = design_info.get("primaryPurpose")

    # protocolSection.conditionsModule
    conditions_module = safe_get(protocol, "conditionsModule", default={}) or {}
    conditions_list = conditions_module.get("conditions", []) or []
    conditions = join_values(conditions_list)
    condition_count = len([c for c in conditions_list if str(c).strip()])

    # protocolSection.armsInterventionsModule.interventions
    arms_module = safe_get(protocol, "armsInterventionsModule", default={}) or {}
    interventions = arms_module.get("interventions", []) or []
    intervention_names: List[str] = []
    intervention_types_set = set()
    drug_intervention_names: List[str] = []

    for intervention in interventions:
        if not isinstance(intervention, dict):
            continue
        intervention_type = intervention.get("type")
        intervention_name = intervention.get("name")
        if intervention_type:
            intervention_types_set.add(str(intervention_type).strip())
        if intervention_name:
            intervention_names.append(str(intervention_name).strip())
        if intervention_type and str(intervention_type).upper() == "DRUG" and intervention_name:
            drug_intervention_names.append(str(intervention_name).strip())

    intervention_types = join_values(sorted(intervention_types_set))
    intervention_names_joined = join_values(intervention_names)
    drug_intervention_names_joined = join_values(drug_intervention_names)
    drug_intervention_count = len(drug_intervention_names)

    # protocolSection.sponsorCollaboratorsModule
    sponsor_module = safe_get(protocol, "sponsorCollaboratorsModule", default={}) or {}
    lead_sponsor_info = sponsor_module.get("leadSponsor", {}) or {}
    lead_sponsor = lead_sponsor_info.get("name")
    sponsor_class = lead_sponsor_info.get("class")
    collaborator_names = [
        collaborator.get("name")
        for collaborator in sponsor_module.get("collaborators", []) or []
        if isinstance(collaborator, dict) and collaborator.get("name")
    ]
    collaborators = join_values(collaborator_names)
    collaborator_count = len(collaborator_names)

    # protocolSection.contactsLocationsModule.locations
    contacts_module = safe_get(protocol, "contactsLocationsModule", default={}) or {}
    locations = contacts_module.get("locations", []) or []
    countries_set = {
        location.get("country")
        for location in locations
        if isinstance(location, dict) and location.get("country")
    }
    countries = join_values(sorted(countries_set))
    location_count = len(locations)
    country_count = len(countries_set)

    # protocolSection.descriptionModule
    description_module = safe_get(protocol, "descriptionModule", default={}) or {}
    brief_summary = description_module.get("briefSummary")

    # protocolSection.eligibilityModule
    eligibility_module = safe_get(protocol, "eligibilityModule", default={}) or {}
    eligibility_criteria = eligibility_module.get("eligibilityCriteria")

    # protocolSection.outcomesModule.primaryOutcomes and secondaryOutcomes
    outcomes_module = safe_get(protocol, "outcomesModule", default={}) or {}
    primary_outcomes = outcomes_module.get("primaryOutcomes", []) or []
    secondary_outcomes = outcomes_module.get("secondaryOutcomes", []) or []
    primary_outcome_measures = join_values(
        outcome.get("measure")
        for outcome in primary_outcomes
        if isinstance(outcome, dict) and outcome.get("measure")
    )
    primary_outcome_timeframes = join_values(
        outcome.get("timeFrame")
        for outcome in primary_outcomes
        if isinstance(outcome, dict) and outcome.get("timeFrame")
    )
    secondary_outcome_measures = join_values(
        outcome.get("measure")
        for outcome in secondary_outcomes
        if isinstance(outcome, dict) and outcome.get("measure")
    )

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "overall_status": overall_status,
        "start_date": start_date,
        "primary_completion_date": primary_completion_date,
        "completion_date": completion_date,
        "study_first_submit_date": study_first_submit_date,
        "last_update_submit_date": last_update_submit_date,
        "phases": phases,
        "study_type": study_type,
        "allocation": allocation,
        "intervention_model": intervention_model,
        "masking": masking,
        "primary_purpose": primary_purpose,
        "enrollment_count": enrollment_count,
        "enrollment_type": enrollment_type,
        "conditions": conditions,
        "condition_count": condition_count,
        "search_query_source": study.get("search_query_source"),
        "intervention_names": intervention_names_joined,
        "intervention_types": intervention_types,
        "drug_intervention_names": drug_intervention_names_joined,
        "drug_intervention_count": drug_intervention_count,
        "lead_sponsor": lead_sponsor,
        "sponsor_class": sponsor_class,
        "collaborators": collaborators,
        "collaborator_count": collaborator_count,
        "location_count": location_count,
        "country_count": country_count,
        "countries": countries,
        "brief_summary": brief_summary,
        "eligibility_criteria": eligibility_criteria,
        "primary_outcome_measures": primary_outcome_measures,
        "primary_outcome_timeframes": primary_outcome_timeframes,
        "secondary_outcome_measures": secondary_outcome_measures,
    }


def studies_to_dataframe(studies: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten raw API records into a de-duplicated V2 DataFrame."""
    records = [flatten_study(study) for study in tqdm(studies, desc="Flattening studies")]
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.drop_duplicates(subset="nct_id", keep="first").reset_index(drop=True)


def filter_drug_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Keep trials with at least one DRUG intervention."""
    if df.empty or "intervention_types" not in df.columns:
        return df.iloc[0:0].copy()
    is_drug = df["intervention_types"].fillna("").str.upper().str.contains("DRUG")
    return df[is_drug].copy().reset_index(drop=True)


def print_diagnostics(df: pd.DataFrame) -> None:
    """Print basic V2 extraction diagnostics after saving."""
    print("\nV2 extraction diagnostics")
    print("-" * 30)
    if df.empty:
        print("Saved rows: 0")
        print("Unique NCT IDs: 0")
        print("Rows with start_date: 0")
        print("Rows with completion_date: 0")
        print("Rows with primary_completion_date: 0")
        print("Drug-related trials: 0")
        return

    drug_rows = df["intervention_types"].fillna("").str.upper().str.contains("DRUG").sum()
    print(f"Saved rows: {len(df):,}")
    print(f"Unique NCT IDs: {df['nct_id'].nunique():,}")
    print(f"Rows with start_date: {df['start_date'].notna().sum():,}")
    print(f"Rows with completion_date: {df['completion_date'].notna().sum():,}")
    print(
        "Rows with primary_completion_date: "
        f"{df['primary_completion_date'].notna().sum():,}"
    )
    print(f"Drug-related trials: {drug_rows:,}")


def main(
    max_total_records: int = V2_MAX_TRIALS,
    page_size: int = V2_PAGE_SIZE,
    output_path: str = V2_RAW_CSV_PATH,
    search_queries: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch a limited sample and write `data/raw/v2_trials_raw.csv`.

    By default, this pulls a small sample across the existing project search
    queries. Increase `max_total_records` later when the V2 extraction shape is
    validated.
    """
    queries = search_queries or SEARCH_QUERIES
    records_per_query = max(1, max_total_records // max(1, len(queries)))
    all_studies: List[Dict[str, Any]] = []

    print("Fetching V2 ClinicalTrials.gov sample")
    print(f"Target maximum records before drug filtering: {max_total_records:,}")

    for query in queries:
        if len(all_studies) >= max_total_records:
            break
        remaining = max_total_records - len(all_studies)
        query_limit = min(records_per_query, remaining)
        print(f"Fetching query '{query}' (up to {query_limit} records)")
        studies = fetch_studies(
            query_term=query,
            page_size=page_size,
            max_records=query_limit,
        )
        for study in studies:
            study["search_query_source"] = query
        all_studies.extend(studies)

    raw_df = studies_to_dataframe(all_studies)
    drug_df = filter_drug_trials(raw_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    drug_df.to_csv(output_path, index=False)

    print(f"\nSaved V2 raw dataset to: {output_path}")
    print_diagnostics(drug_df)
    return drug_df


def parse_args() -> argparse.Namespace:
    """Parse CLI options for V2 ClinicalTrials.gov extraction."""
    parser = argparse.ArgumentParser(
        description="Fetch a V2 ClinicalTrials.gov sample for success/duration modeling.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=V2_MAX_TRIALS,
        help=f"Maximum records to fetch before drug filtering. Default: {V2_MAX_TRIALS}.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=V2_PAGE_SIZE,
        help=f"ClinicalTrials.gov API page size. Default: {V2_PAGE_SIZE}.",
    )
    parser.add_argument(
        "--output",
        default=V2_RAW_CSV_PATH,
        help=f"Output CSV path. Default: {V2_RAW_CSV_PATH}.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        max_total_records=args.max_trials,
        page_size=args.page_size,
        output_path=args.output,
    )
