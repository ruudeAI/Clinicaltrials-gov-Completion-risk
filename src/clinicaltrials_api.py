"""
clinicaltrials_api.py — Fetches clinical trial data from ClinicalTrials.gov API v2.

WHAT THIS FILE DOES:
    1. Connects to the ClinicalTrials.gov REST API (no API key needed!)
    2. Searches for cancer-related clinical trials
    3. Handles pagination automatically (fetches multiple pages of results)
    4. Flattens the deeply nested JSON response into a clean, flat table
    5. Saves a full raw CSV and a 50-row sample CSV

HOW TO RUN:
    python src/clinicaltrials_api.py

WHAT YOU'LL GET:
    data/raw/cancer_trials_raw.csv       — Full dataset of all fetched trials
    data/sample/cancer_trials_sample.csv — Small 50-row sample for quick demos

BEGINNER CONCEPTS:
    - REST API: A way to request data from a website using URLs and parameters
    - JSON: A text format for structured data (like nested Python dictionaries)
    - Pagination: When results are too many for one response, the API sends them
      in "pages" and gives you a token to fetch the next page
    - Flattening: Converting deeply nested data into a simple flat table (CSV)

DATA SOURCE:
    This project uses data from ClinicalTrials.gov (https://clinicaltrials.gov/).
    The data is publicly available under ClinicalTrials.gov's terms of use.
"""

import os
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm

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
    SEARCH_QUERIES,
    TRIALS_PER_CONDITION,
    DRUG_RAW_CSV_PATH as RAW_CSV_PATH,
    DRUG_SAMPLE_CSV_PATH as SAMPLE_CSV_PATH,
    API_RATE_LIMIT_SECONDS as RATE_LIMIT_SECONDS,
    RANDOM_SEED
)

# ==============================================================================
# CONSTANTS
# ==============================================================================
# API endpoint for ClinicalTrials.gov v2
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"



# ==============================================================================
# HELPER: Safely navigate nested dictionaries
# ==============================================================================

def safe_get(data, *keys, default=None):
    """
    Safely navigate a nested dictionary without crashing on missing keys.

    The ClinicalTrials.gov API returns deeply nested JSON. For example:
        study["protocolSection"]["designModule"]["designInfo"]["allocation"]

    If ANY key in that chain is missing, Python crashes with a KeyError.
    This function returns a safe default value instead.

    Parameters
    ----------
    data : dict
        The nested dictionary to navigate.
    *keys : str
        The chain of keys to follow, one after another.
    default : any, optional
        Value to return if any key is missing (default: None).

    Examples
    --------
    >>> d = {"a": {"b": {"c": 42}}}
    >>> safe_get(d, "a", "b", "c")
    42
    >>> safe_get(d, "a", "x", "y")  # "x" doesn't exist
    None
    >>> safe_get(d, "a", "x", "y", default="N/A")
    'N/A'
    """
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current


# ==============================================================================
# CORE FUNCTION 1: Fetch studies from the API
# ==============================================================================

def fetch_studies(query_term="cancer", page_size=100, max_pages=10):
    """
    Fetch clinical trial records from the ClinicalTrials.gov API v2.

    Sends paginated HTTP GET requests to the API, collecting up to
    (page_size × max_pages) trial records. Handles errors gracefully
    and includes rate limiting to avoid overwhelming the server.

    Parameters
    ----------
    query_term : str, optional
        Search term for finding trials (default: "cancer").
    page_size : int, optional
        Number of trials per API request / page (default: 100, max: 1000).
    max_pages : int, optional
        Maximum number of pages to fetch (default: 10).
        Total trials ≈ page_size × max_pages.

    Returns
    -------
    list of dict
        A list of raw study records (nested dictionaries) from the API.

    Example
    -------
    >>> studies = fetch_studies("cancer", page_size=50, max_pages=2)
    >>> print(f"Fetched {len(studies)} trials")
    Fetched 100 trials
    """
    all_studies = []
    page_token = None  # Used for pagination — None means "start from page 1"

    print(f"\n{'='*60}")
    print(f"  Fetching trials from ClinicalTrials.gov API")
    print(f"  Search term: '{query_term}'")
    print(f"  Page size: {page_size} | Max pages: {max_pages}")
    print(f"{'='*60}\n")

    # Loop through pages using tqdm for a progress bar
    for page_num in tqdm(range(1, max_pages + 1), desc="  Fetching pages", unit="page"):

        # ------------------------------------------------------------------
        # Build the query parameters for this request
        # ------------------------------------------------------------------
        params = {
            "query.term": query_term,   # What to search for
            "pageSize": page_size,      # How many results per page
            "format": "json",           # Response format
        }

        # If we have a page token from the previous response, add it
        # to get the NEXT page of results
        if page_token is not None:
            params["pageToken"] = page_token

        # ------------------------------------------------------------------
        # Make the HTTP request with error handling
        # ------------------------------------------------------------------
        try:
            response = requests.get(
                API_BASE_URL,
                params=params,
                timeout=30,  # Wait at most 30 seconds for a response
            )
            # Raise an exception if the HTTP status code indicates an error
            response.raise_for_status()

        except requests.exceptions.Timeout:
            print(f"\n  ⚠️  Page {page_num}: Request timed out. Retrying once...")
            try:
                time.sleep(2)
                response = requests.get(API_BASE_URL, params=params, timeout=60)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Page {page_num}: Retry also failed: {e}")
                break

        except requests.exceptions.RequestException as e:
            print(f"\n  ❌ Page {page_num}: Request failed: {e}")
            break

        # ------------------------------------------------------------------
        # Parse the JSON response
        # ------------------------------------------------------------------
        data = response.json()

        # The API returns studies in a "studies" array
        studies = data.get("studies", [])
        if not studies:
            print(f"\n  ℹ️  No more studies returned on page {page_num}. Stopping.")
            break

        all_studies.extend(studies)

        # ------------------------------------------------------------------
        # Check for next page
        # ------------------------------------------------------------------
        page_token = data.get("nextPageToken")
        if page_token is None:
            # No more pages available
            break

        # ------------------------------------------------------------------
        # Rate limiting: pause between requests
        # ------------------------------------------------------------------
        time.sleep(RATE_LIMIT_SECONDS)

    print(f"\n  ✓ Total studies fetched: {len(all_studies)}")
    return all_studies


# ==============================================================================
# CORE FUNCTION 2: Flatten one study record
# ==============================================================================

def flatten_study(study):
    """
    Extract key fields from a single nested study record into a flat dictionary.

    The API returns deeply nested JSON. This function pulls out the 26 fields
    we care about and puts them into a simple, flat dictionary — one key per
    column in our final CSV.

    Parameters
    ----------
    study : dict
        A single study record from the API (deeply nested).

    Returns
    -------
    dict
        A flat dictionary with 26 key-value pairs.
    """
    # Most fields live under "protocolSection"
    protocol = study.get("protocolSection", {})

    # ------------------------------------------------------------------
    # 1. IDENTIFICATION — What is this trial called?
    # ------------------------------------------------------------------
    id_module = safe_get(protocol, "identificationModule", default={})
    nct_id = id_module.get("nctId")
    brief_title = id_module.get("briefTitle")
    official_title = id_module.get("officialTitle")

    # ------------------------------------------------------------------
    # 2. STATUS — Is this trial still running? Completed? Terminated?
    # ------------------------------------------------------------------
    status_module = safe_get(protocol, "statusModule", default={})
    overall_status = status_module.get("overallStatus")

    # ------------------------------------------------------------------
    # 3. DESIGN — How is the trial structured?
    # ------------------------------------------------------------------
    design_module = safe_get(protocol, "designModule", default={})

    # Study type: INTERVENTIONAL, OBSERVATIONAL, etc.
    study_type = design_module.get("studyType")

    # Phase: ["PHASE1"], ["PHASE2", "PHASE3"], etc.
    # We join multiple phases with "|" to store as a single string
    phases_list = design_module.get("phases", [])
    phases = "|".join(phases_list) if phases_list else None

    # Enrollment: how many participants
    enrollment_info = design_module.get("enrollmentInfo", {}) or {}
    enrollment_count = enrollment_info.get("count")
    enrollment_type = enrollment_info.get("type")  # ACTUAL or ESTIMATED

    # Design details: allocation, masking, etc.
    design_info = design_module.get("designInfo", {}) or {}
    allocation = design_info.get("allocation")
    intervention_model = design_info.get("interventionModel")
    primary_purpose = design_info.get("primaryPurpose")

    # Masking is nested one level deeper
    masking_info = design_info.get("maskingInfo", {}) or {}
    masking = masking_info.get("masking")

    # ------------------------------------------------------------------
    # 4. SPONSOR — Who is funding / running the trial?
    # ------------------------------------------------------------------
    sponsor_module = safe_get(protocol, "sponsorCollaboratorsModule", default={})

    lead_sponsor = sponsor_module.get("leadSponsor", {}) or {}
    sponsor_name = lead_sponsor.get("name")
    sponsor_class = lead_sponsor.get("class")  # INDUSTRY, NIH, OTHER, etc.

    # Count collaborating organizations
    collaborators = sponsor_module.get("collaborators", []) or []
    collaborator_count = len(collaborators)

    # ------------------------------------------------------------------
    # 5. CONDITIONS — What diseases does this trial study?
    # ------------------------------------------------------------------
    conditions_module = safe_get(protocol, "conditionsModule", default={})
    conditions_list = conditions_module.get("conditions", []) or []
    # Join multiple conditions with "|"
    conditions = "|".join(conditions_list) if conditions_list else None

    # ------------------------------------------------------------------
    # 6. INTERVENTIONS — What treatments are being tested?
    # ------------------------------------------------------------------
    arms_module = safe_get(protocol, "armsInterventionsModule", default={})
    interventions = arms_module.get("interventions", []) or []

    # Collect unique intervention types (DRUG, DEVICE, BIOLOGICAL, etc.)
    int_types = set()
    int_names = []
    for intervention in interventions:
        if isinstance(intervention, dict):
            itype = intervention.get("type", "")
            iname = intervention.get("name", "")
            if itype:
                int_types.add(itype)
            if iname:
                int_names.append(iname)

    intervention_types = "|".join(sorted(int_types)) if int_types else None
    intervention_names = "|".join(int_names) if int_names else None

    # ------------------------------------------------------------------
    # 7. ELIGIBILITY — Who can participate in this trial?
    # ------------------------------------------------------------------
    elig_module = safe_get(protocol, "eligibilityModule", default={})
    sex = elig_module.get("sex")
    minimum_age = elig_module.get("minimumAge")
    maximum_age = elig_module.get("maximumAge")
    healthy_volunteers = elig_module.get("healthyVolunteers")
    eligibility_criteria = elig_module.get("eligibilityCriteria")

    # ------------------------------------------------------------------
    # 8. LOCATIONS — Where is this trial being conducted?
    # ------------------------------------------------------------------
    contacts_module = safe_get(protocol, "contactsLocationsModule", default={})
    locations = contacts_module.get("locations", []) or []
    location_count = len(locations)

    # Extract unique countries from all locations
    country_set = set()
    for loc in locations:
        if isinstance(loc, dict):
            country = loc.get("country")
            if country:
                country_set.add(country)
    countries = "|".join(sorted(country_set)) if country_set else None

    # ------------------------------------------------------------------
    # 9. DESCRIPTION — What is this trial about?
    # ------------------------------------------------------------------
    desc_module = safe_get(protocol, "descriptionModule", default={})
    brief_summary = desc_module.get("briefSummary")

    # ------------------------------------------------------------------
    # 10. OUTCOMES — What is the trial measuring?
    # ------------------------------------------------------------------
    outcomes_module = safe_get(protocol, "outcomesModule", default={})
    primary_outcomes = outcomes_module.get("primaryOutcomes", []) or []
    primary_measures = []
    for outcome in primary_outcomes:
        if isinstance(outcome, dict):
            measure = outcome.get("measure")
            if measure:
                primary_measures.append(measure)
    primary_outcome_measures = "|".join(primary_measures) if primary_measures else None

    # ------------------------------------------------------------------
    # BUILD THE FLAT DICTIONARY
    # ------------------------------------------------------------------
    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "overall_status": overall_status,
        "study_type": study_type,
        "phases": phases,
        "enrollment_count": enrollment_count,
        "enrollment_type": enrollment_type,
        "sponsor_name": sponsor_name,
        "sponsor_class": sponsor_class,
        "collaborator_count": collaborator_count,
        "conditions": conditions,
        "intervention_types": intervention_types,
        "intervention_names": intervention_names,
        "allocation": allocation,
        "intervention_model": intervention_model,
        "masking": masking,
        "primary_purpose": primary_purpose,
        "sex": sex,
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "healthy_volunteers": healthy_volunteers,
        "eligibility_criteria": eligibility_criteria,
        "location_count": location_count,
        "countries": countries,
        "brief_summary": brief_summary,
        "primary_outcome_measures": primary_outcome_measures,
        "search_query_source": study.get("search_query_source", "user_input"),
    }



# ==============================================================================
# CORE FUNCTION 3: Convert all studies to a DataFrame
# ==============================================================================

def studies_to_dataframe(studies):
    """
    Convert a list of nested study records into a flat pandas DataFrame.

    Applies flatten_study() to each record, then combines them into a table.

    Parameters
    ----------
    studies : list of dict
        Raw study records from the API.

    Returns
    -------
    pd.DataFrame
        A flat DataFrame with one row per trial and 26 columns.
    """
    print("\n  Flattening nested JSON into a flat table...")
    records = []
    for study in tqdm(studies, desc="  Processing studies", unit=" trials"):
        flat = flatten_study(study)
        records.append(flat)

    df = pd.DataFrame(records)

    # Drop duplicate trials (same NCT ID appearing multiple times)
    before = len(df)
    df = df.drop_duplicates(subset="nct_id", keep="first")
    after = len(df)
    if before > after:
        print(f"  ℹ️  Removed {before - after} duplicate trials")

    return df


# ==============================================================================
# MAIN FUNCTION: Orchestrate everything
# ==============================================================================

def main():
    """
    Main function that runs the full data collection pipeline:
    1. Fetch studies from the API for all configured search queries
    2. Flatten into a DataFrame
    3. Filter to keep only trials with DRUG intervention
    4. Save raw CSV
    5. Save sample CSV
    6. Print summary statistics
    """
    print("=" * 60)
    print("  🏥 Clinical Trials Data Collection Pipeline")
    print("=" * 60)
    print("  Data source: ClinicalTrials.gov (https://clinicaltrials.gov/)")
    print("  This project is for educational/portfolio purposes only.\n")

    # ------------------------------------------------------------------
    # Step 1: Fetch studies from the API across all queries
    # ------------------------------------------------------------------
    all_studies = []
    page_size = 100
    max_pages = max(1, TRIALS_PER_CONDITION // page_size)

    for term in SEARCH_QUERIES:
        print(f"\n--- Fetching trials for query: '{term}' ---")
        studies = fetch_studies(
            query_term=term,
            page_size=page_size,
            max_pages=max_pages,
        )
        for s in studies:
            s["search_query_source"] = term
        all_studies.extend(studies)

    if not all_studies:
        print("  ❌ No studies fetched. Check your internet connection.")
        return

    # ------------------------------------------------------------------
    # Step 2: Flatten into a DataFrame
    # ------------------------------------------------------------------
    df = studies_to_dataframe(all_studies)
    print(f"\n  ✓ Combined DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Keep only trials where intervention_types contains 'DRUG' (case-insensitive)
    before_drug_filter = len(df)
    df = df[df["intervention_types"].fillna("").str.upper().str.contains("DRUG")].copy()
    after_drug_filter = len(df)
    print(f"  ✓ Filtered by DRUG intervention: kept {after_drug_filter:,} trials "
          f"(dropped {before_drug_filter - after_drug_filter:,} non-drug trials)")

    # ------------------------------------------------------------------
    # Step 3: Save the full raw CSV
    # ------------------------------------------------------------------
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(RAW_CSV_PATH), exist_ok=True)

    df.to_csv(RAW_CSV_PATH, index=False)
    size_mb = os.path.getsize(RAW_CSV_PATH) / (1024 * 1024)
    print(f"\n  💾 Saved full dataset to: {RAW_CSV_PATH} ({size_mb:.2f} MB)")

    # ------------------------------------------------------------------
    # Step 4: Save a 50-row sample CSV (for quick demos)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(SAMPLE_CSV_PATH), exist_ok=True)

    sample_size = min(50, len(df))
    sample_df = df.sample(n=sample_size, random_state=RANDOM_SEED)
    sample_df.to_csv(SAMPLE_CSV_PATH, index=False)
    print(f"  💾 Saved sample dataset to: {SAMPLE_CSV_PATH} ({sample_size} rows)")

    # ------------------------------------------------------------------
    # Step 5: Print summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  📊 Collection Summary")
    print("=" * 60)

    print(f"\n  Unique trials collected: {df['nct_id'].nunique()}")

    print(f"\n  Search Query Source Distribution:")
    if "search_query_source" in df.columns:
        source_counts = df["search_query_source"].value_counts()
        for src, count in source_counts.items():
            print(f"    {src:<30s} {count:>5,}")

    print(f"\n  Overall Status Distribution:")
    status_counts = df["overall_status"].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"    {status:<30s} {count:>5,}  ({pct:5.1f}%)")

    print(f"\n  Study Type Distribution:")
    if "study_type" in df.columns:
        type_counts = df["study_type"].value_counts()
        for stype, count in type_counts.items():
            print(f"    {str(stype):<30s} {count:>5,}")

    print(f"\n  Files saved:")
    print(f"    📄 {RAW_CSV_PATH}")
    print(f"    📄 {SAMPLE_CSV_PATH}")

    print("\n" + "=" * 60)
    print("  ✅ Data collection complete!")
    print(f"  Next step: python src/preprocess.py")
    print("=" * 60)

    return df



# ==============================================================================
# ENTRY POINT
# ==============================================================================
# This block runs ONLY when you execute this file directly:
#     python src/clinicaltrials_api.py
# It does NOT run when another file imports functions from this module.

if __name__ == "__main__":
    main()
