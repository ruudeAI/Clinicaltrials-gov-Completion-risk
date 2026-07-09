"""
chembl_api.py - Version 2 ChEMBL enrichment utilities.

Purpose
-------
Map ClinicalTrials.gov drug/intervention names to public ChEMBL molecule
metadata for Version 2 success and duration modeling.

Inputs
------
- A drug/intervention name string, such as "imatinib" or "pembrolizumab".
- Optional local cache at `data/cache/chembl_cache.csv`.

Outputs
-------
Each lookup returns a dictionary with:
- `original_drug_name`
- `chembl_id`
- `preferred_name`
- `molecule_type`
- `max_phase`
- `first_match_confidence`
- `match_notes`

Data Leakage Warnings
---------------------
- ChEMBL `max_phase` can encode development progress that may have happened
  after a historical trial's prediction date. Treat it as exploratory unless a
  time-aware feature policy proves it was known at prediction time.
- Do not use missing ChEMBL matches as evidence that a trial failed.
- Keep ChEMBL enrichment features separate from success labels and progression
  evidence.

TODO Implementation Steps
-------------------------
1. Add stronger intervention-name cleaning for placebo, dose, route, and combo
   therapy strings.
2. Add synonym-aware scoring using ChEMBL molecule detail responses.
3. Add batch enrichment for full ClinicalTrials.gov datasets.
4. Add unit tests with mocked ChEMBL API responses.
5. Decide whether `max_phase` belongs in time-aware V2 model features.
"""

import os
import re
import sys
from typing import Dict, List

import pandas as pd
import requests


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "cache", "chembl_cache.csv")
CHEMBL_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json"
REQUEST_TIMEOUT_SECONDS = 20

OUTPUT_COLUMNS = [
    "original_drug_name",
    "chembl_id",
    "preferred_name",
    "molecule_type",
    "max_phase",
    "first_match_confidence",
    "match_notes",
]

CACHE_COLUMNS = [
    "normalized_drug_name",
    *OUTPUT_COLUMNS,
]

UNKNOWN_RESULT = {
    "chembl_id": "UNKNOWN",
    "preferred_name": "UNKNOWN",
    "molecule_type": "UNKNOWN",
    "max_phase": "UNKNOWN",
    "first_match_confidence": "none",
}


def normalize_drug_name(drug_name: str) -> str:
    """
    Normalize an intervention name for cache keys and ChEMBL search.

    This is intentionally conservative. ClinicalTrials.gov intervention names
    can include routes, doses, placebo text, or combination descriptions, so
    the raw `original_drug_name` is always preserved in outputs.
    """
    if drug_name is None:
        return ""

    normalized = str(drug_name).strip().lower()
    normalized = re.sub(r"\b(placebo|matching placebo|vehicle)\b", " ", normalized)
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[^a-z0-9+\-/ ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _empty_cache() -> pd.DataFrame:
    """Return an empty cache DataFrame with the expected columns."""
    return pd.DataFrame(columns=CACHE_COLUMNS)


def load_cache(cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Load the ChEMBL CSV cache.

    Missing or malformed cache files are treated as an empty cache so enrichment
    never blocks the wider pipeline.
    """
    if not os.path.exists(cache_path):
        return _empty_cache()

    try:
        cache = pd.read_csv(cache_path, dtype=str).fillna("UNKNOWN")
    except Exception:
        return _empty_cache()

    for col in CACHE_COLUMNS:
        if col not in cache.columns:
            cache[col] = "UNKNOWN"

    return cache[CACHE_COLUMNS].copy()


def save_cache(cache: pd.DataFrame, cache_path: str = CACHE_PATH) -> None:
    """Save cache rows to `data/cache/chembl_cache.csv`."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = cache.copy()
    for col in CACHE_COLUMNS:
        if col not in cache.columns:
            cache[col] = "UNKNOWN"
    cache[CACHE_COLUMNS].drop_duplicates(
        subset=["normalized_drug_name"],
        keep="last",
    ).to_csv(cache_path, index=False)


def _unknown_row(original_drug_name: str, normalized_drug_name: str, notes: str) -> Dict[str, str]:
    """Build a safe UNKNOWN result row for no-match and error cases."""
    return {
        "normalized_drug_name": normalized_drug_name,
        "original_drug_name": original_drug_name,
        **UNKNOWN_RESULT,
        "match_notes": notes,
    }


def _public_row(row: Dict[str, str]) -> Dict[str, str]:
    """Return only the stable public enrichment output fields."""
    return {col: str(row.get(col, "UNKNOWN")) for col in OUTPUT_COLUMNS}


def _is_retryable_cached_failure(row: Dict[str, str]) -> bool:
    """Return True for cached transient failures that should not poison retries."""
    chembl_id = str(row.get("chembl_id", "UNKNOWN")).upper()
    notes = str(row.get("match_notes", ""))
    retryable_note = (
        "ChEMBL request failed" in notes
        or "Unexpected ChEMBL lookup error" in notes
        or "could not be parsed as JSON" in notes
    )
    return chembl_id == "UNKNOWN" and retryable_note


def _score_first_match(normalized_drug_name: str, molecule: Dict) -> str:
    """Assign a simple confidence label to the first ChEMBL search result."""
    preferred_name = str(molecule.get("pref_name") or "").strip().lower()

    if preferred_name and preferred_name == normalized_drug_name:
        return "exact_preferred_name"
    if preferred_name and normalized_drug_name in preferred_name:
        return "partial_preferred_name"
    return "first_search_result"


def _row_from_molecule(
    original_drug_name: str,
    normalized_drug_name: str,
    molecule: Dict,
) -> Dict[str, str]:
    """Convert a ChEMBL molecule search result into the cache/output schema."""
    confidence = _score_first_match(normalized_drug_name, molecule)
    return {
        "normalized_drug_name": normalized_drug_name,
        "original_drug_name": original_drug_name,
        "chembl_id": molecule.get("molecule_chembl_id") or "UNKNOWN",
        "preferred_name": molecule.get("pref_name") or "UNKNOWN",
        "molecule_type": molecule.get("molecule_type") or "UNKNOWN",
        "max_phase": str(molecule.get("max_phase", "UNKNOWN")),
        "first_match_confidence": confidence,
        "match_notes": "Matched using ChEMBL molecule search first result.",
    }


def _query_chembl(normalized_drug_name: str) -> Dict[str, str]:
    """
    Query ChEMBL's public molecule search endpoint.

    Returns a raw molecule dictionary for the first result, or an empty
    dictionary when no safe match is available.
    """
    if not normalized_drug_name:
        return {}

    params = {"q": normalized_drug_name, "limit": 1}
    response = requests.get(
        CHEMBL_SEARCH_URL,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    molecules = payload.get("molecules", []) or []
    return molecules[0] if molecules else {}


def lookup_drug(
    drug_name: str,
    cache_path: str = CACHE_PATH,
    use_cache: bool = True,
    retry_cached_failures: bool = True,
) -> Dict[str, str]:
    """
    Look up a drug/intervention name in ChEMBL with CSV caching.

    The function never raises for no-match or request failures. Instead it
    returns UNKNOWN values with `match_notes` explaining the issue.
    """
    original_drug_name = "" if drug_name is None else str(drug_name).strip()
    normalized_drug_name = normalize_drug_name(original_drug_name)

    if not normalized_drug_name:
        return _public_row(
            _unknown_row(
                original_drug_name=original_drug_name,
                normalized_drug_name=normalized_drug_name,
                notes="Missing or blank drug name.",
            )
        )

    cache = load_cache(cache_path) if use_cache else _empty_cache()
    cached_rows = cache[cache["normalized_drug_name"] == normalized_drug_name]
    if not cached_rows.empty:
        cached_row = cached_rows.iloc[-1].to_dict()
        if not retry_cached_failures or not _is_retryable_cached_failure(cached_row):
            return _public_row(cached_row)

    try:
        molecule = _query_chembl(normalized_drug_name)
        if molecule:
            row = _row_from_molecule(
                original_drug_name=original_drug_name,
                normalized_drug_name=normalized_drug_name,
                molecule=molecule,
            )
        else:
            row = _unknown_row(
                original_drug_name=original_drug_name,
                normalized_drug_name=normalized_drug_name,
                notes="No ChEMBL molecule search match found.",
            )
    except requests.exceptions.RequestException as exc:
        row = _unknown_row(
            original_drug_name=original_drug_name,
            normalized_drug_name=normalized_drug_name,
            notes=f"ChEMBL request failed: {exc}",
        )
    except ValueError as exc:
        row = _unknown_row(
            original_drug_name=original_drug_name,
            normalized_drug_name=normalized_drug_name,
            notes=f"ChEMBL response could not be parsed as JSON: {exc}",
        )
    except Exception as exc:
        row = _unknown_row(
            original_drug_name=original_drug_name,
            normalized_drug_name=normalized_drug_name,
            notes=f"Unexpected ChEMBL lookup error: {exc}",
        )

    if use_cache:
        updated_cache = pd.concat([cache, pd.DataFrame([row])], ignore_index=True)
        save_cache(updated_cache, cache_path)

    return _public_row(row)


def lookup_many_drugs(
    drug_names: List[str],
    cache_path: str = CACHE_PATH,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Look up multiple drug names and return a DataFrame of enrichment rows."""
    rows = [
        lookup_drug(drug_name, cache_path=cache_path, use_cache=use_cache)
        for drug_name in drug_names
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def demo_lookup() -> pd.DataFrame:
    """
    Run a small demo lookup using three example drugs.

    This function intentionally writes to the normal cache so repeated demos do
    not keep hitting the ChEMBL API.
    """
    example_drugs = ["aspirin", "imatinib", "pembrolizumab"]
    results = lookup_many_drugs(example_drugs)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    demo_lookup()
