"""
v2_enrich_chembl.py - Add ChEMBL molecule and modality fields to V2 data.

Purpose
-------
Enrich the Version 2 interim dataset with public ChEMBL molecule metadata and a
simple derived drug modality. This supports Narayanan's requirement to include
molecule and drug modality information.

Inputs
------
- `data/interim/v2_trials_with_success_labels.csv`

Outputs
-------
- `data/interim/v2_trials_with_chembl.csv`

Leakage Note
------------
`max_phase` is preserved for reporting/auditing only. Do not use ChEMBL
`max_phase` as a model feature in time-based modeling unless a later explicit
feature policy confirms it is available at the prediction date.
"""

import argparse
import os
import re
import sys
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.chembl_api import lookup_drug


INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "v2_trials_with_success_labels.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "v2_trials_with_chembl.csv")

CHEMBL_COLUMNS = [
    "chembl_id",
    "preferred_name",
    "molecule_type",
    "max_phase",
    "first_match_confidence",
    "match_notes",
]


def _clean_name_token(value: object) -> str:
    """Clean one candidate intervention name for ChEMBL lookup."""
    if value is None or pd.isna(value):
        return ""
    name = str(value).strip()
    if not name:
        return ""
    name = re.sub(r"\([^)]*\)", " ", name)
    name = re.sub(r"\b(placebo|matching placebo|vehicle)\b", " ", name, flags=re.I)
    name = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?)\b", " ", name, flags=re.I)
    name = re.sub(r"[^A-Za-z0-9\-/ ]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _split_candidate_names(value: object) -> List[str]:
    """Split pipe/combo intervention strings into candidate molecule names."""
    if value is None or pd.isna(value):
        return []
    pieces: List[str] = []
    for pipe_piece in str(value).split("|"):
        pieces.extend(re.split(r"\s+\+\s+|,|;|\s+ and \s+|/", pipe_piece, flags=re.I))

    names = []
    for piece in pieces:
        cleaned = _clean_name_token(piece)
        if cleaned and len(cleaned) > 2:
            names.append(cleaned)
    return names


def extract_primary_molecule_name(row: pd.Series) -> str:
    """
    Extract the first usable molecule/drug name from available intervention fields.

    Priority:
    1. `drug_intervention_names`
    2. `intervention_names_normalized`
    3. `intervention_names`
    """
    for col in ["drug_intervention_names", "intervention_names_normalized", "intervention_names"]:
        if col not in row:
            continue
        candidates = _split_candidate_names(row.get(col))
        if candidates:
            return candidates[0]
    return "UNKNOWN"


def derive_drug_modality(molecule_name: object, molecule_type: object, chembl_id: object) -> str:
    """Derive a coarse modality from ChEMBL molecule_type plus name/type rules."""
    chembl = "" if chembl_id is None or pd.isna(chembl_id) else str(chembl_id).strip().upper()
    name = "" if molecule_name is None or pd.isna(molecule_name) else str(molecule_name).lower()
    mtype = "" if molecule_type is None or pd.isna(molecule_type) else str(molecule_type).lower()
    combined = f"{name} {mtype}"

    if not chembl or chembl == "UNKNOWN":
        return "unknown"
    if any(term in combined for term in ["vaccine"]):
        return "vaccine"
    if any(
        term in combined
        for term in [
            "gene therapy",
            "viral vector",
            "aav",
            "car-t",
            "cart",
            "cell therapy",
            "cellular therapy",
        ]
    ):
        return "cell_or_gene_therapy"
    if any(term in combined for term in ["oligonucleotide", "sirna", "antisense", "rna"]):
        return "oligonucleotide"
    if any(term in combined for term in ["antibody", "monoclonal antibody", " mab"]):
        return "biologic_antibody"
    if any(term in combined for term in ["protein", "enzyme", "peptide"]):
        return "biologic_protein"
    if "small molecule" in combined:
        return "small_molecule"
    return "unknown"


def enrich_unique_molecules(
    molecule_names: List[str],
    retry_cached_failures: bool = False,
) -> pd.DataFrame:
    """Look up unique molecule names via ChEMBL and return enrichment rows."""
    rows = []
    for molecule_name in tqdm(molecule_names, desc="ChEMBL molecule lookups"):
        if not molecule_name or molecule_name == "UNKNOWN":
            result = {
                "molecule_name": molecule_name or "UNKNOWN",
                "chembl_id": "UNKNOWN",
                "preferred_name": "UNKNOWN",
                "molecule_type": "UNKNOWN",
                "max_phase": "UNKNOWN",
                "first_match_confidence": "none",
                "match_notes": "Missing molecule name.",
            }
        else:
            result = lookup_drug(
                molecule_name,
                retry_cached_failures=retry_cached_failures,
            )
            result = {
                "molecule_name": molecule_name,
                "chembl_id": result.get("chembl_id", "UNKNOWN"),
                "preferred_name": result.get("preferred_name", "UNKNOWN"),
                "molecule_type": result.get("molecule_type", "UNKNOWN"),
                "max_phase": result.get("max_phase", "UNKNOWN"),
                "first_match_confidence": result.get("first_match_confidence", "none"),
                "match_notes": result.get("match_notes", "UNKNOWN"),
            }
        result["drug_modality"] = derive_drug_modality(
            result["molecule_name"],
            result["molecule_type"],
            result["chembl_id"],
        )
        rows.append(result)
    return pd.DataFrame(rows)


def rituximab_sanity_check(df: pd.DataFrame) -> None:
    """Print a non-blocking sanity check for the rituximab ChEMBL mapping."""
    mask = df["molecule_name"].fillna("").str.lower() == "rituximab"
    if not mask.any():
        print("Rituximab sanity check: rituximab not present in molecule_name.")
        return

    observed_ids = sorted(set(df.loc[mask, "chembl_id"].fillna("UNKNOWN").astype(str)))
    if "CHEMBL1201576" in observed_ids:
        print("Rituximab sanity check: PASS - mapped to CHEMBL1201576.")
    else:
        print(
            "Rituximab sanity check: REVIEW - rituximab present but observed "
            f"ChEMBL IDs were {observed_ids}. Expected CHEMBL1201576 if ChEMBL "
            "returns the standard rituximab match."
        )


def print_diagnostics(df: pd.DataFrame) -> None:
    """Print ChEMBL enrichment diagnostics."""
    matched = df["chembl_id"].fillna("UNKNOWN").ne("UNKNOWN")
    print("\nV2 ChEMBL enrichment diagnostics")
    print("-" * 36)
    print(f"Total rows: {len(df):,}")
    print(f"Unique molecule names: {df['molecule_name'].nunique(dropna=True):,}")
    print(f"ChEMBL matched count: {int(matched.sum()):,}")
    print(f"Unknown/no-match count: {int((~matched).sum()):,}")
    print("\nDrug modality value counts:")
    print(df["drug_modality"].fillna("unknown").value_counts().to_string())
    print("\nSample enriched rows:")
    sample_cols = [
        "molecule_name",
        "chembl_id",
        "preferred_name",
        "molecule_type",
        "drug_modality",
    ]
    print(df[sample_cols].head(12).to_string(index=False))


def enrich_v2_with_chembl(
    input_path: str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
    retry_cached_failures: bool = False,
) -> pd.DataFrame:
    """Load V2 labeled data, add ChEMBL enrichment, and save interim output."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Run src/success_labels.py first."
        )

    df = pd.read_csv(input_path)
    df["molecule_name"] = df.apply(extract_primary_molecule_name, axis=1)

    unique_molecules = sorted(df["molecule_name"].dropna().unique())
    enrichment = enrich_unique_molecules(
        unique_molecules,
        retry_cached_failures=retry_cached_failures,
    )
    enriched = df.merge(enrichment, on="molecule_name", how="left")

    for col in CHEMBL_COLUMNS + ["drug_modality"]:
        if col not in enriched.columns:
            enriched[col] = "UNKNOWN" if col != "drug_modality" else "unknown"
        enriched[col] = enriched[col].fillna("UNKNOWN" if col != "drug_modality" else "unknown")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enriched.to_csv(output_path, index=False)

    print(f"Saved V2 ChEMBL-enriched dataset to: {output_path}")
    rituximab_sanity_check(enriched)
    print_diagnostics(enriched)
    return enriched


def parse_args() -> argparse.Namespace:
    """Parse command-line options for ChEMBL enrichment."""
    parser = argparse.ArgumentParser(description="Enrich V2 trials with ChEMBL metadata.")
    parser.add_argument("--input", default=INPUT_PATH, help=f"Input CSV. Default: {INPUT_PATH}")
    parser.add_argument("--output", default=OUTPUT_PATH, help=f"Output CSV. Default: {OUTPUT_PATH}")
    parser.add_argument(
        "--retry-cached-failures",
        action="store_true",
        help=(
            "Retry cache rows from previous network/API failures. This can be slow "
            "for large datasets but is useful after a network-blocked run."
        ),
    )
    return parser.parse_args()


def main() -> pd.DataFrame:
    """CLI entry point."""
    args = parse_args()
    return enrich_v2_with_chembl(
        input_path=args.input,
        output_path=args.output,
        retry_cached_failures=args.retry_cached_failures,
    )


if __name__ == "__main__":
    main()
