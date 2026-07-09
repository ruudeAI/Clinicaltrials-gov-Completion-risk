"""
Lightweight validation tests for src.chembl_api.

These tests mock ChEMBL network calls so they can run even when the live API or
network is unavailable. Run directly with:

    python tests/test_chembl_api.py
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import requests


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import chembl_api


EXPECTED_OUTPUT_FIELDS = {
    "original_drug_name",
    "chembl_id",
    "preferred_name",
    "molecule_type",
    "max_phase",
    "first_match_confidence",
    "match_notes",
}


class FakeResponse:
    """Small fake `requests` response for mocked ChEMBL lookups."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def molecule_payload(chembl_id, preferred_name, molecule_type, max_phase):
    """Build a minimal ChEMBL search payload."""
    return {
        "molecules": [
            {
                "molecule_chembl_id": chembl_id,
                "pref_name": preferred_name,
                "molecule_type": molecule_type,
                "max_phase": max_phase,
            }
        ]
    }


def mocked_get(_url, params, timeout):
    """Return deterministic fake ChEMBL responses by search query."""
    del timeout
    query = params.get("q")
    payloads = {
        "aspirin": molecule_payload("CHEMBL25", "ASPIRIN", "Small molecule", 4),
        "imatinib": molecule_payload("CHEMBL941", "IMATINIB", "Small molecule", 4),
        "pembrolizumab": molecule_payload(
            "CHEMBL3137343",
            "PEMBROLIZUMAB",
            "Antibody",
            4,
        ),
        "definitelynotarealdrugxyz": {"molecules": []},
    }
    return FakeResponse(payloads.get(query, {"molecules": []}))


class ChemblApiValidationTests(unittest.TestCase):
    """Network-safe checks for ChEMBL enrichment behavior."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.temp_dir.name, "chembl_cache.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    def assert_output_contract(self, row):
        self.assertEqual(set(row.keys()), EXPECTED_OUTPUT_FIELDS)

    @patch("src.chembl_api.requests.get", side_effect=mocked_get)
    def test_known_example_drugs_return_stable_fields(self, _mock_get):
        examples = {
            "aspirin": "CHEMBL25",
            "imatinib": "CHEMBL941",
            "pembrolizumab": "CHEMBL3137343",
        }

        for drug_name, expected_id in examples.items():
            with self.subTest(drug_name=drug_name):
                row = chembl_api.lookup_drug(drug_name, cache_path=self.cache_path)
                self.assert_output_contract(row)
                self.assertEqual(row["original_drug_name"], drug_name)
                self.assertEqual(row["chembl_id"], expected_id)
                self.assertNotIn("normalized_drug_name", row)

    @patch("src.chembl_api.requests.get", side_effect=mocked_get)
    def test_empty_string_returns_unknown_without_api_call(self, mock_get):
        row = chembl_api.lookup_drug("", cache_path=self.cache_path)

        self.assert_output_contract(row)
        self.assertEqual(row["chembl_id"], "UNKNOWN")
        self.assertIn("blank drug name", row["match_notes"])
        mock_get.assert_not_called()

    @patch("src.chembl_api.requests.get", side_effect=mocked_get)
    def test_unknown_fake_drug_returns_unknown_no_match(self, _mock_get):
        row = chembl_api.lookup_drug(
            "definitelynotarealdrugxyz",
            cache_path=self.cache_path,
        )

        self.assert_output_contract(row)
        self.assertEqual(row["chembl_id"], "UNKNOWN")
        self.assertIn("No ChEMBL molecule search match", row["match_notes"])

    @patch("src.chembl_api.requests.get", side_effect=mocked_get)
    def test_repeated_lookup_uses_cache(self, mock_get):
        first = chembl_api.lookup_drug("aspirin", cache_path=self.cache_path)
        second = chembl_api.lookup_drug("aspirin", cache_path=self.cache_path)

        self.assertEqual(first, second)
        self.assertEqual(mock_get.call_count, 1)
        self.assertTrue(os.path.exists(self.cache_path))

        cache = pd.read_csv(self.cache_path, dtype=str)
        self.assertIn("normalized_drug_name", cache.columns)
        self.assertIn("aspirin", set(cache["normalized_drug_name"]))

    @patch(
        "src.chembl_api.requests.get",
        side_effect=requests.exceptions.ConnectionError("network unavailable"),
    )
    def test_network_failure_returns_unknown_with_clear_notes(self, _mock_get):
        row = chembl_api.lookup_drug("aspirin", cache_path=self.cache_path)

        self.assert_output_contract(row)
        self.assertEqual(row["chembl_id"], "UNKNOWN")
        self.assertIn("ChEMBL request failed", row["match_notes"])


def print_validation_summary():
    """Print a short human-readable note for direct script runs."""
    print("ChEMBL validation uses mocked API responses; no live network is required.")
    print(f"Default cache path: {chembl_api.CACHE_PATH}")


if __name__ == "__main__":
    print_validation_summary()
    unittest.main(verbosity=2)

