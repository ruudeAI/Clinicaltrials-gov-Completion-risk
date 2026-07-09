"""
endpoint_features.py - Version 2 endpoint-type feature placeholder.

Purpose
-------
Extract endpoint type indicators from ClinicalTrials.gov outcome text for use
in success classification and duration regression.

Inputs
------
- Primary outcome measure text.
- Optional outcome descriptions and time frames if added to the V2 extractor.

Outputs
-------
- `endpoint_safety`
- `endpoint_efficacy`
- `endpoint_survival`
- `endpoint_biomarker`
- `endpoint_response_rate`
- Optional endpoint count and endpoint text-length features.

Data Leakage Warnings
---------------------
- Use protocol endpoint descriptions, not posted result values.
- Do not extract endpoint success/failure or statistical significance from
  results sections as features.
- Keep endpoint type features separate from success labels.

TODO Implementation Steps
-------------------------
1. Define keyword dictionaries for safety, efficacy, survival, biomarker, and
   response-rate endpoints.
2. Normalize endpoint text.
3. Create binary endpoint flags.
4. Add endpoint text-length and endpoint type-count features.
5. Validate keyword rules on a small audited trial sample.
6. Add tests for common endpoint phrases.
"""


def placeholder():
    """Placeholder to keep the module importable until V2 logic is implemented."""
    raise NotImplementedError("Endpoint feature extraction is planned for Version 2.")

