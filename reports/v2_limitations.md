# Version 2 Limitations Placeholder

## Purpose

This document will track limitations, assumptions, and responsible-use warnings
for the Version 2 clinical trial success and duration models.

## Inputs

- ClinicalTrials.gov metadata quality and coverage limitations
- ChEMBL molecule-matching limitations
- Success label provenance and uncertainty
- Duration date completeness and date revision issues

## Outputs

- Documented limitations for interpreting V2 predictions
- Known leakage risks and mitigation notes
- Dataset and label caveats for reports and dashboard text

## Data Leakage Warnings

- Clinical success labels based on phase progression or approval evidence must
  not become model input features.
- Sponsor history must be computed only from prior trials.
- Planned dates may be revised and can leak future information if used without
  a clear prediction-time assumption.
- ChEMBL fields that summarize future development status may be inappropriate
  for historical validation.

## TODO Implementation Steps

1. Document the active success-label source.
2. Document known weaknesses in drug/disease matching.
3. Document missing and noisy completion dates for duration modeling.
4. Document random-split optimism compared with time-based validation.
5. Document that predictions are educational/research outputs only.
6. Update after each V2 experiment and evaluation report.

