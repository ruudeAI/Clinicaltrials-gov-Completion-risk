# Version 2 Model Card Placeholder

## Purpose

This document will describe the planned Version 2 Clinical Trial Success and
Duration Prediction models. Version 2 is separate from the existing Version 1
completion-risk model.

## Inputs

- ClinicalTrials.gov API v2 trial metadata
- Public ChEMBL molecule/modality enrichment
- Phase, disease indication, trial design, enrollment, sponsor history, and
  endpoint type features
- Optional CTOD labels only if licensing and documentation allow use

## Outputs

- Probability of clinical trial success
- Predicted trial duration in days
- Top predictive feature summaries
- Phase-specific evaluation metrics

## Data Leakage Warnings

- Do not use actual completion outcome as an input feature.
- Do not use future sponsor history relative to the prediction date.
- Do not use approval or phase-progression evidence as input features.
- Do not use actual completion date or observed duration as duration-model
  features.
- Treat time-based validation as the primary performance estimate.

## TODO Implementation Steps

1. Finalize success target definition.
2. Finalize duration target definition.
3. Document feature groups and feature allowlists.
4. Add random, time-based, and phase-specific evaluation results.
5. Add limitations, intended use, and non-use cases.
6. Add model artifacts and reproducibility details after training.

