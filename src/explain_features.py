"""
explain_features.py - Version 2 explainability placeholder.

Purpose
-------
Generate top predictive feature summaries for the Version 2 success classifier
and duration regressor. Basic feature importance should come first; SHAP can be
added later if practical.

Inputs
------
- Trained Version 2 model artifacts.
- Processed Version 2 dataset or held-out evaluation data.
- Preprocessing pipeline feature names.

Outputs
-------
- Top predictive features table.
- Feature-importance plots for success and duration models.
- Optional future SHAP summaries.

Data Leakage Warnings
---------------------
- Explainability should only inspect features that were valid model inputs.
- Do not explain label provenance fields, progression evidence, approval
  evidence, or actual duration as if they were legitimate predictors.
- Report whether feature importance comes from random or time-based evaluation.

TODO Implementation Steps
-------------------------
1. Load trained V2 model pipelines.
2. Extract transformed feature names from the preprocessing step.
3. Support coefficients, tree feature importances, and permutation importance.
4. Write a top predictive features markdown report.
5. Generate plots under `reports/figures/`.
6. Add SHAP support only after baseline reports are stable.
"""


def placeholder():
    """Placeholder to keep the module importable until V2 logic is implemented."""
    raise NotImplementedError("Feature explainability is planned for Version 2.")

