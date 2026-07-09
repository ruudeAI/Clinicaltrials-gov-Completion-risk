"""
sponsor_features.py - Version 2 sponsor-history feature placeholder.

Purpose
-------
Create time-aware sponsor history features for success and duration models.
Sponsor history should summarize a sponsor's prior trial experience without
using information from the future.

Inputs
------
- Trial-level data with sponsor name, phase, start date, outcome/label fields,
  and duration fields.
- Optional disease indication and molecule identifiers for indication-specific
  or molecule-specific sponsor history.

Outputs
-------
- `sponsor_total_trials`
- `sponsor_completed_trials`
- `sponsor_failed_trials`
- `sponsor_success_rate`
- `sponsor_phase_specific_history`
- Optional recent-window and indication-specific sponsor features.

Data Leakage Warnings
---------------------
- Sponsor features must be computed using only trials before the current
  trial's prediction date.
- Do not let future phase progression, future approvals, or future durations
  enter sponsor history features.
- Do not compute aggregate sponsor rates on the full dataset before splitting.

TODO Implementation Steps
-------------------------
1. Normalize sponsor names.
2. Define `prediction_date`, initially using trial start date.
3. Sort records by prediction date.
4. Compute cumulative prior-only sponsor totals and rates.
5. Add phase-specific sponsor history.
6. Add validation checks that no row uses its own or future outcomes.
"""


def placeholder():
    """Placeholder to keep the module importable until V2 logic is implemented."""
    raise NotImplementedError("Sponsor-history features are planned for Version 2.")

