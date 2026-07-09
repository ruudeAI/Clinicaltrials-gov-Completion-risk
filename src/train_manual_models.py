"""
train_manual_models.py - Manual-friendly Version 2 model baselines.

Purpose
-------
Train separate V2 models for Streamlit Manual Trial Query using only fields
that the manual form can reliably populate. Existing Trial Lookup should keep
using the richer V2 models.

Inputs
------
- `data/processed/v2_modeling_dataset.csv`

Outputs
-------
- `models/v2_manual_success_classifier.joblib`
- `models/v2_manual_duration_regressor.joblib`
- `reports/v2_manual_model_results.md`

Leakage Rules
-------------
These models do not use actual completion outcome fields, success-label
provenance, progression evidence, ChEMBL `max_phase`, or free-text protocol
fields that the manual form cannot reliably provide.
"""

import os
import sys
from math import sqrt
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "v2_modeling_dataset.csv")
SUCCESS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "v2_manual_success_classifier.joblib")
DURATION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "v2_manual_duration_regressor.joblib")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "v2_manual_model_results.md")

SUCCESS_TARGET = "target_success"
DURATION_TARGET = "observed_duration_days"
RANDOM_SEED = 42
TEST_SIZE = 0.25

MANUAL_NUMERIC_FEATURES = [
    "condition_count",
    "sponsor_prior_trials",
    "sponsor_prior_completed_trials",
    "sponsor_prior_failed_trials",
    "sponsor_prior_completion_rate",
    "sponsor_prior_phase_trials",
    "sponsor_prior_phase_completion_rate",
    "sponsor_prior_avg_duration_days",
    "sponsor_has_prior_history",
    "enrollment_count",
    "country_count",
    "location_count",
    "collaborator_count",
    "endpoint_safety",
    "endpoint_efficacy",
    "endpoint_survival",
    "endpoint_biomarker",
    "endpoint_response_rate",
    "endpoint_type_count",
    "has_primary_endpoint",
]

MANUAL_CATEGORICAL_FEATURES = [
    "conditions_normalized",
    "molecule_type",
    "drug_modality",
    "sponsor_class",
    "phase_normalized",
    "study_type",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
    "enrollment_type",
]

MANUAL_FEATURES = MANUAL_NUMERIC_FEATURES + MANUAL_CATEGORICAL_FEATURES


def load_dataset(path: str = INPUT_PATH) -> pd.DataFrame:
    """Load the V2 modeling dataset and add split helper columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"V2 modeling dataset not found: {path}")

    df = pd.read_csv(path)
    date_source = "prediction_date" if "prediction_date" in df.columns else "start_date_parsed"
    df["prediction_date_parsed_for_split"] = pd.to_datetime(
        df[date_source],
        errors="coerce",
        format="mixed",
    )

    for col in MANUAL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df


def select_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the manual-friendly feature set."""
    return df[MANUAL_FEATURES].copy()


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing shared by manual-friendly models."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, MANUAL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, MANUAL_CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def build_success_pipeline(calibrate: bool = True) -> Pipeline:
    """Build the manual success classifier pipeline."""
    base = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    if calibrate:
        classifier = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        classifier = base
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_duration_pipeline(model) -> Pipeline:
    """Build one manual duration regressor pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model),
        ]
    )


def random_split(df: pd.DataFrame, target_col: str, stratify: bool = False):
    """Create a random split with optional stratification."""
    y = df[target_col]
    stratify_values = y if stratify and y.value_counts().min() >= 2 else None
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_values,
    )


def time_split(df: pd.DataFrame):
    """Create a time-based train/test split using later trials as test rows."""
    dated = df[df["prediction_date_parsed_for_split"].notna()].copy()
    dated = dated.sort_values("prediction_date_parsed_for_split", kind="mergesort")
    split_idx = max(1, int(len(dated) * (1 - TEST_SIZE)))
    return dated.iloc[:split_idx].copy(), dated.iloc[split_idx:].copy()


def safe_auc(y_true, y_prob) -> Optional[float]:
    """Compute ROC-AUC only when both classes are present."""
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true, y_prob) -> Optional[float]:
    """Compute PR-AUC only when both classes are present."""
    if len(set(y_true)) < 2:
        return None
    return average_precision_score(y_true, y_prob)


def classification_metrics(y_true, y_pred, y_prob) -> Dict[str, Optional[float]]:
    """Compute success-classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": safe_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute duration-regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
    }


def train_success(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, object]]:
    """Train and evaluate the manual-friendly success classifier."""
    target = pd.to_numeric(df[SUCCESS_TARGET], errors="coerce")
    labeled = df[target.isin([0, 1])].copy()
    labeled[SUCCESS_TARGET] = target[target.isin([0, 1])].astype(int)

    random_train, random_test = random_split(labeled, SUCCESS_TARGET, stratify=True)
    time_train, time_test = time_split(labeled)

    # Time split is the operationally realistic model-selection split.
    time_model = build_success_pipeline(calibrate=True)
    time_model.fit(select_manual_features(time_train), time_train[SUCCESS_TARGET])
    time_prob = time_model.predict_proba(select_manual_features(time_test))[:, 1]
    time_pred = (time_prob >= 0.5).astype(int)

    random_model = build_success_pipeline(calibrate=True)
    random_model.fit(select_manual_features(random_train), random_train[SUCCESS_TARGET])
    random_prob = random_model.predict_proba(select_manual_features(random_test))[:, 1]
    random_pred = (random_prob >= 0.5).astype(int)

    final_model = build_success_pipeline(calibrate=True)
    final_model.fit(select_manual_features(labeled), labeled[SUCCESS_TARGET])
    joblib.dump(final_model, SUCCESS_MODEL_PATH)

    results = {
        "total_rows": len(df),
        "labeled_rows": len(labeled),
        "success_count": int((labeled[SUCCESS_TARGET] == 1).sum()),
        "failure_count": int((labeled[SUCCESS_TARGET] == 0).sum()),
        "unknown_rows_excluded": int(target.isna().sum()),
        "random_metrics": classification_metrics(random_test[SUCCESS_TARGET], random_pred, random_prob),
        "time_metrics": classification_metrics(time_test[SUCCESS_TARGET], time_pred, time_prob),
        "calibration": "CalibratedClassifierCV(sigmoid, cv=3) wrapped around LogisticRegression.",
    }
    return final_model, results


def train_duration(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, object]]:
    """Train and evaluate manual-friendly duration regressors."""
    duration = pd.to_numeric(df[DURATION_TARGET], errors="coerce")
    valid = df[duration.notna() & (duration > 0)].copy()
    valid[DURATION_TARGET] = duration[duration.notna() & (duration > 0)]

    random_train, random_test = random_split(valid, DURATION_TARGET)
    time_train, time_test = time_split(valid)

    models = {
        "Ridge": Ridge(random_state=RANDOM_SEED),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }

    rows = []
    best_name = None
    best_time_mae = float("inf")
    best_model = None

    for name, estimator in models.items():
        random_model = build_duration_pipeline(estimator)
        random_model.fit(select_manual_features(random_train), random_train[DURATION_TARGET])
        random_pred = random_model.predict(select_manual_features(random_test))

        time_model = build_duration_pipeline(estimator)
        time_model.fit(select_manual_features(time_train), time_train[DURATION_TARGET])
        time_pred = time_model.predict(select_manual_features(time_test))

        random_metrics = regression_metrics(random_test[DURATION_TARGET], random_pred)
        time_metrics = regression_metrics(time_test[DURATION_TARGET], time_pred)
        rows.append(
            {
                "model": name,
                "random_metrics": random_metrics,
                "time_metrics": time_metrics,
            }
        )

        if time_metrics["MAE"] < best_time_mae:
            best_time_mae = time_metrics["MAE"]
            best_name = name
            best_model = time_model

    if best_name is None:
        raise RuntimeError("No manual duration model was trained.")

    final_model = build_duration_pipeline(models[best_name])
    final_model.fit(select_manual_features(valid), valid[DURATION_TARGET])
    joblib.dump(final_model, DURATION_MODEL_PATH)

    results = {
        "valid_duration_rows": len(valid),
        "target_min": float(valid[DURATION_TARGET].min()),
        "target_median": float(valid[DURATION_TARGET].median()),
        "target_mean": float(valid[DURATION_TARGET].mean()),
        "target_max": float(valid[DURATION_TARGET].max()),
        "model_rows": rows,
        "best_model": best_name,
        "best_time_metrics": next(row["time_metrics"] for row in rows if row["model"] == best_name),
    }
    return final_model, results


def fmt_metric(value: Optional[float]) -> str:
    """Format nullable metric values."""
    return "N/A" if value is None else f"{value:.4f}"


def success_metrics_table(metrics: Dict[str, Optional[float]]) -> str:
    """Render one-row success metrics table."""
    return (
        "| Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"| {metrics['accuracy']:.4f} | {fmt_metric(metrics['roc_auc'])} | "
        f"{fmt_metric(metrics['pr_auc'])} | {metrics['precision']:.4f} | "
        f"{metrics['recall']:.4f} | {metrics['f1']:.4f} |"
    )


def duration_metrics_table(rows: List[Dict[str, object]], split_name: str) -> str:
    """Render duration model comparison table."""
    out = ["| Model | MAE | RMSE | R2 | MedianAE |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        metrics = row[f"{split_name}_metrics"]
        out.append(
            f"| {row['model']} | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} | "
            f"{metrics['R2']:.4f} | {metrics['MedianAE']:.2f} |"
        )
    return "\n".join(out)


def write_report(success_results: Dict[str, object], duration_results: Dict[str, object]) -> None:
    """Write the manual-friendly model results report."""
    report = f"""# V2 Manual-Friendly Model Results

> Generated by `python src/train_manual_models.py`

## Scope

These models are trained specifically for the Streamlit Manual Trial Query.
They use only features that a user can reasonably enter or that the app can
derive from entered sponsor/molecule fields. Existing Trial Lookup continues
to use the richer V2 models.

Manual Query uses a separate manual-friendly model and similar historical
trials for context. It is not clinical, regulatory, or investment advice.

## Manual Feature Set

Features used:

{chr(10).join(f"- `{feature}`" for feature in MANUAL_FEATURES)}

Excluded examples: `observed_duration_days`, `planned_duration_days`,
`overall_status`, progression evidence, success-label source columns, ChEMBL
`max_phase`, molecule IDs/names, sponsor names, and free-text protocol fields.

## Success Classifier

Model: calibrated LogisticRegression using `CalibratedClassifierCV(sigmoid, cv=3)`.

| Metric | Value |
|---|---:|
| Total rows | {success_results['total_rows']:,} |
| Labeled rows used | {success_results['labeled_rows']:,} |
| Success count | {success_results['success_count']:,} |
| Failure count | {success_results['failure_count']:,} |
| Unknown rows excluded | {success_results['unknown_rows_excluded']:,} |

### Random Split

{success_metrics_table(success_results['random_metrics'])}

### Time-Based Split

{success_metrics_table(success_results['time_metrics'])}

Time-based results are more realistic for future-use simulation. The success
label remains a conservative phase-progression proxy, not confirmed clinical
efficacy or regulatory approval.

## Duration Regressor

Best duration model by time-based MAE: **{duration_results['best_model']}**

| Metric | Value |
|---|---:|
| Rows with valid duration target | {duration_results['valid_duration_rows']:,} |
| Target min days | {duration_results['target_min']:.0f} |
| Target median days | {duration_results['target_median']:.0f} |
| Target mean days | {duration_results['target_mean']:.1f} |
| Target max days | {duration_results['target_max']:.0f} |

### Random Split

{duration_metrics_table(duration_results['model_rows'], 'random')}

### Time-Based Split

{duration_metrics_table(duration_results['model_rows'], 'time')}

Duration prediction is an estimate from public registry metadata, not a
guaranteed operational timeline.
"""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write(report)


def main() -> None:
    """Train manual-friendly V2 models and write a report."""
    os.makedirs(os.path.dirname(SUCCESS_MODEL_PATH), exist_ok=True)
    df = load_dataset()
    success_model, success_results = train_success(df)
    duration_model, duration_results = train_duration(df)
    write_report(success_results, duration_results)

    print("V2 manual-friendly model training complete.")
    print(f"Saved success model: {SUCCESS_MODEL_PATH}")
    print(f"Saved duration model: {DURATION_MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print("\nSuccess time-based metrics:")
    print(success_results["time_metrics"])
    print("\nDuration best model:")
    print(duration_results["best_model"])
    print(duration_results["best_time_metrics"])


if __name__ == "__main__":
    main()
