"""
train_duration_model.py - Version 2 duration regression baseline.

Purpose
-------
Train baseline models to predict `observed_duration_days`, the number of days
between ClinicalTrials.gov `start_date` and `completion_date`.

This is a baseline only. Duration prediction is an estimate from historical
registry metadata, not a guaranteed operational timeline.

Inputs
------
- `data/processed/v2_success_duration_dataset.csv`

Outputs
-------
- `models/v2_duration_regressor.joblib`
- `reports/v2_duration_results.md`
- `reports/figures/v2_duration_actual_vs_predicted.png`
- `reports/figures/v2_duration_residuals.png`

Leakage Rules
-------------
`completion_date`, `observed_duration_days`, `overall_status`, success labels,
progression evidence, and last-update fields are excluded from model inputs.
`planned_duration_days` is also excluded from this first baseline because it may
be leakage-prone if registry dates were revised after trial start.
"""

import os
import sys
from math import sqrt
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
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

from src.v2_preprocess import V2_FEATURE_ALLOWLIST, V2_PROCESSED_CSV_PATH


MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "v2_duration_regressor.joblib")
REPORT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "reports", "v2_duration_results.md")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
ACTUAL_VS_PREDICTED_PATH = os.path.join(
    FIGURES_DIR,
    "v2_duration_actual_vs_predicted.png",
)
RESIDUALS_PATH = os.path.join(FIGURES_DIR, "v2_duration_residuals.png")

TARGET = "observed_duration_days"
RANDOM_SEED = 42
TEST_SIZE = 0.2

LEAKAGE_COLUMNS = {
    "observed_duration_days",
    "planned_duration_days",
    "completion_date",
    "completion_date_parsed",
    "primary_completion_date",
    "primary_completion_date_parsed",
    "overall_status",
    "target_success",
    "success_label_source",
    "success_label_notes",
    "progression_nct_id",
    "progression_phase",
    "progression_start_date",
    "last_update_submit_date",
    "last_update_submit_date_parsed",
}

NUMERIC_FEATURES = [
    "prediction_year",
    "enrollment_count",
    "condition_count",
    "drug_intervention_count",
    "collaborator_count",
    "location_count",
    "country_count",
    "endpoint_safety",
    "endpoint_efficacy",
    "endpoint_survival",
    "endpoint_biomarker",
    "endpoint_response_rate",
    "endpoint_type_count",
    "has_primary_endpoint",
    "primary_endpoint_text_length",
]

CATEGORICAL_FEATURES = [
    "phase_normalized",
    "study_type",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
    "enrollment_type",
    "sponsor_class",
    "lead_sponsor_normalized",
    "conditions_normalized",
    "search_query_source",
]

TEXT_FEATURE = "combined_protocol_text"


def load_duration_dataset(path: str = V2_PROCESSED_CSV_PATH) -> pd.DataFrame:
    """Load and filter the V2 dataset to valid observed duration targets."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed V2 dataset not found: {path}")

    df = pd.read_csv(path)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()

    df["prediction_date_parsed_for_split"] = pd.to_datetime(
        df.get("prediction_date", df.get("start_date_parsed")),
        errors="coerce",
        format="mixed",
    )
    df["prediction_year"] = df["prediction_date_parsed_for_split"].dt.year

    text_cols = [
        "brief_summary",
        "eligibility_criteria",
        "primary_outcome_measures",
        "primary_outcome_timeframes",
        "secondary_outcome_measures",
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
    df[TEXT_FEATURE] = (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    """
    Select non-leaky baseline features.

    The V2 preprocessing allowlist is used as the source of candidate features,
    then leakage columns and planned duration are removed.
    """
    candidate_features = [
        feature
        for feature in V2_FEATURE_ALLOWLIST
        if feature not in LEAKAGE_COLUMNS and feature in df.columns
    ]

    # Use a derived year instead of the raw date string in model inputs.
    candidate_features = [
        feature for feature in candidate_features if feature != "prediction_date"
    ]
    if "prediction_year" not in candidate_features:
        candidate_features.append("prediction_year")
    if TEXT_FEATURE not in candidate_features:
        candidate_features.append(TEXT_FEATURE)

    numeric_features = [f for f in NUMERIC_FEATURES if f in candidate_features]
    categorical_features = [f for f in CATEGORICAL_FEATURES if f in candidate_features]

    feature_cols = numeric_features + categorical_features + [TEXT_FEATURE]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected duration feature columns: {missing}")

    leakage_overlap = set(feature_cols).intersection(LEAKAGE_COLUMNS)
    if leakage_overlap:
        raise ValueError(f"Leakage columns selected as features: {sorted(leakage_overlap)}")

    return df[feature_cols].copy(), numeric_features, categorical_features, TEXT_FEATURE


def build_pipeline(model, numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Build a shared preprocessing plus regression pipeline."""
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
    text_transformer = TfidfVectorizer(
        max_features=250,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("text", text_transformer, TEXT_FEATURE),
        ],
        sparse_threshold=0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", clone(model)),
        ]
    )


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute duration regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MedianAE": median_absolute_error(y_true, y_pred),
    }


def random_split(df: pd.DataFrame):
    """Create a random train/test split."""
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)


def time_based_split(df: pd.DataFrame):
    """Create a train/test split where the test set is the latest trials."""
    dated = df[df["prediction_date_parsed_for_split"].notna()].copy()
    dated = dated.sort_values("prediction_date_parsed_for_split", kind="mergesort")
    split_idx = max(1, int(len(dated) * (1 - TEST_SIZE)))
    train_df = dated.iloc[:split_idx].copy()
    test_df = dated.iloc[split_idx:].copy()
    return train_df, test_df


def evaluate_model(
    name: str,
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[Pipeline, Dict[str, float], np.ndarray]:
    """Fit and evaluate one model on one split."""
    X_train, _, _, _ = select_features(train_df)
    X_test, _, _, _ = select_features(test_df)
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    pipeline = build_pipeline(model, numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)
    metrics["model"] = name
    return pipeline, metrics, y_pred


def feature_importance_table(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Extract top feature drivers when supported by the final regressor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(regressor, "feature_importances_"):
        values = regressor.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(regressor, "coef_"):
        values = regressor.coef_
        importance_type = "coefficient"
    else:
        return pd.DataFrame(columns=["feature", "value", "importance_type"])

    values = np.asarray(values).ravel()
    order = np.argsort(np.abs(values))[::-1][:top_n]
    rows = []
    for idx in order:
        rows.append(
            {
                "feature": str(feature_names[idx]).replace("num__", "").replace("cat__", "").replace("text__", ""),
                "value": values[idx],
                "importance_type": importance_type,
            }
        )
    return pd.DataFrame(rows)


def plot_duration_predictions(y_true, y_pred, model_name: str) -> None:
    """Save actual-vs-predicted and residual plots for the best time-split model."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolor="none")
    min_value = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_value, max_value], [min_value, max_value], color="#ef4444", linestyle="--")
    ax.set_title(f"V2 Duration Actual vs Predicted - {model_name}")
    ax.set_xlabel("Actual duration days")
    ax.set_ylabel("Predicted duration days")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ACTUAL_VS_PREDICTED_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, residuals, alpha=0.65, edgecolor="none")
    ax.axhline(0, color="#ef4444", linestyle="--")
    ax.set_title(f"V2 Duration Residuals - {model_name}")
    ax.set_xlabel("Predicted duration days")
    ax.set_ylabel("Residual days")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESIDUALS_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metrics_table(results: List[Dict[str, object]], split_name: str) -> str:
    """Render a markdown metrics table for one split."""
    rows = [
        "| Model | MAE | RMSE | R2 | Median Absolute Error |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result[f"{split_name}_metrics"]
        rows.append(
            f"| {result['model']} | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} | "
            f"{metrics['R2']:.4f} | {metrics['MedianAE']:.2f} |"
        )
    return "\n".join(rows)


def feature_importance_markdown(feature_importances: pd.DataFrame) -> str:
    """Render feature importances without requiring optional tabulate."""
    if feature_importances.empty:
        return "Feature importance is not available for the selected model."

    rows = [
        "| Feature | Value | Importance Type |",
        "|---|---:|---|",
    ]
    for row in feature_importances.itertuples(index=False):
        rows.append(f"| {row.feature} | {row.value:.6f} | {row.importance_type} |")
    return "\n".join(rows)


def write_report(
    df: pd.DataFrame,
    results: List[Dict[str, object]],
    best_name: str,
    best_time_metrics: Dict[str, float],
    feature_importances: pd.DataFrame,
) -> None:
    """Write the V2 duration baseline report."""
    target_summary = df[TARGET].describe(percentiles=[0.25, 0.5, 0.75])
    importance_md = feature_importance_markdown(feature_importances)

    report = f"""# V2 Duration Regression Baseline

> Generated by `python src/train_duration_model.py`

## Scope

This is a first baseline for predicting `observed_duration_days` from public
ClinicalTrials.gov metadata. It is an estimate from historical registry data,
not a guaranteed operational timeline.

## Leakage Warning

The duration target is created from `completion_date - start_date`.
`completion_date`, `observed_duration_days`, `overall_status`, success labels,
progression evidence, and last-update fields are excluded from model inputs.
`planned_duration_days` is also excluded from this first baseline.

## Dataset

| Metric | Value |
|---|---:|
| Rows with valid target | {len(df):,} |
| Target min days | {target_summary['min']:.0f} |
| Target median days | {target_summary['50%']:.0f} |
| Target mean days | {target_summary['mean']:.1f} |
| Target max days | {target_summary['max']:.0f} |

## Random Split Results

{metrics_table(results, 'random')}

## Time-Based Split Results

{metrics_table(results, 'time')}

Time-based performance is more realistic because it tests on later trials,
closer to the way a future prediction tool would be used.

## Best Baseline

Best model selected by lowest time-based MAE: **{best_name}**

| Metric | Value |
|---|---:|
| Time MAE | {best_time_metrics['MAE']:.2f} |
| Time RMSE | {best_time_metrics['RMSE']:.2f} |
| Time R2 | {best_time_metrics['R2']:.4f} |
| Time Median Absolute Error | {best_time_metrics['MedianAE']:.2f} |

## Top Duration-Driving Features

{importance_md}

## Plots

- `reports/figures/v2_duration_actual_vs_predicted.png`
- `reports/figures/v2_duration_residuals.png`
"""

    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)


def train_duration_baselines() -> None:
    """Train and evaluate V2 duration baseline regressors."""
    df = load_duration_dataset()
    X_all, numeric_features, categorical_features, _ = select_features(df)
    del X_all

    random_train, random_test = random_split(df)
    time_train, time_test = time_based_split(df)

    candidates = {
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_SEED),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=RANDOM_SEED, max_iter=10000),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=RANDOM_SEED,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_SEED,
        ),
    }

    results: List[Dict[str, object]] = []
    best_name = None
    best_time_mae = float("inf")
    best_time_pipeline = None
    best_time_predictions = None
    best_time_test = None

    print("Training V2 duration regression baselines")
    print(f"Rows with valid observed duration: {len(df):,}")
    print(f"Random split train/test: {len(random_train):,}/{len(random_test):,}")
    print(f"Time split train/test: {len(time_train):,}/{len(time_test):,}")

    for name, model in candidates.items():
        print(f"\nTraining {name}")
        _, random_metrics, _ = evaluate_model(
            name,
            model,
            random_train,
            random_test,
            numeric_features,
            categorical_features,
        )
        time_pipeline, time_metrics, time_pred = evaluate_model(
            name,
            model,
            time_train,
            time_test,
            numeric_features,
            categorical_features,
        )
        results.append(
            {
                "model": name,
                "random_metrics": random_metrics,
                "time_metrics": time_metrics,
            }
        )
        print(
            f"  Random MAE/RMSE/R2/MedAE: {random_metrics['MAE']:.2f} / "
            f"{random_metrics['RMSE']:.2f} / {random_metrics['R2']:.4f} / "
            f"{random_metrics['MedianAE']:.2f}"
        )
        print(
            f"  Time MAE/RMSE/R2/MedAE:   {time_metrics['MAE']:.2f} / "
            f"{time_metrics['RMSE']:.2f} / {time_metrics['R2']:.4f} / "
            f"{time_metrics['MedianAE']:.2f}"
        )

        if time_metrics["MAE"] < best_time_mae:
            best_time_mae = time_metrics["MAE"]
            best_name = name
            best_time_pipeline = time_pipeline
            best_time_predictions = time_pred
            best_time_test = time_test

    if best_time_pipeline is None or best_time_test is None or best_time_predictions is None:
        raise RuntimeError("No duration model trained successfully.")

    # Refit the selected model family on all valid rows before saving.
    best_model = candidates[best_name]
    X_all, _, _, _ = select_features(df)
    final_pipeline = build_pipeline(best_model, numeric_features, categorical_features)
    final_pipeline.fit(X_all, df[TARGET])

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(final_pipeline, MODEL_OUTPUT_PATH)

    plot_duration_predictions(
        y_true=best_time_test[TARGET].to_numpy(),
        y_pred=best_time_predictions,
        model_name=best_name,
    )
    importances = feature_importance_table(best_time_pipeline)
    best_metrics = next(r["time_metrics"] for r in results if r["model"] == best_name)
    write_report(
        df=df,
        results=results,
        best_name=best_name,
        best_time_metrics=best_metrics,
        feature_importances=importances,
    )

    print("\nV2 duration baseline complete")
    print(f"Best model by time-based MAE: {best_name}")
    print(
        f"Best time metrics MAE/RMSE/R2/MedAE: {best_metrics['MAE']:.2f} / "
        f"{best_metrics['RMSE']:.2f} / {best_metrics['R2']:.4f} / "
        f"{best_metrics['MedianAE']:.2f}"
    )
    print(f"Saved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved report: {REPORT_OUTPUT_PATH}")
    print(f"Saved plot: {ACTUAL_VS_PREDICTED_PATH}")
    print(f"Saved plot: {RESIDUALS_PATH}")


def main() -> None:
    """CLI entry point."""
    train_duration_baselines()


if __name__ == "__main__":
    main()
