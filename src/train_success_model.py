"""
train_success_model.py - Version 2 success classification baseline.

Purpose
-------
Train baseline classifiers to predict `target_success`, a conservative
phase-progression proxy label. This is not confirmed clinical efficacy or
regulatory approval.

Inputs
------
- `data/processed/v2_modeling_dataset.csv`

Outputs
-------
- `models/v2_success_classifier.joblib`
- `reports/v2_success_results.md`
- `reports/figures/v2_success_confusion_matrix.png`
- `reports/figures/v2_success_roc_curve.png`
- `reports/figures/v2_success_pr_curve.png`

Leakage Rules
-------------
Progression evidence, success-label provenance, actual completion fields,
duration targets, `max_phase`, and last-update fields are excluded from model
inputs. Success-based sponsor-history columns are also excluded in this first
baseline because they are derived from the proxy success labels.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

from src.v2_build_modeling_dataset import LEAKAGE_COLUMNS, V2_MODEL_FEATURE_ALLOWLIST


INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "v2_modeling_dataset.csv")
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "v2_success_classifier.joblib")
REPORT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "reports", "v2_success_results.md")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
CONFUSION_MATRIX_PATH = os.path.join(FIGURES_DIR, "v2_success_confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(FIGURES_DIR, "v2_success_roc_curve.png")
PR_CURVE_PATH = os.path.join(FIGURES_DIR, "v2_success_pr_curve.png")

TARGET = "target_success"
RANDOM_SEED = 42
TEST_SIZE = 0.25

# These prior-success columns are label-derived. They may become usable after a
# stricter historical label-availability audit, but are excluded for baseline 1.
LABEL_DERIVED_SPONSOR_COLUMNS = {
    "sponsor_prior_successes",
    "sponsor_prior_failures",
    "sponsor_prior_success_rate",
}

SUCCESS_LEAKAGE_COLUMNS = set(LEAKAGE_COLUMNS).union(LABEL_DERIVED_SPONSOR_COLUMNS)

NUMERIC_FEATURES = [
    "prediction_year",
    "condition_count",
    "enrollment_count",
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
    "sponsor_prior_trials",
    "sponsor_prior_completed_trials",
    "sponsor_prior_failed_trials",
    "sponsor_prior_completion_rate",
    "sponsor_prior_phase_trials",
    "sponsor_prior_phase_completion_rate",
    "sponsor_prior_avg_duration_days",
    "sponsor_has_prior_history",
]

CATEGORICAL_FEATURES = [
    "conditions_normalized",
    "search_query_source",
    "molecule_name",
    "chembl_id",
    "preferred_name",
    "molecule_type",
    "drug_modality",
    "first_match_confidence",
    "lead_sponsor_normalized",
    "sponsor_class",
    "phase_normalized",
    "study_type",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
    "enrollment_type",
    "countries",
]

TEXT_FEATURE = "combined_success_text"


def load_labeled_dataset(path: str = INPUT_PATH) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load the final V2 modeling dataset and keep rows with 0/1 labels."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"V2 modeling dataset not found: {path}")

    df = pd.read_csv(path)
    target = pd.to_numeric(df[TARGET], errors="coerce")
    labeled = df[target.isin([0, 1])].copy()
    labeled[TARGET] = target[target.isin([0, 1])].astype(int)

    diagnostics = {
        "total_rows": len(df),
        "labeled_rows": len(labeled),
        "success_count": int((labeled[TARGET] == 1).sum()),
        "failure_count": int((labeled[TARGET] == 0).sum()),
        "unknown_rows_excluded": int(target.isna().sum()),
    }
    return labeled, diagnostics


def prepare_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived split/model helper columns."""
    df = df.copy()
    date_source = "prediction_date" if "prediction_date" in df.columns else "start_date_parsed"
    df["prediction_date_parsed_for_split"] = pd.to_datetime(
        df[date_source],
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


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Select non-leaky features from the final V2 feature allowlist."""
    allowed = [
        col
        for col in V2_MODEL_FEATURE_ALLOWLIST
        if col in df.columns and col not in SUCCESS_LEAKAGE_COLUMNS
    ]
    allowed = [col for col in allowed if col != "prediction_date"]
    if "prediction_year" not in allowed:
        allowed.append("prediction_year")
    if TEXT_FEATURE not in allowed:
        allowed.append(TEXT_FEATURE)

    numeric = [col for col in NUMERIC_FEATURES if col in allowed]
    categorical = [col for col in CATEGORICAL_FEATURES if col in allowed]
    feature_cols = numeric + categorical + [TEXT_FEATURE]

    leakage_overlap = set(feature_cols).intersection(SUCCESS_LEAKAGE_COLUMNS)
    if leakage_overlap:
        raise ValueError(f"Leakage columns selected as success features: {sorted(leakage_overlap)}")

    return df[feature_cols].copy(), numeric, categorical


def build_pipeline(model, numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Build a shared preprocessing plus classifier pipeline."""
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
            ("classifier", clone(model)),
        ]
    )


def random_split(df: pd.DataFrame):
    """Create a stratified random split when possible."""
    y = df[TARGET]
    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=stratify)


def time_based_split(df: pd.DataFrame):
    """Create a split where the test set contains later trials."""
    dated = df[df["prediction_date_parsed_for_split"].notna()].copy()
    dated = dated.sort_values("prediction_date_parsed_for_split", kind="mergesort")
    split_idx = max(1, int(len(dated) * (1 - TEST_SIZE)))
    return dated.iloc[:split_idx].copy(), dated.iloc[split_idx:].copy()


def _safe_auc(y_true, y_prob) -> Optional[float]:
    """Return ROC-AUC when both classes are present."""
    if len(set(y_true)) < 2:
        return None
    return roc_auc_score(y_true, y_prob)


def _safe_pr_auc(y_true, y_prob) -> Optional[float]:
    """Return PR-AUC when both classes are present."""
    if len(set(y_true)) < 2:
        return None
    return average_precision_score(y_true, y_prob)


def classification_metrics(y_true, y_pred, y_prob) -> Dict[str, object]:
    """Compute classification metrics with one-class split protection."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": _safe_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "test_class_count": dict(pd.Series(y_true).value_counts().sort_index()),
    }


def evaluate_model(
    name: str,
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Tuple[Pipeline, Dict[str, object], np.ndarray, np.ndarray]:
    """Fit and evaluate one model on one split."""
    X_train, _, _ = select_features(train_df)
    X_test, _, _ = select_features(test_df)
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    pipeline = build_pipeline(model, numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, y_pred, y_prob)
    metrics["model"] = name
    return pipeline, metrics, y_pred, y_prob


def feature_importance_table(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    """Extract top success-driving features when supported."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        importance_type = "coefficient"
    else:
        return pd.DataFrame(columns=["feature", "value", "importance_type"])

    values = np.asarray(values).ravel()
    order = np.argsort(np.abs(values))[::-1][:top_n]
    rows = []
    for idx in order:
        rows.append(
            {
                "feature": str(feature_names[idx])
                .replace("num__", "")
                .replace("cat__", "")
                .replace("text__", ""),
                "value": values[idx],
                "importance_type": importance_type,
            }
        )
    return pd.DataFrame(rows)


def _fmt_optional(value: Optional[float]) -> str:
    """Format nullable metric values."""
    return "N/A" if value is None else f"{value:.4f}"


def metrics_table(results: List[Dict[str, object]], split_name: str) -> str:
    """Render markdown metrics table for one split."""
    rows = [
        "| Model | Accuracy | ROC-AUC | PR-AUC | Precision | Recall | F1 | Confusion Matrix [[TN FP], [FN TP]] |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        metrics = result[f"{split_name}_metrics"]
        cm = metrics["confusion_matrix"].tolist()
        rows.append(
            f"| {result['model']} | {metrics['accuracy']:.4f} | {_fmt_optional(metrics['roc_auc'])} | "
            f"{_fmt_optional(metrics['pr_auc'])} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | `{cm}` |"
        )
    return "\n".join(rows)


def feature_importance_markdown(df: pd.DataFrame) -> str:
    """Render feature importances without optional dependencies."""
    if df.empty:
        return "Feature importance is not available for the selected model."
    rows = ["| Feature | Value | Importance Type |", "|---|---:|---|"]
    for row in df.itertuples(index=False):
        rows.append(f"| {row.feature} | {row.value:.6f} | {row.importance_type} |")
    return "\n".join(rows)


def plot_best_model(y_true, y_pred, y_prob, model_name: str) -> None:
    """Save confusion matrix, ROC, and PR plots where possible."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Failure", "Success"])
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"V2 Success Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if len(set(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_prob):.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"V2 Success ROC Curve - {model_name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROC_CURVE_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, y_prob):.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"V2 Success Precision-Recall Curve - {model_name}")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PR_CURVE_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)


def phase_counts(df: pd.DataFrame) -> str:
    """Render phase-specific label counts."""
    table = (
        df.groupby(["phase_normalized", TARGET], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["phase_normalized", TARGET])
    )
    rows = ["| Phase | Target Success | Count |", "|---|---:|---:|"]
    for row in table.itertuples(index=False):
        rows.append(f"| {row.phase_normalized} | {int(row.target_success)} | {row.count} |")
    return "\n".join(rows)


def write_report(
    diagnostics: Dict[str, int],
    labeled_df: pd.DataFrame,
    results: List[Dict[str, object]],
    best_name: str,
    best_time_metrics: Dict[str, object],
    importances: pd.DataFrame,
) -> None:
    """Write V2 success baseline report."""
    report = f"""# V2 Success Classification Baseline

> Generated by `python src/train_success_model.py`

## Scope

This is a first baseline for predicting `target_success`, a conservative
phase-progression proxy label. It does **not** represent confirmed clinical
efficacy, endpoint success, or regulatory approval.

## Label Definition Summary

- Phase 1 success: same normalized drug/intervention appears later in Phase 2 or higher.
- Phase 2 success: same normalized drug/intervention appears later in Phase 3 or higher.
- Phase 3 success: later Phase 4 evidence only for this baseline; approval labeling is deferred.
- Failed status with no later progression can be labeled as likely failure.
- Ongoing/recruiting/uncertain trials usually remain unknown.

## Dataset

| Metric | Value |
|---|---:|
| Total rows | {diagnostics['total_rows']:,} |
| Labeled rows used | {diagnostics['labeled_rows']:,} |
| Success count | {diagnostics['success_count']:,} |
| Failure count | {diagnostics['failure_count']:,} |
| Unknown rows excluded | {diagnostics['unknown_rows_excluded']:,} |

## Phase-Specific Counts

{phase_counts(labeled_df)}

## Random Split Results

{metrics_table(results, 'random')}

## Time-Based Split Results

{metrics_table(results, 'time')}

Time-based performance is more realistic because it tests on later trials.
With only {diagnostics['labeled_rows']:,} labeled rows, these metrics should be
treated as early baseline signals only.

## Best Baseline

Best model selected by time-based ROC-AUC when available, otherwise time-based F1:
**{best_name}**

| Metric | Value |
|---|---:|
| Time Accuracy | {best_time_metrics['accuracy']:.4f} |
| Time ROC-AUC | {_fmt_optional(best_time_metrics['roc_auc'])} |
| Time PR-AUC | {_fmt_optional(best_time_metrics['pr_auc'])} |
| Time Precision | {best_time_metrics['precision']:.4f} |
| Time Recall | {best_time_metrics['recall']:.4f} |
| Time F1 | {best_time_metrics['f1']:.4f} |

## Top Success-Driving Features

{feature_importance_markdown(importances)}

## Plots

- `reports/figures/v2_success_confusion_matrix.png`
- `reports/figures/v2_success_roc_curve.png`
- `reports/figures/v2_success_pr_curve.png`

## Leakage Warning

Progression evidence is used only to create labels. `progression_nct_id`,
`progression_phase`, label-source columns, actual completion fields, duration
targets, `overall_status`, and ChEMBL `max_phase` are excluded from model
inputs.
"""
    os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)


def _selection_score(metrics: Dict[str, object]) -> float:
    """Select by ROC-AUC if available, otherwise F1."""
    return metrics["roc_auc"] if metrics["roc_auc"] is not None else metrics["f1"]


def train_success_baselines() -> None:
    """Train and evaluate V2 success classification baselines."""
    labeled_df, diagnostics = load_labeled_dataset()
    labeled_df = prepare_modeling_columns(labeled_df)
    _, numeric_features, categorical_features = select_features(labeled_df)

    random_train, random_test = random_split(labeled_df)
    time_train, time_test = time_based_split(labeled_df)

    candidates = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_SEED,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_SEED,
        ),
    }

    print("Training V2 success classification baselines")
    print(f"Total rows: {diagnostics['total_rows']:,}")
    print(f"Labeled rows used: {diagnostics['labeled_rows']:,}")
    print(f"Success count: {diagnostics['success_count']:,}")
    print(f"Failure count: {diagnostics['failure_count']:,}")
    print(f"Unknown rows excluded: {diagnostics['unknown_rows_excluded']:,}")
    print(f"Random split train/test: {len(random_train):,}/{len(random_test):,}")
    print(f"Time split train/test: {len(time_train):,}/{len(time_test):,}")

    results: List[Dict[str, object]] = []
    best_name = None
    best_score = -1.0
    best_time_pipeline = None
    best_time_y = None
    best_time_pred = None
    best_time_prob = None

    for name, model in candidates.items():
        print(f"\nTraining {name}")
        _, random_metrics, _, _ = evaluate_model(
            name,
            model,
            random_train,
            random_test,
            numeric_features,
            categorical_features,
        )
        time_pipeline, time_metrics, time_pred, time_prob = evaluate_model(
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
            f"  Random Acc/ROC/PR/F1: {random_metrics['accuracy']:.4f} / "
            f"{_fmt_optional(random_metrics['roc_auc'])} / "
            f"{_fmt_optional(random_metrics['pr_auc'])} / {random_metrics['f1']:.4f}"
        )
        print(
            f"  Time Acc/ROC/PR/F1:   {time_metrics['accuracy']:.4f} / "
            f"{_fmt_optional(time_metrics['roc_auc'])} / "
            f"{_fmt_optional(time_metrics['pr_auc'])} / {time_metrics['f1']:.4f}"
        )

        score = _selection_score(time_metrics)
        if score > best_score:
            best_score = score
            best_name = name
            best_time_pipeline = time_pipeline
            best_time_y = time_test[TARGET].to_numpy()
            best_time_pred = time_pred
            best_time_prob = time_prob

    if best_time_pipeline is None:
        raise RuntimeError("No success model trained successfully.")

    best_model_template = candidates[best_name]
    X_all, _, _ = select_features(labeled_df)
    final_pipeline = build_pipeline(best_model_template, numeric_features, categorical_features)
    final_pipeline.fit(X_all, labeled_df[TARGET])

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(final_pipeline, MODEL_OUTPUT_PATH)

    plot_best_model(best_time_y, best_time_pred, best_time_prob, best_name)
    importances = feature_importance_table(best_time_pipeline)
    best_time_metrics = next(r["time_metrics"] for r in results if r["model"] == best_name)
    write_report(
        diagnostics=diagnostics,
        labeled_df=labeled_df,
        results=results,
        best_name=best_name,
        best_time_metrics=best_time_metrics,
        importances=importances,
    )

    print("\nV2 success baseline complete")
    print(f"Best model: {best_name}")
    print(
        f"Best time metrics Acc/ROC/PR/Precision/Recall/F1: "
        f"{best_time_metrics['accuracy']:.4f} / {_fmt_optional(best_time_metrics['roc_auc'])} / "
        f"{_fmt_optional(best_time_metrics['pr_auc'])} / {best_time_metrics['precision']:.4f} / "
        f"{best_time_metrics['recall']:.4f} / {best_time_metrics['f1']:.4f}"
    )
    print(f"Saved model: {MODEL_OUTPUT_PATH}")
    print(f"Saved report: {REPORT_OUTPUT_PATH}")
    print(f"Saved plot: {CONFUSION_MATRIX_PATH}")
    print(f"Saved plot: {ROC_CURVE_PATH}")
    print(f"Saved plot: {PR_CURVE_PATH}")


def main() -> None:
    """CLI entry point."""
    train_success_baselines()


if __name__ == "__main__":
    main()

