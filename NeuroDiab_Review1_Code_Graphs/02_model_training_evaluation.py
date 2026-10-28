#!/usr/bin/env python3
"""NeuroDiab Review-II: train and compare ANN + five baseline classifiers.

Uses the UCI Early Stage Diabetes Risk Prediction CSV. Review-I files are not
modified. Preprocessing is fitted inside each model pipeline to avoid leakage.
Exact duplicate records are assigned to the same train/test partition.
Dataset: https://doi.org/10.24432/C5VG8H

Run:
  python -m pip install numpy pandas matplotlib scikit-learn joblib
  python 02_model_training_evaluation.py --csv data/diabetes_data_upload.csv
"""
import importlib.util
from pathlib import Path
import argparse
import json
import warnings

missing_packages = [
    package for package, module in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
    )
    if importlib.util.find_spec(module) is None
]
if missing_packages:
    raise SystemExit(
        "Missing required packages: "
        + ", ".join(missing_packages)
        + ". Install them with: python -m pip install numpy pandas matplotlib scikit-learn joblib"
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import joblib

warnings.filterwarnings("ignore", category=UserWarning)


def normalize_target(series):
    """Convert UCI Positive/Negative target labels to 1/0."""
    values = series.astype(str).str.strip().str.lower()
    mapping = {"positive": 1, "negative": 0, "1": 1, "0": 0}
    result = values.map(mapping)
    if result.isna().any() or result.nunique() != 2:
        raise ValueError(
            "Target must contain two classes labelled Positive/Negative (or 1/0). "
            f"Found: {sorted(values.unique().tolist())}"
        )
    return result.astype(int)


def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ], remainder="drop")


def build_models(preprocessor, seed=42):
    models = {
        "ANN": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.001, max_iter=1000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=seed
        ),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    return {name: Pipeline([("preprocess", clone(preprocessor)), ("model", model)]) for name, model in models.items()}


def positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    return model.predict(X).astype(float)


def compute_metrics(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc_average_precision": average_precision_score(y_true, scores),
        "brier_score": brier_score_loss(y_true, scores),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }
    return result


def save_comparison_graphs(results, y_test, predictions, score_map, output_dir):
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(results).set_index("model")

    for metric, title, filename in [
        ("accuracy", "Accuracy by Model", "01_accuracy_comparison.png"),
        ("precision", "Precision by Model", "02_precision_comparison.png"),
        ("recall_sensitivity", "Recall / Sensitivity by Model", "03_recall_comparison.png"),
        ("f1_score", "F1-score by Model", "04_f1_comparison.png"),
    ]:
        ax = metrics_df[metric].sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))
        ax.set_title(title); ax.set_ylabel(metric.replace("_", " ").title()); ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=0.25)
        plt.tight_layout(); plt.savefig(graph_dir / filename, dpi=180); plt.close()

    # Confusion matrix for every model
    for name, pred in predictions.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1],
            display_labels=["Negative", "Positive"], cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{name} — Confusion Matrix")
        plt.tight_layout(); plt.savefig(graph_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=180); plt.close()

    # ROC and precision-recall curves
    plt.figure(figsize=(8, 6))
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison"); plt.legend(loc="lower right", fontsize=8); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(graph_dir / "roc_curve_comparison.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 6))
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve Comparison")
    plt.legend(loc="best", fontsize=8); plt.grid(alpha=0.25)
    plt.tight_layout(); plt.savefig(graph_dir / "precision_recall_comparison.png", dpi=180); plt.close()

    # ANN training loss
    ann = None
    # Stored by caller through attribute below, if ANN exists
    return graph_dir


def main():
    parser = argparse.ArgumentParser(
        description=(
            "NeuroDiab Review-II model training/evaluation "
            "(UCI dataset: https://doi.org/10.24432/C5VG8H)"
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to diabetes_data_upload.csv from the UCI dataset",
    )
    parser.add_argument("--out", default="review2_results", help="Folder for Review-II outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", action="store_true", help="Also run 5-fold grouped cross-validation (takes longer)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    model_dir, report_dir = out_dir / "models", out_dir / "reports"
    for folder in (out_dir, model_dir, report_dir, out_dir / "graphs"):
        folder.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    target_matches = [c for c in df.columns if c.lower().strip() in ("class", "target", "outcome")]
    if not target_matches:
        raise ValueError(f"Could not find target column named class/target/outcome. Columns: {df.columns.tolist()}")
    target_col = target_matches[0]
    y = normalize_target(df[target_col])
    X = df.drop(columns=[target_col]).copy()
    if X.empty:
        raise ValueError("No feature columns found after removing target.")

    # Identical full records (including target) must remain within one partition.
    # This reduces leakage from repeated questionnaire rows without deleting data.
    groups = pd.util.hash_pandas_object(df.astype(str), index=False).astype("uint64").to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Guard against a split that accidentally omits a class.
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Grouped split did not include both classes. Check the dataset and duplicate groups.")

    preprocessor = make_preprocessor(X_train)
    models = build_models(preprocessor, args.seed)
    results, predictions, score_map = [], {}, {}
    fitted_models = {}

    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test).astype(int)
        scores = positive_scores(pipeline, X_test)
        metrics = compute_metrics(y_test, pred, scores)
        metrics["model"] = name
        results.append(metrics)
        predictions[name] = pred
        score_map[name] = scores
        fitted_models[name] = pipeline
        joblib.dump(pipeline, model_dir / f"{name.lower().replace(' ', '_')}_pipeline.joblib")
        with open(report_dir / f"classification_report_{name.lower().replace(' ', '_')}.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, pred, target_names=["Negative", "Positive"], zero_division=0))

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(report_dir / "model_comparison_metrics.csv", index=False)
    with open(report_dir / "model_comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)
    pd.DataFrame({"actual": y_test.to_numpy(), **{f"pred_{k}": v for k, v in predictions.items()},
                  **{f"score_{k}": v for k, v in score_map.items()}}).to_csv(report_dir / "test_predictions.csv", index=False)

    graph_dir = save_comparison_graphs(results, y_test, predictions, score_map, out_dir)

    # ANN loss curve
    ann_model = fitted_models["ANN"].named_steps["model"]
    if hasattr(ann_model, "loss_curve_") and len(ann_model.loss_curve_):
        plt.figure(figsize=(8, 5)); plt.plot(range(1, len(ann_model.loss_curve_) + 1), ann_model.loss_curve_)
        plt.xlabel("Training iteration"); plt.ylabel("Loss"); plt.title("ANN Training Loss Curve")
        plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(graph_dir / "ann_training_loss.png", dpi=180); plt.close()

    # Permutation importance for ANN on held-out data (works on raw columns/pipeline).
    print("Calculating ANN permutation importance...")
    pi = permutation_importance(fitted_models["ANN"], X_test, y_test, n_repeats=10,
                                random_state=args.seed, scoring="f1", n_jobs=-1)
    importance_df = pd.DataFrame({"feature": X.columns, "importance_mean": pi.importances_mean,
                                  "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(report_dir / "ann_permutation_feature_importance.csv", index=False)
    top = importance_df.head(15).sort_values("importance_mean")
    plt.figure(figsize=(9, 6)); plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Mean decrease in F1 after permutation"); plt.title("ANN Permutation Feature Importance")
    plt.tight_layout(); plt.savefig(graph_dir / "ann_permutation_feature_importance.png", dpi=180); plt.close()

    if args.cv:
        print("Running 5-fold grouped cross-validation for each model...")
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        cv_rows = []
        scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
        for name, pipeline in models.items():
            scores = cross_validate(pipeline, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
            row = {"model": name}
            for metric in scoring:
                vals = scores[f"test_{metric}"]
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1))
            cv_rows.append(row)
        pd.DataFrame(cv_rows).to_csv(report_dir / "grouped_5fold_cv_metrics.csv", index=False)

    summary = {
        "dataset_file": csv_path.name,
        "rows": int(df.shape[0]),
        "columns_including_target": int(df.shape[1]),
        "target_column": target_col,
        "feature_count": int(X.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "unique_full_record_groups": int(pd.Series(groups).nunique()),
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "positive_train": int(y_train.sum()), "negative_train": int((y_train == 0).sum()),
        "positive_test": int(y_test.sum()), "negative_test": int((y_test == 0).sum()),
        "best_by_roc_auc_on_this_holdout": str(results_df.iloc[0]["model"]),
        "warning": "Preliminary internal hold-out results only; not external clinical validation or a diagnostic guarantee."
    }
    with open(report_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Test-set model comparison ===")
    display_cols = ["model", "accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc", "pr_auc_average_precision", "brier_score"]
    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved models: {model_dir.resolve()}")
    print(f"Saved reports: {report_dir.resolve()}")
    print(f"Saved graphs:  {graph_dir.resolve()}")
    print("Reminder: use the questionnaire-based UCI dataset responsibly; outputs are not a medical diagnosis.")

if __name__ == "__main__":
    main()
