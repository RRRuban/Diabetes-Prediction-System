"""NeuroDiab Review-II Step 1: train and evaluate an initial ANN.

Uses the same UCI Early Stage Diabetes Risk Prediction CSV as Review-I.
Review-I files/results are not modified. Exact duplicate records are grouped
so identical rows cannot cross the training/held-out split.
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    average_precision_score, brier_score_loss
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "diabetes_data_upload.csv"
GRAPHS = ROOT / "graphs"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for folder in (GRAPHS, REPORTS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5

# Load and validate the same Review-I dataset.
df = pd.read_csv(DATA)
df.columns = [str(c).strip() for c in df.columns]
if df.shape != (520, 17):
    raise ValueError(f"Expected Review-I dataset shape (520, 17), got {df.shape}")
if "class" not in df.columns:
    raise ValueError("Expected target column 'class' was not found.")
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()

yraw = df["class"].astype(str).str.lower().str.strip()
y = yraw.map({"positive": 1, "negative": 0, "1": 1, "0": 0})
if y.isna().any() or y.nunique() != 2:
    raise ValueError(f"Unexpected target values: {sorted(yraw.unique())}")
X = df.drop(columns=["class"])

# Group identical full records, including target, to prevent exact duplicates
# appearing on both sides of any split. This does not delete source records.
groups = pd.factorize(pd.util.hash_pandas_object(df, index=False))[0]
num_duplicates = int(df.duplicated().sum())

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

# The preprocessing operations are inside the model Pipeline so each CV fold
# learns its own imputation/scaling/encoding statistics from that fold's train.
ann = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
        alpha=0.001, learning_rate_init=0.001, max_iter=1000,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=25, random_state=SEED,
    )),
])

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    cv_results = cross_validate(
        ann, X, y, cv=cv, groups=groups, scoring=scoring,
        return_train_score=False, n_jobs=1, error_score="raise"
    )
    # Use the first deterministic group-stratified fold as a held-out example.
    train_idx, test_idx = next(cv.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    ann.fit(X_train, y_train)
    y_pred = ann.predict(X_test)
    y_prob = ann.predict_proba(X_test)[:, 1]
    warning_text = [str(w.message) for w in caught]

metrics = {
    "dataset": "UCI Early Stage Diabetes Risk Prediction Dataset (Review-I CSV)",
    "dataset_rows": int(len(df)),
    "dataset_columns_including_target": int(df.shape[1]),
    "input_features": int(X.shape[1]),
    "positive_records": int((y == 1).sum()),
    "negative_records": int((y == 0).sum()),
    "exact_duplicate_rows_reported": num_duplicates,
    "unique_full_record_groups": int(pd.Series(groups).nunique()),
    "split_method": "StratifiedGroupKFold; exact duplicate full records grouped",
    "seed": SEED,
    "ann_hidden_layer_sizes": [32, 16],
    "ann_activation": "relu",
    "ann_solver": "adam",
    "ann_alpha": 0.001,
    "ann_learning_rate_init": 0.001,
    "ann_max_iter": 1000,
    "ann_early_stopping": True,
    "heldout_train_rows": int(len(train_idx)),
    "heldout_test_rows": int(len(test_idx)),
    "heldout_train_positive": int((y_train == 1).sum()),
    "heldout_train_negative": int((y_train == 0).sum()),
    "heldout_test_positive": int((y_test == 1).sum()),
    "heldout_test_negative": int((y_test == 0).sum()),
    "heldout_accuracy": float(accuracy_score(y_test, y_pred)),
    "heldout_precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "heldout_recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "heldout_specificity": float((confusion_matrix(y_test, y_pred, labels=[0,1])[0,0] / max(1, confusion_matrix(y_test, y_pred, labels=[0,1])[0].sum()))),
    "heldout_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "heldout_roc_auc": float(roc_auc_score(y_test, y_prob)),
    "heldout_pr_auc_average_precision": float(average_precision_score(y_test, y_prob)),
    "heldout_brier_score": float(brier_score_loss(y_test, y_prob)),
    "ann_epochs_run_on_heldout_fit": int(ann.named_steps["classifier"].n_iter_),
    "cv_mean_std": {},
    "warnings": warning_text,
}
for metric in scoring:
    vals = cv_results[f"test_{metric}"]
    metrics["cv_mean_std"][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "fold_scores": [float(v) for v in vals]}

# Persist raw fold scores and final held-out evaluation.
cv_rows = []
for i in range(N_SPLITS):
    cv_rows.append({"fold": i+1, **{m: float(cv_results[f"test_{m}"][i]) for m in scoring}})
pd.DataFrame(cv_rows).to_csv(REPORTS / "ann_grouped_cv_scores.csv", index=False)
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
pd.DataFrame(cm, index=["actual_negative", "actual_positive"], columns=["predicted_negative", "predicted_positive"]).to_csv(REPORTS / "ann_confusion_matrix.csv")
(REPORTS / "ann_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(REPORTS / "ann_classification_report.txt").write_text(classification_report(y_test, y_pred, target_names=["Negative", "Positive"], zero_division=0), encoding="utf-8")
joblib.dump(ann, MODELS / "ann_review2_pipeline.joblib")

# Confusion matrix graph.
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"]).plot(ax=ax, colorbar=False)
ax.set_title("ANN — Held-out Confusion Matrix")
fig.tight_layout(); fig.savefig(GRAPHS / "ann_confusion_matrix.png", dpi=180); plt.close(fig)

# ROC and precision-recall graphs from the held-out predictions.
fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="ANN")
ax.set_title("ANN — Held-out ROC Curve")
fig.tight_layout(); fig.savefig(GRAPHS / "ann_roc_curve.png", dpi=180); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax, name="ANN")
ax.set_title("ANN — Held-out Precision–Recall Curve")
fig.tight_layout(); fig.savefig(GRAPHS / "ann_precision_recall_curve.png", dpi=180); plt.close(fig)

clf = ann.named_steps["classifier"]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, len(clf.loss_curve_) + 1), clf.loss_curve_)
ax.set_xlabel("Training iteration")
ax.set_ylabel("Loss")
ax.set_title("ANN — Training Loss Curve")
ax.grid(True, alpha=0.25)
fig.tight_layout(); fig.savefig(GRAPHS / "ann_training_loss.png", dpi=180); plt.close(fig)

print("NEURODIAB REVIEW-II — ANN STEP 1 COMPLETE")
print(f"Dataset: {len(df)} rows x {df.shape[1]} columns; {X.shape[1]} input features")
print(f"Exact duplicate rows flagged: {num_duplicates}; groups used to prevent duplicate leakage: {pd.Series(groups).nunique()}")
print(f"Grouped holdout: train={len(train_idx)} test={len(test_idx)}")
print("Held-out metrics:")
for k in ["heldout_accuracy", "heldout_precision", "heldout_recall", "heldout_specificity", "heldout_f1", "heldout_roc_auc", "heldout_pr_auc_average_precision", "heldout_brier_score"]:
    print(f"  {k}: {metrics[k]:.4f}")
print("Grouped 5-fold CV:")
for k, v in metrics["cv_mean_std"].items():
    print(f"  {k}: {v['mean']:.4f} +/- {v['std']:.4f}")
print(f"ANN epochs: {clf.n_iter_}; model saved to {MODELS / 'ann_review2_pipeline.joblib'}")
print(f"Graphs saved to {GRAPHS}; reports saved to {REPORTS}")
if warning_text:
    print("Warnings:")
    for w in warning_text: print(" -", w)
