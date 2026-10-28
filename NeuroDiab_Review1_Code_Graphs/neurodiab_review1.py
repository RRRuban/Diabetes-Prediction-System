#!/usr/bin/env python
"""
NeuroDiab Review-I reproducible analysis.
Usage:
  python neurodiab_review1.py --csv path/to/diabetes_data_upload.csv --out results

This script expects the official UCI Early Stage Diabetes Risk Prediction CSV.
It computes checks and creates graphs from the CSV itself. It does not hard-code
the summary counts from the presentation.
"""
from pathlib import Path
import argparse
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PURPLE = "#8754B5"
GREY = "#C9C7D3"

def find_col(df, candidates):
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in norm:
            return norm[key]
    return None

def norm_class(s):
    return s.astype(str).str.strip().str.lower().map({
        "positive": "Positive", "negative": "Negative",
        "1": "Positive", "0": "Negative"
    }).fillna(s.astype(str).str.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the UCI dataset CSV")
    ap.add_argument("--out", default="neurodiab_review1_results", help="Output directory")
    args = ap.parse_args()
    csv_path = Path(args.csv)
    out = Path(args.out)
    graph_dir = out / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    target = find_col(df, ["class", "target", "outcome"])
    if target is None:
        raise ValueError(f"Could not identify target column. Found columns: {df.columns.tolist()}")

    # Data audit
    audit = pd.DataFrame({
        "check": ["rows", "columns_including_target", "missing_cells", "exact_duplicate_rows"],
        "value": [len(df), len(df.columns), int(df.isna().sum().sum()), int(df.duplicated().sum())]
    })
    audit.to_csv(out / "data_audit.csv", index=False)
    df.isna().sum().rename("missing_count").to_csv(out / "missing_by_column.csv")
    df.dtypes.astype(str).rename("dtype").to_csv(out / "column_dtypes.csv")
    df[target].value_counts(dropna=False).rename_axis("class").reset_index(name="count").to_csv(out / "class_counts.csv", index=False)

    # Map target labels only for plotting; retain original dataset for audit.
    y_plot = norm_class(df[target])
    class_counts = y_plot.value_counts()
    plt.figure(figsize=(7,5))
    bars = plt.bar(class_counts.index.astype(str), class_counts.values, color=PURPLE)
    plt.title("Class Distribution", loc="left", fontsize=15, fontweight="bold")
    plt.ylabel("Number of records")
    for b, v in zip(bars, class_counts.values):
        plt.text(b.get_x()+b.get_width()/2, b.get_height(), str(int(v)), ha="center", va="bottom")
    plt.grid(axis="y", alpha=.25); plt.gca().set_axisbelow(True); plt.tight_layout()
    plt.savefig(graph_dir/"01_class_distribution.png", dpi=180); plt.close()

    # Gender by target class
    gender = find_col(df, ["Gender", "Sex"])
    if gender:
        tab = pd.crosstab(df[gender].astype(str).str.strip().str.title(), y_plot)
        ax = tab.plot(kind="bar", figsize=(8,5), color=[PURPLE, GREY] if len(tab.columns)==2 else None)
        ax.set_title("Class Counts by Gender", loc="left", fontsize=15, fontweight="bold")
        ax.set_xlabel("Gender"); ax.set_ylabel("Number of records")
        plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(graph_dir/"02_gender_by_class.png", dpi=180); plt.close()
        tab.to_csv(out/"gender_by_class.csv")

    # Symptom Yes counts
    excluded = {target, gender}
    age_col = find_col(df, ["Age"])
    symptom_cols = [c for c in df.columns if c not in excluded and c != age_col]
    symptom_counts = {}
    for c in symptom_cols:
        vals = df[c].astype(str).str.strip().str.lower()
        yes_n = int(vals.isin(["yes", "y", "1", "true"]).sum())
        if vals.isin(["yes","no","y","n","1","0","true","false"]).any():
            symptom_counts[c] = yes_n
    if symptom_counts:
        sc = pd.Series(symptom_counts).sort_values(ascending=False)
        sc.to_csv(out/"symptom_yes_counts.csv", header=["yes_count"])
        plt.figure(figsize=(11,6))
        bars = plt.bar(sc.index, sc.values, color=PURPLE)
        plt.title('Symptom Counts (Yes responses)', loc="left", fontsize=15, fontweight="bold")
        plt.ylabel("Number of Yes responses"); plt.xticks(rotation=40, ha="right")
        for b,v in zip(bars,sc.values):
            plt.text(b.get_x()+b.get_width()/2,b.get_height(),str(int(v)),ha="center",va="bottom",fontsize=8)
        plt.grid(axis="y",alpha=.25); plt.gca().set_axisbelow(True); plt.tight_layout()
        plt.savefig(graph_dir/"03_symptom_yes_counts.png",dpi=180); plt.close()

    # Age distribution by class
    if age_col:
        tmp = pd.DataFrame({"Age": pd.to_numeric(df[age_col], errors="coerce"), "Class": y_plot}).dropna()
        plt.figure(figsize=(8,5))
        groups = [tmp.loc[tmp["Class"]==c,"Age"].values for c in tmp["Class"].dropna().unique()]
        labels = list(tmp["Class"].dropna().unique())
        plt.boxplot(groups, tick_labels=labels, patch_artist=False)
        plt.title("Age Distribution by Class", loc="left", fontsize=15, fontweight="bold")
        plt.ylabel("Age"); plt.tight_layout()
        plt.savefig(graph_dir/"04_age_by_class_boxplot.png",dpi=180); plt.close()
        tmp.groupby("Class")["Age"].describe().to_csv(out/"age_summary_by_class.csv")

    # Encode binary categories for Pearson correlation. Age remains numeric.
    enc = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            enc[c] = pd.to_numeric(s, errors="coerce")
        else:
            vals = s.astype(str).str.strip().str.lower()
            mapping = {"yes":1, "no":0, "female":1, "male":0, "positive":1, "negative":0}
            unique = set(vals.dropna().unique())
            if unique and unique.issubset(set(mapping)):
                enc[c] = vals.map(mapping)
            else:
                # One-hot encode non-binary categorical values without target leakage.
                dummies = pd.get_dummies(s.astype("category"), prefix=c, dtype=float)
                enc = pd.concat([enc, dummies], axis=1)
    y_map = y_plot.map({"Positive":1, "Negative":0})
    if y_map.notna().all() and y_map.nunique()==2:
        corrs = enc.apply(lambda col: col.corr(y_map)).dropna()
        corr_df = pd.DataFrame({"feature": corrs.index, "pearson_r": corrs.values})
        corr_df["absolute_correlation"] = corr_df["pearson_r"].abs()
        corr_df = corr_df.sort_values("absolute_correlation", ascending=False)
        corr_df.to_csv(out/"absolute_correlation_ranking.csv", index=False)
        top = corr_df.head(20).sort_values("absolute_correlation")
        plt.figure(figsize=(10,7))
        bars = plt.barh(top["feature"], top["absolute_correlation"], color=PURPLE)
        plt.title("Absolute Correlation Ranking", loc="left", fontsize=15, fontweight="bold")
        plt.xlabel("Absolute Pearson correlation with class")
        for b,v in zip(bars,top["absolute_correlation"]):
            plt.text(b.get_width(), b.get_y()+b.get_height()/2, f" {v:.3f}", va="center", fontsize=8)
        plt.tight_layout(); plt.savefig(graph_dir/"05_absolute_correlation_ranking.png",dpi=180); plt.close()

    # Save a concise text report
    with open(out/"summary.txt","w",encoding="utf-8") as f:
        f.write(f"Dataset file: {csv_path.name}\n")
        f.write(f"Rows: {len(df)}\nColumns incl. target: {len(df.columns)}\n")
        f.write(f"Target column: {target}\n")
        f.write(f"Missing cells: {int(df.isna().sum().sum())}\n")
        f.write(f"Exact duplicate rows: {int(df.duplicated().sum())}\n\n")
        f.write("Class counts:\n")
        f.write(y_plot.value_counts().to_string())
        f.write("\n\nGraphs are generated from the supplied CSV. Review all encodings and outputs before using them in an academic submission.\n")
    print(f"Analysis complete. Outputs saved to: {out.resolve()}")

if __name__ == "__main__":
    main()
