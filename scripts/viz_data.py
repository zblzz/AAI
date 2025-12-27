"""
Visualize train.csv quickly.

Usage:
  python scripts/viz_data.py --csv data/train.csv --outdir outputs/viz --target label

This script will:
- Print basic info (shape, dtypes, missing values)
- Plot target distribution (if target exists)
- Plot numeric feature histograms + correlation heatmap
- Plot categorical feature top-k bar charts
- Save figures to --outdir
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for servers/no-display
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".", "+") else "_" for ch in name)


def summarize_dataframe(df: pd.DataFrame, target: Optional[str] = None) -> str:
    lines = []
    lines.append(f"Shape: {df.shape}")
    lines.append("\nDtypes:")
    lines.append(df.dtypes.to_string())

    na = df.isna().sum().sort_values(ascending=False)
    na_nonzero = na[na > 0]
    lines.append("\nMissing values (non-zero):")
    lines.append(na_nonzero.to_string() if len(na_nonzero) else "None")

    dup = df.duplicated().sum()
    lines.append(f"\nDuplicated rows: {dup}")

    if target and target in df.columns:
        lines.append(f"\nTarget '{target}' value counts:")
        lines.append(df[target].value_counts(dropna=False).to_string())

    return "\n".join(lines)


def plot_target_distribution(df: pd.DataFrame, target: str, outdir: Path) -> None:
    if target not in df.columns:
        return

    plt.figure(figsize=(10, 4))
    vc = df[target].value_counts(dropna=False)
    sns.barplot(x=vc.index.astype(str), y=vc.values, color="#4C72B0")
    plt.title(f"Target distribution: {target}")
    plt.xlabel(target)
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / f"target_distribution__{safe_filename(target)}.png", dpi=200)
    plt.close()


def plot_numeric_overview(df: pd.DataFrame, outdir: Path, max_cols: int = 30) -> None:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return

    # limit to avoid huge outputs
    num_cols = num_cols[:max_cols]

    # histograms
    n = len(num_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(4 * ncols, 3 * nrows))
    for i, c in enumerate(num_cols, start=1):
        ax = plt.subplot(nrows, ncols, i)
        series = df[c]
        sns.histplot(series.dropna(), bins=30, kde=False, ax=ax, color="#55A868")
        ax.set_title(c)
    plt.suptitle("Numeric feature histograms", y=1.02)
    plt.tight_layout()
    plt.savefig(outdir / "numeric_histograms.png", dpi=200, bbox_inches="tight")
    plt.close()

    # correlation heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(min(14, 0.6 * len(num_cols) + 6), min(12, 0.6 * len(num_cols) + 4)))
        sns.heatmap(corr, cmap="vlag", center=0, square=False)
        plt.title("Correlation heatmap (numeric features)")
        plt.tight_layout()
        plt.savefig(outdir / "numeric_correlation_heatmap.png", dpi=200)
        plt.close()


def plot_categorical_overview(df: pd.DataFrame, outdir: Path, top_k: int = 20, max_cols: int = 30) -> None:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        return
    cat_cols = cat_cols[:max_cols]

    for c in cat_cols:
        vc = df[c].astype("object").fillna("<NA>").value_counts().head(top_k)
        plt.figure(figsize=(10, max(3.5, 0.35 * len(vc) + 1.5)))
        sns.barplot(y=vc.index.astype(str), x=vc.values, color="#C44E52")
        plt.title(f"Top-{top_k} categories: {c}")
        plt.xlabel("count")
        plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(outdir / f"categorical_top{top_k}__{safe_filename(c)}.png", dpi=200)
        plt.close()


def plot_missingness(df: pd.DataFrame, outdir: Path, max_cols: int = 50) -> None:
    na_ratio = (df.isna().mean() * 100).sort_values(ascending=False)
    na_ratio = na_ratio[na_ratio > 0]
    if na_ratio.empty:
        return

    na_ratio = na_ratio.head(max_cols)
    plt.figure(figsize=(10, max(4, 0.25 * len(na_ratio) + 2)))
    sns.barplot(y=na_ratio.index, x=na_ratio.values, color="#8172B2")
    plt.title("Missingness by column (%)")
    plt.xlabel("missing %")
    plt.ylabel("column")
    plt.tight_layout()
    plt.savefig(outdir / "missingness_by_column.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--outdir", type=str, default="output/viz", help="Directory to save figures")
    parser.add_argument("--target", type=str, default="label", help="Target column name (optional)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K categories for categorical plots")
    parser.add_argument("--max-cols", type=int, default=30, help="Max numeric/categorical cols to plot")
    parser.add_argument("--sample", type=int, default=0, help="If >0, randomly sample rows for faster plotting")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    if args.sample and args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    # write summary
    summary = summarize_dataframe(df, target=args.target)
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    # plots
    sns.set_theme(style="whitegrid")
    plot_missingness(df, outdir=outdir, max_cols=50)
    if args.target:
        plot_target_distribution(df, target=args.target, outdir=outdir)
    plot_numeric_overview(df, outdir=outdir, max_cols=args.max_cols)
    plot_categorical_overview(df, outdir=outdir, top_k=args.top_k, max_cols=args.max_cols)

    print(f"\nSaved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()