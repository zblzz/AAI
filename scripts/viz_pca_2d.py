"""
PCA 2D visualization for train.csv

Usage:
  python scripts/pca_2d.py --csv data/train.csv --outdir output/viz --target label

Outputs:
- pca_2d_scatter__by_{target}.png
- pca_2d_scatter__by_{target}__ellipses.png (optional confidence ellipses)
- pca_2d_summary.txt (explained variance ratios)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".", "+") else "_" for ch in name)


def _confidence_ellipse(ax, x, y, n_std=2.0, **kwargs):
    """
    Draw a covariance confidence ellipse of (x, y) onto ax.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 3:
        return

    cov = np.cov(x, y)
    if not np.isfinite(cov).all():
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # angle in degrees
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # width/height are "full" lengths (2*radius)
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 0))

    from matplotlib.patches import Ellipse
    ell = Ellipse((np.mean(x), np.mean(y)), width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ell)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--outdir", type=str, default="output/viz", help="Directory to save figures")
    parser.add_argument("--target", type=str, default="label", help="Target column name")
    parser.add_argument("--sample", type=int, default=0, help="If >0, randomly sample rows for faster plotting")
    parser.add_argument("--alpha", type=float, default=0.85, help="Point alpha")
    parser.add_argument("--s", type=float, default=28.0, help="Point size")
    parser.add_argument("--ellipses", action="store_true", help="Add class confidence ellipses (2 std)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    if args.sample and 0 < args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError("No numeric features found for PCA.")

    # Basic fill to avoid PCA failing on NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    evr = pca.explained_variance_ratio_
    summary = (
        f"Rows: {len(df)}\n"
        f"Num features used: {X.shape[1]}\n"
        f"PCA explained variance ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}\n"
        f"Total (PC1+PC2): {evr.sum():.4f}\n"
    )
    (outdir / "pca_2d_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)

    plot_df = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], args.target: y.astype(str).values})

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 7))
    ax = sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue=args.target,
        palette="Set2",
        s=args.s,
        alpha=args.alpha,
        edgecolor="none",
    )
    ax.set_title(f"PCA 2D scatter by {args.target}\nEVR: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}")
    ax.legend(title=args.target, loc="best")
    plt.tight_layout()
    plt.savefig(outdir / f"pca_2d_scatter__by_{safe_filename(args.target)}.png", dpi=220)
    plt.close()

    if args.ellipses:
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(9, 7))
        ax = sns.scatterplot(
            data=plot_df,
            x="PC1",
            y="PC2",
            hue=args.target,
            palette="Set2",
            s=args.s,
            alpha=args.alpha * 0.7,
            edgecolor="none",
        )
        for cls, g in plot_df.groupby(args.target):
            _confidence_ellipse(
                ax,
                g["PC1"].values,
                g["PC2"].values,
                n_std=2.0,
                fill=False,
                linewidth=2.0,
                alpha=0.9,
            )
        ax.set_title(f"PCA 2D scatter + 2σ ellipses by {args.target}\nEVR: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}")
        ax.legend(title=args.target, loc="best")
        plt.tight_layout()
        plt.savefig(outdir / f"pca_2d_scatter__by_{safe_filename(args.target)}__ellipses.png", dpi=220)
        plt.close()

    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()