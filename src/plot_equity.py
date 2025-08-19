# file: src/plot_equity.py
from __future__ import annotations
import tomllib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def load_config(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))

def _load_equity(path: Path, name: str) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" not in df.columns or "equity" not in df.columns:
        return None
    s = pd.Series(df["equity"].values, index=pd.to_datetime(df["date"]), name=name)
    return s.sort_index()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="Use log scale on Y axis")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root / "config" / "config.toml")
    outdir = Path(cfg["reporting"]["outdir"])
    curves: list[pd.Series] = []

    for model in cfg["models"]["use"]:
        s = _load_equity(outdir / f"equity_{model}.csv", model)
        if s is not None:
            curves.append(s)

    for method in cfg["allocation"]["methods"]:
        s = _load_equity(outdir / f"equity_alloc_{method}.csv", f"alloc_{method}")
        if s is not None:
            curves.append(s)

    if not curves:
        print("No equity files found.")
        return

    df = pd.concat(curves, axis=1)

    # Normalize each series by its own first valid value (independent starts)
    for col in df.columns:
        first_idx = df[col].first_valid_index()
        if first_idx is not None and pd.notna(df.at[first_idx, col]) and df.at[first_idx, col] != 0:
            df[col] = df[col] / df.at[first_idx, col]

    # Replace non-positive with NaN (log scale cannot plot <= 0)
    if args.log:
        nonpos = (df <= 0)
        if nonpos.any().any():
            df = df.mask(nonpos)
            # Optional: small epsilon fill (uncomment if you prefer tiny floor instead of gaps)
            # df = df.fillna(1e-8)

    coverage = pd.DataFrame({
        "first_date": [df[c].first_valid_index() for c in df.columns],
        "last_date": [df[c].last_valid_index() for c in df.columns],
        "non_na": [df[c].count() for c in df.columns]
    }, index=df.columns)
    coverage.to_csv(outdir / "equity_coverage.csv")

    plt.figure(figsize=(11, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, linewidth=1.2)

    for col in df.columns:
        start = df[col].first_valid_index()
        if start:
            plt.axvline(start, color="gray", alpha=0.1)

    plt.title("Equity Curves (Individually Normalized)" + (" [Log]" if args.log else ""))
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized to 1 at own start)")
    plt.yscale("log")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3, which="both")
    out_name = "equity_comparison_individual_log.png" if args.log else "equity_comparison_individual.png"
    out_path = outdir / out_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()