#!/usr/bin/env python3
"""
build_rates.py
--------------
Build a USD short-end curve from FRED DTB (1M,3M,6M,12M), then compute USD-collateralized foreign
zero rates via Covered Interest Parity (CIP) from a Bloomberg-style FX file with spot and forward points.
Outputs a long-format Parquet (or CSV fallback) with all rates needed for GK/SABR/VV pricing.

Inputs (defaults):
  FRED DTB CSVs:
    data/input/FRED/DTB1.csv, DTB3.csv, DTB6.csv, DTB12.csv
    -> columns: date, DTB1/3/6/12 (percent on discount basis)
  FX forwards file:
    data/input/fxo.parquet
    -> columns like: 'EURUSD Curncy' (spot), 'EURUSD1M Curncy' (forward points), ...

Output:
  data/output/rates.parquet

Notes:
  - Requires pandas + (pyarrow or fastparquet) for Parquet I/O.
  - Optionally uses SciPy for PCHIP (monotone interpolation). Falls back to linear if missing.
  - No look-ahead: USD curve for a given FX date uses the last DTB observation <= that date.
  - Short-end robustness: outside the DTB node range, zeros are CLAMPED to the nearest node (no U-shape).
  - Tenor labels: maps to 1W/2W/3W/1M/... with tolerance; otherwise prints decimal years.
  - Handles FX files already indexed by date (DatetimeIndex) or with a 'date' column.
  - Triangular completion: fills crosses (e.g., GBPNZD) using ln(F/S)/T = r_base - r_quote when one USD leg is missing.

Usage:
  python build_rates.py
  python build_rates.py --fred-dir data/input/FRED --fxo data/input/fxo.parquet --out data/output/rates.parquet
"""
import argparse
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Optional PCHIP for monotone interpolation
try:
    from scipy.interpolate import PchipInterpolator
    HAS_PCHIP = True
except Exception:
    HAS_PCHIP = False


# ------------------------------ Helpers ---------------------------------------

def parse_tenor(label: str) -> Optional[float]:
    """Parse '1W','2W','3W','1M','3M','6M','9M','12M','1Y' -> year fraction (ACT/365 approx)."""
    if label is None:
        return None
    s = str(label).upper()
    m = re.search(r'(\d+)\s*([DWMY])', s)
    if not m:
        return None
    n = int(m.group(1))
    u = m.group(2)
    if u == 'D':
        return n/365.0
    if u == 'W':
        return (n*7)/365.0
    if u == 'M':
        return n/12.0
    if u == 'Y':
        return float(n)
    return None


def discount_basis_to_cc(discount_rate_pct: float, T_years: float) -> float:
    """
    Convert T-bill discount-basis % at maturity T_years to a continuous-compounded zero (cc).
      y_mm = d * 360 / (1 - d * Days/360)
      y_bey ≈ y_mm * 365/360
      r_cc = ln(1 + y_bey * T) / T
    """
    d = float(discount_rate_pct) / 100.0
    days = int(round(T_years * 365))
    if days <= 0:
        return float('nan')
    y_mm = (d * 360.0) / (1.0 - d * (days/360.0))
    y_bey = y_mm * (365.0/360.0)
    r_cc = np.log(1.0 + y_bey * (days/365.0)) / (days/365.0)
    return float(r_cc)


def interp_zero(Tgrid: np.ndarray, Z: np.ndarray, Tq: np.ndarray) -> np.ndarray:
    """Interpolate zero curve; clamp outside node range to edge values; PCHIP inside if available."""
    if len(Tgrid) == 0:
        return np.full_like(Tq, np.nan, dtype=float)
    order = np.argsort(Tgrid)
    Tgrid = np.asarray(Tgrid)[order]
    Z = np.asarray(Z)[order]
    out = np.empty_like(Tq, dtype=float)
    Tq = np.asarray(Tq, dtype=float)
    inside = (Tq >= Tgrid[0]) & (Tq <= Tgrid[-1])
    # clamps
    out[Tq < Tgrid[0]] = Z[0]
    out[Tq > Tgrid[-1]] = Z[-1]
    # interpolate inside
    if inside.any():
        if HAS_PCHIP and len(Tgrid) >= 2:
            f = PchipInterpolator(Tgrid, Z, extrapolate=False)
            out[inside] = f(Tq[inside])
        else:
            out[inside] = np.interp(Tq[inside], Tgrid, Z)
    return out


def asof_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if 'date' in cl or 'time' in cl or 'timestamp' in cl:
            return c
    return None


def pip_size_for_pair(pair: str) -> float:
    """Forward points to price units: quote JPY -> 0.01, else -> 0.0001 (typical for G10 FX)."""
    quote = pair[3:6] if len(pair) >= 6 else ""
    return 0.01 if quote == 'JPY' else 1e-4


def detect_fx_fields_wide(df_fx: pd.DataFrame) -> Dict[str, Dict]:
    """
    Detect spot and forward columns in a Bloomberg-style wide FX dataframe
    (e.g., 'EURUSD Curncy', 'EURUSD1M Curncy', ...). Returns:
      { pair: {"spot": colname or None, "fwds": {colname:Tyears}} }
    """
    out: Dict[str, Dict] = {}
    def root_of(col: str):
        m = re.match(r'([A-Z]{6})(\d+[DWMY])?\s*Curncy$', col.strip())
        if m:
            return m.group(1), m.group(2)
        return None, None

    for c in df_fx.columns:
        if not isinstance(c, str):
            continue
        pair, ten = root_of(c)
        if pair is None:
            continue
        entry = out.setdefault(pair, {"spot": None, "fwds": {}})
        if ten is None:
            entry["spot"] = c
        else:
            T = parse_tenor(ten)
            if T and T > 0:
                entry["fwds"][c] = T
    return out


def load_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read Parquet at {path}. Install pyarrow or fastparquet. Error: {e}")


def write_parquet(df: pd.DataFrame, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        alt = out_path.with_suffix(".csv")
        print(f"[WARN] Failed to write Parquet ({e}). Writing CSV fallback to {alt}")
        df.to_csv(alt, index=False)


# ------------------------------ Core logic ------------------------------------

def build_usd_zero_from_dtb_row(row: pd.Series, tenor_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    tenors, zeros = [], []
    for c in tenor_cols:
        T = parse_tenor(c)
        if T is None or T <= 0:
            continue
        val = row.get(c, np.nan)
        if pd.isna(val):
            continue
        z = discount_basis_to_cc(val, T)
        if np.isfinite(z):
            tenors.append(T); zeros.append(z)
    if not tenors:
        return np.array([]), np.array([])
    order = np.argsort(tenors)
    return np.array(tenors, dtype=float)[order], np.array(zeros, dtype=float)[order]


def main(fred_dir: str, fxo_path: str, out_path: str) -> None:
    fred_dir = Path(fred_dir)

    # Load DTB CSVs
    fred_files = {"1M":"DTB1.csv","3M":"DTB3.csv","6M":"DTB6.csv","12M":"DTB12.csv"}
    fred_frames = []
    for lbl,fname in fred_files.items():
        p = fred_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing FRED file: {p}")
        df = pd.read_csv(p)
        if "date" not in df.columns:
            dcol = asof_column(df)
            if not dcol:
                raise ValueError(f"{p} has no 'date' column")
            df = df.rename(columns={dcol:"date"})
        rate_col = next((c for c in df.columns if c.lower().startswith("dtb")), None)
        if not rate_col:
            rate_col = [c for c in df.columns if c != "date"][0]
        df = df[["date", rate_col]].rename(columns={rate_col: lbl})
        fred_frames.append(df)

    df_dtb = fred_frames[0]
    for df in fred_frames[1:]:
        df_dtb = df_dtb.merge(df, on="date", how="outer")
    df_dtb["date"] = pd.to_datetime(df_dtb["date"], errors="coerce")
    df_dtb = df_dtb.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)

    usd_tenor_cols = ["1M","3M","6M","12M"]

    # Load FX forwards panel
    df_fx = load_parquet(str(fxo_path))
    # Normalize to DateTimeIndex if not already
    if not isinstance(df_fx.index, pd.DatetimeIndex):
        dcol_fx = asof_column(df_fx)
        if dcol_fx:
            df_fx[dcol_fx] = pd.to_datetime(df_fx[dcol_fx], errors="coerce")
            df_fx = df_fx.set_index(dcol_fx)
        else:
            df_fx.index = pd.to_datetime(df_fx.index, errors="coerce")
    df_fx = df_fx[~df_fx.index.isna()].sort_index()

    fx_fields = detect_fx_fields_wide(df_fx)

    # Prepare output collector
    rows = []

    # Index DTB by date for "pad" (no look-ahead)
    df_dtb_idx = df_dtb.set_index("date")

    # Build set of all forward tenors present
    all_Ts = sorted({round(float(T),8) for info in fx_fields.values() for T in info["fwds"].values()})

    # Cache for USD curves by date
    usd_curve_cache: Dict[pd.Timestamp, Tuple[np.ndarray, np.ndarray]] = {}

    def usd_zero_at_T(date: pd.Timestamp, T: float) -> float:
        """Interpolate USD zero for given T at date using DTB nodes; clamp outside range; no look-ahead."""
        if date not in usd_curve_cache:
            try:
                idx = df_dtb_idx.index.get_loc(date, method="pad")
                row = df_dtb_idx.iloc[idx]
            except Exception:
                usd_curve_cache[date] = (np.array([]), np.array([]))
                return float('nan')
            Tgrid, Z = build_usd_zero_from_dtb_row(row, usd_tenor_cols)
            usd_curve_cache[date] = (Tgrid, Z)
        else:
            Tgrid, Z = usd_curve_cache[date]
        if len(Tgrid) == 0:
            return float('nan')
        return float(interp_zero(Tgrid, Z, np.array([T]))[0])

    # Iterate FX rows (dates)
    for fx_date, fx_row in df_fx.iterrows():
        if pd.isna(fx_date):
            continue

        # 1) USD curve slice for all forward tenors used that day
        for T in all_Ts:
            z_usd = usd_zero_at_T(fx_date, T)
            if np.isfinite(z_usd):
                rows.append({
                    "date": fx_date, "tenor": None, "tenor_years": T,
                    "level": "USD", "entity": "USD",
                    "base_ccy": None, "quote_ccy": None, "pair": None,
                    "zero_cc": z_usd, "df": float(np.exp(-z_usd*T))
                })

        # 2a) CCY zeros from USD pairs directly via CIP
        ccy_zeros: Dict[str, Dict[float, float]] = {}
        for pair, info in fx_fields.items():
            spot_col = info["spot"]
            if spot_col is None or spot_col not in df_fx.columns:
                continue
            S = fx_row.get(spot_col, np.nan)
            if not np.isfinite(S) or S <= 0:
                continue
            base, quote = pair[:3], pair[3:6]
            involves_usd = (base == "USD") or (quote == "USD")
            if not involves_usd:
                continue
            pip = pip_size_for_pair(pair)
            for fcol, T in info["fwds"].items():
                if fcol not in df_fx.columns:
                    continue
                pts = fx_row.get(fcol, np.nan)
                if not np.isfinite(pts):
                    continue
                F = S + float(pts) * pip
                if F <= 0:
                    continue
                z_usd = usd_zero_at_T(fx_date, T)
                if not np.isfinite(z_usd):
                    continue
                ccy = quote if base == "USD" else base
                r_foreign = float(z_usd - np.log(F/S)/T)
                if np.isfinite(r_foreign):
                    ccy_zeros.setdefault(ccy, {})[round(float(T),8)] = r_foreign

        # 2b) Triangular completion using cross pairs (recover missing CCY zeros)
        for pair, info in fx_fields.items():
            base = pair[:3]; quote = pair[3:6]
            if (base == "USD") or (quote == "USD"):
                continue  # already handled
            spot_col = info["spot"]
            if spot_col is None or spot_col not in df_fx.columns:
                continue
            S = fx_row.get(spot_col, np.nan)
            if not np.isfinite(S) or S <= 0:
                continue
            pip = pip_size_for_pair(pair)
            for fcol, T in info["fwds"].items():
                if fcol not in df_fx.columns:
                    continue
                pts = fx_row.get(fcol, np.nan)
                if not np.isfinite(pts):
                    continue
                F = S + float(pts) * pip
                if F <= 0 or T <= 0:
                    continue
                delta = np.log(F/S)/T  # r_base - r_quote (USD-collateral measure)
                have_base = base in ccy_zeros and round(float(T),8) in ccy_zeros[base]
                have_quote = quote in ccy_zeros and round(float(T),8) in ccy_zeros[quote]
                if have_base and not have_quote:
                    r_base = ccy_zeros[base][round(float(T),8)]
                    r_quote = r_base - delta
                    ccy_zeros.setdefault(quote, {})[round(float(T),8)] = r_quote
                elif have_quote and not have_base:
                    r_quote = ccy_zeros[quote][round(float(T),8)]
                    r_base = r_quote + delta
                    ccy_zeros.setdefault(base, {})[round(float(T),8)] = r_base

        # Store CCY-level rows
        for ccy, zmap in ccy_zeros.items():
            for T, z in zmap.items():
                rows.append({
                    "date": fx_date, "tenor": None, "tenor_years": T,
                    "level": "CCY", "entity": ccy,
                    "base_ccy": None, "quote_ccy": None, "pair": None,
                    "zero_cc": z, "df": float(np.exp(-z*T))
                })

        # 3) Pair-level domestic/foreign rows when both legs known
        for pair, info in fx_fields.items():
            base, quote = pair[:3], pair[3:6]
            Ts_pair = {round(float(T),8) for T in info["fwds"].values()}
            for T in sorted(Ts_pair):
                z_base = (ccy_zeros.get(base, {}).get(T) if base != "USD" else usd_zero_at_T(fx_date, T))
                z_quote = (ccy_zeros.get(quote, {}).get(T) if quote != "USD" else usd_zero_at_T(fx_date, T))
                if z_base is None or z_quote is None or not np.isfinite(z_base) or not np.isfinite(z_quote):
                    continue
                rows.append({
                    "date": fx_date, "tenor": None, "tenor_years": T,
                    "level": "PAIR_FOREIGN", "entity": pair,
                    "base_ccy": base, "quote_ccy": quote, "pair": pair,
                    "zero_cc": float(z_base), "df": float(np.exp(-float(z_base)*T))
                })
                rows.append({
                    "date": fx_date, "tenor": None, "tenor_years": T,
                    "level": "PAIR_DOMESTIC", "entity": pair,
                    "base_ccy": base, "quote_ccy": quote, "pair": pair,
                    "zero_cc": float(z_quote), "df": float(np.exp(-float(z_quote)*T))
                })

    # Build output DataFrame
    out_df = pd.DataFrame(rows)

    # Tenor label with tolerance (weeks & months)
    def tenor_label(T: float) -> str:
        labels = {
            "1W": 7/365.0, "2W": 14/365.0, "3W": 21/365.0,
            "1M": 1/12.0, "2M": 2/12.0, "3M": 3/12.0, "4M": 4/12.0,
            "6M": 6/12.0, "9M": 9/12.0, "12M": 12/12.0
        }
        best_lbl, best_err = None, 1e9
        for lbl, Tref in labels.items():
            err = abs(T - Tref)
            if err < best_err:
                best_lbl, best_err = lbl, err
        if best_err <= 0.005:  # ~2 days tolerance
            return best_lbl
        return f"{T:.6f}y"

    if not out_df.empty:
        out_df["tenor"] = out_df["tenor_years"].apply(lambda x: tenor_label(float(x)))
        cols = ["date","tenor","tenor_years","level","entity","base_ccy","quote_ccy","pair","zero_cc","df"]
        out_df = out_df[cols].sort_values(["date","entity","level","tenor_years"]).reset_index(drop=True)

    write_parquet(out_df, out_path)
    print(f"Saved {len(out_df):,} rows to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fred-dir", type=str, default="data/input/FRED", help="Directory with DTB1/3/6/12 CSVs")
    ap.add_argument("--fxo", type=str, default="data/input/fxo.parquet", help="Path to FX forwards parquet (spot + points)")
    ap.add_argument("--out", type=str, default="data/output/rates.parquet", help="Output Parquet path")
    args = ap.parse_args()
    main(args.fred_dir, args.fxo, args.out)
