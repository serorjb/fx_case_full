"""
rates.py
Utility for loading and interpolating daily risk-free rates curves built from FRED T-Bill data.
Previously named *discount_curves*; all nomenclature now consolidated as *rates* per request.

Outputs:
- data/rates.parquet with columns: date, tenor, tenor_years, discount_factor, interpolated_rate
- results/rates_3d.png / results/rates_3d.html (optional plots)

If data/rates.parquet is missing, a flat 2% curve fallback is created (or curves are built on-demand).
Foreign (quote currency) rate inferred via covered interest parity when a forward is provided.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, Dict
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import os

# Parquet path renamed
_RATES_PATH = Path("data/rates.parquet")
_FRED_DIR = Path("data/FRED")

# --------------------
# FRED Loading & Curve Build
# --------------------

def _load_fred_series(file: Path, label: str) -> pd.DataFrame:
    if not file.exists():
        return pd.DataFrame(columns=['date', label]).set_index('date')
    df = pd.read_csv(file)
    if df.shape[1] < 2:
        return pd.DataFrame(columns=['date', label]).set_index('date')
    df.columns = ['date', label]
    df['date'] = pd.to_datetime(df['date'])
    df[label] = pd.to_numeric(df[label], errors='coerce') / 100.0  # convert pct to decimal
    return df.set_index('date')


def build_rates(
    fred_dir: Path = _FRED_DIR,
    save_path: Path = _RATES_PATH,
    target_tenors: Dict[str, float] = None,
    make_plots: bool = True
) -> pd.DataFrame:
    """Build & interpolate daily risk-free rates curves from FRED T-Bill data.

    Parameters
    ----------
    fred_dir : Path
        Directory containing DTB1.csv, DTB3.csv, DTB6.csv, DTB12.csv.
    save_path : Path
        Parquet output path for rates curves (data/rates.parquet).
    target_tenors : dict
        Mapping tenor label -> year fraction. Uses default FX grid if None.
    make_plots : bool
        Whether to create surface plots under results/.

    Returns
    -------
    DataFrame with columns: date, tenor, tenor_years, discount_factor, interpolated_rate.
    """
    if target_tenors is None:
        target_tenors = {
            '1W': 1/52,
            '2W': 2/52,
            '3W': 3/52,
            '1M': 1/12,
            '2M': 2/12,
            '3M': 3/12,
            '4M': 4/12,
            '6M': 6/12,
            '9M': 9/12,
            '1Y': 1.0
        }

    # Load available series
    df_1m = _load_fred_series(fred_dir / 'DTB1.csv', '1M')
    df_3m = _load_fred_series(fred_dir / 'DTB3.csv', '3M')
    df_6m = _load_fred_series(fred_dir / 'DTB6.csv', '6M')
    df_1y = _load_fred_series(fred_dir / 'DTB12.csv', '1Y')

    if all(len(df)==0 for df in (df_1m, df_3m, df_6m, df_1y)):
        # Fallback single flat curve
        flat_records = []
        for d in pd.date_range('2007-01-01', periods=1):
            for label, T in target_tenors.items():
                flat_records.append({
                    'date': d,
                    'tenor': label,
                    'tenor_years': T,
                    'discount_factor': np.exp(-0.02*T),
                    'interpolated_rate': 0.02
                })
        df_flat = pd.DataFrame(flat_records)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_flat.to_parquet(save_path, index=False)
        return df_flat

    rates_df = df_1m.join([df_3m, df_6m, df_1y], how='outer').sort_index()

    records = []
    for date, row in rates_df.iterrows():
        avail = row.dropna()
        if len(avail) < 2:
            continue
        tenors = []
        dfs = []
        for lbl, r in avail.items():
            if lbl.endswith('M'):
                T = int(lbl[:-1]) / 12.0
            elif lbl.endswith('Y'):
                T = float(lbl[:-1])
            else:
                continue
            tenors.append(T)
            dfs.append(np.exp(-r * T))
        if len(tenors) < 2:
            continue
        try:
            interp = PchipInterpolator(tenors, dfs, extrapolate=True)
        except Exception:
            continue
        for t_label, T in target_tenors.items():
            df_val = float(interp(T))
            df_val = np.clip(df_val, 1e-6, 1.0)
            if T < 0.025:
                r = 1.0 - df_val
            else:
                r = -np.log(df_val)/T
            if T <= 0.1:
                r = np.clip(r, -0.05, 0.25)
            records.append({
                'date': date,
                'tenor': t_label,
                'tenor_years': T,
                'discount_factor': df_val,
                'interpolated_rate': r
            })
    df_rates = pd.DataFrame(records).sort_values(['date','tenor'])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not df_rates.empty:
        df_rates.to_parquet(save_path, index=False)

    if make_plots and not df_rates.empty:
        try:
            os.makedirs('results', exist_ok=True)
            pivot = df_rates.pivot_table(index='date', columns='tenor_years', values='interpolated_rate')
            dates = mdates.date2num(pivot.index)
            X, Y = np.meshgrid(pivot.columns.values, dates)
            Z = pivot.values
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
            ax.set_xlabel('Tenor (Y)'); ax.set_ylabel('Date'); ax.set_zlabel('Rate'); ax.set_title('Yield Surface')
            ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate(); plt.colorbar(surf, shrink=0.5, aspect=10)
            plt.tight_layout(); plt.savefig('results/rates_3d.png', dpi=180); plt.close(fig)
            fig2 = go.Figure(data=[go.Surface(x=pivot.columns.values, y=pivot.index, z=pivot.values, colorscale='Viridis')])
            fig2.update_layout(title='Yield Curves Over Time', scene=dict(xaxis_title='Tenor (Y)', yaxis_title='Date', zaxis_title='Rate'))
            fig2.write_html('results/rates_3d.html')
        except Exception:
            pass
    return df_rates

# --------------------
# Curve Store & Accessors
# --------------------
class RatesCurveStore:
    def __init__(self, path: Path = _RATES_PATH):
        self.path = path
        self._df = None
        if path.exists():
            try:
                self._df = pd.read_parquet(path)
                self._df['date'] = pd.to_datetime(self._df['date'])
            except Exception:
                self._df = None

    def available(self) -> bool:
        return self._df is not None and len(self._df) > 0

    @lru_cache(maxsize=4096)
    def get_rate(self, date: pd.Timestamp, T: float) -> float:
        if not self.available():
            if not self.path.exists():
                build_rates(make_plots=False)
                if self.path.exists():
                    try:
                        self._df = pd.read_parquet(self.path)
                        self._df['date'] = pd.to_datetime(self._df['date'])
                    except Exception:
                        return 0.02
            else:
                return 0.02
        d = pd.Timestamp(date).normalize()
        df_day = self._df[self._df['date'] == d]
        if df_day.empty:
            idx = (self._df['date'] - d).abs().idxmin()
            d = self._df.loc[idx, 'date']
            df_day = self._df[self._df['date'] == d]
        curve = df_day.sort_values('tenor_years')
        tenors = curve['tenor_years'].values
        rates = curve['interpolated_rate'].values
        if len(tenors)==0:
            return 0.02
        if T <= tenors[0]:
            return float(rates[0])
        if T >= tenors[-1]:
            return float(rates[-1])
        return float(np.interp(T, tenors, rates))

    def get_discount_factor(self, date: pd.Timestamp, T: float) -> float:
        r = self.get_rate(date, T)
        return float(np.exp(-r * T))

    def infer_foreign_rate(self, spot: float, forward: float, T: float, domestic_rate: float) -> float:
        if T <= 0 or spot <= 0 or forward <= 0:
            return domestic_rate
        try:
            diff = np.log(forward / spot) / T
            r_f = domestic_rate - diff
            return float(np.clip(r_f, -0.05, 0.15))
        except Exception:
            return domestic_rate

_curve_store = RatesCurveStore()

def ensure_rates():
    """Ensure data/rates.parquet exists; build if missing."""
    if not _RATES_PATH.exists():
        print("[rates] rates.parquet missing – building now...")
        build_rates(make_plots=False)
    else:
        # quick sanity load
        try:
            _ = pd.read_parquet(_RATES_PATH).head(1)
        except Exception:
            print("[rates] rates.parquet unreadable – rebuilding...")
            build_rates(make_plots=False)


def get_rate(date: pd.Timestamp, T: float) -> float:
    return _curve_store.get_rate(date, T)

def get_discount_factor(date: pd.Timestamp, T: float) -> float:
    return _curve_store.get_discount_factor(date, T)

def get_domestic_foreign_rates(date: pd.Timestamp, T: float, spot: float, forward: Optional[float]) -> Tuple[float, float]:
    r_d = get_rate(date, T)
    if forward is None:
        return r_d, r_d
    r_f = _curve_store.infer_foreign_rate(spot, forward, T, r_d)
    return r_d, r_f

if __name__ == "__main__":
    print("Building rates curves into", _RATES_PATH)
    dfc = build_rates(make_plots=True)
    print("Rows:", len(dfc))
