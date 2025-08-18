from __future__ import annotations
import pandas as pd
from pathlib import Path

TENOR_DAYS = {"1W":7,"2W":14,"3W":21,"1M":30,"2M":60,"3M":90,"4M":120,"6M":182,"9M":273,"1Y":365}

def load_curve(curve_path: str) -> pd.DataFrame:
    p = Path(curve_path)
    df = pd.read_csv(p) if p.suffix=='.csv' else pd.read_parquet(p)
    if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
    return df

def load_fxo(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p) if p.suffix=='.csv' else pd.read_parquet(p)
    if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
    df.columns = [c.strip() for c in df.columns]
    return df

def spot_col(pair: str) -> str: return f"{pair} Curncy"
def fwd_points_col(pair: str, tenor: str) -> str: return f"{pair}{tenor} Curncy"
def atm_vol_col(pair: str, tenor: str) -> str: return f"{pair}V{tenor} Curncy"
def rr_col(pair: str, delta: int, tenor: str) -> str: return f"{pair}{delta}R{tenor} Curncy"
def bf_col(pair: str, delta: int, tenor: str) -> str: return f"{pair}{delta}B{tenor} Curncy"

def tenor_years(tenor: str, day_count: float=365.0) -> float:
    return TENOR_DAYS[tenor]/day_count
