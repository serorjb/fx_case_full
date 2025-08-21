"""
smile_replication.py

Construct FX option smile from ATM, risk reversal (RR) and butterfly (BF) quotes and
calibrate SABR to obtain model vols used for mispricing detection.

Formulas (standard FX quoting conventions):
  RR_25 = vol(25d call) - vol(25d put)
  BF_25 = 0.5 * (vol(25d call) + vol(25d put)) - vol(ATM)
=> vol(25d call) = ATM + BF_25 + 0.5 * RR_25
   vol(25d put)  = ATM + BF_25 - 0.5 * RR_25
Similarly for 10 delta.

We expose utilities to:
  - build a discrete smile in delta space
  - convert delta to strike under Black / GK approximation
  - calibrate SABR (beta fixed) using QuantLib helper in pricing_models.py
  - evaluate mispricing (market vol - sabr vol) per strike

Assumptions / Simplifications:
  - Use forward delta N(d1) ≈ delta (ignoring discount factors) for inversion.
  - Use continuous compounding rates from rates.get_domestic_foreign_rates.
  - If calibration fails, fall back to simple polynomial fit in strike-vol space.
  - Vols provided may be in percent; function normalizes to decimals.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
from rates import get_domestic_foreign_rates
from pricing_models import SABRModelQL
import warnings
warnings.filterwarnings('ignore')

TARGET_DELTAS = [0.10, 0.25, 0.50, 0.75, 0.90]
TENOR_DAYS = {'1W':7,'2W':14,'3W':21,'1M':30,'2M':60,'3M':90,'4M':120,'6M':180,'9M':270,'1Y':365,'12M':365}

@dataclass
class SmilePoint:
    delta: float
    strike: float
    market_vol: float
    sabr_vol: float = np.nan
    vol_edge: float = np.nan

@dataclass
class CalibratedSmile:
    pair: str
    date: pd.Timestamp
    tenor: str
    forward: float
    points: List[SmilePoint]
    sabr_params: Dict[str,float]


def _to_decimal(vol: float) -> float:
    if vol is None or not np.isfinite(vol):
        return np.nan
    v = float(vol)
    if v > 1.5:  # assume in percent
        v /= 100.0
    return v


def build_delta_vol_dict(atm: float, rr25: float, bf25: float, rr10: float, bf10: float) -> Dict[float,float]:
    atm = _to_decimal(atm)
    rr25 = _to_decimal(rr25)
    bf25 = _to_decimal(bf25)
    rr10 = _to_decimal(rr10)
    bf10 = _to_decimal(bf10)
    out = {}
    if np.isnan(atm):
        return out
    # 25 delta
    if not np.isnan(rr25) and not np.isnan(bf25):
        vol_25_call = atm + bf25 + 0.5 * rr25
        vol_25_put = atm + bf25 - 0.5 * rr25
    else:
        vol_25_call = vol_25_put = np.nan
    # 10 delta
    if not np.isnan(rr10) and not np.isnan(bf10):
        vol_10_call = atm + bf10 + 0.5 * rr10
        vol_10_put = atm + bf10 - 0.5 * rr10
    else:
        vol_10_call = vol_10_put = np.nan
    # Assemble
    out[0.50] = atm
    if not np.isnan(vol_25_put):
        out[0.25] = max(vol_25_put, 0.0001)
        out[0.75] = max(vol_25_call, 0.0001)
    if not np.isnan(vol_10_put):
        out[0.10] = max(vol_10_put, 0.0001)
        out[0.90] = max(vol_10_call, 0.0001)
    return out


def delta_to_strike(forward: float, vol: float, T: float, delta: float) -> float:
    """Invert forward delta ≈ N(d1) to strike under Black / GK.
    d1 = N^{-1}(delta); K = F * exp(-vol*sqrt(T)*d1 + 0.5*vol^2 T)
    """
    from math import exp, sqrt
    if vol <= 0 or T <= 0 or forward <= 0:
        return forward
    try:
        d1 = norm.ppf(delta)
        k = forward * np.exp(-vol * np.sqrt(T) * d1 + 0.5 * vol * vol * T)
        return float(k)
    except Exception:
        return forward


def interpolate_vol(delta_vols: Dict[float,float], target_delta: float) -> float:
    if target_delta in delta_vols:
        return delta_vols[target_delta]
    ds = sorted(delta_vols.keys())
    if not ds:
        return np.nan
    # clamp
    if target_delta <= ds[0]:
        return delta_vols[ds[0]]
    if target_delta >= ds[-1]:
        return delta_vols[ds[-1]]
    for i in range(len(ds)-1):
        if ds[i] <= target_delta <= ds[i+1]:
            w = (target_delta - ds[i])/(ds[i+1]-ds[i])
            return (1-w)*delta_vols[ds[i]] + w*delta_vols[ds[i+1]]
    return delta_vols[ds[-1]]


def calibrate_sabr(forward: float, T: float, strikes: List[float], vols: List[float]) -> Tuple[Dict[str,float], List[float]]:
    strikes_arr = np.array(strikes, dtype=float)
    vols_arr = np.array(vols, dtype=float)
    mask = np.isfinite(vols_arr) & (vols_arr>0)
    if mask.sum() < 3:
        return {'alpha':np.nan,'beta':0.5,'rho':0,'nu':0}, [np.nan]*len(strikes)
    try:
        model = SABRModelQL(forward, T)
        params = model.calibrate(strikes_arr[mask], vols_arr[mask], beta=0.5)
        sabr_vols = [model.sabr_vol(k, params['alpha'], params['beta'], params['rho'], params['nu']) for k in strikes]
        return params, sabr_vols
    except Exception:
        # Fallback: quadratic fit vol(k) = a + b k + c k^2
        try:
            k = strikes_arr[mask]
            v = vols_arr[mask]
            coeff = np.polyfit(k, v, 2)
            sabr_vols = list(np.polyval(coeff, strikes_arr))
            return {'alpha':np.nan,'beta':0.5,'rho':0,'nu':0}, sabr_vols
        except Exception:
            return {'alpha':np.nan,'beta':0.5,'rho':0,'nu':0}, [np.nan]*len(strikes)


def build_and_calibrate_smile(pair: str, date: pd.Timestamp, tenor: str, spot: float,
                               forward: float, atm: float, rr25: float, bf25: float,
                               rr10: float, bf10: float) -> Optional[CalibratedSmile]:
    T_days = TENOR_DAYS.get(tenor, 30)
    T = T_days/365.0
    delta_vols = build_delta_vol_dict(atm, rr25, bf25, rr10, bf10)
    if not delta_vols:
        return None
    # Ensure full target delta coverage via interpolation
    full_delta_vols = {d: interpolate_vol(delta_vols, d) for d in TARGET_DELTAS}
    strikes = []
    market_vols = []
    for d in TARGET_DELTAS:
        vol = full_delta_vols[d]
        k = delta_to_strike(forward, vol, T, d)
        strikes.append(k)
        market_vols.append(vol)
    params, sabr_vols = calibrate_sabr(forward, T, strikes, market_vols)
    points = []
    for d, k, mv, sv in zip(TARGET_DELTAS, strikes, market_vols, sabr_vols):
        edge = mv - sv if (sv is not None and np.isfinite(sv)) else np.nan
        points.append(SmilePoint(delta=d, strike=k, market_vol=mv, sabr_vol=sv, vol_edge=edge))
    return CalibratedSmile(pair=pair, date=date, tenor=tenor, forward=forward, points=points, sabr_params=params)


def extract_overpriced_options(smile: CalibratedSmile, min_edge: float=0.005) -> List[SmilePoint]:
    candidates = []
    for p in smile.points:
        if p.vol_edge is not None and np.isfinite(p.vol_edge) and p.vol_edge > min_edge:
            candidates.append(p)
    # sort by edge desc
    candidates.sort(key=lambda x: x.vol_edge if x.vol_edge is not None else -1, reverse=True)
    return candidates

