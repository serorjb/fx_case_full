from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union

NumberLike = Union[float, int]
ArrayLike = Union[NumberLike, np.ndarray, pd.Series]

def option_trade_cost(premium: ArrayLike,
                      abs_delta: ArrayLike,
                      wings_bps: float = 30.0,
                      atm_bps: float = 15.0):
    """
    Compute option trade cost:
      use wings_bps where |delta| <= 0.10 else atm_bps.
    Accepts scalars, numpy arrays, or pandas Series.
    Returns same shape as premium (broadcast if needed).
    """
    # Scalar fast path
    if np.isscalar(premium) and np.isscalar(abs_delta):
        bps = wings_bps if abs_delta <= 0.10 else atm_bps
        return abs(premium) * bps * 1e-4

    # If any input is a Series, operate in pandas to keep index
    if isinstance(premium, pd.Series) or isinstance(abs_delta, pd.Series):
        prem = premium if isinstance(premium, pd.Series) else pd.Series(premium, index=abs_delta.index)  # type: ignore
        delta = abs_delta if isinstance(abs_delta, pd.Series) else pd.Series(abs_delta, index=prem.index)  # type: ignore
        cond = delta.abs() <= 0.10
        bps_series = pd.Series(atm_bps, index=prem.index, dtype=float)
        bps_series[cond] = wings_bps
        return prem.abs() * bps_series * 1e-4

    # Numpy array path
    prem_arr = np.asarray(premium)
    delta_arr = np.asarray(abs_delta)
    bps_arr = np.where(np.abs(delta_arr) <= 0.10, wings_bps, atm_bps)
    return np.abs(prem_arr) * bps_arr * 1e-4

def hedge_cost_from_pips(notional: ArrayLike, pips: ArrayLike):
    """
    Hedge cost given notional and pips (price bps).
    Vectorized for scalars / arrays / Series.
    """
    if np.isscalar(notional) and np.isscalar(pips):
        return abs(notional) * abs(pips) * 1e-4

    if isinstance(notional, pd.Series) or isinstance(pips, pd.Series):
        notl = notional if isinstance(notional, pd.Series) else pd.Series(notional, index=pips.index)  # type: ignore
        pp = pips if isinstance(pips, pd.Series) else pd.Series(pips, index=notl.index)  # type: ignore
        return notl.abs() * pp.abs() * 1e-4

    notl_arr = np.asarray(notional)
    pips_arr = np.asarray(pips)
    return np.abs(notl_arr) * np.abs(pips_arr) * 1e-4