import numpy as np
from math import sqrt, log
from scipy.stats import norm
from quantlib_utils import black_price_ql  # uses QuantLib.blackFormula if available

# ---------- Black-76 (forward) ----------
def _black_price_py(F, K, sigma, T, df=1.0, call=True):
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        intrinsic = max((F - K) if call else (K - F), 0.0)
        return df * intrinsic
    v = sigma * sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if call:
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def black_price(F, K, sigma, T, df=1.0, call=True):
    """Black-76 price on forward with discount factor df (domestic)."""
    pv = black_price_ql(F, K, sigma, T, df, call)
    if pv is not None:
        return float(pv)
    return _black_price_py(F, K, sigma, T, df, call)

def black_forward_delta(F, K, sigma, T, call=True):
    from math import sqrt, log
    from scipy.stats import norm
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        # directional intrinsic delta fallback
        if call:
            return 1.0 if F > K else 0.0
        else:
            return -1.0 if F < K else 0.0
    v = sigma * sqrt(T)
    d1 = (log(F / K) + 0.5 * v * v) / v
    return norm.cdf(d1) if call else norm.cdf(d1) - 1.0


# ---------- Garman–Kohlhagen (spot) ----------
def _rd_q_from(F, S, T, DFd):
    """Recover r_d and r_f from S, F, DF_d: r_d = -ln(DFd)/T, r_f = r_d - ln(F/S)/T."""
    T = max(T, 1e-12)
    if DFd <= 0 or S <= 0 or F <= 0:
        return 0.0, 0.0
    r_d = -np.log(DFd) / T
    r_f = r_d - np.log(F / S) / T
    return float(r_d), float(r_f)

def gk_price(S, K, sigma, T, DFd, F, call=True):
    """
    GK price using spot S with domestic DF (DFd) and F to infer r_f.
    This equals Black-76(F, K, sigma, T, DFd) numerically.
    """
    r_d, r_f = _rd_q_from(F, S, T, DFd)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or DFd <= 0:
        intrinsic = max((S - K) if call else (K - S), 0.0)
        return DFd * intrinsic  # when T~0, DFd≈1
    v = sigma * sqrt(T)
    d1 = (log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    disc_r = DFd
    disc_q = np.exp(-r_f * T)
    if call:
        return disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2)
    else:
        return disc_r * K * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1)

def gk_spot_delta(S, K, sigma, T, DFd, F, call=True):
    """Spot delta under GK: e^{-r_f T} N(d1) for call; minus that for put."""
    r_d, r_f = _rd_q_from(F, S, T, DFd)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if (call and S > K) else (-1.0 if ((not call) and S < K) else 0.0)
    v = sigma * sqrt(T)
    d1 = (log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / v
    phi = 1.0 if call else -1.0
    return np.exp(-r_f * T) * norm.cdf(phi * d1) * phi
