from __future__ import annotations
import math

try:
    import QuantLib as ql
    HAS_QL = True
except Exception:
    HAS_QL = False

def black_price_ql(F, K, sigma, T, df=1.0, call=True):
    if not HAS_QL:
        return None
    payoff = ql.Option.Call if call else ql.Option.Put
    # QuantLib blackFormula takes (payoffType, strike, forward, stdDev)
    stddev = sigma * math.sqrt(max(T, 1e-12))
    return float(df * ql.blackFormula(payoff, float(K), float(F), float(stddev)))

def sabr_vol_ql(F, K, T, alpha, beta, nu, rho):
    if not HAS_QL:
        return None
    if K <= 0 or F <= 0 or T <= 0:
        return None
    return float(ql.sabrVolatility(float(K), float(F), float(T),
                                   float(alpha), float(beta), float(nu), float(rho)))
