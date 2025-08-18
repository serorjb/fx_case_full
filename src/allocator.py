from __future__ import annotations
import numpy as np
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

def _fallback_max_return(rets: np.ndarray) -> np.ndarray:
    mu = rets.mean(axis=0)
    mu = np.maximum(mu, 0.0)
    if mu.sum()==0: return np.ones_like(mu)/len(mu)
    return mu/mu.sum()

def _fallback_max_sharpe(rets: np.ndarray, eps=1e-8) -> np.ndarray:
    mu = rets.mean(axis=0)
    cov = np.cov(rets.T) + eps*np.eye(rets.shape[1])
    inv = np.linalg.pinv(cov)
    w = inv @ mu
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s<=eps: return np.ones_like(w)/len(w)
    return w/s

def allocate(returns_matrix: np.ndarray, method: str="max_sharpe"):
    if returns_matrix.size == 0: return np.array([])
    if not PYPFOPT_AVAILABLE:
        return _fallback_max_return(returns_matrix) if method=='max_return' else _fallback_max_sharpe(returns_matrix)
    import pandas as pd
    df = pd.DataFrame(returns_matrix)
    mu = expected_returns.mean_historical_return(df, frequency=252)
    S = risk_models.sample_cov(df, frequency=252)
    if method=="max_return":
        ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
        ef.max_quadratic_utility(risk_aversion=1e-8, market_neutral=False)
    else:
        ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
        ef.max_sharpe(risk_free_rate=0.0)
    w = ef.clean_weights()
    weights = np.array([w[i] for i in range(len(w))], dtype=float)
    weights = np.clip(weights, 0.0, 1.0)
    s = weights.sum()
    if s<=0: return _fallback_max_sharpe(returns_matrix)
    return weights/s
