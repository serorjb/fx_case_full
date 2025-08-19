from __future__ import annotations
import numpy as np
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
    PYPFOPT_AVAILABLE = True
except Exception:
    PYPFOPT_AVAILABLE = False

def _fallback_max_return(rets: np.ndarray) -> np.ndarray:
    mu = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0).mean(axis=0)
    mu = np.maximum(mu, 0.0)
    if mu.sum()==0: return np.ones_like(mu)/len(mu)
    return mu/mu.sum()

def _fallback_max_sharpe(rets: np.ndarray, eps=1e-8) -> np.ndarray:
    X = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0)
    cov = np.cov(X.T) + eps*np.eye(X.shape[1])
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
    # Clean inputs
    X = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    df = pd.DataFrame(X)

    # Robust moments
    mu = expected_returns.mean_historical_return(df, frequency=252)
    # Shrinkage to ensure PSD + stability
    S = risk_models.CovarianceShrinkage(df, frequency=252).ledoit_wolf()

    try:
        from pypfopt import EfficientFrontier
        ef = EfficientFrontier(mu, S, weight_bounds=(0.0, 1.0))
        # Small L2 regularizer to enforce strong convexity
        ef.add_objective(objective_functions.L2_reg, gamma=1e-4)

        if method == "max_return":
            # Practical "max return": MSR with tiny risk aversion often more stable than quadratic utility
            ef.max_quadratic_utility(risk_aversion=1e-4, market_neutral=False)
        else:
            ef.max_sharpe(risk_free_rate=0.0)

        w = ef.clean_weights()
        weights = np.array([w[i] for i in range(len(w))], dtype=float)
        weights = np.clip(weights, 0.0, 1.0)
        s = weights.sum()
        if s <= 0:
            raise ValueError("weights sum to zero")
        return weights/s
    except Exception:
        # Fallback if solver fails
        return _fallback_max_return(X) if method=='max_return' else _fallback_max_sharpe(X)
