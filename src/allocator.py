# file: src/allocator.py
from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from pypfopt import risk_models, EfficientFrontier

log = logging.getLogger(__name__)

__all__ = [
    "prepare_cov_input",
    "allocate",
]

def prepare_cov_input(
    df: pd.DataFrame,
    clip_std: float = 8.0,
    max_nan_frac: float = 0.40,
    min_rows_factor: float = 1.2,
    abs_min_rows: int = 20,
    fill_method: str = "ffill_bfill",
    verbose: bool = False,
    label: str = ""
) -> Optional[pd.DataFrame]:
    """
    Clean return window for covariance / optimization.

    Steps:
      1. Replace +/-inf with NaN.
      2. Remove columns with all NaN.
      3. Keep columns with at least (1 - max_nan_frac) finite observations.
      4. Clip extreme values per column at clip_std * std (if std > 0).
      5. Fill small gaps (ffill then bfill) if requested.
      6. Drop any residual rows with NaN (only among kept columns).
      7. Drop zero-variance columns.
      8. Require at least max(abs_min_rows, int(min_rows_factor * n_assets)) rows.

    Returns cleaned DataFrame or None if insufficient data.
    """
    if df is None or df.empty:
        return None
    initial_rows, initial_cols = df.shape

    x = df.replace([np.inf, -np.inf], np.nan)
    # Drop all-NaN columns
    x = x.dropna(axis=1, how="all")
    if x.empty:
        if verbose:
            log.debug("Cleaner %s: all columns all-NaN after drop.", label)
        return None

    # Column completeness filter
    finite_counts = np.isfinite(x).sum(axis=0)
    completeness = finite_counts / len(x)
    keep_cols = completeness[(completeness >= (1 - max_nan_frac)) & (finite_counts >= min_non_nan_abs)].index.tolist()
    x = x[keep_cols]
    if x.shape[1] == 0:
        if verbose:
            log.debug("Cleaner %s: no columns pass completeness threshold.", label)
        return None

    # Clip extremes
    if clip_std is not None:
        col_std = x.std(ddof=0)
        for c in x.columns:
            s = col_std[c]
            if s > 0 and np.isfinite(s):
                lim = clip_std * s
                x[c] = x[c].clip(-lim, lim)

    # Fill gaps (small ones) before final drop
    if fill_method == "ffill_bfill":
        x = x.ffill().bfill()
    elif fill_method == "median":
        med = x.median()
        x = x.fillna(med)
    elif fill_method == "zero":
        x = x.fillna(0.0)

    # Final NaN purge
    x = x.dropna(how="any")
    if x.empty:
        if verbose:
            log.debug("Cleaner %s: empty after final NaN drop.", label)
        return None

    # Drop zero-variance columns
    nunique = x.nunique()
    zero_var_cols = nunique[nunique <= 1].index.tolist()
    if zero_var_cols:
        x = x.drop(columns=zero_var_cols)

    if x.shape[1] == 0:
        if verbose:
            log.debug("Cleaner %s: all columns zero variance.", label)
        return None

    # Min rows rule (adaptive)
    required = max(abs_min_rows, int(min_rows_factor * x.shape[1]))
    if x.shape[0] < required:
        if verbose:
            log.debug("Cleaner %s: %d rows < required %d (cols=%d).",
                      label, x.shape[0], required, x.shape[1])
        return None

    if verbose:
        log.debug(
            "Cleaner %s: in_rows=%d in_cols=%d -> out_rows=%d out_cols=%d kept=%.2f%%",
            label, initial_rows, initial_cols, x.shape[0], x.shape[1],
            100.0 * x.shape[1] / max(1, initial_cols)
        )
    return x


def _equal_weight(cols: list[str]) -> pd.Series:
    w = np.full(len(cols), 1.0 / len(cols)) if cols else np.array([])
    return pd.Series(w, index=cols, dtype=float)


def _max_return(returns: pd.DataFrame) -> pd.Series:
    mu = returns.mean()
    pos = mu.clip(lower=0)
    if pos.sum() <= 0:
        return _equal_weight(list(returns.columns))
    return (pos / pos.sum()).astype(float)


def _inverse_vol(returns: pd.DataFrame) -> pd.Series:
    vol = returns.std(ddof=0).replace(0, np.nan)
    inv = 1.0 / vol
    inv = inv.replace([np.inf], np.nan)
    if inv.dropna().sum() <= 0:
        return _equal_weight(list(returns.columns))
    w = inv / inv.sum()
    return w.fillna(0.0).astype(float)


def _mv_opt(returns: pd.DataFrame, objective: str) -> pd.Series:
    try:
        mu = returns.mean() * 252.0
        cov = risk_models.CovarianceShrinkage(returns, frequency=252).ledoit_wolf()
        ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, 1.0))
        if objective == "sharpe":
            ef.max_sharpe()
        else:
            ef.min_volatility()
        w = pd.Series(ef.clean_weights())
        w = w.reindex(returns.columns).fillna(0.0)
        if (w <= 0).all():
            return _equal_weight(list(returns.columns))
        return w.astype(float)
    except Exception as e:
        log.debug("MV optimization failed (%s): %s", objective, e)
        return _equal_weight(list(returns.columns))


def allocate(
    returns: pd.DataFrame,
    method: str = "max_return",
    clean: bool = True,
    clip_std: float = 8.0,
    max_nan_frac: float = 0.40,
    min_rows_factor: float = 1.2,
    abs_min_rows: int = 20,
    fill_method: str = "ffill_bfill",
    verbose_clean: bool = False
) -> pd.Series:
    """
    Allocate weights given a DataFrame of periodic returns.

    Methods:
      - equal_weight
      - max_return
      - inv_vol
      - min_vol
      - sharpe

    Returns pd.Series of weights aligned to original columns (missing -> 0).
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    original_cols = list(returns.columns)
    work = returns

    if clean:
        label = ""
        if len(returns.index):
            try:
                label = str(returns.index[-1].date())
            except Exception:
                label = ""
        cleaned = prepare_cov_input(
            returns,
            clip_std=clip_std,
            max_nan_frac=max_nan_frac,
            min_rows_factor=min_rows_factor,
            abs_min_rows=abs_min_rows,
            fill_method=fill_method,
            verbose=verbose_clean,
            label=label
        )
        if cleaned is None:
            # Single summarized warning (not per step)
            log.warning("Allocation fallback: insufficient clean data -> equal weights.")
            return _equal_weight(original_cols)
        work = cleaned

    if work.shape[1] == 1:
        single = pd.Series([1.0], index=work.columns)
        return single.reindex(original_cols, fill_value=0.0)

    method_l = method.lower()
    if method_l == "equal_weight":
        w_core = _equal_weight(list(work.columns))
    elif method_l == "max_return":
        w_core = _max_return(work)
    elif method_l in ("inv_vol", "risk_parity"):
        w_core = _inverse_vol(work)
    elif method_l == "min_vol":
        w_core = _mv_opt(work, "min_vol")
    elif method_l in ("sharpe", "max_sharpe"):
        w_core = _mv_opt(work, "sharpe")
    else:
        log.debug("Unknown method '%s' -> equal_weight.", method)
        w_core = _equal_weight(list(work.columns))

    # Map back to original
    w_full = w_core.reindex(original_cols, fill_value=0.0)

    total = w_full.sum()
    if total > 0:
        w_full = w_full / total
    else:
        w_full = _equal_weight(original_cols)

    return w_full.astype(float)


if __name__ == "__main__":
    # Simple diagnostic run
    rng = np.random.default_rng(0)
    df_test = pd.DataFrame(rng.normal(0, 0.01, size=(120, 6)),
                           columns=list("ABCDEF"))
    # Introduce NaNs
    df_test.iloc[::7, 2] = np.nan
    df_test.iloc[::11, 4] = np.nan
    w = allocate(df_test, method="sharpe", verbose_clean=True)
    print(w)