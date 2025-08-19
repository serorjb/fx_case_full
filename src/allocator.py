from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from pypfopt import risk_models, EfficientFrontier

log = logging.getLogger("allocator")

__all__ = [
    "prepare_cov_input",
    "allocate",
]

_FALLBACK_WARN_COUNT = 0
_FALLBACK_WARN_LIMIT = 3


def prepare_cov_input(
    df: pd.DataFrame,
    clip_std: float = 8.0,
    max_nan_frac_col: float = 0.55,
    max_nan_frac_row: float = 0.40,
    min_rows_factor: float = 0.9,
    abs_min_rows: int = 30,
    fill_method: str = "ffill_bfill_median",
    min_non_nan_count_col: int = 15,
    min_non_nan_frac_col: float | None = None,
    verbose: bool = False,
    label: str = ""
) -> Optional[pd.DataFrame]:
    """
    Permissive cleaner with early column sparsity pruning.

    Column pruning:
      - Drop columns with zero non-NaN.
      - Drop columns with non-NaN count < min_non_nan_count_col.
      - If min_non_nan_frac_col provided, also require non-NaN fraction >= min_non_nan_frac_col.
      - Always also enforce completeness >= (1 - max_nan_frac_col).

    Remaining steps:
      - Clip outliers.
      - Optional row NaN fraction filter.
      - Gap fill.
      - Drop zero-variance.
      - Soft min rows check.

    Returns cleaned DataFrame or None.
    """
    if df is None or df.empty:
        return None
    in_rows, in_cols = df.shape
    x = df.replace([np.inf, -np.inf], np.nan)

    # Initial drop: all-NaN columns
    x = x.dropna(axis=1, how="all")
    if x.empty:
        return None

    non_nan_counts = x.notna().sum()
    non_nan_frac = non_nan_counts / len(x)

    # Criteria
    completeness_threshold = 1 - max_nan_frac_col
    keep_mask = (
        (non_nan_counts >= min_non_nan_count_col) &
        (non_nan_frac >= completeness_threshold)
    )
    if min_non_nan_frac_col is not None:
        keep_mask &= (non_nan_frac >= min_non_nan_frac_col)

    dropped_sparse = x.columns[~keep_mask].tolist()
    x = x.loc[:, keep_mask]
    if x.shape[1] == 0:
        if verbose:
            log.debug("Cleaner %s: all columns dropped for sparsity.", label)
        return None
    if verbose and dropped_sparse:
        log.debug("Cleaner %s: dropped sparse cols (%d): %s",
                  label, len(dropped_sparse), dropped_sparse[:12] + (["..."] if len(dropped_sparse) > 12 else []))

    # Clip extremes
    if clip_std is not None:
        stds = x.std(ddof=0)
        for c in x.columns:
            s = stds[c]
            if s > 0 and np.isfinite(s):
                lim = clip_std * s
                x[c] = x[c].clip(-lim, lim)

    # Row filter (pre-fill) if method leaves potential NaNs
    if fill_method not in ("zero", "median"):
        row_nan_frac = x.isna().sum(axis=1) / x.shape[1]
        x = x.loc[row_nan_frac <= max_nan_frac_row]
        if x.empty:
            return None

    # Fill methods
    if fill_method == "ffill_bfill":
        x = x.ffill().bfill()
    elif fill_method == "ffill_bfill_median":
        x = x.ffill().bfill()
        if x.isna().any().any():
            med = x.median()
            x = x.fillna(med)
    elif fill_method == "median":
        med = x.median()
        x = x.fillna(med)
    elif fill_method == "zero":
        x = x.fillna(0.0)
    else:
        x = x.ffill().bfill()

    # Safety fill
    if x.isna().any().any():
        med = x.median()
        x = x.fillna(med)

    # Drop zero variance
    var = x.var(ddof=0)
    zvar = var[var <= 1e-14].index
    if len(zvar):
        x = x.drop(columns=zvar)

    if x.shape[1] == 0:
        return None

    # Min rows (soft)
    required_hard = max(abs_min_rows, int(np.ceil(min_rows_factor * x.shape[1])))
    if x.shape[0] < required_hard:
        soft = max(int(0.6 * abs_min_rows), 10)
        if x.shape[0] < soft:
            if verbose:
                log.debug("Cleaner %s: %d rows < soft %d -> reject.", label, x.shape[0], soft)
            return None
        if verbose:
            log.debug("Cleaner %s: %d rows < hard %d but kept (soft).", label, x.shape[0], required_hard)

    if verbose:
        log.debug("Cleaner %s: in_rows=%d in_cols=%d -> out_rows=%d out_cols=%d",
                  label, in_rows, in_cols, x.shape[0], x.shape[1])
    return x


def _equal_weight(cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(cols), index=cols, dtype=float)


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
    max_nan_frac_col: float = 0.55,
    max_nan_frac_row: float = 0.40,
    min_rows_factor: float = 0.9,
    abs_min_rows: int = 30,
    fill_method: str = "ffill_bfill_median",
    min_non_nan_count_col: int = 15,
    min_non_nan_frac_col: float | None = None,
    verbose_clean: bool = False
) -> pd.Series:
    """
    Allocation with early sparse-column pruning.

    New params:
      - min_non_nan_count_col: minimum count of non-NaN observations to retain a column.
      - min_non_nan_frac_col: optional minimum fraction of non-NaN values to retain a column (after count filter).
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    original_cols = list(returns.columns)
    work = returns

    if clean:
        try:
            label = str(work.index[-1].date())
        except Exception:
            label = ""
        cleaned = prepare_cov_input(
            work,
            clip_std=clip_std,
            max_nan_frac_col=max_nan_frac_col,
            max_nan_frac_row=max_nan_frac_row,
            min_rows_factor=min_rows_factor,
            abs_min_rows=abs_min_rows,
            fill_method=fill_method,
            min_non_nan_count_col=min_non_nan_count_col,
            min_non_nan_frac_col=min_non_nan_frac_col,
            verbose=verbose_clean,
            label=label
        )
        if cleaned is None:
            global _FALLBACK_WARN_COUNT
            if _FALLBACK_WARN_COUNT < _FALLBACK_WARN_LIMIT:
                log.warning("Allocation fallback: insufficient clean data -> equal weights.")
                _FALLBACK_WARN_COUNT += 1
            else:
                log.debug("Allocation fallback (suppressed).")
            return _equal_weight(original_cols)
        work = cleaned

    if work.shape[1] == 1:
        return pd.Series([1.0], index=work.columns).reindex(original_cols, fill_value=0.0)

    m = method.lower()
    if m == "equal_weight":
        w_core = _equal_weight(list(work.columns))
    elif m == "max_return":
        w_core = _max_return(work)
    elif m in ("inv_vol", "risk_parity"):
        w_core = _inverse_vol(work)
    elif m == "min_vol":
        w_core = _mv_opt(work, "min_vol")
    elif m in ("sharpe", "max_sharpe"):
        w_core = _mv_opt(work, "sharpe")
    else:
        w_core = _equal_weight(list(work.columns))

    w_full = w_core.reindex(original_cols, fill_value=0.0)
    s = w_full.sum()
    if s > 0:
        w_full = w_full / s
    else:
        w_full = _equal_weight(original_cols)
    return w_full.astype(float)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    df_test = pd.DataFrame(rng.normal(0, 0.01, size=(90, 12)),
                           columns=[f"C{i}" for i in range(12)])
    # Make some sparse / empty columns
    df_test.loc[:, "C3"] = np.nan
    df_test.loc[df_test.index[::6], "C5"] = np.nan
    df_test.loc[df_test.index[::4], "C7"] = np.nan
    w = allocate(df_test, method="sharpe", verbose_clean=True,
                 min_non_nan_count_col=20, min_non_nan_frac_col=0.4)
    print(w)
