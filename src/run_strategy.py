from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from config_io import load_config
from allocator import allocate
from costs import option_trade_cost, hedge_cost_from_pips

log = logging.getLogger("run_strategy")
FREQ = 252


def _normalize_pair(s: pd.Series) -> pd.Series:
    return s.str.upper().str.replace("/", "", regex=False).str.strip()


_TENOR_MAP = {
    "14D": "2W", "21D": "3W", "30D": "1M", "60D": "2M", "90D": "3M",
    "1MO": "1M", "2MO": "2M", "3MO": "3M"
}


def _normalize_tenor(s: pd.Series) -> pd.Series:
    t = s.str.upper().str.strip()
    return t.replace(_TENOR_MAP)


def _load_priced_grid(results_dir: Path, model: str) -> pd.DataFrame:
    f = results_dir / f"priced_grid_{model}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing priced grid file {f}")
    df = pd.read_csv(f)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    # Normalize keys
    if "pair" in df.columns:
        df["pair"] = _normalize_pair(df["pair"])
    if "tenor" in df.columns:
        df["tenor"] = _normalize_tenor(df["tenor"])
    return df.sort_values(["date", "pair", "tenor", "delta", "side"]).reset_index(drop=True)


def _perf_stats(returns: pd.Series, freq: int = FREQ) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    eq = (1 + r).cumprod()
    ann_ret = (1 + r.mean()) ** freq - 1
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(sharpe), "max_drawdown": mdd}


def _save_series(path: Path, series: pd.Series, name: str):
    pd.DataFrame({"date": series.index, name: series.values}).to_csv(path, index=False)


def _simulate_sleeves(priced: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    kappa = float(cfg.get("margin", {}).get("kappa", 0.10))
    floor = float(cfg.get("margin", {}).get("floor", 0.01))
    delta_thr = float(cfg.get("hedging", {}).get("delta_threshold", 0.04))
    pips_maj = float(cfg.get("hedging", {}).get("spot_cost_pips_majors", 1.0))
    pips_xcc = float(cfg.get("hedging", {}).get("spot_cost_pips_crosses", 3.0))
    bps_atm = float(cfg.get("costs", {}).get("option_trade_bps_atm", 15.0))
    bps_wng = float(cfg.get("costs", {}).get("option_trade_bps_wings", 30.0))
    trade_cost_on_diff = bool(cfg.get("costs", {}).get("trade_cost_on_diff", True))
    cost_multiplier = float(cfg.get("costs", {}).get("cost_multiplier", 1.0))

    key = ["pair", "tenor", "delta", "side"]
    df = priced.copy()

    # PnL
    df["pv_diff"] = df.groupby(key)["pv"].diff()
    df["pnl"] = -(df["pv_diff"])
    df.loc[df["pv_diff"].isna(), "pnl"] = 0.0

    # Turnover
    if trade_cost_on_diff:
        prem_turn = df.groupby(key)["pv"].diff().abs()
        first = prem_turn.isna()
        prem_turn[first] = df.loc[first, "pv"].abs()
    else:
        prem_turn = df["pv"].abs()

    abs_delta = df["delta"].abs() if "delta" in df.columns else pd.Series(0.0, index=df.index)
    df["trade_cost"] = option_trade_cost(prem_turn, abs_delta,
                                         wings_bps=bps_wng, atm_bps=bps_atm).fillna(0.0) * cost_multiplier

    dchg = df.groupby(key)["delta"].diff().abs() if "delta" in df.columns else pd.Series(0.0, index=df.index)
    notional = np.nan_to_num(dchg) * df["F"].abs()
    is_major = df["pair"].str.contains("USD", case=False)
    pips_cost = np.where(is_major, pips_maj, pips_xcc)
    hedge_raw = hedge_cost_from_pips(notional, pips_cost)
    df["hedge_cost"] = np.where(dchg > delta_thr, hedge_raw, 0.0) * cost_multiplier

    vol = np.maximum(df["vol"].values, 1e-8)
    T = np.maximum(df["T"].values, 1e-8)
    Fv = np.maximum(df["F"].values, 1e-8)
    approx_vega = Fv * np.sqrt(T) * vol
    margin = np.maximum(kappa * np.abs(approx_vega), floor)

    net_pnl = df["pnl"] - df["trade_cost"] - df["hedge_cost"]
    df["ret"] = net_pnl / margin
    r = df["ret"].to_numpy()
    r[~np.isfinite(r)] = 0.0
    if np.count_nonzero(r) > 30:
        lo, hi = np.nanpercentile(r, [1, 99.5])
        r = np.clip(r, lo, hi)
    df["ret"] = r

    sleeves = (df[["date"] + key + ["ret"]]
               .pivot_table(index="date", columns=key, values="ret", aggfunc="first")
               .sort_index())

    # Equal weight across available
    avail = sleeves.notna().astype(float)
    w = avail.div(avail.sum(axis=1), axis=0).fillna(0.0)
    ew = (sleeves.fillna(0.0) * w).sum(axis=1)
    return sleeves, ew


def _extract_cleaning_cfg(cfg: dict) -> dict:
    section = cfg.get("allocation_cleaning", {}) or {}
    return {
        "clip_std": float(section.get("clip_std", 8.0)),
        "max_nan_frac_col": float(section.get("max_nan_frac_col", 0.55)),
        "max_nan_frac_row": float(section.get("max_nan_frac_row", 0.40)),
        "min_rows_factor": float(section.get("min_rows_factor", 0.9)),
        "abs_min_rows": int(section.get("abs_min_rows", 30)),
        "fill_method": str(section.get("fill_method", "ffill_bfill_median")),
        "min_non_nan_count_col": int(section.get("min_non_nan_count_col", 15)),
        "min_non_nan_frac_col": section.get("min_non_nan_frac_col", None),
    }


def _diagnose_sparse_columns(df: pd.DataFrame, outdir: Path, tag: str):
    counts = df.notna().sum()
    frac = counts / len(df)
    diag = pd.DataFrame({"non_nan_count": counts, "non_nan_frac": frac})
    diag.sort_values("non_nan_frac", inplace=True)
    path = outdir / f"sleeve_completeness_{tag}.csv"
    diag.to_csv(path)
    # Log consistently empty examples
    empty = diag[diag["non_nan_count"] == 0].head(10).index.tolist()
    if empty:
        log.info("Diagnostics %s: %d empty sleeves (showing up to 10): %s",
                 tag, (diag["non_nan_count"] == 0).sum(), empty)



def _run_allocation(sleeves_all: pd.DataFrame, methods, lookback_years: float,
                    outdir: Path, cleaning_cfg: dict):
    if sleeves_all.empty:
        log.warning("No sleeves for allocation.")
        return
    _diagnose_sparse_columns(sleeves_all, outdir, "ALL")
    lb_days = int(lookback_years * FREQ)
    month_ends = sleeves_all.resample("M").last().index

    for method in methods:
        log.info("[ALLOC:%s] rolling %.2f-year lookback ...", method, lookback_years)
        port_ret = pd.Series(np.nan, index=sleeves_all.index, dtype=float)
        first_alloc_date = None
        skipped = 0

        for i in range(len(month_ends) - 1):
            end = month_ends[i]
            start = end - pd.tseries.offsets.BDay(lb_days)
            next_end = month_ends[i + 1]

            window = sleeves_all.loc[(sleeves_all.index > start) & (sleeves_all.index <= end)]
            if window.shape[0] < 25:
                skipped += 1
                continue

            w_series = allocate(window, method=method, **cleaning_cfg)
            if (w_series.abs().sum() == 0) or w_series.isna().all():
                skipped += 1
                continue

            mask = (sleeves_all.index > end) & (sleeves_all.index <= next_end)
            seg = sleeves_all.loc[mask]
            if seg.empty:
                continue

            weights_aligned = w_series.reindex(seg.columns, fill_value=0.0)
            port_ret.loc[mask] = (seg.fillna(0.0) * weights_aligned).sum(axis=1)

            if first_alloc_date is None:
                first_alloc_date = seg.index.min()
                log.info("[ALLOC:%s] first weights applied start=%s window_rows=%d cols=%d",
                         method, first_alloc_date.date(), window.shape[0], window.shape[1])

        if skipped:
            log.info("[ALLOC:%s] skipped windows=%d (insufficient rows or cleaning failure)", method, skipped)

        # Trim leading NaNs (before first allocation) for equity calc
        ret_series = port_ret.copy()
        if first_alloc_date is not None:
            ret_series = ret_series[ret_series.index >= first_alloc_date]

        equity = (1 + ret_series.fillna(0.0)).cumprod()
        if not equity.empty:
            equity /= equity.iloc[0]

        _save_series(outdir / f"equity_alloc_{method}.csv", equity, "equity")

        stats = _perf_stats(ret_series)
        (outdir / f"summary_alloc_{method}.txt").write_text(
            f"Method: {method}\nFirstAlloc: {first_alloc_date}\nSkippedWindows: {skipped}\n"
            f"AnnRet: {stats['ann_return']:.4%}\nAnnVol: {stats['ann_vol']:.4%}\n"
            f"Sharpe: {stats['sharpe']:.2f}\nMaxDD: {stats['max_drawdown']:.2%}\n"
        )
        log.info("[ALLOC:%s] first_alloc=%s Sharpe=%.2f", method,
                 first_alloc_date.date() if first_alloc_date else None, stats["sharpe"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--lookback-years", type=float, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--verbose-clean", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root / "config" / "config.toml")
    results_dir = Path(cfg["reporting"]["outdir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or cfg["models"]["use"]
    lookback_years = args.lookback_years if args.lookback_years is not None else cfg["allocation"]["lookback_years"]
    methods = args.methods or cfg["allocation"]["methods"]
    cleaning_cfg = _extract_cleaning_cfg(cfg)
    if args.verbose_clean:
        cleaning_cfg["verbose_clean"] = True

    all_sleeves = []
    for m in models:
        log.info("[%s] loading priced grid ...", m)
        priced = _load_priced_grid(results_dir, m)
        sleeves, ew_ret = _simulate_sleeves(priced, cfg)
        # Diagnostics per model
        _diagnose_sparse_columns(sleeves, results_dir, m)
        sleeves.to_csv(results_dir / f"sleeve_returns_{m}.csv", index_label="date")
        eq = (1 + ew_ret.fillna(0.0)).cumprod()
        _save_series(results_dir / f"equity_{m}.csv", eq, "equity")
        st = _perf_stats(ew_ret)
        (results_dir / f"summary_{m}.txt").write_text(
            f"Model: {m}\nAnnRet: {st['ann_return']:.4%}\nAnnVol: {st['ann_vol']:.4%}\n"
            f"Sharpe: {st['sharpe']:.2f}\nMaxDD: {st['max_drawdown']:.2%}\n"
        )
        log.info("[%s] Sharpe=%.2f", m, st["sharpe"])
        sleeves_pref = sleeves.copy()
        sleeves_pref.columns = [f"{m}|{c[0]}|{c[1]}|{c[2]}|{c[3]}" for c in sleeves_pref.columns]
        all_sleeves.append(sleeves_pref)

    if all_sleeves:
        sleeves_all = pd.concat(all_sleeves, axis=1).sort_index()
        _run_allocation(sleeves_all, methods, lookback_years, results_dir, cleaning_cfg)
    else:
        log.warning("No sleeves aggregated; allocation skipped.")


if __name__ == "__main__":
    main()