from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd

from config_io import load_config
from allocator import allocate
from pricing_black_gk import black_forward_delta
from costs import option_trade_cost, hedge_cost_from_pips

log = logging.getLogger("run_strategy")
FREQ = 252


# -------------------- IO HELPERS -------------------- #

def _load_priced_grid(results_dir: Path, model: str) -> pd.DataFrame:
    f = results_dir / f"priced_grid_{model}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing priced grid file {f}")
    df = pd.read_csv(f)
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    # Expected columns: date, pair, tenor, delta, side, pv, F, K, vol, T, DF, delta (model)
    needed = {"pair", "tenor", "delta", "side", "pv", "F", "K", "vol", "T", "DF"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{f} missing columns: {missing}")
    return df.sort_values(["date", "pair", "tenor", "delta", "side"]).reset_index(drop=True)


# -------------------- METRICS -------------------- #

def _perf_stats(returns: pd.Series, freq: int = FREQ) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    equity = (1 + r).cumprod()
    ann_ret = (1 + r.mean()) ** freq - 1
    ann_vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    roll_max = equity.cummax()
    mdd = float((equity / roll_max - 1).min())
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": mdd
    }


def _save_series(path: Path, series: pd.Series, name: str):
    series = series.copy()
    df = pd.DataFrame({"date": series.index, name: series.values})
    df.to_csv(path, index=False)


# -------------------- SLEEVE SIMULATION -------------------- #

def _simulate_sleeves(priced: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Produce per-(pair,tenor,delta,side) return series (sleeves) and an equal-weight sleeve portfolio.
    """
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

    # Mark-to-market PnL (assuming pv is value of long position)
    # pnl_t = -(pv_t - pv_{t-1}) so that increasing PV gives positive pnl? Validate sign.
    df["pv_diff"] = df.groupby(key)["pv"].diff()
    df["pnl"] = -(df["pv_diff"])
    df.loc[df["pv_diff"].isna(), "pnl"] = 0.0  # first observation no pnl

    # Premium turnover for transaction costs
    if trade_cost_on_diff:
        prem_turn = df.groupby(key)["pv"].diff().abs()
        # First line per sleeve treated as opening trade
        first_mask = prem_turn.isna()
        prem_turn[first_mask] = df.loc[first_mask, "pv"].abs()
    else:
        prem_turn = df["pv"].abs()

    # Use existing delta column if present; else compute model delta
    if "delta" in df.columns:
        abs_delta = df["delta"].abs()
    else:
        abs_delta = pd.Series(0.0, index=df.index)

    trade_cost_raw = option_trade_cost(prem_turn, abs_delta, wings_bps=bps_wng, atm_bps=bps_atm)
    df["trade_cost"] = np.nan_to_num(trade_cost_raw, nan=0.0) * cost_multiplier

    # Delta hedging costs
    if "delta" in df.columns:
        dchg = df.groupby(key)["delta"].diff().abs()
    else:
        dchg = pd.Series(0.0, index=df.index)

    notional_change = np.nan_to_num(dchg) * df["F"].abs()
    is_major = df["pair"].str.upper().str.contains("USD")
    pips_cost = np.where(is_major, pips_maj, pips_xcc)
    hedge_cost_raw = hedge_cost_from_pips(notional_change, pips_cost)
    df["hedge_cost"] = np.where(dchg > delta_thr, hedge_cost_raw, 0.0) * cost_multiplier

    # Margin approximation (simple Vega * F * sqrt(T))
    # If explicit Vega not available, approximate with F * sqrt(T) * vol
    vol = np.maximum(df["vol"].values, 1e-8)
    T = np.maximum(df["T"].values, 1e-8)
    F = np.maximum(df["F"].values, 1e-8)
    approx_vega = F * np.sqrt(T) * vol  # crude scaling
    margin = np.maximum(kappa * np.abs(approx_vega), floor)

    net_pnl = df["pnl"] - df["trade_cost"] - df["hedge_cost"]
    df["ret"] = net_pnl / np.where(margin > 0, margin, np.nan)

    # Sanitize returns: remove inf, clip tail to reduce explosions
    r = df["ret"].to_numpy()
    r[~np.isfinite(r)] = 0.0
    if np.count_nonzero(r) > 30:
        lo, hi = np.nanpercentile(r, [1, 99])
        r = np.clip(r, lo, hi)
    df["ret"] = r

    sleeves = (
        df[["date"] + key + ["ret"]]
        .dropna(subset=["ret"])
        .pivot_table(index="date", columns=key, values="ret", aggfunc="first")
        .sort_index()
    )

    # Equal-weight daily across available sleeves
    avail = sleeves.notna().astype(float)
    weights = avail.div(avail.sum(axis=1), axis=0).fillna(0.0)
    ew_port = (sleeves.fillna(0.0) * weights).sum(axis=1)

    # Log average cost ratios for diagnostics
    avg_tc = df["trade_cost"].mean()
    avg_hc = df["hedge_cost"].mean()
    avg_abs_pnl = df["pnl"].abs().mean()
    if avg_abs_pnl > 0:
        log.info("Avg trade cost/abs_pnl=%.4f, hedge cost/abs_pnl=%.4f",
                 avg_tc / avg_abs_pnl, avg_hc / avg_abs_pnl)
    return sleeves, ew_port


# -------------------- ALLOCATION -------------------- #

def _run_allocation(sleeves_all: pd.DataFrame,
                    methods: List[str],
                    lookback_years: float,
                    outdir: Path):
    if sleeves_all.empty:
        log.warning("No sleeves available for allocation.")
        return
    lb_days = int(lookback_years * FREQ)
    month_ends = sleeves_all.resample("M").last().index
    for method in methods:
        log.info("[ALLOC:%s] rolling %.2f-year lookback ...", method, lookback_years)
        port_ret = pd.Series(0.0, index=sleeves_all.index, dtype=float)
        for i in range(len(month_ends) - 1):
            end = month_ends[i]
            start = end - pd.tseries.offsets.BDay(lb_days)
            next_end = month_ends[i + 1]
            window = sleeves_all.loc[(sleeves_all.index > start) & (sleeves_all.index <= end)]
            if window.empty or window.shape[0] < 40:
                continue
            # Ensure DataFrame passed
            w_series = allocate(window, method=method)
            # Forward apply until next month end
            seg_mask = (sleeves_all.index > end) & (sleeves_all.index <= next_end)
            seg = sleeves_all.loc[seg_mask]
            if seg.empty:
                continue
            weights_aligned = w_series.reindex(seg.columns, fill_value=0.0)
            port_ret.loc[seg_mask] = (seg.fillna(0.0) * weights_aligned).sum(axis=1)
        stats = _perf_stats(port_ret)
        eq = (1 + port_ret.fillna(0.0)).cumprod()
        _save_series(outdir / f"equity_alloc_{method}.csv", eq, "equity")
        (outdir / f"summary_alloc_{method}.txt").write_text(
            f"Method: {method}\n"
            f"AnnRet: {stats['ann_return']:.4%}\n"
            f"AnnVol: {stats['ann_vol']:.4%}\n"
            f"Sharpe: {stats['sharpe']:.2f}\n"
            f"MaxDD: {stats['max_drawdown']:.2%}\n"
        )
        log.info("[ALLOC:%s] Sharpe=%.2f", method, stats["sharpe"])


# -------------------- MAIN -------------------- #

def main():
    parser = argparse.ArgumentParser(description="Run strategy simulation and allocation.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models to run.")
    parser.add_argument("--lookback-years", type=float, default=None, help="Allocation lookback in years.")
    parser.add_argument("--methods", nargs="*", default=None, help="Allocation methods override.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root / "config" / "config.toml")
    results_dir = Path(cfg["reporting"]["outdir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or cfg["models"]["use"]
    lookback_years = args.lookback_years if args.lookback_years is not None else cfg["allocation"]["lookback_years"]
    methods = args.methods or cfg["allocation"]["methods"]

    all_sleeves = []

    for m in models:
        log.info("[%s] loading priced grid ...", m)
        priced = _load_priced_grid(results_dir, m)
        sleeves, ew_ret = _simulate_sleeves(priced, cfg)

        sleeves.to_csv(results_dir / f"sleeve_returns_{m}.csv", index_label="date")
        eq = (1 + ew_ret.fillna(0.0)).cumprod()
        _save_series(results_dir / f"equity_{m}.csv", eq, "equity")

        stats = _perf_stats(ew_ret)
        (results_dir / f"summary_{m}.txt").write_text(
            f"Model: {m}\n"
            f"AnnRet: {stats['ann_return']:.4%}\n"
            f"AnnVol: {stats['ann_vol']:.4%}\n"
            f"Sharpe: {stats['sharpe']:.2f}\n"
            f"MaxDD: {stats['max_drawdown']:.2%}\n"
        )
        log.info("[%s] Sharpe=%.2f", m, stats["sharpe"])

        # Flatten multi-index style to unique column labels
        sleeves_named = sleeves.copy()
        sleeves_named.columns = [
            f"{m}|{c[0]}|{c[1]}|{c[2]}|{c[3]}" for c in sleeves_named.columns
        ]
        all_sleeves.append(sleeves_named)

    if all_sleeves:
        sleeves_all = pd.concat(all_sleeves, axis=1).sort_index()
        _run_allocation(sleeves_all, methods, lookback_years, results_dir)
    else:
        log.warning("No sleeves aggregated; skipping allocation.")


if __name__ == "__main__":
    main()