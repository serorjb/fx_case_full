from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from config_io import load_config
from allocator import allocate  # PyPortfolioOpt-backed
from costs import option_trade_cost, hedge_cost_from_pips
from pricing_black_gk import black_forward_delta, gk_spot_delta

log = logging.getLogger("run_strategy")
FREQ = 252

# ---------- Loading helpers ----------

def _load_priced_grid(results_dir: Path, model: str) -> pd.DataFrame:
    """
    Load priced grid for a model.
    Expects file: priced_grid_<MODEL>.csv
    Robust to missing 'date' header (will treat first column as date).
    """
    f = results_dir / f"priced_grid_{model}.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}. Run pricing first.")
    try:
        df = pd.read_csv(f, parse_dates=["date"])
    except ValueError:
        # Fallback: read raw and rename first column to 'date'
        raw = pd.read_csv(f)
        first = raw.columns[0]
        if "date" not in raw.columns:
            raw = raw.rename(columns={first: "date"})
        raw["date"] = pd.to_datetime(raw["date"])
        df = raw
    if "date" not in df.columns:
        raise ValueError(f"No 'date' column in {f}")
    if df.empty:
        raise RuntimeError(f"{f} is empty")
    return df.sort_values(["date", "pair", "tenor", "delta", "side"])

def _read_sleeves_csv(f: Path) -> pd.DataFrame:
    df = pd.read_csv(f)
    if "date" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()

# ---------- Analytics helpers ----------

def _simulate_sleeves(priced: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Constant-maturity short option sleeves with costs:
    - PnL_t = -(PV_t - PV_{t-1})
    - Costs:
        * Option trade cost when we (re)open the sleeve each day (daily roll).
        * Delta-hedge cost when |Δ_t - Δ_{t-1}| > threshold (per pair).
    - Return_t = (PnL_t - OptionCost_t - HedgeCost_t) / MarginProxy_t
      MarginProxy ≈ kappa * |vega| * sqrt(T) * F, with floor (from TOML if present).
    """
    import numpy as np
    from scipy.stats import norm

    # ---- config knobs (with safe defaults) ----
    kappa = float(cfg.get("margin", {}).get("kappa", 0.10))
    floor = float(cfg.get("margin", {}).get("floor", 0.01))
    delta_thr = float(cfg.get("hedging", {}).get("delta_threshold", 0.04))
    pips_maj = float(cfg.get("hedging", {}).get("spot_cost_pips_majors", 1.0))
    pips_xcc = float(cfg.get("hedging", {}).get("spot_cost_pips_crosses", 3.0))
    bps_atm = float(cfg.get("costs", {}).get("option_trade_bps_atm", 15))
    bps_wng = float(cfg.get("costs", {}).get("option_trade_bps_wings", 30))

    def _is_major(pair: str) -> bool:
        # Simple rule: majors if USD is in the pair name
        return "USD" in str(pair).upper()

    key = ["pair","tenor","delta","side"]
    df = priced.copy().sort_values(["date"]+key)

    # --- model Greeks used for scaling and hedge signal ---
    # Black forward-based vega: vega = DF * F * phi(d1) * sqrt(T)
    def _vega(F, K, sigma, T, DF):
        if T<=0 or sigma<=0 or F<=0 or K<=0 or DF<=0:
            return 0.0
        v = sigma*np.sqrt(T)
        d1 = (np.log(F/K)+0.5*v*v)/max(v, 1e-12)
        return float(DF * F * norm.pdf(d1) * np.sqrt(T))

    # Delta (forward convention), sign for call/put
    def _delta_row(r):
        call = (str(r["side"]).lower() == "call")
        try:
            # For forward-delta (Black-76):
            return float(black_forward_delta(r["F"], r["K"], r["vol"], r["T"], call=call))
            # If you prefer GK spot-delta instead, use this version (requires spot S & DF in your DF):
            # return float(gk_spot_delta(r["S"], r["K"], r["vol"], r["T"], r["DF"], r["F"], call=call))
        except Exception:
            return np.nan

    # --- base P&L: short premium mark-to-market ---
    df["pv_diff"] = df.groupby(key)["pv"].diff()
    df["pnl"] = -(df["pv_diff"])

    # --- trading cost: charged once per day when we roll (i.e., whenever there is a valid row) ---
    # Use absolute premium as size proxy; bps choice depends on target |delta|
    # Note: the sleeve's target abs-delta is in column 'delta'
    abs_premium = df["pv"].abs()
    abs_delta = df["delta"].abs()
    trade_cost = option_trade_cost(abs_premium, abs_delta, wings_bps=bps_wng, atm_bps=bps_atm)
    df["trade_cost"] = np.nan_to_num(trade_cost, nan=0.0, posinf=0.0, neginf=0.0)

    # --- hedge cost: when |Δ change| > threshold ---
    df["delta_model"] = df.apply(_delta_row, axis=1)
    dchg = df.groupby(key)["delta_model"].diff().abs()
    # notional proxy = |Δ change| * F  (per sleeve)
    notional = dchg * df["F"].abs()
    # pick pips by pair kind
    pips_cost = np.where(df["pair"].map(_is_major), pips_maj, pips_xcc)
    hedge_cost = hedge_cost_from_pips(np.nan_to_num(notional, nan=0.0), pips=np.nan_to_num(pips_cost, nan=0.0))
    # only apply when threshold crossed; else zero
    df["hedge_cost"] = np.where(dchg > delta_thr, np.nan_to_num(hedge_cost, nan=0.0), 0.0)

    # --- margin proxy for return scaling ---
    vega_vec = df.apply(lambda r: _vega(r["F"], r["K"], r["vol"], r["T"], r["DF"]), axis=1)
    margin_base = np.maximum(kappa * np.abs(vega_vec) * np.sqrt(np.maximum(df["T"], 1e-12)) * np.maximum(df["F"], 1e-12), floor)

    # --- returns with costs ---
    net_pnl = df["pnl"] - df["trade_cost"] - df["hedge_cost"]
    df["ret"] = net_pnl / np.where(margin_base>0, margin_base, np.nan)

    # Clean & winsorize
    x = df["ret"].to_numpy()
    x[~np.isfinite(x)] = 0.0
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    x = np.clip(x, lo, hi)
    df["ret"] = x

    # Build sleeve panel and equal-weight diagnostic portfolio
    sleeves = (df.dropna(subset=["ret"])
                 .pivot_table(index="date", columns=key, values="ret", aggfunc="first")
                 .sort_index())

    w = (sleeves.notna()).astype(float)
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
    port = (sleeves.fillna(0.0) * w).sum(axis=1)
    return sleeves, port


def _perf_stats(returns: pd.Series, freq: int = FREQ) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    eq = (1 + r).cumprod()
    ann_ret = (1 + r.mean()) ** freq - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    roll_max = eq.cummax()
    mdd = float((eq / roll_max - 1.0).min())
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": mdd
    }

def _save_series(path: Path, series: pd.Series, name: str = "value"):
    df = pd.DataFrame({"date": series.index, name: series.values})
    df.to_csv(path, index=False)

# ---------- Allocation ----------

def _run_allocation(sleeves_all: pd.DataFrame, methods: list[str], lookback_years: float, outdir: Path):
    if sleeves_all.empty:
        log.warning("No sleeves for allocation.")
        return
    lb_days = int(lookback_years * FREQ)
    monthly = sleeves_all.resample("M").last().index
    for method in methods:
        log.info("[ALLOC:%s] rolling %.2f-year lookback ...", method, lookback_years)
        port_ret = pd.Series(0.0, index=sleeves_all.index)
        for i, dt in enumerate(monthly[:-1]):
            end = dt
            next_end = monthly[i + 1]
            start = end - pd.tseries.offsets.BDay(lb_days)
            window = sleeves_all.loc[(sleeves_all.index > start) & (sleeves_all.index <= end)].fillna(0.0)
            if len(window) < 60:
                continue
            w = allocate(window.to_numpy(), method=method)
            mask = (sleeves_all.index > end) & (sleeves_all.index <= next_end)
            seg = sleeves_all.loc[mask].fillna(0.0)
            if not seg.empty:
                port_ret.loc[mask] = (seg * w).sum(axis=1)
        eq = (1 + port_ret.fillna(0.0)).cumprod()
        _save_series(outdir / f"equity_alloc_{method}.csv", eq, name="equity")
        st = _perf_stats(port_ret)
        (outdir / f"summary_alloc_{method}.txt").write_text(
            f"Method: {method}\nAnnRet: {st['ann_return']:.4%}\nAnnVol: {st['ann_vol']:.4%}\n"
            f"Sharpe: {st['sharpe']:.2f}\nMaxDD: {st['max_drawdown']:.2%}\n"
        )
        log.info("[ALLOC:%s] Sharpe=%.2f", method, st["sharpe"])

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Run strategy & allocation on priced grids.")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of models (default: config)")
    parser.add_argument("--lookback-years", type=float, default=None, help="Allocation lookback in years (default: config)")
    parser.add_argument("--methods", nargs="*", default=None, help="Allocation methods (default: config)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(project_root / "config" / "config.toml")
    results_dir = Path(cfg["reporting"]["outdir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    models = args.models if args.models else cfg["models"]["use"]
    lookback_years = args.lookback_years if args.lookback_years is not None else cfg["allocation"]["lookback_years"]
    methods = args.methods if args.methods else cfg["allocation"]["methods"]

    all_sleeves = []

    # Per-model sleeves & stats
    for m in models:
        log.info("[%s] loading priced grid ...", m)
        priced = _load_priced_grid(results_dir, m)
        sleeves, ew_ret = _simulate_sleeves(priced)

        # Save sleeves (index label ensures later parse)
        sleeves.to_csv(results_dir / f"sleeve_returns_{m}.csv", index_label="date")

        # Equity curve
        eq = (1 + ew_ret.fillna(0.0)).cumprod()
        _save_series(results_dir / f"equity_{m}.csv", eq, name="equity")

        # Stats
        st = _perf_stats(ew_ret)
        (results_dir / f"summary_{m}.txt").write_text(
            f"Model: {m}\nAnnRet: {st['ann_return']:.4%}\nAnnVol: {st['ann_vol']:.4%}\n"
            f"Sharpe: {st['sharpe']:.2f}\nMaxDD: {st['max_drawdown']:.2%}\n"
        )
        log.info("[%s] Sharpe=%.2f", m, st["sharpe"])

        # Prepare for cross-model allocation (prefix columns)
        sleeves_pref = sleeves.copy()
        sleeves_pref.columns = [f"{m}|{c[0]}|{c[1]}|{c[2]}|{c[3]}" for c in sleeves_pref.columns]
        all_sleeves.append(sleeves_pref)

    # Cross-model allocation
    if all_sleeves:
        sleeves_all = pd.concat(all_sleeves, axis=1).sort_index()
        _run_allocation(sleeves_all, methods, lookback_years, results_dir)
    else:
        log.warning("No sleeves aggregated; allocation skipped.")

if __name__ == "__main__":
    main()
