# File: src/backtest.py
from __future__ import annotations
import pandas as pd, numpy as np, logging
from pathlib import Path
from config_io import load_config
import data_io as dio
from forwards import forward_from_points
from delta_strike import strike_from_delta
from pricing_black_gk import black_price, gk_price
from smile import vols_from_atm_rr_bf
from pricing_sabr import sabr_vol, calibrate_sabr
from pricing_vgvv import vgvv_implied_vol

log = logging.getLogger("backtest")


def _ensure_datetime_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col in df.columns:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame needs a DatetimeIndex or column: %s" % date_col)
    return df.sort_index()


def _df_for_T(curve_day, T_years):
    i = (curve_day["tenor_years"] - T_years).abs().idxmin()
    return float(curve_day.loc[i, "discount_factor"])


# --- Helpers to mitigate data mismatches ---

def _tenor_aliases(t: str) -> list[str]:
    """Return tenor aliases to try (primary first), to handle 3M vs 90D etc."""
    t = str(t).upper()
    mapping = {
        "1W": ["1W", "7D"],
        "2W": ["2W", "14D"],
        "3W": ["3W", "21D"],
        "1M": ["1M", "30D"],
        "2M": ["2M", "60D"],
        "3M": ["3M", "90D"],
        "6M": ["6M", "180D"],
        "9M": ["9M", "270D"],
        "12M": ["12M", "360D", "1Y"],
    }
    return mapping.get(t, [t])


def _rev_pair(pair: str) -> str:
    p = pair.upper()
    if "/" in p:
        a, b = p.split("/")
    else:
        a, b = p[:3], p[3:]
    return f"{b}{a}" if "/" not in p else f"{b}/{a}"


def _first_non_nan_from_aliases(row: pd.Series, getter, pair: str, tenor: str):
    """Try primary tenor then aliases with provided getter (dio.*_col)."""
    for ta in _tenor_aliases(tenor):
        col = getter(pair, ta)
        val = row.get(col)
        if pd.notna(val):
            return val, ta
    return np.nan, None


def daily_surface_for_pair(pair, row, tenors, day_count):
    out = {}
    for t in tenors:
        T = dio.tenor_years(t, day_count)

        # Spot and forward: try direct, else try reverse pair and invert
        S = row.get(dio.spot_col(pair))
        points, used_tenor_forw = _first_non_nan_from_aliases(row, dio.fwd_points_col, pair, t)

        if pd.isna(S) or pd.isna(points):
            # Try reversed pair (e.g., CADUSD instead of USDCAD)
            rev = _rev_pair(pair)
            S_rev = row.get(dio.spot_col(rev))
            points_rev, used_tenor_rev = _first_non_nan_from_aliases(row, dio.fwd_points_col, rev, t)
            if pd.notna(S_rev) and pd.notna(points_rev):
                try:
                    F_rev = forward_from_points(S_rev, points_rev)
                    if isinstance(F_rev, (int, float)) and F_rev > 0 and isinstance(S_rev, (int, float)) and S_rev > 0:
                        # Invert
                        S = 1.0 / float(S_rev)
                        F = 1.0 / float(F_rev)
                    else:
                        F = np.nan
                except Exception:
                    F = np.nan
            else:
                F = np.nan
        else:
            F = forward_from_points(S, points)

        if pd.isna(S) or pd.isna(F):
            continue
        if not (isinstance(F, (int, float)) and F > 0):
            continue

        # Vols (ATM + smile): try aliases; ATM vol assumed invariant under inversion
        atm = row.get(dio.atm_vol_col(pair, t))
        if pd.isna(atm):
            # try tenor alias on same pair
            for ta in _tenor_aliases(t):
                atm = row.get(dio.atm_vol_col(pair, ta))
                if pd.notna(atm):
                    break
        if pd.isna(atm):
            # try reverse pair as last resort (ATM vol ~ invariant under inversion)
            rev = _rev_pair(pair)
            for ta in _tenor_aliases(t):
                atm = row.get(dio.atm_vol_col(rev, ta))
                if pd.notna(atm):
                    break

        # RR/BF (if missing, we will gracefully fall back to ATM-only elsewhere)
        rr25 = row.get(dio.rr_col(pair, 25, t))
        bf25 = row.get(dio.bf_col(pair, 25, t))
        rr10 = row.get(dio.rr_col(pair, 10, t))
        bf10 = row.get(dio.bf_col(pair, 10, t))

        # Also try aliases for RR/BF if primary missing
        if pd.isna(rr25) or pd.isna(bf25):
            for ta in _tenor_aliases(t):
                rr25 = row.get(dio.rr_col(pair, 25, ta)) if pd.isna(rr25) else rr25
                bf25 = row.get(dio.bf_col(pair, 25, ta)) if pd.isna(bf25) else bf25
                if pd.notna(rr25) and pd.notna(bf25):
                    break
        if pd.isna(rr10) or pd.isna(bf10):
            for ta in _tenor_aliases(t):
                rr10 = row.get(dio.rr_col(pair, 10, ta)) if pd.isna(rr10) else rr10
                bf10 = row.get(dio.bf_col(pair, 10, ta)) if pd.isna(bf10) else bf10
                if pd.notna(rr10) and pd.notna(bf10):
                    break

        out[t] = {
            "S": S,
            "F": F,
            "T": T,
            "atm": atm,
            "25": vols_from_atm_rr_bf(atm, rr25, bf25) if pd.notna(rr25) and pd.notna(bf25) else (np.nan, np.nan),
            "10": vols_from_atm_rr_bf(atm, rr10, bf10) if pd.notna(rr10) and pd.notna(bf10) else (np.nan, np.nan)
        }
    return out


def price_by_model(model, F, K, T, DF, vols_bundle, beta=1.0, call: bool = True):
    if model == "BLACK":
        sigma = vols_bundle["baseline_vol"]
        pv = black_price(F, K, sigma, T, df=DF, call=call)
        return pv, sigma

    elif model == "GK":
        sigma = vols_bundle["baseline_vol"]
        S = vols_bundle["S"]  # spot must be present in vols_bundle
        pv = gk_price(S, K, sigma, T, DF, F, call=call)
        return pv, sigma

    elif model == "SABR":
        p = vols_bundle["sabr_params"]
        sigma = sabr_vol(F, K, T, p["alpha"], p["beta"], p["nu"], p["rho"])
        pv = black_price(F, K, sigma, T, df=DF, call=call)
        return pv, sigma

    elif model == "VGVV":
        sigma = vgvv_implied_vol(F, K, T,
                                 vols_bundle["anchors"],
                                 vols_bundle["anchor_vols"])
        pv = black_price(F, K, sigma, T, df=DF, call=call)
        return pv, sigma

    else:
        raise ValueError(f"Unknown model {model}")


def run(config_path: str, model_scope=None, outdir=None):
    cfg = load_config(config_path)

    # Ensure DateTimeIndex
    fxo = dio.load_fxo(cfg["data"]["fxo_path"])
    fxo = _ensure_datetime_index(fxo, cfg["data"]["date_col"])

    curve = dio.load_curve(cfg["data"]["curve_path"])
    curve = _ensure_datetime_index(curve, "date" if "date" in curve.columns else cfg["data"]["date_col"])

    tenors = cfg["tenors"]["list"]
    pairs = cfg["pairs"]["universe"]
    abs_d = cfg["target_deltas"]["abs"]
    beta = cfg["sabr"]["beta"]
    day_count = cfg["conventions"]["day_count"]
    if model_scope is None:
        model_scope = cfg["models"]["use"]

    records = []
    curve_by_date = {d: g.reset_index(drop=True) for d, g in curve.groupby(level=0)}
    dates = fxo.index.unique().sort_values()
    total = len(dates)

    for idx, (date, day) in enumerate(fxo.groupby(level=0)):
        if idx % 20 == 0:
            log.info("processing %d/%d dates ... %s", idx + 1, total, date.date())
        if date not in curve_by_date:
            continue
        curve_day = curve_by_date[date]
        for _, row in day.iterrows():
            for p in pairs:
                surf = daily_surface_for_pair(p, row, tenors, day_count)
                for t in tenors:
                    if t not in surf:
                        continue
                    S, F, T, atm = surf[t]["S"], surf[t]["F"], surf[t]["T"], surf[t]["atm"]
                    # Basic sanity checks
                    if any(pd.isna(x) for x in (S, F, T, atm)):
                        continue
                    if S <= 0 or F <= 0 or T <= 0 or atm <= 0:
                        continue

                    DF = _df_for_T(curve_day, T)
                    sc25, sp25 = surf[t]["25"]
                    baseline_vol = float(atm)

                    # Build anchors ONCE per tenor
                    K_atm = F
                    K_25c = strike_from_delta(F, 0.25, float(sc25) if pd.notna(sc25) and sc25 > 0 else baseline_vol, T,
                                              call=True)
                    K_25p = strike_from_delta(F, 0.25, float(sp25) if pd.notna(sp25) and sp25 > 0 else baseline_vol, T,
                                              call=False)
                    # Guard against non-positive strikes
                    if K_atm <= 0 or K_25c <= 0 or K_25p <= 0:
                        continue

                    anchors = (K_atm, K_25c, K_25p)
                    anchor_vols = (
                        baseline_vol,
                        float(sc25) if pd.notna(sc25) and sc25 > 0 else baseline_vol,
                        float(sp25) if pd.notna(sp25) and sp25 > 0 else baseline_vol
                    )

                    # SABR params only if needed
                    if "SABR" in model_scope:
                        sabr_params = calibrate_sabr(F, T, anchors, anchor_vols, beta=beta)
                        sabr_params["beta"] = beta
                    else:
                        sabr_params = None

                    vols_bundle = {
                        "S": S,
                        "baseline_vol": baseline_vol,
                        "anchors": anchors,
                        "anchor_vols": anchor_vols,
                        "sabr_params": sabr_params,
                    }

                    for d_abs in abs_d:
                        for side in ("call", "put"):
                            call = (side == "call")
                            K = strike_from_delta(F, d_abs, baseline_vol, T, call=call)
                            if K <= 0:
                                continue
                            for model in model_scope:
                                pv, used_vol = price_by_model(model, F, K, T, DF, vols_bundle, beta=beta)
                                if not np.isfinite(pv):
                                    continue
                                records.append({
                                    "date": date, "pair": p, "tenor": t, "delta": d_abs, "side": side, "model": model,
                                    "F": F, "K": K, "T": T, "DF": DF, "vol": used_vol, "pv": pv
                                })

    df = pd.DataFrame.from_records(records)
    if outdir is None:
        outdir = cfg["reporting"]["outdir"]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    return df
