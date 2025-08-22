"""
vgvv_backtest.py

Backtest analogous to sabr_backtest but using the EnhancedVolatilityArbitrageStrategy's
ultra_enhanced_vgvv_model to generate model vols (instead of SABR) for each delta point
obtained from smile replication (ATM, RR, BF inputs). Identifies overpriced options
(market_vol - model_vol > threshold) and systematically sells them.

Ensures risk-free rates parquet exists (rates.parquet) before running.
Supports two allocation modes: 'return' and 'sortino', with optional PyPortfolioOpt optimization.
Tracks Greeks, tenor weights, and outputs CSV/plots similar to SABR module for side-by-side comparison.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import os
os.environ.setdefault('CVXPY_SUPPRESS_WARNINGS','1')

from pricing_models import BlackScholesFX, VGVVModel
from smile_replication import build_and_calibrate_smile, TENOR_DAYS, build_delta_vol_dict, interpolate_vol
from rates import get_domestic_foreign_rates, ensure_rates
from trading_strategy import EnhancedVolatilityArbitrageStrategy, ENHANCED_CONFIG
from tearsheet import save_tearsheet  # NEW

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    _HAVE_PPO = True
except Exception:
    _HAVE_PPO = False

import matplotlib.pyplot as plt

TENORS = ["1W","2W","3W","1M","2M","3M","4M","6M","9M","1Y"]

@dataclass
class VGVVTrade:
    trade_id: int
    pair: str
    tenor: str
    delta: float
    strike: float
    option_type: str
    notional: float
    direction: int  # -1 short
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    entry_vol: float
    model_vol: float
    vol_edge: float
    entry_price: float
    premium: float
    domestic_rate: float
    foreign_rate: float
    spot_entry: float
    forward: float
    hedged: bool=False
    hedge_size: float=0.0
    hedge_cost: float=0.0
    exit_price: float=0.0
    pnl: float=0.0
    premium_net_received: float=0.0
    last_mark: float = 0.0
    hedge_last_mark: float = 0.0
    hedge_entry_spot: float = 0.0
    entry_cost: float = 0.0
    first_mtm_done: bool = False  # Track if first MTM has been done

class VGVVSmileBacktester:
    def __init__(self, loader, pairs: List[str], start_date, end_date,
                 initial_capital=10_000_000, vol_edge_threshold=0.0100,
                 bid_ask=0.0125, commission=0.0005, slippage=0.0002,
                 margin_rate=0.20, daily_capital_fraction=0.025,
                 max_notional=2_500_000, allocation_mode='return', seed=42,
                 use_moneyness_cost: bool = False,
                 bid_ask_wing_mult: float = 1.5,
                 report_start_date: str | None = '2007-01-01'):
        # Auto-discover pairs if 'ALL' passed
        if pairs is None or (isinstance(pairs, list) and len(pairs)==1 and pairs[0].upper()=='ALL'):
            fx_dir = Path('data/FX')
            auto = []
            if fx_dir.exists():
                for f in fx_dir.glob('*.parquet'):
                    nm = f.stem
                    if len(nm)<=10:
                        auto.append(nm)
            pairs = sorted(auto)
        self.all_pairs = pairs
        self.loader = loader
        self.pairs = pairs
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.vol_edge_threshold = vol_edge_threshold
        # Store components to allow moneyness-aware bid/ask
        self.bid_ask = bid_ask
        self.commission = commission
        self.slippage = slippage
        self.cost_rate = bid_ask + commission + slippage
        self.use_moneyness_cost = use_moneyness_cost
        self.bid_ask_wing_mult = bid_ask_wing_mult
        self.margin_rate = margin_rate
        self.daily_capital_fraction = daily_capital_fraction
        self.max_notional = max_notional
        self.allocation_mode = allocation_mode
        self.bs = BlackScholesFX()
        self.strategy = EnhancedVolatilityArbitrageStrategy(ENHANCED_CONFIG)
        self.rng = np.random.default_rng(seed)

        self.business_days = pd.bdate_range(self.start_date, self.end_date)
        self.open_trades: List[VGVVTrade] = []
        self.closed_trades: List[VGVVTrade] = []
        self.trade_counter = 0
        self.tenor_weights = {t: 1/len(TENORS) for t in TENORS}
        self.tenor_alloc_history = []
        self.tenor_daily_pnl = defaultdict(list)
        self.tenor_capital_history = {t: [] for t in TENORS}
        self.margin_used = 0.0
        self.daily_records = []
        self.daily_greeks = []
        self.daily_theta_est = []
        # Reporting window (burn-in)
        self.report_start = pd.Timestamp(report_start_date) if report_start_date else self.start_date
        # Per-pair distribution tracking
        self.pair_daily_new = []
        self.pair_daily_open = []

    def _price(self, spot, strike, T, vol, r_d, r_f, opt_type):
        try:
            return self.bs.price(strike, int(max(T*365,1)), vol, spot, r_d, r_f, opt_type)
        except Exception:
            return max((spot-strike) if opt_type=='call' else (strike-spot), 0)

    def _delta(self, spot, strike, T, vol, r_d, r_f, opt_type):
        try:
            return self.bs.delta(spot, strike, r_d, r_f, vol, T, opt_type)
        except Exception:
            return 0.0

    def _effective_cost_rate(self, delta: float) -> float:
        """Optional moneyness-aware transaction cost (disabled by default).
        Scales bid/ask with |delta-0.5|; commission & slippage flat.
        """
        if not self.use_moneyness_cost or delta is None or not np.isfinite(delta):
            return self.cost_rate
        wing_factor = min(1.0, max(0.0, 2.0 * abs(float(delta) - 0.5)))
        ba = self.bid_ask * (1.0 + self.bid_ask_wing_mult * wing_factor)
        return ba + self.commission + self.slippage

    def _optimize_weights(self, current_date):
        # Run at year end
        if current_date.month != 12 or current_date == self.business_days[-1]:
            return
        if (current_date + pd.Timedelta(days=1)).year == current_date.year:
            return
        window = 250
        returns_dict = {}
        for t in TENORS:
            pnl_list = self.tenor_daily_pnl.get(t, [])
            cap_hist = self.tenor_capital_history.get(t, [])
            if not pnl_list or not cap_hist:
                continue
            pnl_series = pd.Series(pnl_list[-window:])
            cap_series = pd.Series(cap_hist[-window:])
            n = min(len(pnl_series), len(cap_series))
            if n == 0:
                continue
            pnl_series = pnl_series.iloc[-n:].reset_index(drop=True)
            cap_series = cap_series.iloc[-n:].replace(0, np.nan).reset_index(drop=True)
            if cap_series.isna().all():
                continue
            ret_series = (pnl_series / cap_series).fillna(0.0)
            returns_dict[t] = ret_series
        if not returns_dict:
            return
        ret_df = pd.DataFrame(returns_dict)
        floor = 0.05
        weights = None
        if _HAVE_PPO and ret_df.shape[0] > 20:
            try:
                clean = ret_df.fillna(0.0)
                mu = expected_returns.mean_historical_return(clean, frequency=252)
                if self.allocation_mode == 'sortino':
                    cov = risk_models.semicovariance(clean, frequency=252)
                else:
                    cov = risk_models.sample_cov(clean, frequency=252)
                ef = EfficientFrontier(mu, cov, weight_bounds=(floor,1.0))
                if self.allocation_mode == 'sortino':
                    ef.max_sharpe()
                else:
                    try:
                        ef.max_quadratic_utility(risk_aversion=1e-6)
                    except Exception:
                        ef.max_sharpe()
                raw = ef.clean_weights()
                weights = {t: float(raw.get(t,0)) for t in TENORS}
            except Exception:
                weights = None
        if weights is None:
            mu_simple = ret_df.mean()*252
            if self.allocation_mode == 'sortino':
                downside = ret_df.clip(upper=0)**2
                semi = (downside.mean()**0.5)*(252**0.5)
                scores = mu_simple / semi.replace(0,np.nan)
            else:
                scores = mu_simple.clip(lower=0)
            scores = scores.replace([np.inf,-np.inf], np.nan).fillna(0)
            if scores.sum() == 0:
                scores = pd.Series([1]*len(scores), index=scores.index)
            w_raw = scores / scores.sum()
            weights = {t: float(w_raw.get(t,0)) for t in TENORS}
        # floor & renorm
        for k,v in list(weights.items()):
            if v>0: weights[k] = max(v, floor)
        # Add missing / zero weights at floor
        for t in TENORS:
            if t not in weights or weights[t] == 0:
                weights[t] = floor
        total = sum(weights.values())
        weights = {t: v/total for t,v in weights.items()}
        self.tenor_weights = weights
        self.tenor_alloc_history.append({'date': current_date, 'mode': self.allocation_mode, **self.tenor_weights})
        print(f"[REBALANCE-VGVV-{self.allocation_mode.upper()}] {current_date.date()} weights: "+", ".join(f"{k}:{v:.1%}" for k,v in self.tenor_weights.items()))

    def _open_trade(self, cand, date, tenor_capital):
        sp = cand['point']
        tenor = cand['tenor']
        T_days = TENOR_DAYS[tenor]
        T = T_days/365
        price = self._price(cand['spot'], sp.strike, T, sp.market_vol, cand['r_d'], cand['r_f'], cand['type'])
        if price <= 0: return None
        per_trade_capital = tenor_capital * self.daily_capital_fraction / 3.0
        if per_trade_capital <=0: return None
        notional = per_trade_capital / price
        notional = min(notional, self.max_notional)
        premium = price * notional
        # Use moneyness-aware cost if enabled
        eff_cost_rate = self._effective_cost_rate(getattr(sp, 'delta', np.nan))
        premium_net = premium * (1 - eff_cost_rate)
        entry_cost = premium * eff_cost_rate
        margin = cand['spot'] * notional * self.margin_rate
        if self.margin_used + margin > self.equity * 0.8:
            return None
        self.margin_used += margin
        tr = VGVVTrade(
            trade_id=self.trade_counter,
            pair=cand['pair'],
            tenor=tenor,
            delta=sp.delta,
            strike=sp.strike,
            option_type=cand['type'],
            notional=notional,
            direction=-1,
            entry_date=date,
            expiry_date=date + pd.Timedelta(days=T_days),
            entry_vol=sp.market_vol,
            model_vol=sp.model_vol,
            vol_edge=sp.vol_edge,
            entry_price=price,
            premium=premium,
            domestic_rate=cand['r_d'],
            foreign_rate=cand['r_f'],
            spot_entry=cand['spot'],
            forward=cand['forward'],
            premium_net_received=premium_net,
            last_mark=(-1) * notional * price,  # fair-value mark at entry (liability for short)
            hedge_last_mark=0.0,
            entry_cost=entry_cost
        )
        self.open_trades.append(tr)
        self.trade_counter += 1
        return tr

    def _hedge(self, tr: VGVVTrade, spot: float):
        T = max((tr.expiry_date - tr.entry_date).days,1)/365
        d = self._delta(spot, tr.strike, T, tr.entry_vol, tr.domestic_rate, tr.foreign_rate, tr.option_type)
        if abs(d)>0.04 and not tr.hedged:
            hedge_units = -d * tr.notional * tr.direction
            cost = abs(hedge_units) * spot * 0.0002
            self.equity -= cost
            tr.hedged = True
            tr.hedge_size = hedge_units
            tr.hedge_cost = cost
            tr.hedge_last_mark = tr.hedge_size * spot
            tr.hedge_entry_spot = spot

    def _value_trade(self, tr: VGVVTrade, date, spot):
        if date >= tr.expiry_date:
            intrinsic = max(spot - tr.strike,0) if tr.option_type=='call' else max(tr.strike - spot,0)
            # Realized PnL for reporting only (equity handled via MTM)
            pnl_realized = tr.premium_net_received - intrinsic * tr.notional
            if tr.hedged:
                pnl_realized += tr.hedge_size * (spot - tr.hedge_entry_spot) - tr.hedge_cost
            tr.exit_price = intrinsic
            tr.pnl = pnl_realized
            # Release margin at expiry
            try:
                self.margin_used -= tr.spot_entry * tr.notional * self.margin_rate
            except Exception:
                pass
            return pnl_realized
        return 0.0

    def _greeks(self, date):
        total_delta=total_gamma=total_vega=total_theta=0.0
        for tr in self.open_trades:
            if date >= tr.expiry_date: continue
            vol_data = self.loader.get_volatility_surface(tr.pair, date)
            if not vol_data: continue
            T = max((tr.expiry_date - date).days,1)/365
            try:
                atm = vol_data.atm_vols.get(tr.tenor)
                rr25 = vol_data.rr_25d.get(tr.tenor,0.0)
                bf25 = vol_data.bf_25d.get(tr.tenor,0.0)
                rr10 = vol_data.rr_10d.get(tr.tenor,0.0)
                bf10 = vol_data.bf_10d.get(tr.tenor,0.0)
                dvols = build_delta_vol_dict(atm, rr25, bf25, rr10, bf10)
                vol_now = interpolate_vol(dvols, tr.delta)
                g = self.bs.calculate_greeks(tr.strike, int(T*365), vol_now, vol_data.spot, tr.domestic_rate, tr.foreign_rate, tr.option_type)
                total_delta += g['delta'] * tr.direction * tr.notional * vol_data.spot
                total_gamma += g['gamma'] * tr.direction * tr.notional * vol_data.spot
                total_vega  += g['vega']  * tr.direction * tr.notional
                total_theta += g['theta'] * tr.direction * tr.notional
            except Exception:
                continue
        return {'delta':total_delta,'gamma':total_gamma,'vega':total_vega,'theta':total_theta}

    def run(self):
        ensure_rates()
        print(f"\n=== VGVV Backtest ({self.allocation_mode}) ===")
        print(f"Period: {self.start_date.date()} -> {self.end_date.date()}  Pairs: {', '.join(self.pairs)}")
        for date in self.business_days:
            self._optimize_weights(date)
            day_mtm_pnl = 0.0
            day_pnl_by_tenor = defaultdict(float)
            new_trades_counter = defaultdict(int)
            # MTM existing trades with current market vol; daily re-hedge
            for tr in list(self.open_trades):
                vd = self.loader.get_volatility_surface(tr.pair, date)
                if not vd: continue
                spot = vd.spot
                expired = date >= tr.expiry_date
                atm = vd.atm_vols.get(tr.tenor)
                rr25 = vd.rr_25d.get(tr.tenor,0.0)
                bf25 = vd.bf_25d.get(tr.tenor,0.0)
                rr10 = vd.rr_10d.get(tr.tenor,0.0)
                bf10 = vd.bf_10d.get(tr.tenor,0.0)
                dvols = build_delta_vol_dict(atm, rr25, bf25, rr10, bf10)
                vol_now = interpolate_vol(dvols, tr.delta)
                if not expired:
                    T = max((tr.expiry_date - date).days,1)/365
                    try:
                        price_now = self.bs.price(tr.strike, int(T*365), vol_now, spot, tr.domestic_rate, tr.foreign_rate, tr.option_type)
                    except Exception:
                        price_now = max((spot-tr.strike) if tr.option_type=='call' else (tr.strike-spot),0)
                else:
                    price_now = max((spot-tr.strike) if tr.option_type=='call' else (tr.strike-spot),0)
                current_mark = tr.direction * tr.notional * price_now

                # MTM: skip on entry day; otherwise compute change from last_mark
                if tr.entry_date == date:
                    delta_pnl = 0.0
                else:
                    delta_pnl = current_mark - tr.last_mark
                    tr.last_mark = current_mark
                day_mtm_pnl += delta_pnl
                day_pnl_by_tenor[tr.tenor] += delta_pnl
                # Hedge MTM
                if tr.hedged:
                    hedge_mark = tr.hedge_size * spot
                    hedge_delta = hedge_mark - tr.hedge_last_mark
                    day_mtm_pnl += hedge_delta
                    day_pnl_by_tenor[tr.tenor] += hedge_delta
                    tr.hedge_last_mark = hedge_mark
                    carry = tr.hedge_size * spot * (tr.domestic_rate - tr.foreign_rate) / 252.0
                    self.equity += carry
                    day_pnl_by_tenor[tr.tenor] += carry
                # Daily re-hedge to |delta|<=4%
                if not expired:
                    T = max((tr.expiry_date - date).days,1)/365
                    try:
                        d_now = self.bs.delta(spot, tr.strike, tr.domestic_rate, tr.foreign_rate, vol_now, T, tr.option_type)
                    except Exception:
                        d_now = 0.0
                    if abs(d_now) > 0.04:
                        desired_units = -d_now * tr.notional * tr.direction
                        dH = desired_units - tr.hedge_size
                        if abs(dH) > 1e-12:
                            cost = abs(dH) * spot * 0.0002
                            self.equity -= cost
                            day_pnl_by_tenor[tr.tenor] -= cost
                            tr.hedged = True
                            tr.hedge_size += dH
                            tr.hedge_cost += cost
                            tr.hedge_last_mark += dH * spot
                            tr.hedge_entry_spot = spot if tr.hedge_entry_spot == 0.0 else tr.hedge_entry_spot
                if expired:
                    self._value_trade(tr, date, spot)
                    self.open_trades.remove(tr)
                    self.closed_trades.append(tr)
            self.equity += day_mtm_pnl
            # Open new trades
            if len(self.open_trades) < 300:
                for tenor in TENORS:
                    tenor_cap = self.equity * self.tenor_weights[tenor]
                    self.tenor_capital_history[tenor].append(tenor_cap)
                    cands = []
                    for pair in self.pairs:
                        vd = self.loader.get_volatility_surface(pair, date)
                        if not vd or tenor not in vd.atm_vols: continue
                        atm = vd.atm_vols.get(tenor)
                        rr25 = vd.rr_25d.get(tenor,0.0)
                        bf25 = vd.bf_25d.get(tenor,0.0)
                        rr10 = vd.rr_10d.get(tenor,0.0)
                        bf10 = vd.bf_10d.get(tenor,0.0)
                        fwd = vd.forwards.get(tenor, vd.spot)
                        smile = build_and_calibrate_smile(pair, date, tenor, vd.spot, fwd, atm, rr25, bf25, rr10, bf10)
                        if not smile: continue
                        # Calibrate VGVV model to the reconstructed smile
                        T = TENOR_DAYS[tenor]/365
                        r_d, r_f = get_domestic_foreign_rates(date, T, vd.spot, fwd)
                        try:
                            vg = VGVVModel(vd.spot, fwd, r_d, r_f, T)
                            k_arr = np.array([pt.strike for pt in smile.points], dtype=float)
                            mv_arr = np.array([pt.market_vol for pt in smile.points], dtype=float)
                            params_vg = vg.calibrate(k_arr, mv_arr)
                        except Exception:
                            vg = None; params_vg = None
                        for p in smile.points:
                            try:
                                if vg and params_vg:
                                    model_vol = vg.vgvv_vol(p.strike, params_vg)
                                else:
                                    model_vol = max(p.market_vol * 0.9, 0.0001)
                                edge = p.market_vol - model_vol
                            except Exception:
                                continue
                            if edge > self.vol_edge_threshold:
                                opt_type = 'call' if p.delta >= 0.5 else 'put'
                                class _Point: pass
                                pt = _Point(); pt.delta=p.delta; pt.strike=p.strike; pt.market_vol=p.market_vol; pt.model_vol=model_vol; pt.vol_edge=edge
                                cands.append({'pair':pair,'tenor':tenor,'point':pt,'spot':vd.spot,'forward':fwd,'type':opt_type,'r_d':r_d,'r_f':r_f})
                    if not cands: continue
                    cands.sort(key=lambda x: x['point'].vol_edge, reverse=True)
                    for cd in cands[:3]:
                        tr = self._open_trade(cd, date, tenor_cap)
                        if tr:
                            # Book entry cash and liability so net is -entry_cost
                            self.equity += tr.premium_net_received + tr.last_mark
                            day_pnl_by_tenor[tr.tenor] += tr.premium_net_received + tr.last_mark
                            new_trades_counter[tr.pair] += 1
            else:
                for tenor in TENORS:
                    self.tenor_capital_history[tenor].append(self.equity * self.tenor_weights[tenor])
            # Financing cost on margin used (allocate by tenor share of margin); recompute margin using current spot when available
            margin_by_tenor = defaultdict(float)
            total_margin = 0.0
            for tr in self.open_trades:
                vd = self.loader.get_volatility_surface(tr.pair, date)
                cur_spot = vd.spot if vd else tr.spot_entry
                m = cur_spot * tr.notional * self.margin_rate
                margin_by_tenor[tr.tenor] += m
                total_margin += m
            self.margin_used = total_margin
            if total_margin > 0:
                from rates import get_rate
                fin_rate = get_rate(date, 1/12)
                fin_cost_total = - total_margin * fin_rate / 252.0
                self.equity += fin_cost_total
                for t, m in margin_by_tenor.items():
                    day_pnl_by_tenor[t] += fin_cost_total * (m/total_margin)
            # Record per-tenor daily PnL
            for t in TENORS:
                self.tenor_daily_pnl[t].append(day_pnl_by_tenor.get(t, 0.0))
            # Per-pair daily logs
            if new_trades_counter:
                for p, n in new_trades_counter.items():
                    self.pair_daily_new.append({'date': date, 'pair': p, 'new_trades': int(n)})
            open_counter = defaultdict(int)
            for tr in self.open_trades:
                open_counter[tr.pair] += 1
            for p, n in open_counter.items():
                self.pair_daily_open.append({'date': date, 'pair': p, 'open_trades': int(n)})
            g = self._greeks(date)
            theta_est = g.get('theta',0.0)
            self.daily_theta_est.append({'date':date,'theta_pnl_est':theta_est})
            cap_util = (self.margin_used / (self.equity*0.8)) if self.equity>0 else 0.0
            cap_util_initial = (self.margin_used / (self.initial_capital*0.8)) if self.initial_capital>0 else 0.0
            self.daily_greeks.append({'date':date, **g})
            self.daily_records.append({
                'date':date,
                'equity':self.equity,
                'open_trades':len(self.open_trades),
                'closed_trades':len(self.closed_trades),
                'margin_used':self.margin_used,
                'capacity_utilization': cap_util,
                'capacity_utilization_initial': cap_util_initial,
                'day_mtm_pnl':day_mtm_pnl,
                'theta_pnl_est':theta_est,
                **g
            })
            if (len(self.daily_records)%250==0) or date==self.business_days[-1]:
                ret=(self.equity-self.initial_capital)/self.initial_capital
                print(f"  {date.date()} | Eq ${self.equity/1e6:.2f}M Ret {ret:+.1%} Open {len(self.open_trades)} Closed {len(self.closed_trades)}")
        return self._finalize()

    def _finalize(self):
        df_daily = pd.DataFrame(self.daily_records).set_index('date')
        trades_df = pd.DataFrame(t.__dict__ for t in self.closed_trades)
        # Adjusted equity rebased at report_start
        eq = df_daily['equity'].copy()
        eq_vals = np.array([])
        if not eq.empty:
            pre = eq[eq.index < self.report_start]
            base = pre.iloc[-1] if len(pre)>0 else (eq.iloc[0] if len(eq)>0 else self.initial_capital)
            adj = eq.copy()
            adj.loc[adj.index >= self.report_start] = self.initial_capital + (eq.loc[eq.index >= self.report_start] - base)
            adj.loc[adj.index < self.report_start] = np.nan
            df_daily['equity_adjusted'] = adj
            post = adj.dropna()
            eq_vals = post.values
            rets = np.diff(eq_vals)/eq_vals[:-1] if len(eq_vals)>1 else np.array([])
        else:
            df_daily['equity_adjusted'] = eq
            rets = np.array([])
        # Drawdown on adjusted equity
        if 'equity_adjusted' in df_daily.columns and df_daily['equity_adjusted'].notna().any():
            adj_series = df_daily['equity_adjusted']
            roll_peak = adj_series.cummax()
            df_daily['drawdown'] = (adj_series - roll_peak) / roll_peak
        else:
            df_daily['drawdown'] = np.nan
        if len(rets)>0:
            tot_ret = (eq_vals[-1]-eq_vals[0])/eq_vals[0]
            ann_ret = (1+tot_ret)**(252/len(rets))-1
            ann_vol = rets.std()*np.sqrt(252)
            sharpe = ann_ret/ann_vol if ann_vol>0 else 0
            peak = np.maximum.accumulate(eq_vals)
            dd = (eq_vals-peak)/peak
            max_dd = dd.min() if len(dd)>0 else 0
        else:
            tot_ret=ann_ret=ann_vol=sharpe=max_dd=0.0
        # Tenor perf unchanged
        tenor_perf=[]
        for t in TENORS:
            pnl_series = pd.Series(self.tenor_daily_pnl.get(t,[]))
            cap_series = pd.Series(self.tenor_capital_history.get(t,[]))
            if pnl_series.empty: continue
            cum_pnl = pnl_series.cumsum().iloc[-1]
            avg_cap = cap_series.replace(0,np.nan).mean()
            ret_series = pnl_series/avg_cap if avg_cap and avg_cap>0 else pd.Series([0])
            ann_rtn = ret_series.mean()*252 if len(ret_series)>0 else 0
            ann_vol_t = ret_series.std()*np.sqrt(252) if ret_series.std()>0 else 0
            sharpe_t = ann_rtn/ann_vol_t if ann_vol_t>0 else 0
            tenor_perf.append({'tenor':t,'cum_pnl':cum_pnl,'ann_return':ann_rtn,'ann_vol':ann_vol_t,'sharpe':sharpe_t})
        tenor_perf_df = pd.DataFrame(tenor_perf)
        out_dir = Path('results'); out_dir.mkdir(exist_ok=True)
        suf = f"_{self.allocation_mode}"
        df_daily.to_csv(out_dir/f'vgvv_daily_equity{suf}.csv')
        trades_df.to_csv(out_dir/f'vgvv_trades{suf}.csv', index=False)
        tenor_perf_df.to_csv(out_dir/f'vgvv_tenor_performance{suf}.csv', index=False)
        # Save per-pair distributions
        if self.pair_daily_new:
            pd.DataFrame(self.pair_daily_new).to_csv(out_dir/f'vgvv_pair_new_trades{suf}.csv', index=False)
        if self.pair_daily_open:
            pd.DataFrame(self.pair_daily_open).to_csv(out_dir/f'vgvv_pair_open_trades{suf}.csv', index=False)
        if self.daily_greeks:
            pd.DataFrame(self.daily_greeks).to_csv(out_dir/f'vgvv_greeks{suf}.csv', index=False)
        if self.daily_theta_est:
            pd.DataFrame(self.daily_theta_est).to_csv(out_dir/f'vgvv_theta_pnl{suf}.csv', index=False)
        # Tearsheets & plots
        try:
            try:
                series_for_tearsheet = df_daily['equity_adjusted'].dropna() if 'equity_adjusted' in df_daily.columns else df_daily['equity']
                save_tearsheet(series_for_tearsheet, f'VGVV ({self.allocation_mode})', str(out_dir / 'tearsheets' / f'vgvv_{self.allocation_mode}.png'))
            except Exception:
                pass
            # Equity curve
            plt.figure(figsize=(12,5))
            if 'equity_adjusted' in df_daily.columns:
                plt.plot(df_daily.index, df_daily['equity_adjusted']/1e6, label='Equity (Adj)')
            else:
                plt.plot(df_daily.index, df_daily['equity']/1e6, label='Equity')
            plt.title('VGVV Equity Curve'); plt.ylabel('Equity (MM)'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'vgvv_equity_curve.png', dpi=200); plt.close()
            # Drawdown
            if 'drawdown' in df_daily.columns and df_daily['drawdown'].notna().any():
                plt.figure(figsize=(12,3))
                plt.plot(df_daily.index, df_daily['drawdown'], color='firebrick')
                plt.title('VGVV Drawdown (Adj)'); plt.ylabel('Drawdown'); plt.tight_layout(); plt.savefig(out_dir/'vgvv_drawdown.png', dpi=200); plt.close()
            # Capacity utilization
            if 'capacity_utilization' in df_daily.columns:
                plt.figure(figsize=(12,3))
                plt.plot(df_daily.index, df_daily['capacity_utilization'], color='slateblue', label='vs current equity')
                if 'capacity_utilization_initial' in df_daily.columns:
                    plt.plot(df_daily.index, df_daily['capacity_utilization_initial'], color='darkorange', alpha=0.8, label='vs initial equity')
                plt.title('VGVV Capacity Utilization (Margin / 80% Equity)'); plt.ylabel('Utilization'); plt.legend(loc='upper right'); plt.tight_layout(); plt.savefig(out_dir/f"vgvv_capacity_utilization{suf}.png", dpi=200); plt.close()
            # Tenor weights
            if self.tenor_alloc_history:
                wdf = pd.DataFrame(self.tenor_alloc_history).set_index('date'); wdf.plot(figsize=(12,5)); plt.title('VGVV Tenor Weights'); plt.tight_layout(); plt.savefig(out_dir/'vgvv_tenor_weights.png', dpi=200); plt.close()
            # Greeks
            if self.daily_greeks:
                gdf = pd.DataFrame(self.daily_greeks).set_index('date'); plt.figure(figsize=(12,6))
                for c in ['delta','gamma','vega','theta']:
                    if c in gdf.columns: plt.plot(gdf.index, gdf[c], label=c)
                plt.legend(); plt.title(f'VGVV Greeks ({self.allocation_mode})'); plt.tight_layout(); plt.savefig(out_dir/f'vgvv_greeks{suf}.png', dpi=200); plt.close()
            # Pair distributions over time
            if self.pair_daily_open:
                pdf = pd.DataFrame(self.pair_daily_open)
                pivot_open = pdf.pivot_table(index='date', columns='pair', values='open_trades', aggfunc='sum').fillna(0)
                pivot_open.plot(figsize=(12,6), lw=1)
                plt.title('Open Trades by Pair Over Time'); plt.ylabel('Open Trades'); plt.tight_layout(); plt.savefig(out_dir/'vgvv_pair_open_timeseries.png', dpi=200); plt.close()
            if self.pair_daily_new:
                ndf = pd.DataFrame(self.pair_daily_new)
                pivot_new = ndf.pivot_table(index='date', columns='pair', values='new_trades', aggfunc='sum').fillna(0)
                pivot_new_7d = pivot_new.rolling(5, min_periods=1).sum()
                pivot_new_7d.plot(figsize=(12,6), lw=1)
                plt.title('New Trades by Pair (5-day rolling)'); plt.ylabel('New Trades'); plt.tight_layout(); plt.savefig(out_dir/'vgvv_pair_new_trades_timeseries.png', dpi=200); plt.close()
        except Exception:
            pass
        print(f"\n=== VGVV Summary ({self.allocation_mode}) ===")
        final_eq_adj = float(df_daily['equity_adjusted'].dropna().iloc[-1]) if 'equity_adjusted' in df_daily.columns and not df_daily['equity_adjusted'].dropna().empty else self.equity
        print(f"Total Return: {tot_ret:+.1%} AnnReturn: {ann_ret:+.1%} Sharpe: {sharpe:.2f} MaxDD: {max_dd:.1%} Closed: {len(self.closed_trades)} FinalEq: ${final_eq_adj:,.0f}")
        if not tenor_perf_df.empty:
            rows = tenor_perf_df.to_dict('records')
            print('Tenor Performance: '+', '.join(f"{r['tenor']}:{r['cum_pnl']:,.0f}" for r in rows))
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_eq_adj,
            'total_return': float(tot_ret),
            'annualized_return': float(ann_ret),
            'annualized_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'num_closed_trades': len(self.closed_trades),
            'allocation_mode': self.allocation_mode,
            'tenor_weights_final': self.tenor_weights
        }

if __name__ == '__main__':
    from data_loader import FXDataLoader
    ensure_rates()
    loader = FXDataLoader()
    pairs = ['ALL']
    # Start 3 months earlier; report from 2007-01-01
    bt_r = VGVVSmileBacktester(loader, pairs, '2019-09-01','2024-12-31', allocation_mode='return', report_start_date='2020-01-01'); bt_r.run()
    bt_s = VGVVSmileBacktester(loader, pairs, '2019-09-01','2024-12-31', allocation_mode='sortino', report_start_date='2020-01-01'); bt_s.run()
