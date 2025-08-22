"""
sabr_backtest.py

End-to-end workflow:
 1. Ensure risk-free rates parquet exists (build via rates.build_rates if missing).
 2. Replicate option smile from ATM, risk reversal (RR) and butterfly (BF) quotes.
 3. Calibrate SABR per pair / tenor / day and identify overpriced options (market vol > SABR model vol by threshold).
 4. Strategy: For each business day & tenor, short the 3 most overpriced options across all currency pairs.
 5. Yearly tenor allocation rebalancing (return or sortino mode) with PyPortfolioOpt if available; min 5% per tenor.
 6. Track equity, Greeks, margin, theta estimate and produce CSV outputs & plots (including tearsheet PNG).

Assumptions:
 - Pricing model: Garman-Kohlhagen via BlackScholesFX.
 - Entry costs (bid/ask + commission + slippage) = 17bp of premium; no exit cost.
 - Margin: 20% spot * notional for shorts (released at expiry).
 - Hedge: If |option delta| > 4% on entry day we hedge to neutral using spot at 1bp cost of hedge notionally.
 - Notional sizing: Per tenor capital = initial_capital * weight_t. Each day allocate daily_capital_fraction (10%) of that to new trades, split equally among up to 3 options selected.
 - P&L recognized at expiry (premium treated as liability until then) -> clearer theta attribution.
 - No early exercise / close outs. Expired options settle intrinsic.
 - Foreign rate inferred from forward via covered interest parity.

Outputs (with suffix _{allocation_mode}):
 - results/sabr_trades_{mode}.csv
 - results/sabr_daily_equity_{mode}.csv
 - results/sabr_tenor_performance_{mode}.csv
 - results/sabr_greeks_{mode}.csv, sabr_theta_pnl_{mode}.csv
 - results/sabr_equity_curve.png (shared curve latest run), sabr_greeks_{mode}.png, sabr_vol_edges_distribution.png
 - results/tearsheets/sabr_{mode}.png (compact tearsheet)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
from pathlib import Path
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import os
os.environ.setdefault('CVXPY_SUPPRESS_WARNINGS','1')

from pricing_models import BlackScholesFX
from smile_replication import build_and_calibrate_smile, extract_overpriced_options, TENOR_DAYS
from rates import get_domestic_foreign_rates, ensure_rates
from tearsheet import save_tearsheet

try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    _HAVE_PPO = True
except Exception:
    _HAVE_PPO = False

warnings.filterwarnings('ignore')

TENORS = ["1W","2W","3W","1M","2M","3M","4M","6M","9M","1Y"]

@dataclass
class OptionTrade:
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
    sabr_vol: float
    vol_edge: float
    entry_price: float
    premium: float
    domestic_rate: float
    foreign_rate: float
    spot_entry: float
    forward: float
    hedged: bool = False
    hedge_size: float = 0.0
    hedge_cost: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    premium_net_received: float = 0.0
    last_mark: float = 0.0   # direction * notional * last option price
    hedge_last_mark: float = 0.0  # hedge_size * last spot
    hedge_entry_spot: float = 0.0  # spot when hedge was placed

class SABRSmileBacktester:
    def __init__(self, loader, pairs: List[str], start_date, end_date,
                 initial_capital: float = 10_000_000.0,
                 min_vol_edge: float = 0.005,
                 bid_ask: float = 0.001,
                 commission: float = 0.0005,
                 slippage: float = 0.0002,
                 margin_rate: float = 0.20,
                 daily_capital_fraction: float = 0.10,
                 max_notional: float = 2_500_000.0,
                 allocation_mode: str = 'return', seed: int = 42):
        # Auto-discover pairs if 'ALL'
        if pairs is None or (isinstance(pairs, list) and len(pairs)==1 and pairs[0].upper()=='ALL'):
            fx_dir = Path('data/FX')
            auto = []
            if fx_dir.exists():
                for f in fx_dir.glob('*.parquet'):
                    nm = f.stem
                    if len(nm) <= 10:
                        auto.append(nm)
            pairs = sorted(auto)
        self.pairs = pairs
        self.loader = loader
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.min_vol_edge = min_vol_edge
        self.cost_rate = bid_ask + commission + slippage
        self.margin_rate = margin_rate
        self.daily_capital_fraction = daily_capital_fraction
        self.max_notional = max_notional
        self.allocation_mode = allocation_mode
        self.rng = np.random.default_rng(seed)
        self.bs = BlackScholesFX()
        self.open_trades: List[OptionTrade] = []
        self.closed_trades: List[OptionTrade] = []
        self.trade_counter = 0
        self.tenor_weights = {t: 1/len(TENORS) for t in TENORS}
        self.tenor_alloc_history = []
        self.tenor_daily_pnl = defaultdict(list)
        self.tenor_capital_history = {t: [] for t in TENORS}
        self.daily_records = []
        self.daily_greeks = []
        self.daily_theta_est = []
        self.margin_used = 0.0
        self.business_days = pd.bdate_range(self.start_date, self.end_date)

    # ---------------- Allocation -----------------
    def _rebalance(self, date: pd.Timestamp):
        if date == self.business_days[-1]:
            return
        if date.month != 12:
            return
        if (date + pd.Timedelta(days=1)).year == date.year:
            return
        window = 250
        returns_dict = {}
        for t in TENORS:
            pnl_list = self.tenor_daily_pnl.get(t, [])
            if not pnl_list:
                continue
            pnl_series = pd.Series(pnl_list[-window:])
            base_cap = self.initial_capital * self.tenor_weights.get(t, 1/len(TENORS))
            if base_cap <= 0:
                continue
            returns_dict[t] = (pnl_series / base_cap).reset_index(drop=True)
        if not returns_dict:
            return
        ret_df = pd.DataFrame(returns_dict)
        floor = 0.05
        weights = None
        if _HAVE_PPO and ret_df.shape[0] > 20:
            try:
                clean = ret_df.fillna(0.0)
                mu = expected_returns.mean_historical_return(clean, frequency=252)
                cov = risk_models.semicovariance(clean, frequency=252) if self.allocation_mode=='sortino' else risk_models.sample_cov(clean, frequency=252)
                ef = EfficientFrontier(mu, cov, weight_bounds=(floor,1.0))
                if self.allocation_mode=='sortino':
                    ef.max_sharpe()
                else:
                    try:
                        ef.max_quadratic_utility(risk_aversion=1e-6)
                    except Exception:
                        ef.max_sharpe()
                raw = ef.clean_weights()
                weights = {t: float(raw.get(t,0)) for t in TENORS}
            except Exception:
                weights=None
        if weights is None:
            mu_simple = ret_df.mean()*252
            if self.allocation_mode=='sortino':
                downside = ret_df.clip(upper=0)**2
                semi = (downside.mean()**0.5)*(252**0.5)
                scores = mu_simple / semi.replace(0,np.nan)
            else:
                scores = mu_simple.clip(lower=0)
            scores = scores.replace([np.inf,-np.inf], np.nan).fillna(0)
            if scores.sum()==0:
                scores = pd.Series([1]*len(scores), index=scores.index)
            w_raw = scores / scores.sum()
            weights = {t: float(w_raw.get(t,0)) for t in TENORS}
        for k,v in list(weights.items()):
            # Enforce 5% floor on ALL tenors (even those without data yet) per requirement
            if v > 0:
                weights[k] = max(v, floor)
        # Add missing tenors at floor
        for t in TENORS:
            if t not in weights or weights[t] == 0:
                weights[t] = floor
        total = sum(weights.values())
        weights = {t: v/total for t,v in weights.items()}
        self.tenor_weights = weights
        self.tenor_alloc_history.append({'date':date,'mode':self.allocation_mode, **self.tenor_weights})
        print(f"[REBALANCE-SABR-{self.allocation_mode.upper()}] {date.date()} weights: "+", ".join(f"{k}:{v:.1%}" for k,v in self.tenor_weights.items()))

    # --------------- Helper pricing ---------------
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

    def _open_trade(self, cand, date, tenor_capital):
        sp = cand['smile_point']
        tenor = cand['tenor']
        T_days = TENOR_DAYS[tenor]
        T = T_days/365
        price = self._price(cand['spot'], sp.strike, T, sp.market_vol, cand['r_d'], cand['r_f'], cand['option_type'])
        if price <= 0:
            return None
        per_trade_capital = tenor_capital * self.daily_capital_fraction / 3.0
        if per_trade_capital <= 0:
            return None
        notional = per_trade_capital / price
        notional = min(notional, self.max_notional)
        premium = price * notional
        premium_net = premium * (1 - self.cost_rate)
        margin = cand['spot'] * notional * self.margin_rate
        if self.margin_used + margin > self.equity * 0.8:
            return None
        self.margin_used += margin
        trade = OptionTrade(
            trade_id=self.trade_counter,
            pair=cand['pair'],
            tenor=tenor,
            delta=sp.delta,
            strike=sp.strike,
            option_type=cand['option_type'],
            notional=notional,
            direction=-1,
            entry_date=date,
            expiry_date=date + pd.Timedelta(days=T_days),
            entry_vol=sp.market_vol,
            sabr_vol=sp.sabr_vol,
            vol_edge=sp.vol_edge,
            entry_price=price,
            premium=premium,
            domestic_rate=cand['r_d'],
            foreign_rate=cand['r_f'],
            spot_entry=cand['spot'],
            forward=cand['forward'],
            premium_net_received=premium_net,
            last_mark=(-1) * notional * price,  # direction is -1
            hedge_last_mark=0.0
        )
        self.open_trades.append(trade)
        self.trade_counter += 1
        return trade

    def _hedge(self, trade: OptionTrade, spot: float):
        T = max((trade.expiry_date - trade.entry_date).days,1)/365
        d = self._delta(spot, trade.strike, T, trade.entry_vol, trade.domestic_rate, trade.foreign_rate, trade.option_type)
        if abs(d) > 0.04 and not trade.hedged:
            hedge_units = -d * trade.notional * trade.direction
            cost = abs(hedge_units) * spot * 0.0001
            self.equity -= cost
            trade.hedged = True
            trade.hedge_size = hedge_units
            trade.hedge_cost = cost
            trade.hedge_last_mark = trade.hedge_size * spot
            trade.hedge_entry_spot = spot

    def _value_trade(self, trade: OptionTrade, date: pd.Timestamp, spot: float):
        if date >= trade.expiry_date:
            intrinsic = max(spot - trade.strike,0) if trade.option_type=='call' else max(trade.strike - spot,0)
            # Realized PnL for reporting only (equity already fully MTM):
            pnl_realized = trade.premium_net_received - intrinsic * trade.notional
            if trade.hedged:
                pnl_realized += trade.hedge_size * (spot - trade.hedge_entry_spot) - trade.hedge_cost
            trade.exit_price = intrinsic
            trade.pnl = pnl_realized
            return pnl_realized
        return 0.0

    def _portfolio_greeks(self, date: pd.Timestamp):
        total_delta=total_gamma=total_vega=total_theta=0.0
        for tr in self.open_trades:
            if date >= tr.expiry_date:
                continue
            vol_data = self.loader.get_volatility_surface(tr.pair, date)
            if not vol_data:
                continue
            T = max((tr.expiry_date - date).days,1)/365
            try:
                g = self.bs.calculate_greeks(tr.strike, int(T*365), tr.entry_vol, vol_data.spot, tr.domestic_rate, tr.foreign_rate, tr.option_type)
                total_delta += g['delta'] * tr.direction * tr.notional * vol_data.spot
                total_gamma += g['gamma'] * tr.direction * tr.notional * vol_data.spot
                total_vega  += g['vega']  * tr.direction * tr.notional
                total_theta += g['theta'] * tr.direction * tr.notional
            except Exception:
                continue
        return {'delta':total_delta,'gamma':total_gamma,'vega':total_vega,'theta':total_theta}

    # --------------- Main run ---------------
    def run(self):
        ensure_rates()
        print(f"\n=== SABR Backtest ({self.allocation_mode}) ===")
        print(f"Period: {self.start_date.date()} -> {self.end_date.date()}  Pairs: {', '.join(self.pairs)}")
        realized_equity_base = self.initial_capital  # dynamic equity base including realized premium & MTM deltas
        for date in self.business_days:
            self._rebalance(date)
            day_mtm_pnl = 0.0
            spot_cache = {}
            # Mark-to-market existing positions
            for tr in list(self.open_trades):
                vd = self.loader.get_volatility_surface(tr.pair, date)
                if not vd:
                    continue
                spot = vd.spot
                spot_cache[tr.pair] = spot
                expired = date >= tr.expiry_date
                if not expired:
                    T = max((tr.expiry_date - date).days,1)/365
                    try:
                        price_now = self.bs.price(tr.strike, int(T*365), tr.entry_vol, spot, tr.domestic_rate, tr.foreign_rate, tr.option_type)
                    except Exception:
                        price_now = max((spot-tr.strike) if tr.option_type=='call' else (tr.strike-spot),0)
                else:
                    price_now = max((spot-tr.strike) if tr.option_type=='call' else (tr.strike-spot),0)
                current_mark = tr.direction * tr.notional * price_now  # negative for short
                delta_pnl = tr.last_mark - current_mark
                day_mtm_pnl += delta_pnl
                tr.last_mark = current_mark
                # Hedge MTM
                if tr.hedged:
                    hedge_mark = tr.hedge_size * spot
                    hedge_delta_pnl = hedge_mark - tr.hedge_last_mark
                    day_mtm_pnl += hedge_delta_pnl
                    tr.hedge_last_mark = hedge_mark
                if expired:
                    # Compute realized PnL for report
                    self._value_trade(tr, date, spot)
                    self.open_trades.remove(tr)
                    self.closed_trades.append(tr)
            # Apply daily MTM to equity
            self.equity += day_mtm_pnl
            # Open new trades (after MTM)
            if len(self.open_trades) < 300:
                for tenor in TENORS:
                    tenor_cap = self.initial_capital * self.tenor_weights[tenor]
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
                        overpriced = extract_overpriced_options(smile, self.min_vol_edge)
                        for op in overpriced:
                            opt_type = 'call' if op.delta >= 0.5 else 'put'
                            T = TENOR_DAYS[tenor]/365
                            r_d, r_f = get_domestic_foreign_rates(date, T, vd.spot, fwd)
                            cands.append({'pair':pair,'tenor':tenor,'smile_point':op,'spot':vd.spot,'forward':fwd,'option_type':opt_type,'r_d':r_d,'r_f':r_f})
                    if not cands: continue
                    cands.sort(key=lambda x: x['smile_point'].vol_edge, reverse=True)
                    for cd in cands[:3]:
                        tr = self._open_trade(cd, date, tenor_cap)
                        if tr:
                            self.equity += tr.premium_net_received  # receive premium
                            self._hedge(tr, cd['spot'])
            else:
                for tenor in TENORS:
                    self.tenor_capital_history[tenor].append(self.initial_capital * self.tenor_weights[tenor])
            g = self._portfolio_greeks(date)
            theta_est = g.get('theta',0.0)
            self.daily_theta_est.append({'date':date,'theta_pnl_est':theta_est})
            self.daily_greeks.append({'date':date, **g})
            self.daily_records.append({'date':date,'equity':self.equity,'open_trades':len(self.open_trades),'closed_trades':len(self.closed_trades),'margin_used':self.margin_used, 'day_mtm_pnl':day_mtm_pnl, **g})
            if (len(self.daily_records)%250==0) or date==self.business_days[-1]:
                ret=(self.equity-self.initial_capital)/self.initial_capital
                print(f"  {date.date()} | Eq ${self.equity/1e6:.2f}M Ret {ret:+.1%} Open {len(self.open_trades)} Closed {len(self.closed_trades)}")
        return self._finalize()

    # --------------- Finalize ---------------
    def _finalize(self):
        df_daily = pd.DataFrame(self.daily_records).set_index('date')
        trades_df = pd.DataFrame(t.__dict__ for t in self.closed_trades)
        eq = df_daily['equity'].values
        rets = np.diff(eq)/eq[:-1] if len(eq)>1 else np.array([])
        tot_ret = (eq[-1]-eq[0])/eq[0] if len(eq)>1 else 0
        ann_ret = (1+tot_ret)**(252/len(rets))-1 if len(rets)>0 else 0
        ann_vol = rets.std()*np.sqrt(252) if rets.std()>0 else 0
        sharpe = ann_ret/ann_vol if ann_vol>0 else 0
        peak = np.maximum.accumulate(eq) if len(eq)>0 else np.array([])
        dd = (eq-peak)/peak if len(eq)>0 else np.array([])
        max_dd = dd.min() if len(dd)>0 else 0
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
        mode_suffix = f"_{self.allocation_mode}"
        df_daily.to_csv(out_dir/f'sabr_daily_equity{mode_suffix}.csv')
        trades_df.to_csv(out_dir/f'sabr_trades{mode_suffix}.csv', index=False)
        tenor_perf_df.to_csv(out_dir/f'sabr_tenor_performance{mode_suffix}.csv', index=False)
        if self.daily_greeks:
            pd.DataFrame(self.daily_greeks).to_csv(out_dir/f'sabr_greeks{mode_suffix}.csv', index=False)
        if self.daily_theta_est:
            pd.DataFrame(self.daily_theta_est).to_csv(out_dir/f'sabr_theta_pnl{mode_suffix}.csv', index=False)
        # Plots
        try:
            plt.figure(figsize=(12,5)); plt.plot(df_daily.index, df_daily['equity']/1e6); plt.title('SABR Equity Curve'); plt.ylabel('Equity (MM)'); plt.tight_layout(); plt.savefig(out_dir/'sabr_equity_curve.png', dpi=200); plt.close()
            if self.tenor_alloc_history:
                wdf = pd.DataFrame(self.tenor_alloc_history).set_index('date'); wdf.plot(figsize=(12,5)); plt.title('SABR Tenor Weights'); plt.tight_layout(); plt.savefig(out_dir/'sabr_tenor_weights.png', dpi=200); plt.close()
            if self.daily_greeks:
                gdf = pd.DataFrame(self.daily_greeks).set_index('date'); plt.figure(figsize=(12,6));
                for c in ['delta','gamma','vega','theta']:
                    if c in gdf.columns: plt.plot(gdf.index, gdf[c], label=c)
                plt.legend(); plt.title(f'SABR Greeks ({self.allocation_mode})'); plt.tight_layout(); plt.savefig(out_dir/f'sabr_greeks{mode_suffix}.png', dpi=200); plt.close()
            edges = [t.vol_edge for t in self.closed_trades if t.vol_edge is not None]
            if edges:
                plt.figure(figsize=(8,4)); plt.hist(edges, bins=40, color='steelblue', edgecolor='black'); plt.title('Distribution of Sold Vol Edges (Market - SABR)'); plt.xlabel('Vol Edge'); plt.tight_layout(); plt.savefig(out_dir/'sabr_vol_edges_distribution.png', dpi=200); plt.close()
            # Tearsheet PNG
            try:
                save_tearsheet(df_daily['equity'], f'SABR ({self.allocation_mode})', str(out_dir / 'tearsheets' / f'sabr_{self.allocation_mode}.png'))
            except Exception:
                pass
        except Exception:
            pass
        print(f"\n=== SABR Summary ({self.allocation_mode}) ===")
        print(f"Total Return: {tot_ret:+.1%} AnnReturn: {ann_ret:+.1%} Sharpe: {sharpe:.2f} MaxDD: {max_dd:.1%} Closed: {len(self.closed_trades)} FinalEq: ${self.equity:,.0f}")
        if not tenor_perf_df.empty:
            rows = tenor_perf_df.to_dict('records')
            print('Tenor Performance: '+', '.join(f"{r['tenor']}:{r['cum_pnl']:,.0f}" for r in rows))
        return {
            'initial_capital': self.initial_capital,
            'final_equity': float(self.equity),
            'total_return': float(tot_ret),
            'annualized_return': float(ann_ret),
            'annualized_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'num_closed_trades': len(self.closed_trades),
            'allocation_mode': self.allocation_mode,
            'tenor_weights_final': self.tenor_weights
        }

# ---------------- Runner ----------------
if __name__ == '__main__':
    from data_loader import FXDataLoader
    ensure_rates()
    loader = FXDataLoader()
    pairs = ['ALL']
    bt_r = SABRSmileBacktester(loader, pairs, '2007-01-01','2024-12-31', allocation_mode='return'); bt_r.run()
    bt_s = SABRSmileBacktester(loader, pairs, '2007-01-01','2024-12-31', allocation_mode='sortino'); bt_s.run()
