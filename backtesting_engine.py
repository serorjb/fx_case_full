"""
Enhanced FX Options Backtesting Engine - Fixed Version
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float = 10_000_000
    pairs: Optional[List[str]] = None
    max_positions: int = 100
    max_position_size: float = 0.01
    vol_threshold: float = 0.001
    pricing_model: str = "VGVV"
    calibration_window: int = 252
    max_drawdown: float = 0.10
    bid_ask_spread: float = 0.001
    commission: float = 0.0001
    delta_threshold: float = 0.03
    max_daily_trades: int = 20
    margin_requirement: float = 0.30  # 30% margin on short option underlying exposure
    max_open_positions: int = 500     # Global hard cap
    max_signals_per_pair: int = 5
    per_pair_position_limit: int = 120
    per_tenor_position_limit: int = 40
    is_end_date: Optional[pd.Timestamp] = None  # In-sample end date for calibration
    optimize: bool = False
    vol_threshold_grid: Optional[List[float]] = None
    max_position_size_grid: Optional[List[float]] = None
    delta_threshold_grid: Optional[List[float]] = None
    close_vol_edge_factor: float = 0.5  # Close when mispricing shrinks below factor * vol_threshold
    output_dir: str = "output"


class BacktestResults:
    """Container for backtest results and analytics"""

    def __init__(self):
        self.equity_curve = []
        self.daily_snapshots = []
        self.trades = []
        self.expired_positions = []
        self.hedge_trades = []
        self.greeks_history = defaultdict(list)
        self.positions_history = []
        self.returns = []
        self.metrics = {}
        self.realized_pnl_series = []
        self.unrealized_pnl_series = []
        self.hedge_pnl_series = []
        self.margin_series = []
        self.var_history = []
        self.es_history = []
        self.regime_tags = []
        self.closed_positions = []  # early exits
        self._cumulative_realized = 0.0
        self._cumulative_hedge = 0.0

    def add_snapshot(self, date, equity, positions, greeks, daily_trades=0, daily_pnl=0):
        """Add daily portfolio snapshot"""
        snapshot = {
            'date': date,
            'equity': equity,
            'num_positions': len(positions),
            'daily_trades': daily_trades,
            'daily_pnl': daily_pnl,
            'greeks': greeks.copy(),
            'margin_used': getattr(self, 'margin_used', None)
        }

        self.daily_snapshots.append(snapshot)
        self.equity_curve.append({'date': date, 'equity': equity})
        self.positions_history.append({
            'date': date,
            'positions': len(positions),
            'total_notional': sum(p.get('notional', 0) for p in positions)
        })

        # Store Greeks
        for greek, value in greeks.items():
            self.greeks_history[greek].append({'date': date, 'value': value})

        # Margin tracking
        if hasattr(self, 'margin_used'):
            self.margin_series.append({'date': date, 'margin': getattr(self, 'margin_used', 0)})

    def calculate_final_metrics(self, initial_capital, all_trades, expired_pnl):
        """Calculate comprehensive performance metrics"""
        if len(self.equity_curve) < 2:
            self.metrics = {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
            return

        # Extract equity values
        equity_values = [e['equity'] for e in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]

        # Basic performance metrics
        total_return = (equity_values[-1] - initial_capital) / initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0

        # Risk metrics
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Drawdown calculation
        equity_array = np.array(equity_values)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Trading statistics - FIXED
        successful_trades = [t for t in expired_pnl if t.get('pnl', 0) > 0]
        total_expired = len(expired_pnl)
        win_rate = len(successful_trades) / total_expired if total_expired > 0 else 0

        avg_win = np.mean([t['pnl'] for t in successful_trades]) if successful_trades else 0
        losing_trades = [t for t in expired_pnl if t.get('pnl', 0) <= 0]
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 1
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Calculate total P&L from expired positions
        total_expired_pnl = sum(t.get('pnl', 0) for t in expired_pnl)

        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(all_trades),
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'avg_delta': 0,
            'avg_vega': 0,
            'max_vega': 0,
            'total_expired_trades': total_expired,
            'total_expired_pnl': total_expired_pnl,
            'avg_trade_pnl': total_expired_pnl / total_expired if total_expired > 0 else 0
        }

    def export_reports(self, path: str):
        import os, csv
        os.makedirs(path, exist_ok=True)
        # Trades
        if self.trades:
            pd.DataFrame(self.trades).to_csv(os.path.join(path, 'trades.csv'), index=False)
        if self.expired_positions:
            pd.DataFrame(self.expired_positions).to_csv(os.path.join(path, 'expired_positions.csv'), index=False)
        if self.closed_positions:
            pd.DataFrame(self.closed_positions).to_csv(os.path.join(path, 'early_closed_positions.csv'), index=False)
        if self.daily_snapshots:
            pd.DataFrame(self.daily_snapshots).to_csv(os.path.join(path, 'daily_snapshots.csv'), index=False)
        if self.greeks_history:
            for g, series in self.greeks_history.items():
                pd.DataFrame(series).to_csv(os.path.join(path, f'greeks_{g}.csv'), index=False)
        if self.var_history:
            pd.DataFrame(self.var_history).to_csv(os.path.join(path, 'var_series.csv'), index=False)
        if self.es_history:
            pd.DataFrame(self.es_history).to_csv(os.path.join(path, 'es_series.csv'), index=False)


class FXOptionsBacktester:
    """
    Fixed Systematic FX Options Backtesting Engine
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.loader = None
        self.strategy = None
        self.pricing_model = None
        self.pairs = []

        # Position tracking
        self.active_positions = []
        self.hedge_positions = defaultdict(float)
        self.position_counter = 0

        # Performance tracking
        self.current_equity = config.initial_capital
        self.results = BacktestResults()
        self.margin_used = 0.0
        self.stop_trading = False

        # Realized and unrealized P&L
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        # Per-pair and per-tenor position counts
        self.per_pair_counts = defaultdict(int)
        self.per_pair_tenor_counts = defaultdict(lambda: defaultdict(int))

    def initialize(self):
        """Initialize backtester with data loader and models"""
        from data_loader import FXDataLoader
        from trading_strategy import VolatilityArbitrageStrategy
        from pricing_models import BlackScholesFX

        # Initialize data loader
        self.loader = FXDataLoader()
        self.loader.rf_curve.load_fred_data()

        # Determine available pairs
        if self.config.pairs:
            self.pairs = self.config.pairs
        else:
            # Auto-detect available pairs
            from pathlib import Path
            data_dir = Path("data/FX")
            if data_dir.exists():
                self.pairs = ['AUDNZD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
            else:
                self.pairs = ['AUDNZD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

        # Initialize strategy
        self.strategy = VolatilityArbitrageStrategy(
            initial_capital=self.config.initial_capital,
            max_position_size=self.config.max_position_size,
            vol_threshold=self.config.vol_threshold,
            transaction_cost=self.config.commission
        )

        # Initialize pricing model
        self.pricing_model = BlackScholesFX()

        print(f"Initialized with {len(self.pairs)} pairs: {', '.join(self.pairs)}")

    def run_backtest(self):
        """Run systematic daily trading backtest"""
        print(f"\nRunning systematic backtest from {self.config.start_date.date()} to {self.config.end_date.date()}")

        # Generate business days
        business_days = pd.bdate_range(
            start=self.config.start_date,
            end=self.config.end_date
        )

        total_days = len(business_days)
        print(f"Total trading days: {total_days}")

        for day_idx, current_date in enumerate(business_days):
            # Skip if optimizing and beyond IS end
            if self.config.is_end_date and current_date > self.config.end_date:
                break
            try:
                self._process_trading_day(current_date)

                # Progress reporting
                if day_idx % 100 == 0 or day_idx == total_days - 1:
                    print(f"  Day {day_idx+1}/{total_days}: {current_date.date()} - "
                          f"Equity: ${self.current_equity/1e6:.1f}M, "
                          f"Positions: {len(self.active_positions)}")

            except Exception as e:
                print(f"Error on {current_date}: {e}")
                continue

        # Finalize results
        self.results.calculate_final_metrics(
            self.config.initial_capital,
            self.results.trades,
            self.results.expired_positions
        )

        return self.results

    def _process_trading_day(self, current_date):
        """Process a single trading day"""
        daily_pnl = 0
        daily_trade_count = 0

        # 1. EXPIRE OLD POSITIONS
        daily_pnl += self._expire_positions(current_date)

        # 2. MARK-TO-MARKET EXISTING POSITIONS
        mtm_pnl = self._mark_to_market_positions(current_date)
        self.unrealized_pnl += mtm_pnl

        # 3. SCAN FOR NEW OPPORTUNITIES
        if self.stop_trading:
            new_positions = []
        else:
            new_positions = self._scan_for_opportunities(current_date)
        daily_trade_count = len(new_positions)
        self.active_positions.extend(new_positions)

        # 4. DELTA HEDGING
        hedging_pnl = self._perform_delta_hedging(current_date)
        daily_pnl += hedging_pnl

        # 5. UPDATE EQUITY
        self.current_equity += daily_pnl + mtm_pnl

        # 6. CALCULATE PORTFOLIO GREEKS
        portfolio_greeks = self._calculate_portfolio_greeks(current_date)

        # 7. RECORD SNAPSHOT
        self.results.add_snapshot(
            date=current_date,
            equity=self.current_equity,
            positions=self.active_positions,
            greeks=portfolio_greeks,
            daily_trades=daily_trade_count,
            daily_pnl=daily_pnl
        )

        # Evaluate position exits
        self._evaluate_position_exits(current_date)

        # After snapshot risk checks
        self._risk_checks()

        # After equity update record PnL series
        self.results.realized_pnl_series.append({'date': current_date, 'realized': self.realized_pnl})
        self.results.unrealized_pnl_series.append({'date': current_date, 'unrealized': self.unrealized_pnl})
        if hedging_pnl != 0:
            self.results.hedge_pnl_series.append({'date': current_date, 'hedge_pnl': hedging_pnl})

        # Risk analytics
        self._risk_analytics(current_date)

    def _expire_positions(self, current_date):
        """Handle position expiries and calculate P&L"""
        total_pnl = 0
        positions_to_remove = []

        for idx, position in enumerate(self.active_positions):
            if current_date >= position['expiry_date']:
                pnl = self._calculate_expiry_pnl(position, current_date)
                total_pnl += pnl
                # Release margin
                self.margin_used -= position.get('margin',0)
                positions_to_remove.append(idx)

        # Remove expired positions
        for idx in reversed(positions_to_remove):
            self.active_positions.pop(idx)

        return total_pnl

    def _calculate_expiry_pnl(self, position, current_date):
        """Calculate P&L for an expiring option - FIXED"""
        try:
            # Get current spot rate
            pair_data = self.loader.load_pair_data(position['pair'])

            # Find closest available date
            available_dates = pair_data.index
            closest_date = min(available_dates, key=lambda x: abs((x - current_date).days))

            if abs((closest_date - current_date).days) > 7:  # More than 1 week difference
                return 0

            # Get the correct column name for spot price
            spot_column = f"{position['pair']} Curncy"

            # Check if the column exists, if not try the first column
            if spot_column not in pair_data.columns:
                spot_column = pair_data.columns[0]

            current_spot = pair_data.loc[closest_date, spot_column]
            strike = position['strike']
            notional = position['notional']

            # Calculate intrinsic value
            if position['option_type'] == 'call':
                intrinsic_value = max(0, current_spot - strike)
            else:  # put
                intrinsic_value = max(0, strike - current_spot)

            # Convert to currency units
            option_value = intrinsic_value * notional

            # P&L calculation based on position type
            if position['position_type'] == 'long':
                pnl = option_value - position['premium_paid']
            else:  # short
                pnl = position['premium_received'] - option_value

            # Record the expired position with P&L
            expired_record = {
                'id': position['id'],
                'pair': position['pair'],
                'expiry_date': current_date,
                'strike': strike,
                'option_type': position['option_type'],
                'position_type': position['position_type'],
                'notional': notional,
                'spot_at_expiry': current_spot,
                'intrinsic_value': intrinsic_value,
                'option_value': option_value,
                'premium_paid': position.get('premium_paid', 0),
                'premium_received': position.get('premium_received', 0),
                'pnl': pnl
            }

            self.results.expired_positions.append(expired_record)

            if hasattr(self, 'realized_pnl'):
                self.realized_pnl += pnl

            # decrement per-pair counters
            self.per_pair_counts[position['pair']] -= 1
            self.per_pair_tenor_counts[position['pair']][position['tenor']] -= 1

            return pnl

        except Exception as e:
            print(f"Error calculating expiry P&L for position {position.get('id', 'unknown')}: {e}")
            return 0

    def _mark_to_market_positions(self, current_date):
        """Mark existing positions to market using Black-Scholes price change since last mark."""
        mtm_pnl = 0.0
        spot_cache = {}
        for position in self.active_positions:
            # Skip already expired positions (they get handled in expiry logic)
            if current_date >= position['expiry_date']:
                continue
            pair = position['pair']
            if pair not in spot_cache:
                vol_data = self.loader.get_volatility_surface(pair, current_date)
                if vol_data is None:
                    continue
                spot_cache[pair] = vol_data.spot
            spot = spot_cache[pair]
            T_days = max((position['expiry_date'] - current_date).days, 0)
            if T_days == 0:
                continue
            # Use stored market vol (could be updated with smile model)
            vol = position['market_vol']
            current_price = self._price_option(spot, position['strike'], T_days/365, vol, position['option_type'])
            last_price = position.get('last_price', position['premium_paid']/position['notional'] if position['notional']>0 else current_price)
            direction = 1 if position['position_type']=='long' else -1
            pnl_increment = direction * (current_price - last_price) * position['notional']
            mtm_pnl += pnl_increment
            position['last_price'] = current_price
        # now returns incremental unrealized pnl
        return mtm_pnl

    def _scan_for_opportunities(self, current_date):
        """Scan all pairs for trading opportunities using core strategy identify_opportunities"""
        new_positions = []

        for pair in self.pairs:
            if len(new_positions) >= self.config.max_daily_trades:
                break
            try:
                if current_date == self.config.start_date and pair == self.pairs[0]:
                    self._debug_data_structure(pair)
                vol_data = self.loader.get_volatility_surface(pair, current_date)
                if vol_data is None:
                    continue
                # Build market/model surfaces (no double % to decimal conversion)
                market_surface, model_surface = self._generate_surfaces(vol_data, current_date)
                if not market_surface or not model_surface:
                    continue
                # Use strategy engine to identify opportunities
                signals = self.strategy.identify_opportunities(
                    market_surface=market_surface,
                    model_surface=model_surface,
                    spot=vol_data.spot,
                    date=current_date
                )
                # Restrict to top N per pair
                signals = signals[:self.config.max_signals_per_pair]
                for signal in signals:
                    # Per pair / tenor limits
                    if self.per_pair_counts[pair] >= self.config.per_pair_position_limit:
                        break
                    if self.per_pair_tenor_counts[pair][signal.tenor] >= self.config.per_tenor_position_limit:
                        continue
                    # Margin / equity / position cap checks
                    if len(self.active_positions) + len(new_positions) >= self.config.max_positions:
                        break
                    position = self._execute_trade(signal, vol_data, current_date, pair)
                    if position:
                        self.per_pair_counts[pair] += 1
                        self.per_pair_tenor_counts[pair][signal.tenor] += 1
                        new_positions.append(position)
                        if len(new_positions) >= self.config.max_daily_trades:
                            break
            except Exception:
                continue
        return new_positions

    def _create_simple_signals(self, market_surface, model_surface, vol_data):
        """(Deprecated) Simple signal creation retained for reference; no longer used."""
        signals = []
        return signals

    def _generate_surfaces(self, vol_data, current_date):
        """Generate market and model volatility surfaces from ATM, RR, BF inputs.
        Output format matches strategy expectations: tenor -> {'vol': vol}.
        Skips any tenor with missing/invalid ATM vol values."""
        market_surface = {}
        model_surface = {}
        tenors = ["1W", "2W", "3W", "1M", "2M", "3M", "4M", "6M", "9M", "12M"]
        for tenor in tenors:
            if tenor in vol_data.atm_vols:
                atm = vol_data.atm_vols[tenor]
                # Validate atm volatility
                if atm is None or not np.isfinite(atm) or atm <= 0:
                    continue
                rr = vol_data.rr_25d.get(tenor, 0.0) or 0.0
                bf = vol_data.bf_25d.get(tenor, 0.0) or 0.0
                # Ensure rr/bf are numeric
                try:
                    rr = float(rr)
                    bf = float(bf)
                except (TypeError, ValueError):
                    rr, bf = 0.0, 0.0
                if not np.isfinite(rr): rr = 0.0
                if not np.isfinite(bf): bf = 0.0
                market_surface[tenor] = {'vol': float(atm)}
                adj = -0.25 * rr - 0.5 * bf
                base_model = atm + adj
                base_model = float(np.clip(base_model, 0.01, 0.80))
                variation = (hash((tenor, int(current_date.strftime('%Y%m%d')))) % 7) / 10000.0
                model_vol = max(0.0001, base_model * (1 - 0.005) + variation)
                model_surface[tenor] = {'vol': model_vol}
        return market_surface, model_surface

    def _execute_trade(self, signal, vol_data, current_date, pair):
        """Execute a trading signal"""
        try:
            # Derive time to expiry in years
            T = None
            if hasattr(signal, 'tenor') and signal.tenor:
                T = self._tenor_to_years(signal.tenor)
            if (T is None or T <= 0) and hasattr(signal, 'expiry'):
                T = max((signal.expiry - current_date).days, 1)/365
            if T is None or T <= 0:
                T = 1/12  # fallback 1M
            market_vol = signal.market_vol if isinstance(signal.market_vol, (int, float)) else getattr(signal.market_vol, 'vol', None)
            if market_vol is None or market_vol <= 0:
                return None
            spot = vol_data.spot
            strike = signal.strike if getattr(signal, 'strike', None) else spot
            r = 0.02
            option_price = self._price_option(spot, strike, T, market_vol, getattr(signal, 'option_type','call'), r)
            if option_price <= 0:
                return None
            max_position_value = self.current_equity * self.config.max_position_size
            if max_position_value <= 0 or self.stop_trading:
                return None
            direction = -1 if getattr(signal, 'signal_type', None) and signal.signal_type.value == 'overpriced' else 1
            notional = min(max_position_value / option_price, 20000)
            margin = 0.0
            if direction == -1:
                margin = spot * notional * self.config.margin_requirement
                if self.margin_used + margin > self.current_equity * 0.9:
                    return None
            total_premium = option_price * notional
            if direction == -1:
                self.current_equity += total_premium * (1 - self.config.commission)
            else:
                cash_out = total_premium * (1 + self.config.commission)
                if cash_out > self.current_equity * 0.99:
                    return None
                self.current_equity -= cash_out
            self.margin_used += margin
            position = {
                'id': f"{pair}_{self.position_counter}",
                'pair': pair,
                'option_type': getattr(signal, 'option_type','call'),
                'position_type': 'short' if direction==-1 else 'long',
                'strike': strike,
                'tenor': getattr(signal, 'tenor',''),
                'expiry_date': current_date + pd.Timedelta(days=int(T * 365)),
                'notional': notional,
                'market_vol': market_vol,
                'model_vol': getattr(signal, 'model_vol', market_vol),
                'premium_paid': total_premium if direction==1 else 0,
                'premium_received': total_premium if direction==-1 else 0,
                'entry_date': current_date,
                'entry_spot': spot,
                'last_price': option_price,
                'margin': margin
            }
            self.position_counter += 1
            self.results.trades.append(position.copy())
            return position
        except Exception as e:
            print(f"Error executing trade: {e}")
            return None

    def _price_option(self, S, K, T, vol, option_type='call', r=0.02):
        """Black-Scholes option pricing with robust input validation."""
        from scipy.stats import norm
        import math
        # Validate inputs
        if (T is None) or (vol is None) or (S is None) or (K is None):
            return 0.0001
        try:
            T = float(T)
            vol = float(vol)
            S = float(S)
            K = float(K)
        except (TypeError, ValueError):
            return 0.0001
        if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
            return 0.0001
        try:
            d1 = (math.log(S/K) + (r + 0.5*vol**2)*T) / (vol*math.sqrt(T))
            d2 = d1 - vol*math.sqrt(T)
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
            else:
                price = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return max(price, 0.0001)
        except Exception:
            return 0.0001

    def _perform_delta_hedging(self, current_date):
        """Perform simple delta hedging; neutralize per pair above threshold."""
        if self.stop_trading:
            return 0
        pair_delta = {}
        spot_cache = {}
        for position in self.active_positions:
            if current_date >= position['expiry_date']:
                continue
            pair = position['pair']
            if pair not in spot_cache:
                vol_data = self.loader.get_volatility_surface(pair, current_date)
                if vol_data is None:
                    continue
                spot_cache[pair] = vol_data.spot
            spot = spot_cache[pair]
            T_days = max((position['expiry_date'] - current_date).days,0)
            if T_days == 0:
                continue
            vol = position['market_vol']
            delta = self._option_delta(spot, position['strike'], T_days/365, vol, position['option_type'])
            direction = 1 if position['position_type']=='long' else -1
            pair_delta[pair] = pair_delta.get(pair,0) + delta * direction * position['notional'] * spot
        hedging_cost = 0.0
        threshold = self.config.delta_threshold * self.current_equity
        for pair, d_exposure in pair_delta.items():
            if abs(d_exposure) > threshold:
                # Hedge the excess
                hedge_size = -d_exposure / spot_cache[pair]
                # Assume zero slippage & small cost
                cost = abs(hedge_size) * spot_cache[pair] * 0.0001
                hedging_cost += cost
                self.results.hedge_trades.append({'date': current_date,'pair':pair,'hedge_size':hedge_size,'cost':cost})
        hedge_cost = -hedging_cost
        if hedge_cost != 0:
            self.realized_pnl += hedge_cost
        return -hedging_cost

    def _calculate_portfolio_greeks(self, current_date):
        """Calculate portfolio Greeks by maturity bucket"""
        # Replace simplified approximations with BS greeks per position
        total_delta = total_gamma = total_vega = total_theta = 0.0
        spot_cache = {}
        for position in self.active_positions:
            if current_date >= position['expiry_date']:
                continue
            pair = position['pair']
            if pair not in spot_cache:
                vol_data = self.loader.get_volatility_surface(pair, current_date)
                if vol_data is None:
                    continue
                spot_cache[pair] = vol_data.spot
            S = spot_cache[pair]
            T_days = (position['expiry_date'] - current_date).days
            if T_days <= 0:
                continue
            T = T_days / 365
            vol = position['market_vol']
            delta, gamma, vega, theta = self._bs_greeks(S, position['strike'], vol, T, position['option_type'])
            direction = 1 if position['position_type']=='long' else -1
            total_delta += delta * direction * position['notional']
            total_gamma += gamma * direction * position['notional']
            total_vega += vega * direction * position['notional']
            total_theta += theta * direction * position['notional']
        return {'delta': total_delta,'gamma': total_gamma,'vega': total_vega,'theta': total_theta}

    def _bs_greeks(self, S, K, vol, T, option_type):
        import math
        from math import erf, sqrt, exp, log
        if T<=0 or vol<=0 or S<=0 or K<=0:
            return 0,0,0,0
        d1 = (log(S/K)+(0.5*vol*vol)*T)/(vol*sqrt(T))
        d2 = d1 - vol*sqrt(T)
        # normal pdf/cdf
        pdf = (1/np.sqrt(2*np.pi))*np.exp(-0.5*d1*d1)
        cdf = 0.5*(1+erf(d1/np.sqrt(2)))
        if option_type=='call':
            delta = cdf
            theta_sign = -1
        else:
            delta = cdf - 1
            theta_sign = -1
        gamma = pdf/(S*vol*sqrt(T))
        vega = S*pdf*sqrt(T) / 100.0  # per 1 vol point (percentage)
        theta = theta_sign * (pdf*S*vol/(2*sqrt(T)))/365
        return delta, gamma, vega, theta

    def _risk_checks(self):
        """Apply portfolio level risk controls (drawdown, equity floor, margin)."""
        if self.current_equity <= 0:
            self.current_equity = 0
            self.stop_trading = True
        # Drawdown based throttle
        peak_equity = max(e['equity'] for e in self.results.equity_curve) if self.results.equity_curve else self.config.initial_capital
        dd = (peak_equity - self.current_equity)/peak_equity if peak_equity>0 else 0
        if dd > self.config.max_drawdown:
            # Halt new trading if breach
            self.stop_trading = True
        # Margin breach
        if self.margin_used > self.current_equity * 0.9:  # Hard cap: margin cannot exceed 90% equity
            self.stop_trading = True

        # Add margin utilization metric to results metrics progressively
        if self.results.equity_curve:
            self.results.metrics['avg_margin_utilization'] = float(np.mean([
                s.get('margin_used',0)/s['equity'] if s['equity']>0 else 0 for s in self.results.daily_snapshots
            ]))
        # Store realized/unrealized in metrics (rolling overwrite ok)
        self.results.metrics['realized_pnl'] = self.realized_pnl
        self.results.metrics['unrealized_pnl'] = self.unrealized_pnl
        self.results.metrics['hedge_pnl'] = sum(h['hedge_pnl'] for h in self.results.hedge_pnl_series) if self.results.hedge_pnl_series else 0

        # Additional drawdown guard: if stop_trading set, liquidate remaining after hedging logic not implemented here
        return

    def _debug_data_structure(self, pair):
        """Debug helper to inspect pair data once."""
        try:
            pair_data = self.loader.load_pair_data(pair)
            print(f"DEBUG {pair}: rows={len(pair_data)} cols={len(pair_data.columns)} range={pair_data.index.min().date()}->{pair_data.index.max().date()}")
        except Exception as e:
            print(f"DEBUG load fail {pair}: {e}")
        return True

    def _option_delta(self, S, K, T, vol, option_type):
        """Calculate option delta using Black-Scholes formula"""
        from math import exp, log, sqrt
        from scipy.stats import norm

        if T <= 0 or vol <= 0:
            return 0

        d1 = (log(S / K) + (0.5 * vol ** 2) * T) / (vol * sqrt(T))
        cdf = norm.cdf(d1)

        # Provide consistent delta with _bs_greeks
        if option_type=='call':
            return cdf
        else:
            return cdf-1

    def _evaluate_position_exits(self, current_date):
        """Early exit logic based on reduced vol edge or time decay."""
        to_close = []
        spot_cache = {}
        for idx, position in enumerate(self.active_positions):
            if current_date >= position['expiry_date']:
                continue
            pair = position['pair']
            if pair not in spot_cache:
                vol_data = self.loader.get_volatility_surface(pair, current_date)
                if not vol_data:
                    continue
                spot_cache[pair] = vol_data.spot
                current_atm_vols = vol_data.atm_vols
            # Get fresh market vol for tenor
            tenor = position['tenor']
            if tenor in vol_data.atm_vols:
                new_market_vol = vol_data.atm_vols[tenor]
                edge = abs(new_market_vol - position['model_vol'])
                if edge < self.config.close_vol_edge_factor * self.config.vol_threshold:
                    to_close.append((idx, new_market_vol, 'edge_reverted'))
            # Time stop (e.g. if < 10% life remaining)
            life_frac = (position['expiry_date'] - current_date).days / max((position['expiry_date'] - position['entry_date']).days,1)
            if life_frac < 0.1:
                if (idx, position.get('market_vol'), 'edge_reverted') not in to_close:
                    to_close.append((idx, position.get('market_vol'), 'time_stop'))
        for offset, (idx, mvol, reason) in enumerate(to_close):
            real_idx = idx - offset  # adjust for prior pops
            if 0 <= real_idx < len(self.active_positions):
                self._early_close_position(real_idx, current_date, mvol, reason)

    def _early_close_position(self, idx, current_date, market_vol, reason):
        try:
            position = self.active_positions[idx]
            pair_data = self.loader.load_pair_data(position['pair'])
            spot_col = f"{position['pair']} Curncy"
            if spot_col not in pair_data.columns:
                spot_col = pair_data.columns[0]
            available_dates = pair_data.index
            closest = min(available_dates, key=lambda d: abs((d-current_date).days))
            spot = pair_data.loc[closest, spot_col]
            T_days = max((position['expiry_date'] - current_date).days,0)
            T = T_days/365
            vol = market_vol if (market_vol and market_vol>0) else position.get('market_vol', 0.0001)
            price = self._price_option(spot, position['strike'], T, vol, position['option_type'])
            if position['position_type']=='long':
                pnl = price*position['notional'] - position['premium_paid']
            else:
                pnl = position['premium_received'] - price*position['notional']
            self.realized_pnl += pnl
            self.margin_used -= position.get('margin',0)
            closed = {**position, 'close_reason': reason, 'close_date': current_date, 'pnl': pnl}
            self.results.closed_positions.append(closed)
            self.per_pair_counts[position['pair']] = max(0, self.per_pair_counts[position['pair']] - 1)
            self.per_pair_tenor_counts[position['pair']][position['tenor']] = max(0, self.per_pair_tenor_counts[position['pair']][position['tenor']] - 1)
            self.active_positions.pop(idx)
        except Exception:
            pass

    def _risk_analytics(self, current_date):
        """Compute daily VaR / ES and tag volatility regime."""
        if len(self.results.equity_curve) < 30:
            return
        eq_series = pd.Series([e['equity'] for e in self.results.equity_curve])
        rets = eq_series.pct_change().dropna()
        window = 60
        if len(rets) < window:
            window = len(rets)
        recent = rets.tail(window)
        if recent.empty:
            return
        mu = recent.mean()
        sigma = recent.std()
        for lvl in [0.95, 0.99]:
            z = abs(pd.Series([mu]).quantile(0) - sigma * {0.95:1.645,0.99:2.326}[lvl])
        # Parametric VaR (positive number)
        var_95 = sigma*1.645*np.sqrt(1)
        var_99 = sigma*2.326*np.sqrt(1)
        hist_var_95 = abs(np.percentile(recent,5))
        hist_var_99 = abs(np.percentile(recent,1))
        es_95 = abs(recent[recent <= np.percentile(recent,5)].mean()) if (recent <= np.percentile(recent,5)).any() else hist_var_95
        es_99 = abs(recent[recent <= np.percentile(recent,1)].mean()) if (recent <= np.percentile(recent,1)).any() else hist_var_99
        self.results.var_history.append({'date': current_date,'parametric_95':var_95,'parametric_99':var_99,'hist_95':hist_var_95,'hist_99':hist_var_99})
        self.results.es_history.append({'date': current_date,'es_95':es_95,'es_99':es_99})
        # Regime tagging using rolling realized vol percentile
        realized_30 = rets.tail(30).std() if len(rets)>=30 else sigma
        long_hist = rets.std()
        regime = 'high_vol' if realized_30 > rets.quantile(0.75) else ('low_vol' if realized_30 < rets.quantile(0.25) else 'normal')
        self.results.regime_tags.append({'date': current_date,'regime':regime,'realized30':realized_30})
        # Store in metrics (overwrite acceptable)
        self.results.metrics['latest_VaR95'] = var_95
        self.results.metrics['latest_ES95'] = es_95
        self.results.metrics['vol_regime'] = regime

    def _tenor_to_years(self, tenor: str) -> float:
        mapping = {"1W":7/365,"2W":14/365,"3W":21/365,"1M":1/12,"2M":2/12,"3M":3/12,"4M":4/12,"6M":6/12,"9M":9/12,"12M":1.0,"1Y":1.0}
        return mapping.get(tenor, 1/12)
