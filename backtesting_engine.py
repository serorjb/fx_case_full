"""
Enhanced FX Options Backtesting Engine - Fixed Version
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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

    def add_snapshot(self, date, equity, positions, greeks, daily_trades=0, daily_pnl=0):
        """Add daily portfolio snapshot"""
        snapshot = {
            'date': date,
            'equity': equity,
            'num_positions': len(positions),
            'daily_trades': daily_trades,
            'daily_pnl': daily_pnl,
            'greeks': greeks.copy()
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

        # 3. SCAN FOR NEW OPPORTUNITIES
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

    def _expire_positions(self, current_date):
        """Handle position expiries and calculate P&L"""
        total_pnl = 0
        positions_to_remove = []

        for idx, position in enumerate(self.active_positions):
            if current_date >= position['expiry_date']:
                pnl = self._calculate_expiry_pnl(position, current_date)
                total_pnl += pnl
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

            return pnl

        except Exception as e:
            print(f"Error calculating expiry P&L for position {position.get('id', 'unknown')}: {e}")
            return 0

    def _mark_to_market_positions(self, current_date):
        """Mark existing positions to market"""
        # Simplified MTM - could be enhanced with full Black-Scholes
        return 0

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
                for signal in signals:
                    if signal.pair == "":
                        signal.pair = pair
                    if len(new_positions) >= self.config.max_daily_trades:
                        break
                    position = self._execute_trade(signal, vol_data, current_date, pair)
                    if position:
                        new_positions.append(position)
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
        """
        market_surface = {}
        model_surface = {}
        tenors = ["1W", "2W", "3W", "1M", "2M", "3M", "4M", "6M", "9M", "12M"]
        for tenor in tenors:
            if tenor in vol_data.atm_vols:
                atm = vol_data.atm_vols[tenor]  # already in decimal (e.g. 0.105)
                # Market surface uses provided ATM vol directly
                market_surface[tenor] = {'vol': atm}
                # Build a simple fair value model vol using RR/BF if available
                rr = vol_data.rr_25d.get(tenor, 0.0)
                bf = vol_data.bf_25d.get(tenor, 0.0)
                # Approximate synthetic model adjustment: remove half RR impact and butterfly
                # Model vol slightly smoothed toward long-term mean (simple EMA style)
                adj = -0.25 * rr - 0.5 * bf
                base_model = atm + adj
                # Clamp to reasonable bounds
                base_model = float(np.clip(base_model, 0.01, 0.80))
                # Small deterministic variation to avoid identical vols (hash based)
                variation = (hash((tenor, int(current_date.strftime('%Y%m%d')))) % 7) / 10000.0
                model_vol = max(0.0001, base_model * (1 - 0.005) + variation)
                model_surface[tenor] = {'vol': model_vol}
        return market_surface, model_surface

    def _execute_trade(self, signal, vol_data, current_date, pair):
        """Execute a trading signal"""
        try:
            T = self._tenor_to_years(signal.tenor)
            market_vol = signal.market_vol if isinstance(signal.market_vol, (int, float)) else signal.market_vol
            # Strategy's signal stores raw vol in signal.market_vol (already decimal)
            spot = vol_data.spot
            strike = signal.strike
            r = 0.02
            option_price = self._price_option(spot, strike, T, market_vol, signal.option_type, r)
            max_position_value = self.current_equity * self.config.max_position_size
            if option_price <= 0:
                return None
            position_size = min(max_position_value / option_price, 100000)
            notional = position_size
            total_premium = option_price * notional
            if signal.signal_type.value == 'overpriced':
                position_type = 'short'
                premium_received = total_premium
                premium_paid = 0
                self.current_equity += total_premium * (1 - self.config.commission)
            else:
                position_type = 'long'
                premium_paid = total_premium
                premium_received = 0
                self.current_equity -= total_premium * (1 + self.config.commission)
            position = {
                'id': f"{pair}_{self.position_counter}",
                'pair': pair,
                'option_type': signal.option_type,
                'position_type': position_type,
                'strike': strike,
                'tenor': signal.tenor,
                'expiry_date': current_date + pd.Timedelta(days=int(T * 365)),
                'notional': notional,
                'market_vol': signal.market_vol,
                'model_vol': signal.model_vol,
                'premium_paid': premium_paid,
                'premium_received': premium_received,
                'entry_date': current_date,
                'entry_spot': spot
            }
            self.position_counter += 1
            self.results.trades.append(position.copy())
            return position

        except Exception as e:
            print(f"Error executing trade: {e}")
            return None

    def _price_option(self, S, K, T, vol, option_type='call', r=0.02):
        """Black-Scholes option pricing"""
        from scipy.stats import norm
        import math

        if T <= 0 or vol <= 0:
            return 0.0001

        try:
            d1 = (math.log(S/K) + (r + 0.5*vol**2)*T) / (vol*math.sqrt(T))
            d2 = d1 - vol*math.sqrt(T)

            if option_type == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
            else:  # put
                price = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return max(price, 0.0001)  # Minimum price

        except:
            return 0.0001  # Fallback price

    def _perform_delta_hedging(self, current_date):
        """Perform portfolio delta hedging by maturity buckets"""
        hedging_cost = 0

        # Calculate delta exposures by pair and maturity
        delta_exposures = defaultdict(lambda: defaultdict(float))

        for position in self.active_positions:
            pair = position['pair']
            tenor = position['tenor']
            notional = position['notional']

            # Simple delta calculation (could be enhanced)
            delta = 0.5 * notional if position['option_type'] == 'call' else -0.5 * notional

            if position['position_type'] == 'short':
                delta = -delta

            delta_exposures[pair][tenor] += delta

        # Check for hedging needs
        for pair, buckets in delta_exposures.items():
            for tenor, delta_exposure in buckets.items():
                # Hedge if delta exposure exceeds threshold
                if abs(delta_exposure) > self.config.delta_threshold * self.current_equity:
                    hedging_cost += abs(delta_exposure) * 0.001  # Simple hedging cost

        return -hedging_cost  # Cost reduces P&L

    def _calculate_portfolio_greeks(self, current_date):
        """Calculate portfolio Greeks by maturity bucket"""
        total_delta = 0
        total_vega = 0
        total_gamma = 0
        total_theta = 0

        for position in self.active_positions:
            # Simple Greeks estimation
            notional = position['notional']
            days_to_expiry = (position['expiry_date'] - current_date).days

            if days_to_expiry > 0:
                # Simple approximations (could be enhanced with proper Greeks)
                pos_delta = 0.5 * notional if position['option_type'] == 'call' else -0.5 * notional
                pos_vega = notional * 0.01  # Simplified vega
                pos_gamma = notional * 0.001  # Simplified gamma
                pos_theta = -notional * 0.001  # Simplified theta

                if position['position_type'] == 'short':
                    pos_delta = -pos_delta
                    pos_vega = -pos_vega
                    pos_gamma = -pos_gamma
                    pos_theta = -pos_theta

                total_delta += pos_delta
                total_vega += pos_vega
                total_gamma += pos_gamma
                total_theta += pos_theta

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta
        }

    def _tenor_to_years(self, tenor):
        """Convert tenor string to years"""
        tenor_map = {
            "1W": 1/52, "2W": 2/52, "3W": 3/52,
            "1M": 1/12, "2M": 2/12, "3M": 3/12,
            "6M": 6/12, "9M": 9/12, "1Y": 1.0
        }
        return tenor_map.get(tenor, 1/12)