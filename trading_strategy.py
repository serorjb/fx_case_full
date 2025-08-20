"""
FX Options Trading Strategy Engine
Implements systematic volatility arbitrage strategies with Greeks management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class SignalType(Enum):
    """Types of trading signals"""
    OVERPRICED = "overpriced"
    UNDERPRICED = "underpriced"
    NEUTRAL = "neutral"


@dataclass
class OptionPosition:
    """Represents an option position"""
    pair: str
    strike: float
    expiry: pd.Timestamp
    option_type: str  # 'call' or 'put'
    position_size: float  # Positive for long, negative for short
    entry_price: float
    entry_vol: float
    entry_date: pd.Timestamp
    model_vol: float  # Model implied volatility
    market_vol: float  # Market implied volatility
    entry_spot: float
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_vega: float = 0.0
    entry_theta: float = 0.0
    pnl: float = 0.0
    is_closed: bool = False
    close_date: Optional[pd.Timestamp] = None
    close_price: Optional[float] = None
    close_reason: Optional[str] = None


@dataclass
class TradingSignal:
    """Trading signal for an option"""
    pair: str
    strike: float
    tenor: str
    expiry: pd.Timestamp
    option_type: str
    signal_type: SignalType
    market_vol: float
    model_vol: float
    vol_diff: float  # market_vol - model_vol
    expected_edge: float  # Expected profit in vol points
    confidence: float  # Signal confidence [0, 1]
    recommended_size: float  # Recommended position size


class VolatilityArbitrageStrategy:
    """
    Core volatility arbitrage strategy
    Identifies mispriced options and manages positions
    """

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 max_position_size: float = 0.02,  # Max 2% per position
                 vol_threshold: float = 0.01,  # 1% vol difference threshold
                 max_positions: int = 50,
                 target_delta_neutral: bool = True,
                 margin_requirement: float = 0.10,  # 10% margin requirement
                 transaction_cost: float = 0.0002):  # 2bp transaction cost

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.vol_threshold = vol_threshold
        self.max_positions = max_positions
        self.target_delta_neutral = target_delta_neutral
        self.margin_requirement = margin_requirement
        self.transaction_cost = transaction_cost  # Store as instance variable

        self.positions: List[OptionPosition] = []
        self.closed_positions: List[OptionPosition] = []
        self.portfolio_metrics = {}

        # Track margin
        self.margin_used = 0.0
        self.margin_available = initial_capital * 0.5  # 50% of capital for margin

    def identify_opportunities(self,
                              market_surface: Dict,
                              model_surface: Dict,
                              spot: float,
                              date: pd.Timestamp) -> List[TradingSignal]:
        """
        Systematically identify ALL trading opportunities
        Trade whenever expected profit > transaction costs
        """
        signals = []

        # Use instance transaction cost
        min_edge = self.transaction_cost + 0.0003  # Transaction cost + 3bp minimum edge

        for tenor in market_surface.keys():
            if tenor not in model_surface:
                continue

            market_data = market_surface[tenor]
            model_data = model_surface[tenor]

            # Get market and model vols - handle both dict and float formats
            if isinstance(market_data, dict):
                market_vol = market_data.get('vol', 0)
            else:
                market_vol = float(market_data)

            if isinstance(model_data, dict):
                model_vol = model_data.get('vol', 0)
            else:
                model_vol = float(model_data)

            if market_vol == 0 or model_vol == 0:
                continue

            # Calculate vol difference
            vol_diff = market_vol - model_vol

            # Trade if edge exceeds transaction costs
            if abs(vol_diff) > min_edge:
                # Determine signal type
                if vol_diff > 0:
                    # Market vol > Model vol: SELL volatility
                    signal_type = SignalType.OVERPRICED
                    recommended_size = -1  # Short
                else:
                    # Market vol < Model vol: BUY volatility
                    signal_type = SignalType.UNDERPRICED
                    recommended_size = 1  # Long

                # Calculate expected edge (profit after transaction costs)
                expected_edge = abs(vol_diff) - self.transaction_cost

                # Calculate confidence based on vol difference magnitude
                confidence = min(abs(vol_diff) / 0.02, 1.0)  # Max confidence at 2% difference

                # Create signal
                expiry = date + pd.Timedelta(days=self._tenor_to_days(tenor))

                signal = TradingSignal(
                    pair="",  # Will be set by backtester
                    strike=spot,  # ATM for now
                    tenor=tenor,
                    expiry=expiry,
                    option_type='call',  # Default to calls
                    signal_type=signal_type,
                    market_vol=market_vol,
                    model_vol=model_vol,
                    vol_diff=vol_diff,
                    expected_edge=expected_edge,
                    confidence=confidence,
                    recommended_size=recommended_size
                )
                signals.append(signal)

            # Also check for strike-specific opportunities if available
            if isinstance(market_data, dict) and 'strikes' in market_data:
                for strike_info in market_data.get('strikes', []):
                    strike = strike_info.get('strike', spot)
                    market_strike_vol = strike_info.get('vol', market_vol)

                    # Get corresponding model vol for this strike
                    model_strike_vol = model_vol  # Default to ATM
                    if isinstance(model_data, dict) and 'strikes' in model_data:
                        for model_strike_info in model_data['strikes']:
                            if abs(model_strike_info.get('strike', 0) - strike) < 0.0001:
                                model_strike_vol = model_strike_info.get('vol', model_vol)
                                break

                    vol_diff_strike = market_strike_vol - model_strike_vol

                    if abs(vol_diff_strike) > min_edge:
                        if vol_diff_strike > 0:
                            signal_type_strike = SignalType.OVERPRICED
                            position_size_strike = -1
                        else:
                            signal_type_strike = SignalType.UNDERPRICED
                            position_size_strike = 1

                        expected_edge_strike = abs(vol_diff_strike) - self.transaction_cost
                        confidence_strike = min(abs(vol_diff_strike) / 0.02, 1.0)

                        # Determine option type based on moneyness
                        option_type = 'call' if strike >= spot else 'put'

                        signal = TradingSignal(
                            pair="",
                            strike=strike,
                            tenor=tenor,
                            expiry=date + pd.Timedelta(days=self._tenor_to_days(tenor)),
                            option_type=option_type,
                            signal_type=signal_type_strike,
                            market_vol=market_strike_vol,
                            model_vol=model_strike_vol,
                            vol_diff=vol_diff_strike,
                            expected_edge=expected_edge_strike,
                            confidence=confidence_strike,
                            recommended_size=position_size_strike
                        )
                        signals.append(signal)

        # Sort by expected edge (best opportunities first)
        signals.sort(key=lambda x: x.expected_edge, reverse=True)

        # Return all profitable opportunities
        return signals

    def _tenor_to_days(self, tenor: str) -> int:
        """Convert tenor string to days"""
        tenor_map = {
            '1W': 7, '2W': 14, '3W': 21, '1M': 30, '2M': 60,
            '3M': 90, '4M': 120, '6M': 180, '9M': 270, '12M': 365
        }
        return tenor_map.get(tenor, 30)

    def execute_signal(self,
                       signal: TradingSignal,
                       spot: float,
                       date: pd.Timestamp,
                       bs_model,
                       r_d: float = None,
                       r_f: float = None) -> Optional[OptionPosition]:
        """
        Execute a trading signal and create a position
        """
        # Check portfolio constraints
        if len(self.positions) >= self.max_positions:
            return None

        # Use provided rates or defaults
        if r_d is None:
            r_d = 0.02
        if r_f is None:
            r_f = 0.025

        T = (signal.expiry - date).days / 365

        if signal.option_type == 'call':
            option_price = bs_model.call_price(spot, signal.strike, r_d, r_f,
                                              signal.market_vol, T)
        else:
            option_price = bs_model.put_price(spot, signal.strike, r_d, r_f,
                                             signal.market_vol, T)

        # Position sizing: allocate fraction of capital to premium, derive units
        risk_capital = self.current_capital * self.max_position_size
        if option_price <= 0:
            return None
        units = (risk_capital / option_price) * np.sign(signal.recommended_size)

        # Calculate Greeks
        delta = bs_model.delta(spot, signal.strike, r_d, r_f, signal.market_vol, T,
                              signal.option_type)
        gamma = bs_model.gamma(spot, signal.strike, r_d, r_f, signal.market_vol, T)
        vega = bs_model.vega(spot, signal.strike, r_d, r_f, signal.market_vol, T)
        theta = bs_model.theta(spot, signal.strike, r_d, r_f, signal.market_vol, T,
                              signal.option_type)

        # Create position
        position = OptionPosition(
            pair=signal.pair,
            strike=signal.strike,
            expiry=signal.expiry,
            option_type=signal.option_type,
            position_size=units,
            entry_price=option_price,
            entry_vol=signal.market_vol,
            entry_date=date,
            model_vol=signal.model_vol,
            market_vol=signal.market_vol,
            entry_spot=spot,
            entry_delta=delta,
            entry_gamma=gamma,
            entry_vega=vega,
            entry_theta=theta
        )

        self.positions.append(position)
        return position

    def update_positions(self, spot: float, date: pd.Timestamp,
                        current_vol: float, bs_model) -> None:
        """
        Update existing positions and apply risk management
        """
        positions_to_close = []

        for i, pos in enumerate(self.positions):
            if pos.is_closed:
                continue

            T = (pos.expiry - date).days / 365

            if T <= 0:
                # Option expired
                positions_to_close.append((i, "expiry"))
                continue

            # Calculate current option value
            if pos.option_type == 'call':
                current_price = bs_model.call_price(spot, pos.strike, 0.02, 0.025,
                                                   current_vol, T)
            else:
                current_price = bs_model.put_price(spot, pos.strike, 0.02, 0.025,
                                                  current_vol, T)

            # Update P&L
            pos.pnl = (current_price - pos.entry_price) * pos.position_size

            # Risk management rules
            # 1. Take profit if vol has mean-reverted significantly
            vol_change = current_vol - pos.entry_vol
            expected_vol_change = pos.model_vol - pos.entry_vol

            if abs(vol_change) > abs(expected_vol_change) * 0.7:
                positions_to_close.append((i, "take_profit"))

            # 2. Stop loss if position moves against us significantly
            if pos.pnl < -self.current_capital * 0.001:  # 0.1% stop loss
                positions_to_close.append((i, "stop_loss"))

            # 3. Close if approaching expiry (5 days)
            if T < 5/365:
                positions_to_close.append((i, "approaching_expiry"))

        # Close marked positions
        for idx, reason in positions_to_close:
            self.close_position(idx, spot, date, reason, bs_model)

    def close_position(self, position_idx: int, spot: float, date: pd.Timestamp,
                      reason: str, bs_model) -> None:
        """
        Close a position and record the trade
        """
        if position_idx >= len(self.positions):
            return

        pos = self.positions[position_idx]
        if pos.is_closed:
            return

        T = max((pos.expiry - date).days / 365, 0)

        # Calculate exit price
        if T > 0:
            # Use current market vol (simplified)
            exit_vol = pos.entry_vol  # Would get from market

            if pos.option_type == 'call':
                exit_price = bs_model.call_price(spot, pos.strike, 0.02, 0.025,
                                                exit_vol, T)
            else:
                exit_price = bs_model.put_price(spot, pos.strike, 0.02, 0.025,
                                               exit_vol, T)
        else:
            # Expired - calculate intrinsic value
            if pos.option_type == 'call':
                exit_price = max(spot - pos.strike, 0)
            else:
                exit_price = max(pos.strike - spot, 0)

        # Update position
        pos.is_closed = True
        pos.close_date = date
        pos.close_price = exit_price
        pos.close_reason = reason

        # Final P&L calculation (per-unit pricing, no extra spot multiplier)
        pos.pnl = (exit_price - pos.entry_price) * pos.position_size

        # Update capital
        self.current_capital += pos.pnl

        # Move to closed positions
        self.closed_positions.append(pos)

    def calculate_portfolio_greeks(self, spot: float, date: pd.Timestamp,
                                  bs_model) -> Dict[str, float]:
        """
        Calculate aggregate portfolio Greeks
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        for pos in self.positions:
            if pos.is_closed:
                continue

            T = (pos.expiry - date).days / 365
            if T <= 0:
                continue

            # Recalculate Greeks with current spot
            delta = bs_model.delta(spot, pos.strike, 0.02, 0.025, pos.entry_vol, T,
                                  pos.option_type)
            gamma = bs_model.gamma(spot, pos.strike, 0.02, 0.025, pos.entry_vol, T)
            vega = bs_model.vega(spot, pos.strike, 0.02, 0.025, pos.entry_vol, T)
            theta = bs_model.theta(spot, pos.strike, 0.02, 0.025, pos.entry_vol, T,
                                 pos.option_type)

            # Aggregate (accounting for position sign)
            total_delta += delta * pos.position_size
            total_gamma += gamma * pos.position_size
            total_vega += vega * pos.position_size
            total_theta += theta * pos.position_size

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta
        }

    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate strategy performance metrics
        """
        if not self.closed_positions:
            return {}

        # Extract P&L series
        pnl_series = [pos.pnl for pos in self.closed_positions]

        if not pnl_series:
            return {}

        total_pnl = sum(pnl_series)
        win_rate = sum(1 for pnl in pnl_series if pnl > 0) / len(pnl_series)

        # Calculate returns
        returns = []
        capital = self.initial_capital
        for pnl in pnl_series:
            ret = pnl / capital
            returns.append(ret)
            capital += pnl

        if returns:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            max_dd = self._calculate_max_drawdown(returns)
        else:
            sharpe = 0
            max_dd = 0

        return {
            'total_pnl': total_pnl,
            'total_return': total_pnl / self.initial_capital,
            'win_rate': win_rate,
            'num_trades': len(self.closed_positions),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_capital': self.current_capital
        }

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = []
        cum_ret = 1.0
        for ret in returns:
            cum_ret *= (1 + ret)
            cumulative.append(cum_ret)

        peak = cumulative[0]
        max_dd = 0.0

        for value in cumulative:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd


class CarryToVolStrategy:
    """
    Carry-to-Volatility ratio strategy
    Trade based on forward premium vs implied volatility
    """

    def __init__(self, carry_threshold: float = 1.5):
        self.carry_threshold = carry_threshold

    def calculate_carry_to_vol(self, spot: float, forward: float,
                               vol: float, T: float) -> float:
        """
        Calculate carry-to-volatility ratio
        Carry = (F-S)/S annualized
        """
        carry = (forward - spot) / spot / T
        carry_to_vol = carry / vol
        return carry_to_vol

    def generate_signal(self, spot: float, forward: float,
                        vol: float, T: float) -> Optional[TradingSignal]:
        """
        Generate trading signal based on carry-to-vol ratio
        """
        ratio = self.calculate_carry_to_vol(spot, forward, vol, T)

        if abs(ratio) > self.carry_threshold:
            # High carry relative to vol - potential opportunity
            if ratio > 0:
                # Positive carry - consider selling vol
                signal_type = SignalType.OVERPRICED
                recommended_size = -1
            else:
                # Negative carry - consider buying vol
                signal_type = SignalType.UNDERPRICED
                recommended_size = 1

            confidence = min(abs(ratio) / (self.carry_threshold * 2), 1.0)

            signal = TradingSignal(
                pair="",
                strike=spot,
                tenor="",
                expiry=pd.Timestamp.now() + pd.Timedelta(days=T*365),
                option_type='call',
                signal_type=signal_type,
                market_vol=vol,
                model_vol=vol * (1 - 0.1 * np.sign(ratio)),  # Adjust by 10%
                vol_diff=vol * 0.1 * np.sign(ratio),
                expected_edge=abs(ratio) / 100,
                confidence=confidence,
                recommended_size=recommended_size
            )
            return signal

        return None


# Example usage
if __name__ == "__main__":
    # Create strategy
    strategy = VolatilityArbitrageStrategy(
        initial_capital=1_000_000,
        max_position_size=0.02,
        vol_threshold=0.005
    )

    # Create sample market and model surfaces
    market_surface = {
        '1M': {'vol': 0.12},  # 12% implied vol
        '3M': {'vol': 0.11},  # 11% implied vol
        '6M': {'vol': 0.10}   # 10% implied vol
    }

    model_surface = {
        '1M': {'vol': 0.10},  # Model thinks 1M is overpriced
        '3M': {'vol': 0.11},  # 3M is fair
        '6M': {'vol': 0.11}   # 6M is underpriced
    }

    # Identify opportunities
    spot = 1.0755
    date = pd.Timestamp.now()

    signals = strategy.identify_opportunities(
        market_surface, model_surface, spot, date
    )

    print(f"Found {len(signals)} trading opportunities:")
    for signal in signals[:3]:
        print(f"\n{signal.signal_type.value} signal:")
        print(f"  Strike: {signal.strike:.4f}, Tenor: {signal.tenor}")
        print(f"  Market Vol: {signal.market_vol:.2%}, Model Vol: {signal.model_vol:.2%}")
        print(f"  Expected Edge: {signal.expected_edge:.2%}, Confidence: {signal.confidence:.2f}")

    # Execute a signal
    if signals:
        from pricing_models import BlackScholesFX
        bs_model = BlackScholesFX()
        position = strategy.execute_signal(signals[0], spot, date, bs_model)
        if position:
            print(f"\nExecuted position:")
            print(f"  Size: {position.position_size:.2f} contracts")
            print(f"  Entry Price: {position.entry_price:.5f}")
            print(f"  Greeks - Delta: {position.entry_delta:.4f}, Vega: {position.entry_vega:.4f}")

    # Calculate portfolio Greeks
    portfolio_greeks = strategy.calculate_portfolio_greeks(spot, date, bs_model)
    print(f"\nPortfolio Greeks:")
    print(f"  Total Delta: {portfolio_greeks['delta']:.4f}")
    print(f"  Total Vega: {portfolio_greeks['vega']:.4f}")

    # Test Carry-to-Vol strategy
    carry_strategy = CarryToVolStrategy(carry_threshold=1.5)
    forward = spot * 1.005  # 0.5% forward premium
    vol = 0.10
    T = 30/365

    carry_signal = carry_strategy.generate_signal(spot, forward, vol, T)
    if carry_signal:
        print(f"\nCarry-to-Vol Signal Generated:")
        print(f"  Type: {carry_signal.signal_type.value}")
        print(f"  Confidence: {carry_signal.confidence:.2f}")