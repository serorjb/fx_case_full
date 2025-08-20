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
                 target_delta_neutral: bool = True):

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.vol_threshold = vol_threshold
        self.max_positions = max_positions
        self.target_delta_neutral = target_delta_neutral

        self.positions: List[OptionPosition] = []
        self.closed_positions: List[OptionPosition] = []
        self.portfolio_metrics = {}

    def identify_opportunities(self,
                              market_surface: Dict,
                              model_surface: Dict,
                              spot: float,
                              date: pd.Timestamp) -> List[TradingSignal]:
        """
        Identify trading opportunities by comparing market vs model surfaces
        """
        signals = []

        for tenor in market_surface.keys():
            if tenor not in model_surface:
                continue

            market_data = market_surface[tenor]
            model_data = model_surface[tenor]

            # Get market and model vols
            market_vol = market_data.get('vol', 0)
            model_vol = model_data.get('vol', 0)

            if market_vol == 0 or model_vol == 0:
                continue

            # Calculate vol difference
            vol_diff = market_vol - model_vol

            # Check if difference is significant (use absolute value for threshold check)
            if abs(vol_diff) > self.vol_threshold:
                # Determine signal type and option to trade
                if vol_diff > self.vol_threshold:
                    # Market vol > Model vol: Sell volatility
                    signal_type = SignalType.OVERPRICED
                    recommended_size = -1  # Short
                else:
                    # Market vol < Model vol: Buy volatility
                    signal_type = SignalType.UNDERPRICED
                    recommended_size = 1  # Long

                # For ATM options, we'll trade straddles
                # For OTM options, we'll trade the appropriate single option
                strike = spot  # ATM strike
                option_type = 'straddle' if abs(strike - spot) / spot < 0.01 else 'call'

                # Calculate expected edge
                expected_edge = abs(vol_diff) * 0.5  # Assume 50% mean reversion

                # Calculate confidence based on vol difference magnitude
                confidence = min(abs(vol_diff) / (self.vol_threshold * 3), 1.0)

                # Adjust size based on confidence
                recommended_size *= confidence * self.max_position_size

                # Create signal
                signal = TradingSignal(
                    pair="AUDNZD",
                    strike=strike,
                    tenor=tenor,
                    expiry=date + pd.Timedelta(days=self._tenor_to_days(tenor)),
                    option_type=option_type,
                    signal_type=signal_type,
                    market_vol=market_vol,
                    model_vol=model_vol,
                    vol_diff=vol_diff,
                    expected_edge=expected_edge,
                    confidence=confidence,
                    recommended_size=recommended_size
                )

                signals.append(signal)

            # Also check for relative value opportunities across strikes
            if 'strikes' in model_data and 'vols' in model_data:
                model_strikes = model_data['strikes']
                model_vols_array = model_data['vols']

                # Sample a few strike points
                for i, strike in enumerate(model_strikes[::2]):  # Check every other strike
                    if i < len(model_vols_array):
                        model_strike_vol = model_vols_array[i]
                        # Estimate market vol at this strike (simplified)
                        strike_moneyness = strike / spot
                        market_strike_vol = market_vol * (1 + 0.1 * (strike_moneyness - 1))

                        vol_diff_strike = market_strike_vol - model_strike_vol

                        if abs(vol_diff_strike) > self.vol_threshold * 0.75:  # Lower threshold for individual strikes
                            signal_type = SignalType.OVERPRICED if vol_diff_strike > 0 else SignalType.UNDERPRICED
                            option_type = 'put' if strike < spot else 'call'

                            signal = TradingSignal(
                                pair="AUDNZD",
                                strike=strike,
                                tenor=tenor,
                                expiry=date + pd.Timedelta(days=self._tenor_to_days(tenor)),
                                option_type=option_type,
                                signal_type=signal_type,
                                market_vol=market_strike_vol,
                                model_vol=model_strike_vol,
                                vol_diff=vol_diff_strike,
                                expected_edge=abs(vol_diff_strike) * 0.4,
                                confidence=min(abs(vol_diff_strike) / (self.vol_threshold * 2), 1.0),
                                recommended_size=np.sign(-vol_diff_strike) * self.max_position_size * 0.5
                            )
                            signals.append(signal)

        # Sort by expected edge
        signals.sort(key=lambda x: abs(x.expected_edge), reverse=True)

        return signals[:10]  # Return top 10 opportunities

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
                       bs_model) -> Optional[OptionPosition]:
        """
        Execute a trading signal and create a position
        """
        # Check portfolio constraints
        if len(self.positions) >= self.max_positions:
            return None

        # Calculate option price using market vol
        r_d, r_f = 0.02, 0.025  # Simplified rates
        T = (signal.expiry - date).days / 365

        if signal.option_type == 'call':
            option_price = bs_model.call_price(spot, signal.strike, r_d, r_f,
                                              signal.market_vol, T)
        else:
            option_price = bs_model.put_price(spot, signal.strike, r_d, r_f,
                                             signal.market_vol, T)

        # Calculate Greeks
        delta = bs_model.delta(spot, signal.strike, r_d, r_f, signal.market_vol, T,
                              signal.option_type)
        gamma = bs_model.gamma(spot, signal.strike, r_d, r_f, signal.market_vol, T)
        vega = bs_model.vega(spot, signal.strike, r_d, r_f, signal.market_vol, T)
        theta = bs_model.theta(spot, signal.strike, r_d, r_f, signal.market_vol, T,
                              signal.option_type)

        # Calculate position size in contracts
        notional = self.current_capital * abs(signal.recommended_size)
        position_size = notional / (option_price * spot * 100)  # Assuming 100 multiplier

        # Create position
        position = OptionPosition(
            pair=signal.pair,
            strike=signal.strike,
            expiry=signal.expiry,
            option_type=signal.option_type,
            position_size=position_size * np.sign(signal.recommended_size),
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

    def calculate_portfolio_greeks(self, spot: float, date: pd.Timestamp,
                                   bs_model) -> Dict:
        """
        Calculate aggregate portfolio Greeks
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        for pos in self.positions:
            if not pos.is_closed:
                T = max((pos.expiry - date).days / 365, 0)

                if T > 0:
                    # Recalculate Greeks at current spot
                    delta = bs_model.delta(spot, pos.strike, 0.02, 0.025,
                                          pos.entry_vol, T, pos.option_type)
                    gamma = bs_model.gamma(spot, pos.strike, 0.02, 0.025,
                                          pos.entry_vol, T)
                    vega = bs_model.vega(spot, pos.strike, 0.02, 0.025,
                                        pos.entry_vol, T)
                    theta = bs_model.theta(spot, pos.strike, 0.02, 0.025,
                                          pos.entry_vol, T, pos.option_type)

                    # Aggregate
                    total_delta += delta * pos.position_size
                    total_gamma += gamma * pos.position_size
                    total_vega += vega * pos.position_size
                    total_theta += theta * pos.position_size

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta,
            'delta_exposure': total_delta * spot,
            'gamma_exposure': total_gamma * spot**2 / 100,
            'vega_exposure': total_vega,
            'theta_exposure': total_theta
        }

    def hedge_delta(self, portfolio_greeks: Dict, spot: float) -> float:
        """
        Calculate spot hedge required for delta neutrality
        """
        if not self.target_delta_neutral:
            return 0.0

        return -portfolio_greeks['delta']

    def update_positions(self, spot: float, date: pd.Timestamp,
                        market_vols: Dict, bs_model) -> None:
        """
        Update position P&L and check for exits
        """
        positions_to_close = []

        for i, pos in enumerate(self.positions):
            if pos.is_closed:
                continue

            T = (pos.expiry - date).days / 365

            # Check if expired
            if T <= 0:
                positions_to_close.append((i, "expired"))
                continue

            # Get current market vol (simplified)
            current_vol = market_vols.get(pos.strike, pos.entry_vol)

            # Calculate current option value
            if pos.option_type == 'call':
                current_price = bs_model.call_price(spot, pos.strike, 0.02, 0.025,
                                                   current_vol, T)
            else:
                current_price = bs_model.put_price(spot, pos.strike, 0.02, 0.025,
                                                  current_vol, T)

            # Calculate P&L
            price_change = current_price - pos.entry_price
            pos.pnl = price_change * pos.position_size * spot * 100

            # Check exit conditions
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

        # Final P&L calculation
        pos.pnl = (exit_price - pos.entry_price) * pos.position_size * spot * 100

        # Update capital
        self.current_capital += pos.pnl

        # Move to closed positions
        self.closed_positions.append(pos)

    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate strategy performance metrics
        """
        if not self.closed_positions:
            return {}

        # Extract P&L series
        pnls = [pos.pnl for pos in self.closed_positions]

        # Calculate metrics
        total_pnl = sum(pnls)
        num_trades = len(pnls)
        win_rate = sum(1 for pnl in pnls if pnl > 0) / num_trades if num_trades > 0 else 0

        # Calculate returns
        returns = [pnl / self.initial_capital for pnl in pnls]

        # Sharpe ratio (simplified - assumes daily returns)
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = np.sqrt(252) * avg_return / std_return if std_return > 0 else 0
        else:
            sharpe = 0

        # Maximum drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / self.initial_capital
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        return {
            'total_pnl': total_pnl,
            'total_return': total_pnl / self.initial_capital,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'current_capital': self.current_capital,
            'active_positions': len([p for p in self.positions if not p.is_closed])
        }


class CarryToVolStrategy:
    """
    Carry to Volatility ratio strategy
    Sells options when carry/vol ratio is attractive
    """

    def __init__(self,
                 carry_threshold: float = 1.5,
                 min_vol: float = 0.05,
                 max_vol: float = 0.30):

        self.carry_threshold = carry_threshold
        self.min_vol = min_vol
        self.max_vol = max_vol

    def calculate_carry(self, forward: float, spot: float, T: float) -> float:
        """
        Calculate carry (forward premium/discount) annualized
        """
        if T <= 0:
            return 0.0

        return (forward / spot - 1) / T

    def generate_signal(self, spot: float, forward: float, vol: float,
                       T: float) -> Optional[TradingSignal]:
        """
        Generate trading signal based on carry/vol ratio
        """
        # Calculate carry
        carry = self.calculate_carry(forward, spot, T)

        # Calculate carry to vol ratio
        if vol < self.min_vol or vol > self.max_vol:
            return None

        carry_vol_ratio = abs(carry) / vol

        # Generate signal if ratio is attractive
        if carry_vol_ratio > self.carry_threshold:
            # Sell volatility (sell straddle/strangle)
            signal_type = SignalType.OVERPRICED
            confidence = min(carry_vol_ratio / (self.carry_threshold * 2), 1.0)

            return TradingSignal(
                pair="AUDNZD",
                strike=forward,  # ATM
                tenor=f"{int(T*365)}D",
                expiry=pd.Timestamp.now() + pd.Timedelta(days=int(T*365)),
                option_type='straddle',
                signal_type=signal_type,
                market_vol=vol,
                model_vol=vol * 0.9,  # Expect vol to decrease
                vol_diff=vol * 0.1,
                expected_edge=carry * 0.5,
                confidence=confidence,
                recommended_size=-0.01  # Short position
            )

        return None


# Example usage
if __name__ == "__main__":
    from fx_pricing_models import BlackScholesFX

    # Initialize strategy
    strategy = VolatilityArbitrageStrategy(
        initial_capital=1_000_000,
        max_position_size=0.02,
        vol_threshold=0.01
    )

    # Create sample market and model surfaces
    market_surface = {
        '1M': {'vol': 0.12},
        '3M': {'vol': 0.11},
        '6M': {'vol': 0.10}
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