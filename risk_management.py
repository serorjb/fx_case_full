"""
risk_management.py
Risk management and position sizing for FX options trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing
    """

    def __init__(self,
                 lookback_window: int = 100,
                 kelly_fraction: float = 0.25,  # Use 25% of full Kelly
                 max_leverage: float = 2.0):
        self.lookback_window = lookback_window
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        self.returns_history = []

    def add_return(self, ret: float):
        """Add a return to history"""
        self.returns_history.append(ret)
        if len(self.returns_history) > self.lookback_window:
            self.returns_history.pop(0)

    def calculate_position_size(self, expected_return: float,
                                win_probability: float = None) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        f* = (p*b - q) / b
        where:
        f* = fraction of capital to bet
        p = probability of winning
        b = odds (amount won on $1 bet)
        q = probability of losing (1-p)
        """
        if len(self.returns_history) < 20:
            # Not enough history, use conservative sizing
            return 0.01

        returns = np.array(self.returns_history)

        if win_probability is None:
            # Estimate from historical returns
            win_probability = np.mean(returns > 0)

        # Calculate average win and loss
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.01

        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        # Kelly formula
        b = avg_win / avg_loss if avg_loss > 0 else 1
        p = win_probability
        q = 1 - p

        kelly_size = (p * b - q) / b if b > 0 else 0

        # Apply Kelly fraction and constraints
        position_size = kelly_size * self.kelly_fraction
        position_size = max(0, min(position_size, self.max_leverage))

        return position_size


class RiskParity:
    """
    Risk Parity allocation across multiple positions
    """

    def __init__(self, target_risk: float = 0.01):
        self.target_risk = target_risk  # Target risk per position

    def calculate_weights(self, volatilities: np.ndarray,
                          correlations: np.ndarray = None) -> np.ndarray:
        """
        Calculate risk parity weights

        Weight is inversely proportional to volatility
        """
        if correlations is None:
            # Assume zero correlation
            weights = 1 / volatilities
        else:
            # Use correlation matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlations
            inv_vols = np.linalg.solve(cov_matrix, np.ones(len(volatilities)))
            weights = inv_vols / np.sum(inv_vols)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Scale to target risk
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(np.outer(volatilities, volatilities), weights)))
        scaling = self.target_risk / portfolio_vol if portfolio_vol > 0 else 1

        return weights * scaling


class VaRCalculator:
    """
    Value at Risk calculator for options portfolio
    """

    @staticmethod
    def parametric_var(portfolio_value: float, volatility: float,
                       confidence: float = 0.95, horizon: int = 1) -> float:
        """
        Calculate parametric VaR

        VaR = Portfolio Value * volatility * z-score * sqrt(horizon)
        """
        z_score = norm.ppf(1 - confidence)
        var = portfolio_value * volatility * abs(z_score) * np.sqrt(horizon / 252)
        return var

    @staticmethod
    def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate historical VaR
        """
        if len(returns) == 0:
            return 0

        percentile = (1 - confidence) * 100
        var = np.percentile(returns, percentile)
        return abs(var)

    @staticmethod
    def monte_carlo_var(portfolio_value: float, mu: float, sigma: float,
                        confidence: float = 0.95, horizon: int = 1,
                        simulations: int = 10000) -> float:
        """
        Calculate VaR using Monte Carlo simulation
        """
        # Generate random returns
        random_returns = np.random.normal(mu * horizon / 252,
                                          sigma * np.sqrt(horizon / 252),
                                          simulations)

        # Calculate portfolio values
        portfolio_values = portfolio_value * (1 + random_returns)

        # Calculate VaR
        percentile = (1 - confidence) * 100
        var_value = np.percentile(portfolio_values, percentile)
        var = portfolio_value - var_value

        return var


class StressTest:
    """
    Stress testing for options portfolio
    """

    def __init__(self):
        self.scenarios = self._define_scenarios()

    def _define_scenarios(self) -> Dict:
        """Define stress test scenarios"""
        return {
            'volatility_spike': {
                'spot_change': 0.0,
                'vol_change': 0.50,  # 50% increase in vol
                'description': 'Sudden volatility spike'
            },
            'market_crash': {
                'spot_change': -0.10,  # 10% drop
                'vol_change': 0.30,
                'description': 'Market crash scenario'
            },
            'flash_crash': {
                'spot_change': -0.05,  # 5% drop
                'vol_change': 1.00,  # Vol doubles
                'description': 'Flash crash with vol explosion'
            },
            'volatility_collapse': {
                'spot_change': 0.02,
                'vol_change': -0.30,  # 30% decrease in vol
                'description': 'Volatility collapse'
            },
            'correlation_breakdown': {
                'spot_change': 0.05,
                'vol_change': -0.10,
                'description': 'Correlation regime change'
            }
        }

    def run_stress_test(self, positions: List, spot: float,
                        bs_model) -> pd.DataFrame:
        """
        Run stress tests on portfolio
        """
        results = []

        for scenario_name, scenario in self.scenarios.items():
            stressed_spot = spot * (1 + scenario['spot_change'])

            scenario_pnl = 0
            for position in positions:
                if position.is_closed:
                    continue

                # Calculate stressed volatility
                stressed_vol = position.entry_vol * (1 + scenario['vol_change'])

                # Calculate time to expiry
                T = max((position.expiry - pd.Timestamp.now()).days / 365, 0)

                if T > 0:
                    # Calculate stressed option price
                    if position.option_type == 'call':
                        stressed_price = bs_model.call_price(
                            stressed_spot, position.strike, 0.02, 0.025,
                            stressed_vol, T
                        )
                    else:
                        stressed_price = bs_model.put_price(
                            stressed_spot, position.strike, 0.02, 0.025,
                            stressed_vol, T
                        )

                    # Calculate P&L
                    price_change = stressed_price - position.entry_price
                    position_pnl = price_change * position.position_size * spot * 100
                    scenario_pnl += position_pnl

            results.append({
                'scenario': scenario_name,
                'description': scenario['description'],
                'spot_change': scenario['spot_change'],
                'vol_change': scenario['vol_change'],
                'pnl': scenario_pnl
            })

        return pd.DataFrame(results)


class PositionLimits:
    """
    Manage position limits and concentration risk
    """

    def __init__(self,
                 max_positions: int = 50,
                 max_concentration: float = 0.10,  # Max 10% in single name
                 max_tenor_concentration: float = 0.30,  # Max 30% in one tenor
                 max_directional_bias: float = 0.60):  # Max 60% long or short

        self.max_positions = max_positions
        self.max_concentration = max_concentration
        self.max_tenor_concentration = max_tenor_concentration
        self.max_directional_bias = max_directional_bias

    def check_limits(self, positions: List, new_position) -> Tuple[bool, str]:
        """
        Check if new position violates any limits
        Returns: (is_allowed, reason)
        """
        active_positions = [p for p in positions if not p.is_closed]

        # Check max positions
        if len(active_positions) >= self.max_positions:
            return False, f"Max positions limit ({self.max_positions}) reached"

        # Check concentration by pair
        pair_exposure = {}
        for pos in active_positions:
            pair_exposure[pos.pair] = pair_exposure.get(pos.pair, 0) + abs(pos.position_size)

        new_pair_exposure = pair_exposure.get(new_position.pair, 0) + abs(new_position.position_size)
        total_exposure = sum(pair_exposure.values()) + abs(new_position.position_size)

        if new_pair_exposure / total_exposure > self.max_concentration:
            return False, f"Concentration limit exceeded for {new_position.pair}"

        # Check tenor concentration
        tenor_exposure = {}
        for pos in active_positions:
            tenor = (pos.expiry - pd.Timestamp.now()).days
            bucket = self._get_tenor_bucket(tenor)
            tenor_exposure[bucket] = tenor_exposure.get(bucket, 0) + abs(pos.position_size)

        new_tenor = (new_position.expiry - pd.Timestamp.now()).days
        new_bucket = self._get_tenor_bucket(new_tenor)
        new_tenor_exposure = tenor_exposure.get(new_bucket, 0) + abs(new_position.position_size)

        if new_tenor_exposure / total_exposure > self.max_tenor_concentration:
            return False, f"Tenor concentration limit exceeded"

        # Check directional bias
        long_exposure = sum(pos.position_size for pos in active_positions if pos.position_size > 0)
        short_exposure = abs(sum(pos.position_size for pos in active_positions if pos.position_size < 0))

        if new_position.position_size > 0:
            long_exposure += new_position.position_size
        else:
            short_exposure += abs(new_position.position_size)

        total = long_exposure + short_exposure
        if total > 0:
            long_ratio = long_exposure / total
            short_ratio = short_exposure / total

            if max(long_ratio, short_ratio) > self.max_directional_bias:
                return False, "Directional bias limit exceeded"

        return True, "OK"

    def _get_tenor_bucket(self, days: int) -> str:
        """Categorize tenor into buckets"""
        if days <= 30:
            return "1M"
        elif days <= 90:
            return "3M"
        elif days <= 180:
            return "6M"
        elif days <= 365:
            return "1Y"
        else:
            return ">1Y"


class RiskMonitor:
    """
    Real-time risk monitoring and alerts
    """

    def __init__(self,
                 max_delta: float = 100000,  # Max delta exposure
                 max_gamma: float = 10000,  # Max gamma exposure
                 max_vega: float = 50000,  # Max vega exposure
                 max_theta: float = 5000):  # Max daily theta

        self.max_delta = max_delta
        self.max_gamma = max_gamma
        self.max_vega = max_vega
        self.max_theta = max_theta
        self.alerts = []

    def check_greeks(self, portfolio_greeks: Dict) -> List[str]:
        """Check if Greeks exceed limits"""
        alerts = []

        if abs(portfolio_greeks.get('delta_exposure', 0)) > self.max_delta:
            alerts.append(f"ALERT: Delta exposure ${portfolio_greeks['delta_exposure']:,.0f} exceeds limit")

        if abs(portfolio_greeks.get('gamma_exposure', 0)) > self.max_gamma:
            alerts.append(f"ALERT: Gamma exposure ${portfolio_greeks['gamma_exposure']:,.0f} exceeds limit")

        if abs(portfolio_greeks.get('vega_exposure', 0)) > self.max_vega:
            alerts.append(f"ALERT: Vega exposure ${portfolio_greeks['vega_exposure']:,.0f} exceeds limit")

        if abs(portfolio_greeks.get('theta_exposure', 0)) > self.max_theta:
            alerts.append(f"ALERT: Theta exposure ${portfolio_greeks['theta_exposure']:,.0f} exceeds limit")

        self.alerts.extend(alerts)
        return alerts

    def calculate_margin_requirement(self, positions: List, spot: float) -> float:
        """
        Calculate margin requirement for positions
        Simplified SPAN-like calculation
        """
        margin = 0

        for position in positions:
            if position.is_closed:
                continue

            # Base margin: 10% of notional
            notional = abs(position.position_size) * spot * 100
            base_margin = notional * 0.10

            # Adjust for moneyness
            moneyness = position.strike / spot
            if 0.95 < moneyness < 1.05:
                # ATM options need more margin
                base_margin *= 1.5

            # Adjust for time to expiry
            days_to_expiry = (position.expiry - pd.Timestamp.now()).days
            if days_to_expiry < 7:
                # Near expiry needs more margin
                base_margin *= 1.5

            margin += base_margin

        return margin


# Example usage
if __name__ == "__main__":
    # Test Kelly Criterion
    kelly = KellyCriterion(kelly_fraction=0.25)

    # Add some historical returns
    for _ in range(50):
        kelly.add_return(np.random.normal(0.001, 0.02))

    optimal_size = kelly.calculate_position_size(expected_return=0.02, win_probability=0.55)
    print(f"Kelly optimal position size: {optimal_size:.2%}")

    # Test Risk Parity
    risk_parity = RiskParity(target_risk=0.01)
    volatilities = np.array([0.10, 0.15, 0.20, 0.12])
    weights = risk_parity.calculate_weights(volatilities)
    print(f"\nRisk Parity weights: {weights}")

    # Test VaR
    portfolio_value = 1_000_000
    volatility = 0.15
    var_95 = VaRCalculator.parametric_var(portfolio_value, volatility, confidence=0.95)
    print(f"\n95% VaR (1-day): ${var_95:,.0f}")

    # Test Stress Testing
    stress_test = StressTest()
    scenarios = stress_test.scenarios
    print(f"\nStress test scenarios:")
    for name, scenario in scenarios.items():
        print(f"  {name}: {scenario['description']}")

    # Test Position Limits
    limits = PositionLimits()
    print(f"\nPosition limits configured:")
    print(f"  Max positions: {limits.max_positions}")
    print(f"  Max concentration: {limits.max_concentration:.0%}")
    print(f"  Max tenor concentration: {limits.max_tenor_concentration:.0%}")

    # Test Risk Monitor
    monitor = RiskMonitor()
    test_greeks = {
        'delta_exposure': 150000,
        'gamma_exposure': 5000,
        'vega_exposure': 45000,
        'theta_exposure': 3000
    }

    alerts = monitor.check_greeks(test_greeks)
    if alerts:
        print(f"\nRisk alerts:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print(f"\nNo risk alerts - all Greeks within limits")