"""
risk_management.py
Risk management and position sizing for FX options trading
Complete implementation of Kelly criterion, risk parity, VaR, and stress testing
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
        self.win_loss_history = []

    def add_return(self, ret: float):
        """Add a return to history"""
        self.returns_history.append(ret)
        if len(self.returns_history) > self.lookback_window:
            self.returns_history.pop(0)

        # Track win/loss
        self.win_loss_history.append(1 if ret > 0 else 0)
        if len(self.win_loss_history) > self.lookback_window:
            self.win_loss_history.pop(0)

    def calculate_position_size(self, expected_return: float,
                                win_probability: float = None,
                                current_capital: float = 10_000_000) -> float:
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
            return min(0.01 * current_capital, current_capital * 0.001)

        returns = np.array(self.returns_history)

        if win_probability is None:
            # Estimate from historical returns
            win_probability = np.mean(returns > 0)

        # Calculate average win and loss
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        if len(wins) == 0 or len(losses) == 0:
            return min(0.01 * current_capital, current_capital * 0.001)

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

        # Convert to dollar amount
        dollar_position = position_size * current_capital

        # Apply maximum position size constraint (1% of capital)
        max_position = 0.01 * current_capital

        return min(dollar_position, max_position)

    def get_confidence(self) -> float:
        """Get confidence level based on recent performance"""
        if len(self.win_loss_history) < 20:
            return 0.5

        recent_wins = sum(self.win_loss_history[-20:])
        win_rate = recent_wins / 20

        # Confidence increases with consistent winning
        if win_rate > 0.7:
            return 0.9
        elif win_rate > 0.6:
            return 0.75
        elif win_rate > 0.5:
            return 0.6
        else:
            return 0.4


class RiskParity:
    """
    Risk Parity allocation across multiple positions
    """

    def __init__(self, target_risk: float = 0.01):
        self.target_risk = target_risk  # Target risk per position

    def calculate_weights(self, volatilities: np.ndarray,
                          correlations: np.ndarray = None,
                          current_weights: np.ndarray = None) -> np.ndarray:
        """
        Calculate risk parity weights

        Weight is inversely proportional to volatility
        """
        n = len(volatilities)

        if n == 0:
            return np.array([])

        if correlations is None:
            # Assume zero correlation
            weights = 1 / volatilities
        else:
            # Use correlation matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlations

            # Solve for equal risk contribution
            try:
                inv_vols = np.linalg.solve(cov_matrix, np.ones(n))
                weights = inv_vols / np.sum(inv_vols)
            except:
                # Fallback to simple inverse volatility
                weights = 1 / volatilities

        # Normalize weights
        weights = weights / np.sum(weights)

        # Scale to target risk
        if correlations is not None:
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        else:
            portfolio_vol = np.sqrt(np.sum((weights * volatilities) ** 2))

        scaling = self.target_risk / portfolio_vol if portfolio_vol > 0 else 1

        return weights * scaling

    def rebalance_positions(self, current_positions: Dict,
                           target_weights: np.ndarray,
                           total_capital: float = 10_000_000) -> Dict:
        """
        Calculate rebalancing trades needed
        """
        rebalance_trades = {}
        position_keys = list(current_positions.keys())

        for i, key in enumerate(position_keys):
            if i < len(target_weights):
                target_value = target_weights[i] * total_capital
                current_value = current_positions[key]
                trade_value = target_value - current_value

                if abs(trade_value) > 0.001 * total_capital:  # Only rebalance if > 0.1% of capital
                    rebalance_trades[key] = trade_value

        return rebalance_trades


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
    def historical_var(returns: np.ndarray, confidence: float = 0.95,
                      portfolio_value: float = 10_000_000) -> float:
        """
        Calculate historical VaR
        """
        if len(returns) == 0:
            return 0

        percentile = (1 - confidence) * 100
        var_return = np.percentile(returns, percentile)
        var = abs(var_return) * portfolio_value
        return var

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

    @staticmethod
    def conditional_var(returns: np.ndarray, confidence: float = 0.95,
                       portfolio_value: float = 10_000_000) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        """
        if len(returns) == 0:
            return 0

        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        conditional_returns = returns[returns <= var_threshold]

        if len(conditional_returns) > 0:
            cvar_return = np.mean(conditional_returns)
            cvar = abs(cvar_return) * portfolio_value
            return cvar
        else:
            return VaRCalculator.historical_var(returns, confidence, portfolio_value)


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
                'correlation_change': -0.2,
                'description': 'Sudden volatility spike (COVID-like)'
            },
            'market_crash': {
                'spot_change': -0.10,  # 10% drop
                'vol_change': 0.30,
                'correlation_change': -0.3,
                'description': 'Market crash scenario (2008-like)'
            },
            'flash_crash': {
                'spot_change': -0.05,  # 5% drop
                'vol_change': 1.00,  # Vol doubles
                'correlation_change': -0.5,
                'description': 'Flash crash with vol explosion'
            },
            'volatility_collapse': {
                'spot_change': 0.02,
                'vol_change': -0.30,  # 30% decrease in vol
                'correlation_change': 0.1,
                'description': 'Volatility collapse (calm market)'
            },
            'correlation_breakdown': {
                'spot_change': 0.05,
                'vol_change': -0.10,
                'correlation_change': 0.5,
                'description': 'Correlation regime change'
            },
            'currency_crisis': {
                'spot_change': -0.15,  # 15% devaluation
                'vol_change': 0.80,
                'correlation_change': -0.4,
                'description': 'Currency crisis (emerging market)'
            },
            'interest_rate_shock': {
                'spot_change': 0.03,
                'vol_change': 0.20,
                'correlation_change': 0.2,
                'description': 'Central bank surprise'
            }
        }

    def run_stress_test(self, positions: List, spot: float,
                        bs_model, current_capital: float = 10_000_000) -> pd.DataFrame:
        """
        Run stress tests on portfolio
        """
        results = []

        for scenario_name, scenario in self.scenarios.items():
            stressed_spot = spot * (1 + scenario['spot_change'])

            scenario_pnl = 0
            scenario_delta = 0
            scenario_vega = 0

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
                        current_price = bs_model.call_price(
                            spot, position.strike, 0.02, 0.025,
                            position.entry_vol, T
                        )
                    elif position.option_type == 'put':
                        stressed_price = bs_model.put_price(
                            stressed_spot, position.strike, 0.02, 0.025,
                            stressed_vol, T
                        )
                        current_price = bs_model.put_price(
                            spot, position.strike, 0.02, 0.025,
                            position.entry_vol, T
                        )
                    else:  # straddle
                        stressed_price = (bs_model.call_price(
                            stressed_spot, position.strike, 0.02, 0.025,
                            stressed_vol, T
                        ) + bs_model.put_price(
                            stressed_spot, position.strike, 0.02, 0.025,
                            stressed_vol, T
                        ))
                        current_price = (bs_model.call_price(
                            spot, position.strike, 0.02, 0.025,
                            position.entry_vol, T
                        ) + bs_model.put_price(
                            spot, position.strike, 0.02, 0.025,
                            position.entry_vol, T
                        ))

                    # Calculate P&L
                    price_change = stressed_price - current_price
                    position_pnl = price_change * position.position_size * spot * 100
                    scenario_pnl += position_pnl

                    # Calculate stressed Greeks
                    scenario_delta += bs_model.delta(
                        stressed_spot, position.strike, 0.02, 0.025,
                        stressed_vol, T, position.option_type if position.option_type != 'straddle' else 'call'
                    ) * position.position_size

                    scenario_vega += bs_model.vega(
                        stressed_spot, position.strike, 0.02, 0.025,
                        stressed_vol, T
                    ) * position.position_size

            # Calculate impact as percentage of capital
            impact_pct = (scenario_pnl / current_capital) * 100

            results.append({
                'scenario': scenario_name,
                'description': scenario['description'],
                'spot_change': scenario['spot_change'],
                'vol_change': scenario['vol_change'],
                'pnl': scenario_pnl,
                'impact_pct': impact_pct,
                'stressed_delta': scenario_delta,
                'stressed_vega': scenario_vega,
                'severity': 'SEVERE' if impact_pct < -5 else 'MODERATE' if impact_pct < -2 else 'MILD'
            })

        return pd.DataFrame(results).sort_values('pnl')


class PositionLimits:
    """
    Manage position limits and concentration risk
    """

    def __init__(self,
                 max_positions: int = 200,  # Allow many positions for $10M capital
                 max_concentration: float = 0.10,  # Max 10% in single name
                 max_tenor_concentration: float = 0.30,  # Max 30% in one tenor
                 max_directional_bias: float = 0.60,  # Max 60% long or short
                 max_vega_per_tenor: float = 50000):  # Max vega per tenor

        self.max_positions = max_positions
        self.max_concentration = max_concentration
        self.max_tenor_concentration = max_tenor_concentration
        self.max_directional_bias = max_directional_bias
        self.max_vega_per_tenor = max_vega_per_tenor

    def check_limits(self, positions: List, new_position,
                    current_capital: float = 10_000_000) -> Tuple[bool, str]:
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
            notional = abs(pos.position_size * pos.entry_price * pos.entry_spot * 100)
            pair_exposure[pos.pair] = pair_exposure.get(pos.pair, 0) + notional

        new_notional = abs(new_position.position_size * new_position.entry_price * new_position.entry_spot * 100)
        new_pair_exposure = pair_exposure.get(new_position.pair, 0) + new_notional

        if new_pair_exposure > self.max_concentration * current_capital:
            return False, f"Concentration limit exceeded for {new_position.pair}"

        # Check tenor concentration
        tenor_exposure = {}
        tenor_vega = {}

        for pos in active_positions:
            tenor = (pos.expiry - pd.Timestamp.now()).days
            bucket = self._get_tenor_bucket(tenor)
            notional = abs(pos.position_size * pos.entry_price * pos.entry_spot * 100)
            tenor_exposure[bucket] = tenor_exposure.get(bucket, 0) + notional
            tenor_vega[bucket] = tenor_vega.get(bucket, 0) + abs(pos.entry_vega * pos.position_size)

        new_tenor = (new_position.expiry - pd.Timestamp.now()).days
        new_bucket = self._get_tenor_bucket(new_tenor)
        new_tenor_exposure = tenor_exposure.get(new_bucket, 0) + new_notional
        new_tenor_vega = tenor_vega.get(new_bucket, 0) + abs(new_position.entry_vega * new_position.position_size)

        if new_tenor_exposure > self.max_tenor_concentration * current_capital:
            return False, f"Tenor concentration limit exceeded for {new_bucket}"

        if new_tenor_vega > self.max_vega_per_tenor:
            return False, f"Vega limit exceeded for {new_bucket}"

        # Check directional bias
        long_exposure = sum(pos.position_size * pos.entry_price * pos.entry_spot * 100
                          for pos in active_positions if pos.position_size > 0)
        short_exposure = abs(sum(pos.position_size * pos.entry_price * pos.entry_spot * 100
                               for pos in active_positions if pos.position_size < 0))

        if new_position.position_size > 0:
            long_exposure += new_notional
        else:
            short_exposure += new_notional

        total = long_exposure + short_exposure
        if total > 0:
            long_ratio = long_exposure / total
            short_ratio = short_exposure / total

            if max(long_ratio, short_ratio) > self.max_directional_bias:
                return False, "Directional bias limit exceeded"

        return True, "OK"

    def _get_tenor_bucket(self, days: int) -> str:
        """Categorize tenor into buckets"""
        if days <= 7:
            return "1W"
        elif days <= 30:
            return "1M"
        elif days <= 90:
            return "3M"
        elif days <= 180:
            return "6M"
        elif days <= 365:
            return "1Y"
        else:
            return ">1Y"

    def get_position_summary(self, positions: List,
                           current_capital: float = 10_000_000) -> Dict:
        """Get summary of current position limits usage"""
        active_positions = [p for p in positions if not p.is_closed]

        # Calculate current usage
        pair_exposure = {}
        tenor_exposure = {}
        long_exposure = 0
        short_exposure = 0
        total_vega = 0
        total_delta = 0

        for pos in active_positions:
            notional = abs(pos.position_size * pos.entry_price * pos.entry_spot * 100)

            # Pair exposure
            pair_exposure[pos.pair] = pair_exposure.get(pos.pair, 0) + notional

            # Tenor exposure
            tenor = (pos.expiry - pd.Timestamp.now()).days
            bucket = self._get_tenor_bucket(tenor)
            tenor_exposure[bucket] = tenor_exposure.get(bucket, 0) + notional

            # Directional exposure
            if pos.position_size > 0:
                long_exposure += notional
            else:
                short_exposure += notional

            # Greeks
            total_vega += abs(pos.entry_vega * pos.position_size)
            total_delta += pos.entry_delta * pos.position_size

        # Calculate usage percentages
        max_pair_pct = max(pair_exposure.values()) / current_capital * 100 if pair_exposure else 0
        max_tenor_pct = max(tenor_exposure.values()) / current_capital * 100 if tenor_exposure else 0

        total_exposure = long_exposure + short_exposure
        directional_bias = max(long_exposure, short_exposure) / total_exposure * 100 if total_exposure > 0 else 0

        return {
            'active_positions': len(active_positions),
            'max_positions': self.max_positions,
            'position_usage_pct': len(active_positions) / self.max_positions * 100,
            'max_pair_concentration_pct': max_pair_pct,
            'max_tenor_concentration_pct': max_tenor_pct,
            'directional_bias_pct': directional_bias,
            'total_vega': total_vega,
            'total_delta': total_delta,
            'total_exposure': total_exposure,
            'exposure_pct_of_capital': total_exposure / current_capital * 100
        }


class RiskMonitor:
    """
    Real-time risk monitoring and alerts
    """

    def __init__(self,
                 max_delta: float = 100000,  # Max delta exposure
                 max_gamma: float = 10000,  # Max gamma exposure
                 max_vega: float = 50000,  # Max vega exposure
                 max_theta: float = 5000,  # Max daily theta
                 alert_callback = None):  # Function to call on alerts

        self.max_delta = max_delta
        self.max_gamma = max_gamma
        self.max_vega = max_vega
        self.max_theta = max_theta
        self.alerts = []
        self.alert_callback = alert_callback

    def check_greeks(self, portfolio_greeks: Dict) -> List[str]:
        """Check if Greeks exceed limits"""
        alerts = []

        if abs(portfolio_greeks.get('delta_exposure', 0)) > self.max_delta:
            alert = f"ALERT: Delta exposure ${portfolio_greeks['delta_exposure']:,.0f} exceeds limit ${self.max_delta:,.0f}"
            alerts.append(alert)

        if abs(portfolio_greeks.get('gamma_exposure', 0)) > self.max_gamma:
            alert = f"ALERT: Gamma exposure ${portfolio_greeks['gamma_exposure']:,.0f} exceeds limit ${self.max_gamma:,.0f}"
            alerts.append(alert)

        if abs(portfolio_greeks.get('vega_exposure', 0)) > self.max_vega:
            alert = f"ALERT: Vega exposure ${portfolio_greeks['vega_exposure']:,.0f} exceeds limit ${self.max_vega:,.0f}"
            alerts.append(alert)

        if abs(portfolio_greeks.get('theta_exposure', 0)) > self.max_theta:
            alert = f"ALERT: Theta exposure ${portfolio_greeks['theta_exposure']:,.0f} exceeds limit ${self.max_theta:,.0f}"
            alerts.append(alert)

        self.alerts.extend(alerts)

        # Call alert callback if provided
        if self.alert_callback and alerts:
            for alert in alerts:
                self.alert_callback(alert)

        return alerts

    def calculate_margin_requirement(self, positions: List, spot: float,
                                    base_margin: float = 0.10) -> float:
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
            position_margin = notional * base_margin

            # Adjust for moneyness
            moneyness = position.strike / spot
            if 0.95 < moneyness < 1.05:
                # ATM options need more margin
                position_margin *= 1.5
            elif moneyness < 0.90 or moneyness > 1.10:
                # Deep OTM/ITM need less margin
                position_margin *= 0.75

            # Adjust for time to expiry
            days_to_expiry = (position.expiry - pd.Timestamp.now()).days
            if days_to_expiry < 7:
                # Near expiry needs more margin
                position_margin *= 1.5
            elif days_to_expiry > 180:
                # Long-dated needs less margin
                position_margin *= 0.8

            # Adjust for volatility
            if position.entry_vol > 0.30:
                # High vol needs more margin
                position_margin *= 1.3
            elif position.entry_vol < 0.10:
                # Low vol needs less margin
                position_margin *= 0.8

            margin += position_margin

        return margin

    def check_margin_usage(self, margin_used: float, margin_available: float) -> List[str]:
        """Check margin utilization and generate alerts"""
        alerts = []
        utilization = margin_used / margin_available if margin_available > 0 else 0

        if utilization > 0.90:
            alerts.append(f"CRITICAL: Margin utilization at {utilization:.1%} - immediate action required")
        elif utilization > 0.80:
            alerts.append(f"WARNING: Margin utilization at {utilization:.1%} - approaching limit")
        elif utilization > 0.70:
            alerts.append(f"CAUTION: Margin utilization at {utilization:.1%}")

        return alerts

    def generate_risk_report(self, positions: List, portfolio_greeks: Dict,
                           current_capital: float = 10_000_000) -> Dict:
        """Generate comprehensive risk report"""
        active_positions = [p for p in positions if not p.is_closed]

        # Calculate various risk metrics
        total_notional = sum(abs(p.position_size * p.entry_price * p.entry_spot * 100)
                           for p in active_positions)

        # Group by pair
        pair_exposure = {}
        for pos in active_positions:
            notional = abs(pos.position_size * pos.entry_price * pos.entry_spot * 100)
            pair_exposure[pos.pair] = pair_exposure.get(pos.pair, 0) + notional

        # Find largest exposures
        top_exposures = sorted(pair_exposure.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'timestamp': pd.Timestamp.now(),
            'active_positions': len(active_positions),
            'total_notional': total_notional,
            'notional_pct_of_capital': total_notional / current_capital * 100,
            'greeks': portfolio_greeks,
            'top_exposures': top_exposures,
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'risk_score': self._calculate_risk_score(portfolio_greeks, total_notional, current_capital)
        }

    def _calculate_risk_score(self, greeks: Dict, notional: float, capital: float) -> float:
        """Calculate overall risk score (0-100)"""
        score = 0

        # Greek risk (40% weight)
        delta_score = min(abs(greeks.get('delta_exposure', 0)) / self.max_delta * 40, 40)
        vega_score = min(abs(greeks.get('vega_exposure', 0)) / self.max_vega * 40, 40)
        greek_score = (delta_score + vega_score) / 2

        # Concentration risk (30% weight)
        concentration_score = min(notional / capital * 100, 30)

        # Alert-based risk (30% weight)
        alert_score = min(len(self.alerts) * 5, 30)

        score = greek_score + concentration_score + alert_score

        return min(score, 100)


# Example usage
if __name__ == "__main__":
    from pricing_models import BlackScholesFX
    from trading_strategy import OptionPosition

    # Test Kelly Criterion with $10M capital
    kelly = KellyCriterion(kelly_fraction=0.25)

    # Add some historical returns
    for _ in range(50):
        kelly.add_return(np.random.normal(0.001, 0.02))

    optimal_size = kelly.calculate_position_size(
        expected_return=0.02,
        win_probability=0.55,
        current_capital=10_000_000
    )
    print(f"Kelly optimal position size: ${optimal_size:,.0f}")
    print(f"As % of capital: {optimal_size/10_000_000:.2%}")

    # Test Risk Parity
    risk_parity = RiskParity(target_risk=0.01)
    volatilities = np.array([0.10, 0.15, 0.20, 0.12])
    weights = risk_parity.calculate_weights(volatilities)
    print(f"\nRisk Parity weights: {weights}")

    # Test rebalancing
    current_positions = {'EURUSD': 1_000_000, 'GBPUSD': 500_000, 'USDJPY': 750_000, 'AUDUSD': 250_000}
    rebalance = risk_parity.rebalance_positions(current_positions, weights, 10_000_000)
    print(f"Rebalancing trades needed:")
    for pair, amount in rebalance.items():
        print(f"  {pair}: ${amount:,.0f}")

    # Test VaR with $10M portfolio
    portfolio_value = 10_000_000
    volatility = 0.15
    var_95 = VaRCalculator.parametric_var(portfolio_value, volatility, confidence=0.95)
    print(f"\n95% VaR (1-day): ${var_95:,.0f}")
    print(f"As % of capital: {var_95/portfolio_value:.2%}")

    # Test CVaR
    returns = np.random.normal(-0.001, 0.02, 1000)
    cvar_95 = VaRCalculator.conditional_var(returns, 0.95, portfolio_value)
    print(f"95% CVaR: ${cvar_95:,.0f}")

    # Test Stress Testing
    stress_test = StressTest()

    # Create sample positions
    positions = []
    import pandas as pd

    for i in range(5):
        pos = OptionPosition(
            pair=f"PAIR{i}",
            strike=1.0 + i*0.01,
            expiry=pd.Timestamp.now() + pd.Timedelta(days=30+i*10),
            option_type='call' if i % 2 == 0 else 'put',
            position_size=100,
            entry_price=0.01,
            entry_vol=0.10 + i*0.01,
            entry_date=pd.Timestamp.now(),
            model_vol=0.09,
            market_vol=0.10,
            entry_spot=1.0,
            entry_forward=1.0,
            entry_vega=1000
        )
        positions.append(pos)

    bs_model = BlackScholesFX()
    stress_results = stress_test.run_stress_test(positions, 1.0, bs_model, 10_000_000)

    print(f"\nStress test scenarios:")
    for _, row in stress_results.iterrows():
        print(f"  {row['scenario']}: P&L ${row['pnl']:,.0f} ({row['impact_pct']:.2%}) - {row['severity']}")

    # Test Position Limits with $10M capital
    limits = PositionLimits(max_positions=200)
    is_allowed, reason = limits.check_limits(positions, positions[0], 10_000_000)
    print(f"\nPosition check: {reason}")

    summary = limits.get_position_summary(positions, 10_000_000)
    print(f"Position summary:")
    for key, value in summary.items():
        if 'pct' in key:
            print(f"  {key}: {value:.1f}%")
        else:
            print(f"  {key}: {value:,.0f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

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

    # Test margin calculation
    margin_req = monitor.calculate_margin_requirement(positions, 1.0)
    print(f"\nTotal margin requirement: ${margin_req:,.0f}")
    print(f"As % of $10M capital: {margin_req/10_000_000:.1%}")

    # Test risk report
    risk_report = monitor.generate_risk_report(positions, test_greeks, 10_000_000)
    print(f"\nRisk Report:")
    print(f"  Risk Score: {risk_report['risk_score']:.1f}/100")
    print(f"  Active Positions: {risk_report['active_positions']}")
    print(f"  Total Notional: ${risk_report['total_notional']:,.0f}")
    print(f"  Notional % of Capital: {risk_report['notional_pct_of_capital']:.1%}")