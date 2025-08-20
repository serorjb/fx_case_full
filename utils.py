"""
utils.py
Utility functions for FX options trading system
Complete set of helper functions for dates, volatility, performance metrics, and visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import json
import pickle
import warnings

warnings.filterwarnings('ignore')


class DateUtils:
    """Date and calendar utilities for options"""

    @staticmethod
    def get_expiry_date(tenor: str, start_date: pd.Timestamp) -> pd.Timestamp:
        """
        Convert tenor string to expiry date
        """
        tenor_map = {
            '1W': 7, '2W': 14, '3W': 21, '1M': 30, '2M': 60,
            '3M': 90, '4M': 120, '6M': 180, '9M': 270, '12M': 365
        }

        if tenor in tenor_map:
            days = tenor_map[tenor]
            return start_date + pd.Timedelta(days=days)
        else:
            # Try to parse as number of days
            try:
                days = int(tenor.replace('D', ''))
                return start_date + pd.Timedelta(days=days)
            except:
                raise ValueError(f"Unknown tenor: {tenor}")

    @staticmethod
    def business_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Count business days between two dates"""
        return len(pd.bdate_range(start, end)) - 1

    @staticmethod
    def is_expiry_date(date: pd.Timestamp) -> bool:
        """Check if date is a standard expiry (3rd Friday)"""
        if date.weekday() == 4:  # Friday
            # Check if it's the third Friday
            first_day = date.replace(day=1)
            first_friday = first_day + pd.Timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + pd.Timedelta(weeks=2)
            return date.date() == third_friday.date()
        return False

    @staticmethod
    def get_maturity_bucket(days_to_expiry: int) -> str:
        """Categorize days to expiry into standard tenor buckets"""
        if days_to_expiry <= 7:
            return '1W'
        elif days_to_expiry <= 30:
            return '1M'
        elif days_to_expiry <= 90:
            return '3M'
        elif days_to_expiry <= 180:
            return '6M'
        elif days_to_expiry <= 365:
            return '12M'
        else:
            return '>1Y'


class VolatilityUtils:
    """Volatility-related utilities"""

    @staticmethod
    def realized_volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate realized volatility from returns"""
        vol = np.std(returns)
        if annualize:
            vol *= np.sqrt(252)
        return vol

    @staticmethod
    def garch_volatility(returns: np.ndarray, p: int = 1, q: int = 1) -> np.ndarray:
        """
        Simple GARCH(p,q) volatility forecast
        """
        # Simplified GARCH - would use arch package in production
        n = len(returns)
        omega = 0.00001
        alpha = 0.1
        beta = 0.85

        variance = np.zeros(n)
        variance[0] = np.var(returns)

        for t in range(1, n):
            variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]

        return np.sqrt(variance)

    @staticmethod
    def parkinson_volatility(high: np.ndarray, low: np.ndarray) -> float:
        """
        Parkinson volatility estimator using high-low prices
        """
        n = len(high)
        sum_sq = np.sum((np.log(high / low)) ** 2)
        vol = np.sqrt(sum_sq / (n * 4 * np.log(2))) * np.sqrt(252)
        return vol

    @staticmethod
    def yang_zhang_volatility(open_: np.ndarray, high: np.ndarray,
                              low: np.ndarray, close: np.ndarray) -> float:
        """
        Yang-Zhang volatility estimator
        """
        n = len(close)

        # Overnight volatility
        log_ho = np.log(open_[1:] / close[:-1])
        overnight_var = np.var(log_ho)

        # Open-to-close volatility
        log_co = np.log(close / open_)
        open_close_var = np.var(log_co)

        # Rogers-Satchell volatility
        log_hl = np.log(high / low)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        rs_var = np.mean(log_hc * log_hl)

        # Yang-Zhang volatility
        k = 0.34 / (1 + (n + 1) / (n - 1))
        yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var

        return np.sqrt(yz_var * 252)

    @staticmethod
    def ewma_volatility(returns: np.ndarray, lambda_param: float = 0.94) -> np.ndarray:
        """
        Exponentially weighted moving average volatility
        """
        n = len(returns)
        variance = np.zeros(n)
        variance[0] = returns[0] ** 2

        for t in range(1, n):
            variance[t] = lambda_param * variance[t-1] + (1 - lambda_param) * returns[t] ** 2

        return np.sqrt(variance) * np.sqrt(252)


class PerformanceMetrics:
    """Calculate various performance metrics"""

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free / 252
        if len(excess_returns) > 0 and np.std(excess_returns) > 0:
            return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        return 0

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free: float = 0.02,
                      target: float = 0) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free / 252
        downside_returns = excess_returns[excess_returns < target]

        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                return np.sqrt(252) * np.mean(excess_returns) / downside_std
        return 0

    @staticmethod
    def calmar_ratio(returns: np.ndarray, max_dd: float) -> float:
        """Calculate Calmar ratio"""
        if abs(max_dd) > 0:
            annual_return = (1 + np.mean(returns)) ** 252 - 1
            return annual_return / abs(max_dd)
        return 0

    @staticmethod
    def maximum_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and peak/trough indices
        """
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min()

        # Find peak and trough
        end_idx = drawdown.argmin()
        start_idx = equity_curve[:end_idx].argmax() if end_idx > 0 else 0

        return max_dd, start_idx, end_idx

    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark: np.ndarray) -> float:
        """Calculate information ratio"""
        active_returns = returns - benchmark
        if len(active_returns) > 0 and np.std(active_returns) > 0:
            return np.sqrt(252) * np.mean(active_returns) / np.std(active_returns)
        return 0

    @staticmethod
    def hit_rate(returns: np.ndarray) -> float:
        """Calculate percentage of positive returns"""
        if len(returns) > 0:
            return np.mean(returns > 0)
        return 0

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())

        if losses > 0:
            return profits / losses
        elif profits > 0:
            return np.inf
        else:
            return 0

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()

        if losses > 0:
            return gains / losses
        elif gains > 0:
            return np.inf
        else:
            return 0


class DataValidator:
    """Validate and clean data"""

    @staticmethod
    def validate_volatility_surface(vol_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate volatility surface data
        Returns: (is_valid, list_of_errors)
        """
        errors = []

        # Check for required fields
        required_fields = ['spot', 'atm_vols']
        for field in required_fields:
            if field not in vol_data or vol_data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Check spot price
        if 'spot' in vol_data:
            if vol_data['spot'] <= 0:
                errors.append("Spot price must be positive")
            if vol_data['spot'] > 1000:  # Sanity check for FX
                errors.append(f"Spot price seems unrealistic: {vol_data['spot']}")

        # Check ATM vols
        if 'atm_vols' in vol_data and vol_data['atm_vols']:
            for tenor, vol in vol_data['atm_vols'].items():
                if vol <= 0:
                    errors.append(f"ATM vol for {tenor} must be positive")
                if vol > 2.0:  # 200% vol seems excessive
                    errors.append(f"ATM vol for {tenor} seems too high: {vol:.1%}")

        # Check butterfly arbitrage
        if all(k in vol_data for k in ['bf_25d', 'bf_10d']):
            for tenor in vol_data['bf_25d'].keys():
                bf_25 = vol_data['bf_25d'].get(tenor, 0)
                bf_10 = vol_data['bf_10d'].get(tenor, 0)

                # 10 delta butterfly should be larger than 25 delta
                if bf_10 < bf_25:
                    errors.append(f"Butterfly arbitrage at {tenor}: 10d < 25d")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def clean_price_data(prices: pd.Series) -> pd.Series:
        """Clean price data"""
        # Remove outliers (more than 10 std devs)
        returns = prices.pct_change()
        mean_ret = returns.mean()
        std_ret = returns.std()

        outliers = abs(returns - mean_ret) > 10 * std_ret

        # Forward fill outliers
        cleaned = prices.copy()
        cleaned[outliers] = np.nan
        cleaned = cleaned.fillna(method='ffill')

        return cleaned

    @staticmethod
    def validate_option_data(strike: float, spot: float, vol: float,
                           maturity: float) -> Tuple[bool, str]:
        """Validate single option data point"""
        if strike <= 0:
            return False, "Strike must be positive"
        if spot <= 0:
            return False, "Spot must be positive"
        if vol <= 0 or vol > 5:
            return False, f"Volatility {vol} out of reasonable range"
        if maturity <= 0:
            return False, "Maturity must be positive"
        if strike / spot > 10 or strike / spot < 0.1:
            return False, "Strike too far from spot"
        return True, "Valid"


class Visualization:
    """Visualization utilities"""

    @staticmethod
    def plot_volatility_surface(strikes: np.ndarray, maturities: np.ndarray,
                                vols: np.ndarray, title: str = "Volatility Surface"):
        """Plot 3D volatility surface"""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create mesh
        K, T = np.meshgrid(strikes, maturities)

        # Plot surface
        surf = ax.plot_surface(K, T, vols.T, cmap='viridis', alpha=0.8)

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity (years)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(title)

        fig.colorbar(surf)
        plt.show()

    @staticmethod
    def plot_pnl_attribution(pnl_components: Dict):
        """Plot P&L attribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart of P&L components
        components = list(pnl_components.keys())
        values = list(pnl_components.values())
        colors = ['green' if v > 0 else 'red' for v in values]

        ax1.barh(components, values, color=colors)
        ax1.set_xlabel('P&L ($)')
        ax1.set_title('P&L Attribution')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Pie chart of absolute contributions
        abs_values = [abs(v) for v in values]
        ax2.pie(abs_values, labels=components, autopct='%1.1f%%')
        ax2.set_title('Absolute P&L Contribution')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_greeks_heatmap(greeks_matrix: pd.DataFrame):
        """Plot Greeks as heatmap"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(greeks_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    center=0, cbar_kws={'label': 'Greek Value'})
        plt.title('Portfolio Greeks Heatmap')
        plt.xlabel('Maturity')
        plt.ylabel('Strike')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_performance_summary(equity_curve: pd.Series, returns: pd.Series):
        """Plot comprehensive performance summary"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Equity curve
        axes[0, 0].plot(equity_curve.index, equity_curve.values)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)

        # Drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax * 100
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0,
                               color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # Returns distribution
        axes[1, 0].hist(returns * 100, bins=50, edgecolor='black')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # Rolling Sharpe
        rolling_sharpe = returns.rolling(252).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
        )
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 1].set_title('Rolling 1-Year Sharpe Ratio')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Performance Summary', fontsize=16)
        plt.tight_layout()
        plt.show()


class DataPersistence:
    """Save and load data"""

    @staticmethod
    def save_results(results: Dict, filename: str):
        """Save results to file"""
        if filename.endswith('.json'):
            # Convert numpy arrays and timestamps to serializable format
            serializable = DataPersistence._make_serializable(results)
            with open(filename, 'w') as f:
                json.dump(serializable, f, indent=2)
        elif filename.endswith('.pkl'):
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        else:
            raise ValueError("Unsupported file format. Use .json or .pkl")

    @staticmethod
    def load_results(filename: str) -> Dict:
        """Load results from file"""
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                return json.load(f)
        elif filename.endswith('.pkl'):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file format")

    @staticmethod
    def _make_serializable(obj):
        """Convert numpy arrays and pandas objects to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: DataPersistence._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataPersistence._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    @staticmethod
    def save_portfolio_state(portfolio: Dict, filename: str):
        """Save portfolio state with positions and Greeks"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'positions': portfolio.get('positions', []),
            'greeks': portfolio.get('greeks', {}),
            'capital': portfolio.get('capital', 0),
            'margin_used': portfolio.get('margin_used', 0),
            'realized_pnl': portfolio.get('realized_pnl', 0)
        }
        DataPersistence.save_results(state, filename)


class Logger:
    """Simple logging utility"""

    def __init__(self, log_file: str = "trading_log.txt", level: str = "INFO"):
        self.log_file = log_file
        self.level = level
        self.levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

    def log(self, message: str, level: str = "INFO"):
        """Log a message"""
        if self.levels.get(level, 1) >= self.levels.get(self.level, 1):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] [{level}] {message}\n"

            # Print to console
            print(log_message.strip())

            # Write to file
            with open(self.log_file, 'a') as f:
                f.write(log_message)

    def debug(self, message: str):
        self.log(message, "DEBUG")

    def info(self, message: str):
        self.log(message, "INFO")

    def warning(self, message: str):
        self.log(message, "WARNING")

    def error(self, message: str):
        self.log(message, "ERROR")

    def log_trade(self, trade_info: Dict):
        """Log trade execution"""
        trade_str = (f"TRADE: {trade_info.get('action', 'UNKNOWN')} "
                    f"{trade_info.get('quantity', 0)} {trade_info.get('instrument', 'UNKNOWN')} "
                    f"@ {trade_info.get('price', 0):.4f}")
        self.info(trade_str)

    def log_position(self, position_info: Dict):
        """Log position update"""
        pos_str = (f"POSITION: {position_info.get('pair', 'UNKNOWN')} "
                  f"Delta: {position_info.get('delta', 0):.4f} "
                  f"Vega: {position_info.get('vega', 0):.4f} "
                  f"P&L: ${position_info.get('pnl', 0):,.2f}")
        self.info(pos_str)


# Example usage
if __name__ == "__main__":
    # Test date utilities
    date_utils = DateUtils()
    start = pd.Timestamp("2024-01-01")
    expiry = date_utils.get_expiry_date("3M", start)
    print(f"3M expiry from {start.date()}: {expiry.date()}")

    bucket = date_utils.get_maturity_bucket(45)
    print(f"45 days maps to bucket: {bucket}")

    # Test volatility utilities
    returns = np.random.normal(0, 0.01, 252)
    vol = VolatilityUtils.realized_volatility(returns)
    print(f"\nRealized volatility: {vol:.2%}")

    ewma_vol = VolatilityUtils.ewma_volatility(returns)
    print(f"EWMA volatility (last): {ewma_vol[-1]:.2%}")

    # Test performance metrics
    metrics = PerformanceMetrics()
    sharpe = metrics.sharpe_ratio(returns)
    print(f"\nSharpe ratio: {sharpe:.2f}")

    omega = metrics.omega_ratio(returns)
    print(f"Omega ratio: {omega:.2f}")

    # Test data validation
    validator = DataValidator()
    is_valid, error_msg = validator.validate_option_data(
        strike=100, spot=100, vol=0.2, maturity=0.25
    )
    print(f"\nOption data validation: {is_valid} - {error_msg}")

    # Test logger
    logger = Logger()
    logger.info("System initialized")
    logger.warning("Low volatility detected")

    logger.log_trade({
        'action': 'BUY',
        'quantity': 100,
        'instrument': 'EURUSD 1M ATM Call',
        'price': 0.0125
    })

    print("\nUtilities module loaded successfully")