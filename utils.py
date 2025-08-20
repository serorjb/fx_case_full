"""
utils.py
Utility functions for FX options trading system
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
        elif isinstance(obj, dict):
            return {k: DataPersistence._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataPersistence._make_serializable(item) for item in obj]
        else:
            return obj


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


# Example usage
if __name__ == "__main__":
    # Test date utilities
    date_utils = DateUtils()
    start = pd.Timestamp("2024-01-01")
    expiry = date_utils.get_expiry_date("3M", start)
    print(f"3M expiry from {start.date()}: {expiry.date()}")

    # Test volatility utilities
    returns = np.random.normal(0, 0.01, 252)
    vol = VolatilityUtils.realized_volatility(returns)
    print(f"\nRealized volatility: {vol:.2%}")

    # Test performance metrics
    metrics = PerformanceMetrics()
    sharpe = metrics.sharpe_ratio(returns)
    print(f"Sharpe ratio: {sharpe:.2f}")

    # Test logger
    logger = Logger()
    logger.info("System initialized")
    logger.warning("Low volatility detected")

    print("\nUtilities module loaded successfully")