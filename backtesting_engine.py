"""
FX Options Backtesting Engine
Event-driven backtesting system for multi-currency options strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_loader import FXDataLoader, FXVolatilityData, VolatilitySurfaceInterpolator
from pricing_models import BlackScholesFX, VGVVModel, SABRModel
from trading_strategy import VolatilityArbitrageStrategy, CarryToVolStrategy, TradingSignal


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float = 1_000_000
    pairs: List[str] = field(default_factory=lambda: ["AUDNZD"])
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

    # Strategy parameters
    max_positions: int = 50
    position_sizing: str = "equal"  # equal, kelly, risk_parity
    max_position_size: float = 0.02
    vol_threshold: float = 0.01

    # Risk management
    stop_loss: float = 0.02  # 2% stop loss per position
    take_profit: float = 0.05  # 5% take profit
    max_drawdown: float = 0.10  # 10% max drawdown

    # Transaction costs
    bid_ask_spread: float = 0.001  # 0.1% bid-ask spread
    commission: float = 0.0001  # 0.01% commission

    # Model selection
    pricing_model: str = "VGVV"  # BS, VGVV, SABR
    calibration_window: int = 60  # Days for model calibration


class BacktestResults:
    """Container for backtest results"""

    def __init__(self):
        self.equity_curve = []
        self.returns = []
        self.positions_history = []
        self.trades = []
        self.greeks_history = defaultdict(list)
        self.metrics = {}

    def add_snapshot(self, date: pd.Timestamp, equity: float,
                    positions: List, greeks: Dict):
        """Add a snapshot of the portfolio state"""
        self.equity_curve.append({'date': date, 'equity': equity})

        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            ret = (equity - prev_equity) / prev_equity
            self.returns.append({'date': date, 'return': ret})

        self.positions_history.append({
            'date': date,
            'num_positions': len(positions),
            'positions': positions.copy()
        })

        for greek, value in greeks.items():
            self.greeks_history[greek].append({'date': date, 'value': value})

    def calculate_metrics(self, risk_free_rate: float = 0.02):
        """Calculate performance metrics"""
        if not self.returns:
            return

        returns_df = pd.DataFrame(self.returns).set_index('date')
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')

        # Basic metrics
        total_return = (equity_df['equity'].iloc[-1] /
                       equity_df['equity'].iloc[0] - 1)

        # Annualized metrics
        days = (returns_df.index[-1] - returns_df.index[0]).days
        years = days / 365
        ann_return = (1 + total_return) ** (1/years) - 1
        ann_vol = returns_df['return'].std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # Maximum drawdown
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_dd = drawdown.min()

        # Win rate
        if self.trades:
            wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
            win_rate = wins / len(self.trades)
        else:
            win_rate = 0

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        self.metrics = {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'avg_trade_pnl': np.mean([t.get('pnl', 0) for t in self.trades]) if self.trades else 0
        }

        return self.metrics

    def plot_results(self):
        """Plot backtest results"""
        if not self.equity_curve:
            print("No results to plot")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Equity curve
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        axes[0, 0].plot(equity_df.index, equity_df['equity'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].grid(True)

        # Drawdown
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax * 100
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0,
                               color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)

        # Returns distribution
        if self.returns:
            returns_df = pd.DataFrame(self.returns).set_index('date')
            axes[1, 0].hist(returns_df['return'] * 100, bins=50, edgecolor='black')
            axes[1, 0].set_title('Returns Distribution')
            axes[1, 0].set_xlabel('Daily Return (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)

        # Number of positions over time
        positions_df = pd.DataFrame(self.positions_history).set_index('date')
        axes[1, 1].plot(positions_df.index, positions_df['num_positions'])
        axes[1, 1].set_title('Number of Positions')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True)

        # Greeks over time
        if 'delta' in self.greeks_history:
            delta_df = pd.DataFrame(self.greeks_history['delta']).set_index('date')
            axes[2, 0].plot(delta_df.index, delta_df['value'])
            axes[2, 0].set_title('Portfolio Delta')
            axes[2, 0].set_xlabel('Date')
            axes[2, 0].set_ylabel('Delta')
            axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[2, 0].grid(True)

        if 'vega' in self.greeks_history:
            vega_df = pd.DataFrame(self.greeks_history['vega']).set_index('date')
            axes[2, 1].plot(vega_df.index, vega_df['value'])
            axes[2, 1].set_title('Portfolio Vega')
            axes[2, 1].set_xlabel('Date')
            axes[2, 1].set_ylabel('Vega')
            axes[2, 1].grid(True)

        plt.tight_layout()
        plt.show()


class FXOptionsBacktester:
    """
    Main backtesting engine for FX options strategies
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_loader = FXDataLoader()
        self.bs_model = BlackScholesFX()
        self.strategy = None
        self.results = BacktestResults()
        self.current_date = config.start_date
        self.spot_hedge = 0.0  # Spot position for delta hedging

    def initialize(self):
        """Initialize the backtester"""
        # Load risk-free rates
        self.data_loader.rf_curve.load_fred_data()

        # Load FX data for all pairs
        for pair in self.config.pairs:
            try:
                self.data_loader.load_pair_data(pair)
                print(f"Loaded data for {pair}")
            except Exception as e:
                print(f"Error loading {pair}: {e}")

        # Initialize strategy
        self.strategy = VolatilityArbitrageStrategy(
            initial_capital=self.config.initial_capital,
            max_position_size=self.config.max_position_size,
            vol_threshold=self.config.vol_threshold,
            max_positions=self.config.max_positions
        )

        print(f"Backtester initialized with {len(self.config.pairs)} pairs")

    def calibrate_model(self, pair: str, date: pd.Timestamp) -> Optional[Dict]:
        """
        Calibrate pricing model to market data
        """
        # Get historical data for calibration
        vol_data = self.data_loader.get_volatility_surface(pair, date)
        if not vol_data:
            return None

        # Get risk-free rates
        r_d = self.data_loader.rf_curve.get_rate(date, 30)  # USD rate
        r_f = r_d + 0.005  # Foreign rate (simplified)

        # Build surface points for calibration
        calibration_points = []

        for tenor, days in FXDataLoader.TENOR_MAP.items():
            T = days / 365

            # Get forward
            forward_points = vol_data.forwards.get(tenor, 0)
            forward = vol_data.spot + forward_points / 10000

            # Get smile points
            strikes, vols = self.data_loader.construct_smile(vol_data, tenor)

            if strikes is not None and len(strikes) > 0:
                for k, v in zip(strikes, vols):
                    calibration_points.append({
                        'strike': k,
                        'maturity': T,
                        'forward': forward,
                        'vol': v
                    })

        if not calibration_points:
            return None

        # Calibrate model based on config
        if self.config.pricing_model == "VGVV":
            # Group by maturity and calibrate VGVV
            params = {}
            for T in set(p['maturity'] for p in calibration_points):
                T_points = [p for p in calibration_points if p['maturity'] == T]
                if T_points:
                    forward = T_points[0]['forward']
                    vgvv = VGVVModel(vol_data.spot, forward, r_d, r_f, T)

                    strikes = np.array([p['strike'] for p in T_points])
                    vols = np.array([p['vol'] for p in T_points])

                    params[T] = vgvv.calibrate(strikes, vols)

            return params

        elif self.config.pricing_model == "SABR":
            # Calibrate SABR for each maturity
            params = {}
            for T in set(p['maturity'] for p in calibration_points):
                T_points = [p for p in calibration_points if p['maturity'] == T]
                if T_points:
                    forward = T_points[0]['forward']
                    sabr = SABRModel(forward, T)

                    strikes = np.array([p['strike'] for p in T_points])
                    vols = np.array([p['vol'] for p in T_points])

                    params[T] = sabr.calibrate(strikes, vols)

            return params

        else:
            # Default to market vols
            return {'market_vols': calibration_points}

    def generate_model_surface(self, params: Dict, vol_data: FXVolatilityData) -> Dict:
        """
        Generate model-implied volatility surface
        """
        model_surface = {}

        for tenor, days in FXDataLoader.TENOR_MAP.items():
            T = days / 365

            if T in params:
                # Get model parameters for this maturity
                model_params = params[T]

                # Generate model volatilities
                forward_points = vol_data.forwards.get(tenor, 0)
                forward = vol_data.spot + forward_points / 10000

                # Sample strikes
                strikes = np.linspace(0.9 * forward, 1.1 * forward, 11)

                if self.config.pricing_model == "VGVV":
                    vgvv = VGVVModel(vol_data.spot, forward, 0.02, 0.025, T)
                    model_vols = vgvv.get_smile(model_params, strikes)
                elif self.config.pricing_model == "SABR":
                    sabr = SABRModel(forward, T)
                    model_vols = [sabr.sabr_vol(k, model_params['alpha'],
                                               model_params['beta'],
                                               model_params['rho'],
                                               model_params['nu'])
                                 for k in strikes]
                else:
                    model_vols = [model_params.get('sigma_atm', 0.10)] * len(strikes)

                model_surface[tenor] = {
                    'strikes': strikes,
                    'vols': model_vols,
                    'vol': np.mean(model_vols)  # Simplified average
                }

        return model_surface

    def run_backtest(self):
        """
        Run the backtest
        """
        print(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")

        # Get date range
        dates = pd.date_range(self.config.start_date, self.config.end_date, freq='B')

        for i, date in enumerate(dates):
            self.current_date = date

            # Process each pair
            for pair in self.config.pairs:
                # Get market data
                vol_data = self.data_loader.get_volatility_surface(pair, date)
                if not vol_data:
                    continue

                spot = vol_data.spot

                # Update existing positions
                market_vols = {}  # Simplified - would extract from vol_data
                self.strategy.update_positions(spot, date, market_vols, self.bs_model)

                # Calibrate model periodically
                if i % self.config.calibration_window == 0:
                    model_params = self.calibrate_model(pair, date)

                    if model_params:
                        # Generate model surface
                        model_surface = self.generate_model_surface(model_params, vol_data)

                        # Extract market surface (simplified)
                        market_surface = {}
                        for tenor in FXDataLoader.TENOR_MAP.keys():
                            if tenor in vol_data.atm_vols:
                                market_surface[tenor] = {
                                    'vol': vol_data.atm_vols[tenor]
                                }

                        # Identify trading opportunities
                        signals = self.strategy.identify_opportunities(
                            market_surface, model_surface, spot, date
                        )

                        # Execute top signals (respecting position limits)
                        for signal in signals[:3]:  # Execute top 3 signals
                            if len(self.strategy.positions) < self.config.max_positions:
                                position = self.strategy.execute_signal(
                                    signal, spot, date, self.bs_model
                                )

                                if position:
                                    # Record trade
                                    self.results.trades.append({
                                        'date': date,
                                        'pair': pair,
                                        'strike': position.strike,
                                        'type': position.option_type,
                                        'size': position.position_size,
                                        'entry_price': position.entry_price
                                    })

            # Calculate portfolio Greeks
            greeks = self.strategy.calculate_portfolio_greeks(spot, date, self.bs_model)

            # Delta hedging
            if self.strategy.target_delta_neutral:
                hedge_required = self.strategy.hedge_delta(greeks, spot)
                self.spot_hedge = hedge_required

            # Record snapshot
            self.results.add_snapshot(
                date,
                self.strategy.current_capital,
                self.strategy.positions,
                greeks
            )

            # Risk management - check for max drawdown breach
            if len(self.results.equity_curve) > 1:
                peak = max(e['equity'] for e in self.results.equity_curve)
                current = self.strategy.current_capital
                drawdown = (peak - current) / peak

                if drawdown > self.config.max_drawdown:
                    print(f"Max drawdown breached on {date}: {drawdown:.2%}")
                    # Close all positions
                    for i in range(len(self.strategy.positions)):
                        if not self.strategy.positions[i].is_closed:
                            self.strategy.close_position(
                                i, spot, date, "max_drawdown", self.bs_model
                            )

            # Progress update
            if i % 50 == 0:
                print(f"Processed {i}/{len(dates)} days, "
                      f"Capital: ${self.strategy.current_capital:,.0f}, "
                      f"Positions: {len([p for p in self.strategy.positions if not p.is_closed])}")

        # Calculate final metrics
        self.results.calculate_metrics()

        print("\n" + "="*50)
        print("BACKTEST COMPLETE")
        print("="*50)

        return self.results

    def print_summary(self):
        """Print backtest summary"""
        if not self.results.metrics:
            print("No results to display")
            return

        print("\nPerformance Summary:")
        print("-" * 30)
        for key, value in self.results.metrics.items():
            if 'return' in key or 'volatility' in key or 'drawdown' in key:
                print(f"{key.replace('_', ' ').title()}: {value:.2%}")
            elif 'ratio' in key or 'rate' in key:
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")


# Example usage
if __name__ == "__main__":
    # Configure backtest
    config = BacktestConfig(
        start_date=pd.Timestamp("2006-01-01"),
        end_date=pd.Timestamp("2006-12-31"),
        initial_capital=1_000_000,
        pairs=["AUDNZD"],
        max_positions=20,
        max_position_size=0.02,
        vol_threshold=0.01,
        pricing_model="VGVV",
        calibration_window=30
    )

    # Initialize and run backtest
    backtester = FXOptionsBacktester(config)
    backtester.initialize()

    # Run backtest
    results = backtester.run_backtest()

    # Print summary
    backtester.print_summary()

    # Plot results
    results.plot_results()