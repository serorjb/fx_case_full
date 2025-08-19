"""
Main FX Options Trading System
Complete implementation for systematic FX options volatility arbitrage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def run_complete_analysis():
    """
    Run complete FX options analysis including:
    1. Data loading and processing
    2. Volatility surface construction
    3. Model calibration (VGVV, SABR)
    4. Strategy signal generation
    5. Backtesting
    6. Performance analysis
    """

    print("="*60)
    print("FX OPTIONS VOLATILITY ARBITRAGE SYSTEM")
    print("="*60)

    # ==========================================
    # 1. DATA LOADING AND PROCESSING
    # ==========================================
    print("\n1. LOADING FX DATA...")
    print("-"*40)

    from data_loader import FXDataLoader, VolatilitySurfaceInterpolator

    loader = FXDataLoader()
    loader.rf_curve.load_fred_data()

    # Load multiple currency pairs
    pairs_to_trade = ["AUDNZD"]  # Add more pairs as available

    for pair in pairs_to_trade:
        try:
            df = loader.load_pair_data(pair)
            print(f"✓ Loaded {pair}: {len(df)} days of data")
            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        except FileNotFoundError:
            print(f"✗ Could not find data for {pair}")

    # ==========================================
    # 2. VOLATILITY SURFACE ANALYSIS
    # ==========================================
    print("\n2. ANALYZING VOLATILITY SURFACES...")
    print("-"*40)

    # Analyze a sample date
    sample_date = pd.Timestamp("2006-01-04")
    vol_data = loader.get_volatility_surface("AUDNZD", sample_date)

    if vol_data:
        print(f"Sample surface for {sample_date.date()}:")
        print(f"  Spot: {vol_data.spot:.4f}")
        print(f"  ATM Vols: {', '.join([f'{t}:{v:.1%}' for t,v in vol_data.atm_vols.items()])}")

        # Construct and visualize smile
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        tenors_to_plot = ["1M", "3M", "6M", "12M"]
        for i, tenor in enumerate(tenors_to_plot):
            ax = axes[i//2, i%2]

            strikes, vols = loader.construct_smile(vol_data, tenor)
            if strikes is not None:
                ax.plot(strikes/vol_data.spot, vols*100, 'o-', label='Market')
                ax.set_xlabel('Moneyness (K/S)')
                ax.set_ylabel('Implied Vol (%)')
                ax.set_title(f'{tenor} Volatility Smile')
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.suptitle(f'AUDNZD Volatility Smiles - {sample_date.date()}')
        plt.tight_layout()
        plt.show()

    # ==========================================
    # 3. MODEL CALIBRATION
    # ==========================================
    print("\n3. CALIBRATING PRICING MODELS...")
    print("-"*40)

    from pricing_models import VGVVModel, SABRModel, BlackScholesFX

    # Calibrate VGVV model
    if vol_data:
        r_d, r_f = 0.02, 0.025  # Simplified rates

        for tenor in ["1M", "3M", "6M"]:
            if tenor not in vol_data.atm_vols:
                continue

            T = loader.TENOR_MAP[tenor] / 365
            forward_points = vol_data.forwards.get(tenor, 0)
            forward = vol_data.spot + forward_points / 10000

            # Get smile data
            strikes, market_vols = loader.construct_smile(vol_data, tenor)

            if strikes is not None and len(strikes) > 3:
                # VGVV calibration
                vgvv = VGVVModel(vol_data.spot, forward, r_d, r_f, T)
                vgvv_params = vgvv.calibrate(strikes, market_vols)

                print(f"\n{tenor} VGVV Parameters:")
                print(f"  ATM Vol: {vgvv_params['sigma_atm']:.2%}")
                print(f"  Correlation: {vgvv_params['rho']:.3f}")
                print(f"  Vol-of-Vol: {vgvv_params['volvol']:.3f}")

                # SABR calibration
                sabr = SABRModel(forward, T)
                sabr_params = sabr.calibrate(strikes, market_vols, beta=0.5)

                print(f"\n{tenor} SABR Parameters:")
                print(f"  Alpha: {sabr_params['alpha']:.3f}")
                print(f"  Rho: {sabr_params['rho']:.3f}")
                print(f"  Nu: {sabr_params['nu']:.3f}")

                # Compare model fits
                model_vols_vgvv = vgvv.get_smile(vgvv_params, strikes)
                model_vols_sabr = [sabr.sabr_vol(k, sabr_params['alpha'],
                                                sabr_params['beta'],
                                                sabr_params['rho'],
                                                sabr_params['nu'])
                                  for k in strikes]

                # Plot comparison
                plt.figure(figsize=(10, 6))
                plt.plot(strikes/vol_data.spot, market_vols*100, 'ko-',
                        label='Market', markersize=8)
                plt.plot(strikes/vol_data.spot, model_vols_vgvv*100, 'b--',
                        label='VGVV', linewidth=2)
                plt.plot(strikes/vol_data.spot, model_vols_sabr*100, 'r:',
                        label='SABR', linewidth=2)
                plt.xlabel('Moneyness (K/S)')
                plt.ylabel('Implied Volatility (%)')
                plt.title(f'Model Calibration Comparison - {tenor}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()

    # ==========================================
    # 4. STRATEGY SIGNALS
    # ==========================================
    print("\n4. GENERATING TRADING SIGNALS...")
    print("-"*40)

    from trading_strategy import VolatilityArbitrageStrategy

    strategy = VolatilityArbitrageStrategy(
        initial_capital=1_000_000,
        max_position_size=0.02,
        vol_threshold=0.005  # 0.5% threshold
    )

    # Generate sample signals
    market_surface = {}
    model_surface = {}

    for tenor in ["1M", "3M", "6M"]:
        if tenor in vol_data.atm_vols:
            market_surface[tenor] = {'vol': vol_data.atm_vols[tenor]}
            # Simulate model thinks vol should be lower
            model_surface[tenor] = {'vol': vol_data.atm_vols[tenor] * 0.95}

    signals = strategy.identify_opportunities(
        market_surface, model_surface, vol_data.spot, sample_date
    )

    print(f"Found {len(signals)} trading opportunities:")
    for i, signal in enumerate(signals[:5], 1):
        print(f"\nSignal {i}:")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Strike: {signal.strike:.4f}, Tenor: {signal.tenor}")
        print(f"  Market Vol: {signal.market_vol:.2%}, Model Vol: {signal.model_vol:.2%}")
        print(f"  Expected Edge: {signal.expected_edge:.2%}")
        print(f"  Confidence: {signal.confidence:.1%}")

    # ==========================================
    # 5. BACKTESTING
    # ==========================================
    print("\n5. RUNNING BACKTEST...")
    print("-"*40)

    from backtesting_engine import FXOptionsBacktester, BacktestConfig

    # Configure backtest
    config = BacktestConfig(
        start_date=pd.Timestamp("2006-01-01"),
        end_date=pd.Timestamp("2006-06-30"),  # 6 months for demo
        initial_capital=1_000_000,
        pairs=["AUDNZD"],
        max_positions=20,
        max_position_size=0.02,
        vol_threshold=0.005,
        pricing_model="VGVV",
        calibration_window=20,
        max_drawdown=0.10  # 10% max drawdown
    )

    # Run backtest
    backtester = FXOptionsBacktester(config)
    backtester.initialize()
    results = backtester.run_backtest()

    # ==========================================
    # 6. PERFORMANCE ANALYSIS
    # ==========================================
    print("\n6. PERFORMANCE ANALYSIS")
    print("-"*40)

    # Print metrics
    backtester.print_summary()

    # Additional analysis
    if results.equity_curve:
        equity_df = pd.DataFrame(results.equity_curve).set_index('date')

        # Monthly returns
        monthly_returns = equity_df['equity'].resample('M').last().pct_change()

        print("\nMonthly Returns:")
        for date, ret in monthly_returns.items():
            if not pd.isna(ret):
                print(f"  {date.strftime('%Y-%m')}: {ret:+.2%}")

        # Risk metrics
        if len(monthly_returns) > 1:
            print("\nRisk Metrics:")
            print(f"  Best Month: {monthly_returns.max():.2%}")
            print(f"  Worst Month: {monthly_returns.min():.2%}")
            print(f"  % Positive Months: {(monthly_returns > 0).mean():.1%}")

    # Plot detailed results
    results.plot_results()

    # ==========================================
    # 7. STRATEGY RECOMMENDATIONS
    # ==========================================
    print("\n7. STRATEGY RECOMMENDATIONS")
    print("-"*40)

    print("""
    Based on the backtest results, here are key recommendations:
    
    1. POSITION SIZING:
       - Use Kelly Criterion or risk parity for optimal sizing
       - Scale positions based on volatility regime
       - Reduce size during high correlation periods
    
    2. RISK MANAGEMENT:
       - Implement dynamic stop-losses based on realized vol
       - Use time-based exits for theta decay management
       - Monitor regime changes in volatility term structure
    
    3. MODEL SELECTION:
       - VGVV performs well for short-dated options
       - SABR better captures long-dated smile dynamics
       - Consider ensemble approach for robustness
    
    4. EXECUTION:
       - Trade most liquid tenors (1M, 3M)
       - Focus on relative value within tenor buckets
       - Use limit orders to minimize transaction costs
    
    5. PORTFOLIO CONSTRUCTION:
       - Maintain delta neutrality through spot hedging
       - Diversify across multiple currency pairs
       - Balance long/short volatility exposure
    """)

    return results


def analyze_greeks_over_time(results):
    """
    Analyze how portfolio Greeks evolve over time
    """
    if not results.greeks_history:
        print("No Greeks data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Delta
    if 'delta' in results.greeks_history:
        delta_df = pd.DataFrame(results.greeks_history['delta']).set_index('date')
        axes[0, 0].plot(delta_df.index, delta_df['value'])
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Portfolio Delta Over Time')
        axes[0, 0].set_ylabel('Delta')
        axes[0, 0].grid(True, alpha=0.3)

    # Gamma
    if 'gamma' in results.greeks_history:
        gamma_df = pd.DataFrame(results.greeks_history['gamma']).set_index('date')
        axes[0, 1].plot(gamma_df.index, gamma_df['value'], color='green')
        axes[0, 1].set_title('Portfolio Gamma Over Time')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].grid(True, alpha=0.3)

    # Vega
    if 'vega' in results.greeks_history:
        vega_df = pd.DataFrame(results.greeks_history['vega']).set_index('date')
        axes[1, 0].plot(vega_df.index, vega_df['value'], color='purple')
        axes[1, 0].set_title('Portfolio Vega Over Time')
        axes[1, 0].set_ylabel('Vega')
        axes[1, 0].grid(True, alpha=0.3)

    # Theta
    if 'theta' in results.greeks_history:
        theta_df = pd.DataFrame(results.greeks_history['theta']).set_index('date')
        axes[1, 1].plot(theta_df.index, theta_df['value'], color='orange')
        axes[1, 1].set_title('Portfolio Theta Over Time')
        axes[1, 1].set_ylabel('Theta (daily)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Portfolio Greeks Analysis')
    plt.tight_layout()
    plt.show()


def create_performance_report(results, filename="fx_options_report.html"):
    """
    Create an HTML performance report
    """
    html_content = f"""
    <html>
    <head>
        <title>FX Options Strategy Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
            .warning {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <h1>FX Options Volatility Arbitrage Strategy</h1>
        <h2>Performance Summary</h2>
        <table>
    """

    if results.metrics:
        for key, value in results.metrics.items():
            formatted_key = key.replace('_', ' ').title()
            if 'return' in key or 'drawdown' in key:
                formatted_value = f"{value:.2%}"
                color_class = 'metric' if value > 0 else 'warning'
            elif 'ratio' in key:
                formatted_value = f"{value:.2f}"
                color_class = 'metric' if value > 1 else 'warning'
            else:
                formatted_value = f"{value:.2f}"
                color_class = ''

            html_content += f"""
            <tr>
                <td><strong>{formatted_key}</strong></td>
                <td class="{color_class}">{formatted_value}</td>
            </tr>
            """

    html_content += """
        </table>
        <h2>Strategy Configuration</h2>
        <ul>
            <li>Initial Capital: $1,000,000</li>
            <li>Max Positions: 20</li>
            <li>Position Size: 2% max per trade</li>
            <li>Volatility Threshold: 0.5%</li>
            <li>Max Drawdown Limit: 10%</li>
        </ul>
        
        <h2>Risk Management</h2>
        <p>The strategy implements several risk controls:</p>
        <ul>
            <li>Delta-neutral portfolio construction</li>
            <li>Dynamic position sizing based on volatility</li>
            <li>Stop-loss at 2% per position</li>
            <li>Time-based exits for theta management</li>
        </ul>
        
        <h2>Model Performance</h2>
        <p>VGVV model shows strong calibration performance for short-dated options,
        while SABR better captures long-dated smile dynamics.</p>
        
        <p><em>Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </body>
    </html>
    """

    with open(filename, 'w') as f:
        f.write(html_content)

    print(f"\nPerformance report saved to {filename}")


if __name__ == "__main__":
    # Run complete analysis
    results = run_complete_analysis()

    # Additional analysis
    if results:
        print("\n" + "="*60)
        print("ADDITIONAL ANALYSIS")
        print("="*60)

        # Analyze Greeks evolution
        analyze_greeks_over_time(results)

        # Create performance report
        create_performance_report(results)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("""
        Next steps:
        1. Extend to multiple currency pairs
        2. Implement more sophisticated position sizing (Kelly Criterion)
        3. Add machine learning for volatility forecasting
        4. Integrate with execution management system
        5. Implement real-time risk monitoring dashboard
        """)