"""
Main FX Options Trading System - FIXED VERSION
Uses the enhanced trading strategy directly without the problematic backtesting engine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')


def run_complete_analysis():
    """
    Run complete FX options analysis using the enhanced trading strategy directly
    """

    print("="*60)
    print("ENHANCED FX OPTIONS VOLATILITY ARBITRAGE SYSTEM")
    print("="*60)

    # ==========================================
    # 1. DATA LOADING AND PROCESSING
    # ==========================================
    print("\n1. LOADING FX DATA...")
    print("-"*40)

    from data_loader import FXDataLoader
    from trading_strategy import EnhancedVolatilityArbitrageStrategy, ENHANCED_CONFIG

    loader = FXDataLoader()

    # Load risk-free rates with error handling
    try:
        loader.rf_curve.load_fred_data()
        print("✓ Loaded USD risk-free rate curve")
    except Exception as e:
        print(f"⚠ Using default risk-free rates: {e}")

    # Load multiple currency pairs with robust error handling
    pairs_to_trade = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "AUDNZD"]
    loaded_pairs = []

    for pair in pairs_to_trade:
        try:
            df = loader.load_pair_data(pair)
            if df is not None and len(df) > 1000:  # Ensure meaningful data
                print(f"✓ Loaded {pair}: {len(df)} days of data")
                print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
                loaded_pairs.append(pair)
        except Exception as e:
            print(f"✗ Could not load {pair}: {e}")

    if not loaded_pairs:
        print("\n⚠ No usable data files found!")
        return None

    print(f"\n✓ Successfully loaded {len(loaded_pairs)} currency pairs")
    print(f"  Pairs: {', '.join(loaded_pairs)}")

    # ==========================================
    # 2. ENHANCED STRATEGY INITIALIZATION
    # ==========================================
    print("\n2. INITIALIZING ENHANCED STRATEGY...")
    print("-"*40)

    # Use enhanced configuration with sophisticated features
    strategy = EnhancedVolatilityArbitrageStrategy(ENHANCED_CONFIG)

    print(f"✓ Enhanced Strategy initialized with:")
    print(f"  Initial Capital: ${ENHANCED_CONFIG['initial_capital']:,}")
    print(f"  Max Position Size: {ENHANCED_CONFIG['max_position_size']:.1%}")
    print(f"  Vol Threshold: {ENHANCED_CONFIG['vol_threshold']:.1%}")
    print(f"  Max Positions: {ENHANCED_CONFIG['max_positions']}")
    print(f"  VGVV Alpha (Skew): {ENHANCED_CONFIG['vgvv_alpha']:.2f}")
    print(f"  VGVV Beta (Smile): {ENHANCED_CONFIG['vgvv_beta']:.2f}")
    print(f"  VGVV Gamma (Carry): {ENHANCED_CONFIG['vgvv_gamma']:.2f}")

    # ==========================================
    # 3. ENHANCED BACKTEST EXECUTION
    # ==========================================
    print("\n3. RUNNING ENHANCED BACKTEST...")
    print("-"*40)

    # Use full period for comprehensive testing
    start_date = pd.Timestamp("2010-01-01")
    end_date = pd.Timestamp("2024-12-31")

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Enhanced strategy with ultra-sophisticated VGVV model!")

    try:
        results = strategy.run_enhanced_backtest(
            data_loader=loader,
            pairs=loaded_pairs,
            start_date=start_date,
            end_date=end_date
        )

        if results and 'error' not in results:
            print("\n✅ ENHANCED BACKTEST COMPLETED SUCCESSFULLY!")
            display_enhanced_results(results)
            create_enhanced_plots(results)
        else:
            print(f"\n❌ Backtest failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ Backtest execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

    return results


def display_enhanced_results(results):
    """Display comprehensive results from enhanced strategy"""
    print("\n" + "="*70)
    print("ENHANCED VOLATILITY ARBITRAGE STRATEGY RESULTS")
    print("="*70)

    # Core Performance
    print(f"\n📊 CORE PERFORMANCE:")
    print(f"  Total Return:        {results['total_return']:+8.1%}")
    print(f"  Annualized Return:   {results['annualized_return']:+8.1%}")
    print(f"  Annualized Vol:      {results['annualized_volatility']:8.1%}")
    print(f"  Sharpe Ratio:        {results['sharpe_ratio']:8.2f}")
    print(f"  Max Drawdown:        {results['max_drawdown']:8.1%}")

    # Capital Evolution
    print(f"\n💰 CAPITAL EVOLUTION:")
    print(f"  Initial Capital:     ${results['initial_capital']:>12,.0f}")
    print(f"  Final Equity:        ${results['final_equity']:>12,.0f}")
    print(f"  Total P&L:           ${results['final_equity'] - results['initial_capital']:>+12,.0f}")
    print(f"  Realized P&L:        ${results['total_realized_pnl']:>+12,.0f}")

    # Trading Activity
    print(f"\n🎯 TRADING ACTIVITY:")
    print(f"  Total Completed:     {results['total_trades']:>8,} trades")
    print(f"  Win Rate:            {results['win_rate']:>8.1%}")
    print(f"  Average Win:         ${results['avg_win']:>8,.0f}")
    print(f"  Average Loss:        ${results['avg_loss']:>8,.0f}")

    if results['avg_loss'] != 0:
        win_loss_ratio = abs(results['avg_win'] / results['avg_loss'])
        print(f"  Win/Loss Ratio:      {win_loss_ratio:>8.2f}")

    # Enhanced Strategy Metrics
    if results['daily_metrics']:
        daily_df = pd.DataFrame(results['daily_metrics'])

        avg_daily_trades = daily_df['trades'].mean()
        max_daily_trades = daily_df['trades'].max()
        trading_days = len(daily_df[daily_df['trades'] > 0])

        print(f"\n📈 ENHANCED DAILY ACTIVITY:")
        print(f"  Avg Daily Trades:    {avg_daily_trades:>8.1f}")
        print(f"  Max Daily Trades:    {max_daily_trades:>8}")
        print(f"  Active Trading Days: {trading_days:>8,}")
        print(f"  Total Days:          {len(daily_df):>8,}")

    # Sophisticated Position Analysis
    if results['completed_positions']:
        print(f"\n🔍 ENHANCED POSITION ANALYSIS:")

        # Tenor breakdown with performance
        tenor_analysis = {}
        signal_type_analysis = {'buy': [], 'sell': []}

        for pos in results['completed_positions']:
            # Get tenor from position (need to reverse-engineer from expiry)
            if hasattr(pos, 'entry_date') and hasattr(pos, 'expiry'):
                days_to_expiry = (pos.expiry - pos.entry_date).days
                if days_to_expiry <= 10:
                    tenor = '1W'
                elif days_to_expiry <= 25:
                    tenor = '3W'
                elif days_to_expiry <= 45:
                    tenor = '1M'
                elif days_to_expiry <= 75:
                    tenor = '2M'
                elif days_to_expiry <= 105:
                    tenor = '3M'
                elif days_to_expiry <= 150:
                    tenor = '4M'
                elif days_to_expiry <= 210:
                    tenor = '6M'
                elif days_to_expiry <= 300:
                    tenor = '9M'
                else:
                    tenor = '12M'

                if tenor not in tenor_analysis:
                    tenor_analysis[tenor] = {'count': 0, 'pnl': 0, 'wins': 0}

                tenor_analysis[tenor]['count'] += 1
                tenor_analysis[tenor]['pnl'] += pos.total_pnl if hasattr(pos, 'total_pnl') else 0
                if hasattr(pos, 'total_pnl') and pos.total_pnl > 0:
                    tenor_analysis[tenor]['wins'] += 1

                # Signal type analysis
                if hasattr(pos, 'direction'):
                    signal_type = 'sell' if pos.direction == -1 else 'buy'
                    signal_type_analysis[signal_type].append(pos.total_pnl if hasattr(pos, 'total_pnl') else 0)

        print(f"  Tenor Performance Breakdown:")
        for tenor in sorted(tenor_analysis.keys()):
            data = tenor_analysis[tenor]
            win_rate = data['wins'] / data['count'] if data['count'] > 0 else 0
            print(f"    {tenor:>6}: {data['count']:>4} trades, ${data['pnl']:>+8,.0f}, {win_rate:.1%} win rate")

        # Signal type performance
        print(f"\n  Signal Type Performance:")
        for signal_type, pnls in signal_type_analysis.items():
            if pnls:
                avg_pnl = np.mean(pnls)
                win_rate = len([p for p in pnls if p > 0]) / len(pnls)
                print(f"    {signal_type.upper():>4}: {len(pnls):>4} trades, ${avg_pnl:>+8,.0f} avg, {win_rate:.1%} win rate")


def create_enhanced_plots(results):
    """Create comprehensive performance plots for enhanced strategy"""
    if not results['daily_metrics']:
        print("\n⚠ No daily metrics available for plotting")
        return

    try:
        daily_df = pd.DataFrame(results['daily_metrics'])
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df.set_index('date', inplace=True)

        # Create enhanced performance dashboard
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Enhanced FX Volatility Arbitrage Strategy Performance', fontsize=16, fontweight='bold')

        # 1. Equity Curve with Drawdown
        ax1 = axes[0,0]
        ax1.plot(daily_df.index, daily_df['equity'] / 1e6, 'b-', linewidth=2, label='Equity')
        ax1.set_title('Equity Curve', fontweight='bold')
        ax1.set_ylabel('Equity ($M)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=results['initial_capital']/1e6, color='gray', linestyle='--', alpha=0.7, label='Initial')
        ax1.legend()

        # 2. Daily Returns Distribution
        if 'daily_return' in daily_df.columns:
            daily_returns = daily_df['daily_return'].dropna()
            axes[0,1].hist(daily_returns * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[0,1].set_title('Daily Returns Distribution', fontweight='bold')
            axes[0,1].set_xlabel('Daily Return (%)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].axvline(x=daily_returns.mean() * 100, color='red', linestyle='--',
                            label=f'Mean: {daily_returns.mean()*100:.3f}%')
            axes[0,1].legend()

        # 3. Position Evolution
        axes[1,0].plot(daily_df.index, daily_df['active_positions'], 'orange', linewidth=2, label='Active')
        axes[1,0].plot(daily_df.index, daily_df['completed_positions'], 'green', linewidth=2, label='Completed')
        axes[1,0].set_title('Position Evolution', fontweight='bold')
        axes[1,0].set_ylabel('Number of Positions')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 4. Daily P&L
        axes[1,1].bar(daily_df.index, daily_df['daily_pnl'] / 1000, alpha=0.7,
                     color=['green' if x > 0 else 'red' for x in daily_df['daily_pnl']])
        axes[1,1].set_title('Daily P&L', fontweight='bold')
        axes[1,1].set_ylabel('Daily P&L ($K)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].axhline(y=0, color='black', linewidth=1)

        # 5. Greeks Evolution - Delta
        if 'portfolio_greeks' in daily_df.columns:
            # Extract delta and vega
            deltas = [g.get('delta', 0) if isinstance(g, dict) else 0 for g in daily_df['portfolio_greeks']]
            vegas = [g.get('vega', 0) if isinstance(g, dict) else 0 for g in daily_df['portfolio_greeks']]

            ax5 = axes[2,0]
            ax5.plot(daily_df.index, deltas, 'blue', linewidth=2, label='Delta')
            ax5.set_title('Portfolio Delta', fontweight='bold')
            ax5.set_ylabel('Delta ($)')
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='black', linewidth=1)

            # Vega on secondary axis
            ax5_twin = ax5.twinx()
            ax5_twin.plot(daily_df.index, vegas, 'red', linewidth=2, alpha=0.7, label='Vega')
            ax5_twin.set_ylabel('Vega ($)', color='red')
            ax5_twin.tick_params(axis='y', labelcolor='red')

        # 6. Rolling Performance Metrics
        if len(daily_df) > 252:  # At least 1 year of data
            rolling_252 = daily_df['daily_return'].rolling(252)
            rolling_returns = rolling_252.mean() * 252
            rolling_vol = rolling_252.std() * np.sqrt(252)
            rolling_sharpe = rolling_returns / rolling_vol

            axes[2,1].plot(daily_df.index, rolling_sharpe, 'purple', linewidth=2)
            axes[2,1].set_title('Rolling 1-Year Sharpe Ratio', fontweight='bold')
            axes[2,1].set_ylabel('Sharpe Ratio')
            axes[2,1].grid(True, alpha=0.3)
            axes[2,1].axhline(y=0, color='black', linewidth=1)
            axes[2,1].axhline(y=1, color='gray', linestyle='--', alpha=0.7)

        plt.tight_layout()

        # Save plot
        plot_filename = 'enhanced_fx_options_performance.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Enhanced performance plots saved to: {plot_filename}")

        plt.show()

    except Exception as e:
        print(f"\n⚠ Could not create plots: {e}")


def analyze_trading_activity(results):
    """Analyze sophisticated trading patterns"""
    print("\n🔬 SOPHISTICATED TRADING ANALYSIS")
    print("-"*50)

    if not results.get('daily_metrics'):
        return

    daily_df = pd.DataFrame(results['daily_metrics'])

    # Trading frequency analysis
    trading_days = daily_df[daily_df['trades'] > 0]
    print(f"Trading Intensity:")
    print(f"  High Activity Days (>5 trades): {len(trading_days[trading_days['trades'] > 5])}")
    print(f"  Medium Activity Days (2-5 trades): {len(trading_days[(trading_days['trades'] >= 2) & (trading_days['trades'] <= 5)])}")
    print(f"  Low Activity Days (1 trade): {len(trading_days[trading_days['trades'] == 1])}")


if __name__ == "__main__":
    print("Starting Enhanced FX Options Analysis...")
    results = run_complete_analysis()

    if results:
        print(f"\n✅ Enhanced analysis completed successfully!")
        print(f"Final Return: {results.get('total_return', 0):+.1%}")
        print(f"Completed Trades: {results.get('total_trades', 0):,}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")

        # Additional sophisticated analysis
        analyze_trading_activity(results)
    else:
        print(f"\n❌ Enhanced analysis failed!")
