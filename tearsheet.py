"""
tearsheet.py
Simplified self-contained performance tearsheet generator.
Generates an equity curve, drawdown chart, monthly heatmap, yearly bar returns, and a stats panel.
Outputs a single PNG file.

Usage:
    from tearsheet import save_tearsheet
    save_tearsheet(equity_series, title="SABR Return Allocation", out_file="results/tearsheets/sabr_return.png")

Expected input:
    equity_series: pandas Series indexed by datetime, values = equity (PnL curve)

Metrics:
    - Daily returns, cumulative returns
    - Sharpe, Sortino, CAGR, annual vol
    - Max drawdown and duration
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec, cm
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from pathlib import Path

# -----------------------------
# Basic statistics helpers
# -----------------------------

def _daily_returns(equity: pd.Series) -> pd.Series:
    r = equity.pct_change().fillna(0.0)
    # Remove extreme outliers to keep charts readable (clip at +/-50%)
    return r.clip(-0.5, 0.5)

def _cum_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()

def create_drawdowns(cum_rets: pd.Series):
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    # duration in days since last peak
    dur = []
    current = 0
    for dd in drawdown:
        if dd == 0:
            current = 0
        else:
            current += 1
        dur.append(current)
    duration = pd.Series(dur, index=cum_rets.index)
    max_dd = drawdown.min() if not drawdown.empty else 0.0
    max_dur = duration.max() if not duration.empty else 0
    return drawdown, max_dd, max_dur

def annualized_vol(returns: pd.Series, periods: int = 252) -> float:
    return returns.std() * np.sqrt(periods)

def sharpe_ratio(returns: pd.Series, periods: int = 252, rf: float = 0.0) -> float:
    ex = returns - rf / periods
    vol = ex.std()
    if vol == 0:
        return 0.0
    return np.sqrt(periods) * ex.mean() / vol

def sortino_ratio(returns: pd.Series, periods: int = 252, rf: float = 0.0) -> float:
    ex = returns - rf / periods
    downside = ex[ex < 0]
    if downside.empty:
        return np.inf
    denom = np.sqrt((downside**2).mean())
    if denom == 0:
        return np.inf
    return np.sqrt(periods) * ex.mean() / denom

def cagr(cum_rets: pd.Series, periods: int = 252) -> float:
    if len(cum_rets) < 2:
        return 0.0
    total_return = cum_rets.iloc[-1] / cum_rets.iloc[0] - 1
    years = len(cum_rets) / periods
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1

def aggregate_period_returns(returns: pd.Series, freq: str) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    df = returns.to_frame('r')
    if freq == 'monthly':
        grouped = df['r'].groupby([returns.index.year, returns.index.month]).apply(lambda x: (1 + x).prod() - 1)
    elif freq == 'yearly':
        grouped = df['r'].groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
    else:
        raise ValueError("freq must be 'monthly' or 'yearly'")
    return grouped

# -----------------------------
# Plotting helpers
# -----------------------------

def _plot_equity(ax, cum_rets: pd.Series, label: str = 'Strategy'):
    ax.plot(cum_rets.index, cum_rets.values, lw=2, color='green', alpha=0.7, label=label)
    ax.axhline(1.0, ls='--', color='black', lw=1)
    ax.yaxis.grid(linestyle=':')
    ax.xaxis.grid(linestyle=':')
    ax.set_ylabel('Cumulative Returns')
    ax.legend(loc='best')
    ax.xaxis.set_major_locator(mdates.YearLocator( max(1, int(len(cum_rets)/252/10)) ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


def _plot_drawdown(ax, drawdown: pd.Series):
    uw = -100 * drawdown
    ax.fill_between(uw.index, uw.values, 0, color='red', alpha=0.3)
    ax.plot(uw.index, uw.values, color='red', lw=1)
    ax.set_title('Drawdown (%)', fontweight='bold')
    ax.set_ylabel('')
    ax.yaxis.grid(linestyle=':')
    ax.xaxis.grid(linestyle=':')
    ax.xaxis.set_major_locator(mdates.YearLocator( max(1, int(len(drawdown)/252/10)) ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


def _plot_monthly(ax, returns: pd.Series):
    monthly = aggregate_period_returns(returns, 'monthly')
    if monthly.empty:
        ax.set_title('Monthly Returns (%)')
        return
    m = monthly.unstack()
    # Ensure months 1..12 columns order
    m = m.reindex(columns=range(1,13))
    m = m.rename(columns={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
    sns.heatmap(m*100.0, annot=True, fmt='.1f', cmap=cm.RdYlGn, center=0.0, cbar=False, ax=ax, linewidths=0.5, linecolor='lightgray')
    ax.set_title('Monthly Returns (%)', fontweight='bold')
    ax.set_ylabel('Year')
    ax.set_xlabel('')


def _plot_yearly(ax, returns: pd.Series):
    yearly = aggregate_period_returns(returns, 'yearly') * 100.0
    if yearly.empty:
        ax.set_title('Yearly Returns (%)')
        return
    yearly.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
    ax.set_title('Yearly Returns (%)', fontweight='bold')
    ax.set_ylabel('%')
    ax.set_xlabel('')
    ax.yaxis.grid(linestyle=':')


def _plot_stats_panel(ax, returns: pd.Series, cum_rets: pd.Series, drawdown: pd.Series, max_dd: float, max_dur: int, periods: int):
    tot_ret = cum_rets.iloc[-1] - 1.0
    sr = sharpe_ratio(returns, periods)
    sor = sortino_ratio(returns, periods)
    vol = annualized_vol(returns, periods)
    cagr_v = cagr(cum_rets, periods)
    text_lines = [
        ('Total Return', f'{tot_ret:,.2%}'),
        ('CAGR', f'{cagr_v:,.2%}'),
        ('Sharpe', f'{sr:,.2f}'),
        ('Sortino', f'{sor:,.2f}' if np.isfinite(sor) else 'inf'),
        ('Ann.Vol', f'{vol:,.2%}'),
        ('Max DD', f'{max_dd:,.2%}'),
        ('Max DD Dur (d)', f'{max_dur}'),
    ]
    ax.set_title('Key Metrics', fontweight='bold')
    ax.axis('off')
    y = 0.95
    for label, val in text_lines:
        ax.text(0.05, y, label, fontsize=9, ha='left')
        ax.text(0.95, y, val, fontsize=9, ha='right', fontweight='bold')
        y -= 0.12

# -----------------------------
# Public API
# -----------------------------

def save_tearsheet(equity_series: pd.Series, title: str, out_file: str, periods: int = 252):
    if equity_series is None or equity_series.empty:
        raise ValueError('Equity series is empty')
    equity_series = equity_series.sort_index()
    returns = _daily_returns(equity_series)
    cum_rets = _cum_returns(returns)
    drawdown, max_dd, max_dur = create_drawdowns(cum_rets)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(14,10))
    fig.suptitle(title, fontweight='bold', fontsize=14)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2,0.8,0.8], hspace=0.35, wspace=0.25)

    ax_eq = plt.subplot(gs[0, :])
    ax_dd = plt.subplot(gs[1, :])
    ax_month = plt.subplot(gs[2, :2])
    ax_year = plt.subplot(gs[2, 2])

    _plot_equity(ax_eq, cum_rets)
    _plot_drawdown(ax_dd, drawdown)
    _plot_monthly(ax_month, returns)
    _plot_yearly(ax_year, returns)

    # Overlay stats panel inside equity plot (inset)
    inset = ax_eq.inset_axes([0.78,0.05,0.2,0.5])
    _plot_stats_panel(inset, returns, cum_rets, drawdown, max_dd, max_dur, periods)

    fig.savefig(out_file, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return {
        'total_return': cum_rets.iloc[-1]-1.0,
        'sharpe': sharpe_ratio(returns, periods),
        'max_drawdown': max_dd,
        'max_drawdown_duration': max_dur
    }

if __name__ == '__main__':
    # Simple self-test with synthetic equity
    idx = pd.date_range('2020-01-01','2022-12-31',freq='B')
    equity = pd.Series(1_000_000 * (1+0.0003)**np.arange(len(idx)), index=idx)
    save_tearsheet(equity, 'Synthetic Strategy', 'results/tearsheets/sample.png')
    print('Sample tearsheet saved.')

