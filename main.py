import pandas as pd
from rates import ensure_rates
from data_loader import FXDataLoader
from sabr_backtest import SABRSmileBacktester
from vgvv_backtest import VGVVSmileBacktester


def main():
    ensure_rates()
    loader = FXDataLoader()
    # pairs = ['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
    pairs = list()
    start = '2006-09-01'
    end = '2024-12-31'

    print('=== Running SABR backtest ===')
    sabr = SABRSmileBacktester(loader, pairs, start, end, allocation_mode='sortino', report_start_date='2007-01-01')
    res_sabr = sabr.run()
    print('SABR summary:', {k: res_sabr[k] for k in ['total_return','annualized_return','sharpe_ratio','max_drawdown']})

    print('\n=== Running VGVV backtest ===')
    vgvv = VGVVSmileBacktester(loader, pairs, start, end, allocation_mode='sortino', report_start_date='2007-01-01')
    res_vgvv = vgvv.run()
    print('VGVV summary:', {k: res_vgvv[k] for k in ['total_return','annualized_return','sharpe_ratio','max_drawdown']})


if __name__ == '__main__':
    main()

