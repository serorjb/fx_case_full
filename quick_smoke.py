import pandas as pd
from rates import ensure_rates
from data_loader import FXDataLoader
from sabr_backtest import SABRSmileBacktester
from vgvv_backtest import VGVVSmileBacktester

if __name__ == '__main__':
    ensure_rates()
    loader = FXDataLoader()
    pairs = ['EURUSD']
    start = '2024-01-02'; end = '2024-03-15'
    print('Quick smoke SABR...')
    sabr = SABRSmileBacktester(loader, pairs, start, end,
                               allocation_mode='return',
                               initial_capital=1_000_000,
                               max_notional=200_000,
                               use_moneyness_cost=True)
    res_s = sabr.run()
    print('SABR result:', {k: res_s[k] for k in ['final_equity','total_return','sharpe_ratio']})

    print('Quick smoke VGVV...')
    vgvv = VGVVSmileBacktester(loader, pairs, start, end,
                               allocation_mode='return',
                               initial_capital=1_000_000,
                               max_notional=200_000,
                               use_moneyness_cost=True)
    res_v = vgvv.run()
    print('VGVV result:', {k: res_v[k] for k in ['final_equity','total_return','sharpe_ratio']})

