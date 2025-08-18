import pandas as pd, numpy as np
def perf_stats(equity: pd.Series, freq=252):
    rets = equity.pct_change().dropna()
    ann_ret = (1+rets.mean())**freq - 1
    ann_vol = rets.std()*np.sqrt(freq)
    sharpe = 0 if ann_vol==0 else ann_ret/ann_vol
    roll_max = equity.cummax()
    dd = (equity/roll_max - 1.0).min()
    return {"ann_return":float(ann_ret), "ann_vol":float(ann_vol), "sharpe":float(sharpe), "max_drawdown":float(dd)}
