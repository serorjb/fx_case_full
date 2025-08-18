import numpy as np
from math import sqrt
from scipy.stats import norm
from quantlib_utils import black_price_ql

def _black_price_py(F,K,sigma,T,df=1.0,call=True):
    if T<=0 or sigma<=0:
        intrinsic = max((F-K) if call else (K-F), 0.0)
        return df*intrinsic
    v = sigma*sqrt(T)
    d1 = (np.log(F/K)+0.5*v*v)/v
    d2 = d1 - v
    return df*(F*norm.cdf(d1)-K*norm.cdf(d2)) if call else df*(K*norm.cdf(-d2)-F*norm.cdf(-d1))

def black_price(F,K,sigma,T,df=1.0,call=True):
    pv = black_price_ql(F,K,sigma,T,df,call)
    if pv is not None:
        return float(pv)
    return _black_price_py(F,K,sigma,T,df,call)
