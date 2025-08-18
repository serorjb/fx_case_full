import numpy as np
from math import sqrt
from scipy.stats import norm

def black_forward_delta(F,K,sigma,T,call=True):
    v = sigma*sqrt(max(T,1e-12))
    if v==0: return 0.0
    d1 = (np.log(F/K)+0.5*v*v)/v
    return norm.cdf(d1) if call else norm.cdf(d1)-1.0

def strike_from_delta(F, target_abs_delta, sigma, T, call=True, tol=1e-10, maxit=50):
    z = norm.ppf(target_abs_delta if call else 1.0-target_abs_delta)
    v = sigma*sqrt(T)
    K = F*np.exp(0.5*v*v - z*v)
    for _ in range(maxit):
        d = black_forward_delta(F,K,sigma,T,call=call)
        err = d - (target_abs_delta if call else -target_abs_delta)
        if abs(err)<tol: break
        grad = - (np.exp(-0.5*z*z)/np.sqrt(2*np.pi)) / (K*(v if v>1e-12 else 1e-12))
        K -= err/grad
        if K<=1e-12: K = 1e-8
    return K
