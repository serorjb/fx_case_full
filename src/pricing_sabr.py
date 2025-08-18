import numpy as np
from math import log, sqrt
from quantlib_utils import sabr_vol_ql

def hagan_vol(F, K, T, alpha, rho, nu, beta=1.0, eps=1e-12):
    if T<=0 or F<=0 or K<=0: return 0.0
    if abs(F-K)<eps:
        fk_beta = (F*K)**((1-beta)/2) if beta!=1.0 else 1.0
        numer = alpha
        denom = fk_beta*(1 + ((1-beta)**2/24)*(log(F/K))**2 + ((1-beta)**4/1920)*(log(F/K))**4 ) if beta!=1.0 else 1.0
        term = 1 + ( (((1-beta)**2)*alpha**2)/(24*(F*K)**(1-beta)) + 0.25*rho*beta*alpha*nu/((F*K)**((1-beta)/2)) + (2-3*rho**2)*nu**2/24 )*T
        return (numer/denom)*term
    z = (nu/alpha) * (F*K)**((1-beta)/2) * log(F/K)
    xz = np.log( (np.sqrt(1-2*rho*z + z*z) + z - rho)/(1-rho) )
    fk_beta = (F*K)**((1-beta)/2) if beta!=1.0 else 1.0
    numer = alpha
    denom = fk_beta*(1 + ((1-beta)**2/24)*(log(F/K))**2 + ((1-beta)**4/1920)*(log(F/K))**4 ) if beta!=1.0 else 1.0
    vol = (numer/denom) * (z/xz) * ( 1 + ( (((1-beta)**2)*alpha**2)/(24*(F*K)**(1-beta)) + 0.25*rho*beta*alpha*nu/((F*K)**((1-beta)/2)) + (2-3*rho**2)*nu**2/24 )*T )
    return max(vol, 1e-8)

def sabr_vol(F,K,T,alpha,beta,nu,rho):
    v = sabr_vol_ql(F,K,T,alpha,beta,nu,rho)
    if v is not None:
        return float(v)
    return hagan_vol(F,K,T,alpha,rho,nu,beta=beta)

def calibrate_sabr(F, T, strikes, target_vols, beta=1.0):
    a_grid  = np.geomspace(0.005, 1.0, 12)
    nu_grid = np.geomspace(0.10, 2.0, 12)
    rho_grid = np.linspace(-0.9, 0.9, 9)
    best = (1e9, 0.2, -0.3, 0.5)

    data = [(K, tv) for K, tv in zip(strikes, target_vols) if (K is not None and tv is not None and K > 0 and tv > 0)]
    if len(data) == 0:
        return {"alpha": 0.2, "rho": 0.0, "nu": 0.5, "beta": beta}

    for a in a_grid:
        for nu in nu_grid:
            for rho in rho_grid:
                err = 0.0
                for K, tv in data:
                    sv = sabr_vol(F, K, T, a, beta, nu, rho)
                    err += (sv - tv) ** 2
                if err < best[0]:
                    best = (err, a, rho, nu)
    return {"alpha": best[1], "rho": best[2], "nu": best[3], "beta": beta}

