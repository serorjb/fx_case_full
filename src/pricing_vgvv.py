import numpy as np
def vgvv_implied_vol(F, K, T, anchors, vols):
    K0, Kc, Kp = anchors
    s0, sc, sp = vols
    eps = 1e-12
    def w(x, xj): return 1.0/max(abs(x-xj), eps)
    x = np.log(F/max(K,1e-12))
    xs = [np.log(F/max(K0,1e-12)), np.log(F/max(Kc,1e-12)), np.log(F/max(Kp,1e-12))]
    sigs = [s0, sc, sp]
    raw = np.array([w(x, xs[0]), w(x, xs[1]), w(x, xs[2])])
    raw = raw/raw.sum()
    return max(1e-6, float((raw*np.array(sigs)).sum()))
