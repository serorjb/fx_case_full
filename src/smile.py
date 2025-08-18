def vols_from_atm_rr_bf(atm, rr, bf):
    sc = atm + 0.5*bf + 0.5*rr
    sp = atm + 0.5*bf - 0.5*rr
    return sc, sp
