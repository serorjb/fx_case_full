def margin_requirement(vega: float, T: float, S: float, kappa=0.10, floor=0.01, notional=1.0):
    vol_term = kappa * abs(vega) * (T**0.5) * S
    return max(vol_term, floor*abs(notional))
