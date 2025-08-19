def option_trade_cost(premium: float, abs_delta: float, wings_bps=30, atm_bps=15):
    bps = wings_bps if abs_delta <= 0.10 else atm_bps
    return abs(premium) * bps * 1e-4

def hedge_cost_from_pips(notional: float, pips: float):
    return abs(notional) * pips * 1e-4
