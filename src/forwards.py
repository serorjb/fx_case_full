import numpy as np
def forward_from_points(spot: float, fwd_points: float) -> float:
    return spot + fwd_points
def r_dom_minus_r_for_from_F(S: float, F: float, T: float) -> float:
    return np.log(max(F,1e-12)/max(S,1e-12))/max(T,1e-10)
