"""
Demand functions:
    - uniform demand between bounds
    - seasonal demand between bounds
"""
import numpy as np

# --- uniform random control --- #
def random_uniform_demand_si(lb, ub):
    range_ = ub - lb
    return np.random.randint(range_) + lb

# --- seasonal random control --- #
def seasonal_random_uniform_control_si(lb, ub, tk):
    range_ = ub - lb
    trend  = (np.sin(tk*365) + 1)/2
    return (np.random.randint(range_) + lb)*trend
