"""
Demand functions:
    - uniform demand between bounds
    - seasonal demand between bounds
"""
import numpy as np

# --- uniform random control --- #
def random_uniform_demand_si(*args):
    range_ = args[0] - args[1]
    return np.random.randint(range_) + args[1]

# --- seasonal random control --- #
def seasonal_random_uniform_control_si(*args):
    range_ = args[0] - args[1]
    trend  = (np.sin(args[2]*365) + 1)/2
    return (np.random.randint(range_) + args[1])*trend
