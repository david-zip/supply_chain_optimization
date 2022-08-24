"""
Heuristic suppy chain agent
"""
import numpy as np

# --- control and storage agent --- #
def agent_control_storage(demand, storage, n_echelon):
    orders    = np.ones(n_echelon)*demand
    if storage>2*demand:
        orders[0] = int(demand*0.8)
    return orders