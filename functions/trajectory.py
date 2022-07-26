"""
J_supply_chain:
    run a complete trajectory
"""
import torch
from functions.demand import random_uniform_demand_si, seasonal_random_uniform_control_si

def J_supply_chain_reinforce(model, SC_run_params, policy):
    '''
    Adjusted version for REINFORCE
    '''
    # problem parameters
    steps_tot  = SC_run_params['steps_tot']
    u_norm     = SC_run_params['u_norm']   
    control_lb = SC_run_params['control_lb'] 
    control_ub = SC_run_params['control_ub']
    demand_lb  = SC_run_params['demand_lb']
    demand_ub  = SC_run_params['demand_ub']
    start_inv  = SC_run_params['start_inv']
    demand_f   = SC_run_params['demand_f']      # random_uniform_demand_si
    x_norm     = SC_run_params['x_norm']

    # se initial inventory
    model.SC_inventory[:,:] = start_inv         # starting inventory
    
    # reward
    r_list   = []
    order_list = []
    backlog = 0                                 # no backlog initially
    
    # first order
    state_norm                     = (model.supply_chain_state()[0,:] - x_norm[0])/x_norm[1]
    state_torch                    = torch.tensor((state_norm))
    order_k                        = policy(state_torch)
    order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]
    order_list.append(order_k)

    # === SC run === #
    for step_k in range(steps_tot):
        d_k_                           = random_uniform_demand_si(demand_lb, demand_ub) #random_uniform_demand_si
        d_k                            = d_k_ + backlog
        sale_product, r_k, backlog     = model.advance_supply_chain_orders_DE(order_k, d_k)
        #r_tot                         += r_k
        r_list.append(r_k)
        # agent makes order
        state_norm                     = (model.supply_chain_state()[0,:] - x_norm[0])/x_norm[1]
        state_torch                    = torch.tensor((state_norm))
        order_k                        = policy(state_torch)
        order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]
        order_list.append(order_k)

    return r_list, order_list

def J_supply_chain_ssa(model, SC_run_params, policy):
    '''
    Original version for stochastic search algorithms with slight modifications
    '''
    # problem parameters
    steps_tot  = SC_run_params['steps_tot']
    u_norm     = SC_run_params['u_norm']   
    control_lb = SC_run_params['control_lb']
    control_ub = SC_run_params['control_ub']
    demand_lb  = SC_run_params['demand_lb']
    demand_ub  = SC_run_params['demand_ub']
    start_inv  = SC_run_params['start_inv']
    demand_f   = SC_run_params['demand_f']          #random_uniform_demand_si
    x_norm     = SC_run_params['x_norm']

    # se initial inventory
    model.SC_inventory[:,:] = start_inv             # starting inventory
    # reward
    r_tot   = 0
    backlog = 0 # no backlog initially
    # first order
    state_norm                     = (model.supply_chain_state()[0,:] - x_norm[0])/x_norm[1]
    state_torch                    = torch.tensor((state_norm))
    order_k                        = policy(state_torch)
    order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]

    # === SC run === #
    for step_k in range(steps_tot):
        d_k_                           = random_uniform_demand_si(demand_lb, demand_ub) #random_uniform_demand_si
        d_k                            = d_k_ + backlog
        sale_product, r_k, backlog     = model.advance_supply_chain_orders(order_k, d_k)
        r_tot                         += r_k
        # agent makes order
        state_norm                     = (model.supply_chain_state()[0,:] - x_norm[0])/x_norm[1]
        state_torch                    = torch.tensor((state_norm))
        order_k                        = policy(state_torch)
        order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]

    return r_tot

def J_supply_chain_ssa_seasonality(model, SC_run_params, policy):
    '''
    Original version for stochastic search algorithms with seasonality
    '''
    # problem parameters
    steps_tot  = SC_run_params['steps_tot']
    u_norm     = SC_run_params['u_norm']   
    control_lb = SC_run_params['control_lb']
    control_ub = SC_run_params['control_ub']
    demand_lb  = SC_run_params['demand_lb']
    demand_ub  = SC_run_params['demand_ub']
    start_inv  = SC_run_params['start_inv']
    demand_f   = SC_run_params['demand_f']          #seasonal_random_uniform_control_si(lb, ub, tk)
    x_norm     = SC_run_params['x_norm']

    # se initial inventory
    model.SC_inventory[:,:] = start_inv             # starting inventory
    # reward
    r_tot   = 0
    backlog = 0 # no backlog initially
    # first order
    state_norm                     = (model.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
    state_torch                    = torch.tensor((state_norm))
    order_k                        = policy(state_torch)
    order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]

    # === SC run === #
    for step_k in range(steps_tot):
        d_k_                           = seasonal_random_uniform_control_si(demand_lb, demand_ub, model.supply_chain_state()[0,-1])
        d_k                            = d_k_ + backlog
        sale_product, r_k, backlog     = model.advance_supply_chain_orders_DE(order_k, d_k)
        r_tot                         += r_k
        # agent makes order
        state_norm                     = (model.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
        state_torch                    = torch.tensor((state_norm))
        order_k                        = policy(state_torch)
        order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]

    return r_tot
     