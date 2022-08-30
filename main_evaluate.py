"""
Main file to evaluate the model performance
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from environment import Multi_echelon_SupplyChain
from neural_nets.model_ssa import Net
from helper_functions.demand import random_uniform_demand_si, seasonal_random_uniform_control_si

def evaluate(**kwargs):

    if kwargs['io'] == "mimo":
        
        # define SC parameters (mimo - storage cost, prod_wt - comment)
        SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                        'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                        'material_cost':{0:12, 1:13, 2:11}, 'product_cost':{0:100, 1:300}}
        """
        SC_params_ = {'echelon_storage_cost':(3,8), 'echelon_storage_cap' :(10,3),
                        'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                        'material_cost':{0:5}, 'product_cost':{0:50}}
        """
    else:   # call siso parameters
        # define SC parameters (siso - ORIGINAL)
        SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                        'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                        'material_cost':{0:12}, 'product_cost':{0:100}}

    n_echelons_ = kwargs["echelons"]

    # build model
    SC_model   = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)


    # policy hyperparameters
    hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                    'output_size': n_echelons_}

    # state and control actions
    u_norm_   = np.array([[20/6 for _ in range(n_echelons_)], 
                            [0 for _ in range(n_echelons_)]]) 
    x_norm_   = np.array([10 for _ in range(n_echelons_)])

    # run parameters
    SC_run_params_ = {}
    SC_run_params_['steps_tot']  = 365
    SC_run_params_['control_lb'] = 0
    SC_run_params_['control_ub'] = 20
    SC_run_params_['demand_lb']  = 12
    SC_run_params_['demand_ub']  = 15
    SC_run_params_['start_inv']  = 10
    SC_run_params_['demand_f']   = random_uniform_demand_si
    SC_run_params_['u_norm']     = u_norm_
    SC_run_params_['x_norm']     = x_norm_
    SC_run_params_['hyparams']   = hyparams_

    # define parameters
    steps_tot  = SC_run_params_['steps_tot']
    u_norm     = SC_run_params_['u_norm']   
    control_lb = SC_run_params_['control_lb'] 
    control_ub = SC_run_params_['control_ub']
    demand_lb  = SC_run_params_['demand_lb']
    demand_lb  = SC_run_params_['demand_lb']
    start_inv  = SC_run_params_['start_inv']
    demand_f   = SC_run_params_['demand_f']
    x_norm     = SC_run_params_['x_norm']

    # simulation parameters
    steps_tot  = SC_run_params_['steps_tot'] 
    control_lb = 0; control_ub = 20
    demand_lb  = 12; demand_ub = 15
    SC_model.SC_inventory[:,:] = 10

    # lists for statistics
    reward_history          = np.zeros((steps_tot))
    backlog_history         = np.zeros((steps_tot))
    demand_history          = np.zeros((steps_tot))
    demand_backlog_history  = np.zeros((steps_tot))
    orders_history          = np.zeros((n_echelons_, steps_tot))
    warehouse_history       = np.zeros((n_echelons_, steps_tot))
    # lists for statistics on reward
    reward_product          = np.zeros((steps_tot))
    reward_raw_mat          = np.zeros((steps_tot))
    reward_storage          = np.zeros((steps_tot))
    reward_bakclog          = np.zeros((steps_tot))
    
    # load best policy
    policy_net = Net(**SC_run_params_['hyparams'])
    policy_net.load_state_dict(torch.load(kwargs['path']))

    # initial order
    state_norm                     = (SC_model.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
    state_time                     = SC_model.supply_chain_state()[0,-1] / 365
    state_torch                    = torch.tensor(np.hstack((state_norm, state_time)))
    order_k                        = policy_net(state_torch)
    order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]

    backlog = 0
    # main loop
    for step_k in range(steps_tot):
        df_params                      = [demand_ub, demand_lb, step_k+1]  # set demand function paramters
        d_k_                           = random_uniform_demand_si(*df_params)
        demand_history[step_k]         = d_k_
        d_k                            = d_k_ + backlog
        sale_product, r_k, backlog     = SC_model.advance_supply_chain_orders(order_k, d_k)
        reward_history[step_k]         = r_k
        backlog_history[step_k]        = backlog
        demand_backlog_history[step_k] = d_k
        # agent makes order
        state_norm                     = (SC_model.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
        state_time                     = SC_model.supply_chain_state()[0,-1] / 365
        state_torch                    = torch.tensor(np.hstack((state_norm, state_time)))
        order_k                        = policy_net(state_torch)
        order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]
        # reward stats
        reward_product[step_k]         = SC_model.r_product
        reward_raw_mat[step_k]         = SC_model.r_raw_mat
        reward_storage[step_k]         = SC_model.r_storage
        reward_bakclog[step_k]         = SC_model.r_bakclog
        orders_history[:,step_k]       = order_k
        warehouse_history[:,step_k]    = SC_model.warehouses

    # plot results
    fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2)
    fig.suptitle('Supply chain management')
    fig.set_size_inches(18.5*.7, 10.5*.7)

    ax1.plot(reward_history, '.-')
    ax1.set_xlabel('Reward history')
    ax1.set_ylabel('£')

    ax2.plot(backlog_history, '.-')
    ax2.set_xlabel('Backlog history')
    ax2.set_ylabel('# items')

    ax3.plot(demand_history, '.-')
    ax3.set_xlabel('Demand history')
    ax3.set_ylabel('£')

    ax4.plot(demand_backlog_history, '.-')
    ax4.set_xlabel('Demand + backlog history')
    ax4.set_ylabel('£')

    ax5.plot(warehouse_history[0,:], '.-')
    ax5.set_xlabel('first warehouse')
    ax5.set_ylabel('£')

    ax6.plot(warehouse_history[1,:], '.-')
    ax6.set_xlabel('second warehouse')
    ax6.set_ylabel('£')
    fig.tight_layout()

    plt.show()

if __name__=="__main__":
    """
    keywords['io']       = 'siso' or 'mimo'
    keywords['echelons'] = int
    keywords['path']     = relative file pathway for parameters
    """
    keywords = {}
    keywords['io']       = 'siso'
    keywords['echelons'] = 2
    keywords['path']     = 'neural_nets/parameters/test/sa.pth'

    evaluate(**keywords)