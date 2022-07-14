"""
Main file to run RL agent using REINFORCE
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Net
from environment import Multi_echelon_SupplyChain
from functions.demand import random_uniform_demand_si
from functions.trajectory import J_supply_chain_reinforce
from algorithms.reinforce import reinforce

### INITIALISE ENVIRONMENT ###
# state and control actions
u_norm_   = np.array([[20/6, 20/6], [0, 0]]);  # here I am assuming entry-wise normalisation
x_norm_   = np.array([10, 10]);                # here I am assuming equal normalisation for all state entries

# define SC parameters (siso - storage cost, prod_wt - comment)
SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                'material_cost':{1:12}, 'product_cost':{1:100}}

n_echelons_ = 2

# initialise environment
SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

# policy hyperparameters
hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                'output_size': 2}

# run parameters
SC_run_params_ = {}
SC_run_params_['steps_tot']  = 365
SC_run_params_['control_lb'] = 0
SC_run_params_['control_ub'] = 20
SC_run_params_['demand_lb']  = 12
SC_run_params_['demand_ub']  = 15
SC_run_params_['start_inv']  = 10
SC_run_params_['demand_f']   = random_uniform_demand_si(12, 15)
SC_run_params_['u_norm']     = u_norm_
SC_run_params_['x_norm']     = x_norm_
SC_run_params_['hyparams']   = hyparams_

### INITIALISE ALGORITHM ###
# define hyperparameters
GAMMA = 0.90
LEARNING_RATE = 1e-2
TOTAL_EPISODES = 1000

# define problem specs
hyparams  = SC_run_params_['hyparams']

# initialise policy and save initial parameters
policy_net = Net(**hyparams)
params1    = policy_net.state_dict()

param_max  =  5   # this can be changed manually
param_min  = -5   # this can be changed manually

# define optimizer

# start training agent
total_rewards = []
for t in range(TOTAL_EPISODES):
    print(f'episode {t+1}')

    # determine rewards and actions
    rewards, orders = J_supply_chain_reinforce(SC_model, SC_run_params_, policy_net)

    total_rewards.append(sum(rewards))

    # get loss
    reinforce(policy_net, rewards, orders, GAMMA, LEARNING_RATE)

params2 = policy_net.state_dict()
print("training complete")
"""
print(f'params before\n{params1}')
print(f'params after\n{params2}')
"""

# plot data
plt.plot(total_rewards)
plt.savefig("testfig.png")