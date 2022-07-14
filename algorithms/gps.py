#################################
# Generalized policy search 
#################################
import torch
import copy
import numpy as np
from model import Net
from functions.trajectory import J_supply_chain

def sample_uniform_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min \
              for k, v in params_prev.items()}              
    return params

def sample_local_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min + v \
              for k, v in params_prev.items()}              
    return params
# afterwards: policy.load_state_dict(params)
# note: remember dict2 = copy.deepcopy(dict1)

def Generalized_policy_search(model, SC_run_params, 
                              shrink_ratio=0.5, radius=0.1, evals_shrink=10, 
                              evals=500, ratio_ls_rs=0.7):
    '''
    Tailores to address function
    '''

    # adapt evaluations    
    evals_rs = round(evals*ratio_ls_rs)
    evals_ls = evals - evals_rs

    # problem specs
    hyparams  = SC_run_params['hyparams']

    # data
    data_rs = {}
    data_rs['R_list'] = np.zeros((evals_rs))

    #######################
    # policy initialization
    #######################

    policy_net = Net(**hyparams)
    params     = policy_net.state_dict()
    # this can be changed manually
    param_max  =  5
    param_min  = -5
    # (-5,5)   337,155 30 => 351,304 60 =>
    # (-30,30) 317,125
    # (-10,10) 317,307 30 => 328,772 

    # == initialise rewards == #
    best_reward = -1e8
    best_policy = copy.deepcopy(params) 

    ###############
    # Random search
    ###############

    for policy_i in range(evals_rs):
        # sample a random policy
        NNparams_RS  = sample_uniform_params(params, param_max, param_min)
        # consrict policy to be evaluated
        policy_net.load_state_dict(NNparams_RS)
        # evaluate policy
        data_rs = J_supply_chain(model, SC_run_params, 
                                 data_rs, policy_net, policy_i)
        # benchmark reward ==> MAX ">"
        if data_rs['R_list'][policy_i]>best_reward:
            best_reward = data_rs['R_list'][policy_i]
            best_policy = copy.deepcopy(NNparams_RS)           

    ###############
    # local search
    ###############

    data_ls = {}
    data_ls['R_list'] = np.zeros((evals_ls))

    # define max radius
    r0 = np.array([param_min, param_max])*radius

    # initialization
    iter_i  = 0
    fail_i  = 0

    while iter_i < evals_ls:

        # shrink radius
        if fail_i >= evals_shrink:
            fail_i = 0
            radius = radius*shrink_ratio
            r0     = np.array([param_min, param_max])*radius

        # new parameters
        NNparams_LS = sample_local_params(best_policy, r0[1], r0[0])

        # == bounds adjustment == #
        # Done via ReLU6 
        
        # evaluate new agent
        policy_net.load_state_dict(NNparams_LS)
        data_ls      = J_supply_chain(model, SC_run_params, 
                                 data_ls, policy_net, iter_i)
        
        # choose the == MAX == value      
        if data_ls['R_list'][iter_i] > best_reward:
            best_reward = data_ls['R_list'][iter_i]
            best_policy = copy.deepcopy(NNparams_LS)
            fail_i = 0
        else:
            fail_i += 1

        # iteration counter
        iter_i += 1  
    print('final reward = ',best_reward)
    return best_policy, best_reward, data_rs, data_ls

