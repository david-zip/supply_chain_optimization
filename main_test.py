"""
Looking to condense all the main files into one, mainly used for testing
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from neural_nets.model_ssa import Net
from neural_nets.model_reinforce import Net_reinforce
from environment import Multi_echelon_SupplyChain
from helper_functions.demand import random_uniform_demand_si, \
                                seasonal_random_uniform_control_si
from helper_functions.trajectory import J_supply_chain_ssa, \
                                    J_supply_chain_ssa_seasonality, \
                                    J_supply_chain_reinforce

from algorithms.sa import Simulated_Annealing, \
                            Parallelized_Simulated_Annealing
from algorithms.pso import Particle_Swarm_Optimization
from algorithms.abc import Artificial_Bee_Colony
from algorithms.ga import Genetic_Algorithm
from algorithms.de import Differential_Evolution
from algorithms.es import Gaussian_Evolutionary_Strategy, \
                            Covariance_Matrix_Adaption_Evolutionary_Strategy

def test_run(args):
    """
    Provide the keyname, run a selected type of algorithm
    """
    ### INITIALISE PARAMETERS ###
    # define SC parameters (siso - ORIGINAL)
    SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                    'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                    'material_cost':{0:12}, 'product_cost':{0:100}}
    """
    # define SC parameters (mimo - storage cost, prod_wt - comment)
    SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                    'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                    'material_cost':{0:12, 1:13, 2:11}, 'product_cost':{0:100, 1:300}}

    SC_params_ = {'echelon_storage_cost':(5/2,10/2,7/2,8/2), 'echelon_storage_cap' :(20,7,10,6),
                    'echelon_prod_cost' :(0,0,0,0), 'echelon_prod_wt' :((5,1),(7,1),(9,1),(11,3)),
                    'material_cost':{0:12}, 'product_cost':{0:100}}
    """
    n_echelons_ = 2

        # state and control actions
    u_norm_   = np.array([[20/6 for _ in range(n_echelons_ * len(SC_params_['product_cost']))],
                            [0 for _ in range(n_echelons_ * len(SC_params_['product_cost']))]])
    x_norm_   = np.array([10 for _ in range(n_echelons_ * len(SC_params_['product_cost']))])

    ### INITIALIZE ENVIRONMENT ###
    SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

    # policy hyperparameters
    hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                    'output_size': n_echelons_ * len(SC_params_['product_cost'])}

    # initialise neural net 
    policy_net = Net(**hyparams_)

    # run parameters
    SC_run_params_ = {}
    SC_run_params_['steps_tot']  = 365
    SC_run_params_['control_lb'] = 0
    SC_run_params_['control_ub'] = 20
    SC_run_params_['demand_lb']  = 12
    SC_run_params_['demand_ub']  = 15
    SC_run_params_['start_inv']  = 10
    SC_run_params_['demand_f']   = seasonal_random_uniform_control_si(12, 15, 365)
    SC_run_params_['u_norm']     = u_norm_
    SC_run_params_['x_norm']     = x_norm_
    SC_run_params_['hyparams']   = hyparams_

    # definig algo hyperparams
    SA_params_ = {}
    SA_params_['bounds']        = [-5, 5]
    SA_params_['temp']          = [1.0, 0.1]
    SA_params_['maxiter']       = 1000

    PSA_params_ = {}
    PSA_params_['bounds']       = [-5, 5]
    PSA_params_['population']   = 5
    PSA_params_['temp']         = [1.0, 0.1]
    PSA_params_['maxiter']      = 1000

    PSO_params_ = {}
    PSO_params_['bounds']       = [-5, 5]
    PSO_params_['weights']      = [0.5, 0.5, 1.0]
    PSO_params_['lambda']       = 0.99
    PSO_params_['population']   = 10
    PSO_params_['maxiter']      = 1000

    ABC_params_ = {}
    ABC_params_['bounds']       = [-5.0, 5.0]
    ABC_params_['population']   = 50
    ABC_params_['maxiter']      = 50

    GA_params_ = {}
    GA_params_['bounds']        = [-5.0, 5.0]
    GA_params_['numbits']       = 16
    GA_params_['population']    = 50
    GA_params_['cut']           = 0.4
    GA_params_['maxiter']       = 50

    GES_params_ = {}
    GES_params_['bounds']       = [-5.0, 5.0]
    GES_params_['population']   = 50
    GES_params_['elite_cut']    = 0.4
    GES_params_['maxiter']      = 50

    CMA_params_ = {}
    CMA_params_['bounds']       = [-5.0, 5.0]
    CMA_params_['population']   = 20
    CMA_params_['mean']         = 0
    CMA_params_['step_size']    = 1
    CMA_params_['elite_cut']    = 0.5
    CMA_params_['maxiter']      = 50

    DE_params_ = {}
    DE_params_['bounds']        = [-5, 5]
    DE_params_['population']    = 100
    DE_params_['scale']         = 0.5
    DE_params_['mutation']      = 0.3
    DE_params_['maxiter']       = 1000

    algo_dict = {
        'sa' :  Simulated_Annealing(model=policy_net, env=SC_model, **SA_params_),
        'psa':  Parallelized_Simulated_Annealing(model=Net, env=Multi_echelon_SupplyChain, echelons=n_echelons_, 
                                                    SC_params=SC_params_, hyparams=hyparams_, **PSA_params_),
        'pso':  Particle_Swarm_Optimization(model=policy_net, env=SC_model, **PSO_params_),
        'abc':  Artificial_Bee_Colony(model=policy_net, env=SC_model, **ABC_params_),
        'ga' :  Genetic_Algorithm(model=policy_net, env=SC_model, **GA_params_),
        'ges':  Gaussian_Evolutionary_Strategy(model=policy_net, env=SC_model, **GES_params_),
        'cma':  Covariance_Matrix_Adaption_Evolutionary_Strategy(model=policy_net, env=SC_model, **CMA_params_),
        'de' :  Differential_Evolution(model=policy_net, env=SC_model, **DE_params_)
    }

    for arg in args:
        best_policy, best_reward, R_list = algo_dict[arg].algorithm(function=J_supply_chain_ssa, 
                                                                        SC_run_params=SC_run_params_, 
                                                                        iter_debug=True)

        print(best_reward)

        # save model parameters
        torch.save(best_policy, f'neural_nets/parameters/test/{arg}.pth')

        print('training done!')

        plt.figure()
        plt.plot(R_list)
        plt.savefig(f'plots/test/training_plots/testfig{arg}.png')

if __name__=="__main__":
    """
    Options:
    - 'sa'          simulated annealing
    - 'psa'         parallelized simulated annealing
    - 'pso'         particle swarm optimization
    - 'abc'         artificial bee colony
    - 'ga'          genetic algorithm
    - 'ges'         gaussian evolutionary strategy
    - 'cma'         covariance matrix adaptation evolutionary strategy
    - 'de'          differential evolution
    """
    keynames = ['de']
    
    test_run(keynames)
