"""
Looking to condense all the main files into one, mainly used for testing
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from neural_nets.model_ssa import Net
from neural_nets.model_reinforce import Net_reinforce
from environment import Multi_echelon_SupplyChain
from functions.demand import random_uniform_demand_si, \
                                seasonal_random_uniform_control_si
from functions.trajectory import J_supply_chain_ssa, \
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
from algorithms.reinforce import reinforce

def test_run(*args):
    """
    Provide the keyname, run a selected type of algorithm
    """
    ### INITIALISE PARAMETERS ###
    # state and control actions
    u_norm_   = np.array([[20/6, 20/6], [0, 0]]);  # here I am assuming entry-wise normalisation
    x_norm_   = np.array([10, 10]);                # here I am assuming equal normalisation for all state entries
    
    # define SC parameters (siso - ORIGINAL)
    SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                    'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                    'material_cost':{1:12}, 'product_cost':{1:100}}
    
    """
    # define SC parameters (mimo - storage cost, prod_wt - comment)
    SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                    'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                    'material_cost':{0:12, 1:13, 2:11}, 'product_cost':{0:100, 1:300}}
    """
    n_echelons_ = 2

    if 'sa' in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                        'output_size': 2}

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

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        SA_params_ = {}
        SA_params_['bounds']    = [-5, 5]
        SA_params_['temp']      = [1.0, 0.1]
        SA_params_['maxiter']   = 1000

        optimizer = Simulated_Annealing(model=policy_net, env=SC_model, **SA_params_)

        best_policy, best_reward, R_list = optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_, iter_debug=True)

        #print(best_policy)
        print(best_reward)
                
        # save model parameters
        torch.save(best_policy, 'neural_nets/parameters/test/sa.pth')

        print('training done!')

        # plot figure
        plt.figure()
        plt.plot(R_list)
        plt.yscale('log')
        plt.savefig('plots/test/training_plots/testfigSA.png')

    if 'psa' in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model1 = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)
        SC_model2 = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)
        SC_model3 = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)
        SC_model4 = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)
        SC_model5 = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        SC_model  = [SC_model1, SC_model2, SC_model3, SC_model4, SC_model5]

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model1.supply_chain_state()[0,:].shape[0], 
                        'output_size': 2}

        # initialise neural net
        policy_net1 = Net(**hyparams_)
        policy_net2 = Net(**hyparams_)
        policy_net3 = Net(**hyparams_)
        policy_net4 = Net(**hyparams_)
        policy_net5 = Net(**hyparams_)

        policy_net = [policy_net1, policy_net2, policy_net3, policy_net4, policy_net5]

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

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        PSA_params_ = {}
        PSA_params_['bounds']     = [-5, 5]
        PSA_params_['population'] = 5
        PSA_params_['temp']       = [1.0, 0.1]
        PSA_params_['maxiter']    = 1000

        optimizer = Parallelized_Simulated_Annealing(model=policy_net, env=SC_model, **PSA_params_)

        best_policy, best_reward, R_list = optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_, iter_debug=True)

        #print(best_policy)
        print(best_reward)
                
        # save model parameters
        torch.save(best_policy, 'neural_nets/parameters/test/psa.pth')

        print('training done!')

        # plot figure
        plt.figure()
        plt.plot(R_list)
        plt.yscale('log')
        plt.savefig('plots/test/training_plots/testfigPSA.png')
    
    if 'pso' in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                            'output_size': 2}

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
        SC_run_params_['demand_f']   = random_uniform_demand_si(12, 15)
        SC_run_params_['u_norm']     = u_norm_
        SC_run_params_['x_norm']     = x_norm_
        SC_run_params_['hyparams']   = hyparams_

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        PSO_params_ = {}
        PSO_params_['bounds']        = [-5, 5]
        PSO_params_['weights']       = [0.5, 0.5, 1.0]
        PSO_params_['lambda']        = 0.99
        PSO_params_['population']    = 10
        PSO_params_['maxiter']       = 1000

        optimizer = Particle_Swarm_Optimization(model=policy_net, env=SC_model, **PSO_params_)

        best_policy, best_reward, R_list = optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_)

        print(best_reward)

        # save model parameters
        torch.save(best_policy, 'neural_nets/parameters/test/pso.pth')
        
        print('training done!')
    
        plt.figure()
        plt.plot(R_list)
        plt.savefig('plots/test/training_plots/testfigPSO.png')
    
    if 'abc' in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                            'output_size': 2}

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
        SC_run_params_['demand_f']   = random_uniform_demand_si(12, 15)
        SC_run_params_['u_norm']     = u_norm_
        SC_run_params_['x_norm']     = x_norm_
        SC_run_params_['hyparams']   = hyparams_

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        ABC_params_ = {}
        ABC_params_['bounds']        = [-5.0, 5.0]
        ABC_params_['population']    = 50
        ABC_params_['maxiter']       = 50

        optimizer = Artificial_Bee_Colony(model=policy_net, env=SC_model, **ABC_params_)

        best_policy, best_reward, R_list = optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_)

        print(best_reward)

        # save model parameters
        torch.save(best_policy, 'neural_nets/parameters/test/abc.pth')

        print('training done!')

        plt.figure()
        plt.plot(R_list)
        plt.savefig('plots/test/training_plots/testfigABC.png')

    if 'ga' in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                            'output_size': 2}

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
        SC_run_params_['demand_f']   = random_uniform_demand_si(12, 15)
        SC_run_params_['u_norm']     = u_norm_
        SC_run_params_['x_norm']     = x_norm_
        SC_run_params_['hyparams']   = hyparams_

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        GA_params_ = {}
        GA_params_['bounds']        = [-5.0, 5.0]
        GA_params_['numbits']       = 16
        GA_params_['population']    = 50
        GA_params_['cut']           = 0.4
        GA_params_['maxiter']       = 100

        optimizer = Genetic_Algorithm(model=policy_net, env=SC_model, **GA_params_)

        best_policy, best_reward, R_list, best_gene = optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_, iter_debug=True)

        print(best_reward)

        # save model parameters
        torch.save(best_policy, 'neural_nets/parameters/test/ga.pth')

        print('training done!')

        plt.figure()
        plt.plot(R_list)
        plt.savefig('plots/test/training_plots/testfigGA.png')
    
    if "ges" in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                            'output_size': 2}

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
        SC_run_params_['demand_f']   = random_uniform_demand_si(12, 15)
        SC_run_params_['u_norm']     = u_norm_
        SC_run_params_['x_norm']     = x_norm_
        SC_run_params_['hyparams']   = hyparams_

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        GES_params_ = {}
        GES_params_['bounds']        = [-5.0, 5.0]
        GES_params_['population']    = 50
        GES_params_['elite_cut']     = 0.4
        GES_params_['maxiter']       = 50

        optimizer = Gaussian_Evolutionary_Strategy(model=policy_net, env=SC_model, **GES_params_)

        best_policy, best_reward, R_list = optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_)

        print(best_reward)

        # save model parameters
        torch.save(best_policy, 'neural_nets/parameters/test/ges.pth')

        print('training done!')

        plt.figure()
        plt.plot(R_list)
        plt.savefig('plots/test/training_plots/testfigGES.png')

    if "cma" in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                            'output_size': 2}

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
        SC_run_params_['demand_f']   = random_uniform_demand_si(12, 15)
        SC_run_params_['u_norm']     = u_norm_
        SC_run_params_['x_norm']     = x_norm_
        SC_run_params_['hyparams']   = hyparams_

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        CMA_params_ = {}
        CMA_params_['bounds']        = [-5.0, 5.0]
        CMA_params_['population']    = 20
        CMA_params_['mean']          = 0
        CMA_params_['step_size']     = 1
        CMA_params_['elite_cut']     = 0.5
        CMA_params_['maxiter']       = 50

        optimizer = Covariance_Matrix_Adaption_Evolutionary_Strategy(model=policy_net, env=SC_model, **CMA_params_)

        best_policy, best_reward, R_list = optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_, iter_debug=True)

        print(best_reward)

        # save model parameters
        torch.save(best_policy, 'neural_nets/parameters/test/cma.pth')

        print('training done!')

        plt.figure()
        plt.plot(R_list)
        plt.savefig('plots/test/training_plots/testfigCMA.png')

    if "de" in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                        'output_size': 2}

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

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        DE_params_ = {}
        DE_params_['bounds']     = [-5, 5]
        DE_params_['population'] = 100
        DE_params_['scale']      = 0.5
        DE_params_['mutation']   = 0.3
        DE_params_['maxiter']    = 1000

        optimizer = Differential_Evolution(model=policy_net, env=SC_model, **DE_params_)

        best_policy, best_reward, R_list = optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_, iter_debug=True)

        #print(best_policy)
        print(best_reward)

        # save model parameters
        torch.save(best_policy, 'neural_nets/parameters/test/de.pth')

        print('training done!')

        # plot figure
        plt.figure()
        plt.plot(R_list)
        #plt.yscale('log')
        plt.savefig('plots/test/training_plots/testfigDE.png')

    if 'reinforce' in args:
        ### INITIALIZE ENVIRONMENT ###
        SC_model = Multi_echelon_SupplyChain(n_echelons=n_echelons_, SC_params=SC_params_)

        # policy hyperparameters
        hyparams_ = {'input_size': SC_model.supply_chain_state()[0,:].shape[0], 
                            'output_size': 2}

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
        SC_run_params_['demand_f']   = random_uniform_demand_si(12, 15)
        SC_run_params_['u_norm']     = u_norm_
        SC_run_params_['x_norm']     = x_norm_
        SC_run_params_['hyparams']   = hyparams_

        ### INITIALISE ALGORITHM ###
        # define hyperparameters
        GAMMA = 0.90
        LEARNING_RATE = 1e-1
        TOTAL_EPISODES = 1000

        # define problem specs
        hyparams  = SC_run_params_['hyparams']

        # initialise policy and save initial parameters
        policy_net = Net_reinforce(**hyparams)
        params1    = policy_net.state_dict()

        print(params1)

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
        plt.figure()
        plt.plot(total_rewards)
        plt.savefig("plots/test/testfigREINFORCE.png")

if __name__=="__main__":
    """
    Options:
    - 'sa'          simulatied annealing
    - 'psa'         parallelized simulated annealing
    - 'pso'         particle swarm optimization
    - 'abc'         artificial bee colony
    - 'ga'          genetic algorithm
    - 'ges'         gaussian evolutionary strategy
    - 'cma'         covariance matrix adaptation evolutionary strategy
    - 'de'          differential evolution
    - 'reinforce'   reinforce
    """
    keynames = ['psa']
    
    test_run(*keynames)
