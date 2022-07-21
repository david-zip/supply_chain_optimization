"""
Main file to training model using algorithms
"""
import numpy as np
import matplotlib.pyplot as plt

from neural_nets.model_ssa import Net
from environment import Multi_echelon_SupplyChain
from functions.demand import random_uniform_demand_si
from functions.trajectory import J_supply_chain_ssa

# stochastic search algorithms
from algorithms.sa import Simulated_Annealing
from algorithms.pso import Particle_Swarm_Optimization
from algorithms.abc import Artificial_Bee_Colony
from algorithms.ga import Genetic_Algorithm
from algorithms.es import Gaussian_Evolutionary_Strategy

def train(maxIter=50, *args):
    # state and control actions
    u_norm_   = np.array([[20/6, 20/6], [0, 0]]);  # here I am assuming entry-wise normalisation
    x_norm_   = np.array([10, 10]);                # here I am assuming equal normalisation for all state entries

    # define SC parameters (siso - storage cost, prod_wt)
    SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                    'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                    'material_cost':{1:12}, 'product_cost':{1:100}}

    n_echelons_ = 2

    ### SIMULATED ANNEALING ###
    if 'sa' in args:
        # initialise environment
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

        # list to strore best rewards found
        SA_best         = []
        SA_mean         = []
        SA_std          = []
        SA_low_err      = []
        SA_high_err     = []

        SA_params_ = {}
        SA_params_['bounds']    = [-5, 5]
        SA_params_['temp']      = [1.0, 0.1]
        SA_params_['maxiter']   = 1000

        for i in range(maxIter):
            print(f"SA {i+1}")
            SA_optimizer = Simulated_Annealing(model=policy_net, env=SC_model, **SA_params_)

            best_policy, best_reward, R_list = SA_optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_)

            SA_best.append(R_list)

        for i in range(SA_optimizer.maxiter):
            SA_value_list = []

            for j in range(maxIter):
                SA_value_list.append(SA_best[j][i])
            
            SA_mean.append(np.mean(SA_value_list))
            SA_std.append(np.std(SA_value_list))

        for i in range(SA_optimizer.maxiter):
            SA_low_err.append(SA_mean[i] - SA_std[i])
            SA_high_err.append(SA_mean[i] + SA_std[i])

        # plot graphs
        fig = plt.figure()
        plt.suptitle(f"Simulated annealing for supply chain test - {maxIter} Iterations")
        plt.plot(range(SA_optimizer.maxiter), SA_mean, 'r-', label='Simulated Annealing')
        plt.fill_between(range(SA_optimizer.maxiter), SA_high_err, SA_low_err, alpha=0.3, edgecolor='r', facecolor='r')

        plt.xlabel("Number of algorithm iterations")
        plt.ylabel("Total reward")
        plt.yscale('log')

        plt.legend(loc="upper right")

        plt.savefig(f'plots/training_plots/simulated_annealing.png')

        print("SA training finished")

    ### PARTICLE SWARM OPTIMIZER ###
    if 'pso' in args:
        # initialise environment
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

        # define hyperparameters
        PSO_params_ = {}
        PSO_params_['bounds']        = [-5, 5]
        PSO_params_['weights']       = [0.2, 0.2, 1.0]
        PSO_params_['lambda']        = 0.99
        PSO_params_['population']    = 10
        PSO_params_['maxiter']       = 500

        # list to strore best rewards found
        PSO_best        = []
        PSO_mean        = []
        PSO_std         = []
        PSO_low_err     = []
        PSO_high_err    = []

        for i in range(maxIter):
            print(f"PSO {i+1}")

            PSO_optimizer = Particle_Swarm_Optimization(model=policy_net, env=SC_model, **PSO_params_)

            best_policy, best_reward, R_list = PSO_optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_)

            PSO_best.append(R_list)

        for i in range(PSO_optimizer.maxiter):
            PSO_value_list = []

            for j in range(maxIter):
                PSO_value_list.append(PSO_best[j][i])
            
            PSO_mean.append(np.mean(PSO_value_list))
            PSO_std.append(np.std(PSO_value_list))

        for i in range(PSO_optimizer.maxiter):
            PSO_low_err.append(PSO_mean[i] - PSO_std[i])
            PSO_high_err.append(PSO_mean[i] + PSO_std[i])

        # plot graphs
        fig = plt.figure()
        plt.suptitle(f"Particle swarm optimization for supply chain test - {maxIter} Iterations")
        plt.plot(range(PSO_optimizer.maxiter), PSO_mean, 'g-', label='Particle Swarm Optimization')
        plt.fill_between(range(PSO_optimizer.maxiter), PSO_high_err, PSO_low_err, alpha=0.3, edgecolor='g', facecolor='g')

        plt.xlabel("Number of algorithm iterations")
        plt.ylabel("Total reward")
        plt.yscale('log')
        plt.legend(loc="upper right")

        plt.savefig(f'plots/training_plots/particle_swarm_opimization.png')

        print("PSO training finished")

    ### ARTIFICIAL BEE COLONY ####
    if 'abc' in args:
        # initialise environment
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
        
        # list to strore best rewards found
        ABC_best        = []
        ABC_mean        = []
        ABC_std         = []
        ABC_low_err     = []
        ABC_high_err    = []

        # define hyperparameters
        ABC_params_ = {}
        ABC_params_['bounds']        = [-5, 5]
        ABC_params_['population']    = 10
        ABC_params_['maxiter']       = 100

        for i in range(maxIter):
            print(f"ABC {i+1}")

            ABC_optimizer = Artificial_Bee_Colony(model=policy_net, env=SC_model, **ABC_params_)

            best_policy, best_reward, R_list = ABC_optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_)

            ABC_best.append(R_list)

        for i in range(ABC_optimizer.maxiter):
            ABC_value_list = []

            for j in range(maxIter):
                ABC_value_list.append(ABC_best[j][i])
            
            ABC_mean.append(np.mean(ABC_value_list))
            ABC_std.append(np.std(ABC_value_list))

        for i in range(ABC_optimizer.maxiter):
            ABC_low_err.append(ABC_mean[i] - ABC_std[i])
            ABC_high_err.append(ABC_mean[i] + ABC_std[i])

        # plot graphs
        fig = plt.figure()
        plt.suptitle(f"Articial bee colony for supply chain test - {maxIter} Iterations")
        plt.plot(range(ABC_optimizer.maxiter), ABC_mean, 'b-', label='Artificial Bee Colony')
        plt.fill_between(range(ABC_optimizer.maxiter), ABC_high_err, ABC_low_err, alpha=0.3, edgecolor='b', facecolor='b')

        plt.xlabel("Number of algorithm iterations")
        plt.ylabel("Total reward")
        plt.legend(loc="upper right")

        plt.savefig(f'plots/training_plots/articial_bee_colony.png')

        print("ABC training finished")

    ### GENETIC ALGORITHM ###
    if 'ga' in args:
        # initialise environment
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

        # list to strore best rewards found
        GA_best         = []
        GA_mean         = []
        GA_std          = []
        GA_low_err      = []
        GA_high_err     = []

        # define hyperparameters
        GA_params_ = {}
        GA_params_['bounds']        = [-5.0, 5.0]
        GA_params_['numbits']       = 16
        GA_params_['population']    = 30
        GA_params_['cut']           = 0.4
        GA_params_['maxiter']       = 100

        for i in range(maxIter):
            print(f"GA {i+1}")
            GA_optimizer = Genetic_Algorithm(model=policy_net, env=SC_model, **GA_params_)

            best_policy, best_reward, R_list, best_gene = GA_optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_)

            GA_best.append(R_list)

        for i in range(GA_optimizer.maxiter):
            GA_value_list = []

            for j in range(maxIter):
                GA_value_list.append(GA_best[j][i])
            
            GA_mean.append(np.mean(GA_value_list))
            GA_std.append(np.std(GA_value_list))

        for i in range(GA_optimizer.maxiter):
            GA_low_err.append(GA_mean[i] - GA_std[i])
            GA_high_err.append(GA_mean[i] + GA_std[i])

        # genetic algorithm
        fig = plt.figure()
        plt.suptitle(f"Genetic algorithm for supply chain test - {maxIter} Iterations")
        plt.plot(range(GA_optimizer.maxiter), GA_mean, 'b-', label='Genetic Algorithm')
        plt.fill_between(range(GA_optimizer.maxiter), GA_high_err, GA_low_err, alpha=0.3, edgecolor='b', facecolor='b')

        plt.xlabel("Number of algorithm iterations")
        plt.ylabel("Total reward")
        plt.legend(loc="upper right")

        plt.savefig(f'plots/training_plots/genetic_algorithm.png')

        print("GA training finished")

    ### GAUSSIAN EVOLUTIONARY STRATEGY ###
    if 'ges' in args:
        # initialise environment
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

        # list to strore best rewards found
        GES_best        = []
        GES_mean        = []
        GES_std         = []
        GES_low_err     = []
        GES_high_err    = []

        # define hyperparameters
        GES_params_ = {}
        GES_params_['bounds']        = [-5.0, 5.0]
        GES_params_['population']    = 30
        GES_params_['elite_cut']     = 0.4
        GES_params_['maxiter']       = 100

        for i in range(maxIter):
            print(f"GES {i+1}")
            GES_optimizer = Gaussian_Evolutionary_Strategy(model=policy_net, env=SC_model, **GES_params_)

            best_policy, best_reward, R_list = GES_optimizer.algorithm(function=J_supply_chain_ssa, SC_run_params=SC_run_params_)

            GES_best.append(R_list)

        for i in range(GES_optimizer.maxiter):
            GES_value_list = []

            for j in range(maxIter):
                GES_value_list.append(GES_best[j][i])
            
            GES_mean.append(np.mean(GES_value_list))
            GES_std.append(np.std(GES_value_list))

        for i in range(GES_optimizer.maxiter):
            GES_low_err.append(GES_mean[i] - GES_std[i])
            GES_high_err.append(GES_mean[i] + GES_std[i])

        # plot graphs
        fig = plt.figure()
        plt.suptitle(f"Gaussian evoluation strategy for supply chain test - {maxIter} Iterations")
        plt.plot(range(GES_optimizer.maxiter), GES_mean, 'b-', label='Gaussian Evolutionary Strategy')
        plt.fill_between(range(GES_optimizer.maxiter), GES_high_err, GES_low_err, alpha=0.3, edgecolor='b', facecolor='b')

        plt.xlabel("Number of algorithm iterations")
        plt.ylabel("Total reward")
        plt.legend(loc="upper right")

        plt.savefig(f'plots/training_plots/gaussian_evolutionary_strategy.png')

        print("GES training finished")

    ### REINFORCE ### (get reinforce working first)
    if 'reinforce' in args:
        pass

if __name__=="__main__":
    """
    Options:
    - 'sa'          simulatied annealing
    - 'pso'         particle swarm optimization
    - 'abc'         artificial bee colony
    - 'ga'          genetic algorithm
    - 'ges'         gaussian evolutionary strategy
    - 'reinforce'   reinforce (not implemented)
    """
    keynames = ['pso']

    train(30, *keynames)
