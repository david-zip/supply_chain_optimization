"""
Main file to evaluate the model performance
"""
import os
import csv
import glob
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from environment import Multi_echelon_SupplyChain
from neural_nets.model_ssa import Net
from helper_functions.demand import random_uniform_demand_si, seasonal_random_uniform_control_si
from helper_functions.agent import agent_control_storage

mpl.style.use('ggplot')

def evaluate(args, **kwargs):

    if kwargs['io'] == "mimo":
        
        # define SC parameters (mimo - storage cost, prod_wt - comment)
        SC_params_ = {'echelon_storage_cost':(5/2,10/2), 'echelon_storage_cap' :(20,7),
                        'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1)),
                        'material_cost':{0:12, 1:13, 2:11}, 'product_cost':{0:100, 1:300}}
    else:   # call siso parameters
        # define SC parameters (siso - ORIGINAL)
        SC_params_ = {'echelon_storage_cost':(5, 10, 7, 8, 6), 'echelon_storage_cap' :(20, 15, 20, 10, 10),
                        'echelon_prod_cost' :(0, 0, 0, 0, 0), 'echelon_prod_wt' :((5,1),(7,1),(10,1),(4,1),(6,1)),
                        'material_cost':{0:20}, 'product_cost':{0:150}}

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
    reward_history          = {arg: np.zeros((steps_tot)) for arg in args}
    backlog_history         = {arg: np.zeros((steps_tot)) for arg in args}
    demand_history          = np.zeros((steps_tot)) # demand should remain constant for each experiment
    demand_backlog_history  = {arg: np.zeros((steps_tot)) for arg in args}
    orders_history          = {arg: np.zeros((n_echelons_, steps_tot)) for arg in args}
    warehouse_history       = {arg: np.zeros((n_echelons_, steps_tot)) for arg in args}
    
    # lists for statistics on reward
    reward_product          = {arg: np.zeros((steps_tot)) for arg in args}
    reward_raw_mat          = {arg: np.zeros((steps_tot)) for arg in args}
    reward_storage          = {arg: np.zeros((steps_tot)) for arg in args}
    reward_backlog          = {arg: np.zeros((steps_tot)) for arg in args}

    # stores average reward per day
    avg_reward = {arg: 0 for arg in args}

    # initialize demand values
    for step_k in range(steps_tot):
        df_params                      = [demand_ub, demand_lb, step_k+1]  # set demand function paramters
        demand_history[step_k]         = random_uniform_demand_si(*df_params)

    with open(f'outputs/final_results/plots/action_plots/demand_data/demand.csv', mode='w') as dcsv:
        dcsv_out = csv.writer(dcsv)
        dcsv_out.writerow(demand_history)

    algo_data = {
        'name': {
            'agent':  'Standard Agent',
            'sa'   :  'Simulated Annealing',
            'psa'  :  'Parallelized Simulated Annealing',
            'pso'  :  'Particle Swarm Optimization',
            'abc'  :  'Artificial Bee Colony',
            'ga'   :  'Genetic Algorithm',
            'ges'  :  'Evolution Strategy',
            'cma'  :  'CMA-ES',
            'de'   :  'Differential Evolution'
        },
        'colours': {
            'agent':  'xkcd:orange',
            'sa'   :  'r',
            'psa'  :  'c',
            'pso'  :  'xkcd:purple',
            'abc'  :  'y',
            'ga'   :  'g',
            'ges'  :  'b',
            'cma'  :  'w',
            'de'   :  'k'
        }
    }
    # lists for statistics
    agent_reward_history          = np.zeros((steps_tot))
    agent_backlog_history         = np.zeros((steps_tot))
    agent_demand_backlog_history  = np.zeros((steps_tot))
    agent_orders_history          = np.zeros((n_echelons_, steps_tot))
    agent_warehouse_history       = np.zeros((n_echelons_, steps_tot))
    # lists for statistics on reward
    agent_reward_product          = np.zeros((steps_tot))
    agent_reward_raw_mat          = np.zeros((steps_tot))
    agent_reward_storage          = np.zeros((steps_tot))
    agent_reward_backlog          = np.zeros((steps_tot))

    # Basic agent action
    SC_model.SC_inventory[:,:] = start_inv             # starting inventory
    SC_model.time_k            = 0

    # initial order
    order_k = np.ones(n_echelons_)*(control_ub - control_lb)/2
    backlog = 0
    # main loop
    for step_k in range(steps_tot):
        d_k                                  = demand_history[step_k] + backlog
        sale_product, r_k, backlog           = SC_model.advance_supply_chain_orders(order_k, d_k)
        agent_reward_history[step_k]         = r_k
        agent_backlog_history[step_k]        = backlog
        agent_demand_backlog_history[step_k] = d_k
        # agent makes order
        order_k                              = agent_control_storage(15, SC_model.storage_tot, n_echelon=n_echelons_)
        # reward stats
        agent_reward_product[step_k]         = SC_model.r_product
        agent_reward_raw_mat[step_k]         = SC_model.r_raw_mat
        agent_reward_storage[step_k]         = SC_model.r_storage
        agent_reward_backlog[step_k]         = SC_model.r_bakclog
        agent_orders_history[:,step_k]       = order_k
        agent_warehouse_history[:,step_k]    = SC_model.warehouses
    
    
    # compute and store average reward
    print(np.average(agent_reward_history))

    # store action list
    with open(f'outputs/final_results/plots/action_plots/action_data/agent/reward_histoy.csv', mode='w') as rewardcsv:
        rewardcsv_out = csv.writer(rewardcsv)
        rewardcsv_out.writerow(agent_reward_history)

    with open(f'outputs/final_results/plots/action_plots/action_data/agent/backlog_histoy.csv', mode='w') as backlogcsv:
        backlogcsv_out = csv.writer(backlogcsv)
        backlogcsv_out.writerow(agent_backlog_history)
    
    with open(f'outputs/final_results/plots/action_plots/action_data/agent/demand_backlog_histoy.csv', mode='w') as dbcsv:
        dbcsv_out = csv.writer(dbcsv)
        dbcsv_out.writerow(agent_demand_backlog_history)
    
    for ii in range(n_echelons_):
        with open(f'outputs/final_results/plots/action_plots/action_data/agent/echelon_{ii}.csv', mode='w') as escsv:
            escsv_out = csv.writer(escsv)
            escsv_out.writerow(agent_warehouse_history[ii, :])
        
    # store rewards list
    with open(f'outputs/final_results/plots/reward_plots/agent/data/reward_product_histoy.csv', mode='w') as rpcsv:
        rpcsv_out = csv.writer(rpcsv)
        rpcsv_out.writerow(agent_reward_product)
    
    with open(f'outputs/final_results/plots/reward_plots/agent/data/raw_material_histoy.csv', mode='w') as rmcsv:
        rmcsv_out = csv.writer(rmcsv)
        rmcsv_out.writerow(agent_reward_raw_mat)

    with open(f'outputs/final_results/plots/reward_plots/agent/data/storage_histoy.csv', mode='w') as scsv:
        scsv_out = csv.writer(scsv)
        scsv_out.writerow(agent_reward_storage)

    with open(f'outputs/final_results/plots/reward_plots/agent/data/backlog_cost_histoy.csv', mode='w') as bcsv:
        bcsv_out = csv.writer(bcsv)
        bcsv_out.writerow(agent_reward_backlog)
    
    # store orders
    for ii in range(n_echelons_):
        with open(f'outputs/final_results/plots/action_plots/action_data/agent/orders_{ii}.csv', mode='w') as orcsv:
            orcsv_out = csv.writer(orcsv)
            orcsv_out.writerow(agent_orders_history[ii, :])

    # Algorithm parameters
    policy_net = Net(**SC_run_params_['hyparams'])

    for arg in args:
        # load best policy
        policy_net.load_state_dict(torch.load(f'neural_nets/parameters/test/{arg}.pth'))

        SC_model.SC_inventory[:,:] = start_inv             # starting inventory
        SC_model.time_k            = 0

        # initial order
        state_norm                     = (SC_model.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
        state_time                     = SC_model.supply_chain_state()[0,-1] / 365
        state_torch                    = torch.tensor(np.hstack((state_norm, state_time)))
        order_k                        = policy_net(state_torch)
        order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]

        backlog = 0
        # main loop
        for step_k in range(steps_tot):
            d_k                                 = demand_history[step_k] + backlog
            sale_product, r_k, backlog          = SC_model.advance_supply_chain_orders(order_k, d_k)
            reward_history[arg][step_k]         = r_k
            backlog_history[arg][step_k]        = backlog
            demand_backlog_history[arg][step_k] = d_k
            
            # agent makes order
            state_norm                     = (SC_model.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
            state_time                     = SC_model.supply_chain_state()[0,-1] / 365
            state_torch                    = torch.tensor(np.hstack((state_norm, state_time)))
            order_k                        = policy_net(state_torch)
            order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]
            
            # reward stats
            reward_product[arg][step_k]         = SC_model.r_product
            reward_raw_mat[arg][step_k]         = SC_model.r_raw_mat
            reward_storage[arg][step_k]         = SC_model.r_storage
            reward_backlog[arg][step_k]         = SC_model.r_bakclog
            orders_history[arg][:,step_k]       = order_k
            warehouse_history[arg][:,step_k]    = SC_model.warehouses
        
        
        # compute and store average reward
        avg_reward[arg] = np.average(reward_history[arg])

        for file in glob.glob(f'outputs/final_results/plots/reward_plots/{arg}/data/*'):
            os.remove(file)

        for file in glob.glob(f'outputs/final_results/plots/action_plots/action_data/{arg}/*'):
            os.remove(file)

        # store action list
        with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/reward_histoy.csv', mode='w') as rewardcsv:
            rewardcsv_out = csv.writer(rewardcsv)
            rewardcsv_out.writerow(reward_history[arg])

        with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/backlog_histoy.csv', mode='w') as backlogcsv:
            backlogcsv_out = csv.writer(backlogcsv)
            backlogcsv_out.writerow(backlog_history[arg])
        
        with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/demand_backlog_histoy.csv', mode='w') as dbcsv:
            dbcsv_out = csv.writer(dbcsv)
            dbcsv_out.writerow(demand_backlog_history[arg])
        
        for ii in range(n_echelons_):
            with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/echelon_{ii}.csv', mode='w') as escsv:
                escsv_out = csv.writer(escsv)
                escsv_out.writerow(warehouse_history[arg][ii, :])
            
        # store rewards list
        with open(f'outputs/final_results/plots/reward_plots/{arg}/data/reward_product_histoy.csv', mode='w') as rpcsv:
            rpcsv_out = csv.writer(rpcsv)
            rpcsv_out.writerow(reward_product[arg])
        
        with open(f'outputs/final_results/plots/reward_plots/{arg}/data/raw_material_histoy.csv', mode='w') as rmcsv:
            rmcsv_out = csv.writer(rmcsv)
            rmcsv_out.writerow(reward_raw_mat[arg])

        with open(f'outputs/final_results/plots/reward_plots/{arg}/data/storage_histoy.csv', mode='w') as scsv:
            scsv_out = csv.writer(scsv)
            scsv_out.writerow(reward_storage[arg])

        with open(f'outputs/final_results/plots/reward_plots/{arg}/data/backlog_cost_histoy.csv', mode='w') as bcsv:
            bcsv_out = csv.writer(bcsv)
            bcsv_out.writerow(reward_backlog[arg])
        
        # store orders
        for ii in range(n_echelons_):
            with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/orders_{ii}.csv', mode='w') as orcsv:
                orcsv_out = csv.writer(orcsv)
                orcsv_out.writerow(orders_history[arg][ii, :])

    print(avg_reward)
    #####-----PLOT RESULTS-----#####
    # reward history
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    
    for arg in args:
        ax.plot(reward_history[arg], c=f'{algo_data["colours"][arg]}', linestyle='-', label=f'{algo_data["name"][arg]}')
        #ax.hlines(y=avg_reward[arg], xmin=0, xmax=steps_tot, color=f'{algo_data["colours"][arg]}', linestyles='--')
        #ax.text(steps_tot, avg_reward[arg], f'{round(avg_reward[arg], 2)}', c=f'{algo_data["colours"][arg]}')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Reward (£)')
    ax.legend(loc="upper right")
    plt.xlim((0, 365))

    plt.savefig(f'outputs/final_results/plots/action_plots/reward_history.png')

    # demand history
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(demand_history, 'k.-')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Demand (# items)')
    plt.xlim((0, 365))
    #plt.yticks(np.arange(demand_lb, demand_ub, 1))

    plt.savefig(f'outputs/final_results/plots/action_plots/demand_history.png')    

    # backlog history
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    
    for arg in args:
        ax.plot(backlog_history[arg], c=f'{algo_data["colours"][arg]}', linestyle='-', label=f'{algo_data["name"][arg]}')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Backlog (# items)')
    ax.legend(loc="upper right")
    plt.xlim((0, 365))

    plt.savefig(f'outputs/final_results/plots/action_plots/backlog_history.png')    

    # demand and backlog history
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    
    for arg in args:
        ax.plot(demand_backlog_history[arg], c=f'{algo_data["colours"][arg]}', linestyle='-', label=f'{algo_data["name"][arg]}')
    
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Demand + backlog (# items)')
    ax.legend(loc="upper right")
    plt.xlim((0, 365))

    plt.savefig(f'outputs/final_results/plots/action_plots/demand_backlog_history.png')

    # plot warehouse for each echelon
    for ii in range(n_echelons_):
        # backlog history
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        
        storage_cap = SC_params_['echelon_storage_cap'][ii]

        for arg in args:
            ax.plot(warehouse_history[arg][ii, :], c=f'{algo_data["colours"][arg]}', label=f'{algo_data["name"][arg]}')
        ax.hlines(y=SC_params_['echelon_storage_cap'][ii], xmin=0, xmax=steps_tot, color='k', linestyles='--')
        ax.text(steps_tot, storage_cap, f'Max: {storage_cap}',c='k')
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Storage (# items)')
        ax.legend(loc="upper right")
        plt.xlim((0, 365))

        plt.savefig(f'outputs/final_results/plots/action_plots/warehouse_{ii+1}.png')
    
    # reward statistic
    for arg in args:
        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111)
        ax.plot(reward_product[arg], label='Reward product')
        ax.plot(reward_raw_mat[arg], label='Reward raw material')  
        ax.plot(reward_storage[arg], label='Reward storage')  
        ax.plot(reward_backlog[arg], label='Reward backlog')
        ax.set_xlabel('time (days)')  # Add an x-label to the axes.
        ax.set_ylabel('£')  # Add a y-label to the axes.  
        plt.xlim((0, 365))
        ax.legend()  # Add a legend.
        
        plt.savefig(f'outputs/final_results/plots/reward_plots/{arg}/plots/reward_stats.png')
    
    # orders history
    for ii in range(n_echelons_):
        # backlog history
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        for arg in args:
            ax.plot(orders_history[arg][ii, :], c=f'{algo_data["colours"][arg]}', label=f'{algo_data["name"][arg]}')
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Orders (# items)')
        ax.legend(loc="upper right")
        plt.xlim((0, 365))

        plt.savefig(f'outputs/final_results/plots/action_plots/orders_{ii+1}.png')

if __name__=="__main__":
    """
    keywords['io']       = 'siso' or 'mimo'
    keywords['echelons'] = int
    keywords['path']     = relative file pathway for parameters

    Algorithm options:
    - 'sa'          simulated annealing
    - 'psa'         parallelized simulated annealing
    - 'pso'         particle swarm optimization
    - 'abc'         artificial bee colony
    - 'ga'          genetic algorithm
    - 'ges'         gaussian evolutionary strategy
    - 'cma'         covariance matrix adaptation evolutionary strategy
    - 'de'          differential evolution
    """
    #keynames = ['sa', 'psa', 'pso', 'abc', 'ga', 'ges', 'de']
    #keynames = ['sa', 'psa', 'pso', 'abc']
    #keynames = ['ga', 'ges', 'de']
    #keynames = ['sa', 'psa']
    keynames = ['psa', 'pso', 'ges']
    #keynames = ['pso']
    
    keywords = {}
    keywords['io']       = 'siso'
    keywords['echelons'] = 5

    evaluate(keynames, **keywords)