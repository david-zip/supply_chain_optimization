"""
Function to plot graphs from csv files
"""
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')

def plot_algorithms(args):
    # create list to store CSV data
    meanls = {
        'sa' :  [],
        'psa':  [],
        'pso':  [],
        'abc':  [],
        'ga' :  [],
        'ges':  [],
        'cma':  [],
        'de' :  [],
    }
    stdls = {
        'sa' :  [],
        'psa':  [],
        'pso':  [],
        'abc':  [],
        'ga' :  [],
        'ges':  [],
        'cma':  [],
        'de' :  [],
    }
    low_err = {
        'sa' :  [],
        'psa':  [],
        'pso':  [],
        'abc':  [],
        'ga' :  [],
        'ges':  [],
        'cma':  [],
        'de' :  [],
    }
    high_err = {
        'sa' :  [],
        'psa':  [],
        'pso':  [],
        'abc':  [],
        'ga' :  [],
        'ges':  [],
        'cma':  [],
        'de' :  [],
    }
    algo_data = {
        'name': {
            'sa' :  'Simulated Annealing',
            'psa':  'Parallelized Simulated Annealing',
            'pso':  'Particle Swarm Optimization',
            'abc':  'Artificial Bee Colony',
            'ga' :  'Genetic Algorithm',
            'ges':  'Evolution Strategy',
            'cma':  'CMA-ES',
            'de' :  'Differential Evolution'
        },
        'colours': {
            'sa' :  'r',
            'psa':  'c',
            'pso':  'xkcd:purple',
            'abc':  'y',
            'ga' :  'g',
            'ges':  'b',
            'cma':  'w',
            'de' :  'k'
        }
    }

    for arg in args:
        # open mean csv 
        with open(f'outputs/final_results/plot_data/{arg}/mean.csv', mode='r') as mean_data:
            mdata_reader = list(csv.reader(mean_data, quoting=csv.QUOTE_NONNUMERIC))[0]
            meanls[arg]  = [float(mean) for mean in mdata_reader]

        # open std csv
        with open(f'outputs/final_results/plot_data/{arg}/std.csv', mode='r') as std_data:
            sdata_reader = list(csv.reader(std_data, quoting=csv.QUOTE_NONNUMERIC))[0]
            stdls[arg]   = [float(std) for std in sdata_reader]

        for i in range(len(meanls[f'{arg}'])):
            low_err[arg].append(meanls[arg][i] - stdls[arg][i])
            high_err[arg].append(meanls[arg][i] + stdls[arg][i])

    # create plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    #plt.suptitle(f"Stochastic seach algorithms for {echelons}-echelon supply chain - {maxIter} Iterations")
    for arg in args:
        plt.plot(range(len(meanls[arg])), meanls[arg], f'{algo_data["colours"][arg]}', label=f'{algo_data["name"][arg]}')
        plt.fill_between(range(len(meanls[arg])), high_err[arg], low_err[arg], alpha=0.3, edgecolor=f'{algo_data["colours"][arg]}', facecolor=f'{algo_data["colours"][arg]}')

    plt.xlabel('Function calls')
    plt.xticks(np.arange(0, len(meanls[arg])+500, 500))
    plt.ylabel('Reward (£)')
    plt.yscale('log')
    #plt.ylim((1e4, 1e6))

    plt.legend(loc="lower right")

    plt.savefig(f'outputs/final_results/plots/secondary_plot.png')

def boxplot_algorihtms(args):
    results = {
        'sa' :  [],
        'psa':  [],
        'pso':  [],
        'abc':  [],
        'ga' :  [],
        'ges':  [],
        'cma':  [],
        'de' :  [],
    }
    algo_data = {
        'name': {
            'sa' :  'Simulated Annealing',
            'psa':  'Parallelized Simulated Annealing',
            'pso':  'Particle Swarm Optimization',
            'abc':  'Artificial Bee Colony',
            'ga' :  'Genetic Algorithm',
            'ges':  'Evolution Strategy',
            'cma':  'CMA-ES',
            'de' :  'Differential Evolution'
        },
        'colours': {
            'sa' :  'r',
            'psa':  'c',
            'pso':  'xkcd:purple',
            'abc':  'y',
            'ga' :  'g',
            'ges':  'b',
            'cma':  'w',
            'de' :  'k'
        }
    }

    for arg in args:
        # open mean csv 
        with open(f'outputs/final_results/plot_data/{arg}/final_solutions.csv', mode='r') as result_values:
            rdata_reader = list(csv.reader(result_values, quoting=csv.QUOTE_NONNUMERIC))[0]
            results[arg] = [float(value) for value in rdata_reader]

    # create plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    bplots = plt.boxplot([results[f'{arg}'] for arg in args], vert=True, patch_artist=True, labels=[f'{algo_data["name"][arg]}' for arg in args])

    # fill with colors
    colours = [algo_data['colours'][arg] for arg in args]
    for patch, median, colour in zip(bplots['boxes'], bplots['medians'], colours):
        patch.set_facecolor(colour)
        median.set_color('yellow')

    ax.set_ylabel('Reward (£)')
    ax.set_xlabel('Algorithms')
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    fig.tight_layout()

    plt.savefig(f'outputs/final_results/plots/box_plot.png')

def plot_evaluations(args, echelons):
    reward_history         = {arg: [] for arg in args}
    backlog_history        = {arg: [] for arg in args}
    demand_history         = []
    demand_backlog_history = {arg: [] for arg in args}
    warehouse_history      = {arg: [[] for _ in range(echelons)] for arg in args}

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

    SC_params_ = {'echelon_storage_cost':(5, 10), 'echelon_storage_cap' :(20, 15, 20, 10, 10),
                    'echelon_prod_cost' :(0,0), 'echelon_prod_wt' :((5,1),(7,1),(10,1),(4,1),(6,1)),
                    'material_cost':{0:20}, 'product_cost':{0:150}}
    
    # retrieve data from csv
    with open(f'outputs/final_results/plots/action_plots/demand_data/demand.csv', mode='r') as demandcsv:
        reader = list(csv.reader(demandcsv, quoting=csv.QUOTE_NONNUMERIC))[0]
        demand_history  = [float(value) for value in reader]

    for arg in args:
        with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/reward_histoy.csv', mode='r') as rewardcsv:
            reader = list(csv.reader(rewardcsv, quoting=csv.QUOTE_NONNUMERIC))[0]
            reward_history[arg]  = [float(value) for value in reader]

        with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/backlog_histoy.csv', mode='r') as backlogcsv:
            reader = list(csv.reader(backlogcsv, quoting=csv.QUOTE_NONNUMERIC))[0]
            backlog_history[arg]  = [float(value) for value in reader]
        
        with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/demand_backlog_histoy.csv', mode='r') as dbcsv:
            reader = list(csv.reader(dbcsv, quoting=csv.QUOTE_NONNUMERIC))[0]
            demand_backlog_history[arg]  = [float(value) for value in reader]
        
        for ii in range(echelons):
            with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/echelon_{ii}.csv', mode='r') as escsv:
                reader = list(csv.reader(escsv, quoting=csv.QUOTE_NONNUMERIC))[0]
                warehouse_history[arg][ii]  = [float(value) for value in reader]
        
        print(f'{arg}: {sum(reward_history[arg])}')

    # plot data
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
    plt.yticks(np.arange(min(demand_history), max(demand_history), 1))

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
    for ii in range(echelons):
        # backlog history
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        
        storage_cap = SC_params_['echelon_storage_cap'][ii]

        for arg in args:
            ax.plot(warehouse_history[arg][ii][:], c=f'{algo_data["colours"][arg]}', label=f'{algo_data["name"][arg]}')
        ax.hlines(y=SC_params_['echelon_storage_cap'][ii], xmin=0, xmax=len(warehouse_history[arg][ii][:]), color='k', linestyles='--')
        ax.text(len(warehouse_history[arg][ii][:]), storage_cap, f'Max: {storage_cap}',c='k')
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Storage (# items)')
        ax.legend(loc="upper right")
        plt.xlim((0, 365))

        plt.savefig(f'outputs/final_results/plots/action_plots/warehouse_{ii+1}.png')

def plot_rewards_stats(args):
    for arg in args:
        with open(f'outputs/final_results/plots/reward_plots/{arg}/data/reward_product_histoy.csv', mode='r') as rewardcsv:
            reader = list(csv.reader(rewardcsv, quoting=csv.QUOTE_NONNUMERIC))[0]
            reward_product  = [float(value) for value in reader]

        with open(f'outputs/final_results/plots/reward_plots/{arg}/data/backlog_cost_histoy.csv', mode='r') as backlogcsv:
            reader = list(csv.reader(backlogcsv, quoting=csv.QUOTE_NONNUMERIC))[0]
            reward_backlog  = [float(value) for value in reader]
        
        with open(f'outputs/final_results/plots/reward_plots/{arg}/data/storage_histoy.csv', mode='r') as scsv:
            reader = list(csv.reader(scsv, quoting=csv.QUOTE_NONNUMERIC))[0]
            reward_storage  = [float(value) for value in reader]
        
        with open(f'outputs/final_results/plots/reward_plots/{arg}/data/storage_histoy.csv', mode='r') as rmcsv:
            reader = list(csv.reader(rmcsv, quoting=csv.QUOTE_NONNUMERIC))[0]
            reward_raw_mat  = [float(value) for value in reader]

        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111)
        ax.plot(reward_product, label='Reward product')
        ax.plot(reward_raw_mat, label='Reward raw material')  
        ax.plot(reward_storage, label='Reward storage')  
        ax.plot(reward_backlog, label='Reward backlog')
        ax.set_xlabel('time (days)')  # Add an x-label to the axes.
        ax.set_ylabel('£')  # Add a y-label to the axes.  
        plt.xlim((0, 365))
        ax.legend()  # Add a legend.

        plt.savefig(f'outputs/final_results/plots/reward_plots/{arg}/plots/reward_stats.png')

def plot_orders(args, echelons):
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
    orders_history = {arg: [[] for _ in range(echelons)] for arg in args}

    for ii in range(echelons):
        for arg in args:
            with open(f'outputs/final_results/plots/action_plots/action_data/{arg}/orders_{ii}.csv', mode='r') as acsv:
                reader = list(csv.reader(acsv, quoting=csv.QUOTE_NONNUMERIC))[0]
                orders_history[arg][ii]  = [float(value) for value in reader]
    
    for arg in args:
        for ii in range(echelons):
            # backlog history
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(111)

            ax.plot(orders_history[arg][ii][:], c=f'{algo_data["colours"][arg]}')
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Orders (# items)')
            plt.xlim((0, 365))

            plt.savefig(f'outputs/final_results/plots/action_plots/orders_plot/{arg}/orders_{ii+1}.png')
