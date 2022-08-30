"""
Function to plot graphs from csv files
"""
import os
import csv

import matplotlib.pyplot as plt

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
            'sa' :  'b',
            'psa':  'g',
            'pso':  'r',
            'abc':  'c',
            'ga' :  'm',
            'ges':  'y',
            'cma':  'w',
            'de' :  'k'
        }
    }
    
    for arg in args:
        # open mean csv 
        with open(f'outputs/final_results/plot_data/{arg}/mean.csv', mode='r') as mean_data:
            mdata_reader     = list(csv.reader(mean_data, quoting=csv.QUOTE_NONNUMERIC))[0]
            meanls[f'{arg}'] = [float(mean) for mean in mdata_reader]
        
        # open std csv
        with open(f'outputs/final_results/plot_data/{arg}/std.csv', mode='r') as std_data:
            sdata_reader    = list(csv.reader(std_data, quoting=csv.QUOTE_NONNUMERIC))[0]
            stdls[f'{arg}'] = [float(std) for std in sdata_reader]
        
        for i in range(len(meanls[f'{arg}'])):
            low_err[arg].append(meanls[arg][i] - stdls[arg][i])
            high_err[arg].append(meanls[arg][i] + stdls[arg][i])

    # create plot
    fig = plt.figure()
    #plt.suptitle(f"Stochastic seach algorithms for {echelons}-echelon supply chain - {maxIter} Iterations")
    for arg in args:
        plt.plot(range(len(meanls[arg])), meanls[arg], f'{algo_data["colours"][arg]}-', label=f'{algo_data["name"][arg]}')
        plt.fill_between(range(len(meanls[arg])), high_err[arg], low_err[arg], alpha=0.3, edgecolor=f'{algo_data["colours"][arg]}', facecolor=f'{algo_data["colours"][arg]}')

    plt.xlabel('Function calls')
    plt.ylabel('Total reward')
    plt.yscale('log')
    plt.ylim((1e4, 1e7))

    plt.legend(loc="lower right")

    plt.savefig(f'outputs/final_results/plots/secondary_plot.png')

def boxplot_algorihtms(args):
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
            'sa' :  'b',
            'psa':  'g',
            'pso':  'r',
            'abc':  'c',
            'ga' :  'm',
            'ges':  'y',
            'cma':  'w',
            'de' :  'k'
        }
    }

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
    keynames = ['sa', 'psa']
    
    plot_algorithms(keynames)