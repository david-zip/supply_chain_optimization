"""
Main file to run training functions and plot
"""
from helper_functions.demand import random_uniform_demand_si
from helper_functions.test_functions import test_reinforce, test_run


if __name__=="__main__":
    """
    Algorithm options:
    - 'sa'          simulated annealing
    - 'psa'         parallelized simulated annealing
    - 'pso'         particle swarm optimization
    - 'abc'         artificial bee colony
    - 'ga'          genetic algorithm
    - 'ges'         gaussian evolutionary strategy
    - 'cma'         covariance matrix adaptation evolutionary strategy (NOT IMPLEMENTED)
    - 'de'          differential evolution
    """
    keynames = ['psa']
    
    #test_run(keynames, random_uniform_demand_si)
    test_reinforce() # testing reinforce only (does not work)