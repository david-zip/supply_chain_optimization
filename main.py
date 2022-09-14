"""
Main file to run training functions and plot
"""
from helper_functions.demand import random_uniform_demand_si
from helper_functions.test_functions import test_run


if __name__=="__main__":
    keynames = ['psa']
    
    test_run(keynames, random_uniform_demand_si)