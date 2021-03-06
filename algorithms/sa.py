"""
Simulated annealing for neural net optimization
"""
import copy
from tkinter.tix import Tree
import torch
import warnings
import numpy as np

from functions.timer import timeit

class Simulated_Annealing():

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameters

        - model                 =   class       # neural network
        - env                   =   class       # supply chain envionment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['temp'][0]     =   1.0         # algorithm initial temperature 
        - kwargs['temp'][1]     =   0.1         # algorithm final temperature 
        - kwargs['maxiter']     =   1000        # maximum number of iterations 
        - kwargs['maxtime']     =   100         # maximum run time in seconds (IMPLEMENT LATER)
        """
        # unpack the arguments
        self.model      = model                 # neural network
        self.env        = env                   # environment 
        self.args       = kwargs

        # store model parameters
        self.params     = self.model.state_dict()   # inital parameter values

        # parameter bounds
        self.lb         = self.args['bounds'][0]
        self.ub         = self.args['bounds'][1]

        # maximum iterations/time
        self.maxiter    = self.args['maxiter']
        #self.maxtime    = self.args['maxtime']      # implement later

        # store algorithm hyperparameters
        self.Ti0        = self.args['temp'][0]
        self.Tf0        = self.args['temp'][1]
        self.T0         = self.args['temp'][0]
        self.eps        = 1 - (self.Tf0/self.Ti0)**(self.maxiter**(-1))

        # initialise list for algorithm
        self.best_rewards       = []                # best reward after every episode
        self.current_reward     = -1e8

    def _initialize(self, function, SC_run_params):
        """
        Initialise algorithm parameters
        """
        # store best solutions
        self.best_value         = function(self.env, SC_run_params, self.model)
        self.best_parameters    = copy.deepcopy(self.params)

        # initialize inital temperatures values
        self.Ti     = self.Ti0
        self.Tf     = self.Tf0
        self.T      = self.Ti0

        return self.best_value, self.best_parameters

    def _neighbourhood_search(self):
        """
        Random neighbourhood search (in parameter bounds)
        """
        # generate random solutions within bounds
        for key, value in self.params.items():
            self.params[key] = torch.rand(value.shape) * (self.ub - self.lb) + self.lb

    def _run_trajectory(self, function, SC_run_params):
        """
        Run J_supply_chain_ssa to collect data on trajectory
        """
        # implement new paramters
        self.model.load_state_dict(self.params)

        self.current_reward = function(self.env, SC_run_params, self.model)

    def _find_best(self):
        """
        Determines best parameters and stores the parameters
        Uses Metropolis acceptance probability for high temperature situations
        """
        # identify best solution
        if self.current_reward > self.best_value:
            self.best_parameters = self.params
            self.best_value = self.current_reward
        else:
            # metropolis acceptance probability
            r = np.random.uniform()
            if r > np.exp((self.best_value - self.current_reward)/self.T):
                self.best_parameters = self.params
                self.best_value = self.current_reward

    def _cooling_schedule(self):
        """
        Geometric cooling schedule
        """
        self.T *= (1 - self.eps)

    @timeit
    def algorithm(self, function: any, SC_run_params: dict, iter_debug: bool = False):
        """
        Simulated annealling algorithm

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - iter_debug    =   if true, prints ever 100 iterations
        """
        # suppress warnings
        warnings.filterwarnings('ignore')

        # initialize solutions
        self._initialize(function, SC_run_params)

        # start algorithm
        niter = 0
        while self.T > self.Tf:
            self._neighbourhood_search()
            self._run_trajectory(function, SC_run_params)
            self._find_best()
            self._cooling_schedule()

            # store values in a list    
            self.best_rewards.append(self.best_value)

            # iteration counter
            niter += 1
            if niter % 100 == 0 and iter_debug == True:
                print(f'{niter}')

        return self.best_parameters, self.best_value, self.best_rewards
    
    def time_algorithm(self, f: any, print_output: bool = True):
        """
        Simulated annealling algorithm

        ### IMPLEMENT LATER ###
        """
        pass

if __name__=="__main__":
    pass