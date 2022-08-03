"""
Simulated annealing for neural net optimization
"""
import copy
import torch
import warnings
import multiprocessing
import numpy as np
from functools import partial

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

class Parallelized_Simulated_Annealing():

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameters

        - model                 =   class       # neural network
        - env                   =   class       # supply chain envionment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['population]   =   5           # population size
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
        self.params     = self.model[0].state_dict()   # inital parameter values

        # parameter bounds
        self.lb         = self.args['bounds'][0]
        self.ub         = self.args['bounds'][1]

        # maximum iterations/time
        self.maxiter    = self.args['maxiter']
        #self.maxtime    = self.args['maxtime']      # implement later

        # store algorithm hyperparameters
        self.population = self.args['population']
        self.Ti0        = self.args['temp'][0]
        self.Tf0        = self.args['temp'][1]
        self.T0         = [self.args['temp'][0] for _ in range(self.population)]
        self.eps        = 1 - (self.Tf0/self.Ti0)**(self.maxiter**(-1))

        # initialise list for algorithm
        self.parameters         = [self.params for _ in range(self.population)] 
        self.best_rewards       = [[] for _ in range(self.population)]
        self.current_reward     = [-1e8 for _ in range(self.population)]
        self.final_rewards      = []

        self.seeds = [np.random.randint(0, 2**32) for _ in range(self.population)]

    def _initialize(self, function, SC_run_params):
        """
        Initialise algorithm parameters
        """
        # store best solutions
        self.best_value         = [function[i](self.env[i], SC_run_params, self.model[i]) for i in range(self.population)]
        self.best_parameters    = [copy.deepcopy(self.params) for _ in range(self.population)]

        # initialize inital temperatures values
        self.Ti     = [self.Ti0 for _ in range(self.population)]
        self.Tf     = self.Tf0
        self.T      = [self.Ti0 for _ in range(self.population)]

    def _neighbourhood_search(self, i):
        """
        Random neighbourhood search (in parameter bounds)
        """
        # generate random solutions within bounds
        for key, value in self.parameters[i].items():
            self.parameters[i][key] = torch.randn(value.shape) * (self.ub - self.lb) + self.lb

    def _run_trajectory(self, function, SC_run_params, i):
        """
        Run J_supply_chain_ssa to collect data on trajectory
        """
        # implement new paramters
        self.model[i].load_state_dict(self.parameters[i])

        self.current_reward[i] = function(self.env[i], SC_run_params, self.model[i])

    def _find_best(self, i):
        """
        Determines best parameters and stores the parameters
        Uses Metropolis acceptance probability for high temperature situations
        """
        # identify best solution
        if self.current_reward[i] > self.best_value[i]:
            self.best_parameters[i]    = copy.deepcopy(self.parameters[i])
            self.best_value[i]         = self.current_reward[i]
        else:
            # metropolis acceptance probability
            r = np.random.uniform()
            if r > np.exp((self.best_value[i] - self.current_reward[i])/self.T[i]):
                self.best_parameters[i] = copy.deepcopy(self.parameters[i])
                self.best_value[i]      = self.current_reward[i]

    def _cooling_schedule(self, i):
        """
        Geometric cooling schedule
        """
        self.T[i] *= (1 - self.eps)
    
    def _run_parallel(self, i, function, SC_run_params, rew_list, param_list, iter_debug):
        """
        Run algorothm task in parallel; represents the task of each population
        """
        # suppress warnings
        warnings.filterwarnings('ignore')

        niter = 0; reward_list = []
        while self.T[i] > self.Tf:
            self._neighbourhood_search(i)
            self._run_trajectory(function[i], SC_run_params, i)
            self._find_best(i)
            self._cooling_schedule(i)

            # store values in a list    
            reward_list.append(self.best_value[i])

            # iteration counter
            niter += 1
            if niter % 100 == 0 and iter_debug == True:
                print(f'{niter}')

        return reward_list

    @timeit
    def algorithm(self, function: any, SC_run_params: dict, iter_debug: bool = False):
        """
        Parallelized simulated annealling algorithm

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - iter_debug    =   if true, prints ever 100 iterations
        """
        function = [function for _ in range(self.population)]

        # initialize solutions
        self._initialize(function, SC_run_params)
        
        # start algorithm for multiple population
        with multiprocessing.Manager() as manager:

            # share list between all workers
            self.best_value = manager.list(self.best_value)
            self.best_parameters = manager.list(self.best_parameters)
            
            # limit number of workers to 5 at a time
            pool = multiprocessing.Pool(5)
            best_rewards_list = list(pool.map(partial(self._run_parallel, \
                                    function=function, SC_run_params=SC_run_params, \
                                    rew_list=self.best_value, param_list=self.best_parameters, \
                                    iter_debug=iter_debug),
                                range(self.population)))

            # create reward list based on all the workers
            for i in range(self.maxiter):
                best = []
                for j in range(self.population):
                    best.append(best_rewards_list[j][i])
                self.final_rewards.append(max(best))

            # determine worker with best reward and parameters
            best_index = np.argmax(self.best_value)

            best_reward = max(self.best_value)
            best_params = self.best_parameters[best_index]

        return best_params, best_reward, self.final_rewards
    
    def time_algorithm(self, f: any, print_output: bool = True):
        """
        Simulated annealling algorithm

        ### IMPLEMENT LATER ###
        """
        pass