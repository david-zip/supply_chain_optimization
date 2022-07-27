"""
Evolutionary strategy for neural net optimization
"""
import copy
import torch
import numpy as np

from functions.timer import timeit

class Gaussian_Evolutionary_Strategy():

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter

        - model                 =   class       # neural network
        - env                   =   class       # supply chain environment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['population']  =   100         # population size
        - kwargs['elite_cut']   =   0.4         # percent of population in elite set
        - kwargs['maxiter']     =   1000        # maximum number of iterations
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

        # store algorithm hyper-parameters
        self.population = self.args['population']
        self.elite_cut  = self.args['elite_cut']
        self.mean       = {}
        self.std        = {}

        # creating list
        self.parameters = []
        self.rewards    = []
        self.elite_set  = []

        # initialise global best
        self.best_parameters  = copy.deepcopy(self.params)
        self.best_reward      = -1e8
        self.reward_list      = []

    def _initialize(self, function, SC_run_params):
        """
        Initialize random starting positions
        """
        for i in range(self.population):

            # generate random solutions
            new_params = {}
            for key, value in self.params.items():
                new_params[key] = torch.rand(value.shape) * (self.ub - self.lb) + self.lb
            self.parameters.append(new_params)
            
            # calculate parameter reward
            self.model.load_state_dict(new_params)
            total_reward = function(self.env, SC_run_params, self.model)
            self.rewards.append(total_reward)

        # find best set of parameters
        self._find_best()
    
    def _elite_set(self):
        """
        Select individuals for the elite set
        """
        for i in range(int(self.population * self.elite_cut)):

            # add fitest to elite set
            current_fittest = max(self.rewards)

            self.elite_set.append(self.parameters[self.rewards.index(current_fittest)])
            self.rewards.remove(current_fittest)

    def _update_mean_std(self):
        """
        Update the mean and standard deviation using the elite set
        """
        for key in self.params.keys():
            
            # create a temporary list and store parameters here
            temp_mean_list = []
            temp_std_list  = []
            for i in range(len(self.elite_set)):
                temp_mean_list.append(torch.mean(self.elite_set[i][key]))
                temp_std_list.append(torch.std(self.elite_set[i][key]))

            temp_mean_list = torch.Tensor(temp_mean_list)
            temp_std_list = torch.Tensor(temp_std_list)
            
            # calculate mean and standard deviation
            self.mean[key] = torch.mean(temp_mean_list)
            self.std[key]  = torch.mean(temp_std_list)

    def _update_parametrs(self, function, SC_run_params):
        """
        Generates new parameters using the updated mean and standard deviation and deteremines
        the new reward 
        """
        self.parameters.clear()
        self.rewards.clear()
        for i in range(self.population):
            # generate random solutions from a Gaussian distribution
            new_params = {}
            for key, value in self.params.items():
                new_params[key] = torch.add(self.mean[key], torch.normal(mean=0, std=1., size=value.shape), alpha=self.std[key])

                # ensure bounds are not breached
                new_params[key] = torch.clamp(new_params[key], min=self.lb, max=self.ub)
            self.parameters.append(new_params)

            # calculate parameter reward
            self.model.load_state_dict(new_params)
            total_reward = function(self.env, SC_run_params, self.model)
            self.rewards.append(total_reward)

    def _find_best(self):
        """
        Finds best solution and stores it
        """
        # determines the best solution
        maximum_reward = max(self.rewards)

        # replaces best parameters found so far if better
        if maximum_reward > self.best_reward:
            self.best_reward     = maximum_reward
            self.best_parameters = copy.deepcopy(self.parameters[self.rewards.index(maximum_reward)])

    @timeit
    def algorithm(self, function: any, SC_run_params: dict, iter_debug: bool = False):
        """
        Gaussian evolutionary strategy algorithm (vanilla ES)

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - iter_debug    =   if true, prints ever 10 iterations
        """
        # initialize algorithm
        self._initialize(function, SC_run_params)

        # store current best reward
        self.reward_list.append(self.best_reward)

        # start algorithm
        niter = 0
        while niter < self.maxiter:
            self._elite_set()                                   # generate elite set
            self._update_mean_std()                             # update mean and std
            self._update_parametrs(function, SC_run_params)     # generate new parameters
            self._find_best()                                   # determine best parameters
            self.reward_list.append(self.best_reward)

            niter += 1

            if niter % 10 == 0 and iter_debug is True:
                print(f'{niter}')
        
        return self.best_parameters, self.best_reward, self.reward_list

class Covariance_Matrix_Adaption_Evolutionary_Strategy():
    
    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter 
        
        ***ADJUST LATER***

        - model                =   class       # neural network
        - env                  =   class       # supply chain environment
        - kwargs['bounds']     =   [lb, ub]    # parameter bounds
        - kwargs['population'] =   20          # population size
        - kwargs['mean']       =   0           # initial mean
        - kwargs['step_size']  =   3           # initial step_size
        - kwargs['elite_cut']  =   0.4         # percent of population in elite set
        - kwargs['maxiter']    =   1000        # maximum number of iterations
        """
        # unpack the arguments
        self.model      = model                     # neural network
        self.env        = env                       # environment 
        self.args       = kwargs

        # store model parameters
        self.params     = self.model.state_dict()   # inital parameter values

        # parameter bounds
        self.lb         = self.args['bounds'][0]
        self.ub         = self.args['bounds'][1]

        # maximum iterations/time
        self.maxiter    = self.args['maxiter']
        #self.maxtime    = self.args['maxtime']      # implement later

        # store algorithm hyper-parameters
        self.population   = self.args['population']
        self.elite_cut    = self.args['elite_cut']
        self.search_space = sum(p.numel() for p in self.model.parameters())
        self.mean         = {}
        self.std          = {}

        # creating list
        self.parameters = []
        self.rewards    = []
        self.elite_set  = []

        # initialise global best
        self.best_parameters  = copy.deepcopy(self.params)
        self.best_reward      = -1e8
        self.reward_list      = []

class Natural_Evolutionary_Strategy():
    
    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter 
        
        ***ADJUST LATER***

        - model                 =   class       # neural network
        - env                   =   class       # supply chain environment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['population']  =   100         # population size
        - kwargs['elite_cut']   =   0.4         # percent of population in elite set
        - kwargs['maxiter']     =   1000        # maximum number of iterations
        """
        # unpack the arguments
        self.model      = model                     # neural network
        self.env        = env                       # environment 
        self.args       = kwargs


if __name__=="__main__":
    pass