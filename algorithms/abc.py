"""
Artificial bee colony for neural net optimization
"""
import copy
import torch
import numpy as np

from algorithms.optim import OptimClass
from helper_functions.timer import timeit

class Artificial_Bee_Colony(OptimClass):

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter

        - model                 =   class       # neural network
        - env                   =   class       # supply chain environment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['population']  =   50          # number of particles in swarm
        - kwargs['maxiter']     =   50          # maximum number of iterations
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

        # creating list
        self.bee_parameters = []
        self.bee_rewards    = []

        self.abandoned      = [0 for i in range(self.population)]   # store number of times solution was not changed

        # initialise global best
        self.best_parameters  = copy.deepcopy(self.params)
        self.best_reward      = -1e8
        self.reward_list      = []

    def _initialize(self, function, SC_run_params):
        """
        Initialize random starting positions
        """
        for i in range(self.population):
            for key, value in self.params.items():
            
                # initialize random parameters
                self.params[key] = torch.rand(value.shape) * (self.ub - self.lb) + self.lb
            self.bee_parameters.append(self.params)

            # caluclate parameter reward
            self.model.load_state_dict(self.params)
            total_reward = function(self.env, SC_run_params, self.model)
            self.bee_rewards.append(total_reward)
        
        # initialize best solution and value
        self._find_best()
    
    def _replace_bee_best(self, new_params, new_reward, i):
        """
        Determines the best solution for each bee
        """
        # determine best solution 
        if new_reward > self.bee_rewards[i]:
            self.bee_parameters[i] = new_params
            self.bee_rewards[i] = new_reward
        else:
            # if solution is worse, iter abandon food counter
            self.abandoned[i] += 1

    def _employed_bee(self, function, SC_run_params, i):
        """
        Employed bee looks for new parameters and calculates reward
        """
        random = {}; random_parameters = {}
        for key, value in self.bee_parameters[i].items():
            
            # randomly generate new solutions
            random_parameters[key] = torch.rand(value.shape) * (self.ub - self.lb) + self.lb
        
            # generate random weights
            random[key] = torch.rand(value.shape)
        
        # update parameters and store their reward
        new_params = {}
        for key, value in self.bee_parameters[i].items():
            
            # update functions
            param_diff = torch.sub(self.bee_parameters[i][key], random_parameters[key])
            new_params[key] = torch.add(self.bee_parameters[i][key], torch.mul(param_diff, random[key]))
        
            # check if bound constraints have been breached and replace
            new_params[key] = torch.clamp(new_params[key], min=self.lb, max=self.ub)

        # determine new reward
        self.model.load_state_dict(new_params)
        new_reward = function(self.env, SC_run_params, self.model)

        return new_params, new_reward

    def _onlooker_bee(self, function, SC_run_params, i):
        """
        Onlooker bee selects solutions via a roulette wheel selection system and 
        searches for better solutions in the neighbourhood
        """
        # onlooker bees generate a new solution using random neighbourhood search
        # generate new solution bounds
        new_bounds = [self.lb*0.1, self.ub*0.1]
        
        # generate new random parameters based on new bounds
        # generate random weights
        neighbour_params = {}; random = {}
        for key, value in self.params.items():
            neighbour_params[key] = torch.rand(value.shape) * (new_bounds[1] - new_bounds[0]) + new_bounds[0] + value
            random[key]           = torch.rand(value.shape)

        # update parameters and store their reward
        new_params = {}
        for key, value in self.bee_parameters[i].items():
            
            # update functions
            param_diff = torch.sub(self.bee_parameters[i][key], neighbour_params[key])
            new_params[key] = torch.add(self.bee_parameters[i][key], torch.mul(param_diff, random[key]))
        
            # check if bound constraints have been breached and replace
            new_params[key] = torch.clamp(new_params[key], min=self.lb, max=self.ub)

        # determine new reward
        self.model.load_state_dict(new_params)
        new_reward = function(self.env, SC_run_params, self.model)

        return new_params, new_reward
        
    def _scout_bee(self, function, SC_run_params, i):
        """
        Scout bee finds new solutions if current solution is considered 'abandoned'
        """
        # Scout bee will generate new solution in provided bounds
        scout_params = {}
        for key, value in self.bee_parameters[i].items():

            # scout hunts for new parameters
            scout_params[key] = torch.rand(value.shape) * (self.ub - self.lb) + self.lb

        # Replaces old solution with new solution found
        self.bee_parameters[i] = scout_params
        self.model.load_state_dict(self.bee_parameters[i])
        
        new_reward          = function(self.env, SC_run_params, self.model)
        self.bee_rewards[i] = new_reward

        # Restarts abandon counter
        self.abandoned[i] = 0

    def _find_best(self):
        """
        Finds best solution and stores it
        """
        # determines best solution
        maximum_reward = max(self.bee_rewards)

        # replaces best parameters found so far if better
        if maximum_reward > self.best_reward:
            self.best_reward     = maximum_reward
            self.best_parameters = copy.deepcopy(self.bee_parameters[self.bee_rewards.index(maximum_reward)])
    
    @timeit
    def algorithm(self, function: any, SC_run_params: dict, iter_debug: bool = False):
        """
        Artificial bee colony algorithm

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - iter_debug    =   if true, prints ever 100 iterations
        """
        # initialize algorithm
        self._initialize(function, SC_run_params)

        # store current best reward
        self.reward_list.append(self.best_reward)

        # start algorithm
        niter = 0
        while niter < self.maxiter:
            for i in range(self.population):
                
                # employed bee
                new_params, new_reward = self._employed_bee(function, SC_run_params, i)
                self._replace_bee_best(new_params, new_reward, i)

                # generate and store solution probability
                # determine minimum reward
                min_reward = min(self.bee_rewards)

                shifted_rewards = []
                if min_reward < 0:
                    # if minimum reward is negative, shift all rewards to be positive
                    for j in range(len(self.bee_rewards)):
                        bee_reward = self.bee_rewards[j] + min_reward * -1
                        shifted_rewards.append(bee_reward)
                else:
                    # otherwise do nothing
                    shifted_rewards = self.bee_rewards

                solution_prob = float(self.bee_rewards[i] / sum(shifted_rewards))

                # roulette wheel selection
                random_prob = np.random.uniform(0, 1/(self.population))
                if random_prob < solution_prob:

                    # onlooker bee
                    new_params, new_reward = self._onlooker_bee(function, SC_run_params, i)
                    self._replace_bee_best(new_params, new_reward, i)

                # scout bee
                if self.abandoned[i] > 5:
                    self._scout_bee(function, SC_run_params, i)

            # Determines best solution found
            self._find_best()

            # Stores best solution found in each iteration
            self.reward_list.append(self.best_reward)

            # Iteration counter
            niter += 1
            if niter % 10 == 0 and iter_debug == True:
                print(f'{niter}')

        return self.best_parameters, self.best_reward, self.reward_list
