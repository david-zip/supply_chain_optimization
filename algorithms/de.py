"""
Differential evolution for neural net optimization
"""
import copy
import torch
import numpy as np

from algorithms.optim import OptimClass
from helper_functions.timer import timeit

class Differential_Evolution(OptimClass):

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter

        - model                 =   class       # neural network
        - env                   =   class       # supply chain environment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['population']  =   100         # population size
        - kwargs['scale']       =   0.5         # scale factor
        - kwargs['mutation']    =   0.7         # mutation rate
        - kwargs['maxiter']     =   1000        # maximum number of iterations
        """
        # unpack the arguments
        self.model      = model                     # neural network
        self.env        = env                       # environment 
        self.args       = kwargs

        # store model parameters
        self.params     = self.model.state_dict()   # inital parameter values

        # maximum iterations/time
        self.maxiter    = self.args['maxiter']

        # parameter bounds
        self.lb         = self.args['bounds'][0]
        self.ub         = self.args['bounds'][1]

        # store algorithm hyper-parameters
        self.scale         = self.args['scale']
        self.population    = self.args['population']
        self.mutation_rate = self.args['mutation']
        self.num_params    = sum(p.numel() for p in self.model.parameters())

        # create required list
        self.solutions  = []
        self.parameters = [copy.deepcopy(self.params) for _ in range(self.population)]
        self.fitness    = []

        # initialize best solution variables
        self.best_parameters = copy.deepcopy(self.params)
        self.best_fitness    = -1e8
        self.fitness_list     = []

        # initialize function call counter and reward list
        self.func_call          = 0
        self.func_call_reward   = []

    def _initialize(self, function, SC_run_params):
        """
        Initialize aglorithm by generate a set of random solutions
        """
        # generate a list of random parameters
        for i in range(self.population):
            solution = torch.rand(self.num_params) * (self.ub - self.lb) + self.lb
            self.solutions.append(solution)

            # replace current parameters with newly found ones
            index = 0
            for key in self.parameters[i].keys():               # loop through keys of parameters
                for tensor in self.parameters[i][key]:          # loop through tensors in parameter dictionary
                    tensor = tensor.unsqueeze(-1)
                    for j in range(len(tensor)):                # iterate through every parameter
                        tensor[j] = torch.tensor(solution[index], 
                                                    dtype=torch.float)
                        index += 1

            # calculate gene fitness and store in list
            self.model.load_state_dict(self.parameters[i])
            fitness = function(self.env, SC_run_params, self.model)
            self.fitness.append(fitness)
        
            # initialize best gene
            self._find_best(i)

    def _find_best(self, i):
        """
        Finds best solution and stores it
        """
        # replaces best parameters found so far if better
        if self.fitness[i] > self.best_fitness:
            self.best_fitness    = self.fitness[i]
            self.best_parameters = copy.deepcopy(self.parameters[i])

    def _mutation(self, a, b, c):
        """
        Mutation operator; adds the difference between two solutions to perturb current solution
        """
        mutated = a + self.scale * (b - c)
        return mutated
    
    def _crossover(self, target, mutated):
        """
        Crossover operator; some parameters are replaced 
        """
        prob = torch.rand(1, self.num_params)

        # replace varaibles in target parameters with mutated parameters
        trial = torch.where(prob < self.mutation_rate, mutated, target)

        return trial.tolist()
    
    def _update_parameters(self, trial, function, SC_run_params, i):
        """
        Update solution parameters and find fitness
        """
        # replace current parameters with newly found ones
        index = 0
        for key in self.parameters[i].keys():               # loop through keys of parameters
            for tensor in self.parameters[i][key]:          # loop through tensors in parameter dictionary
                tensor = tensor.unsqueeze(-1)
                for j in range(len(tensor)):                # iterate through every parameter
                    tensor[j] = torch.tensor(trial[0][index],
                                                dtype=torch.float)
                    index += 1
            self.parameters[i][key] = torch.clamp(self.parameters[i][key], min=self.lb, max=self.ub)

        # update fitness solution
        self.model.load_state_dict(self.parameters[i])
        self.fitness[i] = function(self.env, SC_run_params, self.model)

    @timeit
    def algorithm(self, function: any, SC_run_params: dict, iter_debug: bool = False):
        """
        Differential evolution 

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - iter_debug    =   if true, prints ever 10 iterations
        """
        # initialize algorithm
        self._initialize(function, SC_run_params)

        # store current best reward
        self.fitness_list.append(self.best_fitness)

        # start algorithm
        niter = 0
        while niter < self.maxiter:
            for i in range(self.population):

                # choose three candidates for mutation except current 
                candidates = [int(candidate) for candidate in range(self.population) if candidate != i]
                indexes    = np.random.choice(candidates, 3, replace=False)
                
                # choose three tensors
                a = self.solutions[indexes[0]]
                b = self.solutions[indexes[1]]
                c = self.solutions[indexes[2]]

                # mutation operator
                mutated = self._mutation(a, b, c)

                # crossover operator
                trial = self._crossover(self.solutions[i], mutated)

                # update parameters
                self._update_parameters(trial, function, SC_run_params, i)

                # find fittest solution
                self._find_best(i)
        
                self.fitness_list.append(self.best_fitness)           

                niter += 1
                if niter % 100 == 0 and iter_debug is True:
                    print(f'{niter}')
        
        return self.best_parameters, self.best_fitness, self.fitness_list

    def reinitialize(self):
        """
        Reinitialize class to original state
        """
        self.__init__(self.model, self.env, **self.args)

    @timeit
    def func_algorithm(self, function: any, SC_run_params: dict, func_call_max: int = 10000, 
                        iter_debug: bool = False):
        """
        Simulated annealling algorithm
        Will terminate after a given number of maximum function calls
        
        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - func_call_max =   maximum number of function calls (default; 10000)
        - iter_debug    =   if true, prints ever 1000 function calls
        """
        # initialize algorithm
        self._initialize(function, SC_run_params)

        while self.func_call < func_call_max:
            for i in range(self.population):

                # choose three candidates for mutation except current 
                candidates = [int(candidate) for candidate in range(self.population) if candidate != i]
                indexes    = np.random.choice(candidates, 3, replace=False)
                
                # choose three tensors
                a = self.solutions[indexes[0]]
                b = self.solutions[indexes[1]]
                c = self.solutions[indexes[2]]

                # mutation operator
                mutated = self._mutation(a, b, c)

                # crossover operator
                trial = self._crossover(self.solutions[i], mutated)

                # update parameters
                self._update_parameters(trial, function, SC_run_params, i)

                # find fittest solution
                self._find_best(i)

                # iterate function call counter and store best solution
                self.func_call += 1
                self.func_call_reward.append(self.best_fitness)

                if self.func_call % 1000 == 0 and iter_debug is True:
                    print(f'{self.func_call}')
        
        return self.best_parameters, self.best_fitness, self.func_call_reward
