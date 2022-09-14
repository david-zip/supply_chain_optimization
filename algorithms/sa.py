"""
Simulated annealing for neural net optimization
"""
import copy
import torch
import warnings
import multiprocessing
import numpy as np
from functools import partial

from algorithms.optim import OptimClass
from helper_functions.timer import timeit
from helper_functions.trajectory import J_supply_chain_ssa

class Simulated_Annealing(OptimClass):

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameters

        - model                 =   class       # neural network
        - env                   =   class       # supply chain envionment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['temp'][0]     =   1.0         # algorithm initial temperature 
        - kwargs['temp'][1]     =   0.1         # algorithm final temperature 
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

        # store algorithm hyperparameters
        self.Ti0        = self.args['temp'][0]
        self.Tf0        = self.args['temp'][1]
        self.T0         = self.args['temp'][0]
        self.eps        = 1 - (self.Tf0/self.Ti0)**(self.maxiter**(-1))

        # initialise list for algorithm
        self.best_rewards       = []                # best reward after every episode
        self.current_reward     = -1e8
        
        # set max function call for benchmarking
        self.func_call = 0
        self.func_call_reward = []

        # set max time for benchmarking
        self.time_counter = 0
        self.time_best_value = {}

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
            self.best_parameters = copy.deepcopy(self.params)
            self.best_value = self.current_reward
        else:
            # metropolis acceptance probability
            r = np.random.uniform()
            if r > np.exp((self.best_value - self.current_reward)/self.T):
                self.best_parameters = copy.deepcopy(self.params)
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
        Will terminate after a given number of function calls
        
        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - func_call_max =   maximum number of function calls (default: 10000)
        - iter_debug    =   if true, prints ever 1000 function calls
        """
        # suppress warnings
        warnings.filterwarnings('ignore')

        # reinitialize exponential decay
        self.eps = 1 - (self.Tf0/self.Ti0)**(func_call_max**(-1))

        # initialize solutions
        self._initialize(function, SC_run_params)

        # start algorithm
        while self.func_call <= func_call_max:
            self._neighbourhood_search()
            self._run_trajectory(function, SC_run_params)
            self._find_best()
            self._cooling_schedule()

            # store values in a list    
            self.func_call_reward.append(self.best_value)

            # iteration counter
            self.func_call += 1
            if self.func_call % 1000 == 0 and iter_debug == True:
                print(f'{self.func_call}')

        return self.best_parameters, self.best_value, self.func_call_reward

class Parallelized_Simulated_Annealing(OptimClass):

    def __init__(self, model, env, echelons, SC_params, hyparams, **kwargs):
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
        self.model = []; self.env = []
        for _ in range(kwargs['population']):
            self.model.append(model(**hyparams))
            self.env.append(env(echelons, SC_params))
        self.args       = kwargs
        self.inputs     = [model, env, echelons, SC_params, hyparams, kwargs]

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
        self.func_call          = 0
        self.func_call_reward   = []

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
        functions = [self.env[i].J_supply_chain for i in range(self.population)]

        # initialize solutions
        self._initialize(functions, SC_run_params)
        
        # start algorithm for multiple population
        with multiprocessing.Manager() as manager:

            # share list between all workers
            self.best_value = manager.list(self.best_value)
            self.best_parameters = manager.list(self.best_parameters)
            self.current_reward = manager.list(self.current_reward)
            
            # limit number of workers to 5 at a time
            pool = multiprocessing.Pool(5)
            best_rewards_list = list(pool.map(partial(self._run_parallel, \
                                    function=functions, SC_run_params=SC_run_params, \
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

    def _run_parallel_func(self, i, function, SC_run_params, func_call, func_call_max, 
                            rew_list, param_list, iter_debug):
        """
        Run algorothm task in parallel; represents the task of each population
        """
        # suppress warnings
        warnings.filterwarnings('ignore')

        reward_list = []
        while sum(self.func_call) < func_call_max:
            self._neighbourhood_search(i)
            self._run_trajectory(function[i], SC_run_params, i)
            self._find_best(i)
            self._cooling_schedule(i)

            # store values in a list    
            reward_list.append(self.best_value[i])

            self.func_call_reward.append(max(self.best_value))

            # iteration counter
            self.func_call[i] += 1
            if sum(self.func_call) % 1000 == 0 and iter_debug == True:
                print(f'{self.func_call}')

        return self.func_call_reward

    def reinitialize(self):
        """
        Reinitialize class to original state
        """
        self.__init__(self.inputs[0], self.inputs[1], self.inputs[2], 
                        self.inputs[3], self.inputs[4], **self.inputs[5])

    @timeit
    def func_algorithm(self, function: any, SC_run_params: dict, func_call_max: int = 10000, 
                        iter_debug: bool = False):
        """
        Simulated annealling algorithm
        Will terminate after a given number of function calls

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - func_call_max =   maximum number of function calls (default: 10000)
        - iter_debug    =   if true, prints ever 1000 function calls
        """
        functions = [self.env[i].J_supply_chain for i in range(self.population)]

        # reinitialize exponential decay
        self.eps = 1 - (self.Tf0/self.Ti0)**(func_call_max**(-1))

        # initialize solutions
        self._initialize(functions, SC_run_params)

        # start algorithm for multiple population
        with multiprocessing.Manager() as manager:

            # share list between all workers
            self.best_value         = manager.list(self.best_value)
            self.best_parameters    = manager.list(self.best_parameters)
            self.current_reward     = manager.list(self.current_reward)
            self.func_call_reward   = manager.list(self.func_call_reward)
            self.func_call          = manager.list([0 for _ in range(self.population)])

            # limit number of workers to 5 at a time
            pool = multiprocessing.Pool(5)
            _ = list(pool.map(partial(self._run_parallel_func,
                                        function=functions, SC_run_params=SC_run_params,
                                        func_call=self.func_call, func_call_max=func_call_max,
                                        rew_list=self.best_value, param_list=self.best_parameters, 
                                        iter_debug=iter_debug),
                        range(self.population)))
            pool.terminate()

            R_list = []
            for i in range(len(self.func_call_reward)):
                R_list.append(self.func_call_reward[i])
            
            # determine worker with best reward and parameters
            best_index = np.argmax(self.best_value)

            best_reward = max(self.best_value)
            best_params = self.best_parameters[best_index]

            return best_params, best_reward, R_list

class Parallelized_Simulated_Annealing_with_Differential_Evolution(OptimClass):

    def __init__(self, model, env, echelons, SC_params, hyparams, **kwargs):
        """
        Initialize algorithm hyper-parameters

        - model                 =   class       # neural network
        - env                   =   class       # supply chain envionment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['population]   =   5           # population size
        - kwargs['temp'][0]     =   1.0         # algorithm initial temperature 
        - kwargs['temp'][1]     =   0.1         # algorithm final temperature 
        - kwargs['scale']       =   0.5         # scale factor
        - kwargs['mutation']    =   0.7         # mutation rate
        - kwargs['maxiter']     =   1000        # maximum number of iterations 
        - kwargs['maxtime']     =   100         # maximum run time in seconds (IMPLEMENT LATER)
        """
        # unpack the arguments
        self.model = []; self.env = []
        for _ in range(kwargs['population']):
            self.model.append(model(**hyparams))
            self.env.append(env(echelons, SC_params))

        self.args       = kwargs

        self.inputs     = [model, env, echelons, SC_params, hyparams, kwargs]

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
        self.scale         = self.args['scale']
        self.mutation_rate = self.args['mutation']
        self.num_params    = sum(p.numel() for p in self.model[0].parameters())

        # initialise list for algorithm
        self.parameters         = [self.params for _ in range(self.population)] 
        self.best_rewards       = [[] for _ in range(self.population)]
        self.current_reward     = [-1e8 for _ in range(self.population)]
        self.final_rewards      = []
        self.func_call          = 0
        self.func_call_reward   = []

        self.seeds = [np.random.randint(0, 2**32) for _ in range(self.population)]

        # differential evolution list
        self.trial   = [self.params for _ in range(self.population)]
        self.mutated = [self.params for _ in range(self.population)]
        self.a       = [[self.params, self.params, self.params] for _ in range(self.population)]

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
        for key in self.parameters[i].keys():
            self.parameters[i][key] = torch.clamp(self.parameters[i][key], min=self.lb, max=self.ub)

    def _mutation(self, i):
        """
        Mutation operator; adds the difference between two solutions to perturb current solution
        """
        for key in self.a[i][0].keys():
            self.mutated[i][key] = self.a[i][0][key] + self.scale * (self.a[i][1][key] - self.a[i][2][key])
    
    def _crossover(self, i):
        """
        Crossover operator; some parameters are replaced 
        """
        for key, value in self.mutated[i].items():
            prob = torch.rand(value.shape)

            # replace varaibles in target parameters with mutated parameters
            self.trial[i][key] = torch.where(prob < self.mutation_rate, self.mutated[i][key], self.parameters[i][key])

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
            # choose three candidates for mutation except current 
            candidates = [int(candidate) for candidate in range(self.population) if candidate != i]
            indexes    = np.random.choice(candidates, 3, replace=False)
            
            # choose three tensors
            self.a[i][0] = self.parameters[indexes[0]]
            self.a[i][1] = self.parameters[indexes[1]]
            self.a[i][2] = self.parameters[indexes[2]]

            # mutation operator
            self._mutation(i)

            # crossover operator
            self._crossover(i)

            # update parameters
            self._neighbourhood_search(i)
            self._run_trajectory(function[i], SC_run_params, i)
            
            # find fittest solution
            self._find_best(i)
    
            # cooling schedule
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
        function = [c for _ in range(self.population)]

        # initialize solutions
        self._initialize(function, SC_run_params)
        
        # start algorithm for multiple population
        with multiprocessing.Manager() as manager:

            # share list between all workers
            self.best_value = manager.list(self.best_value)
            self.best_parameters = manager.list(self.best_parameters)
            self.trial = manager.list(self.trial)
            self.mutated = manager.list(self.mutated)
            self.a = manager.list(self.a)
            
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

    def _run_parallel_func(self, i, function, SC_run_params, func_call, func_call_max, 
                            rew_list, param_list, iter_debug):
        """
        Run algorothm task in parallel; represents the task of each population
        """
        # suppress warnings
        warnings.filterwarnings('ignore')

        reward_list = []
        while sum(self.func_call) < func_call_max:
            self._neighbourhood_search(i)
            self._run_trajectory(function[i], SC_run_params, i)
            self._find_best(i)
            self._cooling_schedule(i)

            # store values in a list    
            reward_list.append(self.best_value[i])

            self.func_call_reward.append(max(self.best_value))

            # iteration counter
            self.func_call[i] += 1
            if sum(self.func_call) % 1000 == 0 and iter_debug == True:
                print(f'{self.func_call}')

        return self.func_call_reward

    def reinitialize(self):
        """
        Reinitialize class to original state
        """
        self.__init__(self.inputs[0], self.inputs[1], self.inputs[2], 
                        self.inputs[3], self.inputs[4], **self.inputs[5])

    @timeit
    def func_algorithm(self, function: any, SC_run_params: dict, func_call_max: int = 10000, 
                        iter_debug: bool = False):
        """
        Simulated annealling algorithm
        Will terminate after a given number of function calls

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - func_call_max =   maximum number of function calls (default: 10000)
        - iter_debug    =   if true, prints ever 1000 function calls
        """
        function = [function for _ in range(self.population)]

        # reinitialize exponential decay
        self.eps = 1 - (self.Tf0/self.Ti0)**(func_call_max**(-1))

        # initialize solutions
        self._initialize(function, SC_run_params)

        # start algorithm for multiple population
        with multiprocessing.Manager() as manager:

            # share list between all workers
            self.best_value         = manager.list(self.best_value)
            self.best_parameters    = manager.list(self.best_parameters)
            self.func_call_reward   = manager.list(self.func_call_reward)
            self.func_call          = manager.list([0 for _ in range(self.population)])

            # limit number of workers to 5 at a time
            pool = multiprocessing.Pool(5)
            best_rewards_list = list(pool.map(partial(self._run_parallel_func,
                                                function=function, SC_run_params=SC_run_params,
                                                func_call=self.func_call, func_call_max=func_call_max,
                                                rew_list=self.best_value, param_list=self.best_parameters, 
                                                iter_debug=iter_debug),
                                        range(self.population)))
            pool.terminate()

            R_list = []
            for i in range(len(self.func_call_reward)):
                R_list.append(self.func_call_reward[i])
            
            # determine worker with best reward and parameters
            best_index = np.argmax(self.best_value)

            best_reward = max(self.best_value)
            best_params = self.best_parameters[best_index]

            return best_params, best_reward, R_list
