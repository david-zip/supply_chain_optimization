"""
Particle swarm optimization for neural net optimization
"""
import copy
import torch
import numpy as np

from algorithms.optim import OptimClass
from helper_functions.timer import timeit

class Particle_Swarm_Optimization(OptimClass):

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter

        - model                 =   class       # neural network
        - env                   =   class       # supply chain environment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['weights'][0]  =   0.2         # personal best influence
        - kwargs['weights'][1]  =   0.2         # global best influence
        - kwargs['weights'][2]  =   1.0         # initial velocity weight
        - kwargs['lambda']      =   1.0         # weight decay exponent
        - kwargs['population']  =   50          # number of particles in swarm
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
        self.c1         = self.args['weights'][0]   # personal best influence
        self.c2         = self.args['weights'][1]   # global best influence
        self.w          = self.args['weights'][2]   # velocity weight
        self.w_0        = self.args['weights'][2]   # initial weight value
        self.lbda0      = self.args['lambda']       # weight decay exponent
        self.population = self.args['population']    

        # initalise particle characteristic list
        self.particle_parameters  = [copy.deepcopy(self.params) for _ in range(self.population)]
        self.particle_rewards     = []
        self.particle_velocities  = []

        # initialise particle personal best list
        self.pbest_parameters   = []
        self.pbest_rewards      = []

        # initialise global best
        self.gbest_parameters   = copy.deepcopy(self.params)
        self.gbest_reward       = -1e8
        self.gbest_reward_list  = []

        # initialize function call counter and reward list
        self.func_call          = 0
        self.func_call_reward   = []

    def _initialize(self, function, SC_run_params):
        """
        Initialize random starting positions and velocities
        """
        for i in range(self.population):

            velocity = {}
            for key, value in self.particle_parameters[i].items():
                
                # initialize random parameters and velocity
                self.particle_parameters[i][key] = torch.rand(value.shape) * (self.ub - self.lb) + self.lb
                velocity[key] = torch.rand(value.shape)
            
            # add initial elocity and parameters to list
            self.particle_velocities.append(velocity)

            # calculate parameter reward
            self.model.load_state_dict(self.particle_parameters[i])
            total_reward = function(self.env, SC_run_params, self.model)
            self.particle_rewards.append(total_reward)

        # initialize weight parameters
        self.w      = self.w        # velocity weight
        self.w0     = self.w_0      # initial weight value
        self.lbda   = self.lbda0    # weight decay exponent

        # initialise personal best parameters and reward
        for i in range(self.population):
            self.pbest_rewards.append(self.particle_rewards[i])
            self.pbest_parameters.append(copy.deepcopy(self.particle_parameters[i]))

        best_index = np.argmax(self.pbest_rewards)
        # initialize global best
        self.gbest_reward       = self.pbest_rewards[best_index]
        self.gbest_parameters   = copy.deepcopy(self.pbest_parameters[best_index])

    def _find_best(self, i):
        """
        Determine personal and global best
        """
        # determine personal best
        if self.particle_rewards[i] > self.pbest_rewards[i]:
            # replace previous personal best
            self.pbest_rewards[i]    = self.particle_rewards[i]
            self.pbest_parameters[i] = copy.deepcopy(self.particle_parameters[i])

        # determine global best of the current iteration
        if self.pbest_rewards[i] > self.gbest_reward:
            self.gbest_reward       = self.pbest_rewards[i]
            self.gbest_parameters   = copy.deepcopy(self.pbest_parameters[i])

    def _update(self, function, SC_run_params):
        """
        Update particle parameters and velocity
        """
        for i in range(self.population):
            # look at every dictionary of parameters
            # sort into self.params
            for key, value in self.particle_parameters[i].items():
                # look at the keys and values of self.params

                # generate random values in the range (-1,1) in an tensor 
                r1 = (1 - (-1)) * torch.rand(value.shape) + (-1)
                r2 = (1 - (-1)) * torch.rand(value.shape) + (-1)

                # influence part of update function
                pbest_pt = self.c1 * r1 * (self.pbest_parameters[i][key] - self.particle_parameters[i][key])
                gbest_pt = self.c2 * r2 * (self.gbest_parameters[key] - self.particle_parameters[i][key])
                
                self.particle_velocities[i][key]  = pbest_pt + gbest_pt + self.w * self.particle_velocities[i][key]
                self.particle_parameters[i][key]  = self.particle_parameters[i][key] + self.particle_velocities[i][key]

                # check if bound constraints have been breached and replace
                self.particle_parameters[i][key] = torch.clamp(self.particle_parameters[i][key], min=self.lb, max=self.ub)
                
            # Calculate particle fitness
            self.model.load_state_dict(self.particle_parameters[i])
            self.particle_rewards[i] = function(self.env, SC_run_params, self.model)
            
            # find best solution
            self._find_best(i)

            # step function call counter and store best reward
            self.func_call += 1
            self.func_call_reward.append(self.gbest_reward)

    def _weight_decay(self, niter):
        """
        Exponential weight decay
        """
        self.w = self.w0 * np.exp(-self.lbda*niter)

    @timeit
    def algorithm(self, function: any, SC_run_params: dict, iter_debug: bool = False):
        """
        Particle swarm optimization algorithm

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - iter_debug    =   if true, prints ever 100 iterations
        """
        # initialize solutions
        self._initialize(function, SC_run_params)

        # store current best reward
        self.gbest_reward_list.append(self.gbest_reward)

        # start algorithm
        niter = 0
        while niter < self.maxiter:
            self._weight_decay(niter)
            self._update(function, SC_run_params)

            # Store value in a list
            self.gbest_reward_list.append(self.gbest_reward)

            # Iteration counter
            niter += 1
            if niter % 100 == 0 and iter_debug == True:
                print(f'{niter}')
        
        return self.gbest_parameters, self.gbest_reward, self.gbest_reward_list

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
        # initialize solutions
        self._initialize(function, SC_run_params)

        # start algorithm
        while self.func_call < func_call_max:
            self._weight_decay(self.func_call)
            self._update(function, SC_run_params)

            # iter_debug
            if self.func_call % 1000 == 0 and iter_debug == True:
                print(f'{self.func_call}')
        
        return self.gbest_parameters, self.gbest_reward, self.func_call_reward
