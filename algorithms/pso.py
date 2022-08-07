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
        self.particle_parameters  = []
        self.particle_rewards     = []
        self.particle_velocities  = []

        # initialise particle personal best list
        self.pbest_parameters   = []
        self.pbest_rewards      = []

        # initialise global best
        self.gbest_parameters   = copy.deepcopy(self.params)
        self.gbest_reward       = 0
        self.gbest_reward_list  = []

    def _initialize(self, function, SC_run_params):
        """
        Initialize random starting positions and velocities
        """
        for i in range(self.population):
    
            velocity = {}
            for key, value in self.params.items():
                
                # initialize random parameters
                self.params[key] = torch.rand(value.shape) * (self.ub - self.lb) + self.lb
                self.particle_parameters.append(self.params)

                # initialize random particle velocity
                velocity[key] = torch.randn(value.shape)
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
            self.pbest_parameters.append(copy.deepcopy(self.particle_parameters[i]))
            self.pbest_rewards.append(self.particle_rewards[i])

        # initialize global best
        self.gbest_reward       = max(self.particle_rewards)
        self.gbest_parameters   = copy.deepcopy(self.particle_parameters[self.particle_rewards.index(self.gbest_reward)])

    def _find_best(self):
        """
        Determine personal and global best
        """
        # determine personal best
        for i in range(self.population):
            if self.particle_rewards[i] > self.pbest_rewards[i]:
                # replace previous personal best
                self.pbest_parameters[i]    = copy.deepcopy(self.particle_parameters[i])
                self.pbest_rewards[i]       = self.particle_rewards[i]

        # determine global best of the current iteration
        max_reward = max(self.pbest_rewards)

        if max_reward > self.gbest_reward:
            self.gbest_reward       = max_reward
            self.gbest_parameters   = copy.deepcopy(self.particle_parameters[self.particle_rewards.index(self.gbest_reward)])

    def _update(self, function, SC_run_params):
        """
        Update particle parameters and velocity
        """
        for i in range(self.population):
            # look at every dictionary of parameters
            # sort into self.params

            r1 = {}; r2 = {}
            for key, value in self.particle_parameters[i].items():
                # look at the keys and values of self.params

                # generate random values in the range (-1,1) in an tensor 
                r1 = (-1 - 1) * torch.rand(value.shape) + 1
                r2 = (-1 - 1) * torch.rand(value.shape) + 1

                # influence part of update function
                pbest_pt = self.c1 * torch.mul(r1, (torch.sub(self.pbest_parameters[i][key], self.particle_parameters[i][key])))
                gbest_pt = self.c2 * torch.mul(r2, (torch.sub(self.gbest_parameters[key], self.particle_parameters[i][key])))
                
                self.particle_velocities[i][key]  = torch.add(torch.add(pbest_pt, gbest_pt), self.particle_velocities[i][key], alpha=self.w)
                self.particle_parameters[i][key]  = copy.deepcopy(torch.add(self.particle_parameters[i][key], self.particle_velocities[i][key]))

                # check if bound constraints have been breached and replace
                self.particle_parameters[i][key] = torch.clamp(self.particle_parameters[i][key], min=self.lb, max=self.ub)
                
            # Calculate particle fitness
            self.model.load_state_dict(self.particle_parameters[i])
            new_reward = function(self.env, SC_run_params, self.model)
            self.particle_rewards[i] = new_reward

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
            self._find_best()

            # Store value in a list
            self.gbest_reward_list.append(self.gbest_reward)

            # Iteration counter
            niter += 1
            if niter % 100 == 0 and iter_debug == True:
                print(f'{niter}')
        
        return self.gbest_parameters, self.gbest_reward, self.gbest_reward_list

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
        pass
