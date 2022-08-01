"""
Evolutionary strategy for neural net optimization
"""
import copy
import torch
import warnings
import numpy as np

from functions.timer import timeit

warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
        # sort parameters in order of best rewards
        order = sorted(range(len(self.rewards)), key=lambda i: self.rewards[i], reverse=True)
        self.rewards    = [self.rewards[i] for i in order]
        self.parameters = [self.parameters[i] for i in order]
        
        # added best solutiouns to elite set
        self.elite_set = self.parameters[:int(self.population*self.elite_cut)]

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
            
            # clear elite set for next iteration
            self.elite_set.clear()              

            # store current best into list
            self.reward_list.append(self.best_reward)

            niter += 1
            if niter % 10 == 0 and iter_debug is True:
                print(f'{niter}')
        
        return self.best_parameters, self.best_reward, self.reward_list

class Covariance_Matrix_Adaption_Evolutionary_Strategy():
    
    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter 

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

        # determine search space
        self.search_space   = sum(param.numel() for param in self.model.parameters())

        # store algorithm hyper-parameters
        self.population     = int(4 + np.floor(3 * np.log(self.search_space)))
        self.elite_cut      = self.args['elite_cut']
        self.mean           = torch.full(size=[self.search_space, 1], fill_value=self.args['mean'], dtype=torch.float)
        self.mean_old       = torch.full(size=[self.search_space, 1], fill_value=self.args['mean'], dtype=torch.float)
        self.step_size      = self.args['step_size']
        self.weights        = 1. / (int(self.population * self.elite_cut))
        self.mu_weights     = 1. / sum([self.weights**2 for i in range(int(self.population * self.elite_cut))])
        # initialize learning rates (# are these parameters too small? #)
        self.lr_mean    = self.mu_weights / self.search_space**2    
        self.lr_ss      = 4. / self.search_space
        self.lr_cmep    = 4. / self.search_space    # the same as above?
        self.lr_cmup    = min(1., ((self.population * self.elite_cut) / self.search_space**2))
        self.lr_one     = 2. / self.search_space**2
        self.damping    = 1. + np.sqrt(self.mu_weights / self.search_space)

        # create covariance matrix and evolution path matrices
        #self.D              = torch.ones(self.search_space, 1, dtype=torch.float)
        #self.B              = torch.eye(self.search_space, self.search_space)
        self.C              = torch.eye(self.search_space)
        #self.invsqrtC       = self.B * torch.diag(self.D.pow(-1)) * self.B.t()
        self.evolution_step = torch.zeros(self.search_space, 1)
        self.evolution_cov  = torch.zeros(self.search_space, 1)

        # creating list
        self.parameters    = []
        self.samples       = []
        self.rewards       = []
        self.elite_set     = []
        self.elite_samples = []

        # initialise global best
        self.best_parameters  = copy.deepcopy(self.params)
        self.best_reward      = -1e8
        self.reward_list      = []

    def _initialize(self, function, SC_run_params):
        """
        Initialize random starting parameters using Normal distribution
        """
        for i in range(self.population):

            # generate random solutions
            new_params = {}
            for key, value in self.params.items():
                new_params[key] = torch.randn(value.shape) * (self.ub - self.lb) + self.lb
            self.parameters.append(new_params)
            self.samples.append(torch.randn(self.search_space, 1))

            # calculate parameter reward
            self.model.load_state_dict(new_params)
            total_reward = function(self.env, SC_run_params, self.model)
            self.rewards.append(total_reward)

        # determine best solution
        self._find_best()

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

    def _elite_set(self):
        """
        Select individuals for the elite set
        """
        # sort parameters and samples in order of best rewards
        order = sorted(range(len(self.rewards)), key=lambda i: self.rewards[i], reverse=True)
        self.rewards    = [self.rewards[i] for i in order]
        self.parameters = [self.parameters[i] for i in order]
        self.samples    = [self.samples[i] for i in order]

        # added best solutiouns to elite set
        self.elite_set     = self.parameters[:int(self.population*self.elite_cut)]
        self.elite_samples = self.samples[:int(self.population*self.elite_cut)]

        print(self.rewards)

    def _sample_parameters(self, function, SC_run_params):
        """
        Sample new parameters using an updated covariance matrix
        """
        for i in range(self.population):

            # generate random solutions                
            # sample using covariance matrix and estimate solutions
            self.samples[i] = self.mean + self.step_size * \
                                self.B * (torch.diag(self.D) * torch.randn(self.search_space, 1))

            # enusure bounds are not breached                                    
            self.samples[i] = torch.clamp(self.samples[i], min=self.lb, max=self.ub)
            solutions       = self.samples[i].unsqueeze(-1).tolist()
            
            index = 0
            for key, value in self.parameters[i].items():                   # loop through keys of parameters            
                for tensor in value:                                        # loop through tensors in parameter dictionary
                    tensor = tensor.unsqueeze(-1)
                    for j in range(len(tensor)):                            # iterate through every parameter
                        tensor[j] = torch.tensor(solutions[index][0][0], 
                                                    dtype=torch.float)

                        index += 1

            # calculate parameter reward
            self.model.load_state_dict(self.parameters[i])
            total_reward    = function(self.env, SC_run_params, self.model)
            self.rewards[i] = total_reward

    def _update_mean(self):
        """
        Update algorithm mean
        """
        # initialize mean update list
        mean_update = self.weights * torch.stack(self.elite_samples).sum(dim=0)
            
        # update the mean
        self.mean_old = self.mean
        self.mean     = mean_update

    def _update_step_size_evolution_path(self):
        """
        Step size evolution path is updated using polyak averaging
        """ 
        # calculate pervious contribution and new contribution
        initial_cont = (1 - self.lr_ss) * self.evolution_step
        sqrt_scalar  = (self.lr_ss * (2 - self.lr_ss) * self.mu_weights)**(1/2)
        sqrt_cont    = torch.mul(sqrt_scalar, self.invsqrtC)
        final_cont   = (self.mean - self.mean_old) / self.step_size

        # update step size evolution path
        self.evolution_step = torch.add(initial_cont, torch.mul(sqrt_cont, final_cont))

    def _update_covariance_evolution_path(self):
        """
        Covaiance evolution path is updated using polyak averaging
        """
        # calculate pervious contribution and new contribution
        initial_cont = (1 - self.lr_cmep) * self.evolution_cov
        sqrt_scalar  = (self.lr_cmep * (2 - self.lr_cmep) * self.mu_weights)**(1/2)
        final_cont   = (self.mean - self.mean_old) / self.step_size

        # update covariance evolution path
        self.evolution_cov = torch.add(initial_cont, torch.mul(sqrt_scalar, final_cont))

    def _update_step_size(self):
        """
        Step size is updated using the ratio between the expectation of the step size evolution path
        and the expectation of the norm of a normal distribution N(0, I)
        """
        # calculate inside exponential first
        exp_norm   = np.sqrt(self.search_space) * (1 - (1/(4*self.search_space)) + (1/(21*(self.search_space**2))))
        
        ratio      = torch.linalg.norm(self.evolution_step) / exp_norm
                        
        inside_exp = (self.lr_ss / self.damping) * (ratio - 1)

        # update step size
        self.step_size *= torch.exp(inside_exp)

    def _update_covariance_matrix(self):
        """
        Covariance matrix is updated via two steps
        - Rank-min(lambda, n) update:   uses histroy of covariance matrix
        - Rank-one update           :   estimates the moving steps and the sign information from history
        """
        # initial update
        initial_cont = (1 - self.lr_cmup - self.lr_one) * self.C

        # rank-one update
        rank_one = self.lr_one * torch.mul(self.evolution_cov, self.evolution_cov.t())

        # rank-min(lambda, n) update
        # get sum of all samples   

        # sample all parameters and convert to column tensor
        yi         = (torch.stack(self.elite_samples).sum(dim=0) - self.mean) / self.step_size
        print(yi)
        sample_sum = yi * yi.t()
        rank_min   = self.lr_cmup * self.weights * sample_sum

        # update covariance
        self.C = torch.add(initial_cont, torch.add(rank_one, rank_min))

    def _decompose(self):
        """
        Decompose covariance matrix into eigenvalues and re-estimate invsqrtC
        """
        # calculate B and D (eigen decomposition)
        self.C = (self.C + self.C.t()) / 2
        self.D, self.B = torch.linalg.eigh(self.C)
        
        self.B = self.B.float()
        self.D = self.D.float()

        # D is a vector of standard deviations
        self.D = torch.sqrt(self.D)

        # calculate invsqrtC
        self.invsqrtC = self.B * torch.diag(self.D.pow(-1) * self.B.t())

    @timeit
    def algorithm(self, function: any, SC_run_params: dict, iter_debug: bool = False):
        """
        Covariance matrix adaption evolutionary strategy

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
            self._decompose()
            self._elite_set()
            self._update_mean()
            self._update_step_size_evolution_path()
            self._update_step_size()
            self._update_covariance_evolution_path()
            self._update_covariance_matrix()
            self._sample_parameters(function, SC_run_params)
            self._find_best()
        
            self.reward_list.append(self.best_reward)           

            niter += 1
            if niter % 10 == 0 and iter_debug is True:
                print(f'{niter}')
        
        return self.best_parameters, self.best_reward, self.reward_list

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
