"""
Genetic algorithm for neural net optimization
"""
import copy
import torch
import numpy as np

from algorithms.optim import OptimClass
from helper_functions.timer import timeit

class Genetic_Algorithm(OptimClass):

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter

        - model                 =   class       # neural network
        - env                   =   class       # supply chain environment
        - kwargs['bounds']      =   [lb, ub]    # parameter bounds
        - kwargs['numbits']     =   16          # number of bits in gene
        - kwargs['population']  =   50          # number of genes
        - kwargs['cut']         =   0.4         # percent of genes operated on
        - kwargs['maxiter']     =   100         # maximum number of iterations
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
        self.cut            = self.args['cut']
        self.numbits        = self.args['numbits']
        self.population     = self.args['population']
        self.num_params     = sum(p.numel() for p in self.model.parameters())
        self.gene_length    = self.numbits * self.num_params
        self.mutation_rate  = 1.0 / float(self.numbits)

        # create required list
        self.gene_pool       = []
        self.gene_parameters = {}
        self.gene_fitness    = {}

        # initialize best solution variables
        self.best_gene         = 0
        self.best_parameters   = copy.deepcopy(self.params)
        self.best_gene_fitness = -1e8
        self.reward_list       = []

    def _initialize(self, function, SC_run_params):
        """
        Initialize aglorithm, generate random solutions and creates intial genes
        """
        # generate a list of random genes
        for i in range(self.population):
            gene = np.random.randint(0, 2, self.numbits * self.num_params).tolist()
            gene = ''.join([str(s) for s in gene])
            self.gene_pool.append(gene)
            self.gene_parameters[gene] = self.params
        
        # initialize all gene solutions and fitness
        for gene in self.gene_pool:
            
            # determine gene paramters and store it
            params = self._bitstring_to_decimal(gene)

            index = 0
            for key in self.gene_parameters[gene].keys():       # loop through keys of parameters
                for tensor in self.gene_parameters[gene][key]:  # loop through tensors in parameter dictionary
                    tensor = tensor.unsqueeze(-1)
                    for i in range(len(tensor)):                # iterate through every parameter
                        tensor[i] = torch.tensor(params[index], 
                                                    dtype=torch.float)
                        index += 1

            # calculate gene fitness and store in list
            self.model.load_state_dict(self.gene_parameters[gene])
            fitness = function(self.env, SC_run_params, self.model)
            self.gene_fitness[gene] = fitness
        
        # initialize best gene
        self._find_best()

    def _bitstring_to_decimal(self, gene):
        """
        Convert binary string into float values in decimal
        """
        decoded = []
        largest = 2**self.numbits

        for i in range(self.num_params):
            # extracting substring
            start, end = i * self.numbits, (i * self.numbits) + self.numbits
            substring = gene[start:end]

            # convert bitstring to string of characters
            chars = ''.join([str(s) for s in substring])

            # convert string to integer
            integer = int(chars, 2)

            # scale integer to desired range
            value = self.lb + integer/largest * (self.ub - self.lb)

            # store value
            decoded.append(value)

        return decoded

    def _find_best(self):
        """
        Determines the best gene and stores it
        """
        best_gene_in_generation = max(self.gene_fitness, key=self.gene_fitness.get)

        if self.gene_fitness[best_gene_in_generation] > self.best_gene_fitness:
            self.best_gene         = best_gene_in_generation
            self.best_gene_fitness = self.gene_fitness[best_gene_in_generation]
            self.best_parameters   = copy.deepcopy(self.gene_parameters[best_gene_in_generation])

    def _selection(self):
        """
        Operator to select indiviuals for crossover; tournament selection
        """
        # select k individuals from the population
        cut = np.random.randint(1, len(self.gene_pool))
        np.random.shuffle(self.gene_pool)
        shuffled_gene_pool = self.gene_pool[:cut]

        champion = shuffled_gene_pool[0]

        # select the best gene from the k individuals
        for gene in shuffled_gene_pool:
            if self.gene_fitness[gene] > self.gene_fitness[champion]:
                champion = gene

        return champion

    def _crossover(self, parent1, parent2):
        """
        Operator to create new genes from parents; single point crossover
        """
        # choose crossover point
        point = np.random.randint(1, (len(parent1) - 2))

        # convert from bitstring to list
        parent1 = [int(s) for s in parent1]
        parent2 = [int(s) for s in parent2]

        # create children and perfrom crossover
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        # convert children back into bitstrin
        child1 = ''.join([str(s) for s in child1])
        child2 = ''.join([str(s) for s in child2])

        return child1, child2

    def _mutation(self, gene):
        """
        Operator which flips bits in a gene; bit flip mutation
        """        
        # convert bitstring into list
        gene_int = []
        for char in gene:
            gene_int.append(int(char))
        
        for i in range(len(gene_int)):

            # check if mutation should occur
            if np.random.rand() < self.mutation_rate:
                gene_int[i] = 1 - gene_int[i]
        
        return gene_int

    @timeit
    def algorithm(self, function: any, SC_run_params: dict, iter_debug: bool = False):
        """
        Genetic algorithm

        - function      =   J_supply_chain function (ssa verion)
        - SC_run_params =   J_supply_chain run parameters
        - iter_debug    =   if true, prints ever 10 iterations
        """
        # initialize algorithm
        self._initialize(function, SC_run_params)

        # store current best reward
        self.reward_list.append(self.best_gene_fitness)

        # start algorithm
        # loop through generations
        niter = 0
        while niter < self.maxiter:
            for gene in self.gene_pool:

                # skip in first loop
                if niter == 0:
                    break

                # determine gene parameters
                params = self._bitstring_to_decimal(gene)
                
                index = 0
                for key in self.gene_parameters[gene].keys():       # loop through keys of parameters
                    for tensor in self.gene_parameters[gene][key]:  # loop through tensors in parameter dictionary
                        tensor = tensor.unsqueeze(-1)
                        for i in range(len(tensor)):                # iterate through every parameter
                            tensor[i] = torch.tensor(params[index], 
                                                        dtype=torch.float)
                            index += 1

                # calculate gene fitness
                self.model.load_state_dict(self.gene_parameters[gene])
                fitness = function(self.env, SC_run_params, self.model)
                self.gene_fitness[gene] = fitness

            # determine the best gene
            self._find_best()

            # selection operator
            new_gene_pool = []
            for i in range(int(len(self.gene_pool) * self.cut)):

                # tournament selection
                selected = self._selection()
                new_gene_pool.append(selected)
                self.gene_pool.remove(selected)

            # crossover operator
            for i in range(0, len(new_gene_pool), 2):
                
                # select pairs of parents
                parent1, parent2 = new_gene_pool[i], new_gene_pool[i+1]
                
                # single point crossover
                child1, child2 = self._crossover(parent1, parent2)
                new_gene_pool.append(child1)
                new_gene_pool.append(child2)

            # mutation operator
            for i in range(len(new_gene_pool)):
                gene = self._mutation(new_gene_pool[i])
                new_gene_pool[i] = ''.join([str(s) for s in gene])
            
            # fill in or remove population if needed
            if len(new_gene_pool) < self.population:
                for i in range(len(new_gene_pool), self.population):
                    gene = np.random.randint(0, 2, self.numbits * self.num_params).tolist()
                    gene = ''.join([str(s) for s in gene])
                    new_gene_pool.append(gene)

            elif len(new_gene_pool) > self.population:
                for i in range(self.population, len(new_gene_pool)):
                    np.random.shuffle(new_gene_pool)
                    removed = new_gene_pool.pop(-1)
            
            # define new population; clear dictionary list
            self.gene_pool = new_gene_pool
            self.gene_parameters.clear()
            self.gene_fitness.clear()

            for gene in self.gene_pool:
                self.gene_parameters[gene] = copy.deepcopy(self.best_parameters)

            # Stores best solution found in each iteration
            self.reward_list.append(self.best_gene_fitness)

            # iteration counter
            niter += 1

            if niter % 10 == 0 and iter_debug == True:
                print(f'{niter}')
            
        return self.best_parameters, self.best_gene_fitness, self.reward_list
