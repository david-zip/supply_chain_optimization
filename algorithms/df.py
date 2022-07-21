"""
Differential evolution for neural net optimization
"""
import copy
import torch
import numpy as np

from functions.timer import timeit

class Differential_Evolution():

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameter

        ### ADJUST LATER ###

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