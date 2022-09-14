"""
REINFORCE algorithm for reinforcement learning
"""
import copy
import torch
import numpy as np

from algorithms.optim import OptimClass
from helper_functions.timer import timeit

def reinforce(policy_net, rewards, orders, GAMMA, LEARNING_RATE):
    ### REINFORCE ALGORITHM ###
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    #
    rewards = np.array(rewards)
    orders = np.array(orders)

    # convert from numpy to tensor
    rewards = torch.tensor(rewards, requires_grad=True)
    orders = torch.tensor(orders, requires_grad=True)

    rewards = torch.unsqueeze(rewards, 1)
    orders = torch.unsqueeze(orders, 1)

    # calculate discounted rewards
    discounted_reward = 0
    for t, reward in enumerate(rewards):
        discounted_reward += GAMMA**t * reward

    # calculated log probability
    logprobs = []
    for order in orders:
        logprobs.append(torch.log_softmax(order, dim=-1))
    
    # calculate policy loss
    loss = []
    for logprob in logprobs:
        loss.append(-logprob * discounted_reward)
    loss = torch.cat(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class REINFORCE(OptimClass):

    def __init__(self, model, env, **kwargs):
        """
        Initialize algorithm hyper-parameters

        - model                  =   class       # neural network
        - env                    =   class       # supply chain envionment
        - kwargs['lr']           =   0.01        # learning rate
        - kwargs['gamma']        =   0.99        # discount factor
        - kwargs['maxiter']      =   1000        # maximum number of episodes
        """

        # unpack the arguments
        self.model      = model                 # neural network
        self.env        = env                   # environment 
        self.args       = kwargs

        # store model parameters
        self.params     = self.model.state_dict()   # inital parameter values

        # maximum iterations/time
        self.maxiter    = self.args['maxiter']

        # store algorithm hyperparameters
        self.lr         = kwargs['lr']
        self.gamma      = kwargs['gamma']

        # initialise list for algorithm
        self.total_rewards    = []
        self.logprobs   = []

    def algorithm(self, function, SC_run_params, iter_debug):
        """
        REINFORCE algorithm

        - function      =   J_supply_chain function (unsure yet)
        - SC_run_params =   J_supply_chain run parameters
        - iter_debug    =   if true, prints ever 100 iterations
        """
        return super().algorithm(function, SC_run_params, iter_debug)
    
    def reinitialize(self):
        """
        Reinitialize class to original state
        """
        self.__init__(self.model, self.env, **self.args)

    def func_algorithm(self, function, SC_run_params, func_call_max, iter_debug):
        """
        Not needed for REINFORCE
        """
        return super().func_algorithm(function, SC_run_params, func_call_max, iter_debug)
        