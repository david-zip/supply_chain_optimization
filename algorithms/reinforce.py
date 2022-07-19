"""
REINFORCE algorithm (NEEDS FIXING)
"""
import torch
import numpy as np

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
    