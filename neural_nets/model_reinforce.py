import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net_PG(nn.Module):
  # in current form this is a linear function (wouldn't expect great performance here)
  def __init__(self, **kwargs):
    super(Net_PG, self).__init__()

    # Unpack the dictionary 
    self.args     = kwargs
    self.dtype    = torch.float
    self.use_cuda = torch.cuda.is_available() 
    self.device   = torch.device("cpu")

    # defining ANN topology 
    self.input_size = self.args['input_size']
    self.output_sz  = self.args['output_size']
    self.hs1        = int(self.input_size/2)
    self.hs2        = self.input_size//3
    self.hs3        = self.output_sz//1

    # defining layer 
    self.hidden1 = nn.Linear(self.input_size, self.hs1)
    self.hidden2 = nn.Linear(self.hs1, self.hs2)
    self.hidden3 = nn.Linear(self.hs2, self.hs3)
    self.output  = nn.Linear(self.hs3, self.output_sz)

  def forward(self, x):
    x           = torch.tensor(x.view(1,1,-1)).float()
    y           = F.leaky_relu(self.hidden1(x), 1.0)
    y           = F.relu(self.hidden2(y))
    y           = F.elu(self.hidden3(y))            
    y           = F.sigmoid(self.output(y))       # range (0,1)

    return y
