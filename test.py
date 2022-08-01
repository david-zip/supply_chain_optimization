import torch
from neural_nets.model_ssa import Net


hyparams_ = {'input_size': 1, 'output_size': 1}
net = Net(**hyparams_)

parameters = net.state_dict()
print(parameters)
size = sum(param.numel() for param in net.parameters())

test = torch.rand(size, 1)
print(test)
                                

# enusure bounds are not breached                                    
solutions       = test.unsqueeze(-1).tolist()
index = 0
for key, value in parameters.items():
    for tensor in value:
        tensor = tensor.unsqueeze(-1)
        for i in range(len(tensor)):
            tensor[i] = torch.tensor(solutions[index][0][0], dtype=torch.float)

            index += 1

print(parameters)