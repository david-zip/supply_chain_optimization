# MSc Research Project

## Overview

Python code to reinforcement learning for supply chain optimization using stochasitc serach algorithms.

Supply chain environment consist of single raw material, single product, waiting time stochasicity, and echelon flexibility.

### Directory
- Environment.py     > Supply chain environment
- /Algorihtms/       > Folder containing all the stochastic search algorithms 
- /helper_functions/ > helper functions for demand functions, training, and plotting 
- /neural_net/       > neural network and trained parameters 

## Tasks

- [x] refactor training file
- [x] add constraint bounds to PSO
- [x] add constraint bounds to ABC
- [x] write GA in oop
- [x] add GA to environment
- [x] implement gaussian ES
- [ ] implement CMA-ES
- [x] implement differential evolution
- [ ] implement NES (optional)
- [x] add parallelisation to SA
- [x] add time functionality
- [x] Store solutions in CSV file for reproducibility
- [x] Refactor main files into helper functions
- [x] Formalize readme document
- [ ] implement REINFORCE
- [ ] implement Actor-Critic algorithm
