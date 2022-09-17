import copy
import numpy as np
import torch

class Multi_echelon_SupplyChain():

    def __init__(self, n_echelons, SC_params, connectivity_M='none', reward_f='none'):
        """
        Input parameters:
        - n_echelons                        = 2                  # number of echelons
        - connectivity_M                    = [[],[]]            # Not implemented: a conectivity matrix from each echelon (only for multi product) 
        - SC_params['material_cost']        = {0:12, 1:13, 2:11} # for 3 raw materials
        - SC_params['product_cost']         = {0:100, 1:300}     # for 2 products
        - SC_params['echelon_storage_cost'] = {0:5, 1:10}        # cost of storage for each echelon for 2 echelons
        - SC_params['echelon_storage_cap']  = {0:20, 1:7}        # max storage capacity for each echelon for 2 echelons
        - SC_params['echelon_prod_wt']      = {0:(5,0), 1:(7,0)} # for 2 echelons gaussian <= look at capped distributions (mean, std)
        - SC_params['echelon_prod_cost']    = {0:0, 1:0}         # production cost for each echelon for 2 echelons 1 product
        """
        # SC variable definitions (initial conditions)
        self.SC_params, self.n_echelons = SC_params, n_echelons
        self.connectivity_M             = connectivity_M
        self.reward_f                   = reward_f
        self.time_k                     = 0                      # time step might be important in seasonality
        self.storage_tot                = 0                      # how much storage in total there is
        self.product_tot                = 0                      # total product in the supply chain
        
        # reward statistics
        self.reward                     = 0                      # reward the supply chain has included
        self.r_product                  = 0                      # how much am I selling
        self.r_raw_mat                  = 0                      # how much am I spending in RM
        self.r_storage                  = 0                      # how much am I using in storage
        self.r_bakclog                  = 0                      # how much is the backlog costing me
        
        # constructing supply chain inventory
        self.wt_list = [SC_params['echelon_prod_wt'][ii][0] for ii in range(self.n_echelons)]
        self.wt_std  = [SC_params['echelon_prod_wt'][ii][1] for ii in range(self.n_echelons)]
        self.max_wt  = max([self.wt_list[ii] + self.wt_std[ii] for ii in range(self.n_echelons)])
        SC_inventory = np.zeros((self.n_echelons, self.max_wt + 1))   # (echelon, prod_wt + storage)

        # make inventory self    
        self.SC_inventory = SC_inventory
        self.warehouses   = self.SC_inventory[:,0]

    def advance_supply_chain_orders(self, orders, demand):
        """
        orders: np.array([raw_material=>eche_1, eche_1=>eche_2, ... , eche_n=>sale]) array of number of orders (integer value) in each echelon
        """
        # initialize function
        echelon_prod_wt          = self.SC_params['echelon_prod_wt'] # waiting times for production for each echelon
        wt_list                  = [self.wt_list[i] + self.wt_std[i] * np.random.randint(-1, 1) \
                                        for i in range(self.n_echelons)]    # adds stochasticity to the waiting time
        n_echelons               = self.n_echelons
        self.time_k             += 1
    
        if self.connectivity_M == 'none':

            # minimum between orders and stored capacity
            orders_called     = orders
            orders_called[1:] = np.minimum(self.SC_inventory[:-1,0], orders[1:])  # notice first order is from raw material ('infinite')
            sales_orders      = np.minimum(self.SC_inventory[-1,0],demand)        # you cannot sell more than the demand
            orders_called     = np.hstack((orders_called,sales_orders))
            
            # advance orders and substract from storage
            for i_eche in range(n_echelons):
                self.SC_inventory[i_eche, wt_list[i_eche]] += orders_called[i_eche]   # Add to the current storage what we are ordering
                self.SC_inventory[i_eche, 0]               -= orders_called[i_eche+1] # Remove from the current echelon what we are ordering in the next
            
            # advance all orders to the left by 1 (move a batch of orders down the line until it becomes a demand, shift the next batch order)
            for i_eche in range(n_echelons):
                shift_plus                       = copy.deepcopy(self.SC_inventory[i_eche, 1:])
                self.SC_inventory[i_eche, 0:-1] += copy.deepcopy(shift_plus[:])
                self.SC_inventory[i_eche, 1:]   -= copy.deepcopy(shift_plus[:])

            # sale orders - What is leaving the system
            sale_product = orders_called[-1]
            
            # update reward
            if self.reward_f == 'none':
                self._supply_chain_reward(orders_called, demand)
            
            # extra demand that needs to be covered
            backlog = max(0, demand - orders_called[-1])

            return sale_product, self.reward, backlog

    def supply_chain_state(self):
        """
        Returns the supply chain state vector: the inventory (can be added) + time
        """
        # import state variables
        SC_inventory_, n_echelons = copy.deepcopy(self.SC_inventory), self.n_echelons
        time_k, max_wt            = self.time_k, self.max_wt

        # reshape inventory
        SC_inventory_ = SC_inventory_.reshape((1,(max_wt+1)*n_echelons), order='F')
        
        # add time to state
        SC_state = np.hstack((SC_inventory_,np.array([[time_k]])))
        
        # return state
        return SC_state

    def _supply_chain_reward(self, orders_u, demand):
        """
        reward for multiple raw materials and multiple products
        orders_u:    the orders actually done 'orders_called' which are the control actions.
        inventory_x: notice this is not the state but the inventory.
        demand     : how many sales where asked for.
        """
        # calculate gain from meeting demand
        demand_penalty = 0; product_gain = 0
        for i in range(len(self.SC_params['product_cost'])):
            demand_penalty += self.SC_params['product_cost'][i] * 0.5           # you loose 50% extra for late product
            product_gain   += self.SC_params['product_cost'][i] * orders_u[-1]  # production gains

        # calculate material cost
        raw_mat_cost = 0
        for i in range(len(self.SC_params['material_cost'])):
            raw_mat_cost   += self.SC_params['material_cost'][i] * orders_u[0]      # raw material costs

        # incur additional cost if above storage cap
        storage_cap_cost   = 0
        for ii in range(self.n_echelons):
            storage_cap_exceed      = max(0, self.warehouses[ii] - self.SC_params['echelon_storage_cap'][ii])
            storage_cap_cost       += storage_cap_exceed * (2*self.SC_params['echelon_storage_cost'][ii])

        # check if not all demand was met
        demand_diff      = max(0, demand - orders_u[-1])

        # inventory - storage cost
        storage_cost     = sum([self.SC_params['echelon_storage_cost'][ii]*
                              self.SC_inventory[ii][0] for ii in range(self.n_echelons)])

        # calculate reward
        SC_reward        = product_gain - raw_mat_cost - (demand_diff * demand_penalty) - storage_cost - storage_cap_cost

        # update reward 
        self.reward    = SC_reward
        self.r_product = product_gain
        self.r_raw_mat = raw_mat_cost
        self.r_storage = storage_cost + storage_cap_cost
        self.r_bakclog = demand_diff*demand_penalty

        # update storage
        self.storage_tot = sum([self.SC_inventory[ii][0] for ii in range(self.n_echelons)])
        self.product_tot = np.sum(self.SC_inventory)

        # update warehouse values
        self.warehouses   = self.SC_inventory[:,0]

    def J_supply_chain(self, model, SC_run_params, policy):
        """
        Original version for stochastic search algorithms with slight modifications

        Runs a full trajectory in the environment 
        """
        steps_tot  = SC_run_params['steps_tot']
        u_norm     = SC_run_params['u_norm']   
        _          = SC_run_params['control_lb']
        _          = SC_run_params['control_ub']
        demand_lb  = SC_run_params['demand_lb']
        demand_ub  = SC_run_params['demand_ub']
        start_inv  = SC_run_params['start_inv']
        demand_f   = SC_run_params['demand_f']
        x_norm     = SC_run_params['x_norm']
        # set initial inventory and time
        self.SC_inventory[:,:] = start_inv             # starting inventory
        self.time_k            = 0
        # reward
        r_tot   = 0
        backlog = 0 # no backlog initially
        # first order
        state_norm                     = (self.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
        state_time                     = self.supply_chain_state()[0,-1] / 365
        state_torch                    = torch.tensor(np.hstack((state_norm, state_time)))
        order_k                        = policy(state_torch)
        order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]

        # === SC run === #
        for step_k in range(steps_tot):
            df_params                      = [demand_ub, demand_lb, step_k+1]  # set demand function paramters
            d_k_                           = demand_f(*df_params)
            d_k                            = d_k_ + backlog
            _, r_k, backlog                = self.advance_supply_chain_orders(order_k, d_k)
            r_tot                         += r_k
            # agent makes order
            state_norm                     = (self.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
            state_time                     = self.supply_chain_state()[0,-1] / 365
            state_torch                    = torch.tensor(np.hstack((state_norm, state_time)))
            order_k                        = policy(state_torch)
            order_k                        = (order_k*u_norm[0] + u_norm[1])[0,0]

        return r_tot

    def run_episode(self, model, SC_run_params, policy, backlog):
        """
        Test function for policy gradient algorihtms
        """
        steps_tot  = SC_run_params['steps_tot']
        u_norm     = SC_run_params['u_norm']   
        _          = SC_run_params['control_lb']
        _          = SC_run_params['control_ub']
        demand_lb  = SC_run_params['demand_lb']
        demand_ub  = SC_run_params['demand_ub']
        start_inv  = SC_run_params['start_inv']
        demand_f   = SC_run_params['demand_f']
        x_norm     = SC_run_params['x_norm']

        # initialize environemnt
        if self.time_k == 0:
            self.SC_inventory[:,:] = start_inv             # starting inventory
            self.time_k            = 0
            backlog                = 0

        orders = np.array([0 for _ in range(self.n_echelons)])

        state_norm                     = (self.supply_chain_state()[0,:-1] - x_norm[0])/x_norm[1]
        state_time                     = self.supply_chain_state()[0,-1] / 365
        state_torch                    = torch.tensor(np.hstack((state_norm, state_time)))
        order_k                        = policy(state_torch)
        orders                         = ((order_k*20).detach().numpy())[0,0]
        
        # CATEGORICAL TEST
        prob = order_k.detach().numpy()

        #for i in range(0, self.n_echelons * 20, 20):
        #    m = torch.distributions.Categorical(order_k[0][0][(0 + i):(20 + i)])
        #    action = m.sample()
        #    orders[i//20] = action.item()
        #    prob.append(m.log_prob(action))

        logprob = 0
        for i in range(len(prob)):
            logprob += np.log(prob[i])

        df_params                      = [demand_ub, demand_lb, self.time_k+1]  # set demand function paramters
        d_k_                           = demand_f(*df_params)
        d_k                            = d_k_ + backlog
        _, r_k, backlog                = self.advance_supply_chain_orders(orders, d_k)

        return r_k, logprob, backlog