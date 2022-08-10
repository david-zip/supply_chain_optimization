import copy
import numpy as np


class Multi_echelon_SupplyChain():

    def __init__(self, n_echelons, SC_params, connectivity_M='none', reward_f='none'):
        '''
        Input parameters:
        - n_echelons                        = 2                  # number of echelons
        - connectivity_M                    = [[],[]]            # Not implemented: a conectivity matrix from each echelon (only for multi product) 
        - SC_params['material_cost']        = {0:12, 1:13, 2:11} # for 3 raw materials
        - SC_params['product_cost']         = {0:100, 1:300}     # for 2 products
        - SC_params['echelon_storage_cost'] = {0:5, 1:10}        # cost of storage for each echelon for 2 echelons
        - SC_params['echelon_storage_cap']  = {0:20, 1:7}        # max storage capacity for each echelon for 2 echelons
        - SC_params['echelon_prod_wt']      = {0:(5,1), 1:(7,1)} # for 2 echelons gaussian <= look at capped distributions (mean, std)
        - SC_params['echelon_prod_cost']    = {0:0, 1:0}         # production cost for each echelon for 2 echelons 1 product
        '''
        
        # SC variable definitions (Initial conditions)
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
        
        # constructing supply chain inventory (QUESTION)
        self.wt_list = [SC_params['echelon_prod_wt'][ii][0] for ii in range(n_echelons)]
        self.wt_std  = [SC_params['echelon_prod_wt'][ii][1] for ii in range(n_echelons)]
        self.max_wt  = max([self.wt_list[ii] + self.wt_std[ii] for ii in range(n_echelons)])
        SC_inventory_ = np.asarray(
                                    [np.random.random((len(self.SC_params['product_cost']), self.max_wt + 1)) \
                                    for _ in range(self.n_echelons)]
                        )   # (echelon, prod_wt + storage)

        # make inventory self
        self.SC_inventory = SC_inventory_
        print(self.SC_inventory)
        print(self.SC_inventory[-1,-1,0])
        self.warehouses   = self.SC_inventory[:,:,0]

    def advance_supply_chain_orders(self, orders, demand):
        '''
        Original advance SC orders
        orders: np.array([raw_material=>eche_1, eche_1=>eche_2, ... , eche_n=>sale]) array of number of orders (integer value) in each echelon
        '''
        # Confusion about what a wt_list is. Is it a list of delivery times or something else? (QUESTION)
        echelon_prod_wt, wt_list = self.SC_params['echelon_prod_wt'], self.wt_list # waiting times for production for each echelon
        n_echelons               = self.n_echelons
        self.time_k             += 1
    
        if self.connectivity_M == 'none':
            # minimum between orders and stored capacity
            # (QUESTION) is orders the input of raw material into the system, and the demand the output of products from the system?
            orders_called     = orders
            orders_called[1:] = np.minimum(self.SC_inventory[:-1,0], orders[1:])  # notice first order is from raw material ('infinite')
            sales_orders      = np.minimum(self.SC_inventory[-1,0],demand)        # you cannot sell more than the demand
            orders_called     = np.hstack((orders_called,sales_orders))
            # advance orders and substract from storage
            for i_eche in range(n_echelons):
                self.SC_inventory[i_eche, wt_list[i_eche]] += orders_called[i_eche]   # Add to the current storage what we are ordering (COMMENT)
                self.SC_inventory[i_eche, 0]               -= orders_called[i_eche+1] # Remove from the current echelon what we are ordering in the next (COMMENT)
            # advance all orders by 1 // move a batch of orders down the line until it becomes a demand, shift the next batch order)
            for i_eche in range(n_echelons):
                shift_plus                       = copy.deepcopy(self.SC_inventory[i_eche, 1:])
                self.SC_inventory[i_eche, 0:-1] += copy.deepcopy(shift_plus[:])
                self.SC_inventory[i_eche, 1:]   -= copy.deepcopy(shift_plus[:])
            # sale orders - What is leaving the system (COMMENT)
            sale_product = orders_called[-1]
            # update reward
            if self.reward_f == 'none':
                self.supply_chain_reward_siso(orders_called, demand)
            # extra demand that needs to be covered
            backlog = max(0, demand - orders_called[-1])
            # == return == #
            return sale_product, self.reward, backlog
    
    ####################################
    # --- edited advance SC orders --- #
    ####################################
    def advance_supply_chain_orders_DE(self, orders, demand):
        '''
        orders: np.array([raw_material=>eche_1, eche_1=>eche_2, ... , eche_n=>sale]) array of number of orders (integer value) in each echelon
        '''
        # Confusion about what a wt_list is. Is it a list of delivery times or something else? (QUESTION)
        echelon_prod_wt          = self.SC_params['echelon_prod_wt'] # waiting times for production for each echelon
        wt_list                  = [self.wt_list[i] + self.wt_std[i] * np.random.randint(-1, 1) \
                                        for i in range(self.n_echelons)]    # adds stochasticity to the waiting time
        n_echelons               = self.n_echelons
        self.time_k             += 1
    
        if self.connectivity_M == 'none':
            if len(self.SC_params['product_cost']) == 1:
                # minimum between orders and stored capacity
                orders_called     = orders
                orders_called[1:] = np.minimum(self.SC_inventory[:,0,0], orders[1:])  # notice first order is from raw material ('infinite')
                sales_orders      = np.minimum(self.SC_inventory[-1,-1,0],demand)        # you cannot sell more than the demand
                orders_called     = np.hstack((orders_called,sales_orders))
            else:
                orders_called     = orders
                orders_called[1:] = np.minimum(self.SC_inventory[:,0,0], orders[:,1:].transpose())  # notice first order is from raw material ('infinite')
                sales_orders      = np.minimum(self.SC_inventory[:,-1,0], demand.transpose())        # you cannot sell more than the demand
                orders_called     = np.hstack((orders_called,sales_orders.transpose()))
            # advance orders and substract from storage
            for i_eche in range(n_echelons):
                self.SC_inventory[i_eche, :, wt_list[i_eche]] += orders_called[:, i_eche]   # Add to the current storage what we are ordering (COMMENT)
                self.SC_inventory[i_eche, :, 0]               -= orders_called[:, i_eche+1] # Remove from the current echelon what we are ordering in the next (COMMENT)
            # advance all orders by 1 // move a batch of orders down the line until it becomes a demand, shift the next batch order)
            for i_eche in range(n_echelons):
                shift_plus                        = copy.deepcopy(self.SC_inventory[i_eche,:, 1:])
                print(shift_plus)
                self.SC_inventory[i_eche,:,0:-1] += copy.deepcopy(shift_plus[:])
                self.SC_inventory[i_eche,:,1:]   -= copy.deepcopy(shift_plus[:])
            # sale orders - What is leaving the system (COMMENT)
            sale_product = orders_called[:,-1]
            # update reward
            if self.reward_f == 'none':
                self.supply_chain_reward_mimo(orders_called, demand)
            # extra demand that needs to be covered
            backlog = np.maximum(0, demand - orders_called[:,-1])
            print('backlof=',backlog)
            # == return == #
            return sale_product, self.reward, backlog

    #####################################################
    # --- outputs current state of the supply chain --- #
    #####################################################
    def supply_chain_state(self):
        '''
        returns the supply chain state vector: the inventory (can be added) + time
        '''
        # import state variables
        SC_inventory_, n_echelons = copy.deepcopy(self.SC_inventory), self.n_echelons
        time_k, max_wt            = self.time_k, self.max_wt

        # reshape inventory
        SC_inventory_ = SC_inventory_.reshape((1,(max_wt+1)*n_echelons*len(self.SC_params['product_cost'])), order='F')
        # add time to state
        #SC_state = np.hstack((SC_inventory_,np.array([[time_k]])))
        SC_state  = SC_inventory_    
        # return state
        return SC_state

    def supply_chain_reward_siso(self, orders_u, demand):
        '''
        reward for single raw material and single product
        orders_u:    the orders actually done 'orders_called' which are the control actions.
        inventory_x: notice this is not the state but the inventory.
        demand     : how many sales where asked for.
        '''
        # Is the penalty only for the second product? (QUESTION)
        demand_penalty = self.SC_params['product_cost'][0] * 0.5 # you loose 50% extra for late product

        # check if not all demand was met
        demand_diff    = max(0, demand - orders_u[-1])
        # raw material costs 
        raw_mat_cost   = self.SC_params['material_cost'][0]*orders_u[0]
        # production gains
        product_gain   = self.SC_params['product_cost'][0] *orders_u[-1]
        # inventory - storage cost
        storage_cost   = sum([self.SC_params['echelon_storage_cost'][ii]*
                              self.SC_inventory[ii][0] for ii in range(self.n_echelons)])
        # calculate reward
        SC_reward    = product_gain - raw_mat_cost - demand_diff*demand_penalty - storage_cost
        
        # update reward 
        self.reward    = SC_reward
        self.r_product = product_gain
        self.r_raw_mat = raw_mat_cost
        self.r_storage = storage_cost
        self.r_bakclog = demand_diff*demand_penalty
        # update storage
        self.storage_tot = sum([self.SC_inventory[ii][0] for ii in range(self.n_echelons)])
        self.product_tot = np.sum(self.SC_inventory)
        # update warehouse values
        self.warehouses   = self.SC_inventory[:,0]

    def supply_chain_reward_mimo(self, orders_u, demand):
        '''
        reward for (eventually) multiple raw materials and multiple products
        orders_u:    the orders actually done 'orders_called' which are the control actions.
        inventory_x: notice this is not the state but the inventory.
        demand     : how many sales where asked for.
        '''
        demand_penalty = 0; product_gain = 0
        for i in range(len(self.SC_params['product_cost'])):
            demand_penalty += self.SC_params['product_cost'][i] * 0.5               # you loose 50% extra for late product
            product_gain   += self.SC_params['product_cost'][i] * orders_u[i,-1]    # production gains
        
        raw_mat_cost = 0
        for i in range(len(self.SC_params['material_cost'])):
            raw_mat_cost += self.SC_params['material_cost'][i] * orders_u[i,0]     # raw material costs

        # check if not all demand was met
        demand_diff    = np.maximum(0, demand - orders_u[:, -1])[0,0]

            
        # inventory - storage cost
        storage_cost   = np.sum([self.SC_params['echelon_storage_cost'][ii] *
                              self.SC_inventory[ii,:,0] for ii in range(self.n_echelons)])
        # calculate reward
        SC_reward    = product_gain - raw_mat_cost - (demand_diff * demand_penalty) - storage_cost
        
        # update reward 
        self.reward    = SC_reward
        self.r_product = product_gain
        self.r_raw_mat = raw_mat_cost
        self.r_storage = storage_cost
        self.r_bakclog = demand_diff*demand_penalty
        # update storage
        self.storage_tot = np.sum([self.SC_inventory[ii,:,0] for ii in range(self.n_echelons)])
        self.product_tot = np.sum(self.SC_inventory)
        # update warehouse values
        self.warehouses   = self.SC_inventory[:,:,0]
