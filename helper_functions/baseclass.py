"""
Abstract class for environment
"""
from abc import ABC, abstractmethod

class BaseEnv(ABC):

    @abstractmethod
    def J_supply_chain(self, model, SC_run_params, policy):
        pass
