"""
Abstract class for all algorithms
"""
from abc import ABC, abstractmethod

class OptimClass(ABC):

    @abstractmethod
    def algorithm(self, function, SC_run_params, iter_debug):
        pass

    @abstractmethod
    def func_algorithm(self, function, SC_run_params, func_call_max, iter_debug):
        pass