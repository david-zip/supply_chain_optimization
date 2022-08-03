"""
File stores decorators for timing algorithm and ending algorithm after given time
"""
import time

def timeit(func):
    # wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.2f} seconds')
        return result
    return timeit_wrapper

def timeend(func):
    # wraps(func)
    def timeend_wrapper(*args, **kwargs):
        pass
