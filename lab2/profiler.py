import cProfile
import pstats

# Assuming these functions are defined in the same or another file
def function_a():
    # Some operation
    function_b()

def function_b():
    # Another operation
    for i in range(1000000):
        pass

def main():
    function_a()  # Calls function_a, which calls function_b

# Profile only the `main` function and capture all called functions
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()  # Profile this function and all the functions it calls
    
    profiler.disable()
    
    # Print profiling stats sorted by cumulative time
    profiler.print_stats(sort='cumtime')
