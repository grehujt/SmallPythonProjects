
from profilehooks import profile

# Basic example ========================================================================


@profile
def fib_ver1(n):
    if n < 0:
        raise RuntimeError
    elif n <= 2:
        return 1
    else:
        return fib_ver1(n - 1) + fib_ver1(n - 2)

print fib_ver1(10)
# 55

# *** PROFILER RESULTS ***
# fib_ver1 (XXXX\tmp\profile_examples\main.py:5)
# function called 109 times

#          326 function calls (6 primitive calls) in 0.000 seconds

#    Ordered by: cumulative time, internal time, call count

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#     109/1    0.000    0.000    0.000    0.000 main.py:5(fib_ver1)
#     108/2    0.000    0.000    0.000    0.000 profilehooks.py:235(new_fn)
#     108/2    0.000    0.000    0.000    0.000 profilehooks.py:329(__call__)
#         1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
#         0    0.000             0.000          profile:0(profiler)
# NOTE: When there are two numbers in the ncalls column (for example, 109/1), then the latter is the
# number of primitive calls, and the former is the actual number of calls. When the function does not
# recurse, these two values are the same, and only the single figure is printed.


# Class example ========================================================================
# This example is taken from https://mg.pov.lt/profilehooks/

class SampleClass:

    @profile
    def silly_fibonacci_example(self, n):
        """Return the n-th Fibonacci number.
        This is a method rather rather than a function just to illustrate that
        you can use the 'profile' decorator on methods as well as global
        functions.
        Needless to say, this is a contrived example.
        """
        if n < 1:
            raise ValueError('n must be >= 1, got %s' % n)
        if n in (1, 2):
            return 1
        else:
            return (self.silly_fibonacci_example(n - 1) +
                    self.silly_fibonacci_example(n - 2))


if __name__ == '__main__':
    fib = SampleClass().silly_fibonacci_example
    print fib(10)

# *** PROFILER RESULTS ***
# silly_fibonacci_example (xxx\main.py:44)
# function called 109 times

#          326 function calls (6 primitive calls) in 0.000 seconds

#    Ordered by: cumulative time, internal time, call count

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#     109/1    0.000    0.000    0.000    0.000 main.py:44(silly_fibonacci_example)
#     108/2    0.000    0.000    0.000    0.000 profilehooks.py:235(new_fn)
#     108/2    0.000    0.000    0.000    0.000 profilehooks.py:329(__call__)
#         1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
#         0    0.000             0.000          profile:0(profiler)


# Cache Fib Example ===============================================================
# This example is taken from "Mastering Python High Performance"

class cached:
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            result = self.fn(*args)
            self.cache[args] = result
            return result


@cached
@profile
def fib_ver2(n):
    if n < 0:
        raise RuntimeError
    elif n <= 2:
        return 1
    else:
        return fib_ver2(n - 1) + fib_ver2(n - 2)

print fib_ver2(10)
# *** PROFILER RESULTS ***
# fib_ver2 (xxx\main.py:97)
# function called 10 times

#          45 function calls (6 primitive calls) in 0.000 seconds

#    Ordered by: cumulative time, internal time, call count

#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#      10/1    0.000    0.000    0.000    0.000 main.py:97(fib_ver2)
#      16/2    0.000    0.000    0.000    0.000 main.py:88(__call__)
#       9/1    0.000    0.000    0.000    0.000 profilehooks.py:235(new_fn)
#       9/1    0.000    0.000    0.000    0.000 profilehooks.py:329(__call__)
#         1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
#         0    0.000             0.000          profile:0(profiler)
