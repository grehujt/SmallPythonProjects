# Performant Python Guide

## Profiling
**Tools**:
- [cProfile](https://docs.python.org/2/library/profile.html)
    + comes by default with the standard Python interpreter (cPython) since version 2.5
    + exclusively measures CPU time
    + pays no attention to memory consumption and other memory related stats
    + main APIs:
        * run(command, filename=None, sort=-1)
        * exec(command, \_\_main\_\_.\_\_dict\_\_, \_\_main\_\_.\_\_dict\_\_)
        * runctx(command, globals, locals, filename=None)
    
    ```python
    # *** PROFILER RESULTS ***
    # fib_ver1 (XXXX\tmp\profile_examples\main.py:5)
    # function called 109 times
    #
    #          326 function calls (6 primitive calls) in 0.000 seconds
    #
    #    Ordered by: cumulative time, internal time, call count
    #
    #    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    #     109/1    0.000    0.000    0.000    0.000 main.py:5(fib_ver1)
    #     108/2    0.000    0.000    0.000    0.000 profilehooks.py:235(new_fn)
    #     108/2    0.000    0.000    0.000    0.000 profilehooks.py:329(__call__)
    #         1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
    #         0    0.000             0.000          profile:0(profiler)
    ```

    | column name | meaning |
    | ----------- | ------- |
    | ncalls | reports the number of calls to the function. If there are two numbers in this column, it means there was recursion. The second one is the number of primitive calls, and the first one is the total number of calls. This number can be helpful to identify the possible bugs (unexpected high numbers) or possible inline expansion points.) |
    | tottime | the total time spent inside the function (excluding the time spent doing subcalls to other functions). This particular information can help the developer find long running loops that could be optimized. |
    | percall | the quotient of tottime divided by ncalls. |
    | cumtime | the cumulative time spent inside the function including the time spent in subfunctions (this includes recursive calls as well). This number could help identify higher level errors, such as those in the selection of the algorithm. |
    | percall | the quotient of cumtime divided by primitive calls. |
    | filename:lineno(function) |  the file name, line number, and function name of the analyzed function. |

- [line_profiler](https://github.com/rkern/line_profiler)
    + pip install line_profiler
    + The profiler was designed to be used as a decorator
    + execute the profiler: kernprof -l script_to_profile.py
    
    ![lp](pics/line_profiler.png)

- [RunSnakeRun](http://www.vrplumber.com/programming/runsnakerun/)
    + a small GUI utility that allows you to view (Python) cProfile or Profile profiler dumps in a sortable GUI view.
    + installation:
        ```sh
        apt-get install python-profiler python-wxgtk2.8 python-setuptools
        pip install  SquareMap RunSnakeRun
        ```
    + run:
        ```sh
        python -m cProfile -o xxx.prof xxx.py \<paras\>
        ```
    
    ![png](pics/rsr.png)

## Tips and tricks of Python
- Memoization / lookup tables
- Usage of default arguments
    ```python
    import math 
    #original function
    def degree_sin(deg):
        return math.sin(deg * math.pi / 180.0)
    #optimized function, the factor variable is calculated during function 
    creation time, 
    #and so is the lookup of the math.sin method.
    def degree_sin(deg, factor=math.pi/180.0, sin=math.sin):
        return sin(deg * factor)
    ```
- List comprehension and generators
    + dis module to show the bytecode
    + when generating a list, the for loop should not be your weapon of choice.
    + when generating big lists, consider using generators.
    
    ![png](pics/perf1.png)

- ctypes
    + allows the developer to reach under the hood of Python and tap into the power of the C language.
- String concatenation
    + In Python, strings are immutable, which means that once you create one you can't really change its value.
    + variable interpolation
    + join
- Membership testing
    + set and dict
- collections
- funcion calls can be expensive
- When possible, sort by the key, instead of assign a cmp
- 1 is better than True
- Multiple assignments are slow
- Chained comparisons are good
- Using namedtuples instead of regular (small) objects


**References:**
- Mastering Python High Performance
