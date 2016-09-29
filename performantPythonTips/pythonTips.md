# Python Tips

_[reference](http://litaotao.github.io/python-materials?utm_source=xitu)_

## Enumeration
```python
>>> list(enumerate('abc')) 
[(0, 'a'), (1, 'b'), (2, 'c')] 
 
>>> list(enumerate('abc', 1))  # second para
[(1, 'a'), (2, 'b'), (3, 'c')]
```

## Dict/Set init
```python
my_dict = {i: i * i for i in xrange(100)} 
my_set = {i * 15 for i in xrange(100)}
```

## python 2.7 floating point division
```python
from __future__ import division 
result = 1/2  # 0.5
```

## Use ast.literal_eval whenever you need eval.
```python
ast.literal_eval("__import__('os').system('rm -rf /a-path-you-really-care-about')")
# ast.literal_eval() will raise an error, but eval() will happily wipe your drive.
```

## reverse string/array
```python
>>> a = [1,2,3,4]
>>> a[::-1]
[4, 3, 2, 1]  # This creates a new reversed list. 
# If you want to reverse a list in place you can do:
a.reverse()

>>> foo = "yasoob"
>>> foo[::-1]
'boosay'
```

## ternary operation
```python
[on_true] if [expression] else [on_false]
```

## standard lib -- copy
```python
copy.copy()  # shallow copy
copy.deepcopy()
```

## id, is, ==
```python
id: mem addr
is: same id
==: call __eq__() by default
```

## function paras naming convention
```python
def request(_argv):
# add '_' in front of para name
```

## useful tools
- [pydoc](https://docs.python.org/2/library/pydoc.html), gen docs from doc string.
- [doctest](https://pymotw.com/2/doctest/), lets you test your code by running examples embedded in the documentation and verifying that they produce the expected results.
- [unittest](https://docs.python.org/2/library/unittest.html).
- [trace](https://pymotw.com/2/trace/index.html#module-trace).
- [cgitb](https://docs.python.org/2/library/cgitb.html), after this module is activated, if an uncaught exception occurs, a detailed, formatted report will be displayed. The report includes a traceback showing excerpts of the source code for each level, as well as the values of the arguments and local variables to currently running functions, to help you debug the problem.
- [pdb](https://docs.python.org/2/library/pdb.html), python debugger.
- [ipdb](https://www.safaribooksonline.com/blog/2014/11/18/intro-python-debugger/),ipdb is like pdb but it adds syntax highlightning and completion.
- [cProfile](https://docs.python.org/2/library/profile.html).
- [timeit](https://pymotw.com/2/timeit/).
- [compileall](https://pymotw.com/2/compileall/), the module finds Python source files and compiles them to the byte-code representation, saving the results in .pyc or .pyo files.
- [YAPF](https://github.com/google/yapf), a formatter for Python files by google.
- [pycallgraph](https://github.com/gak/pycallgraph/#python-call-graph), creates call graph visualizations for Python applications.
- [objgraph](https://mg.pov.lt/objgraph/), a module that lets you visually explore Python object graphs, useful when checking memory leaks.

## Usage of default arguments
```python
import math 
#original function
def degree_sin(deg):
    return math.sin(deg * math.pi / 180.0)
#optimized function, the factor variable is calculated during function creation time, 
#and so is the lookup of the math.sin method.
def degree_sin(deg, factor=math.pi/180.0, sin=math.sin):
    return sin(deg * factor)
```

## careful of default para passing by reference
```python
>>> def generate_new_list_with(my_list=[], element=None):
...     my_list.append(element)
...     return my_list
...
>>> list_1 = generate_new_list_with(element=1)
>>> list_1
[1]
>>> list_2 = generate_new_list_with(element=2)
>>> list_2
[1, 2]
```

## pow(x, y, z) == (x ** y) % z

## more on slices
```python
>>> a = [1, 2, 3, 4, 5, 6, 7]
>>> a[1:4] = []  # same as del a[1:4]
>>> a
[1, 5, 6, 7] 

>>> a = [0, 1, 2, 3, 4, 5, 6, 7]
>>> del a[::2]
>>> a
[1, 3, 5, 7]
```

## isinstance
```python
>>> isinstance(1, (float, int))
True
>>> isinstance(1.3, (float, int))
True
>>> isinstance("1.3", (float, int))
False
```

## performance boosters
- [Cython](http://cython.org/), transpiler to C.
- [PyInline](http://pyinline.sourceforge.net/), allows you to put source code from other programming languages directly "inline" in a Python script or module.
- [PyPy](http://pypy.org/).
- [Numba](http://numba.pydata.org), allows you to write high-performance functions in pure Python by generating optimized machine code.
- [Parakeet](http://www.parakeetpython.com), a runtime compiler for scientific computing in Python which uses type inference, data parallel array operators, and a lot of black magic to make your code run faster.

## use key to sort instead of cmp
```python
import operator
somelist = [(1, 5, 8), (6, 2, 4), (9, 7, 5)]
somelist.sort(key=operator.itemgetter(0))
somelist
#Output = [(1, 5, 8), (6, 2, 4), (9, 7, 5)]
somelist.sort(key=operator.itemgetter(1))
somelist
#Output = [(6, 2, 4), (1, 5, 8), (9, 7, 5)]
```

## avoid using '.' operator in loops
```python
lowerlist = ['this', 'is', 'lowercase']
upper = str.upper
upperlist = []
append = upperlist.append
for word in lowerlist:
    append(upper(word))
    print(upperlist)
    #Output = ['THIS', 'IS', 'LOWERCASE']
```

## It's Better to Beg for Forgiveness than to Ask for Permission
```python
# beg for forgiveness
n = 16
myDict = {}
for i in range(0, n):
    char = 'abcd'[i%4]
    try:
        myDict[char] += 1
    except KeyError:
        myDict[char] = 1
    print(myDict)

# ask for permission
n = 16
myDict = {}
for i in range(0, n):
    char = 'abcd'[i%4]
    try:
        myDict[char] += 1
    except KeyError:
        myDict[char] = 1
    print(myDict)
```

## use list comprehension & generator

## decorator
```python
import time
from functools import wraps
 
def timethis(func):
    '''
    Decorator that reports the execution time.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper
 
@timethis
def countdown(n):
    while n > 0:
        n -= 1
 
countdown(100000)
 
# ('countdown', 0.006999969482421875)
```

## with
```python
import time
from functools import wraps
 
def timethis(func):
    '''
    Decorator that reports the execution time.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper
 
@timethis
def countdown(n):
    while n > 0:
        n -= 1
 
countdown(100000)
 
# ('countdown', 0.006999969482421875)
```

## @contextmanager (slower than previous one)
```python
from contextlib import contextmanager
import time
 
@contextmanager
def demo(label):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print('{}: {}'.format(label, end - start))
 
with demo('counting'):
    n = 10000000
    while n > 0:
        n -= 1
 
# counting: 1.32399988174
```

## Descriptors
```python
class Celsius(object):
    def __init__(self, value=0.0):
        self.value = float(value)
    def __get__(self, instance, cls):
        return self.value
    def __set__(self, instance, value):
        self.value = float(value)
 
class Temperature(object):
    celsius = Celsius()
 
temp=Temperature()
temp.celsius #calls Celsius.__get__
```

## Zipping and unzipping lists and iterables
```python
>>> a = [1, 2, 3]
>>> b = ['a', 'b', 'c']
>>> z = zip(a, b)
>>> z
[(1, 'a'), (2, 'b'), (3, 'c')]
>>> zip(*z)
[(1, 2, 3), ('a', 'b', 'c')]
```

## Grouping adjacent list items using zip
```python
>>> a = [1, 2, 3, 4, 5, 6]
>>> # Using iterators
>>> group_adjacent = lambda a, k: zip(*([iter(a)] * k))
>>> group_adjacent(a, 3)
[(1, 2, 3), (4, 5, 6)]
>>> group_adjacent(a, 2)
[(1, 2), (3, 4), (5, 6)]
>>> group_adjacent(a, 1)
[(1,), (2,), (3,), (4,), (5,), (6,)]
>>> # Using slices
>>> from itertools import islice
>>> group_adjacent = lambda a, k: zip(*(islice(a, i, None, k) for i in range(k)))
>>> group_adjacent(a, 3)
[(1, 2, 3), (4, 5, 6)]
>>> group_adjacent(a, 2)
[(1, 2), (3, 4), (5, 6)]
>>> group_adjacent(a, 1)
[(1,), (2,), (3,), (4,), (5,), (6,)]
```
