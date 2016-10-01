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

## Sliding windows (n-grams) using zip and iterators
```python
>>> from itertools import islice
>>> def n_grams(a, n):
...     z = (islice(a, i, None) for i in range(n))
...     return zip(*z)
...
>>> a = [1, 2, 3, 4, 5, 6]
>>> n_grams(a, 3)
[(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
>>> n_grams(a, 2)
[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
>>> n_grams(a, 4)
[(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]
```

## Inverting a dictionary using zip
```python
>>> m = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
>>> m.items()
[('a', 1), ('c', 3), ('b', 2), ('d', 4)]
>>> zip(m.values(), m.keys())
[(1, 'a'), (3, 'c'), (2, 'b'), (4, 'd')]
>>> mi = dict(zip(m.values(), m.keys()))
>>> mi
{1: 'a', 2: 'b', 3: 'c', 4: 'd'}
```

## Inverting a dictionary using zip
```python
>>> m = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
>>> m.items()
[('a', 1), ('c', 3), ('b', 2), ('d', 4)]
>>> zip(m.values(), m.keys())
[(1, 'a'), (3, 'c'), (2, 'b'), (4, 'd')]
>>> mi = dict(zip(m.values(), m.keys()))
>>> mi
{1: 'a', 2: 'b', 3: 'c', 4: 'd'}
```

## Flattening lists
```python
>>> a = [[1, 2], [3, 4], [5, 6]]
>>> list(itertools.chain.from_iterable(a))
[1, 2, 3, 4, 5, 6]
>>> sum(a, [])
[1, 2, 3, 4, 5, 6]
>>> [x for l in a for x in l]
[1, 2, 3, 4, 5, 6]
>>> a = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
>>> [x for l1 in a for l2 in l1 for x in l2]
[1, 2, 3, 4, 5, 6, 7, 8]
>>> a = [1, 2, [3, 4], [[5, 6], [7, 8]]]
>>> flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
>>> flatten(a)
[1, 2, 3, 4, 5, 6, 7, 8]
```

## Dictionary comprehensions
```python
>>> m = {x: x ** 2 for x in range(5)}
>>> m
{0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
>>> m = {x: 'A' + str(x) for x in range(10)}
>>> m
{0: 'A0', 1: 'A1', 2: 'A2', 3: 'A3', 4: 'A4', 5: 'A5', 6: 'A6', 7: 'A7', 8: 'A8', 9: 'A9'}
```

## Be aware of default para
```python
def foo(bar=[]):        # bar is optional and defaults to [] if not specified
    bar.append("baz")    # but this line could be problematic, as we'll see...
    return bar

>>> foo()
["baz"]
>>> foo()
["baz", "baz"]
>>> foo()
["baz", "baz", "baz"]
```

## do not try to add/del elements during iteration
```python
# wrong
>>> odd = lambda x : bool(x % 2)
>>> numbers = [n for n in range(10)]
>>> for i in range(len(numbers)):
...     if odd(numbers[i]):
...         del numbers[i]  # BAD: Deleting item from a list while iterating over it
...
Traceback (most recent call last):
      File "<stdin>", line 2, in <module>
IndexError: list index out of range

# correct
>>> odd = lambda x : bool(x % 2)
>>> numbers = [n for n in range(10)]
>>> numbers[:] = [n for n in numbers if not odd(n)]  # ahh, the beauty of it all
>>> numbers
[0, 2, 4, 6, 8]
```

## generator
```python
%timeit -n 100 a = (i for i in range(100000))
%timeit -n 100 b = [i for i in range(100000)]
100 loops, best of 3: 1.54 ms per loop
100 loops, best of 3: 4.56 ms per loop

%timeit -n 10 for x in (i for i in range(100000)): pass
%timeit -n 10 for x in [i for i in range(100000)]: pass
10 loops, best of 3: 6.51 ms per loop
10 loops, best of 3: 5.54 ms per loop
```

## while 1 is faster than while True
> True is a global variable in python 2.x

## use \*\* instead of pow
```python
%timeit -n 10000 c = pow(2,20)
%timeit -n 10000 c = 2**20
10000 loops, best of 3: 284 ns per loop
10000 loops, best of 3: 16.9 ns per loop
```

## use cProfile/cPickle/cStringIO instead of profile/pickle/stringIO

## deserialization
```python
import json
import cPickle
a = range(10000)
s1 = str(a)
s2 = cPickle.dumps(a)
s3 = json.dumps(a)
%timeit -n 100 x = eval(s1)
%timeit -n 100 x = cPickle.loads(s2)
%timeit -n 100 x = json.loads(s3)
100 loops, best of 3: 16.8 ms per loop
100 loops, best of 3: 2.02 ms per loop
100 loops, best of 3: 798 µs per loop
```


