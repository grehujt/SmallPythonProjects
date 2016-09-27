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
- pydoc, gen docs from doc string
- doctest, lets you test your code by running examples embedded in the documentation and verifying that they produce the expected results.
