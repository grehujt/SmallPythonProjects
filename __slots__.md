# Python \_\_slots\_\_

The special attribute __\_\_slots\_\___ allows you to explicitly state which instance attributes you expect your object instances to have, with the expected results:
- __faster attribute access.__
- __space savings in memory.__

---

The space savings is from:
- Storing value references in slots instead of  __\_\_dict\_\___.
- Denying  __\_\_dict\_\___ and  __\_\_weakref\_\___ creation if parent classes deny them and you declare  __\_\_slots\_\___.

---

Notes:
- Add __\_\_dict\_\___ to __\_\_slots\_\___ to get dynamic assignment
```python
class Base(object): __slots__ = ()

>>> b = Base()
>>> b.a = 'a'
Traceback (most recent call last):
  File "<pyshell#38>", line 1, in <module>
    b.a = 'a'
AttributeError: 'Base' object has no attribute 'a'

class Child(Base): __slots__ = ('a',)
>>> c = Child()
>>> c.a = 'a'
>>> c.b = 'b'
Traceback (most recent call last):
  File "<pyshell#42>", line 1, in <module>
    c.b = 'b'
AttributeError: 'Child' object has no attribute 'b'

class SlottedWithDict(Child): 
    __slots__ = ('__dict__', 'b')

>>> swd = SlottedWithDict()
>>> swd.a = 'a'
>>> swd.b = 'b'
>>> swd.c = 'c'
>>> swd.__dict__
{'c': 'c'}

>>> class NoSlots(Child): pass
>>> ns = NoSlots()
>>> ns.a = 'a'
>>> ns.b = 'b'
>>> ns.__dict__
{'b': 'b'}
	
```
- Add __\_\_weakref\_\___ to __\_\_slots\_\___ explicitly if you need that feature.

---

__\_\_slots\_\___ may cause problems for multiple inheritance:
```python
>>> class BaseA(object): __slots__ = ('a',)
>>> class BaseB(object): __slots__ = ('b',)
>>> class Child(BaseA, BaseB): __slots__ = ()
Traceback (most recent call last):
  File "<pyshell#68>", line 1, in <module>
    class Child(BaseA, BaseB): __slots__ = ()
TypeError: Error when calling the metaclass bases
    multiple bases have instance lay-out conflict

Sol:
>>> class BaseA(object): __slots__ = ()
>>> class BaseB(object): __slots__ = ()
>>> class Child(BaseA, BaseB): __slots__ = ('a', 'b')
>>> c = Child
>>> c.a = 'a'
>>> c.b = 'b'
>>> c.c = 'c'
>>> c.__dict__
<dictproxy object at 0x10C944B0>
>>> c.__dict__['c']
'c'
```

---

Set to empty tuple when subclassing a **namedtuple**:
The namedtuple builtin make immutable instances that are very lightweight (essentially, the size of tuples) but to get the benefits, you need to do it yourself if you subclass them:
```python
from collections import namedtuple
class MyNT(namedtuple('MyNT', 'bar baz')):
    """MyNT is an immutable and lightweight object"""
    __slots__ = ()

usage:
>>> nt = MyNT('bar', 'baz')
>>> nt.bar
'bar'
>>> nt.baz
'baz'
>>> nt.quux = 'quux'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'MyNT' object has no attribute 'quux'
```
You can allow __\_\_dict\_\___ creation by leaving off __\_\_slots\_\___ = (), but you can't use non-empty __\_\_slots\_\___ with subtypes of tuple.

**REF:**
[Usage of __slots__?](https://stackoverflow.com/questions/472000/usage-of-slots)
