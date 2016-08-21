# Python Koans Notes

_click [here](https://github.com/grehujt/python_koans) to check out the koans._

## Asserts

```python
def test_that_sometimes_we_need_to_know_the_class_type(self):
    """
    What is in a class name?
    """

    # Sometimes we will ask you what the class type of an object is.
    #
    # For example, contemplate the text string "navel". What is its class type?
    # The koans runner will include this feedback for this koan:
    #
    #   AssertionError: '-=> FILL ME IN! <=-' != <type 'str'>
    #
    # So "navel".__class__ is equal to <type 'str'>? No not quite. This
    # is just what it displays. The answer is simply str.
    #
    # See for yourself:

    self.assertEqual(str, "navel".__class__) # It's str, not <type 'str'>

    # Need an illustration? More reading can be found here:
    #
    #   http://bit.ly/__class__

def test_what_exception_do_you_get_when_calling_nonexistent_methods(self):
    """
    What is the Exception that is thrown when you call a method that does
    not exist?

    Hint: launch python command console and try the code in the
    block below.

    Don't worry about what 'try' and 'except' do, we'll talk about
    this later
    """
    try:
        None.some_method_none_does_not_know_about()
    except Exception as ex:
        # What exception has been caught?
        #
        # Need a recap on how to evaluate __class__ attributes?
        #   https://github.com/gregmalcolm/python_koans/wiki/Class-Attribute

        self.assertEqual(AttributeError, ex.__class__)

        # What message was attached to the exception?
        # (HINT: replace __ with part of the error message.)
        self.assertMatch("'NoneType' object has no attribute 'some_method_none_does_not_know_about'", ex.args[0])
```

## None

```python
def test_none_is_an_object(self):
    "Unlike NULL in a lot of languages"
    self.assertEqual(True, isinstance(None, object))
```

## List

```python
def test_list_literals(self):
    nums = list()
    self.assertEqual([], nums)

    nums[0:] = [1]
    self.assertEqual([1], nums)

    nums[1:] = [2]
    self.assertEqual([1, 2], nums)

    nums.append(333)
    self.assertEqual([1, 2, 333], nums)

def test_slicing_lists(self):
    noms = ['peanut', 'butter', 'and', 'jelly']

    self.assertEqual(['peanut'], noms[0:1])
    self.assertEqual(['peanut', 'butter'], noms[0:2])
    self.assertEqual([], noms[2:2])
    self.assertEqual(['and', 'jelly'], noms[2:20])
    self.assertEqual([], noms[4:0])
    self.assertEqual([], noms[4:100])
    self.assertEqual([], noms[5:0])
```

## List Assignments

```python
def test_parallel_assignments(self):
    first_name, last_name = ["John", "Smith"]
    self.assertEqual('John', first_name)
    self.assertEqual('Smith', last_name)

def test_swapping_with_parallel_assignment(self):
    first_name = "Roy"
    last_name = "Rob"
    first_name, last_name = last_name, first_name
    self.assertEqual('Rob', first_name)
    self.assertEqual('Roy', last_name)
```

## Dicts

```python
def test_making_a_dictionary_from_a_sequence_of_keys(self):
    cards = {}.fromkeys(
        ('red warrior', 'green elf', 'blue valkyrie', 'yellow dwarf',
         'confused looking zebra'),
        42)

    self.assertEqual(5, len(cards))
    self.assertEqual(42, cards['green elf'])
    self.assertEqual(42, cards['yellow dwarf'])
```

## String Manipulations

```python
def test_strings_can_change_case(self):
    self.assertEqual('Guido', 'guido'.capitalize())
    self.assertEqual('GUIDO', 'guido'.upper())
    self.assertEqual('timbot', 'TimBot'.lower())
    self.assertEqual('Guido Van Rossum', 'guido van rossum'.title())
    self.assertEqual('tOtAlLy AwEsOmE', 'ToTaLlY aWeSoMe'.swapcase())
```

## Tuples

```python
def test_creating_empty_tuples(self):
    self.assertEqual((), ())
    self.assertEqual((), tuple())  # Sometimes less confusing
```

## Methods

```python
# NOTE: Wrong number of arguments is not a SYNTAX error, but a
# runtime error.
def test_calling_functions_with_wrong_number_of_arguments(self):
    try:
        my_global_function()
    except Exception as exception:
        # NOTE: The .__name__ attribute will convert the class
        # into a string value.
        self.assertEqual('TypeError', exception.__class__.__name__)
        self.assertMatch(
            r'my_global_function\(\) takes exactly 2 arguments \(0 given\)',
            exception[0])

    try:
        my_global_function(1, 2, 3)
    except Exception as e:

        # Note, watch out for parenthesis. They need slashes in front!
        self.assertMatch(r'my_global_function\(\) takes exactly 2 arguments \(3 given\)', e[0])

# ------------------------------------------------------------------

def method_with_var_args(self, *args):
    return args

def test_calling_with_variable_arguments(self):
    self.assertEqual((), self.method_with_var_args())
    self.assertEqual(('one', ), self.method_with_var_args('one'))
    self.assertEqual(('one', 'two'), self.method_with_var_args('one', 'two'))

# ------------------------------------------------------------------

def test_pass_does_nothing_at_all(self):
    "You"
    "shall"
    "not"
    pass
    self.assertEqual(True, "Still got to this line" != None)

# ------------------------------------------------------------------

def method_with_documentation(self):
    "A string placed at the beginning of a function is used for documentation"
    return "ok"

def test_the_documentation_can_be_viewed_with_the_doc_method(self):
    self.assertMatch("A string placed at the beginning of a function is used for documentation", self.method_with_documentation.__doc__)

# ------------------------------------------------------------------

class Dog(object):
    def name(self):
        return "Fido"

    def _tail(self):
        # Prefixing a method with an underscore implies private scope
        return "wagging"

    def __password(self):
        return 'password'  # Genius!

def test_calling_methods_in_other_objects(self):
    rover = self.Dog()
    self.assertEqual("Fido", rover.name())

def test_private_access_is_implied_but_not_enforced(self):
    rover = self.Dog()

    # This is a little rude, but legal
    self.assertEqual("wagging", rover._tail())

def test_double_underscore_attribute_prefixes_cause_name_mangling(self):
    """Attributes names that start with a double underscore get
    mangled when an instance is created."""
    rover = self.Dog()
    try:
        #This may not be possible...
        password = rover.__password()
    except Exception as ex:
        self.assertEqual('AttributeError', ex.__class__.__name__)

    # But this still is!
    self.assertEqual('password', rover._Dog__password())

    # Name mangling exists to avoid name clash issues when subclassing.
    # It is not for providing effective access protection
```

## Sets

```python
def test_set_have_arithmetic_operators(self):
    scotsmen = set(['MacLeod', 'Wallace', 'Willie'])
    warriors = set(['MacLeod', 'Wallace', 'Leonidas'])

    self.assertEqual(set(['Willie']), scotsmen - warriors)
    self.assertEqual(set(['MacLeod', 'Wallace', 'Willie', 'Leonidas']), scotsmen | warriors)
    self.assertEqual(set(['MacLeod', 'Wallace']), scotsmen & warriors)
    self.assertEqual(set(['Willie', 'Leonidas']), scotsmen ^ warriors)

# ------------------------------------------------------------------

def test_we_can_compare_subsets(self):
    self.assertEqual(True, set('cake') <= set('cherry cake'))
    self.assertEqual(True, set('cake').issubset(set('cherry cake')))

    self.assertEqual(False, set('cake') > set('pie'))
```

## Exceptions

```python
class MySpecialError(RuntimeError):
    pass

def test_exceptions_inherit_from_exception(self):
    mro = self.MySpecialError.__mro__
    self.assertEqual('RuntimeError', mro[1].__name__)
    self.assertEqual('StandardError', mro[2].__name__)
    self.assertEqual('Exception', mro[3].__name__)
    self.assertEqual('BaseException', mro[4].__name__)

# ------------------------------------------------------------------

def test_try_clause(self):
    result = None
    try:
        self.fail("Oops")
    except StandardError as ex:
        result = 'exception handled'

    self.assertEqual('exception handled', result)

    self.assertEqual(True, isinstance(ex, StandardError))
    self.assertEqual(False, isinstance(ex, RuntimeError))

    self.assertTrue(issubclass(RuntimeError, StandardError), \
        "RuntimeError is a subclass of StandardError")

    self.assertEqual('Oops', ex[0])

# ------------------------------------------------------------------

def test_raising_a_specific_error(self):
    result = None
    try:
        raise self.MySpecialError, "My Message"
    except self.MySpecialError as ex:
        result = 'exception handled'

    self.assertEqual('exception handled', result)
    self.assertEqual("My Message", ex[0])
```

## Generators

```python
def test_generator_expressions_are_a_one_shot_deal(self):
    dynamite = ('Boom!' for n in range(3))

    attempt1 = list(dynamite)
    attempt2 = list(dynamite)

    self.assertEqual(['Boom!'] * 3, list(attempt1))
    self.assertEqual([], list(attempt2))

# ------------------------------------------------------------------

def generator_with_coroutine(self):
    result = yield
    yield result

def test_generators_can_take_coroutines(self):
    generator = self.generator_with_coroutine()

    # THINK ABOUT IT:
    # Why is this line necessary?
    #
    # Hint: Read the "Specification: Sending Values into Generators"
    #       section of http://www.python.org/dev/peps/pep-0342/
    next(generator)  # == generator.send(None)

    self.assertEqual(3, generator.send(1 + 2))

def test_before_sending_a_value_to_a_generator_next_must_be_called(self):
    generator = self.generator_with_coroutine()

    try:
        generator.send(1 + 2)
    except TypeError as ex:
        self.assertMatch("can't send non-None value to a just-started generator", ex[0])
```

## Classes

```python
class Dog(object):
    "Dogs need regular walkies. Never, ever let them drive."

def test_instances_of_classes_can_be_created_adding_parentheses(self):
    fido = self.Dog()
    self.assertEqual('Dog', fido.__class__.__name__)

def test_classes_have_docstrings(self):
    self.assertMatch('Dogs need regular walkies. Never, ever let them drive.', self.Dog.__doc__)

# ------------------------------------------------------------------

def test_you_can_also_access_the_value_out_using_getattr_and_dict(self):
    fido = self.Dog2()
    fido.set_name("Fido")

    self.assertEqual('Fido', getattr(fido, "_name"))
    # getattr(), setattr() and delattr() are a way of accessing attributes
    # by method rather than through assignment operators

    self.assertEqual('Fido', fido.__dict__["_name"])
    # Yes, this works here, but don't rely on the __dict__ object! Some
    # class implementations use optimization which result in __dict__ not
    # showing everything.

# ------------------------------------------------------------------

class Dog5(object):
    def __init__(self, initial_name):
        self._name = initial_name

    @property
    def name(self):
        return self._name

def test_init_provides_initial_values_for_instance_variables(self):
    fido = self.Dog5("Fido")
    self.assertEqual('Fido', fido.name)

def test_args_must_match_init(self):
    self.assertRaises(TypeError, self.Dog5)  # Evaluates self.Dog5()

    # THINK ABOUT IT:
    # Why is this so?

# ------------------------------------------------------------------

def test_all_objects_support_str_and_repr(self):
    seq = [1, 2, 3]

    self.assertEqual('[1, 2, 3]', str(seq))
    self.assertEqual('[1, 2, 3]', repr(seq))

    self.assertEqual("STRING", str("STRING"))
    self.assertEqual("'STRING'", repr("STRING"))
```


## New style class

```python
class OldStyleClass:
        "An old style class"
        # Original class style have been phased out in Python 3.

    class NewStyleClass(object):
        "A new style class"
        # Introduced in Python 2.2
        #
        # Aside from this set of tests, Python Koans sticks exclusively to this
        # kind of class
        pass

    def test_new_style_classes_inherit_from_object_base_class(self):
        self.assertEqual(True, issubclass(self.NewStyleClass, object))
        self.assertEqual(False, issubclass(self.OldStyleClass, object))

    def test_new_style_classes_have_more_attributes(self):
        self.assertEqual(2, len(dir(self.OldStyleClass)))
        self.assertEqual("An old style class", self.OldStyleClass.__doc__)
        self.assertEqual('koans.about_new_style_classes', self.OldStyleClass.__module__)

        self.assertEqual(18, len(dir(self.NewStyleClass)))
        # To examine the available attributes, run
        # 'dir(<Class name goes here>)'
        # from a python console
        # ['__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__']

    # ------------------------------------------------------------------

    def test_old_style_classes_have_type_but_no_class_attribute(self):
        self.assertEqual('classobj', type(self.OldStyleClass).__name__)

        try:
            cls = self.OldStyleClass.__class__.__name__
        except Exception as ex:
            pass

        # What was that error message from the exception?
        self.assertMatch("class OldStyleClass has no attribute '__class__'", ex[0])

    def test_new_style_classes_have_same_class_as_type(self):
        new_style = self.NewStyleClass()
        self.assertEqual(type, self.NewStyleClass.__class__)
        self.assertEqual(
            True,
            type(self.NewStyleClass) == self.NewStyleClass.__class__)
        # print self.NewStyleClass.__class__ == type  # True
        # print type(self.NewStyleClass) == type  # True

    # ------------------------------------------------------------------

    def test_in_old_style_instances_class_is_different_to_type(self):
        old_style = self.OldStyleClass()
        self.assertEqual('OldStyleClass', old_style.__class__.__name__)
        self.assertEqual('instance', type(old_style).__name__)

    def test_new_style_instances_have_same_class_as_type(self):
        new_style = self.NewStyleClass()
        self.assertEqual('NewStyleClass', new_style.__class__.__name__)
        self.assertEqual(True, type(new_style) == new_style.__class__)
```

## With statments

```python
def count_lines(self, file_name):
    try:
        f = open(file_name)
        try:
            return len(f.readlines())
        finally:
            f.close()
    except IOError:
        # should never happen
        self.fail()

def test_counting_lines(self):
    self.assertEqual(4, self.count_lines("example_file.txt"))

# ------------------------------------------------------------------

def find_line(self, file_name):
    try:
        f = open(file_name)
        try:
            for line in f.readlines():
                match = re.search('e', line)
                if match:
                    return line
        finally:
            f.close()
    except IOError:
        # should never happen
        self.fail()

def test_finding_lines(self):
    self.assertEqual('test\n', self.find_line("example_file.txt"))

## ------------------------------------------------------------------
## THINK ABOUT IT:
##
## The count_lines and find_line are similar, and yet different.
## They both follow the pattern of "sandwich code".
##
## Sandwich code is code that comes in three parts: (1) the top slice
## of bread, (2) the meat, and (3) the bottom slice of bread.
## The bread part of the sandwich almost always goes together, but
## the meat part changes all the time.
##
## Because the changing part of the sandwich code is in the middle,
## abstracting the top and bottom bread slices to a library can be
## difficult in many languages.
##
## (Aside for C++ programmers: The idiom of capturing allocated
## pointers in a smart pointer constructor is an attempt to deal with
## the problem of sandwich code for resource allocation.)
##
## Python solves the problem using Context Managers. Consider the
## following code:
##

class FileContextManager():
    def __init__(self, file_name):
        self._file_name = file_name
        self._file = None

    def __enter__(self):
        self._file = open(self._file_name)
        return self._file

    def __exit__(self, cls, value, tb):
        self._file.close()

# Now we write:

def count_lines2(self, file_name):
    with self.FileContextManager(file_name) as f:
        return len(f.readlines())

def test_counting_lines2(self):
    self.assertEqual(4, self.count_lines2("example_file.txt"))
```

## Monkey patching

```python
class AboutMonkeyPatching(Koan):
    class Dog(object):
        def bark(self):
            return "WOOF"

    def test_as_defined_dogs_do_bark(self):
        fido = self.Dog()
        self.assertEqual("WOOF", fido.bark())

    # ------------------------------------------------------------------

    # Add a new method to an existing class.
    def test_after_patching_dogs_can_both_wag_and_bark(self):
        def wag(self):
            return "HAPPY"

        self.Dog.wag = wag

        fido = self.Dog()
        self.assertEqual("HAPPY", fido.wag())
        self.assertEqual("WOOF", fido.bark())

    # ------------------------------------------------------------------

    def test_most_built_in_classes_cannot_be_monkey_patched(self):
        try:
            int.is_even = lambda self: (self % 2) == 0
        except StandardError as ex:
            self.assertMatch("can't set attributes of built-in/extension type 'int'", ex[0])

    # ------------------------------------------------------------------

    class MyInt(int):
        pass

    def test_subclasses_of_built_in_classes_can_be_be_monkey_patched(self):
        self.MyInt.is_even = lambda self: (self % 2) == 0

        self.assertEqual(False, self.MyInt(1).is_even())
        self.assertEqual(True, self.MyInt(2).is_even())
```

## Dice project
```python
class DiceSet(object):
    def __init__(self):
        self._values = None

    @property
    def values(self):
        return self._values

    def roll(self, n):
        # Needs implementing!
        # Tip: random.randint(min, max) can be used to generate random numbers
        self._values = [random.randint(1, 6) for _ in range(n)]
```

## Method bindings
```python
def function():
    return "pineapple"


def function2():
    return "tractor"


class Class(object):
    def method(self):
        return "parrot"


class AboutMethodBindings(Koan):
    def test_methods_are_bound_to_an_object(self):
        obj = Class()
        self.assertEqual(True, obj.method.im_self == obj)

    def test_methods_are_also_bound_to_a_function(self):
        obj = Class()
        self.assertEqual("parrot", obj.method())
        self.assertEqual("parrot", obj.method.im_func(obj))

    def test_functions_have_attributes(self):
        self.assertEqual(31, len(dir(function)))
        # ['__call__', '__class__', '__closure__', '__code__', '__defaults__', '__delattr__', '__dict__', '__doc__', '__format__', '__get__', '__getattribute__', '__globals__', '__hash__', '__init__', '__module__', '__name__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'func_closure', 'func_code', 'func_defaults', 'func_dict', 'func_doc', 'func_globals', 'func_name']
        self.assertEqual(True, dir(function) == dir(Class.method.im_func))

    def test_bound_methods_have_different_attributes(self):
        obj = Class()
        self.assertEqual(23, len(dir(obj.method)))
        # ['__call__', '__class__', '__cmp__', '__delattr__', '__doc__', '__format__', '__func__', '__get__', '__getattribute__', '__hash__', '__init__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__self__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'im_class', 'im_func', 'im_self']

    def test_setting_attributes_on_an_unbound_function(self):
        function.cherries = 3
        self.assertEqual(3, function.cherries)

    def test_setting_attributes_on_a_bound_method_directly(self):
        obj = Class()
        try:
            obj.method.cherries = 3
        except AttributeError as ex:
            self.assertMatch("'instancemethod' object has no attribute 'cherries'", ex[0])

    def test_setting_attributes_on_methods_by_accessing_the_inner_function(self):
        obj = Class()
        obj.method.im_func.cherries = 3
        self.assertEqual(3, obj.method.cherries)

    def test_functions_can_have_inner_functions(self):
        function2.get_fruit = function
        self.assertEqual("pineapple", function2.get_fruit())

    def test_inner_functions_are_unbound(self):
        function2.get_fruit = function
        try:
            cls = function2.get_fruit.im_self
        except AttributeError as ex:
            self.assertMatch("'function' object has no attribute 'im_self'", ex[0])

    # ------------------------------------------------------------------

    class BoundClass(object):
        def __get__(self, obj, cls):
            return (self, obj, cls)

    binding = BoundClass()

    def test_get_descriptor_resolves_attribute_binding(self):
        bound_obj, binding_owner, owner_type = self.binding
        # Look at BoundClass.__get__():
        #   bound_obj = self
        #   binding_owner = obj
        #   owner_type = cls

        self.assertEqual('BoundClass', bound_obj.__class__.__name__)
        self.assertEqual('AboutMethodBindings', binding_owner.__class__.__name__)
        self.assertEqual(AboutMethodBindings, owner_type)

    # ------------------------------------------------------------------

    class SuperColor(object):
        def __init__(self):
            self.choice = None

        def __set__(self, obj, val):
            self.choice = val

    color = SuperColor()

    def test_set_descriptor_changes_behavior_of_attribute_assignment(self):
        self.assertEqual(None, self.color.choice)
        self.color = 'purple'
        self.assertEqual('purple', self.color.choice)
```

## Decorating with functions
```python
class AboutDecoratingWithFunctions(Koan):
    def addcowbell(fn):
        fn.wow_factor = 'COWBELL BABY!'
        return fn

    @addcowbell
    def mediocre_song(self):
        return "o/~ We all live in a broken submarine o/~"

    def test_decorators_can_modify_a_function(self):
        self.assertMatch("o/~ We all live in a broken submarine o/~", self.mediocre_song())
        self.assertEqual('COWBELL BABY!', self.mediocre_song.wow_factor)

    # ------------------------------------------------------------------

    def xmltag(fn):
        def func(*args):
            return '<' + fn(*args) + '/>'
        return func

    @xmltag
    def render_tag(self, name):
        return name

    def test_decorators_can_change_a_function_output(self):
        self.assertEqual('<llama/>', self.render_tag('llama'))
```

## Decorating with classes
```python
class AboutDecoratingWithClasses(Koan):
    def maximum(self, a, b):
        if a > b:
            return a
        else:
            return b

    def test_partial_that_wrappers_no_args(self):
        """
        Before we can understand this type of decorator we need to consider
        the partial.
        """
        max = functools.partial(self.maximum)

        self.assertEqual(23, max(7, 23))
        self.assertEqual(10, max(10, -10))

    def test_partial_that_wrappers_first_arg(self):
        max0 = functools.partial(self.maximum, 0)

        self.assertEqual(0, max0(-4))
        self.assertEqual(5, max0(5))

    def test_partial_that_wrappers_all_args(self):
        always99 = functools.partial(self.maximum, 99, 20)
        always20 = functools.partial(self.maximum, 9, 20)

        self.assertEqual(99, always99())
        self.assertEqual(20, always20())

    # ------------------------------------------------------------------

    class doubleit(object):
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args) + ', ' + self.fn(*args)

        def __get__(self, obj, cls=None):
            if not obj:
                # Decorating an unbound function
                return self
            else:
                # Decorating a bound method
                return functools.partial(self, obj)

    @doubleit
    def foo(self):
        return "foo"

    @doubleit
    def parrot(self, text):
        return text.upper()

    def test_decorator_with_no_arguments(self):
        # To clarify: the decorator above the function has no arguments, even
        # if the decorated function does

        self.assertEqual('foo, foo', self.foo())
        self.assertEqual('PIECES OF EIGHT, PIECES OF EIGHT', self.parrot('pieces of eight'))

    # ------------------------------------------------------------------

    def sound_check(self):
        #Note: no decorator
        return "Testing..."

    def test_what_a_decorator_is_doing_to_a_function(self):
        #wrap the function with the decorator
        self.sound_check = self.doubleit(self.sound_check)

        self.assertEqual('Testing..., Testing...', self.sound_check())

    # ------------------------------------------------------------------

    class documenter(object):
        def __init__(self, *args):
            self.fn_doc = args[0]

        def __call__(self, fn):
            def decorated_function(*args):
                return fn(*args)

            if fn.__doc__:
                decorated_function.__doc__ = fn.__doc__ + ": " + self.fn_doc
            else:
                decorated_function.__doc__ = self.fn_doc
            return decorated_function

    @documenter("Increments a value by one. Kind of.")
    def count_badly(self, num):
        num += 1
        if num == 3:
            return 5
        else:
            return num

    @documenter("Does nothing")
    def idler(self, num):
        "Idler"
        pass

    def test_decorator_with_an_argument(self):
        self.assertEqual(5, self.count_badly(2))
        self.assertEqual('Increments a value by one. Kind of.', self.count_badly.__doc__)

    def test_documentor_which_already_has_a_docstring(self):
        self.assertEqual('Idler: Does nothing', self.idler.__doc__)

    # ------------------------------------------------------------------

    @documenter("DOH!")
    @doubleit
    @doubleit
    def homer(self):
        return "D'oh"

    def test_we_can_chain_decorators(self):
        self.assertEqual("D'oh, D'oh, D'oh, D'oh", self.homer())
        self.assertEqual("DOH!", self.homer.__doc__)
```

## Inheritance
```python
class Greyhound(Dog):
    def __init__(self, name):
        super(AboutInheritance.Greyhound, self).__init__(name)
```

## Mutiple Inheritance
click [here](https://github.com/grehujt/python_koans/python2/koans/about_multiple_inheritance.py).

## Scope
```python
class ...:
    ...

    # ------------------------------------------------------------------

    global deadly_bingo
    deadly_bingo = [4, 8, 15, 16, 23, 42]

    def test_global_attributes_can_be_created_in_the_middle_of_a_class(self):
        self.assertEqual(42, deadly_bingo[5])
```

## Module
```python
class ...:

    ...

    def test_modules_hide_attributes_prefixed_by_underscores(self):
        try:
            private_squirrel = _SecretSquirrel()
        except NameError as ex:
            self.assertMatch( "global name '_SecretSquirrel' is not defined", ex[0])

    def test_private_attributes_are_still_accessible_in_modules(self):
        from local_module import Duck  # local_module.py

        duck = Duck()
        self.assertEqual('password', duck._password)
        # module level attribute hiding doesn't affect class attributes
        # (unless the class itself is hidden).

    def test_a_modules_XallX_statement_limits_what_wildcards_will_match(self):
        """Examine results of from local_module_with_all_defined import *"""

        # 'Goat' is on the __all__ list
        goat = Goat()
        self.assertEqual('George', goat.name)

        # How about velociraptors?
        lizard = _Velociraptor()
        self.assertEqual('Cuddles', lizard.name)

        # SecretDuck? Never heard of her!
        try:
            duck = SecretDuck()
        except NameError as ex:
            self.assertMatch("global name 'SecretDuck' is not defined", ex[0])
```
