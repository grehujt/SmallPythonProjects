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
