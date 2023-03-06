---
title: "Singleton Class Decorator"
author: Jeremy Ribeiro
layout: post
icon: fa-briefcase
icon-style: fas
published: true
hidden: true
---

While coding in Python, I wanted to make a class become a singleton class, ie a class that can only have one instance. So I've looked around, and found some singleton decorators. For the most part, they worked fine, but on some occasions, the classes I created with them did not behave as usual classes would. And the reason was that the singleton classes created with these decorators weren't classes, but callable objects that return an object of the class on which the singleton decorator was called.

I then wondered... What would it take for me to write a singleton decorator that makes classes that behave like normal classes (besides being a singleton)?
Lead by curiosity, I started to draft such a singleton decorator. One of my main goals with this side project was for me to learn more about Python. In particular, I wanted to know more about how the internals of classes worked, how we can change classes' behaviors etc. It ended up being a functional singleton decorator, that any Python programmer can use if they want to (see GitHub repo [here](https://github.com/Jrmy-rbr/singleton-class-decorator)).


Let me tell you a bit about how my singleton decorator works.

## A first "naive" approach

If one were to manually write a singleton class, one of the possible ways of doing is to overwrite the `__new__` function of that class following a classic design pattern as follows:

``` python
class SingletonClass:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def some_method(self): 
        ...
```

My singleton decorator essentially does this: It overwrites the `__new__` function of its input class. Reading this, you may think that an easy implementation of such a decorator would then be,
```python
def singleton(klass):
    # This is the function that will overwirete the `__new__` function of the input `klass`
    def new_overwrite(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(klass, cls).__new__(cls)
        return cls.instance

    klass.__new__ = new_overwrite
    return klass
```

However, with this solution, if the user wants to create a child class `ChildClass` from the initial singleton class, not only the child class will be a singleton class (and the user has no control over this), but worse, the instance of the child class and its parent class will be the same:

```python
# this is the singleton decorator has written above
@singleton
class SingletonClass:
    ...

class ChilClass(SingletonClass):
    ...

# The assert passes
assert SingletonClass() is ChildClass()
```

There are several ways for solving this issue. 
For example, one could simply forbid inheritance for singleton classes. 
It would work but it is not satisfactory in my opinion. Indeed, a singleton classes would behave somewhat differently than ordinary classes (they'd not support inheritance). Also, I do believe that this solution requires the use of [meta-classes](https://www.geeksforgeeks.org/python-metaclasses/). 
And if I were going to use meta-classes I would as well use them for something a bit more sophisticated than just forbidding inheritance.

And this is basically what I have done. I've used a meta-class called `MakeSingleton` that takes care of overwriting the `__new__` method of the input class of my singleton decorator. 

## The MakeSingleton meta-class

Before implementing the `MakeSingleton` meta-class I had to choose how I wanted to handle "singletoness" of child classes. I needed to choose whether the child classes of a singleton class should also be singletons. I decided to let the user of the singleton decorator decide what they want.
In particular, child classes of singleton classes are not singletons by default, unless one uses the singleton decorator on them:
```python
# definition of the Parent with the singleton
@singleton
class ParentSingleton:
    ...

class ChildClass_1(ParentClass):
    # Not a sngleton class
    ...

@singleton
class ChildClass_2(ParentClass):
    # Is a singleton class
    ...
```
As you can see it is also very flexible as it is easy to make some child classes singletons, and others not.

Note that in the actual implementation, I do disable inheritance by default, and one needs to explicitly specify that a singleton class can inherit. Therefore the correct syntax for the above example is,
```python
# definition of the Parent with the singleton decorator, and enabling inheritance by setting `is_final=False`
@singleton(is_finale=False) 
class ParentSingleton:
    ...

class ChildClass_1(ParentClass):
    # Not a sngleton class
    ...

@singleton # here I don't set `is_final` so by default this class cannot be used for inheritance
class ChildClass_2(ParentClass):
    # Is a singleton class
    ...
```

I have chosen to forbid inheritance by default as I believe it's a bit safer if, for example, 
there is an edge case I did not consider in my implementation that causes a bug in child classes. Forbidding inheritance by default prevents the user from mistakenly using a singleton class as a parent class. The user has to make a conscious choice to do so.


Let's now see what the `MakeSingleton` class looks like:

```python
# This is a slightly simplified version of the actual `MakeSingleton` meta-class.
# Note that the extra `make_singleton` argument is used to manage the default "singletoness" of child classes
class MakeSingleton(type):
    def __new__(cls, name, bases, classdict, make_singleton:bool=False):
        old_class = type.__new__(cls, name, bases, classdict)

        # Make the singleton class if make_singleton
        if make_singleton:
            classdict["_old_new"] = old_class.__new__ if "__new__" not in classdict else classdict["__new__"]
            classdict["__new__"] = new_overwrite # as defined before
            return type.__new__(cls, name, bases, classdict)

        # if not make_singleton, simply forward the __new__/_old_new class of the old_class to the new one.
        old_new = classdict["__new__"] if "__new__" in classdict else getattr(old_class, "_old_new", old_class.__new__)
        classdict["__new__"] = old_new

        return type.__new__(cls, name, bases, classdict)
```

With this kind of meta-class, one can now create singleton classes as follows,

```python
# don't forget the `make_singleton` argument, or else the class will not be a Singleton
MySingletonClass = MakeSingleton("MySingletonClass", (,), dict(), make_singleton=True)
```

With this `MakeSingleton` meta-class, one can already write a prototype of a singleton decorator that creates singleton classes whose child classes are not singleton by default. This decorator could be:

```python
def singleton(klass):
    # we create a singletion class, passing `klass` as a parent class
    return MakeSingleton(klass.__name__, (klass,), dict(), make_singleton=True)
```

## Disabling inheritance

As I mentioned before, my decorator disables inheritance by default for the singleton classes it creates.

To disable inheritance I use another meta-class called `MakeFinalSingleton` that inherits from `MakeSingleton` but to which I add a piece of code that disables inheritance:

```python
class MakeFinalSingleton(MakeSingleton):
    def __new__(cls, name, bases, classdict, make_singleton: bool = True):
        # small piece of code that disable inheritance
        for base in bases:
            if isinstance(base, cls):
                raise TypeError(f"type '{base.__name__}' is not an acceptable base type")

        # return the same as its parent class `MakeSingleton`
        return super(cls, cls).__new__(cls, name, bases, classdict, make_singleton)
```

## Putting everything together

Now that we have the `MakeSingleton`, and `MakeFinalSingleton` meta classes, we can add an argument to the singleton decorator so that the user can choose to allow inheritance for their class or not. So here is a version of the decorator that is almost the one I've implemented:

```python
# A slightly simplified version of the actual singleton decorator
def singleton(klass= None, /, *, is_final = True):

    def wrapper(klass):
        # Choose which meta class to use
        MetaClass = MakeFinalSingleton if is_final else MakeSingleton

        # Create and return the new singleton class using MetaClass
        return MetaClass(klass.__name__, (klass,), dict(), make_singleton=True)

    return wrapper(klass) if klass is not None else wrapper
```

Note that the internal `wrapper` function is here as a trick to go around the constraint that normally, decorators can only take one argument: the input class/function. You may 
see this as a sort of [Currying](https://en.wikipedia.org/wiki/Currying) of the decorator.

## One last caveat

The above implementation of the singleton decorator would work for many classes. But there are some classes for which using it would cause an error. These classes are classes that use another meta-class for their creation. Examples of such classes are pydantic classes. To remedy this, I create the meta-classes `MakeSingleton` and `MakeFinalSingleton` as child classes of the meta-class initially used to create the input `klass` to the singleton decorator. 

To do this I define the `__new__` functions of the `MakeSingleton` and `MakeFinalSingleton` as **separate** functions. I then create `MakeSingleton` and `MakeFinalSingleton` as classes that use these `__new__` functions, and that use `type(klass)` as their parent class, where `klass` is the input class to the singleton decorator. Note that `type(klass)` returns the meta-class that has been used to create `klass`.

```python
# this is the function that will be used as a `__new__` method for `MakeSingleton`
def make_singleton__new__(meta_cls, name, bases, classdict, make_singleton: bool = False):
    old_class = type(meta_cls).__new__(meta_cls, name, bases, classdict)

    # Make the singleton class if make_singleton
    if make_singleton:
        classdict["_old_new"] = old_class.__new__ if "__new__" not in classdict else classdict["__new__"]
        classdict["__new__"] = new_overwrite
        return type(meta_cls).__new__(meta_cls, name, bases, classdict)

    # if not make_singleton, simply forward the __new__/_old_new class from the old_class to the new one.
    old_new = classdict["__new__"] if "__new__" in classdict else getattr(old_class, "_old_new", old_class.__new__)
    classdict["__new__"] = old_new

    return type(meta_cls).__new__(meta_cls, name, bases, classdict)

# This is the function that will be used as the __new__ method for the `MakeFinalSingleton`
def make_final_singleton__new__(meta_cls, name, bases, classdict, make_singleton: bool = True):
    for base in bases:
        if isinstance(base, meta_cls):
            raise TypeError(f"type '{base.__name__}' is not an acceptable base type")

    return super(meta_cls, meta_cls).__new__(meta_cls, name, bases, classdict, make_singleton)


# Create `MakeSingleton` and `MakeFinalSingleton` using the above __new__ functions and usign `type(klass)` as a base class.
def _get_metaclasses(klass: Type):
    MakeSingleton = type("MakeSingleton", (type(klass),), {"__new__": make_singleton__new__})
    MakeFinalSingleton = type("MakeFinalSingleton", (MakeSingleton,), {"__new__": make_final_singleton__new__})

    return MakeSingleton, MakeFinalSingleton
```

We can then write the singleton decorator as,

```python
def singleton(klass=None, /, *, is_final = True):

    def wrapper(klass):
        
        # Dynamically create the Metaclass, so that MetaClass inherit from type(klass)
        MetaClass = _get_metaclasses(klass)[int(is_final)]

        # Create and return the singleton class using on of the MetaClass
        return MetaClass(klass.__name__, (klass,), dict(), make_singleton=True)

    return wrapper(klass) if klass is not None else wrapper
```


That's it! This is (almost) everything there is to know about this decorator. 
There are still some small edge cases but I'll keep that for a follow-up post. But don't worry, 
it does not fundamentally affect the global logic of the implementation I've just described.