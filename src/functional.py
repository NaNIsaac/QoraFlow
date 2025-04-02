"""
Functional Programming in Python
This module is designed to support the functional programming paradigm.
"""

from collections.abc import Callable
from functools import reduce


def compose(*funcs: Callable):
    """Combines multiple functions from right to left.
    This means the rightmost function is executed first,
    and its result is passed as input to the next function.
    In this way, creates a single function from multiple functions

    compose(h, g, f)(x) is equivalent to h(g(f(x)))
    """
    return reduce(lambda f, g: lambda x: f(g(x)), funcs)
