"""
Functional Programming in Python
This module is designed to support the functional programming paradigm.
"""

from collections.abc import Callable
from itertools import islice


class ComposedFunction:
    """
    A class-based replacement for the lambda in compose.
    This makes the composed function picklable for multiprocessing.
    """
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, x):
        # Apply functions from right to left
        result = x
        for f in reversed(self.funcs):
            result = f(result)
        return result


def compose(*funcs: Callable):
    """
    Combines multiple functions from right to left.
    Now returns a picklable ComposedFunction object instead of a lambda.
    """
    return ComposedFunction(*funcs)


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def identity(x):
    """Picklable identity function instead of lambda"""
    return x
