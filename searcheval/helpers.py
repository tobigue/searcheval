# -*- coding: utf-8 -*-
from __future__ import division

try:
    from itertools import izip as zip
    range = xrange
except NameError:
    pass


def cumsum(list1):
    """Return the cumulative sum of the elements at each position of list1."""
    return [sum(list1[:i+1]) for i in range(len(list1))]


def divide(list1, list2):
    """Element-wise divison of two lists."""
    assert len(list1) == len(list2)
    return [e1 / e2 for (e1, e2) in zip(list1, list2)]
