# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A module with common functions.
"""

# Import
import numbers


class Transform(object):
    """ A base class for transformations.
    """
    @classmethod
    def apply(cls, arr, fct, *args, **kwargs):
        """ Apply transformation to data.

        Parameters
        ----------
        arr: array or list of array
            the input data.
        fct: callable or str
            the transformation function.
        kwargs: dict
            the function parameters.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        if isinstance(fct, str):
            transformed = [
                getattr(_arr, fct)(*args, **kwargs) for _arr in listify(arr)]
        else:
            transformed = [
                fct(_arr, *args, **kwargs) for _arr in listify(arr)]
        return flatten(transformed)

    @classmethod
    def shape(cls, arr):
        """ Return the shape of an array.

        Parameters
        ----------
        arr: array or list of array
            input array.

        Returns
        -------
        shape: tuple of int
            the elements of the shape tuple give the lengths of the
            corresponding array dimensions.
        """
        _arr = listify(arr)[0]
        return _arr.shape

    @classmethod
    def ndim(cls, arr):
        """ Number of array dimensions.

        Parameters
        ----------
        arr: array or list of array
            input array.

        Returns
        -------
        ndim: int
            the array number of dimensions.
        """
        _arr = listify(arr)[0]
        return _arr.ndim

    @classmethod
    def max(cls, arr, axis=None):
        """ Return the maximum along a given axis.

        Parameters
        ----------
        arr: array or list of array
            input array.

        Returns
        -------
        ndim: int
            the array number of dimensions.
        """
        _arr = listify(arr)[0]
        return _arr.max(axis=axis)


def listify(data):
    """ Ensure that the input is a list or tuple.

    Parameters
    ----------
    arr: list or array
        the input data.

    Returns
    -------
    out: list
        the liftify input data.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        return data
    else:
        return [data]


def flatten(data):
    """ Ensure that the list contains more than one element.

    Parameters
    ----------
    arr: list
        the listify input data.

    Returns
    -------
    out: list or array
        the output data.
    """
    if len(data) == 1:
        return data[0]
    else:
        return data


def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")
    return tuple(obj)
