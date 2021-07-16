# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Core transformations dealing with a label target.
"""

# Imports
from inspect import signature
import numpy as np
import torch


class Compose(object):
    """ Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms
        self.n_params = [
            len(signature(trf).parameters) for trf in self.transforms]

    def __call__(self, image, label=None):
        for trf, size in zip(self.transforms, self.n_params):
            if label is not None and size == 2:
                image, label = trf(image, label)
            else:
                image = trf(image)
        if label is None:
            return image
        else:
            return image, label


class ToTensor(object):
    """ Convert a numpy.ndarray to tensor.
    """
    def __call__(self, image, label=None):
        image = torch.as_tensor(np.array(image), dtype=torch.float32)
        if label is None:
            return image
        else:
            label = torch.as_tensor(np.array(label), dtype=torch.int64)
            return image, label


class RandomApply(object):
    """ Apply randomly a list of transformations with a given probability.
    """
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p
        self.n_params = [
            len(signature(trf).parameters) for trf in self.transforms]

    def __call__(self, image, label=None):
        if self.p < torch.rand(1):
            if label is None:
                return image
            else:
                return image, label
        for trf, size in zip(self.transforms, self.n_params):
            if label is not None and size == 2:
                image, label = trf(image, label)
            else:
                image = trf(image)
        if label is None:
            return image
        else:
            return image, label

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for trf in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(trf)
        format_string += "\n)"
        return format_string
