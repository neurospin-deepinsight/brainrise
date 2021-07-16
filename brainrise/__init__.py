# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Helper module providing common Brain MRI Data Augmentation methods for PyTorch.
"""

# Imports
import sys
import inspect
from .info import __version__
from .utils import Transform
from .core import Compose, ToTensor, RandomApply
from .intensity import (
    RandomOffset, RandomBlur, RandomNoise, RandomGhosting, RandomSpike,
    RandomBiasField, RandomMotion)
from .intensity import Rescale, ZScoreNormalize, KDENormalize
from .spatial import RandomAffine, RandomFlip, RandomDeformation
from .spatial import Padd, Downsample


def get_augmentations():
    """ Get all available augmentation methods.

    Returns
    -------
    transforms: dict
        a dictionnary containing all declared augmentation methods.
    """
    transforms = {}
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if name == "Transform":
            continue
        if inspect.isclass(obj) and issubclass(obj, Transform):
            transforms[name] = obj
    return transforms
