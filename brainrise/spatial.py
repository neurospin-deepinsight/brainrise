# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to transform image.
"""

# Import
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
from .transform import compose
from .transform import gaussian_random_field
from .transform import affine_flow
from .utils import interval
from .utils import Transform


class RandomAffine(Transform):
    """ Random affine transformation.

    The affine translation & rotation parameters are drawn from a lognormal
    distribution - small movements are assumed to occur more often and large
    movements less frequently - or from a uniform distribution.
    """
    def __init__(self, rotation=10, translation=10, zoom=0.2, order=3,
                 dist="uniform", seed=None):
        """ Init class.

        Parameters
        ----------
        rotation: float or 2-uplet, default 10
            the rotation in degrees of the simulated movements. Larger
            values generate more distorted images.
        translation: float or 2-uplet, default 10
            the translation in voxel of the simulated movements. Larger
            values generate more distorted images.
        zoom: float, default 0.2
            the zooming magnitude. Larger values generate more distorted
            images.
        order: int, default 3
            the order of the spline interpolation in the range [0, 5].
        dist: str, default 'uniform'
            the sampling distribution: 'uniform' or 'lognormal'.
        seed: int, default None
            seed to control random number generator.
        """
        self.rotation = interval(rotation)
        self.translation = interval(translation)
        self.zoom = zoom
        self.order = order
        self.dist = dist
        self.seed = seed

    def __call__(self, arr, label_arr=None):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.
        label_arr: array or list of array
            the input label data.

        Returns
        -------
        transformed: array or list of array or 2-uplet
            the transformed input data.
        """
        random_rotations = random_generator(
            self.rotation, self.ndim(arr), dist=self.dist, seed=self.seed)
        random_translations = random_generator(
            self.translation, self.ndim(arr), dist=self.dist, seed=self.seed)
        np.random.seed(self.seed)
        random_zooms = np.random.uniform(
            low=(1 - self.zoom), high=(1 + self.zoom), size=self.ndim(arr))
        random_rotations = Rotation.from_euler(
            "xyz", random_rotations, degrees=True)
        random_rotations = random_rotations.as_matrix()
        affine = compose(random_translations, random_rotations, random_zooms)
        shape = self.shape(arr)
        flow = affine_flow(affine, shape)
        locs = flow.reshape(len(shape), -1)
        transformed = self.apply(
            arr, map_coordinates, locs, order=self.order, cval=0)
        transformed = self.apply(transformed, np.reshape, shape)
        if label_arr is None:
            return transformed
        else:
            transformed_label = self.apply(
                label_arr, map_coordinates, locs, order=0, cval=0)
            transformed_label = self.apply(
                transformed_label, np.reshape, shape)
            return transformed, transformed_label

    def __repr__(self):
        return (
            self.__class__.__name__ +
            "(rotation={0}, translation={1}, zoom={2}, order={3}, "
            "dist={4})".format(
                self.rotation, self.translation, self.zoom, self.order,
                self.dist))


class RandomFlip(Transform):
    """ Apply a random mirror flip.
    """
    def __init__(self, axis=None, seed=None):
        """ Init class.

        Parameters
        ----------
        axis: int, default None
            apply flip on the specified axis. If not specified, randomize the
            flip axis.
        seed: int, default None
            seed to control random number generator.
        """
        self.axis = axis
        self.seed = seed

    def __call__(self, arr, label_arr=None):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.
        label_arr: array or list of array
            the input label data.

        Returns
        -------
        transformed: array or list of array or 2-uplet
            the transformed input data.
        """
        if self.axis is None:
            np.random.seed(self.seed)
            axis = np.random.randint(low=0, high=self.ndim(arr), size=1)[0]
        else:
            axis = self.axis
        if label_arr is None:
            return self.apply(self.apply(arr, np.flip, axis=axis), np.copy)
        else:
            return (self.apply(self.apply(arr, np.flip, axis=axis), np.copy),
                    self.apply(self.apply(label_arr, np.flip, axis=axis),
                               np.copy))

    def __repr__(self):
        return self.__class__.__name__ + "(axis={0})".format(self.axis)


class RandomDeformation(Transform):
    """ Apply dense random elastic deformation.

    Reference: Khanal B, Ayache N, Pennec X., Simulating Longitudinal
    Brain MRIs with Known Volume Changes and Realistic Variations in Image
    Intensity, Front Neurosci, 2017.
    """
    def __init__(self, max_displacement=4, alpha=3, order=3, seed=None):
        """ Init class.

        Parameters
        ----------
        max_displacement: float, default 4
            the maximum displacement in voxel along each dimension. Larger
            values generate more distorted images.
        alpha: float, default 3
            the power of the power-law momentum distribution. Larger values
            genrate smoother fields.
        order: int, default 3
            the order of the spline interpolation in the range [0, 5].
        seed: int, default None
            seed to control random number generator.
        """
        self.max_displacement = max_displacement
        self.alpha = alpha
        self.order = order
        self.seed = seed

    def __call__(self, arr, label_arr=None):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.
        label_arr: array or list of array
            the input label data.

        Returns
        -------
        transformed: array or list of array or 2-uplet
            the transformed input data.
        """
        kwargs = {"seed": self.seed}
        shape = self.shape(arr)
        flow_x = gaussian_random_field(
            shape[:2], alpha=self.alpha, normalize=True, **kwargs)
        flow_x /= flow_x.max()
        flow_x = np.asarray([flow_x] * shape[-1]).transpose(1, 2, 0)
        if self.seed is not None:
            kwargs = {"seed": self.seed + 2}
        flow_y = gaussian_random_field(
            shape[:2], alpha=self.alpha, normalize=True, **kwargs)
        flow_y /= flow_y.max()
        flow_y = np.asarray([flow_y] * shape[-1]).transpose(1, 2, 0)
        if self.seed is not None:
            kwargs = {"seed": self.seed + 4}
        flow_z = gaussian_random_field(
            shape[:2], alpha=self.alpha, normalize=True, **kwargs)
        flow_z /= flow_z.max()
        flow_z = np.asarray([flow_z] * shape[-1]).transpose(1, 2, 0)
        flow = np.asarray([flow_x, flow_y, flow_z])
        flow *= self.max_displacement
        ranges = [np.arange(size) for size in shape]
        locs = np.asarray(np.meshgrid(*ranges)).transpose(0, 2, 1, 3)
        locs = locs.astype(float)
        locs += flow
        locs = locs.reshape(len(locs), -1)
        transformed = self.apply(
            arr, map_coordinates, locs, order=self.order, cval=0)
        transformed = self.apply(transformed, np.reshape, shape)
        if label_arr is None:
            return transformed
        else:
            transformed_label = self.apply(
                label_arr, map_coordinates, locs, order=0, cval=0)
            transformed_label = self.apply(
                transformed_label, np.reshape, shape)
            return transformed, transformed_label

    def __repr__(self):
        return (
            self.__class__.__name__ +
            "(max_displacement={0}, alpha={1}, order={2})".format(
                self.max_displacement, self.alpha, self.order))


class Padd(Transform):
    """ Apply a padding.
    """
    def __init__(self, shape, fill_value=0):
        """ Init class.

        Parameters
        ----------
        shape: list of int
            the desired shape.
        fill_value: int, default 0
            the value used to fill the array.
        """
        self.final_shape = shape
        self.fill_value = fill_value

    def __call__(self, arr, label_arr=None):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.
        label_arr: array or list of array
            the input label data.

        Returns
        -------
        transformed: array or list of array or 2-uplet
            the transformed input data.
        """
        orig_shape = self.shape(arr)
        padding = []
        for orig_i, final_i in zip(orig_shape, self.final_shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append((half_shape_i, half_shape_i))
            else:
                padding.append((half_shape_i, half_shape_i + 1))
        for cnt in range(len(orig_shape) - len(padding)):
            padding.append((0, 0))
        transformed = self.apply(arr, np.pad, padding, mode="constant",
                                 constant_values=self.fill_value)
        if label_arr is None:
            return transformed
        else:
            transformed_label = self.apply(
                label_arr, np.pad, padding, mode="constant",
                constant_values=self.fill_value)
            return transformed, transformed_label

    def __repr__(self):
        return (
            self.__class__.__name__ + "(shape={0}, fill_value={1}".format(
                self.shape, self.fill_value))


class Downsample(Transform):
    """ Apply a downsampling.
    """
    def __init__(self, scale):
        """ Init class.

        Parameters
        ----------
        scale: int or list of int
            the downsampling scale factor in all directions.
        """
        self.scale = scale

    def __call__(self, arr, label_arr=None):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.
        label_arr: array or list of array
            the input label data.

        Returns
        -------
        transformed: array or list of array or 2-uplet
            the transformed input data.
        """
        if not isinstance(self.scale, list):
            self.scale = [self.scale] * self.ndim(arr)
        slices = []
        for cnt, orig_i in enumerate(self.shape(arr)):
            if cnt == 3:
                break
            slices.append(slice(0, orig_i, self.scale[cnt]))
        if label_arr is None:
            return self.apply(arr, "__getitem__", tuple(slices))
        else:
            return (self.apply(arr, "__getitem__", tuple(slices)),
                    self.apply(label_arr, "__getitem__", tuple(slices)))

    def __repr__(self):
        return (
            self.__class__.__name__ + "(scale={0}".format(self.scale))


def random_generator(interval, size, dist="uniform", seed=None):
    """ Random varaible generator.

    Parameters
    ----------
    interval: 2-uplet
        the possible values of the generated random variable.
    size: uplet
        the number of random variables to be drawn from the sampling
        distribution.
    dist: str, default 'uniform'
        the sampling distribution: 'uniform' or 'lognormal'.
    seed: int, default None
        seed to control random number generator.

    Returns
    -------
    random_variables: array
        the generated random variable.
    """
    if dist == "uniform":
        np.random.seed(seed)
        random_variables = np.random.uniform(
            low=interval[0], high=interval[1], size=size)
    # max height occurs at x = exp(mean - sigma**2)
    # FWHM is found by finding the values of x at 1/2 the max height =
    # exp((mean - sigma**2) + sqrt(2*sigma**2*ln(2))) - exp((mean - sigma**2)
    # - sqrt(2*sigma**2*ln(2)))
    elif dist == "lognormal":
        np.random.seed(seed)
        sign = np.random.randint(0, 2, size=size) * 2 - 1
        sign = sign.astype(np.float)
        np.random.seed(seed)
        random_variables = np.random.lognormal(mean=0., sigma=1., size=size)
        random_variables /= 12.5
        random_variables *= (sign * interval[1])
    else:
        raise ValueError("Unsupported sampling distribution.")
    return random_variables
