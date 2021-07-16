# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to change image intensities.
"""

# Import
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
from .transform import compose
from .transform import affine_flow
from .utils import interval
from .utils import Transform
from .hist import get_largest_mode
from .hist import get_last_mode
from .hist import get_first_mode


class RandomOffset(Transform):
    """ Add a random intensity offset (shift and scale).
    """
    def __init__(self, factor, seed=None):
        """ Init class.

        Parameters
        ----------
        arr: array
            the input data.
        factor: float or 2-uplet
            the offset scale factor [0, 1] for the standard deviation
            and the mean.
        seed: int, default None
            seed to control random number generator.
        """
        self.factor = interval(factor, lower=factor)
        self.sigma = interval(factor[0], lower=0)
        self.mean = interval(factor[1])
        self.seed = seed

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        np.random.seed(self.seed)
        sigma_random = np.random.uniform(
            low=self.sigma[0], high=self.sigma[1], size=1)[0]
        np.random.seed(self.seed)
        mean_random = np.random.uniform(
            low=self.mean[0], high=self.mean[1], size=1)[0]
        np.random.seed(self.seed)
        offset = np.random.normal(mean_random, sigma_random, arr.shape)
        offset += 1
        transformed = arr * offset
        return transformed

    def __repr__(self):
        return self.__class__.__name__ + "(sigma={0}, mean={1})".format(
            self.sigma, self.mean)


class RandomBlur(Transform):
    """ Add random blur using a Gaussian filter.
    """
    def __init__(self, snr=None, sigma=None, seed=None):
        """ Init class.

        Parameters
        ----------
        snr: float, default None
            the desired signal-to noise ratio used to infer the standard
            deviation for the noise distribution.
        sigma: float or 2-uplet
            the standard deviation for Gaussian kernel.
        seed: int, default None
            seed to control random number generator.
        """
        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to  "
                             "noise ratio or the standard deviation for the "
                             "noise distribution.")
        if snr is None:
            self.sigma = interval(sigma, lower=0)
        else:
            self.sigma = None
        self.snr = snr
        self.sigma = interval(sigma, lower=0)
        self.seed = seed

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        if self.snr is not None:
            s0 = np.max(arr)
            sigma = s0 / self.snr
            self.sigma = interval(sigma, lower=0)
        np.random.seed(self.seed)
        sigma_random = np.random.uniform(
            low=self.sigma[0], high=self.sigma[1], size=1)[0]
        return gaussian_filter(arr, sigma_random)

    def __repr__(self):
        return self.__class__.__name__ + "(sigma={0})".format(self.sigma)


class RandomNoise(Transform):
    """ Add random Gaussian or Rician noise.

    The noise level can be specified directly by setting the standard
    deviation or the desired signal-to-noise ratio for the Gaussian
    distribution. In the case of Rician noise sigma is the standard deviation
    of the two Gaussian distributions forming the real and imaginary
    components of the Rician noise distribution.

    In anatomical scans, CNR values for GW/WM ranged from 5 to 20 (1.5T and
    3T) for SNR around 40-100 (http://www.pallier.org/pdfs/snr-in-mri.pdf).
    """
    def __init__(self, snr=None, sigma=None, noise_type="gaussian", seed=None):
        """ Init class.

        Parameters
        ----------
        snr: float, default None
            the desired signal-to noise ratio used to infer the standard
            deviation for the noise distribution.
        sigma: float or 2-uplet, default None
            the standard deviation for the noise distribution.
        noise_type: str, default 'gaussian'
            the distribution of added noise - can be either 'gaussian' for
            Gaussian distributed noise, or 'rician' for Rice-distributed noise.
        seed: int, default None
            seed to control random number generator.
        """
        if snr is None and sigma is None:
            raise ValueError("You must define either the desired signal-to "
                             "noise ratio or the standard deviation for the "
                             "noise distribution.")
        if snr is None:
            self.sigma = interval(sigma, lower=0)
        else:
            self.sigma = None
        self.snr = snr
        self.noise_type = noise_type
        self.seed = seed

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        if self.snr is not None:
            s0 = np.max(arr)
            sigma = s0 / self.snr
            self.sigma = interval(sigma, lower=0)
        np.random.seed(self.seed)
        sigma_random = np.random.uniform(
            low=self.sigma[0], high=self.sigma[1], size=1)[0]
        np.random.seed(self.seed)
        noise = np.random.normal(0, sigma_random, [2] + list(arr.shape))
        if self.noise_type == "gaussian":
            transformed = arr + noise[0]
        elif self.noise_type == "rician":
            transformed = np.square(arr + noise[0])
            transformed += np.square(noise[1])
            transformed = np.sqrt(transformed)
        else:
            raise ValueError("Unsupported noise type.")
        return transformed

    def __repr__(self):
        return self.__class__.__name__ + "(type={0}, sigma={1})".format(
            self.noise_type, self.sigma)


class RandomGhosting(Transform):
    """ Add random MRI ghosting artifact.

    Leave first 5% of frequencies untouched.
    """
    def __init__(self, axis, n_ghosts=10, intensity=1, seed=None):
        """ Init class.

        Parameters
        ----------
        axis: int
            the axis along which the ghosts artifact will be created.
        n_ghosts: int or 2-uplet, default 10
            the number of ghosts in the image. Larger values generate more
            distorted images.
        intensity: float or list of float, default 1
            a number between 0 and 1 representing the artifact strength. Larger
            values generate more distorted images.
        seed: int, default None
            seed to control random number generator.
        """
        self.axis = axis
        self.n_ghosts = interval(n_ghosts, lower=0)
        self.intensity = interval(intensity, lower=0)
        self.seed = seed

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        np.random.seed(self.seed)
        n_ghosts_random = np.random.randint(
            low=self.n_ghosts[0], high=self.n_ghosts[1], size=1)[0]
        np.random.seed(self.seed)
        intensity_random = np.random.uniform(
            low=self.intensity[0], high=self.intensity[1], size=1)[0]
        percentage_to_avoid = 0.05
        values = arr.copy()
        slc = [slice(None)] * len(arr.shape)
        for slice_idx in range(values.shape[self.axis]):
            slc[self.axis] = slice_idx
            slice_arr = values[tuple(slc)]
            spectrum = np.fft.fftshift(np.fft.fftn(slice_arr))
            for row_idx, row in enumerate(spectrum):
                if row_idx % n_ghosts_random != 0:
                    continue
                progress = row_idx / arr.shape[0]
                if np.abs(progress - 0.5) < (percentage_to_avoid / 2):
                    continue
                row *= (1 - intensity_random)
            slice_arr *= 0
            slice_arr += np.abs(np.fft.ifftn(np.fft.ifftshift(spectrum)))
        return values

    def __repr__(self):
        return (
            self.__class__.__name__ +
            "(axis={0}, n_ghosts={1}, intensity={2})".format(
                self.axis, self.n_ghosts, self.intensity))


class RandomSpike(Transform):
    """ Add random MRI spike artifacts.
    """
    def __init__(self, n_spikes=1, intensity=(0.1, 1), seed=None):
        """ Init class.

        Parameters
        ----------
        n_spikes: int, default 1
            the number of spikes presnet in k-space. Larger values generate
            more distorted images.
        intensity: float or 2-uplet, default (0.1, 1)
            Ratio between the spike intensity and the maximum of the spectrum.
            Larger values generate more distorted images.
        seed: int, default None
            seed to control random number generator.
        """
        self.n_spikes = n_spikes
        self.intensity = interval(intensity, lower=0)
        self.seed = seed

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        np.random.seed(self.seed)
        spikes_positions = np.random.rand(self.n_spikes)
        np.random.seed(self.seed)
        intensity_factor = np.random.uniform(
            low=self.intensity[0], high=self.intensity[1], size=1)[0]
        spectrum = np.fft.fftshift(np.fft.fftn(arr)).ravel()
        indices = (spikes_positions * len(spectrum)).round().astype(int)
        for index in indices:
            spectrum[index] = spectrum.max() * intensity_factor
        spectrum = spectrum.reshape(arr.shape)
        result = np.abs(np.fft.ifftn(np.fft.ifftshift(spectrum)))
        return result.astype(arr.dtype)

    def __repr__(self):
        return (
            self.__class__.__name__ + "(n_spikes={0}, intensity={1})".format(
                self.n_spikes, self.intensity))


class RandomBiasField(Transform):
    """ Add random MRI bias field artifact.
    """
    def __init__(self, coefficients=0.5, order=3, seed=None):
        """ Init class.

        Parameters
        ----------
        coefficients: float, default 0.5
            the magnitude of polynomial coefficients.
        order: int, default 3
            the order of the basis polynomial functions.
        seed: int, default None
            seed to control random number generator.
        """
        self.coefficients = interval(coefficients)
        self.order = order
        self.seed = seed

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        shape = np.array(self.shape(arr))
        ranges = [np.arange(-size, size) for size in (shape / 2.)]
        bias_field = np.zeros(shape)
        x_mesh, y_mesh, z_mesh = np.asarray(np.meshgrid(*ranges))
        x_mesh /= x_mesh.max()
        y_mesh /= y_mesh.max()
        z_mesh /= z_mesh.max()
        cnt = 0
        np.random.seed(self.seed)
        random_coefficients = np.random.uniform(
            low=self.coefficients[0], high=self.coefficients[1],
            size=(self.order + 1)**3)
        for x_order in range(self.order + 1):
            for y_order in range(self.order + 1 - x_order):
                for z_order in range(self.order + 1 - (x_order + y_order)):
                    random_coefficient = random_coefficients[cnt]
                    new_map = (
                        random_coefficient * x_mesh ** x_order *
                        y_mesh ** y_order * z_mesh ** z_order)
                    bias_field += new_map.transpose(1, 0, 2)
                    cnt += 1
        bias_field = np.exp(bias_field).astype(np.float32)
        return self.apply(arr, np.multiply, bias_field)

    def __repr__(self):
        return (
            self.__class__.__name__ + "(coefficients={0}, order={1})".format(
                self.coefficients, self.order))


class RandomMotion(Transform):
    """ Add random MRI motion artifact on the last axis.

    Reference: Shaw et al., 2019, MRI k-Space Motion Artefact Augmentation:
    Model Robustness and Task-Specific Uncertainty.
    """
    def __init__(self, rotation=10, translation=10, n_transforms=2,
                 perturbation=0.3, axis=None, seed=None):
        """ Init class.

        Parameters
        ----------
        rotation: float or 2-uplet, default 10
            the rotation in degrees of the simulated movements. Larger
            values generate more distorted images.
        translation: floatt or 2-uplet, default 10
            the translation in voxel of the simulated movements. Larger
            values generate more distorted images.
        n_transforms: int, default 2
            the number of simulated movements. Larger values generate more
            distorted images.
        perturbation: float, default 0.3
            control the intervals between movements. If perturbation is 0, time
            intervals between movements are constant.
        axis: int, default None
            the k-space filling axis. If not specified, randomize the k-space
            filling axis.
        seed: int, default None
            seed to control random number generator.
        """
        self.rotation = interval(rotation)
        self.translation = interval(translation)
        self.n_transforms = n_transforms
        self.perturbation = perturbation
        self.axis = axis
        self.seed = seed

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        if self.axis is None:
            np.random.seed(self.seed)
            axis = np.random.randint(low=0, high=arr.ndim, size=1)[0]
        else:
            axis = self.axis
        step = 1. / (self.n_transforms + 1)
        times = np.arange(0, 1, step)[1:]
        shape = arr.shape
        noise = np.random.uniform(
            low=(-step * self.perturbation), high=(step * self.perturbation),
            size=self.n_transforms)
        times += noise
        arrays = [arr]
        np.random.seed(self.seed)
        random_rotations = np.random.uniform(
            low=self.rotation[0], high=self.rotation[1],
            size=(self.n_transforms, arr.ndim))
        np.random.seed(self.seed)
        random_translations = np.random.uniform(
            low=self.translation[0], high=self.translation[1],
            size=(self.n_transforms, arr.ndim))
        for cnt in range(self.n_transforms):
            random_rotations = Rotation.from_euler(
                "xyz", random_rotations[cnt], degrees=True)
            random_rotations = random_rotations.as_matrix()
            zoom = [1, 1, 1]
            affine = compose(random_translations[cnt], random_rotations, zoom)
            flow = affine_flow(affine, shape)
            locs = flow.reshape(len(shape), -1)
            transformed = map_coordinates(arr, locs, order=3, cval=0)
            arrays.append(transformed.reshape(shape))
        spectra = [np.fft.fftshift(np.fft.fftn(array)) for array in arrays]
        n_spectra = len(spectra)
        if np.any(times > 0.5):
            index = np.where(times > 0.5)[0].min()
        else:
            index = n_spectra - 1
        spectra[0], spectra[index] = spectra[index], spectra[0]
        result_spectrum = np.empty_like(spectra[0])
        slc = [slice(None)] * arr.ndim
        slice_size = result_spectrum.shape[axis]
        indices = (slice_size * times).astype(int).tolist()
        indices.append(slice_size)
        start = 0
        for spectrum, end in zip(spectra, indices):
            slc[axis] = slice(start, end)
            result_spectrum[tuple(slc)] = spectrum[tuple(slc)]
            start = end
        result_image = np.abs(np.fft.ifftn(np.fft.ifftshift(result_spectrum)))
        return result_image

    def __repr__(self):
        return (
            self.__class__.__name__ +
            "(rotation={0}, translation={1}, n_transforms={2}, "
            "perturbation={3}, axis={4})".format(
                self.rotation, self.translation, self.n_transforms,
                self.perturbation, self.axis))


class Rescale(Transform):
    """ Performs a rescale of the image intensities to a certain range.
    """
    def __init__(self, mask=None, percentiles=(0, 100), dynamic=(0, 1)):
        """ Init class.

        Parameters
        ----------
        mask: array, default None
            the brain mask.
        percentiles: 2-uplet, default (0, 100)
            percentile values of the input image that will be mapped. This
            parameter can be used for contrast stretching.
        dynamic: 2-uplet, default (0, 1)
            the intensities range of the rescaled data.
        """
        self.mask = mask
        self.percentiles = percentiles
        self.dynamic = dynamic

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        if self.mask is not None:
            values = arr[self.mask]
        else:
            values = arr
        cutoff = np.percentile(values, self.percentiles)
        rescaled = np.clip(arr, *cutoff)
        rescaled -= rescaled.min()  # [0, max]
        array_max = rescaled.max()
        if array_max == 0:
            warnings.warn("Rescaling not possible due to division by zero.")
            return arr
        rescaled /= rescaled.max()  # [0, 1]
        out_range = self.dynamic[1] - self.dynamic[0]
        rescaled *= out_range  # [0, out_range]
        rescaled += self.dynamic[0]  # [out_min, out_max]
        return rescaled

    def __repr__(self):
        return (
            self.__class__.__name__ + "(percentiles={0}, dynamic={1})".format(
                self.percentiles, self.dynamic))


class ZScoreNormalize(Transform):
    """ Performs a batch Z-score normalization.
    """
    def __init__(self, mask=None):
        """ Init class.

        Parameters
        ----------
        mask: array, default None
            the brain mask.
        """
        self.mask = mask

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        if self.mask is not None:
            values = arr[mask == 1]
        else:
            values = arr
        mean = values.mean()
        std = values.std()
        return (arr - mean) / std

    def __repr__(self):
        return self.__class__.__name__ + "()"


class KDENormalize(Transform):
    """ Use kernel density estimation to find the peak of the white
    matter in the histogram of a skull-stripped image. Then normalize
    intensitites to a normalization value.
    """
    def __init__(self, mask=None, modality="T1w", norm_value=1):
        """ Init class.

        Parameters
        ----------
        mask: array, default None
            the brain mask.
        modality str, default 'T1w'
            the modality (T1w, T2w, FLAIR, MD, last, largest, first).
        norm_value: float, default 1
            the new intensity value for the detected WM peak.
        """
        self.mask = mask
        self.modality = modality
        self.norm_value = norm_value

    def __call__(self, arr):
        """ Transform data and return a result of the same type.

        Parameters
        ----------
        arr: array or list of array
            the input data.

        Returns
        -------
        transformed: array or list of array
            the transformed input data.
        """
        return self.apply(arr, self.runtime)

    def runtime(self, arr):
        if self.mask is not None:
            values = arr[self.mask == 1]
        else:
            values = arr[arr > arr.mean()]
        if self.modality.lower() in ["t1w", "flair", "last"]:
            wm_peak = get_last_mode(values)
        elif self.modality.lower() in ["t2w", "largest'"]:
            wm_peak = get_largest_mode(values)
        elif self.modality.lower() in ["md", "first"]:
            wm_peak = get_first_mode(voi)
        else:
            raise ValueError("Invalid modality specified.")
        normalized = (arr / wm_peak) * self.norm_value
        return normalized

    def __repr__(self):
        return (
            self.__class__.__name__ + "(modality={0}, norm_value={1})".format(
                self.modality, self.norm_value))
