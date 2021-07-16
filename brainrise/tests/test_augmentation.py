# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import unittest
import sys
import logging
import numpy as np
import brainrise


class TestAugmentation(unittest.TestCase):
    """ Test the data augmentation methods.
    """
    def setUp(self):
        """ Setup test.
        """
        logging.basicConfig(stream=sys.stderr)
        self.logger = logging.getLogger("unittest")
        self.logger.setLevel(logging.DEBUG)
        self.transforms = brainrise.get_augmentations()
        compose_transform = brainrise.Compose([
            brainrise.RandomApply([brainrise.RandomFlip(axis=1)], p=0.5),
            brainrise.RandomNoise(sigma=4),
            brainrise.ZScoreNormalize(),
            brainrise.ToTensor()])
        self.transform_params = {
            "Downsample": {"scale": 2},
            "KDENormalize": {"mask": None, "modality": "T1w", "norm_value": 1},
            "Padd": {"shape": [128, 128, 128], "fill_value": 0},
            "RandomAffine": {"rotation": 5, "translation": 0, "zoom": 0.05},
            "RandomBiasField": {"coefficients": 0.5},
            "RandomBlur": {"sigma": 4},
            "RandomDeformation": {"max_displacement": 4, "alpha": 3},
            "RandomFlip": {"axis": 0},
            "RandomGhosting": {
                "n_ghosts": (4, 10), "axis": 2, "intensity": (0.5, 1)},
            "RandomMotion": {
                "rotation": 10, "translation": 10, "n_transforms": 2,
                "perturbation": 0.3},
            "RandomNoise": {"snr": 5., "noise_type": "rician"},
            "RandomOffset": {"factor": (0.05, 0.1)},
            "RandomSpike": {"n_spikes": 1, "intensity": (0.1, 1)},
            "Rescale": {
                "mask": None, "percentiles": (0, 100), "dynamic": (0, 1)},
            "ZScoreNormalize": {"mask": None}
        }
        self.x = np.random.random((64, 64, 64))

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_transforms(self):
        """ Test the transforms.
        """
        for key, kwargs in self.transform_params.items():
            trf = self.transforms[key](**kwargs)
            self.logger.debug(trf)
            y = trf(self.x)


if __name__ == "__main__":

    unittest.main()
