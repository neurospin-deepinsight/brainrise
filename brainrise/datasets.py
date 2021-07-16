# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Create the MRI Toy dataset.
"""

# Imports
import os
import shutil
import requests
import nibabel
import numpy as np
from torch.utils.data import Dataset


class MRIToyDataset(Dataset):
    """ Create the MRI Toy dataset.
    """
    lesion_url = (
        "https://raw.github.com/muschellij2/open_ms_data/master/"
        "cross_sectional/coregistered_resampled/patient01/consensus_gt.nii.gz")
    t1w_url = (
        "https://raw.github.com/muschellij2/open_ms_data/master/"
        "cross_sectional/coregistered_resampled/patient01/T1W.nii.gz")
    t2w_url = (
        "https://raw.github.com/muschellij2/open_ms_data/master/"
        "cross_sectional/coregistered_resampled/patient01/T2W.nii.gz")
    flair_url = (
        "https://raw.github.com/muschellij2/open_ms_data/master/"
        "cross_sectional/coregistered_resampled/patient01/FLAIR.nii.gz")
    mask_url = (
        "https://raw.github.com/muschellij2/open_ms_data/master/"
        "cross_sectional/coregistered_resampled/patient01/brainmask.nii.gz")

    def __init__(self, root, transform=None):
        """ Init class.

        Parameters
        ----------
        root: str
            root directory of dataset where data will be saved.
        transform: callable, default None
            optional transform to be applied on a sample.
        """
        super(MRIToyDataset).__init__()
        self.root = root
        self.transform = transform
        self.data_file = os.path.join(root, "mritoy.npz")
        self.download()
        self.data = np.load(self.data_file, mmap_mode="r")

    def download(self):
        """ Download data.
        """
        if not os.path.isfile(self.data_file):

            # Fetch data
            dataset = {}
            for name, url in (("t1w", self.t1w_url),
                              ("t2w", self.t2w_url),
                              ("flair", self.flair_url),
                              ("lesion", self.lesion_url),
                              ("mask", self.mask_url)):
                basename = url.split("/")[-1]
                path = os.path.join(self.root, basename)
                if not os.path.isfile(path):
                    print("Downloading {0}.".format(url))
                    response = requests.get(url, stream=True)
                    with open(path, "wb") as out_file:
                        response.raw.decode_content = False
                        shutil.copyfileobj(response.raw, out_file)
                    del response
                dataset[name] = nibabel.load(path).get_data()

            # Save dataset
            np.savez(self.data_file, **dataset)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        input_data = [self.data["t1w"], self.data["t2w"], self.data["flair"]]
        label_data = [self.data["lesion"], self.data["mask"]]
        if self.transform is not None:
            input_data, label_data = self.transform(input_data, label_data)
        return input_data, label_data
