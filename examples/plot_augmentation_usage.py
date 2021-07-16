# -*- coding: utf-8 -*-
"""
Data augmentation usage
=======================

Credit: A Grigis

A simple example on how to use a data augmentation. More specifically,
learn how to use a set of tools to efficiently augment 3D MRI images. It
includes random affine/non linear transformations, simulation of intensity
artifacts due to MRI magnetic field inhomogeneity or k-space motion
artifacts.
"""

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import brainrise
from brainrise.datasets import MRIToyDataset

############################################################################
# Available augmentation methods
# ------------------------------
#
# First list all available augmentation methods.

trfs = brainrise.get_augmentations()
pprint(trfs)

#############################################################################
# Toy MRI dataset
# ---------------
#
# Use the toy MRI dataset.

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(nrows=len(imgs), squeeze=False)
    for idx, img in enumerate(imgs):
        img = img.detach()
        img = transforms.functional.to_pil_image(img)
        axs[idx, 0].imshow(np.asarray(img))
        axs[idx, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

transform = brainrise.Compose([
    brainrise.Rescale(dynamic=(0, 1), percentiles=(5, 97)),
    brainrise.ToTensor()])
dataset = MRIToyDataset(root="/tmp", transform=transform)
dataloader = DataLoader(dataset, batch_size=1)
batch_input, batch_output = next(iter(dataloader))
batch_data = torch.cat((batch_input, batch_output.type(torch.float32)), dim=1)
batch_data = torch.transpose(batch_data, dim0=0, dim1=1)
mid_slice = (batch_data.shape[-1] // 2)
grid = make_grid(batch_data[..., mid_slice], nrow=5)
show(grid)

#############################################################################
# Data augmentation
# -----------------
#
# Perform a simple A/P random flip and an affine transformation + random noise
# augmentations.

imgs = []
transform = brainrise.Compose([
    brainrise.RandomApply([brainrise.RandomFlip(axis=1)], p=0.5),
    brainrise.Rescale(dynamic=(0, 1), percentiles=(5, 97)),
    brainrise.ToTensor()])
dataset = MRIToyDataset(root="/tmp", transform=transform)
dataloader = DataLoader(dataset, batch_size=1)
for epoch in range(5):
    for batch_input, batch_output in dataloader:
        batch_data = torch.cat((
            batch_input, batch_output.type(torch.float32)), dim=1)
        batch_data = torch.transpose(batch_data, dim0=0, dim1=1)
        mid_slice = (batch_data.shape[-1] // 2)
        imgs.append(make_grid(batch_data[..., mid_slice]))
show(imgs)

imgs = []
transform = brainrise.Compose([
    brainrise.RandomApply([brainrise.RandomNoise(snr=20)], p=0.5),
    brainrise.RandomAffine(rotation=3, translation=4, zoom=0.05, order=1),
    brainrise.Rescale(dynamic=(0, 1), percentiles=(5, 97)),
    brainrise.ToTensor()])
dataset = MRIToyDataset(root="/tmp", transform=transform)
dataloader = DataLoader(dataset, batch_size=1)
for epoch in range(5):
    for batch_input, batch_output in dataloader:
        batch_data = torch.cat((
            batch_input, batch_output.type(torch.float32)), dim=1)
        batch_data = torch.transpose(batch_data, dim0=0, dim1=1)
        mid_slice = (batch_data.shape[-1] // 2)
        imgs.append(make_grid(batch_data[..., mid_slice]))
show(imgs)

plt.show()
