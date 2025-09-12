# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import PIL
import numpy as np
import pickle
import os
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt



PIL.Image.MAX_IMAGE_PIXELS = 933120000

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

# use PIL Image to read image
def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class POMPDataset(Dataset):

    def __init__(self, data, split, loader = default_loader):

        if split == "all":
            self.X_mrna = np.concatenate((data["train"]['x_mrna'], data["validation"]['x_mrna'], data["test"]['x_mrna']))
            self.X_mirna = np.concatenate((data["train"]['x_mirna'], data["validation"]['x_mirna'], data["test"]['x_mirna']))
            self.X_meth = np.concatenate((data["train"]['x_meth'], data["validation"]['x_meth'], data["test"]['x_meth']))
            self.censored = np.concatenate((data["train"]['censored'], data["validation"]['censored'], data["test"]['censored']))
            self.survival = np.concatenate((data["train"]['survival'], data["validation"]['survival'], data["test"]['survival']))
            self.region_pixel_5x = np.concatenate((data["train"]['region_pixel_5x'], data["validation"]['region_pixel_5x'], data["test"]['region_pixel_5x']))
        else:
            self.X_mrna = data[split]['x_mrna']
            self.X_mirna = data[split]['x_mirna']
            self.X_meth = data[split]['x_meth']
            self.censored = data[split]['censored']
            self.survival = data[split]['survival']
            self.region_pixel_5x = data[split]['region_pixel_5x']

        self.max_num_region = 250

    def __getitem__(self, index):

        single_censored = torch.tensor(self.censored[index]).type(torch.LongTensor)
        single_survival = torch.tensor(self.survival[index]).type(torch.FloatTensor)
        single_X_mrna = torch.tensor(self.X_mrna[index]).type(torch.FloatTensor)
        single_X_mirna = torch.tensor(self.X_mirna[index]).type(torch.FloatTensor)
        single_X_meth = torch.tensor(self.X_meth[index]).type(torch.FloatTensor)

        dir_patches = self.region_pixel_5x[index]
        regions = np.load(dir_patches)

        ###
        # if regions.shape[0] > self.max_num_region:
        #     pixs = []
        #     for region in regions:
        #         pix_sum = np.sum(region)
        #         pixs.append(pix_sum)
        #
        #     ids = sorted(np.argsort(pixs)[0: self.max_num_region])
        #     regions = regions[ids, :, :, :]
        ###

        if regions.shape[0] > self.max_num_region:
            regions = regions[:self.max_num_region, :, :, :]

        return regions, single_X_mrna, single_X_mirna, single_X_meth, \
               single_censored, single_survival

    def __len__(self):
        return len(self.X_mrna)


if __name__ == "__main__":

    pass








