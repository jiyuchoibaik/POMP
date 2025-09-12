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

    def __init__(self, data, loader = default_loader):

        self.X_mrna = data['x_mrna']
        self.X_mirna = data['x_mirna']
        self.X_meth = data['x_meth']
        self.region_pixel_5x = data['region_pixel_5x']

        self.max_num_region = 300


    def __getitem__(self, index):

        single_X_mrna = torch.tensor(self.X_mrna[index]).type(torch.FloatTensor)
        single_X_mirna = torch.tensor(self.X_mirna[index]).type(torch.FloatTensor)
        single_X_meth = torch.tensor(self.X_meth[index]).type(torch.FloatTensor)

        dir_patches = self.region_pixel_5x[index]
        regions = np.load(dir_patches)

        if regions.shape[0] > self.max_num_region:
            regions = regions[:self.max_num_region, :, :, :]

        return regions, single_X_mrna, single_X_mirna, single_X_meth

    def __len__(self):
        return len(self.X_mrna)



if __name__ == "__main__":

    pass







