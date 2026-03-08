import PIL
import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 933120000


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        try:
            import accimage
            return accimage.Image(path)
        except IOError:
            return pil_loader(path)
    return pil_loader(path)


class POMPDataset(Dataset):
    """
    원본 대비 변경:
      x_mrna / x_mirna / x_meth (각 300-dim)  →  x_rna (n_genes-dim, 기본 2000)
    pkl 키: x_rna, censored, survival, region_pixel_5x
    """

    def __init__(self, data, split, loader=default_loader, max_num_region=250):
        self.max_num_region = max_num_region
        if split == "all":
            self.X_rna  = np.concatenate([data[s]['x_rna']
                                           for s in ("train", "validation", "test")])
            self.censored = np.concatenate([data[s]['censored']
                                             for s in ("train", "validation", "test")])
            self.survival = np.concatenate([data[s]['survival']
                                             for s in ("train", "validation", "test")])
            self.region_pixel_5x = np.concatenate([data[s]['region_pixel_5x']
                                                    for s in ("train", "validation", "test")])
        else:
            self.X_rna           = data[split]['x_rna']           # (N, n_genes)
            self.censored        = data[split]['censored']         # (N,)
            self.survival        = data[split]['survival']         # (N,)
            self.region_pixel_5x = data[split]['region_pixel_5x'] # (N,) str paths

    def __getitem__(self, index):
        single_censored = torch.tensor(self.censored[index]).type(torch.LongTensor)
        single_survival = torch.tensor(self.survival[index]).type(torch.FloatTensor)
        single_X_rna    = torch.tensor(self.X_rna[index]).type(torch.FloatTensor)

        dir_patches = self.region_pixel_5x[index]
        regions     = np.load(dir_patches)

        if regions.shape[0] > self.max_num_region:
            regions = regions[:self.max_num_region, :, :, :]

        # (regions, X_rna, censored, survival)
        return regions, single_X_rna, single_censored, single_survival

    def __len__(self):
        return len(self.X_rna)