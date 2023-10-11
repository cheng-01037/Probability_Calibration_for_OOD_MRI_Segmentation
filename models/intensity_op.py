import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import dataloaders.augmentation_utils as augutils


class NaiveIntensityWrapper(nn.Module):
    '''
    Photometric transform for estimating aleatoric uncertainty
    '''
    def __init__(self, mode = 'default'):
        super(NaiveIntensityWrapper, self).__init__()
        if mode == 'default':
            self.intensity_augmenter = augutils.get_intensity_transformer_torchcuda(augutils.alea_aug)
        else:
            raise NotImplementedError

    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim = 0) # just to keep the interface the same
        # we can use vector-boardcasting for alea but lets first do things safely and use naive for-loop
        nb, nc, nx, ny = x_in.shape
        out_buffer = torch.stack([self.intensity_augmenter(_img) for _img in x_in  ], dim = 0)
        return out_buffer


