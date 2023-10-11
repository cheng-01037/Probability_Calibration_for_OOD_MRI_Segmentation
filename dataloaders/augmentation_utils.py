"""
implementing intensity transform functions in the format of torch
"""
from os.path import join

import torch
import numpy as np
import torchvision.transforms as deftfx
import dataloaders.image_transforms as myit
import copy
import math

# Standard augmentation for segmenter
aug_config = {
'flip'      : { 'v':False, 'h':False, 't': False, 'p':0.25 },
'affine'    : {
  'rotate':20,
  'shift':(15,15),
  'shear': 20,
  'scale':(0.5, 1.5),
},
'elastic'   : {'alpha':20,'sigma':5}, # medium
'patch': 256,
'reduce_2d': True,
'gamma_range': (0.6, 1.5),
'noise' : {
    'noise_std': 0.15,
    'clip_pm1': False
    },
'bright_contrast': {
    'contrast': (0.60, 1.5),
    'bright': (-1.5,  1.5)
    }

}


# Same transforms as those for training segmenters. More aggresive than usual to reduce degenerate solutions
alea_config = {
'gamma_range': (0.6, 1.5),
'noise' : {
    'noise_std': 0.15,
    'clip_pm1': False
    },
'bright_contrast': {
    'contrast': (0.60, 1.5),
    'bright': (-10.0,  10.0)
    }
}

my_aug = {
    'aug': aug_config
}

alea_aug = {
    'aug': alea_config
        }


def get_geometric_transformer(aug, order=3):
    """order: interpolation degree. Select order=0 for augmenting segmentation """
    # Data augmentation
    affine     = aug['aug'].get('affine', 0)
    alpha      = aug['aug'].get('elastic',{'alpha': 0})['alpha']
    sigma      = aug['aug'].get('elastic',{'sigma': 0})['sigma']
    flip       = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p':0.125})

    tfx = []
    if 'flip' in aug['aug']:
        tfx.append(myit.RandomFlip3D(**flip))

    if 'affine' in aug['aug']:
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso',True),
                                     order=order))

    if 'elastic' in aug['aug']:
        tfx.append(myit.ElasticTransform(alpha, sigma))

    if 'patchify' in aug['aug']:
        tfx.append(Patchify2D(ori_size = aug['aug']['ori_size'],  crop_size = aug['aug']['patch'], bound_pct = aug['aug']['patch_margin']))
    input_transform = deftfx.Compose(tfx)
    return input_transform


class Patchify2D(object):
    def __init__(self, ori_size = 256, crop_size = 192, bound_pct = 0.1):

        """
        x assumed to be nc, nx, ny
        """
        self.nx = ori_size
        self.ny = ori_size
        self.crop_size = crop_size

        self.rx = round(self.nx / 2. * (1 + bound_pct) - crop_size) # range where cropping center can deviate from image center
        self.ry = round(self.ny / 2. * (1 + bound_pct) - crop_size) # range where cropping center can deviate from image center
        self.x_min, self.x_max = round(self.nx / 2. - self.rx), round(self.nx / 2. + self.rx)
        self.y_min, self.y_max = round(self.ny / 2. - self.ry), round(self.ny / 2. + self.ry)

    def __call__(self, x):

        cx = round(np.random.rand() * (self.x_max - self.x_min) + self.x_min)
        cy = round(np.random.rand() * (self.y_max - self.y_min) + self.y_min)


        out = myit.crop_image_at(x, cx, cy, self.crop_size, self.crop_size)

        return out


def get_intensity_transformer(aug):
    """some basic intensity transforms. At least we need to normalize the input anyway"""

    def gamma_tansform(img):
        gamma_range = aug['aug']['gamma_range']
        if isinstance(gamma_range, tuple):
            gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)

            img = img - cmin + 1e-5
            img = irange * np.power(img * 1.0 / irange,  gamma)
            img = img + cmin

        elif gamma_range == False:
            pass
        else:
            raise ValueError("Cannot identify gamma transform range {}".format(gamma_range))
        return img


    def brightness_contrast(img):
        '''
        Chaitanya,   K.,   Karani,   N.,   Baumgartner,   C.F.,   Becker,   A.,   Donati,   O.,863Konukoglu, E., 2019. Semi-Supervised and Task-Driven data augmentation,864in: International Conference on Information Processing in Medical Imaging,865Springer. pp. 29–41.
        '''
        cmin, cmax = aug['aug']['bright_contrast']['contrast']
        bmin, bmax = aug['aug']['bright_contrast']['bright']
        c = np.random.rand() * (cmax - cmin) + cmin
        b = np.random.rand() * (bmax - bmin) + bmin
        img_mean = img.mean()
        img = (img - img_mean) * c + img_mean + b
        return img


    def zm_gaussian_noise(img):
        """
        zero-mean gaussian noise to simulate LDCT
        """
        noise_sigma = aug['aug']['noise']['noise_std']
        noise_vol = np.random.randn(*img.shape) * noise_sigma
        img = img + noise_vol

        if aug['aug']['noise']['clip_pm1']: # if clip to plus-minus 1
            raise Exception('you are not supposed to do this')
            img = np.clip(img, -1.0, 1.0)
        return img

    def compile_transform(img):
        # bright contrast
        if 'bright_contrast' in aug['aug'].keys():
            img = brightness_contrast(img)

        # gamma
        img = gamma_tansform(img) # gamma is by default

        # additive noise
        if 'noise' in aug['aug'].keys():
            img = zm_gaussian_noise(img)

        return img

    return compile_transform

def get_intensity_transformer_torchcuda(aug):
    """the same transform functions but written in torch API
    At least we need to normalize the input anyway"""

    def gamma_tansform(img):
        gamma_range = aug['aug']['gamma_range']
        if isinstance(gamma_range, tuple):
            gamma = torch.rand(1).cuda() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)

            img = img - cmin + 1e-5
            img = irange * torch.pow(img * 1.0 / irange,  gamma)
            img = img + cmin

        elif gamma_range == False:
            pass
        else:
            raise ValueError("Cannot identify gamma transform range {}".format(gamma_range))
        return img


    def brightness_contrast(img):
        '''
        Chaitanya,   K.,   Karani,   N.,   Baumgartner,   C.F.,   Becker,   A.,   Donati,   O.,863Konukoglu, E., 2019. Semi-Supervised and Task-Driven data augmentation,864in: International Conference on Information Processing in Medical Imaging,865Springer. pp. 29–41.
        '''
        cmin, cmax = aug['aug']['bright_contrast']['contrast']
        bmin, bmax = aug['aug']['bright_contrast']['bright']
        c = torch.rand(1).cuda() * (cmax - cmin) + cmin
        b = torch.rand(1).cuda() * (bmax - bmin) + bmin
        img_mean = img.mean()
        img = (img - img_mean) * c + img_mean + b
        return img

    def zm_gaussian_noise(img):
        """
        zero-mean gaussian noise to simulate LDCT
        """
        _, nx, ny = img.shape
        noise_sigma = aug['aug']['noise']['noise_std']
        noise_vol = torch.randn(1, nx, ny ).cuda() * noise_sigma
        img = img + noise_vol

        if aug['aug']['noise']['clip_pm1']: # if clip to plus-minus 1
            raise Exception('you are not supposed to do it!')
            img = torch.clip(img, -1.0, 1.0)
        return img

    def compile_transform(img):
        # bright contrast
        if 'bright_contrast' in aug['aug'].keys():
            img = brightness_contrast(img)

        # gamma
        img = gamma_tansform(img) # gamma is by default

        # additive noise
        if 'noise' in aug['aug'].keys():
            img = zm_gaussian_noise(img)

        return img

    return compile_transform


def transform_with_label(aug, geo_only = False):
    """
    Doing image geometric transform
    Proposed image to have the following configurations
    [H x W x C + CL]
    Where CL is the number of channels for the label. It is NOT a one-hot thing
    """

    # Transform
    geometric_tfx = get_geometric_transformer(aug)
    intensity_tfx = get_intensity_transformer(aug)

    def transform(comp, c_label, c_img, nclass, is_train, use_onehot = False):
        """
        Args
        comp:               a numpy array with shape [H x W x C + c_label]
        c_label:            number of channels for a compact label. Note that the current version only supports 1 slice (H x W x 1)
        nc_onehot:          -1 for not using one-hot representation of mask. otherwise, specify number of classes in the label
        is_train:           whether this is the training set or not. If not, do not perform the geometric transform

        """
        comp = copy.deepcopy(comp)
        if (use_onehot is True) and (c_label != 1):
            raise NotImplementedError("Only allow compact label, also the label can only be 2d")
        assert c_img + 1 == comp.shape[-1], "only allow single slice 2D label"

        # deconcate as image and compact label
        if is_train is True:
            # fixing bug here. Decompose label to onehot, do geo, then compact back
            # _label = comp[..., c_img : ]
            _label = comp[..., c_img ]
            # compact to onehot
            _h_label = np.float32(np.arange( nclass ) == (_label[..., None]) )
            comp = np.concatenate( [comp[...,  :c_img ], _h_label], -1 )
            comp = geometric_tfx(comp)
            # round one_hot labels to 0 or 1
            t_label_h = comp[..., c_img : ]
            t_label_h = np.rint(t_label_h)
            #assert t_label_h.max() <= 1
            t_img = comp[..., 0 : c_img ]

        # intensity transform
        if not geo_only:
            t_img = intensity_tfx(t_img)

        if use_onehot is True:
            t_label = t_label_h
        else:
            t_label = np.expand_dims(np.argmax(t_label_h, axis = -1), -1)
        return t_img, t_label

    return transform

