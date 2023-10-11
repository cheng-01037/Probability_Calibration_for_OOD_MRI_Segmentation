# Clean ACDC dataset
import glob
import numpy as np
import itertools
import dataloaders.niftiio as nio
import dataloaders.augmentation_utils as augutils
# import acdcutils as acdc
import torch
import random
import os
import copy
import platform
import json
import torch.utils.data as torch_data
import dataloaders.data_cfg_acdc as acdc_conf
# from common import BaseDataset, Subset
from pdb import set_trace
###### Configs for data path and selective loading ######

import platform
hostname = platform.node()
BASEDIR='./data/MyACDC/'
LABEL_NAME = ["bg", "lv", "myo", "rv"]

#TODO: merge ACDC and ACDC-C dataset
class ACDCDataset(torch_data.Dataset):
    def __init__(self, idx_split, mode, transforms, base_dir, domain = 'clean', tile_z_dim = 3, cframes = ['ES'], reps = ['0']):
        """
        Clean ACDC
        Args:
            mode:       : 'train' for training the base segmenter, 'tuneval' for fine-tuning the calibrator on the validation data
            idx_split   : index of data split for repeating the experiments: 0, 1, and 2
            tile_z_dim  : number of tile along z dimension. As we are intializing with VGG but using only one slice
                        this seems to be good solution
        """
        super(ACDCDataset, self).__init__()
        self.transforms=transforms
        self.is_train = False
        if mode == 'train' or mode == 'tuneval':
            self.is_train = True
        self.all_label_names = LABEL_NAME
        self.nclass = len(LABEL_NAME)
        self.tile_z_dim = tile_z_dim
        self._base_dir = base_dir
        self.cframes = cframes
        self.domain = domain
        self.reps = reps
        self.scan_ids = self.__get_scanids(mode, idx_split) # patient ids of the entire fold

        self.info_by_scan = None
        self.sample_list = self.__search_samples() # information of scans of the entire fold
        self.pid_curr_load = self.scan_ids
        self.actual_dataset = self.__read_dataset()
        self.size = len(self.actual_dataset)

    def __search_samples(self):
        """ search for filenames """
        out_list = {}
        for curr_id in self.scan_ids:
            out_list[curr_id] = {
                    "img_fid": os.path.join(self._base_dir, f'image_{curr_id}.nii.gz'),
                    "lbs_fid": os.path.join(self._base_dir, f'label_{curr_id}.nii.gz')
                }

        return out_list

    def __get_scanids(self, mode, idx_split):
        """
        60 - 20 - 20 split
        """
        test_ids    = set( str(int(ii)) for ii in acdc_conf.test_data()  )
        raw_ids     = set( itm.split('_')[-1].split('-')[0] for itm in os.listdir(BASEDIR) )
        other_ids   = sorted( raw_ids - test_ids , key = lambda x: int(x) )

        raw_ids = list(raw_ids)
        test_ids = list(test_ids)

        tr_ids = other_ids[:60] # the pool of testing data
        val_ids = other_ids[60:]

        self.subset_fold = idx_split
        tr_ids = tr_ids[20 * self.subset_fold : 20 * (self.subset_fold + 1) ]
        val_ids = val_ids[5 * self.subset_fold : 5 * (self.subset_fold + 1)]

        if mode == 'val' or mode == 'tuneval':
            curr_ids = val_ids
        elif mode == 'test':
            curr_ids = test_ids
        elif mode == 'train':
            curr_ids = tr_ids
        else:
            raise ValueError

        full_id_buffer = []
        for fra in self.cframes:
            for rep in self.reps:
                full_id_buffer += [ pid + '-' + fra + '-' + rep for pid in curr_ids    ]

        print(f'mode {mode}:')
        print(full_id_buffer)
        return full_id_buffer

    def __read_dataset(self):
        """
        Build index pointers to individual slices
        Also keep a look-up table from scan_id, slice to index
        """
        out_list = []
        self.scan_z_idx = {}
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset

        for scan_id, itm in self.sample_list.items():
            img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
            self.info_by_scan[ self.domain + '_' + scan_id] = _info

            img = np.float32(img)

            img = (img - img.mean()) / img.std()
            self.scan_z_idx[scan_id] = [-1 for _ in range(img.shape[-1])]
            lb = nio.read_nii_bysitk(itm["lbs_fid"])
            lb = np.float32(lb)

            img     = np.transpose(img, (1,2,0))
            lb      = np.transpose(lb, (1,2,0))

            assert img.shape[-1] == lb.shape[-1]
            base_idx = img.shape[-1] // 2 # index of the middle slice
            nframe = img.shape[-1]

            # now start writing everthing in
            # write the beginning frame
            out_list.append( {"img": img[..., 0: 1],
                           "lb":lb[..., 0: 0 + 1],
                           "is_start": True,
                           "is_end": False,
                           "nframe": nframe,
                           "domain": self.domain,
                           "scan_id": self.domain + "_" + scan_id,
                           "z_id":0})

            self.scan_z_idx[scan_id][0] = glb_idx
            glb_idx += 1

            for ii in range(1, img.shape[-1] - 1):
                out_list.append( {"img": img[..., ii: ii + 1],
                           "lb":lb[..., ii: ii + 1],
                           "is_start": False,
                           "is_end": False,
                           "domain": self.domain,
                           "nframe": nframe,
                           "scan_id": self.domain + "_" + scan_id,
                           "z_id": ii
                           })
                self.scan_z_idx[scan_id][ii] = glb_idx
                glb_idx += 1

            ii += 1 # last frame, note the is_end flag
            out_list.append( {"img": img[..., ii: ii + 1],
                           "lb":lb[..., ii: ii+ 1],
                           "is_start": False,
                           "is_end": True,
                           "nframe": nframe,
                           "domain": self.domain,
                           "scan_id": self.domain + "_" + scan_id,
                           "z_id": ii
                           })

            self.scan_z_idx[scan_id][ii] = glb_idx
            glb_idx += 1

        return out_list

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]
        if self.is_train is True:
            comp = np.concatenate( [curr_dict["img"], curr_dict["lb"]], axis = -1 )
            if self.transforms:
                img, lb = self.transforms(comp, c_img = 1, c_label = 1, nclass = self.nclass, is_train = self.is_train, use_onehot = False)
        else:
            img = curr_dict['img']
            lb = curr_dict['lb']

        img = np.float32(img)
        lb = np.float32(lb)

        img = np.transpose(img, (2, 0, 1))
        lb  = np.transpose(lb, (2, 0, 1))

        img = torch.from_numpy( img )
        lb  = torch.from_numpy( lb )

        if self.tile_z_dim:
            img = img.repeat( [ self.tile_z_dim, 1, 1] )
            assert img.ndimension() == 3, f'actual dim {img.ndimension()}'

        is_start = curr_dict["is_start"]
        is_end = curr_dict["is_end"]
        nframe = np.int32(curr_dict["nframe"])
        scan_id = curr_dict["scan_id"]
        z_id    = curr_dict["z_id"]

        sample = {"img": img,
                "lb":lb,
                "sup": lb, # dummy variable. We actually do not have sup for that
                "is_start": is_start,
                "is_end": is_end,
                "nframe": nframe,
                "scan_id": scan_id,
                "z_id": z_id
                }
        return sample

    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        return len(self.actual_dataset)

augment_func  = augutils.transform_with_label(augutils.my_aug)
def get_training(idx_split, tile_z_dim = 3, cframes = ['ES'], reps = ['0']):
    return ACDCDataset(idx_split = idx_split,\
        mode = 'train',\
        cframes = cframes,\
        reps = reps,\
        transforms = augment_func,\
        base_dir = BASEDIR,\
        tile_z_dim = tile_z_dim)

def get_validation(idx_split, tile_z_dim = 3, cframes = ['ES'], reps = ['0']):
    return ACDCDataset(idx_split = idx_split,\
        mode = 'val',\
        cframes = cframes,\
        reps = reps,\
        transforms = None,\
        base_dir = BASEDIR,\
        tile_z_dim = tile_z_dim)

def get_tuneval(idx_split, tile_z_dim = 3, cframes = ['ES'], reps = ['0']):
    """
    Tuning the calibrator using the validation fold
    """
    return ACDCDataset(idx_split = idx_split,\
        mode = 'tuneval',\
        cframes = cframes,\
        reps = reps,\
        transforms = augment_func,\
        base_dir = BASEDIR,\
        tile_z_dim = tile_z_dim)

def get_test(idx_split, tile_z_dim = 3, cframes = ['ES'], reps = ['0']):
     return ACDCDataset(idx_split = idx_split,\
        mode = 'test',\
        transforms = None,\
        cframes = cframes,\
        reps = reps,\
        base_dir = BASEDIR,\
        tile_z_dim = tile_z_dim)
