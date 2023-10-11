'''
tuner model for LTS
'''
import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import my_utils.util as util
from .base_model import BaseModel
from . import aux_nets as networks
import numpy as np
from .segmenters import*
import os

from .uni_calibration_models import MyLTS, Temperature_Scaling, VanillaLTS
from .calib_utils import MyNLL, MetricsRoi3D
from .intensity_op import NaiveIntensityWrapper

import models.segloss as segloss
import sys

SOFT_COE = 0.1 # prevent nll loss from exploding

class SegmenterNet(BaseModel):
    def name(self):
        return 'SegmenterNet'

    #def set_encoders_and_decoders(self, opt):
    def set_networks(self, opt):
        n_cls = opt.nclass
        self.n_cls = n_cls
        self.gpu_ids = opt.gpu_ids
        if opt.model == 'scale_unet':
            self.netSeg = scale_unet(nclass = n_cls, in_channel = 3, gpu_ids = opt.gpu_ids, scale_base = opt.unet_scale_base)
        else:
            raise NotImplementedError

        if opt.calib_type == 'lts':
            self.netCalib = MyLTS(n_cls, use_clip = opt.use_clip, clip_range = (opt.clip_min, opt.clip_max), res_se = opt.lts_res_se).cuda()
        elif opt.calib_type == 'ts':
            self.netCalib = Temperature_Scaling().cuda()
        elif opt.calib_type == 'vanilla_lts':
            self.netCalib = VanillaLTS(n_cls, use_clip = opt.use_clip, clip_range = (opt.clip_min, opt.clip_max)).cuda()
        else:
            raise NotImplementedError

        if opt.debug_lts_add_aug:
            print("====== DEBUG MODE: ADDITIONAL AUG FOR LTS =====")
            self.img_transform_node = NaiveIntensityWrapper()

        # auxillary nodes for fast evaluations
        self.onehot_node = segloss.One_Hot(n_cls)
        self.softmax_node = torch.nn.Softmax(dim = 1)


    def initialize(self, opt):
        ## load the model. For now we only have segmentation
        BaseModel.initialize(self, opt)
        self.set_networks(opt)

        if opt.continue_train or opt.phase == 'test':
            self.load_network_by_fid(self.netSeg, opt.reload_model_fid)
            if os.path.isfile(opt.reload_calib_fid):
                self.load_network_by_fid(self.netCalib, opt.reload_calib_fid)
            else:
                print('Warning: NOT reloading CALIB model')

        ## define loss functions
        self.criterionDice  = segloss.SoftDiceLoss(self.n_cls).cuda(self.gpu_ids[0]) # dice loss
        self.ScoreDice      = segloss.SoftDiceScore(self.n_cls, ignore_chan0 = True).cuda(self.gpu_ids[0]) # dice score
        self.ScoreAllEval   = segloss.Efficient_AllScore(self.n_cls, ignore_chan0 = False).cuda(self.gpu_ids[0])
        self.criterionWCE = segloss.My_CE(nclass = self.n_cls,\
                batch_size = self.opt.batchSize, weight = torch.ones(self.n_cls,)).cuda(self.gpu_ids[0])

        self.criterionMyNLL = MyNLL(nclass = self.n_cls).cuda()
        self.dilate_calib_metric_wrapper = MetricsRoi3D()
        self.bypass_bound =  True

        # initialize optimizers
        if self.opt.optimizer == 'adam':
            self.optimizer_Calib = torch.optim.Adam( itertools.chain(self.netCalib.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = opt.adam_weight_decay)
        else:
            raise NotImplementedError
        print(f'OPTIM: using optimizer: {opt.optimizer}')

        # register optimizers
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_Calib)

        # put optimizers into learning rate schedulers
        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netSeg)
        networks.print_network(self.netCalib)

        # register subnets
        self.subnets = [ self.netSeg, self.netCalib ]

        for subnet in self.subnets:
            assert next(subnet.parameters()).is_cuda == True
        print('-----------------------------------------------')

    def set_input(self, input, *args, **kwargs):
        input_img = input['img']
        input_mask = input['lb']

        if len(self.gpu_ids) > 0:
            # check and convert input to a float tensor
            # with shape [nb, nc, nx, ny]
            # send to GPU
            if not isinstance(input_img, torch.FloatTensor):
                if input_img.ndims < 4:
                    input_img = input_img[np.newaxis, ...]
                input_img = torch.FloatTensor(input_img, requires_grad = False).float()
            # input_img = input_img.cuda(self.gpu_ids[0], async=True)
            input_img = input_img.cuda()

            # do the same for mask
            if not isinstance(input_mask, torch.FloatTensor):
                if input_mask.ndims < 4:
                    input_mask = input_mask[np.newaxis, ...]
                input_mask = torch.FloatTensor(input_mask, requires_grad = False).float()
            # input_mask = input_mask.cuda(self.gpu_ids[0], async=True)
            input_mask = input_mask.cuda()

        self.input_img = Variable(input_img)
        self.input_mask = input_mask
        self._nb_current = input_img.shape[0] # batch size of the current batch

    def set_input_add_aug(self, input, *args, **kwargs):
        self.set_input(input)
        self.input_img = self.img_transform_node(self.input_img)

    # run validation
    def validate(self): # used to be called test
        # change everything to eval
        # NOTE: we still keep the name of seen and unseen as in future we need it
        for subnet in self.subnets:
            subnet.eval()

        with torch.no_grad():
            img_val       = self.input_img
            mask_val      = self.input_mask
            pred_val_uncalib, aux_pred   = self.netSeg(img_val)
            temp_val = self.netCalib(pred_val_uncalib, img_val)
            pred_val_calib = pred_val_uncalib / (temp_val + SOFT_COE )

            loss_dice_val = self.ScoreDice(pred_val_calib, mask_val)
            loss_calib_val  = self.criterionMyNLL(pred_val_calib, mask_val, False, bypass_bound = self.bypass_bound)

            self.loss_dice_val = loss_dice_val.data
            self.loss_calib_val  = loss_calib_val.data

            self.pred_val_calib = pred_val_calib.data
            self.gth_val  = mask_val.data
            self.temp_val = temp_val.data

        for subnet in self.subnets:
            subnet.zero_grad()
            subnet.train()


    def get_calib_gpu(self):
        for subnet in self.subnets:
            subnet.eval()

        with torch.no_grad():
            img_val         = self.input_img
            mask_val        = self.input_mask
            seg_val, _      = self.netSeg(img_val)

            seg_out = torch.argmax(seg_val, 1)

            temp = self.netCalib(seg_val, img_val)
            seg_val = seg_val / (temp + SOFT_COE)

            seg_out = torch.argmax(seg_val, dim = 1)

            for subnet in self.subnets:
                subnet.zero_grad()
                subnet.train()

        return mask_val, seg_out, seg_val, temp

    def forward_calib_tune(self, input_img):
        """
        run a forward segmentation in training mode
        """
        lambda_Seg  = self.opt.lambda_Seg
        lambda_wce  = self.opt.lambda_wce
        lambda_dice = self.opt.lambda_dice

        self.set_requires_grad(self.netSeg, False)
        self.netSeg.eval()
        self.netCalib.train()

        lambda_calib            = self.opt.lambda_calib
        pred_uncalib, aux_pred  = self.netSeg(input_img)
        temp_calib              = self.netCalib(pred_uncalib, input_img)
        self.temp_mean_tr = temp_calib.mean().data
        self.temp_min_tr = temp_calib.min().data
        self.temp_max_tr = temp_calib.max().data

        pred_calib              = pred_uncalib / (temp_calib + SOFT_COE)
        self.temp_calib_tr      = temp_calib.data

        loss_calib   = self.criterionMyNLL(pred_calib, self.input_mask, False, bypass_bound = self.bypass_bound)

        pred_calib_tr = pred_calib[: self._nb_current]

        assert lambda_dice == 0 and lambda_wce == 0 # they are not in use
        loss_dice   = self.criterionDice(input = pred_calib_tr, target = self.input_mask)
        loss_wce    = self.criterionWCE(inputs = pred_calib_tr, targets = self.input_mask.long() )

        self.seg_tr         = pred_calib.detach()
        self.loss_calib     = loss_calib * lambda_calib + (loss_dice * lambda_dice + loss_wce * lambda_wce) * lambda_Seg
        self.loss_calib_tr  = loss_calib.data
        self.loss_wce_tr    = loss_wce.data

        return pred_calib, pred_uncalib, temp_calib


    def optimize_parameters(self, **kwargs):
        self.set_requires_grad(self.subnets, False)
        self.set_requires_grad(self.netCalib, True)

        pred_calib, pred_uncalib, temp_calib = self.forward_calib_tune(self.input_img)
        self.optimizer_Calib.zero_grad()
        self.loss_calib.backward()
        self.optimizer_Calib.step()

        self.set_requires_grad(self.subnets, False) # just in case


    def get_current_errors_tr(self):
        ret_errors = [ ('loss_calib_tr', self.loss_calib_tr),\
                ]

        ret_errors = OrderedDict(ret_errors)
        return ret_errors

    def get_current_errors_val(self):
        ret_errors = [ ('loss_calib_val', self.loss_calib_val.mean())]
        ret_errors = OrderedDict(ret_errors)
        return ret_errors

    def get_current_visuals_val(self):
        """
        """
        img_val    = t2n(self.input_img.data)
        gth_val    = t2n(self.gth_val.data)
        pred_val   = t2n( torch.argmax(self.pred_val_calib.data, dim =1, keepdim = True ))
        temp_val   = t2n( self.temp_val.data )

        ret_visuals = [\
                ('img_seen_val', img_val),\
                ('pred_val', pred_val * 1.0 / (self.n_cls + 0.01 )),\
                ('gth_val', gth_val * 1.0 / (self.n_cls + 0.01 ))
                ]

        return OrderedDict(ret_visuals)

    # detech visualization for validation from these for training
    def get_current_visuals_tr(self):
        img_tr  = t2n( to01(self.input_img.data, True))
        pred_tr = t2n( torch.argmax(self.seg_tr.data, dim =1, keepdim = True )  )
        gth_tr  = t2n(self.input_mask.data )
        ret_visuals = [\
                ('gth_seen_tr', (gth_tr + 0.01) * 1.0 / (self.n_cls + 0.01 )), \
                ]

        return OrderedDict(ret_visuals)

    def plot_image_in_tb(self, writer, result_dict, epoch = None):
        for key, img in result_dict.items():

            if epoch is not None:
                writer.add_image(key, img, epoch)
            else:
                writer.add_image(key, img)

    def track_scalar_in_tb(self, writer, result_dict, which_iter):
        for key, val in result_dict.items():
            writer.add_scalar(key, val, which_iter)

    # NOTE: remeber to modify this when expanding the model
    def save(self, label):
        self.save_network(self.netSeg, 'Seg', label, self.gpu_ids)
        self.save_network(self.netCalib, 'Calib', label, self.gpu_ids)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def t2n(x):
    if isinstance(x, np.ndarray):
        return x
    if x.is_cuda:
        x = x.data.cpu()
    else:
        x = x.data

    return np.float32(x.numpy())

def to01(x, by_channel = False):
    if not by_channel:
        out = (x - x.min()) / (x.max() - x.min())
    else:
        nb, nc, nh, nw = x.shape
        xmin = x.view(nb, nc, -1).min(dim = -1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,nh, nw)
        xmax = x.view(nb, nc, -1).max(dim = -1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,nh, nw)
        out = (x - xmin + 1e-5) / (xmax - xmin + 1e-5)

    return out

