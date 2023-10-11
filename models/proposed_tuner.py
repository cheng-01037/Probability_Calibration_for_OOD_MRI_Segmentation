'''
tuner model for the proposed method
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

from .intensity_op import NaiveIntensityWrapper

from .calib_utils import MyNLL, MetricsRoi3D
from .uni_calibration_models import OODCalibNet
from .unets import ScaleUNetDrop

import models.segloss as segloss
import sys

SOFT_COE = 0.1 # prevent nll loss from exploding


class SeqTrans(object):
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, x):
        for func in self.funcs:
            x = func(x)
        return x

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

        if opt.calib_type == 'proposed':
            self.netCalib = OODCalibNet(n_cls, opt.use_clip, [opt.clip_min, opt.clip_max], res_se = opt.lts_res_se ).cuda()
            self.set_input_aug_sup = self.alts_set_input_alea
        else:
            raise NotImplementedError
        # construct the DAE
        if opt.dae_mode == 'unet':
            dae_fn = ScaleUNetDrop # a small unet as DAE with heavy dropout at the front
        else:
            raise Exception
        self.netDAE = dae_fn(n_cls, n_cls, scale_base = opt.dae_scale_base, drop_p = opt.dae_enc_drop_p, bilinear = opt.dae_bilinear, use_leaky = True).cuda()

        # auxillary nodes for fast evaluations
        self.onehot_node = segloss.One_Hot(n_cls)
        self.softmax_node = torch.nn.Softmax(dim = 1)

        # aleatoric uncertainty estimation. We have also added similar further augmentation to lts but it does not help
        self.img_transform_node = NaiveIntensityWrapper() # does not include artifacts in test data

    def initialize(self, opt):
        ## load the model. For now we only have segmentation
        BaseModel.initialize(self, opt)
        self.set_networks(opt)

        if opt.continue_train or opt.phase == 'test':
            assert os.path.isfile(opt.reload_model_fid)
            self.load_network_by_fid(self.netSeg, opt.reload_model_fid)
            if os.path.isfile(opt.reload_calib_fid):
                self.load_network_by_fid(self.netCalib, opt.reload_calib_fid)
            else:
                print('Warning: NOT reloading CALIB model')
            if os.path.isfile(opt.reload_dae_fid):
                self.load_network_by_fid(self.netDAE, opt.reload_dae_fid)
            else:
                print('Warning: NOT reloading DAE model for shape prior')

        ## define loss functions
        self.criterionDice  = segloss.SoftDiceLoss(self.n_cls).cuda(self.gpu_ids[0]) # dice loss
        self.ScoreDice      = segloss.SoftDiceScore(self.n_cls, ignore_chan0 = True).cuda(self.gpu_ids[0]) # dice score
        self.ScoreAllEval   = segloss.Efficient_AllScore(self.n_cls, ignore_chan0 = False).cuda(self.gpu_ids[0])
        self.criterionWCE   = segloss.My_CE(nclass = self.n_cls,\
                batch_size  = self.opt.batchSize, weight = torch.ones(self.n_cls,)).cuda(self.gpu_ids[0])
        self.criterionL1 = torch.nn.L1Loss().cuda(self.gpu_ids[0]) # for DAE validation
        self.criterionMyNLL = MyNLL(nclass = self.n_cls).cuda()
        self.dilate_calib_metric_wrapper = MetricsRoi3D()
        self.bypass_bound =  True

        # initialize optimizers
        if self.opt.optimizer == 'adam':
            self.optimizer_Calib = torch.optim.Adam( itertools.chain(self.netCalib.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = opt.adam_weight_decay)
            self.optimizer_DAE = torch.optim.Adam( itertools.chain(self.netDAE.parameters()), lr=opt.dae_lr, betas=(opt.beta1, 0.999), weight_decay = opt.adam_weight_decay)
        else:
            raise NotImplementedError
        print(f'OPTIM: using optimizer: {opt.optimizer}')

        # register optimizers
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_Calib)
        self.optimizers.append(self.optimizer_DAE)

        # put optimizers into learning rate schedulers
        for optimizer in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netSeg)
        networks.print_network(self.netCalib)
        networks.print_network(self.netDAE)

        # register subnets
        self.subnets = [ self.netSeg, self.netCalib, self.netDAE ]

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
        self.input_img_3copy = input_img # NOTE: dummy variable. This is not used for training

    def alts_set_input_alea(self, input):
        '''
        for the ease of aleatoric uncertainty computation
        '''
        input_img   = input['img']
        input_mask  = input['lb']

        if len(self.gpu_ids) > 0:
            # check and convert input to a float tensor
            # with shape [nb, nc, nx, ny]
            # input_img = input_img.float().cuda(self.gpu_ids[0], async=True)
            input_img = input_img.float().cuda()

            # do the same for mask
            # input_mask = input_mask.float().cuda(self.gpu_ids[0], async=True)
            input_mask = input_mask.float().cuda()

        # random non-linear augmentation
        self._nb_distinct = input_img.shape[0] # batch size of the current batch
        self._n_rep = self.opt.alea_n_rep
        self._nb_current = self._nb_distinct * self._n_rep
        assert self._n_rep % 3 == 0

        input_buffer = self.img_transform_node(input_img.repeat(self._n_rep, 1, 1, 1))
        self.input_img_3copy = input_buffer
        self.input_mask = input_mask

    def __instant_alea_aug(self, input_img, te_stable = False):
        """
        Augmentation for aleatoric uncertainty, on the fly, assuming data has a batch size of 1
        """
        # FIXME: fix the replication mechanism
        assert input_img.shape[0] == 1, "For now only we support batchsize of 1"
        self._nb_distinct = input_img.shape[0] # batch size of the current batch
        local_n_rep = self.opt.alea_n_rep
        if te_stable == True:
            if self.opt.te_reps > 0:
                local_n_rep = self.opt.te_reps * 3
            else:
                local_n_rep *= 3 # use a larger number to stablize testing

        self._nb_current = self._nb_distinct * local_n_rep
        assert local_n_rep % 3 == 0

        input_img = input_img.repeat(self._nb_current, 1, 1, 1)
        input_buffer = self.img_transform_node(input_img)

        return input_buffer

    # run validation. Still with aleatoric uncertainty
    def validate(self): # used to be called test
        for subnet in self.subnets:
            subnet.eval()

        with torch.no_grad():
            img_val       = self.input_img
            mask_val      = self.input_mask
            pred_val_uncalib, aux_pred   = self.netSeg(img_val)

            prob_val_uncalib = F.softmax(pred_val_uncalib, dim = 1)

            denoised_prob = F.softmax(self.netDAE( pred_val_uncalib ), dim = 1)
            diff_denoised = denoised_prob - prob_val_uncalib

            # aleatoric
            augs = self.__instant_alea_aug(img_val)
            alea_mean, alea_std = self.get_alea( self.netSeg(augs)[0] )
            logits_comp = torch.cat([pred_val_uncalib, alea_mean, alea_std] , dim = 1  )

            temp_val = self.netCalib(logits_comp, img_val, diff_denoised)
            pred_val_calib = pred_val_uncalib / (temp_val + SOFT_COE )


            diff_recon_val = self.criterionL1( denoised_prob, self.onehot_node( mask_val  )  )
            loss_dice_val = self.ScoreDice(pred_val_calib, mask_val)
            loss_calib_val  = self.criterionMyNLL(pred_val_calib, mask_val, False, bypass_bound = self.bypass_bound)

            self.loss_dice_val      = loss_dice_val.data
            self.loss_calib_val     = loss_calib_val.data
            self.diff_recon_val     = diff_recon_val.data # not the same as the loss: for the loss we use cross entropy instead

            self.pred_val_calib = pred_val_calib.data
            self.gth_val  = mask_val.data
            self.temp_val = temp_val.data

            self.diff_denoised_val = torch.abs(diff_denoised).sum(dim = 1, keepdim = True)

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

            prob_uncalib = F.softmax(seg_val, dim = 1)

            #denoised_prob = F.softmax(self.netDAE( prob_uncalib ), dim = 1)
            denoised_prob = F.softmax(self.netDAE( seg_val ), dim = 1)
            diff_denoised = denoised_prob - prob_uncalib

            augs = self.__instant_alea_aug(img_val, te_stable = True)
            alea_mean, alea_std = self.get_alea( self.netSeg(augs)[0] )
            logits_comp = torch.cat([seg_val, alea_mean, alea_std] , dim = 1  )
            temp = self.netCalib(logits_comp, img_val, diff_denoised)

            seg_val = seg_val / (temp + SOFT_COE)
            seg_out = torch.argmax(seg_val, dim = 1)

            for subnet in self.subnets:
                subnet.zero_grad()
                subnet.train()

            return mask_val, seg_out, seg_val, temp, alea_std

    def forward_dae_train(self, input_img, dae_has_grad, copy_mode = True):
        """
        Train the DAE
        """

        self.set_requires_grad(self.netSeg, False)
        self.set_requires_grad(self.netCalib, False)
        self.set_requires_grad(self.netDAE, dae_has_grad)

        self.netSeg.eval()
        self.netCalib.eval()
        if dae_has_grad:
            self.netDAE.train()
        else:
            self.netDAE.eval()

        lambda_dae  = self.opt.lambda_dae

        uncalib_logits, aux_pred     = self.netSeg(input_img)

        prob_uncalib = F.softmax(uncalib_logits, dim = 1)
        denoised_logits = self.netDAE( uncalib_logits )

        denoised_prob = F.softmax(denoised_logits, dim = 1)
        diff_denoised = denoised_prob - prob_uncalib

        if copy_mode: # as we are repeating input images
            loss_dae = self.criterionWCE(inputs = denoised_logits, targets = self.input_mask.repeat(self.opt.alea_n_rep, 1, 1, 1).long()) # for convenience, use WCE
        else:
            loss_dae = self.criterionWCE(inputs = denoised_logits, targets = self.input_mask.long()) # for convenience, use WCE

        self.loss_dae_tr = loss_dae.data
        self.loss_dae = loss_dae * lambda_dae
        self.denoised_diff_tr = torch.abs(diff_denoised).sum(dim = 1, keepdim = True)
        self.denoised_prob_tr = denoised_prob.data

        return uncalib_logits, diff_denoised

    def get_alea(self, input_ft):
        """
        compute aleatoric uncertainty (mean and var)
        """
        _nb         = input_ft.shape[0]
        input_ft    = input_ft.clone().detach() # detach from training loop
        raw_probs   = F.softmax(input_ft, dim = 1)

        meanmap     = raw_probs.mean(dim = 0, keepdim = True) # 1, nc, nh, nw
        _ctr_probs  = raw_probs - meanmap.repeat(_nb, 1 ,1 ,1) # nr, nc, nh, hw
        std_map     = (torch.sum(_ctr_probs ** 2, dim = 0, keepdim = True) /  _nb ) ** 0.5 # 1, nc, nh, nw

        return meanmap, std_map

    def forward_calib_alea_tune(self, input_img, pred_uncalib, diff_denoised):
        lambda_Seg  = self.opt.lambda_Seg
        lambda_wce  = self.opt.lambda_wce
        lambda_dice = self.opt.lambda_dice

        self.set_requires_grad(self.netSeg, False)
        self.set_requires_grad(self.netDAE, False)
        self.set_requires_grad(self.netCalib, True)
        self.netSeg.eval()
        self.netDAE.eval()
        self.netCalib.train()

        lambda_calib            = self.opt.lambda_calib
        _nb = input_img.shape[0]
        _nsplit = _nb // 3 # only use gradient of 1/3 of augmented samples
        meanmap, stdmap         = self.get_alea(pred_uncalib)
        logits_comp             = torch.cat([pred_uncalib[:_nsplit], meanmap.repeat(_nsplit, 1, 1, 1), stdmap.repeat(_nsplit, 1, 1 ,1)] , dim = 1  )

        temp_calib              = self.netCalib(logits_comp, input_img[:_nsplit], diff_denoised[:_nsplit])
        self.temp_mean_tr = temp_calib.mean().data
        self.temp_min_tr = temp_calib.min().data
        self.temp_max_tr = temp_calib.max().data

        pred_calib              = pred_uncalib[: _nsplit] / (temp_calib + SOFT_COE)
        self.temp_calib_tr      = temp_calib.data
        self.calibrated_pred_mean_tr = pred_calib.data.mean()
        self.calibrated_pred_max_tr  = pred_calib.data.max()
        self.calibrated_pred_min_tr  = pred_calib.data.min()

        loss_calib   = self.criterionMyNLL(pred_calib, self.input_mask.repeat(_nsplit, 1, 1, 1), False, bypass_bound = self.bypass_bound)
        # The loss is averaged anyway ... FIXME make it looks nicer

        pred_calib_tr = pred_calib
        eff_mask    = self.input_mask.repeat(_nsplit, 1, 1 ,1)
        loss_dice   = self.criterionDice(input = pred_calib_tr, target = eff_mask)
        loss_wce    = self.criterionWCE(inputs = pred_calib_tr, targets = eff_mask.long() )

        self.seg_tr         = pred_calib.detach()
        self.stdmap_tr      = torch.sum(stdmap.detach(), dim = 1, keepdim = True)
        self.loss_calib     = loss_calib * lambda_calib + (loss_dice * lambda_dice + loss_wce * lambda_wce) * lambda_Seg
        self.loss_calib_tr  = loss_calib.data
        self.loss_wce_tr    = loss_wce.data

        return pred_calib, pred_uncalib, temp_calib


    def optimize_parameters(self, is_pretrain = False, flip_mode = False):
        self.set_requires_grad(self.subnets, False)
        # first train the denoiser
        if is_pretrain: # in pre-train, dae needs gradient
            self.set_requires_grad(self.netDAE, True)
            pred_uncalib, diff_denoised = self.forward_dae_train(self.input_img, dae_has_grad = True, copy_mode = False)
            self.optimizer_DAE.zero_grad()
            self.loss_dae.backward()
            self.optimizer_DAE.step()
        elif not flip_mode: # if train jointly, gradient is needed. This leads to shortcut learning and perform worse than joint training
            raise Exception
            self.set_requires_grad(self.netDAE, True)
            pred_uncalib, diff_denoised = self.forward_dae_train(self.input_img_3copy, dae_has_grad = True)
            self.optimizer_DAE.zero_grad()
            self.loss_dae.backward(retain_graph = True)
            self.optimizer_DAE.step()
        else:
            pred_uncalib, diff_denoised = self.forward_dae_train(self.input_img_3copy, dae_has_grad = False)
            self.optimizer_DAE.zero_grad()

        # then train the calibrator
        if not is_pretrain:
            self.set_requires_grad(self.subnets, False)
            self.set_requires_grad(self.netCalib, True)

            pred_calib, pred_uncalib, temp_calib = self.forward_calib_alea_tune(self.input_img_3copy, pred_uncalib, diff_denoised)
            self.optimizer_Calib.zero_grad()
            self.loss_calib.backward()
            self.optimizer_Calib.step()


        # close everything
        self.set_requires_grad(self.subnets, False) # just in case
        for subnet in self.subnets:
            subnet.eval()

    def get_current_errors_tr(self):
        ret_errors = []
        if hasattr(self, 'loss_calib_tr'):
            ret_errors += [ ('loss_calib_tr', self.loss_calib_tr),\
                ]

        ret_errors = OrderedDict(ret_errors)
        return ret_errors

    def get_current_errors_val(self):
        ret_errors = [ ('loss_calib_val', self.loss_calib_val.mean()), ('l1_recon_val', self.diff_recon_val.mean()) ]
        ret_errors = OrderedDict(ret_errors)
        return ret_errors

    def get_current_visuals_val(self):
        """
        """
        img_val    = t2n(self.input_img.data)
        gth_val    = t2n(self.gth_val.data)
        pred_val   = t2n( torch.argmax(self.pred_val_calib.data, dim =1, keepdim = True ))
        temp_val   = t2n( self.temp_val.data )
        dae_diff_val   = t2n( self.diff_denoised_val ) / 2.1

        ret_visuals = [\
               # ('img_seen_val', img_val),\
                ('pred_val', pred_val * 1.0 / self.n_cls + 0.01),\
                ('gth_val', gth_val * 1.0 / self.n_cls + 0.01),\
                ]

        return OrderedDict(ret_visuals)

    # detech visualization for validation from these for training
    def get_current_visuals_tr(self):
        img_tr  = t2n( to01(self.input_img.data, True))
        gth_tr  = t2n(self.input_mask.data )
        ret_visuals = [\
                ('gth_seen_tr', (gth_tr + 0.01) * 1.0 / (self.n_cls + 0.01 )), \
                ]
        if hasattr(self, 'seg_tr'):
            pred_tr = t2n( torch.argmax(self.seg_tr.data, dim =1, keepdim = True )  )
            ret_visuals += [('pred_seen_tr', pred_tr)]
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
        self.save_network(self.netDAE, 'DAE', label, self.gpu_ids)

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

