import argparse
import os
from my_utils import util
import itertools
import glob

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('Calibration')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './my_utils', './scripts', './extern_calib_tools']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))

for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    exp_type = 'erm'
    calib_type = 'proposed'
    name = 'myexp'
    phase = 'train'
    batchSize = 1
    fineSize = 192
    gpu_ids = [0]
    nThreads = 2
    persistent_workers = False
    load_dir = './exp_logs'
    checkpoints_dir = './exp_logs'
    reload_model_fid = '' # path for reloading segmentation model
    reload_calib_fid = '' # path for reliading calibration model

    ###### validation configs ######
    print_freq = 1000
    validation_freq = 1000
    save_epoch_freq = 500
    infer_epoch_freq = 500
    save_prediction = False
    display_freq = 1000

    ###### training configs ######
    data_name = 'ACDC'
    model = 'scale_unet'
    eval_fold = 0
    nclass = 4

    acdcc_domains = ['Bias', 'Motion', 'Ghosting', 'Spike']
    # unet
    unet_scale_base = 32

    continue_train = False
    epoch_count = 1
    which_epoch = 'latest'
    niter = 50
    niter_decay = 1950

    optimizer = 'adam'
    # configs for adam
    beta1 = 0.5
    lr = 0.0003
    adam_weight_decay = 0.00003

    lr_policy = 'lambda' # step/ plateau
    lr_decay_iters = 50
    early_stop_epoch = 2001

    lambda_Seg = 1.0
    lambda_wce = 1.0
    lambda_dice = 1.0
    lambda_calib = 1.0

    use_clip = False # has nothing to do with vl pretraining
    clip_min = 1e5
    clip_max = -1e5

    save_prediction = True

    init_type = 'normal' # [normal|xavier|kaiming|orthogonal]')
    alea_n_rep = 6 # will by multiplied by 3. number of inference steps for aleotoric uncertainty. Small number for quick training, large number for accurate calibration at inference
    debug_lts_add_aug = False # additional augmentation can be added for lts but usually does not work
    te_reps = -1 # will be multiplied by 3. number of inference steps for alea in testing. -1 is to set the same as training

    # specific for DAE shape model only
    dae_enc_drop_p = 0.5
    reload_dae_fid = ''
    dae_scale_base = 8
    lambda_dae = 1.0
    pretrain_epoch = -1
    dae_lr = 1e-3
    dae_mode = 'unet'
    dae_bilinear = True
    dae_flip_mode = False
    daebatchSize = 20
    lts_res_se = False

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["name"]}'
    observer = FileStorageObserver.create(os.path.join(config['checkpoints_dir'], exp_name))
    ex.observers.append(observer)
    return config
