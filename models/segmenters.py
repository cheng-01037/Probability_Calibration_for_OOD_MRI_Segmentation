import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.unets import ScaleUNet

def scale_unet(nclass, in_channel, scale_base, gpu_ids = []):
    model = ScaleUNet(in_channel, nclass, scale_base = scale_base)

    if len(gpu_ids) > 0:
        return model.cuda(gpu_ids[0])
    else:
        raise Error("GPU not found!")

