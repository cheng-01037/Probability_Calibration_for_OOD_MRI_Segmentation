""" Parts of the U-Net model
Extended from Zijun Deng's implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_leaky = False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)


class MCDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2
        contains MC dropout
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, use_leaky = False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p = 0.1, inplace = True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p = 0.1, inplace = True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.2)
        )


    def train(self, mode = True):
        if mode == True:
            super().train(mode)
        else:
            self.eval_w_dropout()


    def eval_w_dropout(self):
        for _name, _module in self.named_children():
            # print(_name)
            if 'double_conv' in _name:
                for _subname, _layer in _module.named_children():
                    # print(_layer)
                    if isinstance(_layer, nn.Dropout2d):
                        # import pdb; pdb.set_trace()
                        _layer.train()
                    else:
                        _layer.eval()
            else:
                _module.eval()

    def forward(self, x):
        return self.double_conv(x)


class DoubleConvDropout(nn.Module):
    """(convolution => [BN] => ReLU) * 2
    Doubleconv with standard dropout
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, drop_p = 0.1, use_leaky = False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p = drop_p, inplace = True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p = drop_p, inplace = True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, use_mc = False, use_leaky = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_leaky) if not use_mc else MCDoubleConv(in_channels, out_channels, use_leaky)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_mc = False, use_leaky = False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2, use_leaky = use_leaky) if not use_mc else MCDoubleConv(in_channels, out_channels // 2, in_channels // 2, use_leaky = use_leaky )
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_leaky = use_leaky ) if not use_mc else MCDoubleConv(in_channels, out_channels , use_leaky = use_leaky )


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class DownDrop(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop_p, use_leaky = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,use_leaky = use_leaky ) if not drop_p <= 0 else DoubleConvDropout(in_channels, out_channels, drop_p = drop_p, use_leaky = use_leaky )
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpNoSkip(nn.Module):
    """Upscaling then double conv. No skip connection"""

    def __init__(self, in_channels, out_channels, bilinear=True, use_leaky = False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels, use_leaky = use_leaky)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_leaky = use_leaky)

    def forward(self, x1, ref_x2):
        """
        x2 is the reference size, not really used!
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([ref_x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([ref_x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return self.conv(x1)



