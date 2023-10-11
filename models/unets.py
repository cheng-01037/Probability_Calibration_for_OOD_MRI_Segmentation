import torch.nn.functional as F
from .unet_parts import *

class ScaleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, scale_base = 64):
        """
        easy-to-scale unet by changing scale_base. 64 is the vanilla unet
        """
        super(ScaleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sb = scale_base

        self.inc = DoubleConv(n_channels, self.sb)
        self.down1 = Down(self.sb, self.sb * 2)
        self.down2 = Down(self.sb * 2, self.sb * 4)
        self.down3 = Down(self.sb * 4, self.sb * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.sb * 8, self.sb * 16 // factor)
        self.up1 = Up(self.sb*16, self.sb*8, bilinear)
        self.up2 = Up(self.sb*8, self.sb*4, bilinear)
        self.up3 = Up(self.sb*4, self.sb*2, bilinear)
        self.up4 = Up(self.sb*2, self.sb * factor, bilinear)
        self.outc = OutConv(self.sb, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, F.interpolate(x, [32,32])

class ScaleUNetDrop(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, scale_base = 64, drop_p = 0.1,use_leaky = False ):
        """
        easy-to-scale unet by changing scale_base. 64 is the vanilla unet
        Also supports mc-dropout
        """
        super(ScaleUNetDrop, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.sb = scale_base

        if drop_p <= 0:
            self.inc = DoubleConv(n_channels, self.sb, use_leaky = use_leaky)
        else:
            self.inc = DoubleConvDropout(n_channels, self.sb, drop_p = drop_p, use_leaky = use_leaky)

        self.down1 = DownDrop(self.sb, self.sb * 2,  drop_p = drop_p, use_leaky = use_leaky)
        self.down2 = DownDrop(self.sb * 2, self.sb * 4, drop_p = drop_p, use_leaky = use_leaky)
        self.down3 = DownDrop(self.sb * 4, self.sb * 8, drop_p = drop_p, use_leaky = use_leaky)
        factor = 2 if bilinear else 1
        self.down4 = DownDrop(self.sb * 8, self.sb * 16 // factor, drop_p = drop_p, use_leaky = use_leaky)
        self.up1 = Up(self.sb*16, self.sb*8, bilinear, use_leaky = use_leaky)
        self.up2 = Up(self.sb*8, self.sb*4, bilinear, use_leaky = use_leaky)
        self.up3 = Up(self.sb*4, self.sb*2, bilinear, use_leaky = use_leaky)
        self.up4 = Up(self.sb*2, self.sb * factor, bilinear)
        self.outc = OutConv(self.sb, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


