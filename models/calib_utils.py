# utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import extern_calib_tools.dev_reliability_diagrams as drd
import math

c2n = lambda x: x.data.cpu().numpy()

def rely_wrapper_cs(pred_t, gth_t, eff_mask_t = None):
    """
    Computing ece and sce
    Args:
        pred: nc, nb,  nx, ny
        gth: nb,nx, ny
        eff_mask: nb, nx, ny. When only computing within a certain region
    """
    #pred_t = torch.Tensor(pred).cuda()
    nc = pred_t.shape[0]
    pred_t = pred_t.reshape( nc, -1)
    gth_t = gth_t.long().reshape(-1)

    if eff_mask_t is not None:
        eff_mask_t  = eff_mask_t.reshape(-1)
        fetch_idx   = eff_mask_t > 0.5

        gth_t = gth_t[eff_mask_t > 0.5 ]
        pred_t = pred_t[:, eff_mask_t > 0.5]

    try:
        pred_t_hard = torch.argmax(pred_t, dim = 0, keepdim = False) # ns
    except:
        print(f'sum of prediction {pred_t.sum()}')
        pred_t_hard = pred_t[0]

    confidence = pred_t.max( dim = 0)[0] # confidence of the highest-scoring class

    calib_out = drd.compute_calibration(true_labels = c2n(gth_t), pred_labels = c2n(pred_t_hard),\
                                       confidences = c2n(confidence))

    scev2 = drd.get_sce_calibrationv2(true_labels = c2n(gth_t), pred_labels = c2n(pred_t_hard),\
                                       probs = c2n(pred_t), nclass = 4)

    calib_out['static_calibration_errorv2'] = scev2['static_calibration_errorv2']
    return calib_out

class MetricsRoi3D(nn.Module):
    def __init__(self, margins = (0, 10, 10)):
        super(MetricsRoi3D, self).__init__()
        self.margins = margins
        self.myroi_op = DilateOP(1, 1, kernel_size = margins[-1]).cuda()

    def __call__(self, pred_t, gth_t, pred_b_flg = False):
        """
        pred_b: if True, compute the boundary using binarized pred.
        gth:    nb/nz,  nx, ny
        pred_t: nc, nb/nz, nx, ny ugly practice don't ask me why
        """
        out_buffer = {}
        myroi = self.myroi_op(gth_t)
        _c_k    = rely_wrapper_cs(pred_t, gth_t, myroi)
        return _c_k

# https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d
class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1) # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L)
        else:
            x = torch.logsumexp(x*self.beta, dim=2, keepdim=False) / self.beta # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)
        return x

class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')

class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class DilateOP(nn.Module):
    """
    boundary op, allowing going to cuda
    """
    def __init__(self, in_channels, out_channels, kernel_size = 5, soft_max = False):
        super(DilateOP, self).__init__()
        self.skip_bound = False
        if kernel_size < 0:
            kernel_size = 5 # not really in use
            self.skip_bound = True # actuall returning all ones
        dilate_fn = Dilation2d(in_channels, out_channels, kernel_size = kernel_size, soft_max = soft_max)

        self.dilate_fn = dilate_fn

    def __call__(self, y_in):
        """
        y_in: binary, nb, nc, nx, ny
        """
        if y_in.ndim == 3:
            y_in = y_in.unsqueeze(1)
        y_in = torch.clamp(y_in, min = 0., max = 1.) # binarize
        if not self.skip_bound:
            y_d = self.dilate_fn(y_in)
            return y_d.squeeze(1)

        else:
            return torch.ones(y_in.shape).cuda()


class MyNLL(nn.Module):
    def __init__(self, nclass = 5):
        super(MyNLL, self).__init__()
        self.nll_node = nn.CrossEntropyLoss() # see LTS implementation
        self.n_class = nclass
        self.forward = self.forward_k_ineff

    def forward_k_ineff(self, logits, gth, copy3, **kwargs):
        """
        when k < 0: use entire image as region for calculating nll loss
        """
        if copy3:
            raise Exception
            gth = gth.repeat(3,1,1,1)

        nb, nc, _, _ = logits.shape

        loss = self.nll_node(input = logits.view(nb, nc, -1), target = gth.view(nb, -1).long() )
        return loss

