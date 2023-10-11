"""
Calibration models used in experiments. Modified from https://github.com/uncbiag/LTS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.aux_nets as cusnet

class Temperature_Scaling(nn.Module):
    def __init__(self):
        super(Temperature_Scaling, self).__init__()
        self.temperature_single = nn.Parameter(torch.ones(1))

    def weights_init(self):
        pass

    def forward(self, logits, image):
        temperature = self.temperature_single.expand(logits.size()).cuda( )
        #return logits / temperature
        return temperature

class VanillaLTS(nn.Module):
    def __init__(self, nclass, use_clip = False, clip_range = (1.0, -1.0 )):
        super(VanillaLTS, self).__init__()
        assert use_clip == False

        self.nclass = nclass
        self.temperature_level_2_conv1 = nn.Conv2d(nclass, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv2 = nn.Conv2d(nclass, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv3 = nn.Conv2d(nclass, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv4 = nn.Conv2d(nclass, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param1 = nn.Conv2d(nclass, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param2 = nn.Conv2d(nclass, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_param3 = nn.Conv2d(nclass, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        self.temperature_level_2_conv_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True)
        #self.temperature_level_2_param_img = nn.Conv2d(3, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) # c8. probably a typo here as c8 interacts with the logits in the original paper
        self.temperature_level_2_param_img = nn.Conv2d(nclass, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) # c8. probably a typo here as c8 interacts with the logits in the original paper

        self.weights_init()

        self.use_clip = use_clip
        self.clip_min, self.clip_max = clip_range


    def weights_init(self):
        torch.nn.init.zeros_(self.temperature_level_2_conv1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv4.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param1.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param2.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param3.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_conv_img.bias.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.weight.data)
        torch.nn.init.zeros_(self.temperature_level_2_param_img.bias.data)

    def forward(self, logits, image):
        temperature_1 = self.temperature_level_2_conv1(logits)
        temperature_1 += (torch.ones(1)).cuda( )
        temperature_2 = self.temperature_level_2_conv2(logits)
        temperature_2 += (torch.ones(1)).cuda( )
        temperature_3 = self.temperature_level_2_conv3(logits)
        temperature_3 += (torch.ones(1)).cuda( )
        temperature_4 = self.temperature_level_2_conv4(logits)
        temperature_4 += (torch.ones(1)).cuda( )
        temperature_param_1 = self.temperature_level_2_param1(logits)
        temperature_param_2 = self.temperature_level_2_param2(logits)
        temperature_param_3 = self.temperature_level_2_param3(logits)
        temp_level_11 = temperature_1 * torch.sigmoid(temperature_param_1) + temperature_2 * (1.0 - torch.sigmoid(temperature_param_1))
        temp_level_12 = temperature_3 * torch.sigmoid(temperature_param_2) + temperature_4 * (1.0 - torch.sigmoid(temperature_param_2))
        temp_1 = temp_level_11 * torch.sigmoid(temperature_param_3) + temp_level_12 * (1.0 - torch.sigmoid(temperature_param_3))
        temp_2 = self.temperature_level_2_conv_img(image) + torch.ones(1).cuda( )
        temp_param = self.temperature_level_2_param_img(logits) # this is the original code. c8
        # temp_param = self.temperature_level_2_param_img(image) # modifiyed by co1818
        self.c8_act = torch.sigmoid(temp_param)
        temperature = temp_1 * self.c8_act + temp_2 * (1.0 - self.c8_act)
        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).cuda( )) + sigma
        temperature = temperature.repeat(1, self.nclass, 1, 1)
        #return logits / temperature # original implementation. As we need to visualize temperature, we take it out
        if self.use_clip:
            temperature = torch.clamp(temperature, min = self.clip_min, max = self.clip_max)

        return temperature



class MyLTS(nn.Module):
    def __init__(self, nclass, interm_dim = 16, use_clip = False, clip_range = (1.0, -1.0 ), res_se = False):
        super(MyLTS, self).__init__()
        assert res_se == False

        self.nclass = nclass
        self.conv_img = nn.Conv2d(3, interm_dim, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.res1_img = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_img = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.img_branch = nn.Sequential(*[ self.conv_img, self.res1_img, self.res2_img  ])

        self.conv_logits = nn.Conv2d(nclass , interm_dim, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.res1_logits = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_logits = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.logits_branch = nn.Sequential(*[ self.conv_logits, self.res1_logits, self.res2_logits  ])

        self.resse_comb = cusnet.ResSE(interm_dim * 2, reduction_ratio = 2 ) if res_se else cusnet.SE(interm_dim * 2, reduction_ratio = 2 )
        self.conv_comb = nn.Conv2d(interm_dim * 2 , interm_dim * 4, kernel_size=1, bias=True) #
        self.res1_comb = cusnet.ResnetBlock( interm_dim * 4, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_comb = cusnet.ResnetBlock( interm_dim * 4, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.conv_out = nn.Conv2d(interm_dim * 4, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.comb_branch = nn.Sequential(*[self.resse_comb, self.conv_comb, self.res1_comb, self.res2_comb, self.conv_out]  )

        self.use_clip = use_clip
        self.clip_min, self.clip_max = clip_range

    def forward(self, logits, image):
        img_out = self.img_branch(image)
        logits_out = self.logits_branch(logits)
        _comb = torch.cat( [img_out, logits_out], dim = 1  )
        temperature = self.comb_branch(_comb)

        sigma = 1e-8
        temperature = F.relu(temperature + torch.ones(1).cuda( )) + sigma
        temperature = temperature.repeat(1, self.nclass, 1, 1)
        if self.use_clip:
            raise NotImplementedError

        return temperature

class OODCalibNet(nn.Module):
    """
    full version
    """
    def __init__(self, nclass, use_clip = False, clip_range = (1.0, -1.0 ), interm_dim = 16, res_se = False ):
        super(OODCalibNet, self).__init__()
        assert use_clip == False

        self.nclass = nclass
        self.conv_img = nn.Conv2d(3, interm_dim, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.res1_img = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_img = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.img_branch = nn.Sequential(*[ self.conv_img, self.res1_img, self.res2_img  ])

        self.conv_logits = nn.Conv2d(nclass , interm_dim, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.res1_logits = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_logits = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.logits_branch = nn.Sequential(*[ self.conv_logits, self.res1_logits, self.res2_logits  ])

        self.conv_diff = nn.Conv2d(nclass , interm_dim, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.res1_diff = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_diff = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.diff_branch = nn.Sequential(*[ self.conv_diff, self.res1_diff, self.res2_diff ])

        self.conv_mean = nn.Conv2d(nclass , interm_dim, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.res1_mean = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_mean = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.mean_branch = nn.Sequential(*[ self.conv_mean, self.res1_mean, self.res2_mean ])

        self.conv_std = nn.Conv2d(nclass , interm_dim, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.res1_std = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_std = cusnet.ResnetBlock( interm_dim, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.std_branch = nn.Sequential(*[ self.conv_std, self.res1_std, self.res2_std  ])

        self.resse_comb = cusnet.ResSE(interm_dim * 5, reduction_ratio = 5 ) if res_se == True else cusnet.SE(interm_dim * 5, reduction_ratio = 5 )
        self.conv_comb = nn.Conv2d(interm_dim * 5 , interm_dim * 4, kernel_size=1, bias=True) #
        self.res1_comb = cusnet.ResnetBlock( interm_dim * 4, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.res2_comb = cusnet.ResnetBlock( interm_dim * 4, padding_type = 'reflect', norm_layer = nn.BatchNorm2d, act_layer = nn.ReLU, use_dropout = False, use_bias = True)
        self.conv_out = nn.Conv2d(interm_dim * 4, 1, kernel_size=5, stride=1, padding=4, padding_mode='reflect', dilation=2, bias=True) #
        self.comb_branch = nn.Sequential(*[self.resse_comb, self.conv_comb, self.res1_comb, self.res2_comb, self.conv_out]  )

        self.use_clip = use_clip
        self.clip_min, self.clip_max = clip_range

    def forward(self, logits, image, dae_diff):

        _logits, _meanmap, _stdmap = logits[:, :self.nclass], logits[:, self.nclass: self.nclass * 2], logits[:, self.nclass * 2: self.nclass * 3]
        img_out     = self.img_branch(image)
        logits_out  = self.logits_branch(_logits)
        std_out     = self.std_branch(_stdmap)
        mean_out    = self.mean_branch(_meanmap)
        dae_out     = self.diff_branch(dae_diff)
        _comb       = torch.cat( [img_out, logits_out, dae_out, mean_out, std_out], dim = 1  )
        temperature = self.comb_branch(_comb)

        sigma = 1e-8 # does not really matter as we are adding another softening coefficient in the main script
        temperature = F.relu(temperature + torch.ones(1).cuda( )) + sigma
        temperature = temperature.repeat(1, self.nclass, 1, 1)
        if self.use_clip:
            assert NotImplementedError

        return temperature


