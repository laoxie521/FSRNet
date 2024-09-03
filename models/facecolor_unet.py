

import math
import torch

from torch import nn
from inspect import isfunction
from models.help import Block, FSABlock

def exists(x):
    return x is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun
class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('gaussian'))#对卷积层进行函数的初始化

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x

# class gradblock(nn.Module):
#     def __init__(self,connel=4):
#         super(gradblock, self).__init__()
#         self.connel=connel
#
#
#     def forward(self,x):
#         f = torch.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3).to(x)
#         f = f.repeat(( self.connel, 1, 1, 1))  # 让f为一维的3x3，变成三维的3x3层一样的\
#
#         grad_b = F.conv2d(x, f, padding=1, groups=self.connel)
#         return grad_b

class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        #nn.ConvTranspose2d进行上采样 反卷积
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv.apply(weights_init('gaussian'))

        if after=='BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after=='Tanh':
            self.after = torch.tanh
        elif after=='sigmoid':
            self.after = torch.sigmoid

        if before=='ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before=='LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if hasattr(self, 'before'):
            x = self.before(x)
        x = self.conv(self.up(x))
        if hasattr(self, 'after'):
            x = self.after(x)
        return x


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Face_UNet(nn.Module):
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            inner_channel=64,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8),
            res_blocks=1,
            attn_res=16,
            dropout=0,
            image_size=128,
            feature=False
    ):
        super(Face_UNet, self).__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        self.feature=feature

        downs = [nn.Conv2d(in_channel, inner_channel,
                       kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(FSABlock(
                pre_channel, channel_mult,  norm_groups=norm_groups,
                dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            FSABlock(pre_channel, pre_channel, norm_groups=norm_groups,
                           dropout=dropout, with_attn=True),
            FSABlock(pre_channel, pre_channel,  norm_groups=norm_groups,
                           dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(FSABlock(
                    pre_channel + feat_channels.pop(), channel_mult,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
       
    def forward(self, x):
        # if self.grad== "grad":
        #    x=gradblock()(x)
        #encoder
        inp =x
        _,_,H,W=inp.shape
        feats = []
        edgefeats=[]
        facefeats = []
        for layer in self.downs:
            x = layer(x)
            feats.append(x)
            if isinstance(layer, FSABlock):

                edgefeats.append(x)

        for layer in self.mid:
            x = layer(x)
            facefeats.append(x)
        for layer in self.ups:
            if isinstance(layer, FSABlock):
                feat = feats.pop()
                # if x.shape[2]!=feat.shape[2] or x.shape[3]!=feat.shape[3]:
                #     feat = F.interpolate(feat, x.shape[2:])
                x = layer(torch.cat((x, feat), dim=1))
            else:
                x = layer(x)

        if self.feature:
            return self.final_conv(x), edgefeats
        else:
            return self.final_conv(x), facefeats

