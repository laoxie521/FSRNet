import this

import math
import torch
import torch.nn.functional as F
import torchvision.utils
from torch import nn
from inspect import isfunction
from models.help import Block, FSABlock, TFModule,  CFModule


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
        elif after=='LN':
            self.after= nn.InstanceNorm2d(out_channels)
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
        # self.up = nn.PixelShuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class GradientLossBlock(nn.Module):
    def __init__(self):
        super(GradientLossBlock, self).__init__()

    def forward(self, pred):
        _, cin, _, _ = pred.shape
        # _, cout, _, _ = target.shape
        assert cin == 3
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(pred)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(pred)
        kx = kx.repeat((cin, 1, 1, 1))
        ky = ky.repeat((cin, 1, 1, 1))

        pred_grad_x = F.conv2d(pred, kx, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, ky, padding=1, groups=3)
        # target_grad_x = F.conv2d(target, kx, padding=1, groups=3)
        # target_grad_y = F.conv2d(target, ky, padding=1, groups=3)

        # loss = (
        #     nn.L1Loss(reduction=self.reduction)
        #     (pred_grad_x, target_grad_x) +
        #     nn.L1Loss(reduction=self.reduction)
        #     (pred_grad_y, target_grad_y))
        return torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
#res_block可以替换，原本2现在变1，先看看2的效果
class Generator(nn.Module):
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            inner_channel=64,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8),
            attn_res=16,
            res_blocks=1,
            dropout=0,
            image_size=128,
            grad=""
    ):
        super(Generator, self).__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        self.grad=grad

        downs = [nn.Conv2d(in_channel, inner_channel,
                       kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            #当now_res为16的时候，才进行注意力机制
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(FSABlock(
                pre_channel, channel_mult, norm_groups=norm_groups,
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
            CFModule(
                pre_channel,
                ffn_expansion_factor=2.66,
                bias=False,alpha=1.0),
            FSABlock(pre_channel, pre_channel,  norm_groups=norm_groups,
                           dropout=dropout, with_attn=False),
            CFModule(
                pre_channel,
                ffn_expansion_factor=2.66,
                bias=False,alpha=1.0)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (str(now_res)in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for a in range(0, res_blocks+2):
                final=(a==res_blocks+1)
                if not final:
                    ups.append(FSABlock(
                        pre_channel + feat_channels.pop(), channel_mult,
                        norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                    pre_channel = channel_mult
                else:
                    ups.append( TFModule(pre_channel, ffn_expansion_factor=16, bias=False))

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
        # self.to_edge = nn.Sequential(
        #     nn.Conv2d( default(out_channel, in_channel), 1, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x,facefeaturemaps,featuremaps):
        # if self.grad== "grad":
        #    x=gradblock()(x)
        #encoder
        inp =x
        _,C,H,W=inp.shape
        feats = []
        for layer in self.downs:
            x = layer(x)
            # print(x.shape)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, CFModule):
                facefeaturemap = facefeaturemaps.pop()
                x = layer(x,facefeaturemap)
            else:
                x = layer(x)

            # print(x.shape)
        for layer in self.ups:

            if isinstance(layer, FSABlock):
                # if x.shape[2]!=feat.shape[2] or x.shape[3]!=feat.shape[3]:
                #     feat = F.interpolate(feat, x.shape[2:])
                feat = feats.pop()
                x = layer(torch.cat((x, feat), dim=1))

            elif isinstance(layer,TFModule):
                featuremap = featuremaps.pop()
                x = layer(x, featuremap)
            else:
                # torch.cat((x, featuremap), dim=1)
                x = layer(x)
            # print(x.shape)

        x=self.final_conv(x)
        # #针对输入通道为4的
        # K, B = torch.split(inp, (1, 3), dim=1)
        # x = K * x - B + x
        # x = x[:, :, :H, :W]
        # x+inp针对输入通道为3的newtest不加会失真只会探测阴影部分
        return x+inp
        # return x
def color_loss(x,y):
    c = nn.MSELoss()
    conv1 = nn.Conv2d(3, 3, (3, 3)).cuda()
    conv1.apply(weights_init('gaussian'))
    x1 = conv1(x)
    y1 = conv1(y)
    loss = c(x1,y1)*10
    return loss
# class Discriminator(nn.Module):
#     def __init__(self, input_channels=4):
#         super(Discriminator, self).__init__()
#         model=[nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
#                     nn.LeakyReLU(0.2, inplace=True) ]
#
#
#         model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
#                     nn.InstanceNorm2d(128),
#                     nn.LeakyReLU(0.2, inplace=True) ]
#
#         model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
#                     nn.InstanceNorm2d(256),
#                     nn.LeakyReLU(0.2, inplace=True) ]
#
#         model += [  nn.Conv2d(256, 512, 4, padding=1),
#                     nn.InstanceNorm2d(512),
#                     nn.LeakyReLU(0.2, inplace=True) ]
#
#         # FCN classification layer
#         model += [nn.Conv2d(512, 1, 4, padding=1)]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#        out=self.model(input)
#
#        return  F.avg_pool2d(out, out.size()[2:])
       # return out
class Discriminator(nn.Module):
    def __init__(self, input_channels=4):
        super(Discriminator, self).__init__()

        self.Cv0 = Cvi(input_channels, 64)

        self.Cv1 = Cvi(64, 128, before='LReLU', after='LN')

        self.Cv2 = Cvi(128, 256, before='LReLU', after='LN')

        self.Cv3 = Cvi(256, 512, before='LReLU', after='LN')

        self.Cv4 = Cvi(512, 1, before='LReLU', after='sigmoid')

    def forward(self, input):
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        out = self.Cv4(x3)
        return out


