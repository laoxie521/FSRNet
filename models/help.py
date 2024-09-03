import numbers

import torch
from einops import rearrange
from torch import nn
import math

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        # print(dim)
        self.block = nn.Sequential(
            #分组归一化
            nn.GroupNorm(groups, dim),
            Swish(),
            # nn.Dropout，防止网络过拟合nn.Identity()
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)
class EdgeBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        # print(dim)
        self.block = nn.Sequential(
            #分组归一化
            nn.GroupNorm(groups, dim),
            Swish(),
            # nn.Dropout，防止网络过拟合nn.Identity()
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out,  dropout=0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        # nn.Identity()相当于一个恒等函数得到他之前的结果
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        #对比ppdm多加了一个block1
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x
#spatial attention
# class SelfAttention(nn.Module):
#     def __init__(self, in_channel, n_head=1, norm_groups=32):
#         super().__init__()
#
#         self.n_head = n_head
#
#         self.norm = nn.GroupNorm(norm_groups, in_channel)
#         self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
#         self.out = nn.Conv2d(in_channel, in_channel, 1)
#
#     def forward(self, input):
#         batch, channel, height, width = input.shape
#         n_head = self.n_head
#         head_dim = channel // n_head
#
#         norm = self.norm(input)
#         qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
#         query, key, value = qkv.chunk(3, dim=2)  # bhdyx
#
#         attn = torch.einsum(
#             "bnchw, bncyx -> bnhwyx", query, key
#         ).contiguous() / math.sqrt(channel)
#         attn = attn.view(batch, n_head, height, width, -1)
#         attn = torch.softmax(attn, -1)
#         attn = attn.view(batch, n_head, height, width, height, width)
#
#         out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
#         out = self.out(out.view(batch, channel, height, width))
#
#         return out + input
class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.temperature = nn.Parameter(torch.ones(n_head, 1, 1))
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(in_channel*3, in_channel * 3, kernel_size=3,stride=1,padding=1,groups=in_channel*3, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        # head_dim = channel // n_head
        head_dim = channel

        norm = self.norm(input)
        qkv = self.qkv_dwconv(self.qkv(norm)).view(batch,head_dim*3,height,width)
        query, key, value = qkv.chunk(3, dim=1)  # bhdyx
        q = rearrange(query, 'b (head c) h w -> b head c (h w)', head=self.n_head)
        k = rearrange(key, 'b (head c) h w -> b head c (h w)', head=self.n_head)
        v = rearrange(value, 'b (head c) h w -> b head c (h w)', head=self.n_head)

        q = torch.nn.functional.normalize(q, dim=-1)

        k = torch.nn.functional.normalize(k, dim=-1)

        q_fft = torch.fft.rfft2(q.float())
        q_fft=torch.fft.fftshift(q_fft)
        # print(q_fft.shape)
        k_fft = torch.fft.rfft2(k.float())
        k_fft = torch.fft.fftshift(k_fft)


        _, _, C, _ = q.shape

        mask1 = torch.zeros(batch, self.n_head, C, C, device=input.device, requires_grad=False)
        mask2 = torch.zeros(batch, self.n_head, C, C, device=input.device, requires_grad=False)
        mask3 = torch.zeros(batch, self.n_head, C, C, device=input.device, requires_grad=False)
        mask4 = torch.zeros(batch, self.n_head, C, C, device=input.device, requires_grad=False)
        attn = (q_fft @ k_fft.transpose(-2, -1)) * self.temperature
        # print(attn.shape)
        attn = torch.fft.ifftshift(attn)
        attn = torch.fft.irfft2(attn,s=(C,C))
        # print(attn.shape)
        #torch.topk用来求tensor中某个dim的前k大或者前k小的值以及对应的index。 k是维度
        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        #把1的数按照scatter的第一个参数叫维度，按行或按列把index的tensor，填入到mask中，
        #具体操作看https://blog.csdn.net/guofei_fly/article/details/104308528?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-104308528-blog-108311046.235%5Ev38%5Epc_relevant_default_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-104308528-blog-108311046.235%5Ev38%5Epc_relevant_default_base&utm_relevant_index=2
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        #torch.where 满足条件返回attn，不满足返回 torch.full_like(attn, float('-inf')表示负无穷的意思)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.n_head, h=height, w=width)

        out = self.out(out)

        return out
class TransformerBlock(nn.Module):
    def __init__(self, dim, norm_groups,ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = SelfAttention(dim,norm_groups=norm_groups)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
class FSABlock(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = TransformerBlock(dim_out, norm_groups=norm_groups,ffn_expansion_factor=2.66,bias=False)

    def forward(self, x):
        x = self.res_block(x)
        if self.with_attn:
            x = self.attn(x)
        return x

class TFModule(nn.Module):
    def __init__(self, pre_channel, ffn_expansion_factor, bias):
        super( ResCrossBlock, self).__init__()

        self.norm = LayerNorm(pre_channel)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.xpatchembedding = nn.Conv2d(pre_channel , pre_channel , kernel_size=3, stride=1, padding=1,
                              groups=pre_channel , bias=bias)
        self.featurepatchembedding = nn.Conv2d(pre_channel , pre_channel , kernel_size=3, stride=1, padding=1,
                              groups=pre_channel , bias=bias)

        self.relu3 = nn.ReLU()
        self.xline1=nn.Conv2d(pre_channel,pre_channel//ffn_expansion_factor,1,padding=0,bias=False)
        self.xline2=nn.Conv2d(pre_channel//ffn_expansion_factor,pre_channel,1,padding=0,bias=False)
        self.faceline1=nn.Conv2d(pre_channel,pre_channel//ffn_expansion_factor,1,padding=0,bias=False)
        self.faceline2=nn.Conv2d(pre_channel//ffn_expansion_factor,pre_channel,1,padding=0,bias=False)
        self.x3x3_1 = nn.Conv2d(pre_channel , pre_channel, kernel_size=3, stride=1, padding=1,
                                groups=pre_channel, bias=bias)

        self.face3x3_1 = nn.Conv2d(pre_channel , pre_channel, kernel_size=3, stride=1, padding=1,
                                   groups=pre_channel, bias=bias)
        self.relux_1 = Swish()
        self.reluface_1 = Swish()

        self.project_out = nn.Conv2d(pre_channel * 2, pre_channel, kernel_size=1, bias=bias)

    def forward(self, x, feature):
        b,c,_,_=x.shape
        x_1 = self.xpatchembedding(self.norm(x))
        feature_1 = self.featurepatchembedding(self.norm(feature))
        x_1=self.avg_pool(x_1)

        feature_1=self.avg_pool(feature_1)
        x_1 = self.relu3(self.xline1(x_1))
        feature_1 = self.relu3(self.faceline1(feature_1))

        x_1=torch.sigmoid(self.xline2(x_1))
        feature_1=torch.sigmoid(self.faceline2(feature_1))

        new_x=feature*x_1.expand_as(x)
        feature=x*feature_1.expand_as(feature)
        new_x = self.relux_1(self.x3x3_1(new_x))
        feature = self.reluface_1(self.face3x3_1(feature))

        g = torch.cat([new_x, feature], dim=1)

        g = self.project_out(g)

        return g

# class FaceFuseBlock(nn.Module):
#     def __init__(self, pre_channel, ffn_expansion_factor, bias):
#         super(FaceFuseBlock, self).__init__()
#
#         hidden_features = int(pre_channel * ffn_expansion_factor)
#         self.norm=LayerNorm(pre_channel)
#         self.project_in = nn.Conv2d(pre_channel, hidden_features * 2, kernel_size=1, bias=bias)
#
#         self.x3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
#         self.face3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
#         self.relu3 = nn.ReLU()
#         self.relu5 = nn.ReLU()
#
#         self.x3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#         self.face3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#
#         self.relu3_1 = Swish()
#         self.relu5_1 = Swish()
#
#         self.project_out = nn.Conv2d(hidden_features * 2, pre_channel, kernel_size=1, bias=bias)
#
#     def forward(self, x,feature):
#
#         h = self.project_in(self.norm(x))
#         feature = self.project_in(self.norm(feature))
#         x1_3, x2_3 = self.relu3(self.x3x3(h)).chunk(2, dim=1)
#         face1_3, face2_3 = self.relu5(self.face3x3(feature)).chunk(2, dim=1)
#
#         x1 = torch.cat([x1_3, face1_3], dim=1)
#         x2 = torch.cat([x2_3, face2_3], dim=1)
#
#         x1 = self.relu3_1(self.x3x3_1(x1))
#         x2 = self.relu5_1(self.face3x3_1(x2))
#
#         g = torch.cat([x1, x2], dim=1)
#
#         g = self.project_out(g)
#
#         return g+x


def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

class CFModule(nn.Module):
    def __init__(self, pre_channel, ffn_expansion_factor, bias,alpha=1.0):
        super(FaceFuseBlockplus, self).__init__()

        hidden_features = int(pre_channel * ffn_expansion_factor)
        self.alpha=alpha
        self.norm=LayerNorm(pre_channel)
        self.project_in = nn.Conv2d(pre_channel, hidden_features * 2, kernel_size=1, bias=bias)

        self.x3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.face3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.x3x3_1 = nn.Conv2d(hidden_features , hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.face3x3_1 = nn.Conv2d(hidden_features , hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)

        self.relu3_1 = Swish()
        self.relu5_1 = Swish()

        self.project_out = nn.Conv2d(hidden_features*2 , pre_channel, kernel_size=1, bias=bias)


    def forward(self, x,feature):

        h = self.project_in(self.norm(x))
        feature = self.project_in(self.norm(feature))

        x1_3, x2_3 = self.relu3(self.x3x3(h)).chunk(2, dim=1)
        face1_3, face2_3 = self.relu5(self.face3x3(feature)).chunk(2, dim=1)
        t1 = adain(x1_3, face1_3)
        t2 = adain(x2_3, face2_3)
        t1 = self.alpha * t1 + (1 - self.alpha) * x1_3
        t2 = self.alpha * t2 + (1 - self.alpha) * x2_3
        # x1 = torch.cat([x1_3, face1_3], dim=1)
        # x2 = torch.cat([x2_3, face2_3], dim=1)

        h_feature = self.relu3_1(self.x3x3_1(t1))
        face_feature =  self.relu5_1(self.face3x3_1(t2))
        g = torch.cat([h_feature, face_feature], dim=1)
        # print(t.shape)
        g = self.project_out(g)

        return g+x

