from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
#import pytorch_lightning as pl
import math
import importlib
from torch import einsum
from einops import rearrange
import numpy as np
import os, hashlib
import requests
from tqdm import tqdm
import functools
from collections import namedtuple
from torchvision import models
import einops
from pathlib import Path
from torch.nn import init

CACHE_PATH = os.path.expanduser(BasicArgs.root_dir)

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


# ============= Perceptual Part ==============

class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = LPIPS.get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name is not "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = LPIPS.get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = LPIPS.normalize_tensor(outs0[kk]), LPIPS.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [LPIPS.spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val

    @staticmethod
    def normalize_tensor(x, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    @staticmethod
    def spatial_average(x, keepdim=True):
        return x.mean([2, 3], keepdim=keepdim)

    @staticmethod
    def download(url, local_path, chunk_size=1024):
        os.makedirs(os.path.split(local_path)[0], exist_ok=True)
        with requests.get(url, stream=True) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                with open(local_path, "wb") as f:
                    for data in r.iter_content(chunk_size=chunk_size):
                        if data:
                            f.write(data)
                            pbar.update(chunk_size)

    @staticmethod
    def md5_hash(path):
        with open(path, "rb") as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()

    @staticmethod
    def get_ckpt_path(name, root, check=False):
        URL_MAP = {
            "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
        }

        CKPT_MAP = {
            "vgg_lpips": "vgg.pth"
        }

        MD5_MAP = {
            "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
        }
        assert name in URL_MAP
        path = os.path.join(root, CKPT_MAP[name])
        if not os.path.exists(path) or (check and not LPIPS.md5_hash(path) == MD5_MAP[name]):
            print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
            LPIPS.download(URL_MAP[name], path)
            md5 = LPIPS.md5_hash(path)
            assert md5 == MD5_MAP[name], md5
        return path


# ============= VAE Part ==================

class MaskConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, mask, dmask, skip_mask=False):
        if skip_mask:
            feature = self.conv(x)
        else:
            B, W, H = mask.shape
            mask = mask.view(B, 1, W, H)
            B, W, H = dmask.shape
            dmask = dmask.view(B, 1, W, H)
            full_feature = self.conv(x)
            masked_feature = self.conv(x * ~mask)
            feature = full_feature * dmask + masked_feature * ~dmask
        return feature

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class MaskGroupNorm(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        super(MaskGroupNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.empty(1, num_channels, 1, **factory_kwargs))
            self.bias = torch.nn.Parameter(torch.empty(1, num_channels, 1, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)

    def forward(self, x, mask=None):
        original_shape = x.shape
        N = original_shape[0]
        C = original_shape[1]
        x = x.view(N, self.num_groups, -1)
        D = x.shape[-1]

        if mask is not None:
            mask = mask.view(N, -1)
            mask = einops.repeat(mask, "b wh -> b c wh", c=C)
            mask = mask.view(N, self.num_groups, -1)
            m = einops.repeat(einops.reduce(mask, "b c d -> b c", 'sum'), "b c -> b c d", d=D)
            n = torch.ones_like(m) * D
            mean_bias = x.mean(dim=-1, keepdim=True)
            mean_real = mean_bias * n / (n - m + self.eps)
            x_fulfill = x * ~mask + mean_real * mask
            var_bias = x_fulfill.var(dim=-1, keepdim=True)
            var_real = var_bias * (n - 1) / (n - m - 1 + self.eps)
        else:
            mean_real = x.mean(dim=-1, keepdim=True)
            var_real = x.var(dim=-1, keepdim=True)

        x = ((x - mean_real) / (var_real + self.eps).sqrt())
        if self.affine:
            x = x.view(N, C, -1)
            x = x * self.weight + self.bias

        return x.view(*original_shape)

class MaskNormalize(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = MaskGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x, mask, skip_mask=False):
        if skip_mask:
            normed = self.norm(x)
        else:
            B, W, H = mask.shape
            mask_c = mask.view(B, 1, W, H)
            full_normed = self.norm(x)
            masked_normed = self.norm(x * ~mask_c, mask)

            normed = full_normed * mask_c + masked_normed * ~mask_c
        return normed


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class MaskDownsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = MaskConv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x, mask, dmask, skip_mask=False):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            if skip_mask:
                pass
            else:
                mask = torch.nn.functional.pad(mask, pad, mode="constant", value=0)
            x = self.conv(x, mask, dmask, skip_mask=skip_mask)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class MaskResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = MaskNormalize(in_channels)
        #self.norm1 = Normalize(in_channels)
        self.conv1 = MaskConv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = MaskNormalize(out_channels)
        #self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = MaskConv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = MaskConv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = MaskConv2d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)

    def forward(self, x, temb, mask, skip_mask=False):
        h = x
        h = self.norm1(h, mask, skip_mask=skip_mask)
        h = nonlinearity(h)
        h = self.conv1(h, mask, mask, skip_mask=skip_mask)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h, mask, skip_mask=skip_mask)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h, mask, mask, skip_mask=skip_mask)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, mask, mask, skip_mask=skip_mask)
            else:
                x = self.nin_shortcut(x, mask, mask, skip_mask=skip_mask)

        return x + h

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class MaskAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = MaskNormalize(in_channels)
        #self.norm = Normalize(in_channels)
        self.q = MaskConv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = MaskConv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = MaskConv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = MaskConv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, mask, skip_mask=False):
        h_ = x
        h_ = self.norm(h_, mask, skip_mask=skip_mask)
        q = self.q(h_, mask, mask, skip_mask=skip_mask)
        k = self.k(h_, mask, mask, skip_mask=skip_mask)
        v = self.v(h_, mask, mask, skip_mask=skip_mask)
        b, c, h, w = q.shape

        if skip_mask:
            pass
        else:
            attn_mask = mask.reshape(b, -1) # b, h, w -> b, hw, hw
            attn_mask = einops.repeat(attn_mask, "b hw -> b hw2 hw", hw2=h*w)

            attn_mask = attn_mask & ~einops.rearrange(attn_mask, "b hw hw1 -> b hw1 hw")
        #attn_mask = ~attn_mask

        # compute attention
        
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        if skip_mask:
            pass
        else:
            mask_value = -torch.finfo(w_.dtype).max
            w_.masked_fill_(attn_mask, mask_value)
        #del attn_mask
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_, mask, mask, skip_mask=skip_mask)

        return x + h_

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class MaskEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = MaskConv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        self.down_pool = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(MaskResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MaskAttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            pool = None
            if i_level != self.num_resolutions - 1:
                down.downsample = MaskDownsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
                pool = torch.nn.MaxPool2d(2 ** (i_level + 1), 2 ** (i_level + 1), padding=0)
            self.down.append(down)
            
            self.down_pool.append(pool)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = MaskResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = MaskAttnBlock(block_in)
        self.mid.block_2 = MaskResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = MaskNormalize(block_in)
        #self.norm_out = Normalize(block_in)
        self.conv_out = MaskConv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, mask, skip_mask=False):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        if skip_mask:
            cur_mask = None
            hms = None
        else:
            cur_mask = mask
            hms = [mask]
        hs = [self.conv_in(x, mask, mask, skip_mask=skip_mask)]
        for i_level in range(self.num_resolutions):
            if skip_mask:
                pass
            else:
                pad_mask = torch.nn.functional.pad(mask, (0, 2 ** (i_level + 1) - 1, 0, 2 ** (i_level + 1) - 1), mode="constant", value=0)
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb, cur_mask, skip_mask=skip_mask)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h, cur_mask, skip_mask=skip_mask)
                hs.append(h)
                if skip_mask:
                    pass
                else:
                    hms.append(cur_mask)
            if i_level != self.num_resolutions - 1:
                if skip_mask:
                    dmask = None
                else:
                    dmask = self.down_pool[i_level](pad_mask.float()).bool()
                hs.append(self.down[i_level].downsample(hs[-1], cur_mask, dmask, skip_mask=skip_mask))
                if skip_mask:
                    pass
                else:
                    hms.append(dmask)
                    cur_mask = dmask

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb, cur_mask, skip_mask=skip_mask)
        h = self.mid.attn_1(h, cur_mask, skip_mask=skip_mask)
        h = self.mid.block_2(h, temb, cur_mask, skip_mask=skip_mask)

        # end
        h = self.norm_out(h, cur_mask, skip_mask=skip_mask)
        h = nonlinearity(h)
        h = self.conv_out(h, cur_mask, cur_mask, skip_mask=skip_mask)
        return h, hs, hms

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, mask=None):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class MaskDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * in_ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
                if i_block != self.num_res_blocks - 1:
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_in,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout))
                else:
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout))
                    block_in = block_out
                
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, hs, hms):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                if hs is not None and hms is not None:
                    skip_h = hs[-1]
                    mask = hms[-1]
                    B, W, H = mask.shape
                    mask = mask.view(B, 1, W, H)
                    h_ = (skip_h * ~mask + h * mask + h) / 2
                    hs = hs[:-1]
                    hms = hms[:-1]
                else:
                    h_ = h
                if len(self.up[i_level].attn) > 0:
                    h_ = self.up[i_level].attn[i_block](h_)
                h = self.up[i_level].block[i_block](h_, temb)
                    
            if i_level != 0:
                if hs is not None and hms is not None:
                    skip_h = hs[-1]
                    mask = hms[-1]
                    B, W, H = mask.shape
                    mask = mask.view(B, 1, W, H)
                    h_ = (skip_h * ~mask + h * mask + h) / 2
                    hs = hs[:-1]
                    hms = hms[:-1]
                else:
                    h_ = h
                h = self.up[i_level].upsample(h_)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

# This is VectorQuantizer 2
class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class DFVQGAN(nn.Module):
    class Arg8192(BasicArgs):
        coder_config = {'double_z': False, 'z_channels': 256, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
                        'ch_mult': [1, 1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [16], 'dropout': 0.0}

        quantize_config = {'n_e': 8192, 'e_dim': 256, 'beta': 0.25, 'remap': None, 'sane_index_shape': False, 'legacy': False}
        discriminator_config = {'input_nc': 3, 'n_layers': 3, 'use_actnorm': False, 'ndf': 64}
    def __init__(self, args):
        super(DFVQGAN, self).__init__()
        self.args = args
        self.encoder = MaskEncoder(**args.coder_config)
        self.decoder = MaskDecoder(**args.coder_config)

        self.quantize = VectorQuantizer(**args.quantize_config)
        self.quant_conv = torch.nn.Conv2d(args.coder_config["z_channels"], args.quantize_config["e_dim"], 1)
        self.post_quant_conv = torch.nn.Conv2d(args.quantize_config["e_dim"], args.coder_config["z_channels"], 1)

    def encode(self, x, mask, skip_mask=False):
        h, hs, hms = self.encoder(x, mask, skip_mask=skip_mask)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, hs, hms

    def decode(self, quant, hs, hms):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, hs, hms)
        return dec

class DFVQGAN8192(nn.Module):
    def __init__(self):
        super().__init__()
        model_filename = 'checkpoint/DF-VQGAN/epoch14-chunk-0.pth'

        state = torch.load(str(Path(CACHE_PATH) / model_filename), map_location = 'cpu')["model"]
        self.model = DFVQGAN(DFVQGAN.Arg8192)
        self.model.load_state_dict(state, strict=False)
        self.image_size = DFVQGAN.Arg8192.coder_config["resolution"]
        self.num_tokens = DFVQGAN.Arg8192.quantize_config["n_e"]

    @torch.no_grad()
    def get_codebook_indices(self, img, mask):
        mask = mask.bool()
        b = img.shape[0]
        # img = (2 * img) - 1
        _, _, [_, _, indices], hs, hms = self.model.encode(img, mask)
        return rearrange(indices.flatten(), '(b n)-> b n', b = b), hs, hms

    @torch.no_grad()
    def get_codebook_indices_no_hidden(self, img):
        b = img.shape[0]
        # img = (2 * img) - 1
        _, _, [_, _, indices], hs, hms = self.model.encode(img, None, skip_mask=True)
        return rearrange(indices.flatten(), '(b n)-> b n', b = b)

    def decode(self, img_seq, hs, hms):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(math.sqrt(n)))
        img = self.model.decode(z, hs, hms)

        # img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def decode_debug(self, img_seq, hs, hms):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(math.sqrt(n)))
        #img = self.model.decode(z, hs, hms)
        img=None

        # img = (img.clamp(-1., 1.) + 1) * 0.5
        return img, z

    def forward(self, input, mask, hs, hms, stage=None):
        if stage == 'encode':
            return self.get_codebook_indices(input, mask)
        elif stage == 'decode':
            return self.decode(input, hs, hms)
