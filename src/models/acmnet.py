from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from models.base_model import BaseModel

# Original code from authors GitHub https://github.com/sshan-zhao/ACMNet
# Small API modifications to adapt this environment.

###############################################################################
# Helper Functions
###############################################################################

def conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, relu=True):
    layers = []
    layers += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)]
    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size=4, padding=1, stride=2, relu=True):
    layers = []
    layers += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)]
    if relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


##############################################################################
# Classes
##############################################################################
class MLP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.ln1 = nn.Linear(in_channels, in_channels//2)
        self.ln2 = nn.Linear(in_channels//2, 1)

    def forward(self, x):

       return self.ln2(F.leaky_relu(self.ln1(x), 0.2))


class CoAttnGPBlock(nn.Module):

    def __init__(self, in_channels=64, out_channels=64, downsample=False):
        super().__init__()

        if downsample:
            stride = 2
        else:
            stride = 1
        
        # Depth path
        self.d_conv0 = conv2d(in_channels, out_channels, stride=stride)
        self.d_conv1 = conv2d(in_channels, out_channels, stride=stride, relu=False)
        self.d_mlp = MLP(out_channels*2+3)  # output is hardcoded to 1 channel
        self.d_bias = nn.Parameter(torch.zeros(out_channels))
        self.d_conv2 = conv2d(out_channels, out_channels, relu=False)

        # RGB path
        self.r_conv0 = conv2d(in_channels, out_channels, stride=stride)
        self.r_conv1 = conv2d(in_channels, out_channels, stride=stride, relu=False)
        self.r_conv2 = conv2d(out_channels, out_channels, relu=False)
        self.r_mlp = MLP(out_channels*2+3)  # output is hardcoded to 1 channel
        self.r_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        rgb: Tensor,
        sdepth: Tensor,
        pc_idx: Tensor,
        nbrs_idx: Tensor,
        nbrs_disp: Tensor
    ):
        """
        Arguments:
            input : (B,C,H,W) tensor containing a batch of feature maps which are the input to the guided concolution
            
            pc_idx : (B,1,Ns) tensor that contains the points indices, range [0,HW-1]

            nbrs_idx : (B,1,Ns,Ks) tensor  that contains the neighbors indices, range [0,HW-1]

            nbrs_disp : (B,3,Ns,Ks) tensor that contain the 3d displacement vector between a point and its neighbors

        Return:
            output : a [B,C,H,W] tensor containing the output of gcc block
        """

        d_feat0 = self.d_conv0(sdepth)
        d_feat1 = self.d_conv1(sdepth)

        r_feat0 = self.r_conv0(rgb)
        r_feat1 = self.r_conv1(rgb)

        B, C, H, W = r_feat1.shape
        _, _, N, K = nbrs_idx.shape

        d_sfeat = torch.gather(input=d_feat0.view(B,C,-1), dim=2, index=pc_idx.repeat(1,C,1))  # (B,C,N)
        d_nnfeat = torch.gather(input=d_feat0.view(B,C,-1), dim=2, index=nbrs_idx.repeat(1,C,1,1).view(B,C,-1)).view(B,C,N,K)  # (B,C,N,K)

        r_sfeat = torch.gather(input=r_feat0.view(B,C,-1), dim=2, index=pc_idx.repeat(1,C,1))  # (B,C,N)
        r_nnfeat = torch.gather(input=d_feat0.view(B,C,-1), dim=2, index=nbrs_idx.repeat(1,C,1,1).view(B,C,-1)).view(B,C,N,K)  # (B,C,N,K)

        d_feat_dist = d_nnfeat - d_sfeat.unsqueeze(-1)  # (B,C,N,K)
        r_feat_dist = r_nnfeat - r_sfeat.unsqueeze(-1)  # (B,C,N,K)
        feats = torch.cat((d_feat_dist, r_feat_dist, nbrs_disp), 1).view(B,2*C+3,-1).permute(0,2,1).contiguous()  # (B,C,NK)->(B,NK,2C+3)

        d_attn = torch.softmax(self.d_mlp(feats).view(B,-1,K,1), 2).permute(0, 3, 1, 2)  # (B,NK,2C+3)->(B,N,K,1)->(B,1,N,K)
        r_attn = torch.softmax(self.r_mlp(feats).view(B,-1,K,1), 2).permute(0, 3, 1, 2)  # (B,NK,2C+3)->(B,N,K,1)->(B,1,N,K)

        d_feat = torch.sum(d_attn * d_nnfeat, 3) + self.d_bias.view(1,C,1)  # (B,C,N)
        r_feat = torch.sum(r_attn * r_nnfeat, 3) + self.r_bias.view(1,C,1)  # (B,C,N)

        d_feat_new = torch.zeros_like(r_feat1.view(B,C,-1)).scatter(dim=2, index=pc_idx.repeat(1,C,1), src=d_feat).reshape(B,C,H,W)  # (B,C,H,W)
        r_feat_new = torch.zeros_like(r_feat1.view(B,C,-1)).scatter(dim=2, index=pc_idx.repeat(1,C,1), src=r_feat).reshape(B,C,H,W)  # (B,C,H,W)

        masks = torch.zeros_like(r_feat1.view(B,C,-1)).scatter(dim=2, index=pc_idx.repeat(1,1,1), src=torch.ones_like(sdepth).view(B,C,-1)).reshape(B,C,H,W).contiguous()  # (B,C,H,W)
        d_feat0 = (1 - masks) * d_feat0 + d_feat_new
        r_feat0 = (1 - masks) * r_feat0 + r_feat_new

        d_feat2 = self.d_conv2(d_feat0)
        r_feat2 = self.r_conv2(r_feat0)

        return F.relu_(d_feat2+d_feat1), F.relu_(r_feat2+r_feat1)


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, channels=64, downsample=False):
        super().__init__()

        if downsample:
            stride = 2
        else:
            stride = 1
        self.conv0 = conv2d(in_channels, channels, stride=stride)
        self.conv1 = conv2d(in_channels, channels, stride=stride, relu=False)
        self.conv2 = conv2d(channels, channels, relu=False)

    def forward(self, feat):

        feat0 = self.conv0(feat)
        feat1 = self.conv1(feat)

        feat2 = self.conv2(feat0)

        return F.relu_(feat2+feat1)


class ACMNet(BaseModel):
    def __init__(self, num_channels=64, **kwargs):

        super().__init__()
        if 'depth_scale' in kwargs:
            self._depth_scale = kwargs.get('depth_scale')
        else:
            self._depth_scale = 1

        self.d_conv00 = conv2d(1, 32)
        self.d_conv01 = conv2d(32, 32)
        self.r_conv00 = conv2d(3, 32)
        self.r_conv01 = conv2d(32, 32)

        self.cpblock10 = CoAttnGPBlock(32, num_channels, True)
        self.cpblock11 = CoAttnGPBlock(num_channels, num_channels, False)
        self.cpblock20 = CoAttnGPBlock(num_channels, num_channels, True)
        self.cpblock21 = CoAttnGPBlock(num_channels, num_channels, False)
        self.cpblock30 = CoAttnGPBlock(num_channels, num_channels, True)
        self.cpblock31 = CoAttnGPBlock(num_channels, num_channels, False)

        self.d_gate4 = conv2d(num_channels, num_channels, relu=False)
        self.d_resblock40 = ResBlock(num_channels*2, num_channels, False)
        self.d_resblock41 = ResBlock(num_channels, num_channels, False)
        self.d_deconv3 = deconv(num_channels, num_channels)
        self.d_gate3 = conv2d(num_channels, num_channels, relu=False)
        self.d_resblock50 = ResBlock(num_channels*3, num_channels, False)
        self.d_resblock51 = ResBlock(num_channels, num_channels, False)
        self.d_deconv2 = deconv(num_channels, num_channels)
        self.d_gate2 = conv2d(num_channels, num_channels, relu=False)
        self.d_resblock60 = ResBlock(num_channels*3, num_channels, False)
        self.d_resblock61 = ResBlock(num_channels, num_channels, False)
        self.d_deconv1 = deconv(num_channels, num_channels)
        self.d_gate1 = conv2d(num_channels, 32, relu=False)
        self.d_last_conv = conv2d(num_channels+64, 32)
        self.d_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.r_gate4 = conv2d(num_channels, num_channels, relu=False)
        self.r_resblock40 = ResBlock(num_channels*2, num_channels, False)
        self.r_resblock41 = ResBlock(num_channels, num_channels, False)
        self.r_deconv3 = deconv(num_channels, num_channels)
        self.r_gate3 = conv2d(num_channels, num_channels, relu=False)
        self.r_resblock50 = ResBlock(num_channels*3, num_channels, False)
        self.r_resblock51 = ResBlock(num_channels, num_channels, False)
        self.r_deconv2 = deconv(num_channels, num_channels)
        self.r_gate2 = conv2d(num_channels, num_channels, relu=False)
        self.r_resblock60 = ResBlock(num_channels*3, num_channels, False)
        self.r_resblock61 = ResBlock(num_channels, num_channels, False)
        self.r_deconv1 = deconv(num_channels, num_channels)
        self.r_gate1 = conv2d(num_channels, 32, relu=False)
        self.r_last_conv = conv2d(num_channels+64, 32)
        self.r_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.f_conv4_1 = conv2d(num_channels*2, num_channels, relu=False)
        self.f_conv4_2 = nn.Sequential(
            conv2d(num_channels, num_channels, stride=2), 
            conv2d(num_channels, num_channels, relu=False)
            )
        self.f_deconv3 = deconv(num_channels, num_channels)
        self.f_conv3_1 = conv2d(num_channels*3, num_channels, relu=False)
        self.f_conv3_2 = nn.Sequential(conv2d(num_channels, num_channels, stride=2), conv2d(num_channels, num_channels, relu=False))
        self.f_deconv2 = deconv(num_channels, num_channels)
        self.f_conv2_1 = conv2d(num_channels*3, num_channels, relu=False)
        self.f_conv2_2 = nn.Sequential(conv2d(num_channels, num_channels, stride=2), conv2d(num_channels, num_channels, relu=False))
        self.f_deconv1 = deconv(num_channels, num_channels)
        self.f_conv1_1 = conv2d(num_channels+64, 32, relu=False)
        self.f_conv1_2 = nn.Sequential(conv2d(32, 32, stride=2), conv2d(32, 32, relu=False))
        self.f_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        
    def forward(
        self,
        rgb: Tensor,
        sdepth: Tensor,
        pc_idx: List,
        nbrs_idx: List,
        nbrs_disp: List,
        **kwargs
    ) -> Tuple:
        """
        Arguments:
            rgb : (B,C,H,W) tensor containing a batch of RGB images
            
            sdepth : (B,1,H,W) tensor containing a batch of Sparse Depth Maps
            
            pc_idx : list of (B,1,Ns) tensor per scale that contains the points indices, range [0,HW-1]

            nbrs_idx : list of (B,1,Ns,Ks) tensor per scale that contains the neighbors indices, range [0,HW-1]

            nbrs_disp : list of (B,3,Ns,Ks) tensor per scale that contain the 3d displacement vector between a point and its neighbors

        Return:
            output : a tuple of predicted depth maps

                f_out : (B,1,H,W) depth map tensor from feature fusion branch

                d_out : (B,1,H,W) depth map tensor from depth branch

                r_out : (B,1,H,W) depth map tensor from rgb branch

        """

        ###########
        # ENCODER #
        ###########

        # Scale 0
        d_feat0 = self.d_conv01(self.d_conv00(sdepth/self._depth_scale))
        r_feat0 = self.r_conv01(self.r_conv00(rgb))

        # CGPM - Scale 1
        d_feat1, r_feat1 = self.cpblock10(r_feat0, d_feat0, pc_idx[0], nbrs_idx[0], nbrs_disp[0])
        d_feat1, r_feat1 = self.cpblock11(r_feat1, d_feat1, pc_idx[0], nbrs_idx[0], nbrs_disp[0])

        # CGPM -  Scale 2
        d_feat2, r_feat2 = self.cpblock20(r_feat1, d_feat1, pc_idx[1], nbrs_idx[1], nbrs_disp[1])
        d_feat2, r_feat2 = self.cpblock21(r_feat2, d_feat2, pc_idx[1], nbrs_idx[1], nbrs_disp[1])

        # CGPM -  Scale 3
        d_feat3, r_feat3 = self.cpblock30(r_feat2, d_feat2, pc_idx[2], nbrs_idx[2], nbrs_disp[2])
        d_feat3, r_feat3 = self.cpblock31(r_feat3, d_feat3, pc_idx[2], nbrs_idx[2], nbrs_disp[2])

        ###########
        # DECODER #
        ###########
        d_gate4 = torch.sigmoid(self.d_gate4(d_feat3))
        d_feat = self.d_resblock40(torch.cat([d_feat3, d_gate4*r_feat3], 1))
        d_feat = self.d_resblock41(d_feat)

        r_gate4 = torch.sigmoid(self.r_gate4(r_feat3))
        r_feat = self.r_resblock40(torch.cat([r_feat3, r_gate4*d_feat3], 1))
        r_feat = self.r_resblock41(r_feat)

        f_feat = self.f_conv4_1(torch.cat([d_feat, r_feat], 1))
        f_feat_res = F.interpolate(self.f_conv4_2(F.relu(f_feat)), scale_factor=2, mode='bilinear')
        f_feat = self.f_deconv3(F.relu_(f_feat + f_feat_res))

        d_ufeat3 = self.d_deconv3(d_feat)
        r_ufeat3 = self.r_deconv3(r_feat)

        d_gate3 = torch.sigmoid(self.d_gate3(d_ufeat3))
        d_feat = self.d_resblock50(torch.cat([d_feat2, d_ufeat3, d_gate3*r_feat2], 1))
        d_feat = self.d_resblock51(d_feat)

        r_gate3 = torch.sigmoid(self.r_gate3(r_ufeat3))
        r_feat = self.r_resblock50(torch.cat([r_feat2, r_ufeat3, r_gate3*d_feat2], 1))
        r_feat = self.r_resblock51(r_feat)

        f_feat = self.f_conv3_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv3_2(F.relu(f_feat)), scale_factor=2, mode='bilinear')
        f_feat = self.f_deconv2(F.relu_(f_feat + f_feat_res))

        d_ufeat2 = self.d_deconv2(d_feat)
        r_ufeat2 = self.r_deconv2(r_feat)

        d_gate2 = torch.sigmoid(self.d_gate2(d_ufeat2))
        d_feat = self.d_resblock60(torch.cat([d_feat1, d_ufeat2, d_gate2*r_feat1], 1))
        d_feat = self.d_resblock61(d_feat)

        r_gate2 = torch.sigmoid(self.r_gate2(r_ufeat2))
        r_feat = self.r_resblock60(torch.cat([r_feat1, r_ufeat2, r_gate2*d_feat1], 1))
        r_feat = self.r_resblock61(r_feat)

        f_feat = self.f_conv2_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv2_2(F.relu(f_feat)), scale_factor=2, mode='bilinear')
        f_feat = self.f_deconv1(F.relu_(f_feat + f_feat_res))

        d_ufeat1 = self.d_deconv1(d_feat)
        r_ufeat1 = self.r_deconv1(r_feat)

        d_gate1 = torch.sigmoid(self.d_gate1(d_ufeat1))
        d_feat = torch.cat((d_feat0, d_ufeat1, d_gate1*r_feat0), 1)

        r_gate1 = torch.sigmoid(self.r_gate1(r_ufeat1))
        r_feat = torch.cat((r_feat0, r_ufeat1, r_gate1*d_feat0), 1)

        d_feat = self.d_last_conv(d_feat)
        r_feat = self.r_last_conv(r_feat)

        f_feat = self.f_conv1_1(torch.cat([d_feat, r_feat, f_feat], 1))
        f_feat_res = F.interpolate(self.f_conv1_2(F.relu(f_feat)), scale_factor=2, mode='bilinear')
        f_feat = F.relu_(f_feat + f_feat_res)

        d_out = self.d_out(d_feat)
        r_out = self.r_out(r_feat)

        f_out = self.f_out(f_feat)
    
        return {
            'pred': f_out,
            'd_out': d_out,
            'r_out': r_out
            }
