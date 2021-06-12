from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet

from models.base_model import BaseModel


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers


class SparseToDense(BaseModel):
    def __init__(
        self,
        **kwargs
        ):

        super().__init__()

        self.conv1_d = conv_bn_relu(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1)

        self.conv1_img = conv_bn_relu(
            in_channels=1,
            out_channels=48,
            kernel_size=3,
            stride=1,
            padding=1)

        pretrained_model = resnet.__dict__['resnet34'](pretrained=False)
        pretrained_model.apply(init_weights)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory

        # define number of intermediate channels
        num_channels = 512

        self.conv6 = conv_bn_relu(
            num_channels,
            512,
            kernel_size=3,
            stride=2,
            padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2

        self.convt5 = convt_bn_relu(
            in_channels=512,
            out_channels=256,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)
            
        self.convt4 = convt_bn_relu(
            in_channels=768,
            out_channels=128,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)

        self.convt3 = convt_bn_relu(
            in_channels=(256 + 128),
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)

        self.convt2 = convt_bn_relu(
            in_channels=(128 + 64),
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            output_padding=1)

        self.convt1 = convt_bn_relu(
            in_channels=128,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding=1)

        self.convtf = conv_bn_relu(
            in_channels=128,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bn=False,
            relu=False)

    def forward(
        self,
        gray_curr,
        sdepth,
        **kwargs
    ) -> Tensor:

        conv1_d = self.conv1_d(sdepth)
        conv1_img = self.conv1_img(gray_curr)
        conv1 = torch.cat((conv1_d, conv1_img), 1)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  # batchsize * ? * 176 * 608
        conv4 = self.conv4(conv3)  # batchsize * ? * 88 * 304
        conv5 = self.conv5(conv4)  # batchsize * ? * 44 * 152
        conv6 = self.conv6(conv5)  # batchsize * ? * 22 * 76

        # decoder
        convt5 = self.convt5(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.convt4(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.convt3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.convt2(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.convt1(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.convtf(y)

        return {'pred': 100 * y}

    def val_post_process(
        self,
        output: Dict
    ) -> Dict:

        min_distance = 0.9
        output['pred'] = F.relu(output['pred'] - min_distance) + min_distance  # the minimum range of Velodyne is around 3 feet ~= 0.9m

        return output