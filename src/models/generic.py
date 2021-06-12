import torch
import torch.nn as nn
from torch import Tensor

from guided_conv import SeparableGuidedConv2d


class Resblock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        downsample: bool = False,
        activation: bool = True
    ) -> None:

        super().__init__()

        self._downsample = downsample
        self._activation = activation

        _expansion = output_channels == 2 * input_channels
        _stride = 2 if downsample else 1

        self._conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, stride=_stride, bias=False),
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, bias=False),
            nn.BatchNorm2d(num_features=output_channels)
        )

        if downsample or _expansion:
            self._identity = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=_stride, bias=False),
                nn.BatchNorm2d(num_features=output_channels)
            )
        else:
            self._identity = nn.Identity()

    def forward(
        self,
        x: Tensor
    ) -> Tensor:

        out = self._conv(x) + self._identity(x)

        if self._activation:
            out = torch.relu(out)

        return out


class EncoderBlock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:

        super().__init__()

        self._layers = nn.Sequential(
            Resblock(
                input_channels=input_channels, output_channels=output_channels,
                downsample=True, activation=True),
            Resblock(
                input_channels=output_channels, output_channels=output_channels,
                downsample=False, activation=True)
        )

    def forward(
        self,
        x: Tensor
    ) -> Tensor:

        return self._layers(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:

        super().__init__()

        self._layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_channels, out_channels=output_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        x: Tensor
    ) -> Tensor:

        return self._layers(x)


class GuideBlock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        guidance_channels: int,
        output_channels: int
    ) -> None:

        super().__init__()
        self._cw_kgn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels + guidance_channels, out_channels=input_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channels, out_channels=9*input_channels, kernel_size=3, padding=1, bias=True)
        )

        self._dw_kgn = nn.Sequential(
            nn.Conv2d(in_channels=input_channels + guidance_channels, out_channels=input_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=input_channels, out_channels=input_channels * output_channels, kernel_size=1, padding=0)
        )


        self._guidance = SeparableGuidedConv2d(input_channels=input_channels, output_channels=output_channels, kernel_size=3)
        
        self._norm = nn.Sequential(
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU()
        )

    def forward(
        self,
        input: Tensor,
        guidance: Tensor
    ) -> Tensor:

        # Weights generation
        kgn_in = torch.cat((input, guidance), dim=1)
        cw_weights = self._cw_kgn(kgn_in)
        dw_weights = self._dw_kgn(kgn_in)

        # Guidance
        x = self._guidance(
            input=input,
            channel_wise_weights=cw_weights,
            depth_wise_weights=dw_weights
        )

        out = self._norm(x)

        return out


class GuidedEncoderBlock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        guidance_channels: int,
        output_channels: int
    ) -> None:

        super().__init__()
        
        self._res_blk = EncoderBlock(input_channels=input_channels, output_channels=output_channels)
        self._guide_blk = GuideBlock(input_channels=output_channels, guidance_channels=guidance_channels, output_channels=output_channels)  # 1/2

    def forward(
        self,
        input: Tensor,
        guidance: Tensor
    ) -> Tensor:

        x = self._res_blk(input)
        out = self._guide_blk(input=x, guidance=guidance)

        return out