from typing import List, Tuple, Callable, Union

import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
import torchvision.transforms.functional as TF


def apply_transform(
    transform: Callable,
    imgs: Tuple,
    **kwargs
) -> Tuple:

    """
    Applies a transform on a sequence of images (PIL, Tensor or np.ndarray), ignoring None's
    """

    return tuple([transform(img, **kwargs) if img is not None else None for img in imgs])


class IdentityTransform(object):

    def __init__(
        self
    ) -> None:

        self.transform = transforms.Lambda(lambda x: x)
    
    def __call__(
        self,
        imgs,
        **kwargs
    ) -> Tensor:

        """
        Arguments:
            imgs : tuple of images (PIL, Tensors or numpy.ndarray) to transform.
        
        Returns:
            Tuple of images (identical to input images)
        """

        return apply_transform(self.transform, imgs)


class ToNumpy:
    """
    Converts a PIL image to numpy array.
    Note: required before converting PIL float image to tensor
          which is not supported natively in transforms.ToTensor()
    """

    def __call__(
        self,
        imgs,
        **kwargs
    ) -> Tuple:

        """
        Convert to numpy

        Arguments:
            imgs : tuple of images (PIL) to transform.
        
        Returns:
            Tuple of numpy images
        """

        return apply_transform(np.array, imgs, **kwargs)


class ToTensor:
    """
    Similar to torchvision.transforms.ToTensor class with an additional support for sequence of inputs including Nones.
    """

    def __call__(
        self,
        imgs: Tuple
    ) -> Tuple:

        """
        Arguments:
            imgs : tuple of images (PIL or numpy.ndarray) to transform.

        Returns:
            Tuple of tensor images
        """

        return apply_transform(TF.to_tensor, imgs)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Grayscale(nn.Module):
    """
    Similar to torchvision.transforms.Grayscale class with an additional support for sequence of inputs, including Nones.
    """

    def __init__(
        self,
        num_output_channels: int = 1
    ) -> None:

        super().__init__()
        self.num_output_channels = num_output_channels

    def forward(
        self,
        imgs: Tuple
    ) -> Tuple:

        """
        Arguments:
            imgs : tuple of images (PIL or Tensor) to transform.

        Returns:
            Tuple of grayscale images.
        """

        return apply_transform(TF.rgb_to_grayscale, imgs, num_output_channels=self.num_output_channels)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)


class CenterCrop(nn.Module):
    """
    Similar to torchvision.transforms.CenterCrop class with an additional support for sequence of inputs, including Nones.
    """

    def __init__(
        self,
        output_size: Union[int, List, Tuple]
    ) -> None:

        super().__init__()
        self.output_size = transforms.transforms._setup_size(output_size, error_msg="Please provide only two dimensions (h, w) for size.")

    def forward(
        self,
        imgs: Tuple
    ) -> Tuple:

        """
        Arguments:
            imgs : tuple of images (PIL or numpy.ndarray) to crop.

        Returns:
            Tuple of cropped images.
        """

        return apply_transform(TF.center_crop, imgs, output_size=self.output_size)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(output_size={0})'.format(self.output_size)


class BottomCrop(nn.Module):
    """
    Applies center crop on the horizontal axis and bottom crop on the vertical axis.
    """

    def __init__(
        self,
        output_size: Union[int, List, Tuple]
    ) -> None:

        super().__init__()
        self.output_size = transforms.transforms._setup_size(output_size, error_msg="Please provide only two dimensions (h, w) for size.")

    def forward(
        self,
        imgs: Tuple
    ) -> Tuple:

        """
        Arguments:
            imgs : tuple of images (PIL or numpy.ndarray) to crop.

        Returns:
            Tuple of cropped images.
        """

        image_width, image_height = TF._get_image_size(imgs[0])
        crop_height, crop_width = self.output_size

        if crop_width == image_width and crop_height == image_height:
            crop_top, crop_left = 0, 0
        elif crop_width <= image_width and crop_height <= image_height:
            crop_top = image_height - crop_height
            crop_left = int(round((image_width - crop_width) / 2.))

        return apply_transform(TF.crop, imgs, top=crop_top, left=crop_left, height=crop_height, width=crop_width)


class RandomHorizontalFlip(nn.Module):
    """
    Similar to torchvision.transforms.RandomHorizontalFlip class with an additional support for sequence of inputs including Nones.
    """

    def __init__(
        self,
        p: float = 0.5
    ) -> None:

        super().__init__()
        self.p = p

    def forward(
        self,
        imgs: Tuple
    ) -> Tuple:

        """
        Arguments:
            imgs : tuple of images (PIL or Tensor) to transform.

        Returns:
            Tuple of randomly flipped images (same operation is applied on all inputs)
        """

        if torch.rand(1) < self.p:
            imgs = apply_transform(TF.hflip, imgs)
            
        return imgs

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ColorJitter(transforms.ColorJitter):
    """
    Override torchvision.transforms.ColorJitter forward method to add suppor for sequence of inputs, inlcuding Nones.
    """

    def forward(
        self,
        imgs: Tuple
    ) -> Tuple:

        """
        Arguments:
            imgs : tuple of images (PIL or Tensor) to transform.

        Returns:
            Tuple of randomly jittered images (same operation is applied on all inputs)
        """

        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                imgs = apply_transform(TF.adjust_brightness, imgs, brightness_factor=brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                imgs = apply_transform(TF.adjust_contrast, imgs, contrast_factor=contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                imgs = apply_transform(TF.adjust_saturation, imgs, saturation_factor=saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                imgs = apply_transform(TF.adjust_hue, imgs, hue_factor=hue_factor)

        return imgs