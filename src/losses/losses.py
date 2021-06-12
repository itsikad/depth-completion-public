from typing import Tuple, List, Optional, Union

from utils.projections import scale_intrinsics, homography_from

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """
    Calculates MSE loss for masked pixels only.

    Arguments:
        pred : (B,1,H,W) predictions tensor

        target : (B,1,H,W) groundtruth/target tensor

        target_mask : (B,1,H,W) boolean tensor

    Return:
        loss value, scalar tensor
    """

    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        target_mask: Tensor,
        **kwargs
    ) -> Tensor:

        numel = target_mask.sum()
        mse = F.mse_loss(pred, target, reduction='none')  # element wise MSE
        masked_mse = (mse * target_mask).sum() / numel

        return masked_mse


class MaskedSmoothL1Loss(nn.Module):
    """
    Calculates Smooth L1 loss for masked pixels only.

    Arguments:
        pred : (B,1,H,W) predictions tensor

        target : (B,1,H,W) groundtruth/target tensor

        target_mask : (B,1,H,W) boolean tensor
    
    Return:
        loss value, scalar tensor
    """

    def __init__(
        self
    ) -> None:
        super().__init__()

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        target_mask: Tensor,
        **kwargs
    ) -> Tensor:

        numel = target_mask.sum()
        smooth_L1 = F.smooth_l1_loss(pred, target, reduction='none')
        masked_smooth_L1 = (smooth_L1 * target_mask).sum() / numel
        
        return masked_smooth_L1


class SmoothnessLoss(nn.Module):
    """
    Calculates Smoothness loss.

    Arguments:
        pred : (B,1,H,W) predictions tensor

        rgb : (B,3,H,W) groundtruth/target tensor

    Return:
        loss value, scalar tensor
    """

    def __init__(
        self
    ) -> None:
        super().__init__()

    def forward(
        self,
        pred: Tensor,
        **kwargs
    ) -> Tensor:

        pred_ddx, pred_ddy = self._laplacian(pred)
        smoothness = pred_ddx.abs() + pred_ddy.abs()

        return smoothness.mean()

    def _laplacian(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Tensor]:

        ddx = 2 * x[:,:,1:-1,1:-1] - x[:,:,1:-1,:-2] - x[:,:,1:-1,2:]
        ddy = 2 * x[:,:,1:-1,1:-1] - x[:,:,:-2,1:-1] - x[:,:,2:,1:-1]

        return ddx, ddy


class EdgeAwareSmoothnessLoss(nn.Module):
    """
    Calculates Edge Aware Smoothness loss.

    Arguments:
        pred : (B,1,H,W) predictions tensor

        rgb : (B,3,H,W) groundtruth/target tensor

    Return:
        loss value, scalar tensor
    """

    def __init__(
        self
    ) -> None:
        super().__init__()

    def forward(
        self,
        pred: Tensor,
        rgb: Tensor,
        **kwargs
    ) -> Tensor:

        pred_dx, pred_dy = self._gradient(pred)
        rgb_dx, rgb_dy = self._gradient(rgb)

        weights_x = torch.exp(-torch.mean(rgb_dx.abs(), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(rgb_dy.abs(), dim=1, keepdim=True))

        smoothness = torch.mean(pred_dx.abs() * weights_x) + torch.mean(pred_dy.abs() * weights_y)

        return 0.5*smoothness

    def _gradient(
        self,
        input: Tensor
    ) -> Tuple[Tensor, Tensor]:

        dx = input[:,:,:,:-1] - input[:,:,:,1:]
        dy = input[:,:,:-1,:] - input[:,:,1:,:]

        return dx, dy


class ACMNetLoss(nn.Module):
    """
    A weighted sum of L2 and Smooth L1 losses with paramter gamma.

    Arguments:
        pred : (B,1,H,W) predictions tensor from feature fusion branch

        d_out : (B,1,H,W) predictions tensor from depth branch

        r_out : (B,1,H,W) predictions tensor from rgb branch

        target : (B,1,H,W) groundtruth/target tensor

        mask : (B,1,H,W) boolean tensor

        rgb : (B,3,H,W) groundtruth/target tensor
    
    Retunr:
        scalar tensor
    """

    def __init__(
        self,
        gamma: Tuple[float,float,float] = [0.5,0.5,0.01],
        **kwargs
    ) -> None:
        """
        Arguments:
            gamma : the weight of SmoothL1Loss
        """
        super().__init__()

        self.gamma = gamma
        self.masked_mse = MaskedMSELoss()
        self.smoothness = EdgeAwareSmoothnessLoss()

    def forward(
        self, 
        pred: Tensor,
        d_out: Tensor,
        r_out: Tensor,
        target: Tensor,
        target_mask: Tensor,
        rgb: Tensor,
        **kwargs
    ) -> Tensor:

        mse_f = self.masked_mse(pred, target, target_mask)
        mse_d = self.masked_mse(d_out, target, target_mask)
        mse_r = self.masked_mse(r_out, target, target_mask)
        smoothness = self.smoothness(pred, rgb)

        return mse_f + self.gamma[0] * mse_d + self.gamma[1] * mse_r + self.gamma[2] * smoothness


class MaskedPhotometricLoss(nn.Module):
    """

    """
    def __init__(
        self
    ) -> None:
        super().__init__()

    def forward(
        self,
        img_1: Tensor,
        img_2: Tensor,
        mask: Tensor
    ) -> Tensor:

        # Compare only non black pixels
        valid_mask = mask * (torch.sum(img_1, dim=1, keepdim=True) > 0.) * (torch.sum(img_2, dim=1, keepdim=True)  > 0.)

        if valid_mask.sum() > 0:
            # Calculate absolute distance and sum over channels
            diff = F.l1_loss(img_1, img_2, reduction='none').sum(dim=1, keepdim=True)

            # Mean over valid pixels only
            loss = diff[valid_mask].mean()

            # Scale 
            scale = 255.0 if img_1.max() <= 1.0 else 1.0
            loss = scale * loss
        else:
            loss = torch.zeros(1, device=img_1.device)

        return loss


class MultiScaleMaskedPhotometricLoss(nn.Module):
    """

    """
    def __init__(
        self,
        num_scales: int
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.photometric_loss = MaskedPhotometricLoss()

    def forward(
        self,
        pred: Tensor,
        rgb: Tensor,
        rgb_near: Tensor,
        sdepth_mask: Tensor,
        intrinsics: Tensor,
        rotation: Tensor,
        translation: Tensor
    ) -> Tensor:

        height, width = rgb.shape[2::]

        loss = torch.zeros(1, device=pred.device)
        mask = (~sdepth_mask).float()

        # Create multi-scale pyrmaids
        rgb_array = self._multiscale(rgb)            
        rgb_near_array = self._multiscale(rgb_near)
        pred_array = self._multiscale(pred)
        mask_array = self._multiscale(mask)

        for scale in range(self.num_scales):
            # Average pool inputs
            rgb_ = rgb_array[scale]
            rgb_near_ = rgb_near_array[scale]
            pred_ = pred_array[scale]
            mask_ = mask_array[scale].bool()

            # Compute corresponding intrinsic paramters
            height_new, width_new = rgb_.shape[2::]
            intrinsics_new = scale_intrinsics(intrinsics=intrinsics, old_dims=(height, width), new_dims=(height_new, width_new))
            
            # Inverse warp from a nearby frame to the current frame
            warped_ = homography_from(
                rgb_near=rgb_near_,
                pred=pred_,
                rotation=rotation,
                translation=translation,
                intrinsics=intrinsics_new
            )

            # Compute photometric loss
            loss += (2 ** -(scale+1)) * self.photometric_loss(rgb_, warped_, mask_)

        return loss

    def _multiscale(
        self,
        img: Tensor
    ) -> Tuple[Tensor, ...]:

        scales = []
        for i in range(self.num_scales):
            img = self.avg_pool(img)
            scales.append(img)

        return scales

class SelfSupSparseToDenseLoss(nn.Module):
    """
    Consists of masked depth RMSE loss, photometric loss with warped RGB image and smoothness loss.

    Arguments:
        pred : (B,1,H,W) predictions tensor

        target : (B,1,H,W) groundtruth/target tensor

        mask : (B,1,H,W) boolean tensor

        rgb : (B,3,H,W) groundtruth/target tensor

        gamma : losses' weights
    
    Return:
        scalar tensor
    """

    def __init__(
        self,
        gamma: Union[List, Tuple, Tensor],
        **kwargs
    ) -> None:

        super().__init__()

        self.depth_loss = MaskedMSELoss()
        self.photometric_loss = MultiScaleMaskedPhotometricLoss(num_scales=5)
        self.smoothness_loss = SmoothnessLoss()
        self.gamma = gamma

    def forward(
        self, 
        pred: Tensor,
        sdepth: Tensor,
        sdepth_mask: Tensor,
        rgb: Tensor,
        rgb_near: Tensor,
        intrinsics: Tensor,
        rotation: Tensor,
        translation: Tensor,
        **kwargs
    ) -> Tensor:

        # Depth loss
        depth_loss = self.depth_loss(pred, sdepth, sdepth_mask)

        # Photmetric loss (training only, rotation is valid)
        photometric_loss = 0
        if rotation.numel() > 0:
            photometric_loss += self.photometric_loss(
                pred=pred, 
                rgb=rgb, 
                rgb_near=rgb_near, 
                sdepth_mask=sdepth_mask, 
                intrinsics=intrinsics, 
                rotation=rotation, 
                translation=translation
            )

        # Smoothness loss
        smoothness_loss = self.smoothness_loss(pred)

        loss = depth_loss + self.gamma[0] * photometric_loss + self.gamma[1] * smoothness_loss
        
        return loss