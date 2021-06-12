from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities.checks import _check_same_shape


def compute_masked_rmse(
        pred: Tensor, 
        target: Tensor, 
        target_mask: Tensor
) -> Tensor:

    squared_error = torch.pow(pred - target, 2)  # elementwise squared error
    sample_mse = (squared_error * target_mask).sum() / target_mask.sum()  # MSE per sample in batch (mean over valid entries only)
    
    return torch.sqrt(sample_mse)  # RMSE per sample in batch


class AverageMaskedRMSE(Metric):
    """
    Computes the Averaged Root Mean Square Error (RMSE) with valid mask

    Arguments:
        pred: Predictions from model

        target: Ground truth values

        target_mask: mask of valid target values
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        scale: Union[int, float] = 1e3
    ) -> None:

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group
            )

        self.scale = scale
        self.add_state('sum_rmse', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(
        self, 
        pred: Tensor, 
        target: Tensor, 
        target_mask: Tensor,
        **kwargs
    ) -> None:

        _check_same_shape(pred, target)

        squared_error = torch.pow(pred - target, 2)  # elementwise squared error
        sample_mse = (squared_error * target_mask).sum(dim=(1,2,3)) / target_mask.sum(dim=(1,2,3))  # MSE per sample in batch (mean over valid entries only)
        sample_rmse = torch.sqrt(sample_mse)  # RMSE per sample in batch

        self.sum_rmse += torch.sum(sample_rmse)
        self.total += sample_rmse.numel()

    def compute(self) -> Tensor:
        return self.scale * self.sum_rmse / self.total


class AverageMaskediRMSE(Metric):
    """
    Computes the Averaged Inverse Root Mean Square Error (iRMSE) with valid mask

    Arguments:
        pred : Predictions from model
        
        target : Ground truth values
        
        target_mask : mask of valid target values
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        scale: Union[int, float] = 1e3
    ) -> None:

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group
            )

        self.scale = scale
        self.add_state("sum_irmse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, 
        pred: Tensor, 
        target: Tensor, 
        target_mask: Tensor,
        **kwargs
    ) -> None:
        
        _check_same_shape(pred, target)

        eps = torch.tensor(1e-8, device=pred.device)
        inv_squared_error = torch.pow(torch.pow(pred + eps, -1) - torch.pow(target + eps, -1), 2)  # elementwise inverse squared error
        sample_imse = (inv_squared_error * target_mask).sum(dim=(1,2,3)) / target_mask.sum(dim=(1,2,3))  # iMSE per sample in batch
        sample_irmse = torch.sqrt(sample_imse)  # iRMSE per sample  in batch

        self.sum_irmse += torch.sum(sample_irmse)
        self.total += sample_irmse.numel()

    def compute(self) -> Tensor:
        return self.scale * self.sum_irmse / self.total


class AverageMaskedMAE(Metric):
    """
    Computes the Averaged Mean Absolute Error (MAE) with valid mask

    Arguments:
        pred : Predictions from model
        
        target : Ground truth values
        
        target_mask : mask of valid target values
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        scale: Union[int, float] = 1e3
    ) -> None:

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group
            )

        self.scale = scale
        self.add_state("sum_mae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, 
        pred: Tensor, 
        target: Tensor, 
        target_mask: Tensor,
        **kwargs
    ) -> None:

        _check_same_shape(pred, target)

        abs_error = torch.abs(pred - target)  # elementwise absolute error
        sample_mae = (abs_error * target_mask).sum(dim=(1,2,3)) / target_mask.sum(dim=(1,2,3))  # iMSE per sample in batch

        self.sum_mae += torch.sum(sample_mae)
        self.total += sample_mae.numel()

    def compute(self) -> Tensor:
        return self.scale * self.sum_mae / self.total


class AverageMaskediMAE(Metric):
    """
    Computes the Averaged Inverse Mean Absolute Error (iMAE) with valid mask

    Arguments:
        pred: Predictions from model

        target: Ground truth values

        target_mask: mask of valid target values
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        scale: Union[int, float] = 1e3
    ) -> None:

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group
            )

        self.scale = scale
        self.add_state("sum_imae", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, 
        pred: Tensor, 
        target: Tensor, 
        target_mask: Tensor,
        **kwargs
    ) -> None:

        _check_same_shape(pred, target)

        eps = torch.tensor(1e-8, device=pred.device)
        inv_abs_error = torch.abs(torch.pow(pred + eps, -1) - torch.pow(target + eps, -1))  # elementwise inverse absolute error
        sample_imae = (inv_abs_error * target_mask).sum(dim=(1,2,3)) / target_mask.sum(dim=(1,2,3))  # iMSE per sample in batch

        self.sum_imae += torch.sum(sample_imae)
        self.total += sample_imae.numel()

    def compute(self) -> Tensor:
        return self.scale * self.sum_imae / self.total


class AverageMaskedDeltaInliersRatio(Metric):
    """
    Computes the Delta Inliers Ratio of some order.

    Arguments:
        pred : Predictions from model
        
        target : Ground truth values
        
        target_mask : target_mask of valid target values
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        order: int = 1
    ) -> None:
        """
            order: Order of inlier ratio
        """

        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group)

        self.order = order
        self.threshold = 1.25 ** self.order
        self.add_state("sum_delta", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, 
        pred: Tensor, 
        target: Tensor, 
        target_mask: Tensor,
        **kwargs
        ) -> None:

        _check_same_shape(pred, target)

        rel_err = torch.max(pred / target, target / pred)  # elementwise relative error
        rel_err_thr = (rel_err < self.threshold).float()  # elementwise threshold
        sample_delta_ratio = (rel_err_thr * target_mask).sum(dim=(1,2,3)) / target_mask.sum(dim=(1,2,3))  # delta ratio per sample in batch

        self.sum_delta += torch.sum(sample_delta_ratio)
        self.total += sample_delta_ratio.numel()

    def compute(self) -> Tensor:
        return self.sum_delta / self.total


depth_comp_metrics = MetricCollection({
    'avg_rmse': AverageMaskedRMSE(scale=1e3),
    'avg_irmse': AverageMaskediRMSE(scale=1e3),
    'avg_mae': AverageMaskedMAE(scale=1e3),
    'avg_imae': AverageMaskediMAE(scale=1e3),
    'avg_delta_1': AverageMaskedDeltaInliersRatio(order=1),
    'avg_delta_2': AverageMaskedDeltaInliersRatio(order=2),
    'avg_delta_3': AverageMaskedDeltaInliersRatio(order=3)
})