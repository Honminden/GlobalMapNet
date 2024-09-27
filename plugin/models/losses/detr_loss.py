import torch
from torch import nn as nn
from torch.nn import functional as F
from mmdet.models.losses import l1_loss, smooth_l1_loss
from mmdet.models.losses.utils import weighted_loss
import mmcv

from mmdet.models.builder import LOSSES
from ..utils.buffered_gaussian import buffered_gaussian_loss


@LOSSES.register_module()
class LinesL1Loss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, beta=0.5):
        """
            L1 loss. The same as the smooth L1 loss
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of loss.
        """

        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else:
            loss = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        num_points = pred.shape[-1] // 2
        loss = loss / num_points

        return loss*self.loss_weight


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def bce(pred, label, class_weight=None):
    """
        pred: B,nquery,npts
        label: B,nquery,npts
    """

    if label.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == label.size()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')

    return loss


@LOSSES.register_module()
class MasksLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MasksLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            xxx
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = bce(pred, target, weight, reduction=reduction,
                   avg_factor=avg_factor)

        return loss*self.loss_weight

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ce(pred, label, class_weight=None):
    """
        pred: B*nquery,npts
        label: B*nquery,
    """

    if label.numel() == 0:
        return pred.sum() * 0

    loss = F.cross_entropy(
        pred, label, weight=class_weight, reduction='none')

    return loss


@LOSSES.register_module()
class LenLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(LenLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            xxx
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss = ce(pred, target, weight, reduction=reduction,
                   avg_factor=avg_factor)

        return loss*self.loss_weight


@LOSSES.register_module()
class BGM_L1_MixLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, l1_loss_weight=1.0, beta=0.5, buffer_distance=1.0, buffer_mode='add', pc_range=None, 
                 loss_type='kld', fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
        """
            Buffered Gaussian loss and L1 loss.
            Args:
                reduction (str, optional): The method to reduce the loss.
                    Options are "none", "mean" and "sum".
                loss_weight (float, optional): The weight of Buffered Gaussian loss.
                l1_loss_weight (float, optional): The weight of L1 loss.
        """

        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.beta = beta
        self.buffer_distance = buffer_distance
        self.buffer_mode = buffer_mode
        self.pc_range = pc_range
        self.loss_type = loss_type
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.sqrt = sqrt # gwd kwarg: normalize

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
                shape: [bs, ...]
            target (torch.Tensor): The learning target of the prediction.
                shape: [bs, ...]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None. 
                it's useful when the predictions are not all valid.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.beta > 0:
            loss_l1 = smooth_l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor, beta=self.beta)
        
        else:
            loss_l1 = l1_loss(
                pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        
        num_points = pred.shape[-1] // 2
        loss_l1 = loss_l1 / num_points

        num_boxes = num_points - 1
        pred_temp = pred.view(-1, num_points, 2)
        target_temp = target.view(-1, num_points, 2)
        weight_temp = weight.view(-1, num_points, 2)
        loss_giou = buffered_gaussian_loss(
            pred_temp, target_temp, weight_temp[:, :-1, 0].flatten(), reduction=reduction, avg_factor=avg_factor * num_boxes, 
            pc_range=self.pc_range, buffer_distance=self.buffer_distance, buffer_mode=self.buffer_mode, 
            loss_type=self.loss_type, fun=self.fun, tau=self.tau, alpha=self.alpha, sqrt=self.sqrt)

        return loss_giou * self.loss_weight + loss_l1 * self.l1_loss_weight
