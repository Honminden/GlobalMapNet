import torch
import mmcv
from mmdet.models import weighted_loss


def x1y1x2y2_2_xy_sigma(segments, pc_range, buffer_distance=1.0, buffer_mode='add'):
    # shape [num_samples, num_pts, 2]
    segments_denorm = torch.cat([segments[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0], 
                                 segments[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]], dim=-1)

    x1y1, x2y2 = segments_denorm[..., :-1, :], segments_denorm[..., 1:, :]
    xy = (x1y1 + x2y2) / 2.0 # [num_samples, num_pts, 2]
    delta = (x2y2 - x1y1).clamp(-1e7, 1e7)
    _shape = xy.shape

    l = torch.linalg.norm(delta, dim=-1).clamp(1e-7, 1e7)
    if buffer_mode == 'mul':
        w = l + 2.0 * l * buffer_distance
        h = 2.0 * l * buffer_distance
    else:
        w = l + 2.0 * buffer_distance
        h = 2.0 * torch.ones_like(l) * buffer_distance

    delta_x, delta_y = delta[..., 0], delta[..., 1]

    R = torch.stack((delta_x / l, -delta_y / l, delta_y / l, delta_x / l), dim=-1).unflatten(-1, (2, 2)).flatten(0, 1)
    zeros = torch.zeros_like(w)
    S = torch.stack((w / 2.0, zeros, zeros, h / 2.0), dim=-1).unflatten(-1, (2, 2)).flatten(0, 1)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2)) # [num_samples, num_pts, 2, 2]

    return xy, sigma


def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True, is_aligned=True):
    """Gaussian Wasserstein distance loss.
    Derivation and simplification:
        Given any positive-definite symmetrical 2*2 matrix Z:
            :math:`Tr(Z^{1/2}) = λ_1^{1/2} + λ_2^{1/2}`
        where :math:`λ_1` and :math:`λ_2` are the eigen values of Z
        Meanwhile we have:
            :math:`Tr(Z) = λ_1 + λ_2`

            :math:`det(Z) = λ_1 * λ_2`
        Combination with following formula:
            :math:`(λ_1^{1/2}+λ_2^{1/2})^2 = λ_1+λ_2+2 *(λ_1 * λ_2)^{1/2}`
        Yield:
            :math:`Tr(Z^{1/2}) = (Tr(Z) + 2 * (det(Z))^{1/2})^{1/2}`
        For gwd loss the frustrating coupling part is:
            :math:`Tr((Σ_p^{1/2} * Σ_t * Σp^{1/2})^{1/2})`
        Assuming :math:`Z = Σ_p^{1/2} * Σ_t * Σ_p^{1/2}` then:
            :math:`Tr(Z) = Tr(Σ_p^{1/2} * Σ_t * Σ_p^{1/2})
            = Tr(Σ_p^{1/2} * Σ_p^{1/2} * Σ_t)
            = Tr(Σ_p * Σ_t)`
            :math:`det(Z) = det(Σ_p^{1/2} * Σ_t * Σ_p^{1/2})
            = det(Σ_p^{1/2}) * det(Σ_t) * det(Σ_p^{1/2})
            = det(Σ_p * Σ_t)`
        and thus we can rewrite the coupling part as:
            :math:`Tr(Z^{1/2}) = (Tr(Z) + 2 * (det(Z))^{1/2})^{1/2}`
            :math:`Tr((Σ_p^{1/2} * Σ_t * Σ_p^{1/2})^{1/2})
            = (Tr(Σ_p * Σ_t) + 2 * (det(Σ_p * Σ_t))^{1/2})^{1/2}`

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    if is_aligned:
        _shape = xy_p.shape # [num_samples, num_pts, 2]

        xy_p = xy_p.reshape(-1, 2)
        xy_t = xy_t.reshape(-1, 2)
        Sigma_p = Sigma_p.reshape(-1, 2, 2)
        Sigma_t = Sigma_t.reshape(-1, 2, 2)
    else:
        xy_p = xy_p.transpose(0, 1).unsqueeze(2) # [num_pts, num_query, 1, 2]
        xy_t = xy_t.transpose(0, 1).unsqueeze(1) # [num_pts, 1, num_gts * num_orders, 2]
        Sigma_p = Sigma_p.transpose(0, 1).unsqueeze(2) # [num_pts, num_query, 1, 2, 2]
        Sigma_t = Sigma_t.transpose(0, 1).unsqueeze(1) # [num_pts, 1, num_gts * num_orders, 2, 2]

    # is aligned: [num_samples * num_pts] F: [num_pts, num_query, num_gts * num_orders]
    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = torch.matmul(Sigma_p, Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    if is_aligned:
        distance = distance.reshape(_shape[:-1])

    return postprocess(distance, fun=fun, tau=tau)


def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True, is_aligned=True):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    if is_aligned:
        _shape = xy_p.shape # [num_samples, num_pts, 2]

        xy_p = xy_p.reshape(-1, 2)
        xy_t = xy_t.reshape(-1, 2)
        Sigma_p = Sigma_p.reshape(-1, 2, 2)
        Sigma_t = Sigma_t.reshape(-1, 2, 2)
    else:
        xy_p = xy_p.transpose(0, 1).unsqueeze(2) # [num_pts, num_query, 1, 2]
        xy_t = xy_t.transpose(0, 1).unsqueeze(1) # [num_pts, 1, num_gts * num_orders, 2]
        Sigma_p = Sigma_p.transpose(0, 1).unsqueeze(2) # [num_pts, num_query, 1, 2, 2]
        Sigma_t = Sigma_t.transpose(0, 1).unsqueeze(1) # [num_pts, 1, num_gts * num_orders, 2, 2]

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).unflatten(-1, (2, 2))
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    # is aligned: [num_samples * num_pts] F: [num_pts, num_query, num_gts * num_orders]
    xy_distance = 0.5 * torch.matmul(torch.matmul(dxy.transpose(-2, -1), Sigma_p_inv), dxy).squeeze(-1).squeeze(-1)

    whr_distance = 0.5 * torch.matmul(Sigma_p_inv, Sigma_t).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(1e-7).sqrt()

    if is_aligned:
        distance = distance.reshape(_shape[:-1])

    return postprocess(distance, fun=fun, tau=tau)


def postprocess(distance, fun='log1p', tau=1.0):
    """Convert distance to loss.

    Args:
        distance (torch.Tensor)
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance.clamp(1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def buffered_gaussian_loss(pred, target, pc_range, eps=1e-7, buffer_distance=1.0, buffer_mode='add', loss_type='kld', fun='log1p', tau=1.0, alpha=1.0, sqrt=True, is_aligned=True):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_pts, num_coords]

    Return:
        Tensor: Loss tensor.
    """
    if target.numel() == 0:
        return pred.sum() * 0

    if isinstance(buffer_distance, list):
        buffer_distances = buffer_distance
    else:
        buffer_distances = [buffer_distance]

    ious_bufs = list()
    for buffer_distance in buffer_distances:
        xy_pred, sigma_pred = x1y1x2y2_2_xy_sigma(pred, pc_range, buffer_distance, buffer_mode)
        xy_target, sigma_target = x1y1x2y2_2_xy_sigma(target, pc_range, buffer_distance, buffer_mode)
        if loss_type == 'gwd':
            ious = gwd_loss((xy_pred, sigma_pred), (xy_target, sigma_target), fun=fun, tau=tau, alpha=alpha, normalize=sqrt, is_aligned=is_aligned)
        elif loss_type == 'kld':
            ious = kld_loss((xy_pred, sigma_pred), (xy_target, sigma_target), fun=fun, tau=tau, alpha=alpha, sqrt=sqrt, is_aligned=is_aligned)
        else:
            raise NotImplementedError()
        ious_bufs.append(ious)
    ious_bufs = torch.stack(ious_bufs, dim=0).mean(dim=0)

    if is_aligned:
        loss = ious_bufs.flatten()
        return loss
    else:
        return ious_bufs.mean(dim=0)