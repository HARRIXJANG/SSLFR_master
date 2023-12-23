from chamferdist import ChamferDistance, knn_points
import torch.nn as nn
import torch
from typing import Optional
import warnings

class ChamferDistance_Mod(ChamferDistance):
    def __init__(self):
        super(ChamferDistance_Mod, self).__init__()
    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
        bidirectional: Optional[bool] = False,
        reverse: Optional[bool] = False,
        reduction: Optional[str] = "mean",
    ):
        #super(ChamferDistance_Mod, self).forward(source_cloud, target_cloud, bidirectional, reverse, reduction)
        if not isinstance(source_cloud, torch.Tensor):
            raise TypeError(
                "Expected input type torch.Tensor. Got {} instead".format(type(source_cloud))
            )
        if not isinstance(target_cloud, torch.Tensor):
            raise TypeError(
                "Expected input type torch.Tensor. Got {} instead".format(type(target_cloud))
            )
        if source_cloud.device != target_cloud.device:
            raise ValueError(
                "Source and target clouds must be on the same device. "
                f"Got {source_cloud.device} and {target_cloud.device}."
            )

        batchsize_source, lengths_source, dim_source = source_cloud.shape
        batchsize_target, lengths_target, dim_target = target_cloud.shape

        lengths_source = (
            torch.ones(batchsize_source, dtype=torch.long, device=source_cloud.device)
            * lengths_source
        )
        lengths_target = (
            torch.ones(batchsize_target, dtype=torch.long, device=target_cloud.device)
            * lengths_target
        )

        #chamfer_dist = None

        if batchsize_source != batchsize_target:
            raise ValueError(
                "Source and target pointclouds must have the same batchsize."
            )
        if dim_source != dim_target:
            raise ValueError(
                "Source and target pointclouds must have the same dimensionality."
            )
        if bidirectional and reverse:
            warnings.warn(
                "Both bidirectional and reverse set to True. "
                "bidirectional behavior takes precedence."
            )
        if reduction != "sum" and reduction != "mean":
            raise ValueError('Reduction must either be "sum" or "mean".')

        source_nn = knn_points(
            source_cloud,
            target_cloud,
            lengths1=lengths_source,
            lengths2=lengths_target,
            K=1,
        )

        target_nn = None
        if reverse or bidirectional:
            target_nn = knn_points(
                target_cloud,
                source_cloud,
                lengths1=lengths_target,
                lengths2=lengths_source,
                K=1,
            )

        # Forward Chamfer distance (batchsize_source, lengths_source)
        chamfer_forward = source_nn.dists[..., 0]
        chamfer_backward = None
        if reverse or bidirectional:
            # Backward Chamfer distance (batchsize_source, lengths_source)
            chamfer_backward = target_nn.dists[..., 0]

        chamfer_forward = chamfer_forward.sum(1)  # (batchsize_source,)
        if reverse or bidirectional:
            chamfer_backward = chamfer_backward.sum(1)  # (batchsize_target,)

        if reduction == "sum":
            chamfer_forward = chamfer_forward.sum()  # (1,)
            if reverse or bidirectional:
                chamfer_backward = chamfer_backward.sum()  # (1,)
        elif reduction == "mean":
            chamfer_forward = chamfer_forward.mean()  # (1,)
            if reverse or bidirectional:
                chamfer_backward = chamfer_backward.mean()  # (1,)

        if bidirectional:
            return chamfer_forward + chamfer_backward, source_nn[1].squeeze(), target_nn[1].squeeze()
        if reverse:
            return chamfer_backward, target_nn[1].squeeze()

        return chamfer_forward, source_nn[1].squeeze()

class ChamferDistance_only(nn.Module):
    def __init__(self):
        super(ChamferDistance_only, self).__init__()

    def forward(
            self,
            source_cloud: torch.Tensor,  # B, G, 3
            target_cloud: torch.Tensor,  # B, G, 3
            bidirectional: bool = True,
            mode: str = 'mean'  # 'sum'
    ):
        B, G, _ = source_cloud.shape
        source_cloud_coor = source_cloud[:, :, :3]
        target_cloud_coor = target_cloud[:, :, :3]
        chamferdistance, _, _ = ChamferDistance_Mod()(source_cloud_coor, target_cloud_coor, bidirectional=bidirectional)
        if mode == 'mean':
            return chamferdistance/G
        else:
            return chamferdistance

class MSE_CrossEntropy(nn.Module):
    def __init__(self):
        super(MSE_CrossEntropy, self).__init__()

    def forward(
            self,
            source_cloud: torch.Tensor,  # B, G, 12
            target_cloud: torch.Tensor,  # B, G, 12
            a: float = 0.25,
            b: float = 0.25,
            c: float = 0.125,
            foe: bool = True,
    ):
        B, G, _ = source_cloud.shape
        source_cloud_normal = source_cloud[:, :, :3]
        target_cloud_normal = target_cloud[:, :, :3]
        MSE_loss = nn.MSELoss()(source_cloud_normal, target_cloud_normal)


        source_cloud_type = source_cloud[:, :, 3:7].view(B*G, 4)
        target_cloud_type = target_cloud[:, :, 3:7].view(B*G, 4)

        cross_entropy_t = nn.CrossEntropyLoss()(source_cloud_type, target_cloud_type)

        if foe:
            source_cloud_foe = source_cloud[:, :, 7:].view(B * G, 2)
            target_cloud_foe = target_cloud[:, :, 7:].view(B * G, 2)
            cross_entropy_f = nn.CrossEntropyLoss()(source_cloud_foe, target_cloud_foe)
            loss = a * MSE_loss + b * cross_entropy_t + c * cross_entropy_f
            return loss, MSE_loss, cross_entropy_t, cross_entropy_f
        else:
            loss = a * MSE_loss + b * cross_entropy_t
            return loss, MSE_loss, cross_entropy_t