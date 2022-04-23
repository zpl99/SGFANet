""""
Code is modified based on : https://github.com/lxtGH/PFSegNets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Nets.SGFANet.mynn import Norm2d
from Nets.ground_transformer import GroundTrans


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords


class PointFlowModuleWithCornerEdgeSampling(nn.Module):
    def __init__(self, in_planes, dim=64, matcher_kernel_size=3,
                 edge_points=32, corner_points=32, gated=False, gt_tag=True):
        super(PointFlowModuleWithCornerEdgeSampling, self).__init__()
        self.dim = dim
        self.down_h = nn.Conv2d(in_planes, dim, 1)
        self.down_l = nn.Conv2d(in_planes, dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.edge_points = edge_points
        self.corner_points = corner_points
        self.gated = gated
        self.gt_tag = gt_tag
        if self.gt_tag:
            self.gt = GroundTrans(in_channels=in_planes, dimension=2)
            print("Ground Transformer")
        print(f"edge points:{self.edge_points},corner points:{self.corner_points}")
        if self.gated:
            print("weight gate is added")
            self.channel_gate = nn.Sequential(nn.Linear(in_planes, in_planes), nn.Dropout(0.1), nn.ReLU(),
                                              nn.Linear(in_planes, in_planes), nn.Sigmoid())
            self.feature_inportance = nn.Sequential(nn.Linear(in_planes, in_planes), nn.Dropout(0.1), nn.ReLU(),
                                                    nn.Linear(in_planes, 1), nn.Sigmoid())
        self.edge_final = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=1, bias=False),
            Norm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=3, padding=1, bias=False)
        )
        self.corner_final = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=1, bias=False),
            Norm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=1, kernel_size=1, bias=False)
        )

    def forward(self, x):

        x_high, x_low = x  # 8,8 16,16

        stride_ratio = x_low.shape[2] / x_high.shape[2]
        N, C, H, W = x_low.shape
        N_h, C_h, H_h, W_h = x_high.shape

        # edge part
        x_high_edge = x_high
        edge_pred = self.edge_final(x_high_edge)
        point_indices, point_coords = get_uncertain_point_coords_on_grid(edge_pred,
                                                                         num_points=self.edge_points)  # torch.Size([2, K, 2])
        sample_x = point_indices % W_h * stride_ratio
        sample_y = point_indices // W_h * stride_ratio
        low_edge_indices = sample_x + sample_y * W
        low_edge_indices = low_edge_indices.unsqueeze(1).expand(-1, C, -1).long()
        high_edge_feat = point_sample(x_high, point_coords)  # torch.Size([2, 256, K])
        low_edge_feat = point_sample(x_low, point_coords)
        if self.gated:
            high_edge_feat = self.channel_gate(high_edge_feat.permute(0, 2, 1)) * high_edge_feat.permute(0, 2, 1)
            high_edge_feat = high_edge_feat.permute(0, 2, 1)
            low_edge_feat = self.channel_gate(low_edge_feat.permute(0, 2, 1)) * low_edge_feat.permute(0, 2, 1)
            low_edge_feat = low_edge_feat.permute(0, 2, 1)
        affinity_edge = torch.bmm(high_edge_feat.transpose(2, 1), low_edge_feat).transpose(2, 1)
        affinity = self.softmax(affinity_edge)
        high_edge_feat = torch.bmm(affinity, high_edge_feat.transpose(2, 1)).transpose(2, 1)
        if self.gated:
            f_i = self.feature_inportance(high_edge_feat.permute(0, 2, 1))
            f_i = f_i.permute(0, 2, 1)
            fusion_edge_feat = f_i * high_edge_feat + (1 - f_i) * low_edge_feat
        else:
            fusion_edge_feat = high_edge_feat + low_edge_feat

        # corner part
        x_high_corner = x_high
        corner_pred = self.corner_final(x_high_corner)
        corner_point_indices, corner_point_coords = get_uncertain_point_coords_on_grid(corner_pred,
                                                                                       num_points=self.corner_points)
        corner_sample_x = corner_point_indices % W_h * stride_ratio
        corner_sample_y = corner_point_indices // W_h * stride_ratio
        low_corner_indices = corner_sample_x + corner_sample_y * W
        low_corner_indices = low_corner_indices.unsqueeze(1).expand(-1, C, -1).long()
        high_corner_feat = point_sample(x_high, corner_point_coords)
        low_corner_feat = point_sample(x_low, corner_point_coords)
        if self.gated:
            high_corner_feat = self.channel_gate(high_corner_feat.permute(0, 2, 1)) * high_corner_feat.permute(0, 2, 1)
            high_corner_feat = high_corner_feat.permute(0, 2, 1)
            low_corner_feat = self.channel_gate(low_corner_feat.permute(0, 2, 1)) * low_corner_feat.permute(0, 2, 1)
            low_corner_feat = low_corner_feat.permute(0, 2, 1)
        affinity_corner = torch.bmm(high_corner_feat.transpose(2, 1), low_corner_feat).transpose(2, 1)
        affinity_corner = self.softmax(affinity_corner)
        high_corner_feat = torch.bmm(affinity_corner, high_corner_feat.transpose(2, 1)).transpose(2, 1)
        if self.gated:
            f_i_c = self.feature_inportance(high_corner_feat.permute(0, 2, 1))
            f_i_c = f_i_c.permute(0, 2, 1)
            fusion_corner_feat = f_i_c * high_corner_feat + (1 - f_i_c) * low_corner_feat
        else:
            fusion_corner_feat = high_corner_feat + low_corner_feat

        # GT Part
        if self.gt_tag == True:
            x_low = self.gt(x_low, x_high)

        final_features = x_low.reshape(N, C, H * W).scatter(2, low_edge_indices, fusion_edge_feat)  # edge
        final_features = final_features.scatter(2, low_corner_indices, fusion_corner_feat)  # corner
        final_features = final_features.view(N, C, H, W)  #
        return final_features, edge_pred, corner_pred
        # end
