
import torch.nn as nn
import torch
from Nets.SGFANet.operators import PSPModule
from Nets.SGFANet.point_flow import PointFlowModuleWithCornerEdgeSampling
import Nets.SGFANet.resnet_d as Resnet_Deep
from Nets.SGFANet.mynn import Norm2d, Upsample
import numpy as np

import os


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        normal_layer(out_planes),
        nn.ReLU(inplace=True),
    )


class TopDown(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 reduce_dim=64, edge_points=32, corner_points=32, gated=False, gt=True):
        super(TopDown, self).__init__()

        self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []

        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            self.fpn_out_align.append(
                PointFlowModuleWithCornerEdgeSampling(fpn_dim, dim=reduce_dim, edge_points=edge_points,
                                                      corner_points=corner_points, gated=gated, gt_tag=gt))
        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1))

    def forward(self, conv_out, imagename=None):
        psp_out = self.ppm(conv_out[-1])
        f = psp_out

        fpn_feature_list = [f]
        edge_preds = []
        corner_preds = []
        out = []

        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f, edge_pred, corner_pred = self.fpn_out_align[i](
                [f, conv_x])  # f: high semantic with low resolution
            f = conv_x + f
            edge_preds.append(edge_pred)
            corner_preds.append(corner_pred)
            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        return x, edge_preds, corner_preds


class SGFANet(nn.Module):
    def __init__(self, num_classes, trunk='resnet-50-deep', criterion=None, variant='D', skip='m1', skip_num=48,reduce_dim=64, edge_points=32, corner_points=32, gated=False, gt=True):
        super(SGFANet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

        if trunk == trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()  # modified resnet
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
        else:
            raise ValueError("Not a valid network arch")

        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
        else:
            print("Not using Dilation ")

        inplane_head = 2048
        self.head = TopDown(inplane_head, num_class=num_classes, norm_layer=Norm2d,
                            reduce_dim=reduce_dim, edge_points=edge_points,
                            corner_points=corner_points, gated=gated, gt=gt)

    def forward(self, x, gts=None, imagename=None):
        x_size = x.size()
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.head([x1, x2, x3, x4], imagename)
        main_out = Upsample(x[0], x_size[2:])
        edge_preds = [Upsample(edge_pred, x_size[2:]) for edge_pred in x[1]]
        corner_preds = [Upsample(corner_pred, x_size[2:]) for corner_pred in x[2]]
        if gts != None:
            return self.criterion([main_out, edge_preds, corner_preds], gts), main_out
        else:
            return None, main_out


def DeepR101_SGFANet(num_classes, criterion, reduce_dim=64, edge_points=32,
                     corner_points=32, gated=False):
    """
    ResNet-50 Based Network
    """
    return SGFANet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1',
                   reduce_dim=reduce_dim,
                   edge_points=edge_points, corner_points=corner_points,
                   gated=gated)


def DeepR50_SGFANet(num_classes, criterion, reduce_dim=64, edge_points=32, corner_points=32, gated=False, gt=True):
    """
    ResNet-50 Based Network
    """
    return SGFANet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1',
                   reduce_dim=reduce_dim, edge_points=edge_points, corner_points=corner_points,
                   gated=gated, gt=gt)


if __name__ == '__main__':
    net = DeepR50_SGFANet(1, None, gated=True)
    net.training = False
    x = torch.rand(3, 3, 224, 224)
    out = net(x)
    print(out.shape)
