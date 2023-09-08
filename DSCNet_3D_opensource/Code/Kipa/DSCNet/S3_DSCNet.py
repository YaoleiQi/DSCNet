# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
from S3_DSConv import DCN_Conv


class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()

        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x


class DSCNet(nn.Module):
    def __init__(self, n_channels, n_classes, kernel_size, extend_scope, if_offset, device, number, dim):
        super(DSCNet, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        self.dim = dim

        # Unet
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0x = DCN_Conv(n_channels, self.number, 3, self.extend_scope, 0, self.if_offset, self.device)
        self.conv0y = DCN_Conv(n_channels, self.number, 3, self.extend_scope, 1, self.if_offset, self.device)
        self.conv0z = DCN_Conv(n_channels, self.number, 3, self.extend_scope, 2, self.if_offset, self.device)
        self.conv1 = EncoderConv(4*self.number, self.number)

        self.conv20 = EncoderConv(self.number, 2*self.number)
        self.conv2x = DCN_Conv(self.number, 2*self.number, 3, self.extend_scope, 0, self.if_offset, self.device)
        self.conv2y = DCN_Conv(self.number, 2*self.number, 3, self.extend_scope, 1, self.if_offset, self.device)
        self.conv2z = DCN_Conv(self.number, 2*self.number, 3, self.extend_scope, 2, self.if_offset, self.device)
        self.conv3 = EncoderConv(8*self.number, 2*self.number)

        self.conv40 = EncoderConv(2*self.number, 4*self.number)
        self.conv4x = DCN_Conv(2*self.number, 4*self.number, 3, self.extend_scope, 0, self.if_offset, self.device)
        self.conv4y = DCN_Conv(2*self.number, 4*self.number, 3, self.extend_scope, 1, self.if_offset, self.device)
        self.conv4z = DCN_Conv(2*self.number, 4*self.number, 3, self.extend_scope, 2, self.if_offset, self.device)
        self.conv5 = EncoderConv(16*self.number, 4*self.number)

        self.conv60 = EncoderConv(4*self.number, 8*self.number)
        self.conv6x = DCN_Conv(4*self.number, 8*self.number, 3, self.extend_scope, 0, self.if_offset, self.device)
        self.conv6y = DCN_Conv(4*self.number, 8*self.number, 3, self.extend_scope, 1, self.if_offset, self.device)
        self.conv6z = DCN_Conv(4*self.number, 8*self.number, 3, self.extend_scope, 2, self.if_offset, self.device)
        self.conv7 = EncoderConv(32*self.number, 8*self.number)

        self.conv120 = EncoderConv(12*self.number, 4*self.number)
        self.conv12x = DCN_Conv(12*self.number, 4*self.number, 3, self.extend_scope, 0, self.if_offset, self.device)
        self.conv12y = DCN_Conv(12*self.number, 4*self.number, 3, self.extend_scope, 1, self.if_offset, self.device)
        self.conv12z = DCN_Conv(12*self.number, 4*self.number, 3, self.extend_scope, 2, self.if_offset, self.device)
        self.conv13 = EncoderConv(16*self.number, 4*self.number)

        self.conv140 = DecoderConv(6*self.number, 2*self.number)
        self.conv14x = DCN_Conv(6*self.number, 2*self.number, 3, self.extend_scope, 0, self.if_offset, self.device)
        self.conv14y = DCN_Conv(6*self.number, 2*self.number, 3, self.extend_scope, 1, self.if_offset, self.device)
        self.conv14z = DCN_Conv(6*self.number, 2*self.number, 3, self.extend_scope, 2, self.if_offset, self.device)
        self.conv15 = DecoderConv(8*self.number, 2*self.number)

        self.conv160 = DecoderConv(3*self.number, self.number)
        self.conv16x = DCN_Conv(3*self.number, self.number, 3, self.extend_scope, 0, self.if_offset, self.device)
        self.conv16y = DCN_Conv(3*self.number, self.number, 3, self.extend_scope, 1, self.if_offset, self.device)
        self.conv16z = DCN_Conv(3*self.number, self.number, 3, self.extend_scope, 2, self.if_offset, self.device)
        self.conv17 = DecoderConv(4*self.number, self.number)

        self.out_conv = nn.Conv3d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # block0
        # x = self.maxpooling(x)
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0z(x)
        x_0y_0 = self.conv0y(x)
        x_0z_0 = self.conv0z(x)
        x_0_1 = self.conv1(cat([x_00_0, x_0x_0, x_0y_0, x_0z_0], dim=1))

        # block1
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_2z_0 = self.conv2z(x)
        x_1_1 = self.conv3(cat([x_20_0, x_2x_0, x_2y_0, x_2z_0], dim=1))

        # block2
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4x_0 = self.conv4x(x)
        x_4y_0 = self.conv4y(x)
        x_4z_0 = self.conv4z(x)
        x_2_1 = self.conv5(cat([x_40_0, x_4x_0, x_4y_0, x_4z_0], dim=1))

        # block3
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6x_0 = self.conv6x(x)
        x_6y_0 = self.conv6y(x)
        x_6z_0 = self.conv6z(x)
        x_3_1 = self.conv7(cat([x_60_0, x_6x_0, x_6y_0, x_6z_0], dim=1))

        # block4
        x = self.up(x_3_1)
        x_120_2 = self.conv120(cat([x, x_2_1], dim=1))
        x_12x_2 = self.conv12x(cat([x, x_2_1], dim=1))
        x_12y_2 = self.conv12y(cat([x, x_2_1], dim=1))
        x_12z_2 = self.conv12z(cat([x, x_2_1], dim=1))
        x_2_3 = self.conv13(cat([x_120_2, x_12x_2, x_12y_2, x_12z_2], dim=1))

        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(cat([x, x_1_1], dim=1))
        x_14x_2 = self.conv14x(cat([x, x_1_1], dim=1))
        x_14y_2 = self.conv14y(cat([x, x_1_1], dim=1))
        x_14z_2 = self.conv14z(cat([x, x_1_1], dim=1))
        x_1_3 = self.conv15(cat([x_140_2, x_14x_2, x_14y_2, x_14z_2], dim=1))

        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(cat([x, x_0_1], dim=1))
        x_16x_2 = self.conv16x(cat([x, x_0_1], dim=1))
        x_16y_2 = self.conv16y(cat([x, x_0_1], dim=1))
        x_16z_2 = self.conv16z(cat([x, x_0_1], dim=1))
        x_0_3 = self.conv17(cat([x_160_2, x_16x_2, x_16y_2, x_16z_2], dim=1))
        out = self.out_conv(x_0_3)
        out = self.softmax(out)
        # out = self.up(out)

        return out