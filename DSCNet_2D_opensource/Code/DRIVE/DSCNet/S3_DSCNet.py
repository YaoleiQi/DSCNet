# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
from S3_DSConv import DSConv


# Define a standard convolution kernel
class EncoderConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
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
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x


class DSCNet(nn.Module):

    def __init__(
        self,
        n_channels,
        n_classes,
        kernel_size,
        extend_scope,
        if_offset,
        device,
        number,
        dim,
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super(DSCNet, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0x = DSConv(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv0y = DSConv(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv1 = EncoderConv(3 * self.number, self.number)

        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = DSConv(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv2y = DSConv(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv3 = EncoderConv(6 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = DSConv(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv4y = DSConv(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv5 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6x = DSConv(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv6y = DSConv(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv7 = EncoderConv(24 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12x = DSConv(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv12y = DSConv(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv13 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = DSConv(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv14y = DSConv(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv15 = DecoderConv(6 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = DSConv(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv16y = DSConv(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv17 = DecoderConv(3 * self.number, self.number)

        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # block0
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_0_1 = self.conv1(cat([x_00_0, x_0x_0, x_0y_0], dim=1))

        # block1
        x = self.maxpooling(x_0_1)
        x_20_0 = self.conv20(x)
        x_2x_0 = self.conv2x(x)
        x_2y_0 = self.conv2y(x)
        x_1_1 = self.conv3(cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        # block2
        x = self.maxpooling(x_1_1)
        x_40_0 = self.conv40(x)
        x_4x_0 = self.conv4x(x)
        x_4y_0 = self.conv4y(x)
        x_2_1 = self.conv5(cat([x_40_0, x_4x_0, x_4y_0], dim=1))

        # block3
        x = self.maxpooling(x_2_1)
        x_60_0 = self.conv60(x)
        x_6x_0 = self.conv6x(x)
        x_6y_0 = self.conv6y(x)
        x_3_1 = self.conv7(cat([x_60_0, x_6x_0, x_6y_0], dim=1))

        # block4
        x = self.up(x_3_1)
        x_120_2 = self.conv120(cat([x, x_2_1], dim=1))
        x_12x_2 = self.conv12x(cat([x, x_2_1], dim=1))
        x_12y_2 = self.conv12y(cat([x, x_2_1], dim=1))
        x_2_3 = self.conv13(cat([x_120_2, x_12x_2, x_12y_2], dim=1))

        # block5
        x = self.up(x_2_3)
        x_140_2 = self.conv140(cat([x, x_1_1], dim=1))
        x_14x_2 = self.conv14x(cat([x, x_1_1], dim=1))
        x_14y_2 = self.conv14y(cat([x, x_1_1], dim=1))
        x_1_3 = self.conv15(cat([x_140_2, x_14x_2, x_14y_2], dim=1))

        # block6
        x = self.up(x_1_3)
        x_160_2 = self.conv160(cat([x, x_0_1], dim=1))
        x_16x_2 = self.conv16x(cat([x, x_0_1], dim=1))
        x_16y_2 = self.conv16y(cat([x, x_0_1], dim=1))
        x_0_3 = self.conv17(cat([x_160_2, x_16x_2, x_16y_2], dim=1))
        # x = self.dropout(x)
        out = self.out_conv(x_0_3)
        out = self.sigmoid(out)

        return out
