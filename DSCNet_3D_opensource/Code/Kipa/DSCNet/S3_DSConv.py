# -*- coding: utf-8 -*-
import torch
from torch import nn, cat


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DCN_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, extend_scope, morph, if_offset, device):
        super(DCN_Conv, self).__init__()
        self.offset_conv = nn.Conv3d(in_ch, 3 * 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm3d(3 * 2 * kernel_size)
        self.kernel_size = kernel_size
        self.device = device

        self.if_offset = if_offset
        self.morph = morph
        self.extend_scope = extend_scope

        self.dcn_conv_x = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 1, kernel_size), stride=(1, 1, kernel_size), padding=0)  #
        self.dcn_conv_y = nn.Conv3d(in_ch, out_ch, kernel_size=(1, kernel_size, 1), stride=(1, kernel_size, 1), padding=0)  #
        self.dcn_conv_z = nn.Conv3d(in_ch, out_ch, kernel_size=(kernel_size, 1, 1), stride=(kernel_size, 1, 1), padding=0)  #

        self.dcn_conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=kernel_size, padding=0)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, f):
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset)
        input_shape = f.shape

        dcn = DCN(input_shape, self.kernel_size, self.extend_scope, self.morph, self.device)
        deformed_feature = dcn.deform_conv(f, offset, self.if_offset)
        if self.morph == 0:
            x = self.dcn_conv_x(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        elif self.morph == 1:
            x = self.dcn_conv_y(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x
        else:
            x = self.dcn_conv_z(deformed_feature)
            x = self.gn(x)
            x = self.relu(x)
            return x

class DCN(object):
    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points = kernel_size
        self.depth = input_shape[2]
        self.width = input_shape[3]
        self.height = input_shape[4]
        self.morph = morph
        self.device = device
        self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope
        self.num_batch = input_shape[0]   # (N,C,D,W,H)
        self.num_channels = input_shape[1]

    '''
    input: offset [N,3*K,D,W,H]
    output: [N,1,K*D,W,H]   coordinate map
    output: [N,1,K,K*W,H]   coordinate map
    output: [N,1,D,W,K*H]   coordinate map
    '''
    def _coordinate_map_3D(self, offset, if_offset):
        # offset
        offset1, offset2 = torch.split(offset, 3 * self.num_points, dim=1)
        z_offset1, y_offset1, x_offset1 = torch.split(offset1, self.num_points, dim=1)
        z_offset2, y_offset2, x_offset2 = torch.split(offset2, self.num_points, dim=1)

        z_center = torch.arange(0, self.depth).repeat([self.width*self.height])
        z_center = z_center.reshape(self.width, self.height, self.depth)
        z_center = z_center.permute(2, 1, 0)
        z_center = z_center.reshape([-1, self.depth, self.width, self.height])
        z_center = z_center.repeat([self.num_points, 1, 1, 1]).float()
        z_center = z_center.unsqueeze(0)

        y_center = torch.arange(0, self.width).repeat([self.height * self.depth])
        y_center = y_center.reshape(self.height, self.depth, self.width)
        y_center = y_center.permute(1, 2, 0)
        y_center = y_center.reshape([-1, self.depth, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1, 1]).float()
        y_center = y_center.unsqueeze(0)

        x_center = torch.arange(0, self.height).repeat([self.depth * self.width])
        x_center = x_center.reshape(self.depth, self.width, self.height)
        x_center = x_center.permute(0, 1, 2)
        x_center = x_center.reshape([-1, self.depth, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1, 1]).float()
        x_center = x_center.unsqueeze(0)

        if self.morph == 0:
            z = torch.linspace(0, 0, 1)
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(-int(self.num_points//2), int(self.num_points//2), int(self.num_points))
            z, y, x = torch.meshgrid(z, y, x)
            z_spread = z.reshape(-1, 1)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            z_grid = z_spread.repeat([1, self.depth * self.width * self.height])
            z_grid = z_grid.reshape([self.num_points, self.depth, self.width, self.height])
            z_grid = z_grid.unsqueeze(0)  # [N*K,D,W,H]

            y_grid = y_spread.repeat([1, self.depth * self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.depth, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            x_grid = x_spread.repeat([1, self.depth * self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.depth, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            z_new = z_center + z_grid
            y_new = y_center + y_grid
            x_new = x_center + x_grid                           # [N*K*K*K,D,W,H]

            z_new = z_new.repeat(self.num_batch, 1, 1, 1, 1)
            y_new = y_new.repeat(self.num_batch, 1, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1, 1)

            z_new = z_new.to(self.device)
            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)

            z_offset1_new = z_offset1.detach().clone()
            y_offset1_new = y_offset1.detach().clone()

            if if_offset:
                z_offset1_new = z_offset1_new.permute(1, 0, 2, 3, 4)
                y_offset1_new = y_offset1_new.permute(1, 0, 2, 3, 4)
                z_offset1 = z_offset1.permute(1, 0, 2, 3, 4)
                y_offset1 = y_offset1.permute(1, 0, 2, 3, 4)
                center = int(self.num_points // 2)
                z_offset1_new[center] = 0
                y_offset1_new[center] = 0
                for index in range(1, center + 1):
                    z_offset1_new[center + index] = z_offset1_new[center + index - 1] + z_offset1[center + index]
                    z_offset1_new[center - index] = z_offset1_new[center - index + 1] + z_offset1[center - index]
                    y_offset1_new[center + index] = y_offset1_new[center + index - 1] + y_offset1[center + index]
                    y_offset1_new[center - index] = y_offset1_new[center - index + 1] + y_offset1[center - index]
                z_offset1_new = z_offset1_new.permute(1, 0, 2, 3, 4).to(self.device)
                y_offset1_new = y_offset1_new.permute(1, 0, 2, 3, 4).to(self.device)
                z_new = z_new.add(z_offset1_new.mul(self.extend_scope))
                y_new = y_new.add(y_offset1_new.mul(self.extend_scope))

                z_new = z_new.reshape([self.num_batch, 1, 1, self.num_points, self.depth, self.width, self.height])
                z_new = z_new.permute(0, 4, 1, 5, 2, 6, 3)
                z_new = z_new.reshape([self.num_batch, self.depth, self.width, self.num_points*self.height])

                y_new = y_new.reshape([self.num_batch, 1, 1, self.num_points, self.depth, self.width, self.height])
                y_new = y_new.permute(0, 4, 1, 5, 2, 6, 3)
                y_new = y_new.reshape([self.num_batch, self.depth, self.width, self.num_points*self.height])

                x_new = x_new.reshape([self.num_batch, 1, 1, self.num_points, self.depth, self.width, self.height])
                x_new = x_new.permute(0, 4, 1, 5, 2, 6, 3)
                x_new = x_new.reshape([self.num_batch, self.depth, self.width, self.num_points*self.height])
            return z_new, y_new, x_new

        elif self.morph == 1:
            z = torch.linspace(0, 0, 1)
            y = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points))
            x = torch.linspace(0, 0, 1)
            z, y, x = torch.meshgrid(z, y, x)
            z_spread = z.reshape(-1, 1)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            z_grid = z_spread.repeat([1, self.depth * self.width * self.height])
            z_grid = z_grid.reshape([self.num_points, self.depth, self.width, self.height])
            z_grid = z_grid.unsqueeze(0)  # [N*K,D,W,H]

            y_grid = y_spread.repeat([1, self.depth * self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.depth, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            x_grid = x_spread.repeat([1, self.depth * self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.depth, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            z_new = z_center + z_grid
            y_new = y_center + y_grid
            x_new = x_center + x_grid  # [N*K*K*K,D,W,H]

            z_new = z_new.repeat(self.num_batch, 1, 1, 1, 1)
            y_new = y_new.repeat(self.num_batch, 1, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1, 1)

            z_new = z_new.to(self.device)
            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset1_new = x_offset1.detach().clone()
            z_offset2_new = z_offset2.detach().clone()

            if if_offset:
                x_offset1_new = x_offset1_new.permute(1, 0, 2, 3, 4)
                z_offset2_new = z_offset2_new.permute(1, 0, 2, 3, 4)
                x_offset1 = x_offset1.permute(1, 0, 2, 3, 4)
                z_offset2 = z_offset2.permute(1, 0, 2, 3, 4)
                center = int(self.num_points // 2)
                x_offset1_new[center] = 0
                z_offset2_new[center] = 0
                for index in range(1, center + 1):
                    x_offset1_new[center + index] = x_offset1_new[center + index - 1] + x_offset1[center + index]
                    x_offset1_new[center - index] = x_offset1_new[center - index + 1] + x_offset1[center - index]
                    z_offset2_new[center + index] = z_offset2_new[center + index - 1] + z_offset2[center + index]
                    z_offset2_new[center - index] = z_offset2_new[center - index + 1] + z_offset2[center - index]
                x_offset1_new = x_offset1_new.permute(1, 0, 2, 3, 4).to(self.device)
                z_offset2_new = z_offset2_new.permute(1, 0, 2, 3, 4).to(self.device)
                z_new = z_new.add(z_offset2_new.mul(self.extend_scope))
                x_new = x_new.add(x_offset1_new.mul(self.extend_scope))
            z_new = z_new.reshape([self.num_batch, 1, self.num_points, 1, self.depth, self.width, self.height])
            z_new = z_new.permute(0, 4, 1, 5, 2, 6, 3)
            z_new = z_new.reshape([self.num_batch, self.depth, self.num_points * self.width, self.height])
            y_new = y_new.reshape([self.num_batch, 1, self.num_points, 1, self.depth, self.width, self.height])
            y_new = y_new.permute(0, 4, 1, 5, 2, 6, 3)
            y_new = y_new.reshape([self.num_batch, self.depth, self.num_points * self.width, self.height])
            x_new = x_new.reshape([self.num_batch, 1, self.num_points, 1, self.depth, self.width, self.height])
            x_new = x_new.permute(0, 4, 1, 5, 2, 6, 3)
            x_new = x_new.reshape([self.num_batch, self.depth, self.num_points * self.width, self.height])
            return z_new, y_new, x_new

        else:
            z = torch.linspace(-int(self.num_points // 2), int(self.num_points // 2), int(self.num_points))
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(0, 0, 1)
            z, y, x = torch.meshgrid(z, y, x)
            z_spread = z.reshape(-1, 1)
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            z_grid = z_spread.repeat([1, self.depth * self.width * self.height])
            z_grid = z_grid.reshape([self.num_points, self.depth, self.width, self.height])
            z_grid = z_grid.unsqueeze(0)  # [N*K,D,W,H]

            y_grid = y_spread.repeat([1, self.depth * self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.depth, self.width, self.height])
            y_grid = y_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            x_grid = x_spread.repeat([1, self.depth * self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.depth, self.width, self.height])
            x_grid = x_grid.unsqueeze(0)  # [N*K*K*K,D,W,H]

            z_new = z_center + z_grid
            y_new = y_center + y_grid
            x_new = x_center + x_grid  # [N*K*K*K,D,W,H]

            z_new = z_new.repeat(self.num_batch, 1, 1, 1, 1)
            y_new = y_new.repeat(self.num_batch, 1, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1, 1)
            
            z_new = z_new.to(self.device)
            y_new = y_new.to(self.device)
            x_new = x_new.to(self.device)
            x_offset2_new = x_offset2.detach().clone()
            y_offset2_new = y_offset2.detach().clone()

            if if_offset:
                x_offset2_new = x_offset2_new.permute(1, 0, 2, 3, 4)
                y_offset2_new = y_offset2_new.permute(1, 0, 2, 3, 4)
                x_offset2 = x_offset2.permute(1, 0, 2, 3, 4)
                y_offset2 = y_offset2.permute(1, 0, 2, 3, 4)
                center = int(self.num_points // 2)
                x_offset2_new[center] = 0
                x_offset2_new[center] = 0
                for index in range(1, center + 1):
                    x_offset2_new[center + index] = x_offset2_new[center + index - 1] + x_offset2[center + index]
                    x_offset2_new[center - index] = x_offset2_new[center - index + 1] + x_offset2[center - index]
                    y_offset2_new[center + index] = y_offset2_new[center + index - 1] + y_offset2[center + index]
                    y_offset2_new[center - index] = y_offset2_new[center - index + 1] + y_offset2[center - index]
                x_offset2_new = x_offset2_new.permute(1, 0, 2, 3, 4).to(self.device)
                y_offset2_new = y_offset2_new.permute(1, 0, 2, 3, 4).to(self.device)
                x_new = x_new.add(x_offset2_new.mul(self.extend_scope))
                y_new = y_new.add(y_offset2_new.mul(self.extend_scope))

            z_new = z_new.reshape([self.num_batch, self.num_points, 1, 1, self.depth, self.width, self.height])
            z_new = z_new.permute(0, 4, 1, 5, 2, 6, 3)
            z_new = z_new.reshape([self.num_batch, self.num_points*self.depth, self.width, self.height])

            y_new = y_new.reshape([self.num_batch, self.num_points, 1, 1, self.depth, self.width, self.height])
            y_new = y_new.permute(0, 4, 1, 5, 2, 6, 3)
            y_new = y_new.reshape([self.num_batch, self.num_points*self.depth, self.width, self.height])

            x_new = x_new.reshape([self.num_batch, self.num_points, 1, 1, self.depth, self.width, self.height])
            x_new = x_new.permute(0, 4, 1, 5, 2, 6, 3)
            x_new = x_new.reshape([self.num_batch, self.num_points*self.depth, self.width, self.height])
            return z_new, y_new, x_new

    '''
    input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H] 
    output: [N,1,K*D,K*W,K*H]  deformed feature map
    '''
    def _bilinear_interpolate_3D(self, input_feature, z, y, x):
        z = z.reshape([-1]).float()
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()  # [N*KD*KW*KH]

        zero = torch.zeros([]).int()
        max_z = self.depth - 1
        max_y = self.width - 1
        max_x = self.height - 1

        # find 8 grid locations
        z0 = torch.floor(z).int()
        z1 = z0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume以外的点
        z0 = torch.clamp(z0, zero, max_z)
        z1 = torch.clamp(z1, zero, max_z)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)  # [N*KD*KW*KH]

        # convert input_feature and coordinate X, Y to 3D，for gathering
        # input_feature_flat = input_feature.reshape([-1, self.num_channels])   # [N*D*W*H, C]
        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(self.num_batch, self.num_channels, self.depth, self.width,
                                                        self.height)
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 4, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width * self.depth

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()  # [N,1]

        repeat = torch.ones([self.num_points * self.depth * self.width * self.height]).unsqueeze(0)
        repeat = repeat.float()  # [1,D*W*H*K*K*K]

        base = torch.matmul(base, repeat)  # [N,1] * [1,D*W*H*K*K*K]  ==> [N,D*W*H*K*K*K]
        base = base.reshape([-1])  # [D*W*H*K*K*K]

        base = base.to(self.device)

        base_z0 = base + z0 * self.height * self.width
        base_z1 = base + z1 * self.height * self.width
        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        # top rectangle of the neighbourhood volume
        index_a0 = base_y0 + base_z0 - base + x0
        index_b0 = base_y0 + base_z1 - base + x0
        index_c0 = base_y0 + base_z0 - base + x1
        index_d0 = base_y0 + base_z1 - base + x1  # [N*KD*KW*KH]

        # bottom rectangle of the neighbourhood volume
        index_a1 = base_y1 + base_z0 - base + x0
        index_b1 = base_y1 + base_z1 - base + x0
        index_c1 = base_y1 + base_z0 - base + x1
        index_d1 = base_y1 + base_z1 - base + x1  # [N*KD*KW*KH]

        # get 8 grid values  ([N*D*W*H,C], [N*D*W*H*27])
        value_a0 = input_feature_flat[index_a0.type(torch.int64)]
        value_b0 = input_feature_flat[index_b0.type(torch.int64)]
        value_c0 = input_feature_flat[index_c0.type(torch.int64)]
        value_d0 = input_feature_flat[index_d0.type(torch.int64)]
        value_a1 = input_feature_flat[index_a1.type(torch.int64)]
        value_b1 = input_feature_flat[index_b1.type(torch.int64)]
        value_c1 = input_feature_flat[index_c1.type(torch.int64)]
        value_d1 = input_feature_flat[index_d1.type(torch.int64)]  # [N*KD*KW*KH, C]

        # find 8 grid locations
        z0 = torch.floor(z).int()
        z1 = z0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        # clip out coordinates exceeding feature map volume以外的点
        z0 = torch.clamp(z0, zero, max_z + 1)
        z1 = torch.clamp(z1, zero, max_z + 1)
        y0 = torch.clamp(y0, zero, max_y + 1)
        y1 = torch.clamp(y1, zero, max_y + 1)
        x0 = torch.clamp(x0, zero, max_x + 1)
        x1 = torch.clamp(x1, zero, max_x + 1)  # [N*KD*KW*KH]

        x0_float = x0.float()
        x1_float = x1.float()
        y0_float = y0.float()
        y1_float = y1.float()
        z0_float = z0.float()
        z1_float = z1.float()

        vol_a0 = ((z1_float - z) * (y1_float - y) * (x1_float - x)).unsqueeze(-1)
        vol_b0 = ((z - z0_float) * (y1_float - y) * (x1_float - x)).unsqueeze(-1)
        vol_c0 = ((z1_float - z) * (y1_float - y) * (x - x0_float)).unsqueeze(-1)
        vol_d0 = ((z - z0_float) * (y1_float - y) * (x - x0_float)).unsqueeze(-1)
        vol_a1 = ((z1_float - z) * (y - y0_float) * (x1_float - x)).unsqueeze(-1)
        vol_b1 = ((z - z0_float) * (y - y0_float) * (x1_float - x)).unsqueeze(-1)
        vol_c1 = ((z1_float - z) * (y - y0_float) * (x - x0_float)).unsqueeze(-1)
        vol_d1 = ((z - z0_float) * (y - y0_float) * (x - x0_float)).unsqueeze(-1)  # [N*KD*KW*KH, C]

        outputs = value_a0 * vol_a0 + value_b0 * vol_b0 + value_c0 * vol_c0 + value_d0 * vol_d0 + value_a1 * vol_a1 + value_b1 * vol_b1 + value_c1 * vol_c1 + value_d1 * vol_d1

        if self.morph == 0:
            outputs = outputs.reshape([self.num_batch, self.depth, self.width, self.num_points*self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        elif self.morph == 1:
            outputs = outputs.reshape([self.num_batch, self.depth, self.num_points*self.width, self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        else:
            outputs = outputs.reshape([self.num_batch, self.num_points*self.depth, self.width, self.height, self.num_channels])
            outputs = outputs.permute(0, 4, 1, 2, 3)
        return outputs

    def deform_conv(self, input, offset, if_offset):
        z, y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input, z, y, x)
        return deformed_feature
