import torch
import torch.nn as nn
from collections.abc import Iterable
import config
from utils.tools_self import standar_gaussian, conver_tensor_to_numpy

cofficient = standar_gaussian(config.get_value())

class Conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(Conv_block, self).__init__()
        self.padding = [(size[0] - 1) // 2, (size[1] - 1) // 2]
        self.conv = nn.Conv2d(in_ch, out_ch, size, padding=self.padding, stride=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.norm(x1)
        x3 = self.act(x2)
        return x3

        # return self.act(self.norm(self.conv(x)))

class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double_conv, self).__init__()
        self.branchs = 6
        self.in_ch = in_ch
        self.mid_mid = out_ch // self.branchs
        self.out_ch = out_ch
        self.conv1x1_mid = Conv_block(self.in_ch, self.out_ch, [1, 1])
        self.conv1x1_2 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.conv3x3_2_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x1_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv1x3_1 = Conv_block(self.mid_mid, self.mid_mid, [1, 3])
        self.conv3x3_1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv3x1_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])
        self.conv1x3_2 = Conv_block(self.mid_mid, self.mid_mid, [3, 1])

        self.conv1x1_1 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.num = max(self.mid_mid // 2, 12)
        self.fc = nn.Linear(in_features=self.mid_mid, out_features=self.num)
        self.fcs = nn.ModuleList([])
        for i in range(self.branchs):
            self.fcs.append(nn.Linear(in_features=self.num, out_features=self.mid_mid))
        self.softmax = nn.Softmax(dim=1)

        self.rel = nn.ReLU(inplace=True)
        if self.in_ch > self.out_ch:
            self.short_connect = nn.Conv2d(in_ch, out_ch, 1, padding=0)

    def forward(self, x):
        short = x
        if self.in_ch > self.out_ch:
            short = self.short_connect(x)
        xxx = self.conv1x1_mid(x)
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x1 = self.conv1x3_1(x1 + x0)
        x2 = self.conv3x1_1(x2 + x1)
        x3 = self.conv3x3_1(x3 + x2)
        x4 = self.conv3x3_1_1(x4 + x3)
        x5 = self.conv3x3_2_1(x5 + x4)
        xx = x0 + x1 + x2 + x3 + x4 + x5
        sk1 = self.avg_pool(xx)
        sk1 = sk1.view(sk1.size(0), -1)

        sk2 = self.fc(sk1)
        for i, fc in enumerate(self.fcs):
            vector = fc(sk2).unsqueeze(1)
            if i == 0:
                attention_vector = vector
            else:
                attention_vector = torch.cat([attention_vector, vector], dim=1)
        attention_vector = self.softmax(attention_vector)
        attention_vector = attention_vector.unsqueeze(-1).unsqueeze(-1)

        x0 = x0 * attention_vector[:, 0, ...]
        x1 = x1 * attention_vector[:, 1, ...]
        x2 = x2 * attention_vector[:, 2, ...]
        x3 = x3 * attention_vector[:, 3, ...]
        x4 = x4 * attention_vector[:, 4, ...]
        x5 = x5 * attention_vector[:, 5, ...]
        xx = torch.cat((x0, x1, x2, x3, x4, x5), dim=1)
        xxx = self.conv1x1_1(xx)
        x0 = xxx[:, 0:self.mid_mid, ...]
        x1_2 = xxx[:, self.mid_mid:self.mid_mid * 2, ...]
        x2_2 = xxx[:, self.mid_mid * 2:self.mid_mid * 3, ...]
        x3_2 = xxx[:, self.mid_mid * 3:self.mid_mid * 4, ...]
        x4_2 = xxx[:, self.mid_mid * 4:self.mid_mid * 5, ...]
        x5_2 = xxx[:, self.mid_mid * 5:self.mid_mid * 6, ...]
        x1 = self.conv1x3_2(x1_2 + x0)
        x2 = self.conv3x1_2(x1 + x2_2)
        x3 = self.conv3x3_2(x2 + x3_2)
        x4 = self.conv3x3_1_2(x3 + x4_2)
        x5 = self.conv3x3_2_2(x4 + x5_2)
        xx = torch.cat((x0, x1, x2, x3, x4, x5), dim=1)
        xxx = self.conv1x1_2(xx)
        return self.rel(xxx + short)


class Conv_down_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_down_2, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=2, bias=True)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Conv_down(nn.Module):
    def __init__(self, in_ch, out_ch, flage):
        super(Conv_down, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.flage = flage
        if self.in_ch == 1:
            self.first = nn.Sequential(
                Conv_block(self.in_ch, self.out_ch, [3, 3]),
                Double_conv(self.out_ch, self.out_ch),
            )
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_down = Conv_down_2(self.out_ch, self.out_ch)
        else:
            self.conv = Double_conv(self.in_ch, self.out_ch)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv_down = Conv_down_2(self.in_ch, self.out_ch)

    def forward(self, x):
        if self.in_ch == 1:
            x = self.first(x)
            pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
        else:
            x = self.conv(x)
            if self.flage == True:
                pool_x = torch.cat((self.pool(x), self.conv_down(x)), dim=1)
            else:
                pool_x = None
        return pool_x, x

class side_output(nn.Module):
    def __init__(self, inChans, outChans, factor, padding):
        super(side_output, self).__init__()
        self.conv0 = nn.Conv2d(inChans, outChans, 3, 1, 1)
        self.transconv1 = nn.ConvTranspose2d(outChans, outChans, 2 * factor, factor, padding=padding)

    def forward(self, x):
        out = self.conv0(x)
        out = self.transconv1(out)
        return out


def hdc(image, device, num=2):
    x1 = torch.Tensor([]).to(device)
    for i in range(num):
        for j in range(num):
            x3 = image[:, :, i::num, j::num]
            x1 = torch.cat((x1, x3), dim=1)
    return x1


class Conv_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_up, self).__init__()
        self.up = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        self.conv = Double_conv(in_ch, out_ch)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x1 = self.interp(x1)
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


class WDA_Net_r(nn.Module):
    def __init__(self, in_channels, out_channels, device, has_dropout=False):
        super(WDA_Net_r, self).__init__()
        self.has_dropout = has_dropout
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.device = device
        self.first = Conv_block(4, 72, [3, 3])
        self.Conv_down1 = Conv_down(72, 72, True)
        self.Conv_down2 = Conv_down(144, 144, True)
        self.Conv_down3 = Conv_down(288, 288, True)
        self.Conv_down5 = Conv_down(576, 576, False)

        self.Conv_up2 = Conv_up(576, 288)
        self.Conv_up3 = Conv_up(288, 144)
        self.Conv_up4 = Conv_up(144, 72)

        self.Conv_up5 = Double_conv(72, 72)
        self.Conv_up6 = Double_conv(72, 72)
        self.Conv_up7 = Double_conv(72, 72)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Conv_out = nn.Conv2d(72, out_channels, 1, padding=0, stride=1)

        self.up_c = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.Conv_out_c = nn.Conv2d(72, 1, 1, padding=0, stride=1)
        self.act = nn.ReLU(inplace=True)

        self.two = Conv_block(1, 72, [3, 3])
        self.Conv_down9 = Double_conv(72, 72)
        self.Conv_out_count = nn.Conv2d(72, 1, 1, padding=0, stride=1)

    def forward(self, x):
        x = hdc(x, self.device)
        x = self.first(x)
        x, conv1 = self.Conv_down1(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        _, x = self.Conv_down5(x)
        x = self.Conv_up2(x, conv3)
        x = self.Conv_up3(x, conv2)
        x = self.Conv_up4(x, conv1)

        seg_x = self.Conv_up5(x)
        seg_x = self.up(seg_x)
        seg_x = self.Conv_out(seg_x)

        det_x = self.Conv_up6(x)
        det_x = self.Conv_up7(det_x)
        det_x = self.up_c(det_x)
        det_x = self.Conv_out_c(det_x)
        det_x = self.act(det_x)

        count_x = self.two(det_x)
        count_x = self.Conv_down9(count_x)
        count_x = self.Conv_out_count(count_x)
        count_x = self.act(count_x)

        return seg_x, det_x, count_x


class Counting_Model(nn.Module):

    def __init__(self, in_channels, out_channels, device, has_dropout=False):
        super(Counting_Model, self).__init__()

        self.has_dropout = has_dropout
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.device = device
        self.first = Conv_block(4, 72, [3, 3])
        self.Conv_down1 = Conv_down(72, 72, True)
        self.Conv_down2 = Conv_down(144, 144, True)
        self.Conv_down3 = Conv_down(288, 288, True)
        self.Conv_down5 = Conv_down(576, 576, False)

        self.Conv_up2 = Conv_up(576, 288)
        self.Conv_up3 = Conv_up(288, 144)
        self.Conv_up4 = Conv_up(144, 72)
        self.Conv_up5 = Double_conv(72, 72)

        self.Conv_out_c = nn.Conv2d(72, out_channels, 1, padding=0, stride=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = hdc(x, self.device)
        x = self.first(x)
        x, conv1 = self.Conv_down1(x)
        x, conv2 = self.Conv_down2(x)
        x, conv3 = self.Conv_down3(x)
        _, x = self.Conv_down5(x)

        x = self.Conv_up2(x, conv3)
        x = self.Conv_up3(x, conv2)
        x = self.Conv_up4(x, conv1)

        count_x = self.up(x)
        count_x = self.Conv_out_c(count_x)
        count_x = self.act(count_x)
        return count_x


if __name__ == "__main__":
    pass
