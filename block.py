import torch
from torch import nn
from torch.nn import init


class ResAT(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResAT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.at =ADPAM()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.at(out)


        out += residual
        out = self.relu(out)
        return out


class ADPAM(nn.Module):
    def __init__(self, kernel_size=3):
        super(ADPAM, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        avg = self.gap(x)
        max = self.gmp(x)
        y = torch.cat([avg, max], dim=1)
        y = y.squeeze(-1).squeeze(-1)
        y = y.view(x.size(0), 2, x.size(1))
        y = self.conv(y)
        y = self.sigmoid_channel(y)
        y = y.view(x.size(0), x.size(1), 1, 1)
        x_channel = x * y.expand_as(x)

        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        x_spatial = self.conv_spatial(x_spatial)
        x_spatial = self.sigmoid_spatial(x_spatial)
        out = x_channel * x_spatial

        return out



import torch
from torch import nn
from torch.nn import init

class APCM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(APCM, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adaptavgpool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_pool = self.avgpool(x)
        x_minus_mu_square = (x_pool - x_pool.mean(dim=[2, 3], keepdim=True)).pow(2)
        y_pool = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / (n/4) + self.e_lambda)) + 0.5
        y = self.act(y_pool)
        y_upsampled = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)


        x_global_pool = self.adaptavgpool(x)
        x_global_pool_repeated = x_global_pool.repeat(1, 1, h, w)
        x_minus_mu_global_square = (x - x_global_pool_repeated).pow(2)
        y_global = x_minus_mu_global_square / (4 * (x_minus_mu_global_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        y_global = self.act(y_global)

        y_final = x * (y_upsampled+y_global)/2
        return y_final

class ResATZJ(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResATZJ, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.at =APCM()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.at(out)


        out += residual
        out = self.relu(out)
        return out






