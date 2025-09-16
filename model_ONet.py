import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



from model.block import ResATZJ

class ONet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter, deep_supervision=False):
        super(ONet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layerZJ(input_channels, nb_filter[0])
        self.conv1_0 = self._make_layerZJ(nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layerZJ(nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layerZJ(nb_filter[2],  nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layerZJ(nb_filter[3],  nb_filter[4], num_blocks[3])

        self.conv0_1 = self._make_layer(block, nb_filter[0] + nb_filter[1] * 2,  nb_filter[0])
        self.conv1_1 = self._make_layer(block, nb_filter[1] + nb_filter[2] + nb_filter[0],  nb_filter[1], num_blocks[0])
        self.conv2_1 = self._make_layer(block, nb_filter[2] + nb_filter[3] + nb_filter[1],  nb_filter[2], num_blocks[1])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4] + nb_filter[2],  nb_filter[3], num_blocks[2])

        self.conv0_2 = self._make_layer(block, nb_filter[0] * 2 + nb_filter[1] * 2, nb_filter[0])
        self.conv1_2 = self._make_layer(block, nb_filter[1]*2 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_2 = self._make_layer(block, nb_filter[2]*2 + nb_filter[3]+ nb_filter[1], nb_filter[2], num_blocks[1])

        self.conv0_3 = self._make_layer(block, nb_filter[0] * 3 + nb_filter[1] * 2, nb_filter[0])
        self.conv1_3 = self._make_layer(block, nb_filter[1]*3 + nb_filter[2]+ nb_filter[0], nb_filter[1], num_blocks[0])

        self.conv0_4 = self._make_layer(block, nb_filter[0] * 4 + nb_filter[1] * 2, nb_filter[0])

        self.Oconv1_0 = Conv(nb_filter[0], nb_filter[1])
        self.Oconv2_0 = Conv(nb_filter[1], nb_filter[2])
        self.Oconv3_0 = Conv(nb_filter[2], nb_filter[3])
        self.Oconv4_0 = Conv(nb_filter[3], nb_filter[4])

        self.Oconv1_1 = Conv(nb_filter[1] + nb_filter[2] + nb_filter[0], nb_filter[1])
        self.Oconv2_1 = Conv(nb_filter[2] + nb_filter[3] + nb_filter[1], nb_filter[2])
        self.Oconv3_1 = Conv(nb_filter[3] + nb_filter[4] + nb_filter[2], nb_filter[3])

        self.Oconv1_2 = Conv(nb_filter[1] * 2 + nb_filter[2] + nb_filter[0], nb_filter[1])
        self.Oconv2_2 = Conv(nb_filter[2] * 2 + nb_filter[3] + nb_filter[1], nb_filter[2])

        self.Oconv1_3 = Conv(nb_filter[1] * 3 + nb_filter[2] + nb_filter[0], nb_filter[1])



        self.conv0_4_final = self._make_layer(block, nb_filter[0] * 5, nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def _make_layerZJ(self, input_channels, output_channels, num_blocks=1):
        layers = []
        blockZJ=ResATZJ
        layers.append(blockZJ(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(blockZJ(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)  #
        x1_0 = self.conv1_0(self.pool(x0_0))
        O1_0 = self.Oconv1_0(self.pool(x0_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0), self.up(O1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0), self.down(x0_1)], 1))
        O2_0 = self.Oconv2_0(self.pool(O1_0))
        O1_1 = self.Oconv1_1(torch.cat([O1_0, self.up(O2_0), self.down(x0_1)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1), self.up(O1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0), self.down(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1), self.down(x0_2)], 1))
        O3_0 = self.Oconv3_0(self.pool(O2_0))
        O2_1 = self.Oconv2_1(torch.cat([O2_0, self.up(O3_0), self.down(O1_1)], 1))
        O1_2 = self.Oconv1_2(torch.cat([O1_0, O1_1, self.up(O2_1), self.down(x0_2)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2), self.up(O1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))  # output
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0), self.down(x2_1)], 1))  # output
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1), self.down(x1_2)], 1))  # output
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2), self.down(x0_3)], 1))  # output
        O4_0 = self.Oconv4_0(self.pool(O3_0))  # output
        O3_1 = self.Oconv3_1(torch.cat([O3_0, self.up(O4_0), self.down(O2_1)], 1))  # output
        O2_2 = self.Oconv2_2(torch.cat([O2_0, O2_1, self.up(O3_1), self.down(O1_2)], 1))  # output
        O1_3 = self.Oconv1_3(torch.cat([O1_0, O1_1, O1_2, self.up(O2_2), self.down(x0_3)], 1))  # output

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3), self.up(O1_3)], 1))  # output

        # Onet-2
        # O0_0 = self.Oconv0_0(input)  #

        # O0_1 = self.Oconv0_1(torch.cat([O0_0, self.up(O1_0)], 1))

        # O0_2 = self.Oconv0_2(torch.cat([O0_0, O0_1, self.up(O1_1)], 1))

        # O0_3 = self.Oconv0_3(torch.cat([O0_0, O0_1, O0_2, self.up(O1_2)], 1))

        # O0_4 = self.Oconv0_4(torch.cat([O0_0, O0_1, O0_2, O0_3, self.up(O1_3)], 1))   #output
        # Onet-2
        xO4_0 = x4_0 + O4_0
        xO3_1 = x3_1 + O3_1
        xO2_2 = x2_2 + O2_2
        xO1_3 = x1_3 + O1_3

        xO0_4 = x0_4
        xO0_1 = x0_1
        xO0_2 = x0_2
        xO0_3 = x0_3

        Final_xO0_4 = self.conv0_4_final(
            torch.cat([self.up_16(self.conv0_4_1x1(xO4_0)), self.up_8(self.conv0_3_1x1(xO3_1)),
                       self.up_4(self.conv0_2_1x1(xO2_2)), self.up(self.conv0_1_1x1(xO1_3)), xO0_4], 1))  #

        if self.deep_supervision:
            output1 = self.final1(xO0_1)
            output2 = self.final2(xO0_2)
            output3 = self.final3(xO0_3)
            output4 = self.final4(Final_xO0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(Final_xO0_4)
            return output



