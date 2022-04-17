import math
import torch.nn as nn

BN_MOMENTUM = 0.1

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Deconv(nn.Module):
    def __init__(self, downsample=1):
        super(Deconv, self).__init__()

        deconv_num = 4 - int(math.log(downsample, 2))
        deconv_kernel = [4] * deconv_num
        deconv_planes = [256] * deconv_num
        self.deconv_with_bias = False
        self.inplanes = 256
        self.deconv_layers = self._make_deconv_layer(deconv_num, deconv_kernel, deconv_planes)

        self.init_weights()

    def _make_deconv_layer(self, num_layers, kernels, planes):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = kernels[i], 1, 0
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes[i],
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes[i], momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes[i]

        return nn.Sequential(*layers)

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.deconv_layers(x)
        return x