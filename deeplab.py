import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from modules.SampleConv import SampleConv
import numpy as np
from torch.autograd import grad, Variable
from util import *
from tensorboardX import SummaryWriter
from Printer import Printer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # deeplab change
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)  # deeplab change
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)  # deeplab change

        samples = 9
        self.conv_to_offset0 = nn.Conv2d(512 * block.expansion, samples * 2, kernel_size=3,
                                         stride=1, dilation=6, padding=6)  # huiyu change
        self.conv_to_offset1 = nn.Conv2d(512 * block.expansion, samples * 2, kernel_size=3,
                                         stride=1, dilation=12, padding=12)  # huiyu change
        self.conv_to_offset2 = nn.Conv2d(512 * block.expansion, samples * 2, kernel_size=3,
                                         stride=1, dilation=18, padding=18)  # huiyu change
        self.conv_to_offset3 = nn.Conv2d(512 * block.expansion, samples * 2, kernel_size=3,
                                         stride=1, dilation=24, padding=24)  # huiyu change

        self.sample_conv = SampleConv(512 * block.expansion, num_classes, samples, 0, 0, groups=1)  # huiyu change

        # base_offset = torch.from_numpy(
        #     np.array([-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1]).astype(np.float32))  # h, w
        base_offset = torch.from_numpy(
            np.array([-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1]).astype(np.float32))  # w, h

        self.gradient_to_delta = nn.Conv2d(samples * 2 * 21, samples * 2, kernel_size=1)
        self.offset_to_delta = nn.Conv2d(samples * 2, samples * 2, kernel_size=1)

        self.aux_loss = nn.CrossEntropyLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # special init for offset layers
        self.conv_to_offset0.weight.data.zero_()
        self.conv_to_offset1.weight.data.zero_()
        self.conv_to_offset2.weight.data.zero_()
        self.conv_to_offset3.weight.data.zero_()

        if samples == 36:
            four_bases = torch.cat([base_offset * 6, base_offset * 12, base_offset * 18, base_offset * 24], 0)
        if samples == 9:
            four_bases = torch.cat([base_offset * 6], 0)
        self.conv_to_offset0.bias.data = four_bases / 4
        self.conv_to_offset1.bias.data = four_bases / 4
        self.conv_to_offset2.bias.data = four_bases / 4
        self.conv_to_offset3.bias.data = four_bases / 4

        self.gradient_to_delta.weight.data.zero_()
        self.gradient_to_delta.bias.data.zero_()

        self.offset_to_delta.weight.data.zero_()
        self.offset_to_delta.bias.data.zero_()

        # set to zero as chenxi did in init model
        self.sample_conv.conv.weight.data.zero_()
        self.sample_conv.conv.bias.data.zero_()

        self.writer = SummaryWriter()
        self.global_step = 0
        self.hooks = list()
        # self.gradient_to_delta.weight.register_hook(
        #     Printer('Backward: gradient_to_delta.weight', self.writer, self.global_step).print_var_par)
        # self.gradient_to_delta.bias.register_hook(
        #     Printer('Backward: gradient_to_delta.bias', self.writer, self.global_step).print_var_par)
        # self.offset_to_delta.weight.register_hook(
        #     Printer('Backward: offset_to_delta.weight', self.writer, self.global_step).print_var_par)
        # self.offset_to_delta.bias.register_hook(
        #     Printer('Backward: offset_to_delta.bias', self.writer, self.global_step).print_var_par)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        self.global_step += 1
        for hook in self.hooks:
            hook.remove()
        self.hooks = list()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

        # huiyu
        offset0 = self.conv_to_offset0(feature)
        offset1 = self.conv_to_offset1(feature)
        offset2 = self.conv_to_offset2(feature)
        offset3 = self.conv_to_offset3(feature)

        offset_step0 = offset0 + offset1 + offset2 + offset3

        output_step0 = self.sample_conv(feature, offset_step0)

        # auxiliary loss
        label0 = torch.t(output_step0.view(21, -1))
        gradients = []
        for i in range(21):
            aux_loss = self.aux_loss(label0, Variable(torch.LongTensor(label0.size()[0]).cuda().zero_() + i))
            # aux_loss = torch.sum(label0)
            offset_grad, = grad(aux_loss, offset_step0, retain_graph=True, create_graph=True)
            gradients.append(offset_grad)

        gradient = torch.cat(gradients, dim=1).detach()
        delta_from_offset = self.offset_to_delta(offset_step0)
        delta_from_gradient = self.gradient_to_delta(gradient * 0)
        offset_step1 = offset_step0 + delta_from_offset + delta_from_gradient

        output_step1 = self.sample_conv(feature, offset_step1)

        # printing
        Printer('Forward:  feature', self.writer, self.global_step).print_var_par(feature)
        feature.register_hook(Printer('Backward: feature', self.writer, self.global_step).print_var_par)

        Printer('Forward:  offset_step0', self.writer, self.global_step).print_var_par(offset_step0)

        Printer('Forward:  gradient', self.writer, self.global_step).print_var_par(gradient)

        Printer('Forward:  offset_to_delta.weight', self.writer, self.global_step).print_var_par(self.offset_to_delta.weight)
        Printer('Forward:  offset_to_delta.bias', self.writer, self.global_step).print_var_par(self.offset_to_delta.bias)
        Printer('Forward:  gradient_to_delta.weight', self.writer, self.global_step).print_var_par(self.gradient_to_delta.weight)
        Printer('Forward:  gradient_to_delta.bias', self.writer, self.global_step).print_var_par(self.gradient_to_delta.bias)
        self.hooks.append(self.gradient_to_delta.weight.register_hook(
            Printer('Backward: gradient_to_delta.weight', self.writer, self.global_step).print_var_par))
        self.hooks.append(self.gradient_to_delta.bias.register_hook(
            Printer('Backward: gradient_to_delta.bias', self.writer, self.global_step).print_var_par))
        self.hooks.append(self.offset_to_delta.weight.register_hook(
            Printer('Backward: offset_to_delta.weight', self.writer, self.global_step).print_var_par))
        self.hooks.append(self.offset_to_delta.bias.register_hook(
            Printer('Backward: offset_to_delta.bias', self.writer, self.global_step).print_var_par))

        Printer('Forward:  delta_from_offset', self.writer, self.global_step).print_var_par(delta_from_offset)

        Printer('Forward:  delta_from_gradient', self.writer, self.global_step).print_var_par(delta_from_gradient)

        Printer('Forward:  offset_step1', self.writer, self.global_step).print_var_par(offset_step1)
        offset_step1.register_hook(Printer('Backward: offset_step1', self.writer, self.global_step).print_var_par)

        Printer('Forward:  output_step1', self.writer, self.global_step).print_var_par(output_step1)
        output_step1.register_hook(Printer('Backward: output_step1', self.writer, self.global_step).print_var_par)

        return output_step1


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model
