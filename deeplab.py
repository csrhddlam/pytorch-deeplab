import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from modules.DeformConv import ConvOffset2d
import numpy as np
from torch.autograd import grad, Variable

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

        # self.fc1_voc12_c0 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
        #                               stride=1, dilation=6, padding=6)  # deeplab change
        # self.fc1_voc12_c1 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
        #                               stride=1, dilation=12, padding=12)  # deeplab change
        # self.fc1_voc12_c2 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
        #                               stride=1, dilation=18, padding=18)  # deeplab change
        # self.fc1_voc12_c3 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
        #                               stride=1, dilation=24, padding=24)  # deeplab change
        samples = 9
        self.offset_c0 = nn.Conv2d(512 * block.expansion, samples * 2, kernel_size=3,
                                   stride=1, dilation=6, padding=6)  # huiyu change
        self.offset_c1 = nn.Conv2d(512 * block.expansion, samples * 2, kernel_size=3,
                                   stride=1, dilation=12, padding=12)  # huiyu change
        self.offset_c2 = nn.Conv2d(512 * block.expansion, samples * 2, kernel_size=3,
                                   stride=1, dilation=18, padding=18)  # huiyu change
        self.offset_c3 = nn.Conv2d(512 * block.expansion, samples * 2, kernel_size=3,
                                   stride=1, dilation=24, padding=24)  # huiyu change

        self.fc1_voc12 = ConvOffset2d(512 * block.expansion, num_classes, samples, 0, 0, groups=1)  # huiyu change

        # base_offset = torch.from_numpy(
        #     np.array([-1, -1, -1, 0, -1, 1, 0, -1, 0, 0, 0, 1, 1, -1, 1, 0, 1, 1]).astype(np.float32))  # h, w
        base_offset = torch.from_numpy(
            np.array([-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1]).astype(np.float32))  # w, h
        self.optimizer = nn.Conv2d(samples * 2 * 21, samples * 2, kernel_size=1)

        self.aux_loss = nn.CrossEntropyLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # special init for offset layers
        self.offset_c0.weight.data.zero_()
        self.offset_c1.weight.data.zero_()
        self.offset_c2.weight.data.zero_()
        self.offset_c3.weight.data.zero_()

        # four_bases = torch.cat([base_offset * 6, base_offset * 12, base_offset * 18, base_offset * 24], 0)
        four_bases = torch.cat([base_offset], 0)
        self.offset_c0.bias.data = four_bases / 4
        self.offset_c1.bias.data = four_bases / 4
        self.offset_c2.bias.data = four_bases / 4
        self.offset_c3.bias.data = four_bases / 4

        self.optimizer.weight.data.zero_()
        self.optimizer.bias.data.zero_()
        # set to zero as chenxi did in init model
        self.fc1_voc12.conv.weight.data.zero_()
        self.fc1_voc12.conv.bias.data.zero_()

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = x = self.layer4(x)

        # deeplab
        # x0 = self.fc1_voc12_c0(x)
        # x1 = self.fc1_voc12_c1(x)
        # x2 = self.fc1_voc12_c2(x)
        # x3 = self.fc1_voc12_c3(x)
        #
        # x = torch.add(x0, x1)
        # x = torch.add(x, x2)
        # x = torch.add(x, x3)

        # huiyu
        offset_c0 = self.offset_c0(feature)
        offset_c1 = self.offset_c1(feature)
        offset_c2 = self.offset_c2(feature)
        offset_c3 = self.offset_c3(feature)
        offset = offset_c0 + offset_c1 + offset_c2 + offset_c3

        output = self.fc1_voc12(feature, offset)

        # auxiliary loss
        # outputs = torch.t(output.view(21, -1))
        # delta_offsets = list()
        # for i in range(21):
        #     aux_loss = self.aux_loss(outputs, Variable(torch.LongTensor(outputs.size()[0]).cuda().zero_() + i))
        #     offset_grad, = grad(aux_loss, offset, retain_graph=True, create_graph=True)
        #     delta_offsets.append(offset_grad)
        #
        # temp = torch.cat(delta_offsets, dim=1)
        # delta_offset = self.optimizer(temp.detach())
        # offset = offset + delta_offset
        # output = self.fc1_voc12(feature, offset)

        return output


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