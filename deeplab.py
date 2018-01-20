import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from modules.SampleConv import SampleConv
from modules.UpdateConv2d import UpdateConv2d
from modules.SampleBottleneck import SampleBottleneck, UpdateSampleBottleneck
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # deeplab change
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
    def __init__(self, block, layers, num_classes, samples, update):
        self.inplanes = 64
        self.update = update
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)  # deeplab change
        self.layer4 = self._make_layer(SampleBottleneck, 512, layers[3], stride=1, dilation=4)  # deeplab change

        self.top_offset = SampleConv(512 * block.expansion, samples * 2, samples, 0, 0, groups=1)
        self.top = SampleConv(512 * block.expansion, num_classes, samples, 0, 0, groups=1)  # huiyu change

        self.aux_loss = nn.CrossEntropyLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # special init for offset layers
        base_offset = torch.from_numpy(
            np.array([-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1]).astype(np.float32))  # w, h

        if samples == 36:
            self.top_offset_bias = torch.cat([base_offset * 6, base_offset * 12, base_offset * 18, base_offset * 24], 0)
        if samples == 9:
            self.top_offset_bias = torch.cat([base_offset * 6], 0)

        self.layer4[0].conv2_offset.weight.data.zero_()
        self.layer4[0].conv2_offset.bias.data = torch.cat([base_offset * 4], 0)
        self.layer4[1].conv2_offset.weight.data.zero_()
        self.layer4[1].conv2_offset.bias.data = torch.cat([base_offset * 4], 0)
        self.layer4[2].conv2_offset.weight.data.zero_()
        self.layer4[2].conv2_offset.bias.data = torch.cat([base_offset * 4], 0)
        self.top_offset.conv.weight.data.zero_()
        self.top_offset.conv.bias.data = self.top_offset_bias

        # set to zero as chenxi did in init model
        self.top.conv.weight.data.zero_()
        self.top.conv.bias.data.zero_()

        self.writer = SummaryWriter()
        self.global_step = 0
        self.hooks = list()

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

        for module in self.modules():
            if hasattr(module, 'need_update'):
                module.need_update = None
                module.grad_update.clear()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # deformable convolution offset sample points
        temp = self.top_offset_bias.cuda().view(1, -1, 1, 1).expand(1, -1, x.size(2), x.size(3))
        offset_offset = Variable(temp, requires_grad=False)

        # first forward: output_step0
        # x_detach = x.detach()
        # feature = self.layer4(x_detach)
        # offset = self.top_offset(feature, offset_offset)
        # output = self.top(feature, offset)

        # initialize lists
        need_updates = list()
        grad_updates = list()
        module_updates = list()
        for module in self.modules():
            if hasattr(module, 'need_update'):
                need_updates.append(module.need_update)
                grad_updates.append(module.grad_update)
                module_updates.append(module)

        for index in range(len(need_updates)):
            Printer('Forward:  update0_block' + str(index), self.writer, self.global_step).print_var_par(module_updates[index].need_update)
            # module_updates[index].need_update.register_hook(Printer('Backward: update0_block' + str(index), self.writer, self.global_step).print_var_par)

        if self.update:
            # auxiliary losses and compute gradients
            label0 = torch.t(output.view(21, -1))
            for class_id in range(21):
                aux_loss = self.aux_loss(label0, Variable(torch.LongTensor(label0.size()[0]).cuda().zero_() + class_id))
                gradient = grad(aux_loss, need_updates, retain_graph=True, create_graph=False)
                for variable_id in range(len(grad_updates)):
                    gradient[variable_id].volatile = False
                    # gradients are not volatile and do not require gradient
                    grad_updates[variable_id].append(gradient[variable_id])
        else:
            # trivial zero gradients
            for class_id in range(21):
                for variable_id in range(len(grad_updates)):
                    grad_updates[variable_id].append(need_updates[variable_id].detach() * 0)

        # last forward: output_step1
        feature = self.layer4(x)
        offset = self.top_offset(feature, offset_offset)
        output = self.top(feature, offset)

        for index in range(len(module_updates)):
            Printer('Forward:  update1_block' + str(index), self.writer, self.global_step).print_var_par(module_updates[index].need_update)
            # module_updates[index].need_update.register_hook(Printer('Backward: update1_block' + str(index), self.writer, self.global_step).print_var_par)

        # printing
        # Printer('Forward:  feature', self.writer, self.global_step).print_var_par(feature)
        # feature.register_hook(Printer('Backward: feature', self.writer, self.global_step).print_var_par)

        # Printer('Forward:  offset_step0', self.writer, self.global_step).print_var_par(offset_step0)
        # Printer('Forward:  gradient', self.writer, self.global_step).print_var_par(gradient)

        # Printer('Forward:  offset_to_bottle.weight', self.writer, self.global_step).print_var_par(self.offset_to_bottle.weight)
        # Printer('Forward:  offset_to_bottle.bias', self.writer, self.global_step).print_var_par(self.offset_to_bottle.bias)
        # Printer('Forward:  gradient_to_bottle.weight', self.writer, self.global_step).print_var_par(self.gradient_to_bottle.weight)
        # Printer('Forward:  gradient_to_bottle.bias', self.writer, self.global_step).print_var_par(self.gradient_to_bottle.bias)
        # Printer('Forward:  bottle_to_delta.weight', self.writer, self.global_step).print_var_par( self.bottle_to_delta.weight)
        # Printer('Forward:  bottle_to_delta.bias', self.writer, self.global_step).print_var_par(self.bottle_to_delta.bias)

        # self.hooks.append(self.gradient_to_bottle.weight.register_hook(
        #     Printer('Backward: gradient_to_bottle.weight', self.writer, self.global_step).print_var_par))
        # self.hooks.append(self.gradient_to_bottle.bias.register_hook(
        #     Printer('Backward: gradient_to_bottle.bias', self.writer, self.global_step).print_var_par))
        # self.hooks.append(self.offset_to_bottle.weight.register_hook(
        #     Printer('Backward: offset_to_bottle.weight', self.writer, self.global_step).print_var_par))
        # self.hooks.append(self.offset_to_bottle.bias.register_hook(
        #     Printer('Backward: offset_to_bottle.bias', self.writer, self.global_step).print_var_par))
        # self.hooks.append(self.bottle_to_delta.weight.register_hook(
        #     Printer('Backward: bottle_to_delta.weight', self.writer, self.global_step).print_var_par))
        # self.hooks.append(self.bottle_to_delta.bias.register_hook(
        #     Printer('Backward: bottle_to_delta.bias', self.writer, self.global_step).print_var_par))
        #
        # Printer('Forward:  bottle_from_offset', self.writer, self.global_step).print_var_par(bottle_from_offset)
        # Printer('Forward:  bottle_from_gradient', self.writer, self.global_step).print_var_par(bottle_from_gradient)
        # Printer('Forward:  delta_offset', self.writer, self.global_step).print_var_par(delta_offset)

        # Printer('Forward:  offset_step1', self.writer, self.global_step).print_var_par(offset_step1)
        # offset_step1.register_hook(Printer('Backward: offset_step1', self.writer, self.global_step).print_var_par)

        # Printer('Forward:  output_step1', self.writer, self.global_step).print_var_par(output_step1)
        # output_step1.register_hook(Printer('Backward: output_step1', self.writer, self.global_step).print_var_par)

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


def resnet101(pretrained=False, num_classes=21, samples=36, update=False):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes
        samples
        update
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, samples=samples, update=update)
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model
