import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


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
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2) # deeplab change
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # deeplab change
        self.fc1_voc12_c0 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                      stride=1, dilation=6, padding=6) # deeplab change
        self.fc1_voc12_c1 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                      stride=1, dilation=12, padding=12) # deeplab change
        self.fc1_voc12_c2 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                      stride=1, dilation=18, padding=18) # deeplab change
        self.fc1_voc12_c3 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                      stride=1, dilation=24, padding=24) # deeplab change

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.layer4(x)

        x0 = self.fc1_voc12_c0(x)
        x1 = self.fc1_voc12_c1(x)
        x2 = self.fc1_voc12_c2(x)
        x3 = self.fc1_voc12_c3(x)

        x = torch.add(x0, x1)
        x = torch.add(x, x2)
        x = torch.add(x, x3)

        return x

class ResSubModel1(nn.Module):
    def __init__(self, block, layers, num_classes=21):
          self.inplanes = 64
          super(ResSubModel1, self).__init__()
          self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
          self.bn1 = nn.BatchNorm2d(64)
          self.relu = nn.ReLU(inplace=True)
          self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
          self.layer1 = self._make_layer(block, 64, layers[0])
          self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
          self.layer31 = self._make_layer(block, 256, layers[2][0], stride=1, dilation=2) # deeplab change
          self.layer32 = self._make_layer(block, 256, layers[2][1], stride=1, dilation=2)
          self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # deeplab change
          self.fc1_voc12_c0 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                        stride=1, dilation=6, padding=6) # deeplab change
          self.fc1_voc12_c1 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                        stride=1, dilation=12, padding=12) # deeplab change
          self.fc1_voc12_c2 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                        stride=1, dilation=18, padding=18) # deeplab change
          self.fc1_voc12_c3 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                        stride=1, dilation=24, padding=24) # deeplab change

          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                  m.weight.data.normal_(0, math.sqrt(2. / n))
              elif isinstance(m, nn.BatchNorm2d):
                  m.weight.data.fill_(1)
                  m.bias.data.zero_()

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

        return x

class ResSubModel2(nn.Module):
    def __init__(self, block, layers, num_classes=21):
          self.inplanes = 64
          super(ResSubModel2, self).__init__()
          self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
          self.bn1 = nn.BatchNorm2d(64)
          self.relu = nn.ReLU(inplace=True)
          self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
          self.layer1 = self._make_layer(block, 64, layers[0])
          self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
          self.layer31 = self._make_layer(block, 256, layers[2][0], stride=1, dilation=2) # deeplab change
          self.layer32 = self._make_layer(block, 256, layers[2][1], stride=1, dilation=2)
          self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # deeplab change
          self.fc1_voc12_c0 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                        stride=1, dilation=6, padding=6) # deeplab change
          self.fc1_voc12_c1 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                        stride=1, dilation=12, padding=12) # deeplab change
          self.fc1_voc12_c2 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                        stride=1, dilation=18, padding=18) # deeplab change
          self.fc1_voc12_c3 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                        stride=1, dilation=24, padding=24) # deeplab change

          for m in self.modules():
              if isinstance(m, nn.Conv2d):
                  n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                  m.weight.data.normal_(0, math.sqrt(2. / n))
              elif isinstance(m, nn.BatchNorm2d):
                  m.weight.data.fill_(1)
                  m.bias.data.zero_()

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
        x = self.layer31(x)

        return x

class ResSubModel3(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(ResSubModel3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer31 = self._make_layer(block, 256, layers[2][0], stride=1, dilation=2) # deeplab change
        self.layer32 = self._make_layer(block, 256, layers[2][1], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # deeplab change
        self.fc1_voc12_c0 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                      stride=1, dilation=6, padding=6) # deeplab change
        self.fc1_voc12_c1 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                      stride=1, dilation=12, padding=12) # deeplab change
        self.fc1_voc12_c2 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                      stride=1, dilation=18, padding=18) # deeplab change
        self.fc1_voc12_c3 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                      stride=1, dilation=24, padding=24) # deeplab change

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.layer32(x)

        return x

class ResSubModel4(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        self.inplanes = 64
        super(ResSubModel4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer31 = self._make_layer(block, 256, layers[2][0], stride=1, dilation=2) # deeplab change
        self.layer32 = self._make_layer(block, 256, layers[2][1], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4) # deeplab change
        self.fc1_voc12_c0 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                      stride=1, dilation=6, padding=6) # deeplab change
        self.fc1_voc12_c1 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                      stride=1, dilation=12, padding=12) # deeplab change
        self.fc1_voc12_c2 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3, 
                                      stride=1, dilation=18, padding=18) # deeplab change
        self.fc1_voc12_c3 = nn.Conv2d(512 * block.expansion, num_classes, kernel_size=3,
                                      stride=1, dilation=24, padding=24) # deeplab change

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.layer4(x)

        x0 = self.fc1_voc12_c0(x)
        x1 = self.fc1_voc12_c1(x)
        x2 = self.fc1_voc12_c2(x)
        x3 = self.fc1_voc12_c3(x)

        x = torch.add(x0, x1)
        x = torch.add(x, x2)
        x = torch.add(x, x3)

        return x

class ResNetModelParallel(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        super(ResNetModelParallel, self).__init__()
        self.submodel1 = ResSubModel1(block, layers, num_classes)
        self.submodel2 = ResSubModel2(block, layers, num_classes)
        self.submodel3 = ResSubModel3(block, layers, num_classes)
        self.submodel4 = ResSubModel4(block, layers, num_classes)

        self.submodel1.cuda(0)
        self.submodel2.cuda(1)
        self.submodel3.cuda(2)
        self.submodel4.cuda(3)

    def forward(self, x):
        x = self.submodel1(x)
        x = x.cuda(1) # P2P GPU transfer
        x = self.submodel2(x)
        x = x.cuda(2)
        x = self.submodel3(x)
        x = x.cuda(3)
        x = self.submodel4(x)
        return x.cuda(0)

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
    model = ResNetModelParallel(Bottleneck, [3, 4, [10,13], 3])
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model
