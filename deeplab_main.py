import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import sys
import deeplab
from PIL import Image
import math
import datetime

from util import *

tensorboard_step = 10

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        # self.avg = self.sum / self.count
        self.avg = self.avg * 0.99 + self.val * 0.01


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    pascal_dir = '/mnt/4T-HD/why/Data/VOCdevkit2012/VOC2012/'
    list_dir = '/mnt/4T-HD/why/Data/deeplab_list/'
    model_fname = '/home/why/Documents/pytorch-deeplab/model/deeplab101_grad_conv1_10_10_10000_epoch%d.pth'

    model = getattr(deeplab, 'resnet101')()
    num_epochs = 4

    if 'train' in sys.argv[2]:
        model.eval()  # in order to fix batchnorm
        state_dict = torch.load('/home/why/Documents/pytorch-deeplab/model/deeplab101_init.pth')
        model.load_state_dict(state_dict, strict=False)

        if use_gpu:
            model = model.cuda()

        iter_size = 10
        base_lr = 0.00025 / iter_size 
        power = 0.9
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([{'params': model.conv1.parameters()},
            {'params': model.bn1.parameters()},
            {'params': model.layer1.parameters()},
            {'params': model.layer2.parameters()},
            {'params': model.layer3.parameters()},
            {'params': model.layer4.parameters()},
            {'params': iter([model.sample_conv.conv.weight])},
            {'params': iter([model.sample_conv.conv.bias]), 'weight_decay': 0.},
            {'params': iter([model.conv_to_offset0.weight,
                             model.conv_to_offset1.weight,
                             model.conv_to_offset2.weight,
                             model.conv_to_offset3.weight])},
            {'params': iter([model.conv_to_offset0.bias,
                             model.conv_to_offset1.bias,
                             model.conv_to_offset2.bias,
                             model.conv_to_offset3.bias]), 'weight_decay': 0.},
            {'params': iter([model.gradient_to_bottle.weight,
                             model.offset_to_bottle.weight,
                             model.bottle_to_delta.weight])},
            {'params': iter([model.gradient_to_bottle.bias,
                             model.offset_to_bottle.bias,
                             model.bottle_to_delta.bias]), 'weight_decay': 0.}],
            lr=base_lr, weight_decay=0.0005)

        losses = AverageMeter()
        lines = np.loadtxt(list_dir + 'train_aug.txt', dtype=str)
        for epoch in range(num_epochs):
            lines = np.random.permutation(lines)
            for i, line in enumerate(lines):
                global_step = epoch * len(lines) + i

                lr = base_lr * math.pow(1 - float(epoch * len(lines) + i) / (num_epochs * len(lines)), power)
                for g in range(6):
                    optimizer.param_groups[g]['lr'] = lr
                optimizer.param_groups[6]['lr'] = lr * 10
                optimizer.param_groups[7]['lr'] = lr * 20

                optimizer.param_groups[8]['lr'] = lr * 10
                optimizer.param_groups[9]['lr'] = lr * 10
                optimizer.param_groups[10]['lr'] = lr * 10
                optimizer.param_groups[11]['lr'] = lr * 10

                imname, labelname = line
                im = datasets.folder.default_loader(pascal_dir + imname)
                label = Image.open(pascal_dir + labelname)
                inputs = data_transforms(im)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                else:
                    inputs = Variable(inputs)
                outputs = model(inputs.unsqueeze(0))
                w, h = outputs.size()[2], outputs.size()[3]
                label_down = label.resize((h, w), Image.NEAREST)
                target_down = torch.LongTensor(np.array(label_down).astype(np.int64))
                if use_gpu:
                    target_down = Variable(target_down.cuda())
                else:
                    target_down = Variable(target_down)

                target_down = target_down.view(-1,)
                mask = torch.lt(target_down, 21)
                target_down = torch.masked_select(target_down, mask)
                outputs = torch.masked_select(outputs.view(-1), mask.repeat(21).view(-1))
                outputs = torch.t(outputs.view(21, -1))

                loss = criterion(outputs, target_down)
                losses.update(loss.data[0], 1)

                if i % iter_size == 0:
                    optimizer.zero_grad()
                loss.backward()
                if i % iter_size == iter_size - 1:
                    optimizer.step()

                print(datetime.datetime.now())
                print('epoch: {0}\t'
                      'iter: {1}/{2}\t'
                      'lr: {3:.6f}\t'
                      'loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch+1, i+1, len(lines), lr, loss=losses))

                if i % tensorboard_step == 0:
                    # for name, param in model.named_parameters():
                        # if 'layer' not in name:
                        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch * len(lines) + i)
                    model.writer.add_scalar('training_loss', losses.val, model.global_step)

            torch.save(model.state_dict(), model_fname % (epoch+1))

    if 'eval' in sys.argv[2]:
        model.eval()
        state_dict = torch.load(model_fname % num_epochs)
        model.load_state_dict(state_dict, strict=False)
        if use_gpu:
            model = model.cuda()

        lines = np.loadtxt(list_dir + 'val_id.txt', dtype=str)
        for i, imname in enumerate(lines):
            im = datasets.folder.default_loader(pascal_dir + 'JPEGImages/' + imname + '.jpg')
            w, h = np.shape(im)[0], np.shape(im)[1]
            inputs = data_transforms(im)
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            outputs = model(inputs.unsqueeze(0))
            outputs_up = nn.UpsamplingBilinear2d((w, h))(outputs)
            _, pred = torch.max(outputs_up, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            seg = Image.fromarray(pred)
            seg.save('data/val' + sys.argv[1] + '/' + imname + '.png')
            print('processing %d/%d' % (i + 1, len(lines)))


