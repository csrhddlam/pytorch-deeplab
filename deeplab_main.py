import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from DataParallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather, scatter
import os
import sys
import deeplab
from PIL import Image
import math
import datetime

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


def extract(xVar):
    a = 0


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    pascal_dir = './VOC2012/'
    list_dir = './deeplab_list/'
    model_fname = 'model/deeplab101_epoch%d.pth'

    model = getattr(deeplab, 'resnet101')()

    if sys.argv[2] == 'train':
        model.eval()  # in order to fix batchnorm
        model.load_state_dict(torch.load('model/deeplab101_init.pth'), strict=False)
        if use_gpu:
            model = model.cuda()
        num_epochs = 2
        iter_size = 10
        base_lr = 0.00025 / iter_size 
        power = 0.9
        criterion = nn.CrossEntropyLoss().cuda()

        optimizer = optim.SGD([{'params': model.conv1.parameters()},
            {'params': model.bn1.parameters()},
            {'params': model.layer1.parameters()},
            {'params': model.layer2.parameters()},
            {'params': model.layer3.parameters()},
            {'params': model.layer4.parameters()},
            {'params': iter([model.fc1_voc12_c0.weight,
                model.fc1_voc12_c1.weight,
                model.fc1_voc12_c2.weight,
                model.fc1_voc12_c3.weight])},
            {'params': iter([model.fc1_voc12_c0.bias,
                model.fc1_voc12_c1.bias,
                model.fc1_voc12_c2.bias,
                model.fc1_voc12_c3.bias]), 'weight_decay': 0.}],
            lr=base_lr, momentum=0.9, weight_decay=0.0005)
        np.random.seed(1234)
        losses = AverageMeter()
        lines = np.loadtxt(list_dir + 'train_aug.txt', dtype=str)
        for epoch in range(num_epochs):
            lines = np.random.permutation(lines)
            for i, line in enumerate(lines):
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                lr = base_lr * math.pow(1 - float(epoch * len(lines) + i) / (num_epochs * len(lines)), power)
                for g in range(6):
                    optimizer.param_groups[g]['lr'] = lr
                optimizer.param_groups[6]['lr'] = lr * 10
                optimizer.param_groups[7]['lr'] = lr * 20

                imname, labelname = line
                im = datasets.folder.default_loader(pascal_dir + imname)
                label = Image.open(pascal_dir + labelname)
                inputs = data_transforms(im)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                else:
                    inputs = Variable(inputs)
                inputs = inputs.unsqueeze(0)
                inputs.requires_grad = True

                # im_size = 1200  # fake images
                # inputs = torch.arange(3 * im_size * im_size).view(1, 3, im_size, im_size)
                stride = math.ceil((inputs.size(3) / len(model.layer0_0.device_ids)) / 8) * 8
                temp = list()
                for d in range(len(model.layer0_0.device_ids)):
                    if d < len(model.layer0_0.device_ids) - 1:
                        temp.append((inputs[:, :, :, stride * d: stride * (d + 1)].cuda(model.layer0_0.device_ids[d]), ))
                    else:
                        temp.append((inputs[:, :, :, stride * d:].cuda(model.layer0_0.device_ids[d]), ))

                outputs = model(*temp)

                outputs = torch.cat([outputs[d][0].cuda(model.layer0_0.device_ids[0])
                                     for d in range(len(model.layer0_0.device_ids))], dim=3)

                w, h = outputs.size()[2], outputs.size()[3]
                label_down = label.resize((h, w), Image.NEAREST)
                target_down = torch.LongTensor(np.array(label_down).astype(np.int64))
                if use_gpu:
                    target_down = Variable(target_down.cuda(model.layer0_0.device_ids[0]))
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

                print(str(datetime.datetime.now()) + '\t'
                      'epoch: {0}\t'
                      'iter: {1}/{2}\t'
                      'lr: {3:.6f}\t'
                      'loss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch+1, i+1, len(lines), lr, loss=losses))
                # prof.export_chrome_trace('trace')

            torch.save(model.state_dict(), model_fname % (epoch+1))

    elif sys.argv[2] == 'eval':
        model.eval()
        model.load_state_dict(torch.load('model/deeplab101_trainaug.pth'), strict=False)
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

            inputs = inputs.unsqueeze(0)

            stride = math.ceil((inputs.size(3) / len(model.layer0_0.device_ids)) / 8) * 8
            temp = list()
            for d in range(len(model.layer0_0.device_ids)):
                if d < len(model.layer0_0.device_ids) - 1:
                    temp.append((inputs[:, :, :, stride * d: stride * (d + 1)].cuda(model.layer0_0.device_ids[d]),))
                else:
                    temp.append((inputs[:, :, :, stride * d:].cuda(model.layer0_0.device_ids[d]),))

            outputs = model(*temp)
            outputs = torch.cat([outputs[d][0].cuda(model.layer0_0.device_ids[0])
                                 for d in range(len(model.layer0_0.device_ids))], dim=3)

            outputs_up = nn.UpsamplingBilinear2d((w, h))(outputs)
            _, pred = torch.max(outputs_up, 1)
            pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
            seg = Image.fromarray(pred)
            seg.save('data/val/' + imname + '.png')
            print('processing %d/%d' % (i + 1, len(lines)))