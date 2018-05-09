import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import deeplab_model_parallel

criterion = nn.CrossEntropyLoss().cuda()
model = getattr(deeplab_model_parallel, 'resnet101')()

input = torch.randn(1,3,2560,1600).cuda()
#target = torch.zeros(1,21,1920,1280).long().view(-1).cuda()


for i in range(11):
    if i == 1:
        t0 = time.time()
    output = model(input)
    #print(output.size())
    output = output.view(-1, 21)
    #print(output.size())
    target = torch.zeros(output.size()[0]).long().view(-1).cuda()
    #print(target.size())
    loss = criterion(output, target)
    loss.backward()

t1 = time.time()

print(t1 - t0)
