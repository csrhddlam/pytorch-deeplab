# pytorch-deeplab (cs420 graduate project code)

This repository contains image parallel training and testing codes for DeepLab ResNet 101.

You can train on `train_aug` and evaluate on `val` of PASCAL VOC2012 on multiple GPUs by 

```
python deeplab_main.py 0,1,2,3 train
python deeplab_main.py 0,1,2,3 eval
```

During evaluation you will also need to run `EvalSegResults.m` in the `matlab` folder to compute the mean IOU.

## Acknowledgement
A large part of the code is borrowed from [https://github.com/chenxi116/pytorch-deeplab](https://github.com/chenxi116/pytorch-deeplab). (see commit history)

## Future Work
Current parallel code doesn't work perfectly for very small images, but it can be easily modified to use only one GPU for small images, and still use multiple GPUs for large ones.
