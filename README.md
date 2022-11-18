# DLP_HW2
Second programming assignment for deep learning practice：
Read the paper of LeNet、AlexNet、ResNet and compare the differences and connections between them.Comparative tests are carried out on the basis of Resnet
------
## 1. ResNet
This experiment was performed on the Tiny-ImageNet dataset with a image size of 32*32.So I use a shallow network ResNet18.
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [256, 64, 64, 64]           1,728
       BatchNorm2d-2          [256, 64, 64, 64]             128
              ReLU-3          [256, 64, 64, 64]               0
         MaxPool2d-4          [256, 64, 32, 32]               0
            Conv2d-5          [256, 64, 16, 16]          36,864
       BatchNorm2d-6          [256, 64, 16, 16]             128
              ReLU-7          [256, 64, 16, 16]               0
            Conv2d-8          [256, 64, 16, 16]          36,864
       BatchNorm2d-9          [256, 64, 16, 16]             128
           Conv2d-10          [256, 64, 16, 16]           4,096
      BatchNorm2d-11          [256, 64, 16, 16]             128
       BasicBlock-12          [256, 64, 16, 16]               0
           Conv2d-13          [256, 64, 16, 16]          36,864
      BatchNorm2d-14          [256, 64, 16, 16]             128
             ReLU-15          [256, 64, 16, 16]               0
           Conv2d-16          [256, 64, 16, 16]          36,864
      BatchNorm2d-17          [256, 64, 16, 16]             128
       BasicBlock-18          [256, 64, 16, 16]               0
           Conv2d-19           [256, 128, 8, 8]          73,728
      BatchNorm2d-20           [256, 128, 8, 8]             256
             ReLU-21           [256, 128, 8, 8]               0
           Conv2d-22           [256, 128, 8, 8]         147,456
      BatchNorm2d-23           [256, 128, 8, 8]             256
           Conv2d-24           [256, 128, 8, 8]           8,192
      BatchNorm2d-25           [256, 128, 8, 8]             256
       BasicBlock-26           [256, 128, 8, 8]               0
           Conv2d-27           [256, 128, 8, 8]         147,456
      BatchNorm2d-28           [256, 128, 8, 8]             256
             ReLU-29           [256, 128, 8, 8]               0
           Conv2d-30           [256, 128, 8, 8]         147,456
      BatchNorm2d-31           [256, 128, 8, 8]             256
       BasicBlock-32           [256, 128, 8, 8]               0
           Conv2d-33           [256, 256, 4, 4]         294,912
      BatchNorm2d-34           [256, 256, 4, 4]             512
             ReLU-35           [256, 256, 4, 4]               0
           Conv2d-36           [256, 256, 4, 4]         589,824
      BatchNorm2d-37           [256, 256, 4, 4]             512
           Conv2d-38           [256, 256, 4, 4]          32,768
      BatchNorm2d-39           [256, 256, 4, 4]             512
       BasicBlock-40           [256, 256, 4, 4]               0
           Conv2d-41           [256, 256, 4, 4]         589,824
      BatchNorm2d-42           [256, 256, 4, 4]             512
             ReLU-43           [256, 256, 4, 4]               0
           Conv2d-44           [256, 256, 4, 4]         589,824
      BatchNorm2d-45           [256, 256, 4, 4]             512
       BasicBlock-46           [256, 256, 4, 4]               0
           Conv2d-47           [256, 512, 2, 2]       1,179,648
      BatchNorm2d-48           [256, 512, 2, 2]           1,024
             ReLU-49           [256, 512, 2, 2]               0
           Conv2d-50           [256, 512, 2, 2]       2,359,296
      BatchNorm2d-51           [256, 512, 2, 2]           1,024
           Conv2d-52           [256, 512, 2, 2]         131,072
      BatchNorm2d-53           [256, 512, 2, 2]           1,024
       BasicBlock-54           [256, 512, 2, 2]               0
           Conv2d-55           [256, 512, 2, 2]       2,359,296
      BatchNorm2d-56           [256, 512, 2, 2]           1,024
             ReLU-57           [256, 512, 2, 2]               0
           Conv2d-58           [256, 512, 2, 2]       2,359,296
      BatchNorm2d-59           [256, 512, 2, 2]           1,024
       BasicBlock-60           [256, 512, 2, 2]               0
        AvgPool2d-61           [256, 512, 1, 1]               0
           Linear-62                 [256, 200]         102,600
================================================================
Total params: 11,275,656
Trainable params: 11,275,656
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 12.00
Forward/backward pass size (MB): 2505.39
Params size (MB): 43.01
Estimated Total Size (MB): 2560.40
----------------------------------------------------------------
