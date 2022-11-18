import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.residual_function(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class _resnet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=200):
        super(_resnet18, self).__init__()
        self.in_channel = 64
        # Input: 3*32*32 Output: 64*32*32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(BasicBlock, 64, 2, stride=2)     # Output: 64*16*16
        self.conv3_x = self._make_layer(BasicBlock, 128, 2, stride=2)    # Output: 128*8*8
        self.conv4_x = self._make_layer(BasicBlock, 256, 2, stride=2)    # Output: 256*4*4
        self.conv5_x = self._make_layer(BasicBlock, 512, 2, stride=2)    # Output: 256*2*2
        self.avg = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(512, num_classes)
        # self.dp = nn.Dropout(0.5)

    def _make_layer(self, block, channels, num_blocks, stride):
        # strides=[1, 1] or [2, 1]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return _resnet18(BasicBlock)