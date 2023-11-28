import torch
import torch.nn as nn

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(EfficientNetB0, self).__init__()

        # Define the blocks and layers for EfficientNetB0
        self.conv1 = nn.Conv2d(3, int(32 * width_mult), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * width_mult))
        self.act1 = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            self._make_block(in_channels=int(32 * width_mult), out_channels=int(16 * width_mult), num_layers=1, stride=1),
            self._make_block(in_channels=int(16 * width_mult), out_channels=int(24 * width_mult), num_layers=2, stride=2),
            self._make_block(in_channels=int(24 * width_mult), out_channels=int(40 * width_mult), num_layers=2, stride=2),
            self._make_block(in_channels=int(40 * width_mult), out_channels=int(80 * width_mult), num_layers=3, stride=2),
            self._make_block(in_channels=int(80 * width_mult), out_channels=int(112 * width_mult), num_layers=3, stride=1),
            self._make_block(in_channels=int(112 * width_mult), out_channels=int(192 * width_mult), num_layers=4, stride=2),
            self._make_block(in_channels=int(192 * width_mult), out_channels=int(320 * width_mult), num_layers=1, stride=1),
        )

        self.conv2 = nn.Conv2d(int(320 * width_mult), int(1280 * width_mult), kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(1280 * width_mult))
        self.act2 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1280 * width_mult), num_classes)

    def _make_block(self, in_channels, out_channels, num_layers, stride):
        layers = []
        for i in range(num_layers):
            layers.append(MBConvBlock(in_channels, out_channels, stride if i == 0 else 1))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(MBConvBlock, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))

        if self.stride == 1 and identity.shape[1] == x.shape[1]:
            x += identity

        return x

def efficientnet_b0():
    return EfficientNetB0()
