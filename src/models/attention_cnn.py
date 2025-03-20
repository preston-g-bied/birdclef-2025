import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class AttentionCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()

        # feature extraction layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.se4 = SEBlock(256)

        # classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.max_pool2d(x, 2)

        # block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = F.max_pool2d(x, 2)

        # block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = F.max_pool2d(x, 2)

        # block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
        x = F.max_pool2d(x, 2)

        # global pooling and classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x