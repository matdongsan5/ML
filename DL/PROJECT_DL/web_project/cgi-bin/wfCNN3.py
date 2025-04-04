import torch
import torch.nn as nn
import torch.nn.functional as F


class wfCNN(nn.Module):
    def __init__(self, num_classes=1, isDebug=False):
        super().__init__()
        self.isDebug = isDebug

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(128)  

        self.pool = nn.MaxPool2d(2, 2)  # Downsampling
        self.dropout = nn.Dropout(0.4)

        # AdaptiveAvgPool으로 Flatten 크기 자동 조정
        self.global_avg_pool = nn.AdaptiveAvgPool2d((4, 4))  

        # Fully Connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)

        x = self.global_avg_pool(x)  # (B, 128, 4, 4)
        x = torch.flatten(x, 1)  # Flatten (B, 128*4*4)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x  # 다중 분류는 softmax 없이 raw logits 반환