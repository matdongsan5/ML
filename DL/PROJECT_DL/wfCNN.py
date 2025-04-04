import torch
import torch.nn as nn
import torch.nn.functional as F

class wfCNN(nn.Module):
    def __init__(self, isDebug=False):
        super().__init__()
        self.isDebug = isDebug

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # (B, 3, 50, 50) -> (B, 32, 50, 50)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # (B, 32, 50, 50) -> (B, 64, 50, 50)
        self.pool = nn.MaxPool2d(2, 2)  # (B, 64, 50, 50) -> (B, 64, 25, 25)
        self.drop = nn.Dropout(0.25)

        # Fully Connected layers
        self.fc1 = nn.Linear(9216, 512)  # Flatten 후 FC 연결
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)  # 10개 클래스 (출력층)

    def forward(self, x):
        # if self.isDebug: print(f"Input shape: {x.shape}")

        x = self.pool(F.relu(self.conv1(x)))  # Conv + ReLU + MaxPooling
        x = self.pool(F.relu(self.conv2(x)))  

        # if self.isDebug: print(f"After Conv shape: {x.shape}")

        x = torch.flatten(x, 1)  # Flatten (B, 64, 25, 25) → (B, 64*25*25)
        # if self.isDebug: print(f"After Flatten shape: {x.shape}")

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Softmax 밖에서 적용 가능

        return x  # CrossEntropyLoss 사용 시 그대로 반환
