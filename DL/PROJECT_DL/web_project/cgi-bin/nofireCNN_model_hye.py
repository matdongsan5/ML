import cgi
import os
import sys
import codecs
import joblib  # 🔹 pkl 파일 로드를 위한 라이브러리
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 🔹 손상된 이미지 강제 로드

# --------------------------------------
# 1. HTTP 응답 헤더 출력 (HTML 콘텐츠 타입 설정)
# --------------------------------------

# 2. 데이터 전처리 함수 정의
# --------------------------------------
def get_preprocessing():
    """
    이미지 전처리 파이프라인을 반환하는 함수.

    Returns:
        torchvision.transforms.Compose: 이미지 변환을 수행하는 Compose 객체
    """
    return transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# --------------------------------------
# 3. CNN 모델 정의
# --------------------------------------
class NoFireForestCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
             
        )

    def forward(self, x):
        x = self.conv_layers(x) 
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
