
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms

# 경로 상수
IMG_ROOT_tr = '../group7/imgDS/Forest Fire_Dataset/train/'
IMG_ROOT_ts = '../group7/imgDS/Forest Fire_Dataset/test/'
IMG_ROOT_va = '../group7/imgDS/Forest Fire_Dataset/val/'
MODEL_DIR = '../group7/models/'
WEIGHTS_FILE = MODEL_DIR + 'Fire_DETECT_epoch7_0.808.pt'


def get_datasets():
    preprocessing = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainDS = ImageFolder(IMG_ROOT_tr, transform=preprocessing)
    testDS = ImageFolder(IMG_ROOT_ts, transform=preprocessing)
    validDS = ImageFolder(IMG_ROOT_va, transform=preprocessing)
    return trainDS, testDS, validDS

# 모델 정의
class FDetectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(weights_path):
    model = FDetectionCNN()
    states = torch.load(weights_path, weights_only=True)
    model.load_state_dict(states)
    return model


def evaluate(model, dataset):
    count = 0
    for feature, target in dataset:
        feature = feature.unsqueeze(0)  # 배치 차원 추가
        output = model(feature)
        predicted = (output.sigmoid() > 0.5).long().squeeze()
        if predicted.item() == target:
            count += 1
    accuracy = (count / len(dataset)) * 100
    return accuracy


if __name__ == '__main__':
    trainDS, testDS, validDS = get_datasets()
    model = load_model(WEIGHTS_FILE)
    acc = evaluate(model, testDS)
    print(f"Accuracy = {acc:.2f}%")




