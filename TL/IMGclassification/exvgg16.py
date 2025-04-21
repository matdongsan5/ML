#!/usr/bin/env python
# coding: utf-8

# #### [ VGG16 MODEL 살펴보기 ]

# In[1]:


## 모듈로딩
## 모듈 로딩
import torch                                    # 텐서
import torch.nn as nn                           # 인공신경망
import torch.nn.functional as F                 # 인공신경망함수

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms     # 이미지 전처리 변형
from torchvision.models import vgg16, VGG16_Weights         # 내장 데이터 셋
from torchinfo import summary    # 학습 데이터 로딩 관련

import numpy as np                                 # 데이터 저장 형식 관련 모듈
import matplotlib.pyplot as plt                 #   이미지 시각화 모듈


# In[2]:


WEIGHTS = VGG16_Weights.DEFAULT
WEIGHTS


# In[3]:


model16 = vgg16(WEIGHTS)

## 모델 층별 추출
FeatureLayers = model16.features    ## 이미지 특징 추출
ClassifierLayer = model16.classifier    ## 분류기 부분


# In[4]:


ClassifierLayer


# In[5]:


## 기존 vgg16의 출력층만 변경
model16.classifier[6]=nn.Linear(4096,2)


# In[6]:


FeatureLayers


# - 데이터셋 로드
# - 데이터로더 만들기
# - 학습/평가 
# - 테스트데이터 넣어보기

# In[7]:


## 준비 ==> 전처리용 transforms 인스턴ㅅ, 저장위치

ROOT = '../_data/pet/'

import os
if not os.path.exists(ROOT):
    os.makedirs(ROOT)
else:
    print(f'{ROOT}: 존재함')


# In[8]:


## 이미지크기가 64*64
TRANSFROM16 = VGG16_Weights.DEFAULT.transforms()
trainDS = ImageFolder(ROOT+'train',transform=TRANSFROM16)
testDS = ImageFolder(ROOT+'test',transform=TRANSFROM16)


# In[9]:


import numpy as np
len(testDS)


# In[10]:


from torch.utils.data import Subset
testlen = np.arange(len(testDS))
np.random.shuffle(testlen)

validDS = Subset(testDS, indices=testlen[:int(len(testDS)*0.5)])
testDS = Subset(testDS, indices=testlen[int(len(testDS)*0.5):])


# In[11]:


print(len(trainDS))
print(len(validDS))
print(len(testDS))


# In[12]:


for a, b in trainDS:
    print(a,b)
    break


# In[13]:


BATCH_SIZE = 500
trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True)
validDL = DataLoader(validDS, batch_size=BATCH_SIZE, shuffle=True)
testDL = DataLoader(testDS, batch_size=BATCH_SIZE, shuffle=True)


# In[14]:


for idx, (image, label) in enumerate(trainDL):
    print(idx, image.shape,label.shape)
    break


# In[15]:


import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR


# In[16]:


LOSSFN = nn.BCEWithLogitsLoss()
LR              = 0.01
EPOCHS          = 100
STEP_SIZE       = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[17]:


MODEL = model16
OPTIMIZER = torch.optim.Adam(filter(lambda p: p.requires_grad, MODEL.parameters()), lr=LR)
SCHEDULER = StepLR(OPTIMIZER, STEP_SIZE, gamma=0.1)


# In[18]:


def train(DL):
    MODEL.train()
    
    total_loss, total_acc = 0, 0
    
    for idx, (image, label) in enumerate(DL):
        
        OPTIMIZER.zero_grad()
        pre = MODEL(image)
        
        loss = LOSSFN(pre,label.reshape(-1,1))
        loss.backward()
        
        OPTIMIZER.step()
        
        total_loss += loss.item()
        total_acc += (pre.argmax(dim=1) == label).sum().item()
        
    return total_loss/(idx+1), total_acc/(idx+1)


# In[19]:


def evaluate(DL):
    MODEL.eval()
    
    total_loss, total_acc = 0, 0
    
    for idx, (image, label) in enumerate(DL):
        
        pre = MODEL(image)
        
        loss = LOSSFN(pre,label.reshape(-1,1))
        total_loss += loss.item()
        total_acc += (pre.argmax(dim=1) == label).sum().item()
        
    return total_loss/(idx+1), total_acc/(idx+1)


# In[ ]:


EPOCH = 5

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(trainDL)
    valid_loss, valid_acc = evaluate(validDL)
    
    SCHEDULER.step()

    print("-" * 59)
    print(f'| end of epoch {epoch:3d} | train acc {train_acc:8.3f}  | valid acc {valid_acc:8.3f}')
    print(f'| end of epoch {epoch:3d} | train loss {train_loss:8.3f}  | valid loss {valid_loss:8.3f}')
    print("-" * 59)


# In[20]:


for image, label in testDS:
        pre = MODEL(image)
        print(pre.argmax(), label)
        break
        

