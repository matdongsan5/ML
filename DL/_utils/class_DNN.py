#### [ DNN 클래스 모듈 ]
""" 
    1. class Module_DNN
        학습 모델 제작
    
    2. class Module_TT
        training()  train
        evaluate()  test용 함수.
"""
## 모듈로딩
import pandas as pd  # 데이터 모듈
import numpy as np

import torch         
# tensor 및 기본 함수 모듈   
import torch.nn as nn
# 인공신경망 관련 모듈
import torch.nn.functional as F
# 인공신경망 관련 함수
import torch.optim as optim
# 최적화 모듈

from collections import OrderedDict

from sklearn.model_selection import train_test_split
# 학습용 데이터셋 관련 함수
class Module_DNN(nn.Module):
    """ 
    훈련 데이터 넣을 때 데이터타입에 맞는 텐서로 넣기  권장.
    순서가 있는 dict를 
    ('conv1', nn.Conv2d(1,20,5)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(20,64,5)),
    ('relu2', nn.ReLU()) 로 입력
    """
    def __init__(self, dict_ordered):
        super().__init__() #부모(nn.Module)의 init 생성
        print('__init__()')
        self.dict_ordered = dict_ordered
        model = nn.Sequential(
                        OrderedDict([*[(key, value) for key, value in enumerate(self.dict_ordered)]
                        ]))
        
    def forward(self, data):
        return self.layers(data)
#학습 클래스
class Module_TT():
    """ 
    데이터를 텐서 타입으로 바꾸어서 넣어야함.
    그 중에서도 데이터타입
    
    ## [5-1] 학습 관련 설정들
    EPOCH = 1000                                         # 학습용 DS를 처음부터 끝까지 1번 학습하는 것을 에포크
    BATCH_SIZE = 1200                                      # DS를 학습량 만큼 나눈 사이즈
    ITERATION = int(X_train.shape[0]/BATCH_SIZE)         # 학습용 DS이 분리된 수 => 1에포크에 W, b 업데이트 횟수
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # [5-2] 학습 관련 인스턴스들
    LR      = 0.0001                                # Learning Rate
    MODEL  = Model()                           #학습 모델
    OPTIMIZER = optim.Adam(MODEL.parameters(), lr=LR)      #최적화 즉, 경사하강법 알고리즘으로 W, b의 값 개신
    LOSS_FN = nn.CrossEntropyLoss()
 """
    def __init__(self, model, lr=0.0001, batch_size=100, epoch=100, device=None):
            self.EPOCH = epoch
            self.BATCH_SIZE = batch_size
            
            self.DEVICE = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

            self.LR = lr
            self.MODEL = model.to(self.DEVICE)
            self.OPTIMIZER = optim.Adam(self.MODEL.parameters(), lr=self.LR)
            self.LOSS_FN = nn.CrossEntropyLoss()
            
            
    def training(self, X_train, y_train):
        """ 
        손실계산 부분. 데이터 shape, type 문제 발생 가능성
        """
        
        #학습 모드 설정
        self.MODEL.train()
        self.ITERATION = int(X_train.shape[0] / self.BATCH_SIZE)
        E_LOSS = 0
        for i in range(self.ITERATION):
            start = i*self.BATCH_SIZE
            end = start + self.BATCH_SIZE
            
            #nadrray ==> tensor 변환
            # 처음부터 텐서로 데이터 가르기를 하면 추가 변환이 필요없다.
            # x = torch.FloatTensor(X_train.values[start:end])
            # y = torch.FloatTensor(y_train.values[start:end])
            
            #가중치 기울기 0 초기화
            self.OPTIMIZER.zero_grad()
            
            #학습 진행
            # pre_y = MODEL(x)
            pre_y = self.MODEL(X_train[start:end])
            
            # 손실 계산
            # loss = LOSS_FN(pre_y, y.reshape(-1,1))
            loss = self.LOSS_FN(pre_y, y_train[start:end].reshape(-1))
            
            #역전파 진행
            loss.backward()
            
            #가중치/절편 업데이트
            self.OPTIMIZER.step()
            
            E_LOSS += loss.item()
            
        # print(f"[EPOCH = {epoch}], LOSS {E_LOSS/ITERATION}")
        return loss.item()  
      
    def evaluate(self, X_test, y_test):
        """ 
        손실계산 부분. 데이터 shape, type 문제 발생 가능성
        """
        
        # 에포크 단위로 검증 => 검증 모드
        self.MODEL.eval()
        
        # W, b가 업데이트 해제
        with torch.no_grad():
            # 검증용 데이터셋 => 텐서화
            # ndarray  ==> tensor 변환
            # x = torch.FloatTensor(X_test.values)
            # y = torch.FloatTensor(y_test.values)
            
            #검증진행
            # pre_y = MODEL(x)
            pre_y = self.MODEL(X_test)
            
            # 손실 계산
            # loss = LOSS_FN(pre_y, y.reshape(-1,1))
            loss = self.LOSS_FN(pre_y, y_test.reshape(-1))
            # 1차원 long 타입 요구.
            
            
        return loss.item()  