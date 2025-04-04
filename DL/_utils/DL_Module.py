""" 
    1. class Common_Dataset
        공용 DS 모델
    
    2. class MyModel
        동적 은닉층 모델(linear only)
    
    3. class TT_classifier
        training()  train
        evaluate()  test용 함수.
    
    4. class TT_regressior
        training()  train
        evaluate()  test용 함수.
"""
import torch                                            ## Tensor 및 기본 함수들 관련 모듈들
import torch.nn as nn                                   ## 인공신경망 관련 모듈들
import torch.nn.functional as F                         ## 인공신경망 관련 함수들
import torch.optim as optim                             ## 최적화 모듈
from torch.utils.data import Dataset, DataLoader 

from torchmetrics.regression import *
from torchmetrics.classification import *
import torchinfo

from sklearn.model_selection import train_test_split    ## 학습용 데이터셋 관련 함수

from torch.optim.lr_scheduler import ReduceLROnPlateau 

import matplotlib.pyplot as plt

import time
## 범용 데이터셋 클래스 
class Common_Dataset(Dataset):
    # 피쳐와 타겟 분리 및 전처리 진행 
    def __init__(self, featureDF, targetSR):
        super().__init__()
        self.feature = featureDF
        self.target  = targetSR
        self.rows = featureDF.shape[0]
        self.cols = featureDF.shape[1]
    
    # 데이터셋의 샘플 수 반환 메서드 
    def __len__(self):
        return self.rows 

    # DataLoader에서 batch_size만큼 호출하는 메서드
    # 인덱스에 해당하는 피쳐와 타겟 반환 단, Tensor 형태
    def __getitem__(self, index):
       arrFeature = self.feature.iloc[index].values   # ndarray
       arrTarget = self.target[index].reshape(-1)     # ndarray
   
       return torch.FloatTensor(arrFeature), torch.Tensor(arrTarget)
# 은닉층 개수 동적인 모델 ---------------------------------------------------------------------
class MyModel(nn.Module):
    def __init__(self, in_in, out_out, h_in, h_list=[]):
        """ 
        in_in : 입력층의 입력. 피처 개수
        h_in : 입력층 출력. 최초 히든입력.
        out_out : 최종 출력값.
        h_list = [] 리스트
        """
        # 부모클래스 생성
        super().__init__()
        # 자식클래스의 인스턴스 속성 설정
        self.input_layer = nn.Linear(in_in, h_in)
        
        self.h1_layer = nn.ModuleDict() 
        for idx in range(len(h_list)):
            h_in = h_list[idx-1] if idx else h_in
            h_out = h_list[idx]
            self.h1_layer[f"hl_{str(idx)}"] = nn.Linear(h_in,h_out)
            
        self.output_layer = nn.Linear(h_out, out_out)
        
    def forward(self, x):
        y=F.relu(self.input_layer(x))
    
        for linear in self.h1_layer.values():
            y=F.relu(linear(y))
            
        return self.output_layer(y)
class TT_classifier():
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
    def __init__(self, model, trainDS, testDS, lr=0.001, batch_size=100, epoch=100, LOSS_FN=nn.CrossEntropyLoss(), device=None):
            self.trainDS = trainDS
            self.testDS = testDS
            self.TRAINDL   = DataLoader(trainDS, batch_size=batch_size, shuffle=True) ## 학습용 데이터로더
            self.TESTDL    = DataLoader(testDS,  batch_size=batch_size, shuffle=True) ## 테스트용 데이터로더
            self.EPOCHS = epoch
            self.BATCH_SIZE = batch_size
            self.ITERATION = int(len(trainDS)/batch_size)
            self.DEVICE = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.MODEL = model
            self.MODEL.to(self.DEVICE)
            self.LR = lr
            
            self.OPTIMIZER = optim.Adam(self.MODEL.parameters(), lr=self.LR)
            self.LOSS_FN = LOSS_FN
            print(f"EPOCHS : {self.EPOCHS}")
            print(f"ITERATION : {self.ITERATION}")
            print(f"LOSS_FN : {self.LOSS_FN}")
            
            
            self.HIST ={'Train':[], 'Valid':[]}
            
            
            
    def training(self):
    # 학습 모드 설정
        self.MODEL.train()

        E_LOSS, E_ACC = 0, 0
        for feature, target in self.TRAINDL:
            # 배치크기만큼 feature, target로딩
            #print('로딩 데이터 :', feature.shape, target.shape)

            # 가중치 기울기 0 초기화
            self.OPTIMIZER.zero_grad()

            # 학습 진행
            pre_y = self.MODEL(feature)
            # target = torch.Tensor(target).long()
            # 손실 계산
            loss = self.LOSS_FN(pre_y, target.reshape(-1,1).float())

            # 정확도 계산
            # accuracy = MulticlassAccuracy(num_classes=10)
            accuracy = BinaryAccuracy()
            acc = accuracy(pre_y, target.reshape(-1,1))

            # 역전파 진행
            loss.backward()

            # 가중치/절편 업데이트
            self.OPTIMIZER.step()

            E_LOSS += loss.item()
            E_ACC  += acc.item()

        return E_LOSS/self.ITERATION, E_ACC/self.ITERATION
      
    def evaluate(self):
        # 에포크 단위로 검증 => 검증 모드
        self.MODEL.eval()
        
        # W, b가 업데이트 해제
        with torch.no_grad():
            # 검증용 데이터셋 => 텐서화 ndarray ==> tensor변환
            # x = torch.FloatTensor(X_test.values) 
            # y = torch.Tensor(y_test.values)
            x = torch.empty(0)  # 빈 텐서 초기화
            y = torch.empty(0)

            for a, b in self.TESTDL:
                x = torch.cat((x, a), dim=0)  # 입력 데이터 (a) 추가
                y = torch.cat((y, torch.Tensor(b)), dim=0)  # 라벨 데이터 차원 변경 후 추가

                        
            # 검증진행
            print(x.shape)
            print(y.shape)
            pre_y= self.MODEL(x)
            # y = torch.Tensor(y).long()
            # 손실 계산
            loss = self.LOSS_FN(pre_y, y.reshape(-1,1).float())

            # 정확도 계산
            accuracy = BinaryAccuracy()
            acc = accuracy(pre_y, y.reshape(-1,1))

            return loss.item(), acc.item()
    
    
    def cycling(self,patience=3, mode='min'):
        # 에포크 : DS 처음부터 ~ 끝까지 학습  

        # 학습 스케쥴러 생성
        lrScheduler = ReduceLROnPlateau(self.OPTIMIZER, patience=patience, mode=mode)

        #조기 종료 카운팅 ==> 모델 성능 개선없이 불필요한 학습 막기 위해서.
        E_STOP_CNT = 5

        # 에포크 단위 학습/검증 진행 
        for epoch in range(self.EPOCHS):
            a = time.time()
            trainLoss, trainAcc = self.training()
            validLoss, validAcc = self.evaluate()

            self.HIST['Train'].append((trainLoss, trainAcc))
            self.HIST['Valid'].append((validLoss, validAcc))
            

            print(f'\nEPOCH[{epoch}/{self.EPOCHS}]----------------')
            print(f'- TRAIN_LOSS {trainLoss:.5f}  ACC {trainAcc:.5f}')
            print(f'- VALID_LOSS {validLoss:.5f}  ACC {validAcc:.5f}')
            
            
            ## 모델 저장부분
            MODEL_PATH = './models/'
            MODEL_FILE = 'catdog_model.pt'

            MAX_ACC  = 0.8

            #모델 저장 기준
            if MAX_ACC < validAcc :
                # torch.save(MODEL, MODEL_PATH+MODEL_FILE)
                torch.save(self.MODEL, f'{MODEL_PATH}cd_model_epoch_{epoch:02d}')
                MAX_ACC = validAcc
                
                
            lrScheduler.step(validLoss)
            print(f"{time.time()-a:.2f}초")
            
            ## num_bad_epochs
            ## 성능개선이 안될 시. 카운트.
            print(f'[{epoch}] - num_bad_epochs : {lrScheduler.num_bad_epochs} ')
            if lrScheduler.num_bad_epochs >= lrScheduler.patience:
                print('Early Stopping')
                E_STOP_CNT -= 1
            
            if not E_STOP_CNT:
                print(f'{epoch}까지 학습 후 성능 개선이 없어서 조기 종료합니다.')
                break
        print(time.strftime("%H:%M:%S")) 
        return self.HIST
            
            
    def draw_graph(self):
        # MODEL.eval()
        # xdata = range(self.EPOCH)
        ydata1Loss = []
        ydata1Acc = []
        for loss, acc in self.HIST['Train']:
            ydata1Loss.append(loss)
            ydata1Acc.append(acc)
        ydata2Loss = []
        ydata2Acc = []
        for loss, acc in self.HIST['Valid']:
            ydata2Loss.append(loss)
            ydata2Acc.append(acc)

        fig, ax = plt.subplots(1,2)
        ax = ax.flatten()
        ax[0].plot(ydata1Loss, 'b^--', label='TrainLOSS')
        ax[1].plot(ydata1Acc, 'b^--', label='TrainACC')

        ax[0].plot(ydata2Loss, 'r^--', label='ValidLOSS')
        ax[1].plot(ydata2Acc, 'r^--', label='ValidACC')


        ax[0].grid()
        ax[1].grid()
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel('EPOCHS')
        ax[1].set_xlabel('EPOCHS')

        ax[0].set_ylabel('LOSS')
        ax[1].set_ylabel('ACC')
        plt.tight_layout()
        plt.show()


class TT_regressor():
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
    def __init__(self, model, trainDS, testDS, lr=0.001, batch_size=100, epoch=100, device=None):
            self.trainDS = trainDS
            self.testDS = testDS
            self.TRAINDL   = DataLoader(trainDS, batch_size=batch_size) ## 학습용 데이터로더
            self.TESTDL    = DataLoader(testDS,  batch_size=batch_size) ## 테스트용 데이터로더
            self.EPOCH = epoch
            self.BATCH_SIZE = batch_size
            self.ITERATION = int(len(trainDS)/batch_size)
            self.DEVICE = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

            self.LR = lr
            self.MODEL = model.to(self.DEVICE)
            self.OPTIMIZER = optim.Adam(self.MODEL.parameters(), lr=self.LR)
            self.LOSS_FN = nn.CrossEntropyLoss()
            self.HIST ={'Train':[], 'Valid':[]}  
            
    def training(self):
    # 학습 모드 설정
        self.MODEL.train()

        E_LOSS = 0
        E_MSE = 0
        E_MAE = 0
        E_R2 = 0
        for feature, target in self.TRAINDL:
            # 배치크기만큼 feature, target로딩
            #print('로딩 데이터 :', feature.shape, target.shape)

            # 가중치 기울기 0 초기화
            self.OPTIMIZER.zero_grad()

            # 학습 진행
            pre_y = self.MODEL(feature)

            # 손실 계산
            loss = self.LOSS_FN(pre_y, target.reshape(-1).long())

            # 지표 계산
            MSE = MeanSquaredError()
            mse = MSE(pre_y, target.reshape(-1,1))
            MAE = MeanAbsoluteError()
            mae = MAE(pre_y, target.reshape(-1,1))
            R2 = R2Score()
            r2 = R2(pre_y, target.reshape(-1,1))
    
            # 역전파 진행
            loss.backward()

            # 가중치/절편 업데이트
            self.OPTIMIZER.step()

            E_R2 += r2.item()
            E_MAE += mae.item()
            E_MSE += mse.item()
            E_LOSS += loss.item()

        return E_LOSS/self.ITERATION, E_MSE/self.ITERATION, E_MAE/self.ITERATION, E_R2/self.ITERATION
      
    def evaluate(self):
        # 에포크 단위로 검증 => 검증 모드
        self.MODEL.eval()
        
        # W, b가 업데이트 해제
        with torch.no_grad():
            E_LOSS = 0
            E_MSE = 0
            E_MAE = 0
            E_R2 = 0
            CNT = 0
            # 검증용 데이터셋 => 텐서화 ndarray ==> tensor변환
            # x = torch.FloatTensor(X_test.values) 
            # y = torch.Tensor(y_test.values)
            x= torch.tensor(4,1)
            y= torch.tensor()
            for a, b in self.testDS:
                x = torch.cat((x, a), dim=0)
                y = torch.cat((x, b), dim=0)
                        
            # 검증진행
            pre_y= self.MODEL(x)
            
            # 손실 계산
            loss = self.LOSS_FN(pre_y, y.reshape(-1).long())

            # 정확도 계산
            MSE = MeanSquaredError()
            mse = MSE(pre_y, y.reshape(-1,1))
            MAE = MeanAbsoluteError()
            mae = MAE(pre_y, y.reshape(-1,1))
            R2 = R2Score()
            r2 = R2(pre_y, y.reshape(-1,1))

            E_R2 += r2.item()
            E_MAE += mae.item()
            E_MSE += mse.item()
            E_LOSS += loss.item()
            E_LOSS += loss.item()
            CNT += 1
            return E_LOSS/self.ITERATION, E_MSE/self.ITERATION, E_MAE/self.ITERATION, E_R2/self.ITERATION
        
        
    def cycling(self,patience=3, mode='min', E_STOP_CNT=5):
        # 에포크 : DS 처음부터 ~ 끝까지 학습  

        # 학습 스케쥴러 생성
        lrScheduler = ReduceLROnPlateau(self.OPTIMIZER, patience=patience, mode=mode)

        #조기 종료 카운팅 ==> 모델 성능 개선없이 불필요한 학습 막기 위해서.
        E_STOP_CNT = E_STOP_CNT

        # 에포크 단위 학습/검증 진행 
        for epoch in range(self.EPOCHS):
            a = time.time()
            trainLoss, trainMse, trainMae, trainR2 = self.training()
            validLoss, validMse, validMae, validR2 = self.evaluate()

            self.HIST['Train'].append((trainLoss, trainMse, trainMae, trainR2 ))
            self.HIST['Valid'].append((validLoss, validMse, validMae, validR2))

            print(f'\nEPOCH[{epoch}/{self.EPOCHS}]----------------')
            print(f'- TRAIN_LOSS {trainLoss:.5f}    Mse {trainMse:.5f}   MAE {trainMae:.5f}  R2{trainR2:.5f}')
            print(f'- VALID_LOSS {validLoss:.5f}    Mse {validMse:.5f}   MAE {validMae:.5f}  R2{validR2:.5f}')
            
            
            ## 모델 저장부분
            MODEL_PATH = './models/'
            MODEL_FILE = 'catdog_model.pt'

            MIN_VLOSS  = 100000.

            #모델 저장 기준
            if MIN_VLOSS > validLoss :
                # torch.save(MODEL, MODEL_PATH+MODEL_FILE)
                torch.save(self.MODEL, f'{MODEL_PATH}cd_model_epoch_{epoch:02d}')
                MIN_VLOSS = validLoss
          
            
            print(time.time()-a)
        
            lrScheduler.step(validLoss)

            
            ## num_bad_epochs
            ## 성능개선이 안될 시. 카운트.
            print(f'[{epoch}] - num_bad_epochs : {lrScheduler.num_bad_epochs} ')
            if lrScheduler.num_bad_epochs >= lrScheduler.patience:
                print('Early Stopping')
                E_STOP_CNT -= 1
            
            if not E_STOP_CNT:
                print(f'{epoch}까지 학습 후 성능 개선이 없어서 조기 종료합니다.')
                break
        print(time.strftime("%H:%M:%S")) 
        return self.HIST
            
    def draw_graph(self):
        # xdata = range(self.EPOCH)
        ydata1Loss = []
        ydata1MSE = []
        ydata1MAE = []
        ydata1R2 = []

        for loss, Mse, Mae, R2 in self.HIST['Train']:
            ydata1Loss.append(loss)
            ydata1MSE.append(Mse)
            ydata1MAE.append(Mae)
            ydata1R2.append(R2)

        ydata2Loss = []
        ydata2MSE = []
        ydata2MAE = []
        ydata2R2 = []

        for loss, Mse, Mae, R2 in self.HIST['Valid']:
            ydata2Loss.append(loss)
            ydata2MSE.append(Mse)
            ydata2MAE.append(Mae)
            ydata2R2.append(R2)
            
        fig, ax = plt.subplots(1,4, figsize=(12,6))
        ax = ax.flatten()
        ax[0].plot(ydata1Loss, 'r^--', label='TrainLOSS')
        ax[0].plot(ydata2Loss, 'b^--', label='ValidLOSS')

        ax[1].plot(ydata1MSE, 'r^--', label='TrainMSE')
        ax[1].plot(ydata2MSE, 'b^--', label='ValidMSE')

        ax[2].plot(ydata1MAE, 'r^--', label='TrainMAE')
        ax[2].plot(ydata2MAE, 'b^--', label='ValidMAE')

        ax[3].plot(ydata1R2, 'r--', label='TrainR2', alpha=0.5)
        ax[3].plot(ydata2R2, 'b--', label='ValidR2', alpha=0.5)

        ax[0].grid();ax[1].grid();ax[2].grid();ax[3].grid()
        ax[0].legend();ax[1].legend();ax[2].legend();ax[3].legend()

        ax[0].set_xlabel('EPOCHS')
        ax[1].set_xlabel('EPOCHS')

        ax[0].set_ylabel('LOSS')
        ax[1].set_ylabel('MSE')
        ax[2].set_ylabel('MAE')
        ax[3].set_ylabel('R2')
        plt.tight_layout()
        plt.show()
    