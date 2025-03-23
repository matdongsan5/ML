'''
    클래스 설계
    1. 클래스명 ClassfierModule
    2. 용도     회귀 모델을 편하게 적용하기
    3. 구성
        1. knn분류
        2. 로지스틱 회귀
        3. 랜덤포레스트.분류
        4. 의사결정나무.분류
        5.
        
'''
#==============================================================
import pandas as pd # 데이터 분석 및 전처리
import numpy as np # 숫자처리
import matplotlib.pyplot as plt # 데이터 시각화

from sklearn.metrics import *
                            ## 성능평가 모듈
                            
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
                            ## 데이터셋 분리 관련 모듈
                            ## 학습/검증/테스트 
                                                    ## 교차검증\
from sklearn.linear_model import LogisticRegression                                                        
from sklearn.neighbors import KNeighborsClassifier                                                    
from sklearn.preprocessing import PolynomialFeatures # 폴리. 컬럼추가
from sklearn.ensemble import RandomForestRegressor

#==============================================================

class ClassfierModule:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.name = __name__ 
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def knn_classification(self, n_neighbors=5):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # 학습
        model.fit(self.X_train, self.y_train)
        
        # 훈련 데이터와 테스트 데이터의 정확도 계산
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)

        # 예측을 기반으로 분류 성능 평가 (정확도)
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        train_loss = 1 - accuracy_score(self.y_train, y_train_pred)
        test_loss = 1 - accuracy_score(self.y_test, y_test_pred)
        
        # 정확도 차이
        diff = train_score - test_score
        
        # 결과 데이터프레임 준비
        resultDF = pd.DataFrame(columns=['train_score', 'test_score', 'diff', 
                                        'train_loss', 'test_loss', 'classification_report'])
        resultDF.loc[self.name] = [train_score, test_score, diff, 
                                train_loss, test_loss, classification_report(self.y_test, y_test_pred)]
        
        return model, resultDF
    
 
def logistic_regression(self, max_iter=100):
    model = LogisticRegression(max_iter=max_iter)
    
    # 학습
    model.fit(self.X_train, self.y_train)
    
    # 훈련 데이터와 테스트 데이터에서 정확도 계산
    train_score = model.score(self.X_train, self.y_train)
    test_score = model.score(self.X_test, self.y_test)

    # 예측값 계산
    y_train_pred = model.predict(self.X_train)
    y_test_pred = model.predict(self.X_test)
    
    # 정확도 차이 계산
    diff = train_score - test_score
    
    # 평가 지표
    train_accuracy = accuracy_score(self.y_train, y_train_pred)
    test_accuracy = accuracy_score(self.y_test, y_test_pred)
    
    # F1-Score 등 더 상세한 평가 지표
    train_classification_report = classification_report(self.y_train, y_train_pred)
    test_classification_report = classification_report(self.y_test, y_test_pred)
    
    # 혼동 행렬
    train_confusion_matrix = confusion_matrix(self.y_train, y_train_pred)
    test_confusion_matrix = confusion_matrix(self.y_test, y_test_pred)
    
    # 결과 데이터프레임 준비
    resultDF = pd.DataFrame(columns=['train_score', 'test_score', 'diff', 
                                     'train_accuracy', 'test_accuracy', 
                                     'train_classification_report', 'test_classification_report',
                                     'train_confusion_matrix', 'test_confusion_matrix'])
    resultDF.loc[self.name] = [train_score, test_score, diff, 
                               train_accuracy, test_accuracy, 
                               train_classification_report, test_classification_report,
                               train_confusion_matrix, test_confusion_matrix]
    
    return model, resultDF
