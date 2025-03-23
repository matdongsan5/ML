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
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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

    


    def decision_tree_classification(self, max_depth=None, min_samples_split=2):
        # 의사결정트리 모델 생성
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        
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
    
   

    def rf_classifier(self, 
                    n_splits=5,  # KFold에서 사용할 폴드 개수
                    n_estimators=1000,  # 트리 개수
                    max_depth=8,  
                    min_samples_split=2,  
                    min_samples_leaf=1,  
                    bootstrap=True,  
                    random_state=42,  
                    n_jobs=-1):
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
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
        resultDF = pd.DataFrame(columns = ['train_score', 'test_score', 'diff', 
                                        'train_loss', 'test_loss'])
        resultDF.loc[self.name] = [train_score, test_score, diff, 
                                train_loss, test_loss]
        
        return model, resultDF


    
    
    def cv_classifier(self, model, cv=5):
        """
        분류 모델에 대해 교차 검증을 수행하는 함수

        Parameters:
            model : 학습된 분류 모델
            cv (int): 교차 검증 폴드 수 (기본값: 5)
        
        Returns:
            DF: 평균 성능 지표 (정확도, 정밀도, 재현율, F1-score)
                손실율 계산을 위한 log loss. 
        """
        # 정확도(accuracy) 계산을 위한 커스텀 스코어러
        accuracy_scorer = make_scorer(accuracy_score)
        precision_scorer = make_scorer(precision_score, average='macro')  # 다중 클래스의 경우 'macro' 평균 사용
        recall_scorer = make_scorer(recall_score, average='macro')  # 다중 클래스의 경우 'macro' 평균 사용
        f1_scorer = make_scorer(f1_score, average='macro')  # 다중 클래스의 경우 'macro' 평균 사용

        # log_loss 계산을 위한 커스텀 스코어러
        log_loss_scorer = make_scorer(log_loss, greater_is_better=False)  # 손실 값은 낮을수록 좋기 때문에 greater_is_better=False로 설정

        
        # 정확도, 정밀도, 재현율, F1-score 계산
        train_accuracy_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=accuracy_scorer)
        test_accuracy_scores = cross_val_score(model, self.X_test, self.y_test, cv=cv, scoring=accuracy_scorer)
        
        train_precision_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=precision_scorer)
        test_precision_scores = cross_val_score(model, self.X_test, self.y_test, cv=cv, scoring=precision_scorer)
        
        train_recall_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=recall_scorer)
        test_recall_scores = cross_val_score(model, self.X_test, self.y_test, cv=cv, scoring=recall_scorer)
        
        train_f1_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=f1_scorer)
        test_f1_scores = cross_val_score(model, self.X_test, self.y_test, cv=cv, scoring=f1_scorer)

        # log_loss 계산
        train_log_loss_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=log_loss_scorer)
        test_log_loss_scores = cross_val_score(model, self.X_test, self.y_test, cv=cv, scoring=log_loss_scorer)



        return pd.DataFrame({
            'train_mean_accuracy': np.mean(train_accuracy_scores),
            'train_std_accuracy': np.std(train_accuracy_scores),
            'test_mean_accuracy': np.mean(test_accuracy_scores),
            'test_std_accuracy': np.std(test_accuracy_scores),
            
            'train_mean_precision': np.mean(train_precision_scores),
            'train_std_precision': np.std(train_precision_scores),
            'test_mean_precision': np.mean(test_precision_scores),
            'test_std_precision': np.std(test_precision_scores),
            
            'train_mean_recall': np.mean(train_recall_scores),
            'train_std_recall': np.std(train_recall_scores),
            'test_mean_recall': np.mean(test_recall_scores),
            'test_std_recall': np.std(test_recall_scores),
            
            'train_mean_f1': np.mean(train_f1_scores),
            'train_std_f1': np.std(train_f1_scores),
            'test_mean_f1': np.mean(test_f1_scores),
            'test_std_f1': np.std(test_f1_scores),
            
            'train_mean_log_loss': np.mean(train_log_loss_scores),
            'train_std_log_loss': np.std(train_log_loss_scores),
            'test_mean_log_loss': np.mean(test_log_loss_scores),
            'test_std_log_loss': np.std(test_log_loss_scores),
            
            'diff_accuracy': np.mean(train_accuracy_scores) - np.mean(test_accuracy_scores),
            'diff_precision': np.mean(train_precision_scores) - np.mean(test_precision_scores),
            'diff_recall': np.mean(train_recall_scores) - np.mean(test_recall_scores),
            'diff_f1': np.mean(train_f1_scores) - np.mean(test_f1_scores)
        }, index=[model.__class__.__name__])