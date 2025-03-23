'''
    클래스 설계
    1. 클래스명 RegressionModule
    2. 용도     회귀 모델을 편하게 적용하기
    3. 구성
        - 선형회귀      LinearREgression
        - 릿지          Ridge
        - 라쏘          Lasso
        - 엘라스틱 넷   ElasticNet    
        - 로지스틱 회귀 LogisticRegression
        
'''
#================================================================
import pandas as pd # 데이터 분석 및 전처리
import numpy as np # 숫자처리
import matplotlib.pyplot as plt # 데이터 시각화

from sklearn.linear_model import LinearRegression ## ML 알고리즘
from sklearn.metrics import *
                            ## 성능평가 모듈
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
                            ## 데이터셋 분리 관련 모듈
                            ## 학습/검증/테스트 
                                                    ## 교차검증\
from sklearn.neighbors import KNeighborsRegressor                                                         
from sklearn.preprocessing import PolynomialFeatures # 폴리. 컬럼추가
from sklearn.linear_model import Ridge, Lasso, ElasticNet                 
from sklearn.ensemble import RandomForestRegressor
#================================================================

class RegressionModule:
    alphaList1 = [0.1, 0.5, 1.0, 1.5, 2, 2.5, 3, 5, 10, 50, 100]        

    def __init__(self, X_train, X_test, y_train, y_test):
        self.name = __name__ 
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def knn_regression(self, n_neighbors=5):
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        
        # 학습
        model.fit(self.X_train, self.y_train)
        
        # 훈련 데이터와 테스트 데이터에서 R^2 (결정 계수) 계산
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)

        # 예측값 계산
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # RMSE 계산
        train_loss = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_loss = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        # 정확도 차이
        diff = train_score - test_score
        
        # 결과 데이터프레임 준비
        resultDF = pd.DataFrame(columns=['train_score', 'test_score', 'diff', 
                                        'train_loss', 'test_loss'])
        resultDF.loc[self.name] = [train_score, test_score, diff, 
                                train_loss, test_loss]
        
        return model, resultDF
            
    def lr(self):
        model = LinearRegression()
        #학습
        model.fit(self.X_train, self.y_train)
        
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)

        train_loss = root_mean_squared_error(self.y_train, model.predict(self.X_train))
        test_loss = root_mean_squared_error(self.y_test, model.predict(self.X_test))
        diff = train_score - test_score
                
        coef = model.coef_
        intercept = model.intercept_
        
        resultDF = pd.DataFrame(columns = 
                                ['train_score', 'test_score','diff', 
                                 'train_loss', 'test_loss', 'coef','intercept'])
        resultDF.loc[self.name] = [train_score, test_score, diff, 
                                   train_loss, test_loss, coef, intercept]
        return model, resultDF


    def ridge_f(self, alphaList=alphaList1):
        resultDF = pd.DataFrame(columns = 
                                ['train_score', 'test_score','diff', 
                                 'train_loss', 'test_loss', 'coef','intercept'])
        for alpha in alphaList:
            model = Ridge()
        #학습
            model.fit(self.X_train, self.y_train)
            
            train_score = model.score(self.X_train, self.y_train)
            test_score = model.score(self.X_test, self.y_test)

            train_loss = root_mean_squared_error(self.y_train, model.predict(self.X_train))
            test_loss = root_mean_squared_error(self.y_test, model.predict(self.X_test))
            diff = train_score - test_score
                    
            coef = model.coef_
            intercept = model.intercept_
            resultDF.loc[alpha] = [train_score, test_score, diff, 
                                   train_loss, test_loss, coef, intercept]
        return model, resultDF
    
    def lasso_f(self, alphaList=alphaList1):
        resultDF = pd.DataFrame(columns = 
                                ['train_score', 'test_score','diff', 
                                 'train_loss', 'test_loss', 'coef','intercept'])
        for alpha in alphaList:
            model = Lasso()
        #학습
            model.fit(self.X_train, self.y_train)
            
            train_score = model.score(self.X_train, self.y_train)
            test_score = model.score(self.X_test, self.y_test)

            train_loss = root_mean_squared_error(self.y_train, model.predict(self.X_train))
            test_loss = root_mean_squared_error(self.y_test, model.predict(self.X_test))
            diff = train_score - test_score
                    
            coef = model.coef_
            intercept = model.intercept_
            resultDF.loc[alpha] = [train_score, test_score, diff, 
                                   train_loss, test_loss, coef, intercept]
        return model, resultDF
    
    def elasticNet_f(self, alphaList=alphaList1):
        resultDF = pd.DataFrame(columns = 
                                ['train_score', 'test_score','diff', 
                                 'train_loss', 'test_loss', 'coef','intercept'])
        for alpha in alphaList:
            model = ElasticNet()
        #학습
            model.fit(self.X_train, self.y_train)
            
            train_score = model.score(self.X_train, self.y_train)
            test_score = model.score(self.X_test, self.y_test)

            train_loss = root_mean_squared_error(self.y_train, model.predict(self.X_train))
            test_loss = root_mean_squared_error(self.y_test, model.predict(self.X_test))
            diff = train_score - test_score
                    
            coef = model.coef_
            intercept = model.intercept_
            resultDF.loc[alpha] = [train_score, test_score, diff, 
                                   train_loss, test_loss, coef, intercept]
        return model, resultDF
        
        
    def cv_regressor(self, model, cv=5):
        """
        Regressor 모델에 대해 교차 검증을 수행하는 함수

        Parameters:
            model : 학습된 Regressor 회귀 모델
            cv (int): 교차 검증 폴드 수 (기본값: 5)
        
        Returns:
            DF: 평균 성능 지표 (R² 점수, RMSE)
        """
        # RMSE(평균 제곱근 오차) 계산을 위한 커스텀 스코어러
        # rmse_scorer = make_scorer(mean_squared_error, squared=False)
        rmse_scorer = make_scorer(mean_squared_error)
        
        # R^2 점수와 RMSE 계산
        train_r2_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='r2')
        test_r2_scores = cross_val_score(model, self.X_test, self.y_test, cv=cv, scoring='r2')
        train_rmse_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=rmse_scorer)
        test_rmse_scores = cross_val_score(model, self.X_test, self.y_test, cv=cv, scoring=rmse_scorer)

        return pd.DataFrame({
            'train_mean_r2': np.mean(train_r2_scores),
            'train_std_r2': np.std(train_r2_scores),
            'test_mean_r2': np.mean(test_r2_scores),
            'test_std_r2': np.std(test_r2_scores),
            'diff': np.mean(train_r2_scores) - np.mean(test_r2_scores),
            'train_mean_rmse': np.mean(train_rmse_scores),
            'train_std_rmse': np.std(train_rmse_scores),
            'test_mean_rmse': np.mean(test_rmse_scores),
            'test_std_rmse': np.std(test_rmse_scores)
        }, index = [model.__class__.__name__])
        
        
    def rf_regressor(self, 
                     n_splits=5,  # KFold에서 사용할 폴드 개수
                    n_estimators=1000,  # 트리 개수
                    max_depth=8,  
                    # max_features='auto',
                    min_samples_split=2,  
                    min_samples_leaf=1,  
                    bootstrap=True,  
                    random_state=42,  
                    n_jobs=-1):
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            # max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs
        )
        #학습
        model.fit(self.X_train, self.y_train)
        
        train_score = model.score(self.X_train, self.y_train)
        test_score = model.score(self.X_test, self.y_test)

        train_loss = root_mean_squared_error(self.y_train, model.predict(self.X_train))
        test_loss = root_mean_squared_error(self.y_test, model.predict(self.X_test))
        diff = train_score - test_score
        
        resultDF = pd.DataFrame(columns = 
                                ['train_score', 'test_score','diff', 
                                 'train_loss', 'test_loss'])
        resultDF.loc[self.name] = [train_score, test_score, diff, 
                                   train_loss, test_loss]
        return model, resultDF
        

    def RLE(self, alphaList=alphaList1):
        resultDF = pd.DataFrame(columns = ['alpha','train_score', 'test_score','diff', 'train_loss', 'test_loss', 'coef','intercept'])
        kf = KFold()
        ## alpha값에 따른 Ridge 모델 성능 비교
        for alpha in alphaList:
            if type == 'rid':
                print('Ridge')
                model = Ridge(alpha)
            elif type == 'las':
                print('Lasso')
                model = Lasso(alpha,max_iter=5000, tol=1e-10)
            elif type == 'ela':
                print('ElasticNet')
                model = ElasticNet(alpha)
            
            train_stotal , test_stotal = 0, 0
            train_ltotal, test_ltotal = 0, 0
                
            for i, (train_index, test_index) in enumerate(kf.split(self.X_train, self.y_train)):
                # print(self.X_train, self.y_train)
                ## 학습용 / 테스트용 피쳐와 타겟 추출
                train_data, train_label = self.X_train[train_index], self.y_train.iloc[train_index]
                test_data, test_label = self.X_train[test_index], self.y_train.iloc[test_index]

                #학습
                # print(train_data.shape,train_data.ndim,'D','/', len(train_label))
                model.fit(train_data, train_label)
                
                train_score = model.score(train_data, train_label)
                test_score = model.score(test_data, test_label)

                train_loss = root_mean_squared_error(train_label, model.predict(train_data))
                test_loss = root_mean_squared_error(test_label, model.predict(test_data))

                coef = model.coef_
                intercept = model.intercept_
                train_stotal += train_score
                test_stotal += test_score
                train_ltotal += train_loss
                test_ltotal += test_loss
            #alpha값 별로 성능과 손실값 평균 저장하기    
            resultDF.loc[alpha] = [alpha, train_stotal/5,test_stotal/5,train_stotal/5-test_stotal/5,train_ltotal/5,test_ltotal/5, coef.round(4), intercept]
        print(resultDF)
        return model

    #===============
    def fives(resultDF, title='None'):
        fig, axe = plt.subplots(1,5, figsize=(12,6), sharex=True)
        axe = axe.flatten()
        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 5, 20)]
        for ax, col, color1 in zip(axe, resultDF.columns[1:], colors):
            ax.plot(resultDF['alpha'], resultDF[col], color = color1, label=col)
            ax.legend()
            ax.set_title(col)
        fig.suptitle(title)