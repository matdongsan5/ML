'''
    클래스 설계
    1. 클래스명 RegressionModule
    2. 용도     회귀 모델을 편하게 적용하기
    3. 구성
        - kfold 선형회귀 추가?하나
        - 선형회귀      LinearREgression
        - 다항     PolyFeature
        - 릿지          Ridge
        - 라쏘          Lasso
        - 엘라스틱 넷   ElasticNet    
        
        x트레인을 공용변수로 넣을까?
        분류에 따로 넣기?
        - 로지스틱 회귀 LogisticRegression
        
'''
#================================================================
import pandas as pd # 데이터 분석 및 전처리
import numpy as np # 숫자처리
import matplotlib.pyplot as plt # 데이터 시각화
from sklearn.linear_model import LinearRegression ## ML 알고리즘
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
                            ## 성능평가 모듈
from sklearn.model_selection import train_test_split, KFold
                            ## 데이터셋 분리 관련 모듈
                            ## 학습/검증/테스트 
                                                    ## 교차검증\
from sklearn.preprocessing import PolynomialFeatures # 폴리. 컬럼추가
from sklearn.linear_model import Ridge, Lasso, ElasticNet                                                        
#================================================================

class RegressionModule:
    alphaList1 = [0.1, 0.5, 1.0, 1.5, 2, 2.5, 3, 5, 10, 50, 100]
    
    def __init__(self):
        self.name = __name__ 
        X_train = np.array()
        X_test = np.array()
        y_train = np.array()
        y_test = np.array()

        
        pass
        
    def train_test_cut(self, feature_df, target_sr, TestSize=0.25, RandomState=5):
        # print(f"featureDF => {featureDF.ndim}D, targetSr => {targetSR.ndim}D")
        ## 학습용 : 테스트용 = 9:1
        self.feature_df = feature_df
        self.target_sr = target_sr
        self.TestSize = TestSize
        self.RandomState = RandomState
        
        X_train, X_test, y_train, y_test = train_test_split(self.feature_df,
                                                            self.target_sr,
                                                            test_size=self.TestSize,
                                                            random_state=self.RandomState)
        print(f"X_train => {X_train.ndim}D {X_train.shape} / X_test => {X_test.ndim}D, {X_test.shape}")
        print(f"y_train => {y_train.ndim}D {y_train.shape}, / y_test => {y_test.ndim}D, {y_test.shape}")
        return X_train, X_test, y_train, y_test
     
    
    def polyFeature(self, X_train, X_test, y_train, y_test, deGree=2, interactionTF = False, include_biasTF = True):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        pl = PolynomialFeatures(degree=deGree, 
                            interaction_only = interactionTF,
                            include_bias = include_biasTF,
                            order= "C")

        pl.fit(X_train)
        X_Ptrain = pl.transform(X_train)
        X_Ptest  = pl.transform(X_test)
        
        print(f"X_train => {X_Ptrain.ndim}D {X_Ptrain.shape} / X_test => {X_Ptest.ndim}D, {X_Ptest.shape}")
        print(f"y_train => {y_train.ndim}D {y_train.shape}, / y_test => {y_test.ndim}D, {y_test.shape}")
        
        return X_Ptrain, X_Ptest, y_train, y_test
    
    
    def lr(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        train_stotal , test_stotal = 0, 0
        train_ltotal, test_ltotal = 0, 0
        
        resultDF = pd.DataFrame(columns = ['train_score', 'test_score','diff', 'train_loss', 'test_loss', 'coef','intercept'])

        model = LinearRegression()
        
        train_data, train_label = X_train, y_train      
        test_data, test_label = X_train, y_train
        #학습
        
        print(train_data.shape,train_data.ndim,'D','/', len(train_label))
        
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
        
        resultDF.loc[self.name] = [train_stotal/5,test_stotal/5,train_stotal/5-test_stotal/5,train_ltotal/5,test_ltotal/5, coef.round(4), intercept]
        return resultDF

    def RLE(self, X_train, X_test, y_train, y_test, type, alphaList=alphaList1):
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
                
            for i, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
                # print(X_train, y_train)
                ## 학습용 / 테스트용 피쳐와 타겟 추출
                train_data, train_label = X_train[train_index], y_train.iloc[train_index]
                test_data, test_label = X_train[test_index], y_train.iloc[test_index]

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