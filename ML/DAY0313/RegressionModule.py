# 모듈화하기
# Boston price 예측모델
# - 데이터셋 : boston.csv
# - 학습종류: 지도학습 / 회귀
# - 알고리즘 : Ridge, Lasso, Elastine
# - 피처: 모두.
# - 타겟: medv   
## 1-1
import pandas as pd
# 데이터 분석 및 전처리
import numpy as np
# 숫자처리
import matplotlib.pyplot as plt
# 데이터 시각화

from sklearn.linear_model import LinearRegression
                            ## ML 알고리즘
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
                            ## 성능평가 모듈
from sklearn.model_selection import train_test_split, KFold
                            ## 데이터셋 분리 관련 모듈
                            ## 학습/검증/테스트 
                                                    ## 교차검증\
from sklearn.preprocessing import PolynomialFeatures
                                # 폴리. 컬럼추가
from sklearn.linear_model import Ridge, Lasso, ElasticNet                                                        


alphaList = [0.1, 0.5, 1.0, 1.5, 2, 2.5, 3, 5, 10, 50, 100]
def ft_choice(feature_df, target_sr, type, alphaList=alphaList):
    print(feature_df.columns)
    featureDF = feature_df
    targetSR = target_sr

    # print(f"featureDF => {featureDF.ndim}D, targetSr => {targetSR.ndim}D")
    ## 학습용 : 테스트용 = 9:1
    X_train, X_test, y_train, y_test = train_test_split(featureDF,
                                                        targetSR,
                                                        random_state=5)
    print(f"X_train => {X_train.ndim}D {X_train.shape} / X_test => {X_test.ndim}D, {X_test.shape}")
    print(f"y_train => {y_train.ndim}D {y_train.shape}, / y_test => {y_test.ndim}D, {y_test.shape}")
    return RLE(type, alphaList,X_train, X_test, y_train, y_test)
#aglt = Ridge(alpha), Lasso(alpha), ElasticNet()
#alphaList = [0.1, 0.5, 1.0, 1.5, 2, 2.5, 3]

def RLE(type, alphaList,X_train, X_test, y_train, y_test):
    if type=='poly':
        pl = PolynomialFeatures()
        pl.fit(X_train,y_train)
        X_train = pd.DataFrame(pl.transform(X_train))
        X_test  = pd.DataFrame(pl.transform(X_test))
        print('poly 적용')
    else:
        pass
    resultDF = pd.DataFrame(columns = ['alpha','train_score', 'test_score','diff', 'train_loss', 'test_loss', 'coef','intercept'])
    kf = KFold()
    ## alpha값에 따른 Ridge 모델 성능 비교
    for alpha in alphaList:
        if type == 'lr':
            # print('Ridge')
            model = LinearRegression()
        if type == 'poly':
            # print('Ridge')
            
            model = LinearRegression()
        elif type == 'rid':
            # print('Ridge')
            model = Ridge(alpha)
        elif type == 'las':
            # print('Lasso')
            model = Lasso(alpha,max_iter=5000, tol=1e-10)
        elif type == 'ela':
            # print('ElasticNet')
            model = ElasticNet(alpha)
                        
        
        train_stotal , test_stotal = 0, 0
        train_ltotal, test_ltotal = 0, 0
        
            
        for i, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
            # print(X_train.info(), y_train.info())
            #        X -> DF    y- > sr
            
            ## 학습용 / 테스트용 피쳐와 타겟 추출
            train_data, train_label = X_train.iloc[train_index, :], y_train.iloc[train_index]
            test_data, test_label = X_train.iloc[test_index, :], y_train.iloc[test_index]

            #학습
            
            print(train_data.shape,train_data.ndim,'D','/', len(train_label))
            
            # if len(X_train.columns) == 1:
            #     train_data = pd.DataFrame(train_data)
            
            model.fit(train_data, train_label)
            
            train_score = model.score(train_data, train_label)
            test_score = model.score(test_data, test_label)

            train_loss = root_mean_squared_error(train_label, model.predict(train_data))
            test_loss = root_mean_squared_error(test_label, model.predict(test_data))

            # if type =='poly':
            #     coef = model.coef_
            # else:
            coef = model.coef_
            intercept = model.intercept_
            
            train_stotal += train_score
            test_stotal += test_score
            train_ltotal += train_loss
            test_ltotal += test_loss
        #alpha값 별로 성능과 손실값 평균 저장하기    
        resultDF.loc[alpha] = [alpha, train_stotal/5,test_stotal/5,train_stotal/5-test_stotal/5,train_ltotal/5,test_ltotal/5, coef.round(4), intercept]
    
    print(resultDF)
    fives(resultDF, type)
    
def fives(resultDF, title='None'):
    fig, axe = plt.subplots(1,5, figsize=(12,6), sharex=True)
    axe = axe.flatten()
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 5, 20)]
    for ax, col, color1 in zip(axe, resultDF.columns[1:], colors):
        ax.plot(resultDF['alpha'], resultDF[col], color = color1, label=col)
        ax.legend()

    fig.suptitle(title)
