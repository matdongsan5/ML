""" 
 함수 설계

 1.서브플롯으로 여러개 뽑아보기.
 2.전처리도구 모음
 3.
 
 """
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split

from sklearn.utils.discovery import *
from sklearn.datasets import load_iris
from sklearn.metrics import *

import warnings
import matplotlib.pyplot as plt
import koreanize_matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import math


def pair_grid(data, *, hue=None, vars=None, x_vars=None, y_vars=None, 
            hue_order=None, palette=None, hue_kws=None, corner=False, 
            diag_sharey=True, height=2.5, aspect=1, layout_pad=0.5, 
            despine=True, dropna=False):
    g = sns.PairGrid(data=data, hue=hue, vars=vars, x_vars=x_vars, y_vars=y_vars, 
            hue_order=hue_order, palette=palette, hue_kws=hue_kws, corner=corner, 
            diag_sharey=diag_sharey, height=height, aspect=aspect, layout_pad=layout_pad, 
            despine=despine, dropna=dropna)
    g.map_upper(sns.kdeplot)
    g.map_diag( sns.kdeplot)
    # g.map_diag(lambda data, 
            #    **kwargs: 
            #     sns.histplot(data, kde=False, **{key: value for key, value in kwargs.items() if key != 'label'})
            #    or sns.kdeplot(data, **{key: value for key, value in kwargs.items() if key != 'label'}))

    # g.map_diag( sns.histplot, kde=True)
    g.map_lower(sns.scatterplot)    
    g.add_legend()
    
###############################################################################
###### 데이터 preprocessing 함수 사용
# 2-1) DataFrame을 입력받아 PowerTransformer로 데이터를 정규화 후 반환
def power_transform_dataframe(df):
    """
    df: pandas DataFrame
    """
    # PowerTransformer 객체 생성
    pt = PowerTransformer()
    
    # fit_transform 수행
    transformed_data = pt.fit_transform(df)
    
    # 원본 컬럼명 유지한 DataFrame으로 변환
    df_transformed = pd.DataFrame(transformed_data, columns=df.columns, index=df.index)
    
    return df_transformed

# 2-2) DataFrame을 입력받아 modified Z-score로 이상치 처리
##     modified Z-score는 이상치를 감지하는 용도지만 추가로 이상치를 중앙값으로 대체하는 기능을 넣음
def fill_outliers_with_median_modified_zscore(df, threshold=3.5):
    """
    1) 숫자형 컬럼(each numeric column)에 대해:
    - 중앙값(median) 계산
    - MAD(median absolute deviation) 계산
    - Modified Z-Score(M_i) = 0.6745 * (x_i - median) / MAD
    - abs(M_i) > threshold 인 경우 → “이상치”로 간주
    - 이상치인 해당 셀의 값을 “그 컬럼의 중앙값”으로 대체
    2) 반환: outlier가 처리된 새로운 DataFrame
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_data = df_copy[col]
        
        # 중앙값
        median_val = col_data.median()
        # MAD: median of absolute deviations
        mad_val = np.median(np.abs(col_data - median_val))
        if mad_val == 0:
            # 모든 값이 동일하거나 MAD가 0이면 이상치 식별 불가능
            continue
        
        # Modified Z-Score
        M_i = 0.6745 * (col_data - median_val) / mad_val
        
        # threshold 초과시 이상치로 간주
        outlier_mask = np.abs(M_i) > threshold
        
        # 이상치 → 해당 컬럼의 중앙값으로 대체
        df_copy.loc[outlier_mask, col] = median_val
    
    return df_copy

# 2-3) DataFrame을 입력받아 MinMaxScaler로 데이터를 정규화 후 반환
def MinMaxScaler_dataframe(df):
    """
    df: pandas DataFrame
    """
    # MinMaxScaler 객체 생성
    mm = MinMaxScaler()
    
    # fit_transform 수행
    transformed_data = mm.fit_transform(df)
    
    # 원본 컬럼명 유지한 DataFrame으로 변환
    df_transformed = pd.DataFrame(transformed_data, columns=df.columns, index=df.index)
    
    return df_transformed

# 2-4) DataFrame을 입력받아 RobustScaler로 데이터를 정규화 후 반환
##     modified Z-score와 다르게 이상치를 따로 처리하는 것이 아니라 이상치에 영향을 덜 받으며 데이터 정규화
def RobustScaler_dataframe(df):
    """
    df: pandas DataFrame
    """
    # RobustScaler 객체 생성
    rb = RobustScaler()
    
    # fit_transform 수행
    transformed_data = rb.fit_transform(df)
    
    # 원본 컬럼명 유지한 DataFrame으로 변환
    df_transformed = pd.DataFrame(transformed_data, columns=df.columns, index=df.index)
    
    return df_transformed


########################################


def train_test_cut(feature_df, target_sr, TestSize=0.25, RandomState=5, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(feature_df,
                                                        target_sr,
                                                        test_size=TestSize,
                                                        random_state=RandomState,
                                                        stratify=stratify)
    
    print(f"X_train => {X_train.ndim}D {X_train.shape} / X_test => {X_test.ndim}D, {X_test.shape}")
    print(f"y_train => {y_train.ndim}D {y_train.shape}, / y_test => {y_test.ndim}D, {y_test.shape}")
    return X_train, X_test, y_train, y_test
    

def polyFeature(X_train, X_test, y_train, y_test, deGree=2, interactionTF = False, include_biasTF = True):
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



def find_estimator(X_train, y_train, type_filter):
    # {“classifier”, “regressor”, “cluster”, “transformer”} 

    rets=all_estimators(type_filter=type_filter)
    
    result=[]
    for name, estimator_ in rets:
        try:
            model=estimator_()
            if 'Logistic' in name or 'SGD' in name or 'MLP' in name:
                model.set_params(max_iter=10000)
            if 'SV' in name:
                model.set_params(max_iter=100000, dual='auto')   
    
            model.fit(X_train, y_train)
            sc=model.score(X_train, y_train)
            result.append((name, round(sc, 2)))
        except Exception:
            pass
    return sorted(result, key=lambda x : x[1], reverse=True)