""" 
 함수 설계

 1.서브플롯으로 여러개 뽑아보기.
 2.전처리도구 모음
 3.
 
 """
from sklearn.preprocessing import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
 
 
def plot_corr(df, feature_col, target_col):
    # 3개면 13 4개면 22 9개까지 33 그이상 4~
    if len(feature_col)-1 > 9:
        a, b = 4, math.ceil(len(feature_col)-1)/4
    elif len(feature_col)-1 > 4 and len(feature_col)-1 <= 9:
        a, b = 3, 3
    elif len(feature_col)-1 == 4:
        a, b = 2, 2
    elif len(feature_col)-1 <= 3:
        a, b = 1, len(feature_col)
        
    fig, axes = plt.subplots(a, b, figsize=(12,6))
    axes = axes.flatten()
    
    for ax, col in zip(axes, feature_col):
        ax.plot(df[col], df[target_col], 'o')
        ax.set_title(f"{col}, corr {df[target_col].corr(df[col]):.2f}")
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
    
    plt.tight_layout()
    plt.show()
    
    
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