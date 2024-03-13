#!/usr/bin/env python
# coding: utf-8

# ### 📝 회귀 성능 지표
# ##### MSE
# - 예측값과 실제값의 차이를 제곱한 값을 모두 더하여 평균을 한 값이다.
# - 제곱을 하기 때문에 이상치 때문에 차이가 크게 나타나면 크기가 많이 늘어난다(이상치에 민감하다).
# 
# ##### MAE
# - 모델의 예측값과 실제값의 차이를 절댓값을 취해 모두 더하여 평균을 한 값이다. 
# - 절대값을 취하기 때문에 실제값보다 모델이 높은 값인지 적은 값인지는 알 수 없고, 차이의 크기만 알 수 있다.
# 
# ##### RSME
# - MSE에 루트를 씌운 값이다.
# - 오류지표를 최대한 실제값과 비슷하게 만들어서 이상치에 예민한 부분을 제거하며, 해석을 용이하게 한다.
# 
# ##### R2 score(R-sqared)
# - 분산 기반으로 예측 성능을 평가하기 때문에, 실제값의 분산이 높을 경우 예측은 어려워지며, 실제값의 분산이 낮을 수록 예측은 쉬워진다.
# - 0부터 1사이의 값을 나타내기 때문에 스케일의 영향 없이 r2의 값만 보고 성능을 판단할 수 있는 장점을 가진다.
# - 가중치가 늘어나거나 데이터의 개수가 많아지면 r2의 값도 같이 늘어나기 때문에 r2 score 하나만으로는 정확한 성능 지표가 되기 어렵다.  
# - R² = 예측값 Variance / 실제값 Variance
#   
# <img src="./images/evaluation02.png" style="margin: 20px; margin-left: -10px">
# 
# ##### RMSLE
# - 로그로 변환하기 때문에 큰 폭의 이상치에 강건하다(영향을 덜 받는다).
# - 상대적 Error를 측정해준다.
# > 예측값 = 100, 실제값 = 90일 때, <strong>RMSLE = 0.1053</strong>, RMSE = 10  
# > 예측값 = 10,000, 실제값 = 9,000일 때, <strong>RMSLE = 0.1053</strong>, RMSE = 1,000  
# 
# -  Under Estimation(예측값이 실제값보다 작을 때)에 큰 패널티를 부여한다.
# > 예측값 = 600, 실제값 = 1,000일 때 RMSE = 400, RMSLE = 0.510  
# > 예측값 = 1,400, 실제값 = 1,000일 때 RMSE = 400, RMSLE = 0.33  
# 🚩 <strong>작업 완료까지 30분으로 예측하였으나 20분이 걸리면 문제가 없지만, 40분이 걸리면 문제이므로 이런 경우에는 RMSLE를 사용한다.</strong>
# 
# <img src="./images/evaluation01.png" width="300" style="margin: 20px; margin-left: -10px">
# 
# ---
# ##### 🚩 회귀 계수에 따라 오차율의 면적을 구하는 다양한 방법을 통해 성능을 평가할 수 있어야 한다.
# <img src="./images/evaluation03.gif" width="400" style="margin: 20px; margin-left: -10px">

# ##### 1인당 건강 보험 비용
# - age: 1차 수혜자의 연령.
# - sex: 보험계약자의 성별(여성 또는 남성).
# - bmi: 체질량지수, 키 대비 체중을 측정하는 척도.
# - children: 건강보험의 적용을 받는 자녀의 수 또는 부양가족의 수.
# - smoker: 흡연 상태(흡연자 또는 비흡연자).
# - region: 수혜자의 미국 내 거주지역(북동, 남동, 남서, 북서).
# - charges: 건강보험에서 청구하는 개인별 의료비.

# In[1]:


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error

def get_evaluation(y_test, prediction):
    MAE =  mean_absolute_error(y_test, prediction)
    MSE = mean_squared_error(y_test, prediction)
    RMSE = np.sqrt(MSE)
    MSLE = mean_squared_log_error(y_test, prediction)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, prediction))
    R2 = r2_score(y_test, prediction)

    print('MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, MSLE: {:.4f}, RMSLE: {:.4f}, R2: {:.4f}'.format(MAE, MSE, RMSE, MSLE, RMSLE, R2))


# In[2]:


import pandas as pd
medical_cost_df = pd.read_csv('./datasets/medical_cost.csv')
medical_cost_df


# In[3]:


from sklearn.preprocessing import LabelEncoder

columns = ['sex', 'smoker', 'region']
encoders = []
for column in columns:
    encoder = LabelEncoder()
    category = encoder.fit_transform(medical_cost_df[column])
    medical_cost_df[column] = category
    encoders.append(encoder)


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features, targets = medical_cost_df.iloc[:, :-1], medical_cost_df.charges

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

prediction = linear_regression.predict(X_test)

get_evaluation(np.log1p(y_test), prediction)

