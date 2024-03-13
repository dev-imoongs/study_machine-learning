#!/usr/bin/env python
# coding: utf-8

# ### Support Vector Regression (SVR)
# ##### 출처: JIYOON LEE
# - 과적합이 발생하면 회귀 계수 W의 크기도 증가하기 때문에 회귀계수의 크기가 너무 커지지 않도록 계수의 크기를 제한하는 정규화 방법을 적용한다.
# <img src="./images/support_vector_regression01.png" width="450" style="margin:10px; margin-left: 0">
# - L2 규제를 사용하는 릿지(Ridge)의 목적은 실제값과 추정값의 차이를 작게 하되, 회귀계수 크기도 작게 하는 선을 찾는 것이다.
# - 패널티를 회귀 계수에 부여한다.
# <img src="./images/support_vector_regression02.png" width="350" style="margin:10px; margin-left: 0">
# - SVR(Support Vector Regression)도 L2 규제를 사용하지만 목적은 회귀계수 크기를 작게 하여 회귀식을 평평하게 만들되, 실제값과 추정값의 차이를 작게 하는 선을 찾는 것이다.
# - 패널티를 손실 함수에 부여한다.
# <img src="./images/support_vector_regression03.png" width="350" style="margin:10px; margin-left: 0">
# 
# ##### ϵ(epsilon)-insensitive Loss function
# -  epsilon: 절대값에서 양수만 남긴다.
# -  SVR의 손실함수를 ϵ-insensitive함수를 사용한 SVR식으로 표현하면 아래와 같다.
# > - ϵ: 회귀식 마진(튜브)
# > - ξ:튜브 위 방향으로 벗어난 거리
# > - ξ<sup>∗</sup>튜브 아래 방향으로 벗어난 거리
# <img src="./images/support_vector_regression04.png" width="300" style="margin:10px; margin-left: 0">
# <img src="./images/support_vector_regression05.png" width="500" style="margin:10px; margin-left: 0">
# - SVR은 회귀식이 추정되면 회귀식 위아래 2ϵ(−ϵ,ϵ)만큼 튜브를 생성하여 회귀선에 대한 상한선, 하한선을 주게된다.
# ##### 🚩 데이터에 노이즈가 있다고 가정하며, 실제 값을 완벽히 추정하는 것을 추구하지 않는다. 적정 범위(2ϵ) 내에 실제값과 예측값의 차이를 허용한다.
# ##### 🚩 SVR은 속도가 많이 느리다.
# 
# ---
# ##### All Loss function hyper parameter
# <img src="./images/support_vector_regression06.png" width="550" style="margin:10px; margin-left: 0">

# ##### SVR(kernel='rbf', degree=3, gamma='scale', C=1.0, epsilon=0.1)
# - kernel: 주어진 데이터에 사용하는 커널함수에 따라 feature space의 특징이 달라지기 때문에 데이터 특성에 적합한 커널함수를 결정한다.
# - degree: 'poly' kernel일 경우만 사용하며, 양수만 가능하고 다른 kernel에서는 무시된다.
# <img src="./images/support_vector_regression07.png" width="500" style="margin:10px; margin-left: 0">
# 
# - gamma: 커널의 폭을 제어하게 되며, gamma가 클수록 회귀선 커브가 심해진다.
# <img src="./images/support_vector_regression08.png" width="500" style="margin:10px; margin-left: 0">
# 
# - C: Cost가 작아지면 잘못 예측한 값에 대해, penalty 부여를 적게 하기 때문에 실제 값과의 차이가 무시된다. 회귀식이 평평해지며, 예측성능도 감소한다.
# <img src="./images/support_vector_regression09.png" width="500" style="margin:10px; margin-left: 0">
# 
# - epsilon: 값이 커질 수록 잘못 예측한 값을 많이 허용해주기 때문에, support vector의 수도 감소하게 되고, 평평한 회귀식이 나타난다.
# <img src="./images/support_vector_regression10.png" width="500" style="margin:10px; margin-left: 0">

# ##### 1인당 건강 보험 비용
# - age: 1차 수혜자의 연령.
# - sex: 보험계약자의 성별(여성 또는 남성).
# - bmi: 체질량지수, 키 대비 체중을 측정하는 척도.
# - children: 건강보험의 적용을 받는 자녀의 수 또는 부양가족의 수.
# - smoker: 흡연 상태(흡연자 또는 비흡연자).
# - region: 수혜자의 미국 내 거주지역(북동, 남동, 남서, 북서).
# - charges: 건강보험에서 청구하는 개인별 의료비.

# In[5]:


import pandas as pd
medical_cost_df = pd.read_csv('./datasets/medical_cost.csv')
medical_cost_df


# In[6]:


from sklearn.preprocessing import LabelEncoder

columns = ['sex', 'smoker', 'region']
encoders = []
for column in columns:
    encoder = LabelEncoder()
    category = encoder.fit_transform(medical_cost_df[column])
    medical_cost_df[column] = category
    encoders.append(encoder)


# In[7]:


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


# In[12]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features, targets = medical_cost_df.iloc[:, :-1], medical_cost_df.charges

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=124)

y_train = np.log1p(y_train)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# 기울기(가중치)
print(linear_regression.coef_)
# 절편(상수)
print(linear_regression.intercept_)

prediction = linear_regression.predict(X_test)
print(linear_regression.score(X_test, np.log1p(y_test)))
print(r2_score(np.log1p(y_test), prediction))
get_evaluation(np.log1p(y_test), prediction)


# In[11]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

features, targets = medical_cost_df.iloc[:, :-1], medical_cost_df.charges

parmas = {
    'gamma': [0.01, 0.1, 1, 10, 100], 
    'C': [0.01, 0.1, 1, 10, 100], 
    'epsilon': [0, 0.01, 0.1, 1, 10, 100]
}

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

grid_svr = GridSearchCV(SVR(kernel='linear'), param_grid=parmas, cv=3, refit=True, return_train_score=True, scoring='r2')


# 로그 변환
y_train = np.log1p(y_train)

grid_svr.fit(X_train, y_train)

prediction = grid_svr.predict(X_test)

# 기울기(가중치)
print(grid_svr.best_estimator_.coef_)

get_evaluation(np.log1p(y_test), prediction)


# In[13]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_svr.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')

