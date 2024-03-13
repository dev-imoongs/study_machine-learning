#!/usr/bin/env python
# coding: utf-8

# ### Polynomial Regression (다항 회귀)
# - 하나의 종속변수와 여러 독립변수 사이의 관계를 분석하는 것은 다중 회귀(Mutlivariate Linear Regession)이며,  
# 각 독립변수에 제곱을 추가하여 비선형 데이터를 선형 회귀 모델로 분석하는 것을 다항 회귀(Polynomial Regression)라고 한다.
# - 데이터 세트에 대해서 독립 변수와 종속 변수의 관계를 단순 선형 회귀로 표현하는 것 보다 다항 회귀 곡선형으로 표현하는 것이 예측성능이 좋다.
# <img src="./images/polynomial_regression01.png" width="500" style="margin:20px; margin-left:-20px">
# ---
# - 선형 회귀와 비선형 회귀를 나누는 기준은 회귀 계수이며, 독립변수의 차수가 증가하는 것은 치환을 통해 모두 선형회귀로 전환된다.
# - 선형 회귀식은 곡선 모형을 가질 수 있으며, 독립변수를 치환하였을 때 선형회귀 식이 된다면 모두 선형 회귀라고 한다.
# - 비선형 회귀에는 대표적으로 로지스틱 회귀가 있으며, 이는 확률을 사용하여 회귀가 아닌 분류에 활용된다.
# 
# ##### 선형 회귀식
# 
# <img src="./images/polynomial_regression02.png" width="300" style="margin:20px; margin-left:40px">
# 
# ##### 비선형 회귀식
# 
# <img src="./images/polynomial_regression03.png" width="240" style="margin:20px; margin-left:20px">

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

    print('MAE: {:.4f}, MSE: {:.2f}, RMSE: {:.2f}, MSLE: {:.2f}, RMSLE: {:.2f}, R2: {:.2f}'.format(MAE, MSE, RMSE, MSLE, RMSLE, R2))


# In[2]:


import pandas as pd
mediacl_cost_df = pd.read_csv('./datasets/medical_cost.csv')
mediacl_cost_df


# In[3]:


from sklearn.preprocessing import LabelEncoder

columns = ['sex', 'smoker', 'region']
encoders = []
for column in columns:
    encoder = LabelEncoder()
    category = encoder.fit_transform(mediacl_cost_df[column])
    mediacl_cost_df[column] = category
    encoders.append(encoder)


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


features, targets = mediacl_cost_df.iloc[:, :-1], mediacl_cost_df.charges

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

prediction = linear_regression.predict(X_test)
get_evaluation(np.log1p(y_test), prediction)


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


features, targets = mediacl_cost_df.iloc[:, :-1], mediacl_cost_df.charges

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(poly_features, targets, test_size=0.2, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

prediction = linear_regression.predict(X_test)
get_evaluation(np.log1p(y_test), prediction)


# In[1]:


new_df


# In[ ]:





# In[ ]:





# In[ ]:




