#!/usr/bin/env python
# coding: utf-8

# ### Decision Tree Regression (회귀 트리)
# - 결정 트리와 결정 트리 기반의 앙상블 알고리즘은 분류뿐만 아니라 회귀도 가능하다.
# - 분류와 유사하게 분할을 하며, 최종 분할 후 각 분할 영역에서 실제 데이터까지의 거리들의 평균 값으로 학습 및 예측을 수행한다.
# <img src="./images/decision_tree_regression01.png" width="600" style="margin: 10px; margin-left: 0">
# - 회귀 트리 역시 복잡한 트리 구조를 가질 경우 과적합이 위험이 있고, 트리의 크기와 노드의 개수의 제한 등으로 개선할 수 있다.
# <img src="./images/decision_tree_regression02.png" width="600" style="margin:20px; margin-left: 0">
# - 독립 변수들과 종속 변수 사이의 관계가 상당히 비선형적일 경우 사용하는 것이 좋다.
# <img src="./images/decision_tree_regression03.png" width="800" style="margin:20px; margin-left: 0">
# - 🚩 하지만, 다른 회귀 모델보다 전체적인 성능은 떨어진다.

# ##### 한우 가격 예측

# In[138]:


import chardet

rawdata = open('./datasets/korea_cow.csv', 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']
charenc


# In[139]:


import pandas as pd

cow_df = pd.read_csv('./datasets/korea_cow.csv', encoding='euc-kr')
cow_df


# In[140]:


cow_df.isna().sum()


# In[141]:


cow_df = cow_df.drop(columns=['개체번호', '출하주', 'kpn', '비고', '최저가', '일자', '번호', '지역'])
cow_df.isna().sum()


# In[142]:


cow_df.info()


# In[143]:


cow_df.종류.value_counts()


# In[144]:


cow_df.상태.value_counts()


# In[145]:


cow_df = cow_df[cow_df.상태 == '낙찰']
cow_df


# In[146]:


cow_df = cow_df.drop(columns='상태', axis=1)


# In[147]:


cow_df


# In[148]:


cow_df.describe().T


# In[149]:


from matplotlib import font_manager

plt.rc('font', family='Malgun Gothic')
cow_df.hist()


# In[150]:


from matplotlib import font_manager
import numpy as np

plt.rc('font', family='Malgun Gothic')
np.log1p(cow_df.낙찰가).hist()


# In[151]:


cow_df['가격'] = cow_df.낙찰가


# In[152]:


cow_df = cow_df.drop(columns='낙찰가', axis=1)


# In[153]:


cow_df


# In[154]:


cow_df.성별.value_counts()


# In[155]:


cond1 = cow_df.성별 == '수' 
cond2 = cow_df.성별 == '암'
cond = cond1 | cond2
cow_df = cow_df[cond]
cow_df.성별.value_counts()


# In[156]:


cow_df.info()


# In[157]:


from sklearn.preprocessing import LabelEncoder

encoders = []
for column in cow_df[['성별', '종류']]:
    encoder = LabelEncoder()
    cow_df.loc[:, column] = encoder.fit_transform(cow_df[column])
    encoders.append(encoder)
    print(encoder.classes_)


# In[158]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scaled_cow_df = pd.DataFrame(scale.fit_transform(cow_df.iloc[:, :-1]), columns=cow_df.iloc[:, :-1].columns)


# In[159]:


scaled_cow_df


# In[160]:


from matplotlib import font_manager

plt.rc('font', family='Malgun Gothic')
scaled_cow_df.hist()


# In[161]:


scaled_cow_df = scaled_cow_df[scaled_cow_df.중량.between(-1.96, 1.96)]
scaled_cow_df = scaled_cow_df[scaled_cow_df.계대.between(-1.96, 1.96)]


# In[162]:


from matplotlib import font_manager

plt.rc('font', family='Malgun Gothic')
scaled_cow_df.hist()


# In[165]:


cow_df = cow_df.iloc[scaled_cow_df.index, :]
cow_df


# In[166]:


cow_df = cow_df.reset_index(drop=True)
cow_df


# In[167]:


scaled_cow_df = scaled_cow_df.reset_index(drop=True)
scaled_cow_df['가격'] = cow_df.가격
scaled_cow_df


# In[168]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = scaled_cow_df.corr()
fig = plt.figure(figsize=(7, 5))
heatmap = sns.heatmap(corr, annot=True)
heatmap.set_title("Correlation")


# In[169]:


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


# In[170]:


scaled_cow_df.가격.isna().sum()


# In[171]:


cow_df = cow_df[~cow_df.계대.isna()]
cow_df


# In[173]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

features, targets = cow_df.drop(columns='가격', axis=1), cow_df.가격

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(poly_features, targets, test_size=0.3, random_state=124)


# In[175]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

dt_reg = DecisionTreeRegressor(random_state=124, max_depth=4)
rf_reg = RandomForestRegressor(random_state=124, n_estimators=1000, max_depth=8)
gb_reg = GradientBoostingRegressor(random_state=124, n_estimators=1000, max_depth=8)
xgb_reg = XGBRegressor(n_estimators=1000, max_depth=8)
lgb_reg = LGBMRegressor(n_estimators=1000, max_depth=8)

# 트리 기반의 회귀 모델을 반복하면서 평가 수행 
models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg]
for model in models:  
    model.fit(X_train, np.log1p(y_train))
    prediction = model.predict(X_test)
    print(model.__class__.__name__)
    get_evaluation(np.log1p(y_test), prediction)

