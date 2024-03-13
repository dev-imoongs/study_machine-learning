#!/usr/bin/env python
# coding: utf-8

# ### Decision Tree Regression Task
# 
# ##### 기온 및 강수량 별 모기 비율 예측
# 
# - date: 년-월-일
# - mosquito_Indicator: 모기 비율
# - rain(mm): 일 강수량
# - mean_T(℃): 일 평균 기온
# - min_T(℃): 일 최저 기온
# - max_T(℃): 일 최고 기온

# In[16]:


import pandas as pd

mos_df = pd.read_csv('./datasets/korea_mosquito.csv')
mos_df


# In[17]:


mos_df.describe().T


# In[18]:


mos_df.isna().sum()


# In[19]:


mos_df.duplicated().sum()


# In[20]:


mos_df = mos_df.drop_duplicates()
mos_df = mos_df.reset_index(drop=True)
mos_df


# In[21]:


mos_df = mos_df.drop(columns='date', axis=1)
mos_df.info()


# In[22]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].scatter(x='rain(mm)', y='mosquito_Indicator', data=mos_df, c='blue', s=3)
ax[1].scatter(x='mean_T(℃)', y='mosquito_Indicator', data=mos_df, c='red', marker='^', s=3)
plt.show()


# In[23]:


mos_df.mosquito_Indicator.hist()


# In[24]:


from sklearn.preprocessing import StandardScaler

mos_df['scale_target'] = StandardScaler().fit_transform(mos_df[['mosquito_Indicator']])
mos_df[~mos_df.scale_target.between(-1.96, 1.96)].shape[0]


# In[27]:


mos_df = mos_df[mos_df.scale_target.between(-1.96, 1.96)].reset_index(drop=True)
mos_df


# In[28]:


mos_df = mos_df.drop(columns='scale_target', axis=1)
mos_df


# In[29]:


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


# In[30]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

features, targets = mos_df.iloc[:, 1:], mos_df.mosquito_Indicator

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=124)

scale = StandardScaler()

X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

y_train = np.log1p(y_train)

dt_reg = DecisionTreeRegressor(random_state=124, max_depth=4)
rf_reg = RandomForestRegressor(random_state=124, n_estimators=3000, max_depth=8)
gb_reg = GradientBoostingRegressor(random_state=124, n_estimators=3000, max_depth=8)
xgb_reg = XGBRegressor(n_estimators=3000, max_depth=8)
lgb_reg = LGBMRegressor(n_estimators=3000, max_depth=8)

# 트리 기반의 회귀 모델을 반복하면서 평가 수행 
models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg]
for model in models:  
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(model.__class__.__name__)
    get_evaluation(np.log1p(y_test), prediction)

