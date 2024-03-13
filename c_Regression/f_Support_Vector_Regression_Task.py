#!/usr/bin/env python
# coding: utf-8

# ### Support Vector Regression Task
# 
# ##### 한국 방문자 수 예측
# 
# - date: 날짜 (년-월)
# - nation: 방문자의 국가
# - visitor: 방문자 수
# - growth: 전년 동월 대비 방문객 수 성장률
# - share: 해당 월의 전체 방문자 중 해당 국가 비율
# 
# ##### 년도와 국가를 입력해서 예상 방문자 수를 예측하세요.

# In[1]:


import pandas as pd

visitor_df = pd.read_csv('./datasets/korea_visitor.csv')
visitor_df


# In[2]:


visitor_df.isna().sum()


# In[3]:


visitor_df.duplicated().sum()


# In[4]:


visitor_df.info()


# In[5]:


visitor_df.date = visitor_df.date.apply(lambda date: date.split("-")[0])


# In[7]:


visitor_df = visitor_df.rename(columns={'date': 'year'})
visitor_df


# In[8]:


visitor_df.year.value_counts()


# In[11]:


visitor_df = visitor_df.visitor.groupby(by=[visitor_df.year, visitor_df.nation]).sum().reset_index()
visitor_df


# In[12]:


visitor_df.info()


# In[13]:


visitor_df.year = visitor_df.year.astype('int16')
visitor_df.info()


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plot = sns.lineplot(x='nation', y='visitor', data=visitor_df, hue='year')
plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
plt.show()


# In[15]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
visitor_df.nation = encoder.fit_transform(visitor_df.nation)
print(encoder.classes_)


# In[16]:


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


# In[17]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR

features, targets = visitor_df.iloc[:, :-1], visitor_df.visitor

parmas = {
    'gamma': [0.1, 1], 
    'C': [0.01, 0.1, 1, 10, 100], 
    'epsilon': [0, 0.01, 0.1]
}

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=124)

grid_svr = GridSearchCV(SVR(), param_grid=parmas, cv=3, refit=True, return_train_score=True, scoring='neg_mean_squared_log_error')

# 로그 변환
y_train = np.log1p(y_train)

grid_svr.fit(X_train, y_train)

prediction = grid_svr.predict(X_test)

get_evaluation(np.log1p(y_test), prediction)


# In[18]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_svr.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[19]:


visitor_df[visitor_df.nation == 56]


# In[22]:


import numpy as np
prediction = grid_svr.predict(pd.DataFrame([[2021, 56]], columns=['year', 'nation']))
np.expm1(prediction)

