#!/usr/bin/env python
# coding: utf-8

# ### OSL(Ordinary Least Square) Task
# 
# ##### 서울 자전거 대여 수 분석
# 
# - Date: 대여 날짜
# - Rented Bike Count: 대여 수
# - Hour: 대여 시간
# - Temperature(°C): 온도(섭씨)
# - Humidity(%): 습도 (%)
# - Wind speed (m/s): 풍속 (m/s)
# - Visibility (10m): 가시거리 (10m)
# - Dew point temperature(°C): 이슬점(°C), 이슬점이 낮으면 10~15°C 정도로 공기가 편안하게 느껴진다.
# - Solar Radiation (MJ/m2): 태양복사 (MJ/m2)
# - Rainfall(mm): 강우량/비 (mm)
# - Snowfall (cm): 강우량/눈 (cm)
# - Seasons: 계절
# - Holiday: 공휴일

# In[1]:


import chardet

rawdata = open('./datasets/seoul_bicycle.csv', 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']
charenc


# In[2]:


import pandas as pd

bicycle_df = pd.read_csv('./datasets/seoul_bicycle.csv', encoding='ISO-8859-1')
bicycle_df


# In[3]:


bicycle_df[bicycle_df['Functioning Day']=='Yes'].reset_index(drop=True)
bicycle_df.drop(columns='Functioning Day',axis=1,inplace=True)


# In[4]:


print('='*40)
print(bicycle_df.info())
print('='*40)
print(bicycle_df.isna().sum())
print('='*40)
print(bicycle_df.duplicated().sum())
print('='*40)


# In[5]:


from sklearn.preprocessing import LabelEncoder

encoders = []
columns = ['Seasons', 'Holiday']

for column in columns:
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(bicycle_df[column])
    bicycle_df[column] = encoded_feature
    print(encoder.classes_)
    encoders.append(encoder)


# In[6]:


bicycle_df.describe().T


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(bicycle_df)
plt.show()


# In[11]:


import statsmodels.api as sm
##### 성별과 가족 구성원 수가 가장 큰 영향을 미친다.
model = sm.OLS(bicycle_df[['Rented Bike Count']], bicycle_df.drop(columns=['Date'],axis=1))
print(model.fit().summary())


# In[9]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def feature_engineering_VIF(features):
    vif = pd.DataFrame()
    vif['vif_score'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    vif['feature'] = features.columns
    return vif


# In[10]:


print(feature_engineering_VIF(bicycle_df.iloc[:, 2:]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




