#!/usr/bin/env python
# coding: utf-8

# ### Evaluation Task
# 
# ##### https://drive.google.com/file/d/1v3eNjo3TjTgUPlq6Uw_v_oEQ8sVjaItw/view?usp=share_link
# ##### 한국 아파트 가격 예측

# In[1]:


import pandas as pd
apart_df = pd.read_csv('./datasets/korean_apart.csv', low_memory=False)
apart_df


# In[2]:


apart_df.info()
apart_df = apart_df.drop_duplicates().reset_index(drop=True)
apart_df.info()


# In[3]:


print(apart_df.isna().sum())
apart_df.dropna(inplace=True)
print(apart_df.isna().sum())


# In[4]:


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


# In[5]:


new_apart_df = apart_df.drop(columns=['동'])


# In[6]:


new_apart_df['거래일'] = pd.to_datetime(new_apart_df['거래일'], format='mixed')


# In[7]:


apart_df


# In[8]:


new_apart_df['거래일'] = pd.to_datetime(new_apart_df['거래일'], dayfirst=False)


# In[9]:


new_apart_df[new_apart_df['층'] == ' ']


# In[10]:


new_apart_df.loc[new_apart_df['층'] == ' ', '층'] = 1


# In[13]:


new_apart_df['층'] = new_apart_df['층'].astype('float')
new_apart_df['층'] = new_apart_df['층'].astype('int')
new_apart_df['건축년도'] = new_apart_df['건축년도'].astype('int')


# In[14]:


new_apart_df['거래금액'] = new_apart_df['거래금액'].str.replace(',','').astype('float')


# In[15]:


new_apart_df.info()


# In[16]:


from sklearn.preprocessing import LabelEncoder

columns = ['아파트', '지번',]
encoders = []

for column in columns:
    encode = LabelEncoder()
    encoded_feature = encode.fit_transform(new_apart_df[column])
    new_apart_df[column] = encoded_feature
    encoders.append(encode)
    print(encode.classes_)


# In[17]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔 폰트 경로 설정 (일반적으로 아래와 같은 경로를 사용합니다)
font_path = 'C:/Windows/Fonts/malgunbd.ttf'

# 폰트 설정
font_name = fm.FontProperties(fname=font_path, size=10).get_name()
plt.rc('font', family=font_name)


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = new_apart_df.corr()
heatmap = sns.heatmap(correlation_matrix, cmap='Oranges', annot=True, fmt='.2f')
heatmap.set_title("Correlation")


# In[19]:


train_df = new_apart_df.drop(columns=['거래일','아파트','지번'])
train_df


# In[20]:


new_apart_df.hist(figsize=(15,10),bins=50)
new_apart_df.describe().T


# In[21]:


new_apart_df.info()


# In[22]:


train_df


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features, targets = train_df.iloc[:, :-1], train_df['거래금액']

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

prediction = linear_regression.predict(X_test)

get_evaluation(np.log1p(y_test), prediction)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




