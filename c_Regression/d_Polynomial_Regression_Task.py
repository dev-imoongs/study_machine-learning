#!/usr/bin/env python
# coding: utf-8

# ### Polynomial Regression Task
# 
# ##### 한국인 수익 예측
# - id : 식별 번호
# - year : 조사 년도
# - wave : 2005년 wave 1위부터 2018년 wave 14위까지
# - region: 1)서울 2)경기 3)경남 4)경북 5)충남 6)강원 & 충북 7)전라 & 제주
# - income: 연간 수입 M원(백만원.1100원=1USD)
# - family_member: 가족 구성원 수
# - gender: 1) 남성 2) 여성
# - year_born: 태어난 년도
# - education_level:1)무교육(7세 미만) 2)무교육(7세 이상) 3)초등학교 4)중학교 5)고등학교 6)대학 학위 8)MA 9)박사 학위
# - marriage: 혼인상태. 1)해당없음(18세 미만) 2)혼인중 3)사망으로 별거중 4)별거중 5)미혼 6)기타
# - religion: 1) 종교 있음 2) 종교 없음  
# - occupation: 직종 코드, 별도 첨부
# - company_size: 기업 규모
# - reason_none_worker: 1)능력 없음 2)군 복무 중 3)학교에서 공부 중 4)학교 준비 5)직장인 7)집에서 돌보는 아이들 8)간호 9)경제 활동 포기 10)일할 의사 없음 11)기타

# In[1]:


import pandas as pd


# In[85]:


income_path = './datasets/korean_income.csv'
job_code_path = './datasets/korean_job_code1.csv'
income_df = pd.read_csv(income_path)
job_code_df = pd.read_csv(job_code_path)


# In[12]:


income_df


# In[88]:


job_code_df


# In[135]:


length = income_df['occupation'].values
temp_list = []
for i in length:
    if(i[:2]==' '):
        temp_list.append(0)
    elif(i[:2]== 10):
        temp_list.append(10)
    else:
        temp_list.append(i[:1])
temp_list


# In[136]:


income_df['occupation'] = temp_list


# In[137]:


income_df


# In[13]:


print('='*30)
print(income_df.info())
print('='*30)
print(income_df.isna().sum())
print('='*30)
print(income_df.duplicated().sum())
print('='*30)


# In[14]:


income_df.describe().T


# In[22]:


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


# In[138]:


from sklearn.preprocessing import LabelEncoder

columns = ['company_size', 'reason_none_worker']
encoders = []
for column in columns:
    encoder = LabelEncoder()
    category = encoder.fit_transform(income_df[column])
    income_df[column] = category
    encoders.append(encoder)


# In[139]:


income_df


# In[140]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = income_df.corr()
heatmap = sns.heatmap(correlation_matrix, cmap='Oranges', annot=True, fmt='.2f')
heatmap.set_title("Correlation")


# In[141]:


new_df = income_df.drop(columns=['id','region','religion','company_size'])


# In[142]:


temp = new_df['income']
new_df = new_df.drop(columns='income')
new_df['income'] = temp
new_df


# In[143]:


new_df.hist(figsize=(15,10), bins=80)
new_df.describe().T


# In[144]:


new_df = new_df[new_df['income']>=0].reset_index(drop=True)


# In[147]:


new_df


# In[145]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


features, targets = new_df.iloc[:, :-1], new_df.income

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

y_train.isna().sum()
linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

prediction = linear_regression.predict(X_test)
get_evaluation(np.log1p(y_test), prediction)


# In[161]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


features, targets = new_df.iloc[:, :-1], new_df.income

poly_features = PolynomialFeatures(degree=4).fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(poly_features, targets, test_size=0.2, random_state=0)

# 로그 변환
# y_train = np.log1p(y_train)

# linear_regression = LinearRegression()

# linear_regression.fit(X_train, y_train)

# prediction = linear_regression.predict(X_test)
# get_evaluation(np.log1p(y_test), prediction)

