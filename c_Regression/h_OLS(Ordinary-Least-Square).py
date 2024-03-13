#!/usr/bin/env python
# coding: utf-8

# ##### Ordinary Least Square(OSL)
# - 선형 회귀 모델을 평가하는 데 유용한 방법이며, 모델 전체와 모델의 각 feature에 대한 통계적 성능 지표를 사용하여 수행된다.
# - 다양한 유형의 통계 모델을 추정하고 여러 통계 테스트를 수행하는 여러 클래스와 기능을 제공한다.
# - 관측된 데이터에 선형 방정식을 적용하여 생성되며, 가장 일반적인 방법이다.
# - P>|t| (p-value): 0.05보다 작으면 독립 변수가 종속 변수에 영향을 미치는 것이 유의미하다는 것을 의미한다.
# - Durbin-Watson: 보통 1.5에서 2.5 사이이면 독립으로 판단하고 회귀 모형이 적합하다는 것을 의미한다.
# - 🚩 단, R<sup>2</sup> 값을 유지 또는 개선시키는 방향으로만 수행한다.
# 
# ##### VIF(Variance Inflation Factor)
# - 분산 팽창 요인 수치가 5 또는 10 이상일 경우 다중 공선성의 문제가 있다는 뜻이다.
# 
# ##### 다중 공선성(Multicollinearity)
# - 회귀분석에서 독립변수들 간에 강한 상관관계가 나타나는 문제를 의미한다.
# <img src="./images/multicollinearity.png" style="margin-left: 0">

# ##### 쇼핑 고객 데이터
# 
# - Customer ID: 고객 아이디
# - Gender: 고객의 성별
# - Age: 고객의 나이
# - Annual Income: 고객의 연소득
# - Spending Score: 고객 행동 및 지출 성격에 따라 상점에서 할당한 점수
# - Profession: 직업, 전문직
# - Work Experience: 고객의 근무 경력(연 단위)
# - Family Size: 가족 구성원 수

# In[2]:


import pandas as pd

customer_df = pd.read_csv('./datasets/customers.csv')
customer_df


# In[3]:


customer_df.isna().sum()


# In[4]:


customer_df.Profession.value_counts()


# In[5]:


customer_df = customer_df[~customer_df.Profession.isna()]
customer_df = customer_df.reset_index(drop=True)
customer_df.isna().sum()


# In[6]:


customer_df.duplicated().sum()


# In[7]:


customer_df = customer_df.drop(columns='CustomerID', axis=1)
customer_df


# In[8]:


customer_df['Score'] = customer_df['Spending Score (1-100)']
customer_df = customer_df.drop(columns='Spending Score (1-100)', axis=1)
customer_df


# In[9]:


from sklearn.preprocessing import LabelEncoder

encoders = []
columns = ['Gender', 'Profession']

for column in columns:
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(customer_df[column])
    customer_df[column] = encoded_feature
    print(encoder.classes_)
    encoders.append(encoder)


# In[10]:


import matplotlib.pyplot as plt
# conda install -c conda-forge seaborn   (0.12.2 이상)
import seaborn as sns

sns.pairplot(customer_df[['Gender', 'Age', 'Annual Income ($)', 'Profession', 'Work Experience', 'Family Size']])
plt.show()


# In[11]:


import statsmodels.api as sm
##### 성별과 가족 구성원 수가 가장 큰 영향을 미친다.
model = sm.OLS(customer_df[['Score']], customer_df[['Gender', 'Age', 'Annual Income ($)', 'Profession', 'Work Experience', 'Family Size']])
print(model.fit().summary())


# In[13]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def feature_engineering_VIF(features):
    vif = pd.DataFrame()
    vif['vif_score'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    vif['feature'] = features.columns
    return vif


# In[14]:


print(feature_engineering_VIF(customer_df.iloc[:, :-1]))

