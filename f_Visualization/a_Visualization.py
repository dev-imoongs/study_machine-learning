#!/usr/bin/env python
# coding: utf-8

# ### Visualization (시각화)
# 
# ##### 데이터의 유형
# 1. 범주형(Categorical): 개체의 상태를 일정한 기준으로 구분하기 위해 사용한다, 숫자 또는 문자 형태의 값이다.
# - 명목형(Nominal): 개체의 상태를 분류하기 위해 사용하고 서열(순서)이 존재하지 않는 데이터, 예)상품 카테고리: 생활용품, 전자제품, 의류
# - 순위형(Ordinal): 개체의 상태를 분류하기 위해 사용하고 서열(순서)이 존재하는 데이터, 예)학생 수준: High, Medium, Low, 측정년도: 2020, 2021
# 
# 2. 집계형(Numerical): 개체의 상태를 정량적으로 표현하기 위해 사용한다. 숫자 형태의 값이다.
# - 이산형(Discrete): 유한 간격으로서, 소수점으로 나누어질 수 없는 형태이다, 예) 학번: 1, 2, 구매 횟수: 323, 453
# - 연속형(Continuos): 무한 간격으로서, 무한히 반복되는 소수점의 형태이다, 예) 가격: 1280.15648..., 식물의 높이: 10.25158...
# 
# <img src="./images/visual01.png" width="600" style="margin-left:0">
# <img src="./images/visual02.png" width="600" style="margin-left:0">
# <img src="./images/visual03.png" width="600" style="margin-left:0">
# 
# ##### 범주형 (상품 카테고리: 생활용품, 전자제품, 의류, 학생 수준: High, Medium, Low, 측정년도: 2020, 2021)
# - 바이올린 차트
# - 스캐터 플롯  
# - 막대 차트
# - 누적 막대 차트  
# 
# ##### 집계형 (학번: 1, 2, 구매 횟수: 323, 453, 가격: 1280.15648..., 식물의 높이: 10.25158...)
# - 막대그래프(숫자가 적을 경우)
# - 선그래프(숫자가 많을 경우)
# - 바이올린 차트
# - 스캐터 플롯  
# - 히스토그램
# - KDE

# In[1]:


import pandas as pd

titanic_df = pd.read_csv('./datasets/titanic.csv')
titanic_df.head(5)


# In[2]:


titanic_df.isna().sum()


# In[3]:


titanic_df.Age = titanic_df.Age.fillna(titanic_df.Age.mean())


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.bar(titanic_df['Pclass'], titanic_df['Age'])
plt.show()

sns.barplot(x='Pclass', y='Age', data=titanic_df, err_kws={'linewidth': 0})
plt.show()


# In[8]:


plt.violinplot(titanic_df['Age'])
plt.show()

sns.violinplot(x=titanic_df['Age'], hue=titanic_df['Pclass'])
plt.show()


# In[9]:


plt.scatter(titanic_df['Age'], titanic_df['Fare'])
plt.show()

sns.scatterplot(x='Age', y='Fare', hue="Pclass", data=titanic_df)
plt.show()

titanic_df.plot(x='Age', y='Fare', kind='scatter')


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(x='Age', data=titanic_df, hue='Sex', multiple='stack')
plt.show()

sns.histplot(x='Age', data=titanic_df, hue='Sex', multiple='dodge')
plt.show()


# In[16]:


import matplotlib.pyplot as plt

plt.hist(titanic_df['Age'])
plt.show()

sns.histplot(titanic_df['Age'], kde=True)
plt.grid()
plt.show()

titanic_df['Age'].hist(grid=False)
plt.show()


# In[18]:


plt.figure(figsize=(12, 6))
sns.histplot(x='Age', data=titanic_df, kde=True, bins=30)


# In[23]:


plt.figure(figsize=(12, 6))
sns.displot(titanic_df['Age'], kde=True, rug=True, height=6, aspect=2)


# In[24]:


import pandas as pd

avocado_df = pd.read_csv('./datasets/avocado.csv')
avocado_df


# In[25]:


avocado_df.isna().sum()


# In[26]:


sns.lineplot(x='Date', y='AveragePrice', data=avocado_df)


# In[29]:


avocado_df['Date'] = pd.to_datetime(avocado_df.Date)
line = sns.lineplot(x='Date', y='AveragePrice', data=avocado_df)
line.set_xticklabels(line.get_xticklabels(), rotation=90)
plt.show()


# In[30]:


avocado_df['Date'] = pd.to_datetime(avocado_df.Date)
line = sns.lineplot(x='Date', y='AveragePrice', hue='type', data=avocado_df)
line.set_xticklabels(line.get_xticklabels(), rotation=90)
plt.show()


# In[31]:


avocado_df['Date'] = pd.to_datetime(avocado_df.Date)
line = sns.lineplot(x='Date', y='Total Volume', data=avocado_df)
line.set_xticklabels(line.get_xticklabels(), rotation=90)
plt.show()


# In[32]:


avocado_df['Date'] = pd.to_datetime(avocado_df.Date)
line = sns.lineplot(x='Date', y='Total Volume', hue="type", data=avocado_df)
line.set_xticklabels(line.get_xticklabels(), rotation=90)
plt.show()


# In[33]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))

avocado_df['Date'] = pd.to_datetime(avocado_df.Date)
line = sns.lineplot(x='Date', y='AveragePrice', hue='type', data=avocado_df, ax=ax[0])
line.set_xticklabels(line.get_xticklabels(), rotation=90)

avocado_df['Date'] = pd.to_datetime(avocado_df.Date)
line = sns.lineplot(x='Date', y='Total Volume', hue="type", data=avocado_df, ax=ax[1])
line.set_xticklabels(line.get_xticklabels(), rotation=90)

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




