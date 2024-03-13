#!/usr/bin/env python
# coding: utf-8

# ### Linear Regression Task
# 
# ##### 다이아몬드 가격 예측
# 
# - price: 미국 달러로 표시된 가격 (＄326 ~ ＄18,823)
# - carat: 다이아몬드의 무게(0.2 ~ 5.01)
# - cut: 품질(공정, 좋음, 매우 좋음, 프리미엄, 이상적)
# - color: 다이아몬드 색상, J(최악)부터 D(최우수)까지
# - clarity: 다이아몬드가 얼마나 선명한지에 대한 측정값 (I1(최악), SI2, SI1, VS2, VS1, VVS2, VVS1, IF(최우수))
# - x: 길이(mm) (0 ~ 10.74)
# - y: 너비(mm)(0 ~ 58.9)
# - z: 깊이(mm)(0 ~ 31.8)
# - depth: 총 깊이 백분율 = z / 평균(x, y) = 2 * z / (x + y) (43–79)
# - table: 가장 넓은 점에 대한 다이아몬드 상단 폭(43 ~ 95)

# In[2]:


import pandas as pd
path = './datasets/diamond.csv'
dia_price_df = pd.read_csv(path)


# In[3]:


dia_price_df


# In[4]:


dia_price_df.drop(columns='Unnamed: 0',axis=1, inplace=True)


# In[7]:


print('='*50)
print(dia_price_df.info())
print('='*50)
print(f'결측치 갯수: \n{dia_price_df.isna().sum()}')
print('='*50)
print(f'중복행 갯수: {dia_price_df.duplicated().sum()}')
print('='*50)


# In[9]:


dia_price_df = dia_price_df.drop_duplicates().reset_index(drop=True)


# In[11]:


dia_price_df.describe().T


# ##### 컬럼별 설명
# 
# - price: 미국 달러로 표시된 가격 (＄326 ~ ＄18,823)
# - carat: 다이아몬드의 무게(0.2 ~ 5.01)
# - cut: 품질(공정, 좋음, 매우 좋음, 프리미엄, 이상적)
# - color: 다이아몬드 색상, J(최악)부터 D(최우수)까지
# - clarity: 다이아몬드가 얼마나 선명한지에 대한 측정값 (I1(최악), SI2, SI1, VS2, VS1, VVS2, VVS1, IF(최우수))
# - x: 길이(mm) (0 ~ 10.74)
# - y: 너비(mm)(0 ~ 58.9)
# - z: 깊이(mm)(0 ~ 31.8)
# - depth: 총 깊이 백분율 = z / 평균(x, y) = 2 * z / (x + y) (43–79)
# - table: 가장 넓은 점에 대한 다이아몬드 상단 폭(43 ~ 95)

# In[15]:


dia_price_df.hist(figsize=(10,6), bins=50)


# In[20]:


import seaborn as sns
price = dia_price_df['price'].groupby(dia_price_df.color).sum().sort_values(ascending=True)
price = price.reset_index()
sns.barplot(x='color', y='price', data=price)


# In[21]:


import seaborn as sns
price = dia_price_df['price'].groupby(dia_price_df.cut).sum().sort_values(ascending=True)
price = price.reset_index()
sns.barplot(x='cut', y='price', data=price)


# In[23]:


from sklearn.preprocessing import LabelEncoder

columns = ['cut', 'color', 'clarity']
encoders = []

for column in columns:
    encode = LabelEncoder()
    encoded_feature = encode.fit_transform(dia_price_df[column])
    dia_price_df[column] = encoded_feature
    encoders.append(encode)
    print(encode.classes_)


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = dia_price_df.corr()
heatmap = sns.heatmap(correlation_matrix, cmap='Oranges', annot=True, fmt='.2f')
heatmap.set_title("Correlation")


# In[25]:


temp_data = dia_price_df['price']
dia_price_df = dia_price_df.drop(columns='price', axis=1)
dia_price_df['price'] = temp_data
dia_price_df


# In[40]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scaled_data = scale.fit_transform(dia_price_df)
scaled_df = pd.DataFrame(columns=dia_price_df.columns, data=scaled_data)
scaled_df


# In[45]:


result_df = dia_price_df[(scaled_df >= -1.96) & (scaled_df <= 1.96)]
result_df = result_df.dropna().reset_index(drop=True)
result_df


# In[58]:


result_df.hist(figsize=(10,10))


# In[46]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features, targets = result_df.iloc[:, :-1], result_df.price

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=124)

y_train = np.log1p(y_train)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# 기울기(가중치)
print(linear_regression.coef_)
# 절편(상수)
print(linear_regression.intercept_)


# In[48]:


result_df.columns[linear_regression.coef_.argsort()[::-1]]


# In[49]:


np.expm1(linear_regression.predict(X_test))


# In[50]:


# 문제(X_test)를 전달해서 예측한 값은 로그값이다.
prediction = linear_regression.predict(X_test)
# 실제 성능 평가를 할 때에도 정답(실제 값)도 로그로 변환해야 한다.
print(linear_regression.score(X_test, np.log1p(y_test)))
print(r2_score(np.log1p(y_test), prediction))


prediction = np.expm1(linear_regression.predict(X_test))
print(linear_regression.score(X_test, np.log1p(y_test)))
# 예측한 로그값을 지수로 변환하여 
print(r2_score(y_test, prediction))


# ##### 실제 데이터로 훈련했을 때(로그로 변환하지 않았을 때)

# In[53]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features, targets = result_df.iloc[:, :-1], result_df.price

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=124)

# y_train = np.log1p(y_train)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# 기울기(가중치)
print(linear_regression.coef_)
# 절편(상수)
print(linear_regression.intercept_)


# In[54]:


# 문제(X_test)를 전달해서 예측한 값은 로그값이다.
prediction = linear_regression.predict(X_test)
# 실제 성능 평가를 할 때에도 정답(실제 값)도 로그로 변환해야 한다.
print(linear_regression.score(X_test, y_test))
print(r2_score(y_test, prediction))


# In[ ]:





# In[ ]:





# In[ ]:




