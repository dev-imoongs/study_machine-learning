#!/usr/bin/env python
# coding: utf-8

# ### Dimension Reduction (차원 축소)
# - 우리가 다루는 데이터들은 보통 3차원 공간에선 표현하기 힘든 고차원(high dimension)의 데이터인 경우가 많다.
# - 차원이 커질 수록 데이터 간 거리가 크게 늘어나며, 데이터가 희소화(Spares) 된다(차원의 저주).
# > - 희소 데이터(Spares Data): 차원/전체 공간에 비해 데이터가 있는 공간이 매우 협소한 데이터
# > - 밀집 데이터(Dense Data): 차원/전체 공간에 비해 데이터가 있는 공간이 빽빽하게 차 있는 데이터
# - 고차원을 이루는 피처 중 상대적으로 중요도가 떨어지는 피처가 존재할 수 있기 때문에 계산 비용이 많고 분석에 필요한 시각화가 어렵다.
# - 머신러닝에서는 고차원 데이터를 다루는 경우가 많으며, 희소 데이터를 학습 시 예측 성능이 좋지 않다.
# - 차원 축소를 통해 Spares Data를 Dense하게 만들 필요가 있다.
# - feature가 많을 경우 feauture간 상관관계가 높아질 가능성이 높고, 이로 인해 선형 회귀 모델에서 다중 공선성(Multicollinearity) 문제가 발생할 수 있다.
# - 차원 축소로 인해 표현력이 일부 손실되지만, 손실을 감수하더라도 계산 효율을 얻기 위해 사용한다.
# ---
# 
# #### PCA (Principal Component Analysis), 주성분 분석
# - 고차원(x<sub>1</sub>, ···, x<sub>n</sub>)의 데이터를 저차원 (x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>)으로 압축하는 대표적인 차원 축소(Dimension reduction) 방법이다.
# - 데이터의 특성을 눈으로 쉽게 파악할 수 있도록 하며, 연산 속도에 큰 이점을 얻을 수 있다.
# - 고차원 데이터를 저차원 데이터로 압축하기 위해서는 먼저 데이터를 가장 잘 표현하는 축(Principal Component(PC), 주성분)을 설정해야 한다.
# - 주성분 분석을 통해 저차원 공간으로 변환할 때 feature 추출(feature extraction)을 진행한다. feature 추출은 기존 피처를 조합해 새로운 feature로 만드는 것을 의미하며, 새로운 feature와 기존 feature 간 연관성이 없어야 한다.
# - 연관성이 없도록 하기 위해서는 내적했을 때 0이 나와야 하고 이는 서로 직교(직각, 90도)하는 것을 의미한다. 따라서 직교 변환을 수행한다.
# - 직교 변환을 수행할 때 기존 feature와 관련 없으면서 기존 데이터의 표현력을 잘 보존해야하고 이는 다른 말로 "분산(Variance)을 최대로 하는 주축을 찾는다" 라고 할 수 있다.
# - 예를 들어, 2차원 공간이 있고 1차원 공간으로 차원을 축소 한다면 1차원 공간상에서 데이터 분포가 가장 넓게 퍼지게 만드는 벡터(Eigen vector, 고유 벡터)를 찾아야 한다.
# - 찾은 고유 벡터 축에 feature들을 투영시키면 이게 바로 주성분이 된다.
# - Eigen vector를 구하기 위해서는 고유값 분해(EVD, Eigen Value decomposition) 혹은 특이값 분해 (SVD, Singular Value Decomposition)가 수행되어야 하고, 이를 통해 구한 주축에 투영을 하기 위해서 공분산 행렬이 필요하다. 즉, PCA는 데이터들의 공분산 행렬에 대한 특이값 분해(SVD)이다.
# 
# <div style="display: flex">
#     <div>
#         <img src="./images/pca02.gif" style="margin-left: -200px">
#     </div>
#     <div>
#         <img src="./images/pca01.gif" width="700" style="margin-top:50px; margin-left: -350px">
#     </div>
# </div>
# 
# > ##### 공분산 행렬 (Covariance matrix)
# > - 공분산이란, 두 피처가 함께 변하는 정도를 의미하며 구하는 공식은 아래와 같다.
# <img src="./images/pca03.png" width="150" style="margin:20px; margin-left: 0">
# > - n은 행렬 X에 있는 데이터 개수를 나타내며 X의 열축은 feature이고 X의 행축은 데이터 개수이다.
# > - 예를 들어, X의 feature가 x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>라고 가정하면, X<sup>T</sup> * X는 다음과 같다.
# <img src="./images/pca04.png" width="250" style="margin:20px; margin-left: 0">
# > - 전치 행렬과 행렬을 내적하면, 아래와 같은 대칭행렬이 만들어지며, 모든 대각 성분은 같은 피처 간 내적이므로 분산에 해당한다.
# <img src="./images/pca05.png" width="250" style="margin:20px; margin-left: 0">
# > - feature 간 내적을 통해 수치적으로 i번 피처와 j번 피처가 공변하는 정도를 알 수 있게 되고, 공분산이 커질 수 있으므로 데이터 개수(n)로 나누어 평균을 구한다.<br>
# > 🚩위와 같은 행렬을 공분산 행렬이라 하고, 임의의 벡터가 있을 때 공분산 행렬을 곱해주게 되면 그 벡터의 선형 변환(투영)이 이루어진다.
# 
# > ##### 고유값 분해(Eigen Value decomposition)와 특이값 분해 (Singular Value Decomposition)
# > - 고유값 분해가 주축을 이루는 벡터를 찾는 것이라면, 특이값 분해는 직교하는 벡터들을 찾아내는 것이다.
# > - 고유값 분해는 투영시 크기만 변하고 여전히 방향이 변하지 않는 벡터(V)를 찾아내는 것이며, 특이값 분해는 직교하는 벡터를 투영 시 여전히 직교하는 벡터를 찾아내는 것이다.
# > - 공분산 행렬을 통해 고유값 또는 특이값을 분해함으로써 PCA에 필요한 주축인 고유 벡터와, 고유 값을 얻을 수 있다.
# > - 고유값을 얻은 뒤 내림차순으로 정렬했을 때 가장 첫 번째 값이 분산을 최대로 하는 값이 된다.
# > - 고유값 분해는 행렬의 크기가 커질 수록 연산량이 증가하므로 계산 시간이 오래걸릴 수 있으며, 대칭 행렬(m * m)이 아니라면 사용할 수 없다.
# > - sklearn의 PCA는 내부적으로 SVD를 사용한다.
# 
# #### LDA (Linear Discriminant Analysis), 선형 판별 분석
# - PCA와 매우 유사하지만, 분류에서 사용하기 쉽도록 개별 클래스를 분별할 수 있는 기준을 최대한 유지하면서 차원을 축소한다.
# - PCA는 가장 큰 분산을 가지는 축을 찾았지만, LDA는 입력 데이터의 클래스(카테고리)를 최대한 분리할 수 있는 축을 찾는다.  
# 즉, 같은 클래스(카테고리)의 데이터에 최대한 근접해서, 다른 클래스(카테고리)의 데이터를 최대한 떨어뜨리는 축을 매핑한다.
# <div style="display: flex">
#     <div>
#         <img src="./images/lda01.png" width="650" style="margin:20px; margin-left: -20px">
#     </div>
#     <div>
#         <img src="./images/lda02.png" width="650" style="margin:20px; margin-left: 0">
#     </div>
# </div>
# 
# - 클래스를 최대한 분리하기 위해서 클래스 간 분산 을 최대화 하고 클래스 내부 분산을 최소화 하는 방식으로 차원을 축소한다.
# - 클래스별 산포 행렬을 구하여 이를 통해 클래스 간 분산과 클래스 내부 분산을 구한다.
# <img src="./images/lda03.png" width="250" style="margin:20px; margin-left: 0">

# ##### PCA

# ##### 회사 파산 데이터

# In[1]:


import pandas as pd

company_df = pd.read_csv('./datasets/company.csv')
print(company_df.shape)
company_df.columns


# In[2]:


company_df


# In[3]:


company_df.isna().sum().sum()


# In[4]:


company_df.duplicated().sum()


# In[5]:


company_df['Bankrupt?'].value_counts()


# In[6]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=124)
featrues, targets = company_df.iloc[:, 1:], company_df['Bankrupt?']
over_features, over_targets = smote.fit_resample(featrues, targets)

print('SMOTE 적용 전:\n',pd.Series(targets).value_counts() )
print('SMOTE 적용 후:\n',pd.Series(over_targets).value_counts() )


# In[7]:


over_company_df = pd.DataFrame(over_features, columns=company_df.iloc[:, 1:].columns)
over_company_df.shape


# In[8]:


over_company_df


# In[9]:


from sklearn.preprocessing import StandardScaler

over_company_scaled = StandardScaler().fit_transform(over_company_df)
over_company_scaled.shape


# In[10]:


over_company_scaled_df = pd.DataFrame(over_company_scaled, columns=company_df.iloc[:, 1:].columns)
over_company_scaled_df['target'] = over_targets


# In[11]:


over_company_scaled_df


# In[13]:


for column in company_df.iloc[:, 1:].columns:
    over_company_scaled_df = over_company_scaled_df[over_company_scaled_df[column].between(-1.96, 1.96)]

over_company_scaled_df.shape


# In[14]:


over_company_scaled_df.target.value_counts()


# In[18]:


over_company_df = pd.DataFrame(over_features, columns=company_df.iloc[:, 1:].columns)
over_company_df['target'] = over_targets
over_company_df.shape


# In[19]:


over_company_scaled_df


# In[20]:


over_company_df = over_company_df.iloc[over_company_scaled_df.index, :]
over_company_df.shape


# In[21]:


over_company_df = over_company_df.reset_index(drop=True)
over_company_scaled_df = over_company_scaled_df.reset_index(drop=True)
print(over_company_df.shape[0], over_company_scaled_df.shape[0])


# In[22]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

company_pca = pca.fit_transform(over_company_scaled_df.iloc[:, :-1])
print(company_pca.shape)


# In[23]:


# PCA 환된 데이터의 컬럼명을 각각 pca1, pca2, ..., pcan으로 명명
pca_columns=[f'pca{i+1}' for i in range(2)]
company_pca_df = pd.DataFrame(company_pca, columns=pca_columns)
company_pca_df['target']=over_company_scaled_df.target
company_pca_df.head(10)


# In[24]:


print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())


# In[25]:


import seaborn as sns

sns.scatterplot(x="pca1", y="pca2", hue='target', data=company_pca_df, alpha=0.5)


# In[27]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rfc = RandomForestClassifier(random_state=156, max_depth=3, min_samples_split=30)
scores = cross_val_score(rfc, over_company_df.iloc[:, 1:], over_company_df.target, scoring='accuracy', cv=5)
print('원본 데이터 교차 검증 개별 정확도:',scores)
print('원본 데이터 평균 정확도:', np.mean(scores))


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rfc = RandomForestClassifier(random_state=156, max_depth=3, min_samples_split=30)
scores = cross_val_score(rfc, company_pca_df.iloc[:, :-1], company_pca_df.target, scoring='accuracy', cv=5)
print('PCA 데이터 교차 검증 개별 정확도:',scores)
print('PCA 데이터 평균 정확도:', np.mean(scores))


# ##### LDA

# In[30]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 타겟 클래스(카테고리) 개수 -n을 전달한다.
lda = LinearDiscriminantAnalysis(n_components=1)

lda.fit(over_company_scaled_df.iloc[:, :-1], over_company_scaled_df.target)
company_lda = lda.transform(over_company_scaled_df.iloc[:, :-1])

print(company_lda.shape)


# In[31]:


company_lda_df = pd.DataFrame()
company_lda_df['lda']=company_lda.flatten()
company_lda_df['target']=over_company_scaled_df.target
company_lda_df.head(10)


# In[32]:


print(lda.explained_variance_ratio_)
print(lda.explained_variance_ratio_.sum())


# In[33]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rfc = RandomForestClassifier(random_state=156, max_depth=3, min_samples_split=30)
scores = cross_val_score(rfc, over_company_df.iloc[:, 1:], over_company_df.target, scoring='accuracy', cv=5)
print('원본 데이터 교차 검증 개별 정확도:',scores)
print('원본 데이터 평균 정확도:', np.mean(scores))


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rfc = RandomForestClassifier(random_state=156, max_depth=3, min_samples_split=30)
scores = cross_val_score(rfc, company_lda_df[['lda']], company_lda_df.target, scoring='accuracy', cv=5)
print('LDA 데이터 교차 검증 개별 정확도:',scores)
print('LDA 데이터 평균 정확도:', np.mean(scores))

