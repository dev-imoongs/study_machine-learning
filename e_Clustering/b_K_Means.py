#!/usr/bin/env python
# coding: utf-8

# ### K-means Clustering (K-평균 군집 분석)
# - 데이터를 K개의 Cluster(군집)으로 묶는 알고리즘으로서 각 군집의 평균을 활용하여 K개의 군집으로 묶는다.
# - 비슷한 특성을 지닌 데이터들끼리 묶어 K개의 군집으로 군집화하는 대표적인 군집화 기법이며, 거리를 기반으로 군집화한다.
# - 각 집합별 중심점에서 집합 내 오브젝트간 거리의 제곱합을 최소로 하는 집합 S를 찾는 것이 K-Means 알고리즘의 목표이다.
# > 1. 초기 k "평균값" (위의 경우 k=3) 은 데이터 오브젝트 중에서 무작위로 뽑힌다.
# <img src="./images/k_means01.png" width="300" style="margin-left: 0; margin-bottom: 20px">
# > 2. k 각 데이터 오브젝트들은 가장 가까이 있는 평균값을 기준으로 묶인다.
# <img src="./images/k_means02.png" width="300" style="margin-left: 0; margin-bottom: 20px">
# > 3. k개의 클러스터의 중심점을 기준으로 평균값이 재조정된다.
# <img src="./images/k_means03.png" width="300" style="margin-left: 0; margin-bottom: 20px">
# > 4. 수렴할 때까지 2, 3 과정을 반복한다.
# <img src="./images/k_means04.png" width="300" style="margin-left: 0; margin-bottom: 20px">
# > 5. 결과
# <img src="./images/k_means05.gif" width="500" style="margin-left: -40px">

# ##### 국가별 요소 분석을 통한 NGO 지원 순위 지정
# 국제인도주의 NGO는 약 1,000만 달러를 모금했습니다.  
# 이제 NGO의 CEO는 이 돈을 어떻게 전략적이고 효과적으로 사용할 것인지 결정해야 합니다.  
# 우리의 업무는 데이터 분석가로서 국가의 전반적인 발전을 결정하는 사회경제적, 건강적 요소들을 사용하여 국가들을 분류하는 것입니다.  
# 이 분석 이후에 우리는 CEO가 중점을 두고 가장 높은 우선순위를 두어야 할 국가들을 제안해야 합니다.
# 
# - country: 국가명
# - child_mort: 출생아 수 1000명당 5세 미만 아동 사망률
# - exports: 상품 및 서비스의 수출 전체 GDP의 %로 부여되는 상품 및 서비스의 수출
# - health: 총 GDP 대비 총 보건 지출 비율(%)
# - imports: 상품 및 서비스 수입액(총 GDP의 %)
# - income: 1인당 순이익
# - inflation: 총 GDP의 연간 성장률 측정
# - life_expec: 현재의 사망률 패턴을 그대로 유지할 경우 신생아가 살 수 있는 평균 연수
# - total_fer: 현재의 연령-출산율이 동일한 경우 각 여성이 낳은 자녀의 수
# - gdpp: 1인당 GDP

# In[1]:


import pandas as pd

country_df = pd.read_csv('./datasets/country.csv')
country_df


# In[2]:


country_df.isna().sum()


# In[3]:


country_df.duplicated().sum()


# In[4]:


country_df.describe().T


# In[5]:


country_df.info()


# In[8]:


country_df.country.value_counts().count()


# In[9]:


import matplotlib.pyplot as plt
plt.figure(figsize=(30, 3))
country_df.groupby('country')['child_mort'].mean().sort_values(ascending=False).plot(kind='bar')


# <div style="width:3000px">
#     <img src="./images/country_df01.png">
# <div>

# In[10]:


import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.lineplot(x=country_df['income'], y=country_df['child_mort'], ax=ax[0])
sns.lineplot(x=country_df['total_fer'], y=country_df['child_mort'], ax=ax[1])


# In[11]:


from sklearn.preprocessing import StandardScaler

scaled_country_df = pd.DataFrame()

scaler = StandardScaler()
columns = ['child_mort', 'health', 'income', 'life_expec', 'total_fer']
scaled_country_df[columns] = scaler.fit_transform(country_df[columns])


# In[13]:


scaled_country_df['country'] = country_df.country
scaled_country_df


# In[15]:


# conda install -c plotly plotly_express
import plotly_express as px

custom_colors = ['#FFA500', '#A32EFF', '#73D393', '#6988E7']

cluster_data = pd.DataFrame({'country': scaled_country_df['country'],
                             'child_mort': scaled_country_df['child_mort']})

# Creating the choropleth map
fig = px.choropleth(data_frame=cluster_data,
                    locations='country',
                    locationmode='country names', 
                    color='child_mort', 
                    color_continuous_scale=custom_colors, 
                    range_color=[0, 3])

# Updating the layout to include a title
fig.update_layout(title='Countries clustered by priority')

# Showing the plot
fig.show()


# In[16]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
scaled_country_df['country'] = encoder.fit_transform(scaled_country_df['country'])
print(encoder.classes_)


# In[17]:


scaled_country_df


# In[18]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_df = pca.fit_transform(scaled_country_df.drop(columns='country', axis=1))

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())


# In[19]:


pca_columns=[f'pca{i+1}' for i in range(2)]
pca_df = pd.DataFrame(pca_df, columns=pca_columns)
pca_df.head(10)


# In[20]:


from sklearn.cluster import KMeans

# k 개수
x = []

# 응집도
y = []

for k in range(1, 6):
    k_means = KMeans(n_clusters=k, random_state=124)
    k_means.fit(pca_df)
    
    x.append(k)
    y.append(k_means.inertia_)
    
plt.plot(x, y)


# In[21]:


k_means = KMeans(n_clusters=3, random_state=124)
k_means.fit_predict(pca_df)
pca_df['cluster'] = k_means.labels_
pca_df


# In[22]:


centroids = k_means.cluster_centers_

centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=pca_df)
sns.scatterplot(x=centroids_x, y=centroids_y, color='red')


# In[24]:


pca_df['country'] = scaled_country_df.country
pca_df.groupby(by='cluster')['country'].count()


# In[25]:


country_df['cluster'] = pca_df.cluster
country_df


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

sns.barplot(x='child_mort', hue='cluster', data=country_df, ax=ax[0])
sns.barplot(x='income', hue='cluster', data=country_df, ax=ax[1])


# In[29]:


country_df[country_df.cluster == 0].country

