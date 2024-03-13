#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 계층적 군집분석
# 필요 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set_palette("hls")

# 값이 깨지는 문제 해결을 위해 파라미터값 설정
import os
if os.name == 'nt' : # windows OS
    font_family = "Malgun Gothic"
else : #Mac OS
    font_family = "AppleGothic"
sns.set(font=font_family, rc ={"axes.unicode_minus" : False})

from scipy.cluster.hierarchy import linkage, dendrogram


# In[2]:


path = './datasets/pro/'

file_name_1 = '공공자전거 대여소 정보(23.06월 기준).csv'
file_name_2 = '서울특별시 공공자전거 대여이력 정보_2306.csv'

rental_place = pd.read_csv(path + file_name_1, low_memory=False, encoding='CP949')
rental_info = pd.read_csv(path + file_name_2, low_memory=False, encoding='CP949')


# In[3]:


display(rental_place)
display(rental_info)


# In[4]:


rental_info.describe().T


# In[5]:


# rental_info 데이터세트 전처리

condition1 = rental_info['이용거리(M)'] > 0
condition2 = rental_info['이용시간(분)'] > 0
rental_info_df = rental_info[condition1 & condition2].reset_index(drop=True)

# NaN값 드랍
rental_info = rental_info.dropna().reset_index(drop=True)

# 미반납 데이터는 반납 대여소에 0으로 대입
rental_info.loc[rental_info['반납대여소번호']=='\\N',['반납대여소번호','반납대여소명','반납거치대']] = 0


# In[6]:


set_columns_name = ['대여소번호', '대여소명', '자치구', '상세주소', '위도', '경도', '설치시기', '거치대수(LCD)', '거치대수(QR)', '운영방식']


# In[7]:


rental_place = rental_place.iloc[4:,:10]


# In[8]:


for i, j in enumerate(rental_place.columns):
    rental_place = rental_place.rename(columns={j: set_columns_name[i]})


# In[9]:


rental_place


# In[10]:


import pandas as pd

region_list = ['마포구', '서대문구', '용산구', '종로구', '중구', '동대문구', '성동구']
selected_regions = []

for region in region_list:
    selected_rows = rental_place[rental_place['자치구'] == region]
    selected_regions.append(selected_rows)

# 모든 선택된 지역을 하나의 DataFrame으로 합치기
merged_rental_place = pd.concat(selected_regions, ignore_index=True)

# 결과 확인
merged_rental_place


# In[11]:


# ['마포구', '서대문구', '용산구', '종로구', '중구', '동대문구', '성동구'] 지역만 남겼다
rental_num_list = merged_rental_place.대여소번호.to_list()


# In[12]:


print(rental_place.info())
print('='*30)
print(rental_place.isna().sum())
print('='*30)
print(rental_place.duplicated().sum())


# In[13]:


print(rental_info_df.info())
print('='*30)
print(rental_info_df.isna().sum())
print('='*30)
print(rental_info_df.duplicated().sum())


# ##### 필요한 컬럼들만 추출하고 rental_place_df로 이름 붙임
#     rental_place['대여소번호', '자치구', '위도', '경도']

# In[14]:


# 데이터 타입 변환
rental_place['대여소번호'] = rental_place['대여소번호'].astype('int64')
rental_info['반납대여소번호'] = rental_info['반납대여소번호'].astype('int64')


# In[15]:


rental_place_df = rental_place[['대여소번호', '자치구', '위도', '경도']]
rental_place_df


# In[16]:


rental_info


# In[17]:


rental_num_list

# '대여소번호'가 rental_info_list에 속하는 행만 남기기
rental_info = rental_info[rental_info['대여 대여소번호'].isin(rental_num_list)]
rental_info = rental_info[rental_info['반납대여소번호'].isin(rental_num_list)].reset_index(drop=True)

# 결과 확인
rental_info


# In[18]:


rental_info = rental_info.drop(columns=['생년', '자전거번호', '대여 대여소명', '대여거치대', '반납거치대', '반납대여소명', '대여일시', '반납일시', '이용자종류', '대여대여소ID', '반납대여소ID'], axis=1)
rental_info


# In[19]:


result_info_df = pd.merge(rental_info, rental_place_df, how='left', left_on='대여 대여소번호', right_on='대여소번호', suffixes=('_대여', '_대여소'))
result_info_df = result_info_df.rename(columns={'위도':'대여위도','경도':'대여경도'})
result_info_df = result_info_df.drop(columns='대여소번호', axis=1).rename(columns={'자치구':'대여자치구'})


# In[20]:


result_info_df = pd.merge(result_info_df, rental_place_df, how='left', left_on='반납대여소번호', right_on='대여소번호', suffixes=('_대여', '_대여소'))
result_info_df = result_info_df.rename(columns={'위도':'반납위도','경도':'반납경도'})
result_info_df = result_info_df.drop(columns='대여소번호', axis=1).rename(columns={'자치구':'반납자치구'})


# In[21]:


result_info_df


# In[22]:


result_info_df.info()


# In[23]:


to_float_columns = ['대여위도', '대여경도', '반납위도', '반납경도']
for i in to_float_columns:
    result_info_df[i] = result_info_df[i].astype('float')


# In[24]:


from sklearn.preprocessing import LabelEncoder
to_label_columns = ['대여자치구', '반납자치구', '성별']

encoder = LabelEncoder()
for i in to_label_columns:
    result_info_df[i] = encoder.fit_transform(result_info_df[i])
    print(encoder.classes_)


# In[25]:


from sklearn.cluster import KMeans

# k 개수
x = []

# 응집도
y = []

for k in range(1, 10):
    k_means = KMeans(n_clusters=k, random_state=124, n_init=10)
    k_means.fit(result_info_df)
    
    x.append(k)
    y.append(k_means.inertia_)
    
plt.plot(x, y)


# In[26]:


k_means = KMeans(n_clusters=3, random_state=124, n_init=10)
k_means.fit_predict(result_info_df)
result_info_df['cluster'] = k_means.labels_
result_info_df


# In[27]:


result_info_df.describe().T


# In[28]:


# centroids = k_means.cluster_centers_

# centroids_x = centroids[:, 0]
# centroids_y = centroids[:, 1]

# sns.scatterplot(x='대여위도', y='대여경도', hue='cluster', data=result_info_df)
# sns.scatterplot(x=centroids_x, y=centroids_y, color='red')


# In[29]:


# result_info_df[result_info_df['cluster'] == 0].head(15)


# In[30]:


# result_info_df[result_info_df['cluster'] == 1].head(15)


# In[31]:


result_info_df[result_info_df['cluster'] == 2].head(15)


# In[32]:


display(result_info_df[result_info_df['cluster'] == 0].describe().T)
display(result_info_df[result_info_df['cluster'] == 1].describe().T)
display(result_info_df[result_info_df['cluster'] == 2].describe().T)


# In[33]:


### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
import numpy as np
def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0, n_init=10)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")


# In[126]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaled_df = pd.DataFrame()

scaler = StandardScaler()
columns = result_info_df.columns[:-1].to_list()
for i in columns:
    scaled_df[columns] = scaler.fit_transform(result_info_df[columns])

pca = PCA(n_components=2)
pca_df = pca.fit_transform(result_info_df)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())


# In[47]:


pca_columns=[f'pca{i+1}' for i in range(2)]
pca_df = pd.DataFrame(pca_df, columns=pca_columns)
pca_df.head(10)


# In[50]:


k_means = KMeans(n_clusters=3, random_state=124, n_init=10)
k_means.fit_predict(pca_df)
pca_df['cluster'] = k_means.labels_
pca_df


# In[51]:


centroids = k_means.cluster_centers_

centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

sampled_data = pca_df.groupby('cluster').apply(lambda group: group.sample(n=1000))

sns.scatterplot(x=centroids_x, y=centroids_y, color='red')
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=sampled_data, palette='viridis', alpha=0.5)


# In[53]:


import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.title("Dendograms")
dend = shc.dendrogram(shc.linkage(sampled_data, method='ward'))


# In[ ]:


# visualize_silhouette([2, 3, 4, 5], pca_df.sample(3000))


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 3차원 산점도를 그리기 위한 Figure 객체 생성
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 각 클러스터에서 무작위로 샘플을 300개씩 뽑기
sampled_data = pca_df.groupby('cluster').apply(lambda group: group.sample(n=300))

# 클러스터별로 3차원 산점도 그리기
scatter = ax.scatter(sampled_data['pca1'], sampled_data['pca2'], sampled_data['pca3'], c=sampled_data['cluster'], cmap='viridis',s=80, alpha=0.8)

# 축 이름 설정
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')

# 범례 표시
legend = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend)

# 그래프 표시
plt.show()


# In[ ]:


sampled_data


# In[54]:


result_info_df


# In[ ]:


# centroids = k_means.cluster_centers_

# centroids_x = centroids[:, 0] 
# centroids_y = centroids[:, 1] 

sampled_data = result_info_df.groupby('cluster').apply(lambda group: group.sample(n=300))

sns.scatterplot(x='이용거리(M)', y='성별', hue='cluster', data=sampled_data)
# sns.scatterplot(x=centroids_x, y=centroids_y, color='red')


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 3차원 산점도를 그리기 위한 Figure 객체 생성
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 각 클러스터에서 무작위로 샘플을 300개씩 뽑기

# 클러스터별로 3차원 산점도 그리기
scatter = ax.scatter(sampled_data['대여자치구'], sampled_data['반납자치구'], sampled_data['이용거리(M)'], c=sampled_data['cluster'], cmap='viridis',s=80, alpha=0.8)

# 축 이름 설정
ax.set_xlabel('대여자치구')
ax.set_ylabel('반납자치구')
ax.set_zlabel('이용거리(M)')

# 범례 표시
legend = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend)

# 그래프 표시
plt.show()


# In[ ]:


result_info_df


# In[124]:


geo_df = result_info_df.drop(columns=['대여 대여소번호','반납대여소번호','이용시간(분)','이용거리(M)', '성별', 'cluster','대여자치구','반납자치구','반납위도','반납경도'])


# In[131]:


k_means = KMeans(n_clusters=7, n_init=10)
k_means.fit_predict(geo_df)
geo_df['cluster'] = k_means.labels_
geo_df


# In[132]:


geo_df.cluster.value_counts()


# In[130]:


k_means = KMeans(n_clusters=6, random_state=124, n_init=10)
k_means.fit_predict(geo_df)
geo_df['cluster'] = k_means.labels_
geo_df


# In[133]:


sampled_geo_data = geo_df.groupby('cluster').apply(lambda group: group.sample(n=1500))


# In[136]:


sns.scatterplot(x='대여위도', y='대여경도', hue='cluster', data=sampled_geo_data, palette='viridis')


# In[123]:


# centroids = k_means.cluster_centers_

# centroids_x = centroids[:, 0]
# centroids_y = centroids[:, 1]

# sns.scatterplot(x=centroids_x, y=centroids_y, color='red')
sns.scatterplot(x='대여위도', y='대여경도', data=sampled_geo_data[sampled_geo_data['cluster']==0], color='red', alpha=0.8)
sns.scatterplot(x='반납위도', y='반납경도', data=sampled_geo_data[sampled_geo_data['cluster']==0], color='blue', alpha=0.8)
# sns.scatterplot(x='대여위도', y='대여경도', data=sampled_geo_data[sampled_geo_data['cluster']==2], color='red')
# sns.scatterplot(x='대여위도', y='대여경도', data=sampled_geo_data[sampled_geo_data['cluster']==3], color='red')
# y축 범위 설정
plt.ylim(126.875, 127.1)

# 그래프 표시
plt.show()


# In[121]:


sns.scatterplot(x='대여위도', y='대여경도', data=sampled_geo_data[sampled_geo_data['cluster']==1], color='yellow', alpha=0.8)
sns.scatterplot(x='반납위도', y='반납경도', data=sampled_geo_data[sampled_geo_data['cluster']==1], color='green', alpha=0.8)
# y축 범위 설정
plt.ylim(126.875, 127.1)

# 그래프 표시
plt.show()


# In[122]:


sns.scatterplot(x='대여위도', y='대여경도', data=sampled_geo_data[sampled_geo_data['cluster']==2], color='orange', alpha=0.8)
sns.scatterplot(x='반납위도', y='반납경도', data=sampled_geo_data[sampled_geo_data['cluster']==2], color='purple', alpha=0.8)
# y축 범위 설정
plt.ylim(126.875, 127.1)

# 그래프 표시
plt.show()


# In[117]:


sampled_geo_data[sampled_geo_data['cluster']==2].describe().T


# In[94]:


sns.scatterplot(x='대여위도', y='대여경도', data=sampled_geo_data[sampled_geo_data['cluster']==3], color='red', alpha=0.8)
sns.scatterplot(x='반납위도', y='반납경도', data=sampled_geo_data[sampled_geo_data['cluster']==3], color='blue', alpha=0.8)


# In[138]:


result_info_df['cluster'] = geo_df['cluster']
result_info_df


# In[149]:


sampled_geo_data = result_info_df.groupby('cluster').apply(lambda group: group.sample(n=1500))

sns.scatterplot(x='이용거리(M)', y='cluster', data = sampled_geo_data ,hue='cluster', color='blue', alpha=0.8, palette='viridis')


# In[ ]:





# In[ ]:





# In[ ]:




