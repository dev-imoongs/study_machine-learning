#!/usr/bin/env python
# coding: utf-8

# ### Dendrogram
# 
# ##### scipy.cluster.hierarchy
# - 계층적 군집화(Hierarchical Clustering)는 Tree기반의 모델이다.
# - 2차원 데이터를 통해 군집화를 시각화할 수 있기 때문에, 다차원 데이터에서 2개의 feature를 선택하거나 차원 축소화를 진행한다.
# - 데이터 사이의 거리를 구할 때 거리 계산 방법으로서, 중심연결, 단일연결, 완전연결, 평균연결, 와드연결 등이 있고, 하이퍼 파라미터로 선택이 가능하다.
# <img src="./images/dendrogram.png" width="400" style="margin-left: 0">
# 
# ##### sklearn.cluster.AgglomerativeClustering
# - 계층적 군집을 반복하여 만들고 모든 데이터 포인트들이 하나의 포인트를 가진 클러스터에서 마지막 클러스터까지 이동하여 최종 군집이 형성된다.
# - 최초의 중앙점으로 부터 포인트들이 증가해가며 군집이 커지는 모양을 가진다.
# - 데이터 사이의 거리를 구할 때 거리 계산 방법으로서, 중심연결, 단일연결, 완전연결, 평균연결, 와드연결 등이 있고, 하이퍼 파라미터로 선택이 가능하다.
# <img src="./images/agglomerative_clustering.gif" width="400" style="margin-left: 0">

# In[1]:


import pandas as pd

mall_df = pd.read_csv('./datasets/mall.csv')
mall_df


# In[2]:


mall_df.isna().sum()


# In[3]:


mall_df.duplicated().sum()


# In[4]:


mall_df.describe().T


# In[5]:


mall_df.info()


# In[6]:


from sklearn.preprocessing import StandardScaler

scaled_mall_df = StandardScaler().fit_transform(mall_df.iloc[:, 2:])
scaled_mall_df = pd.DataFrame(scaled_mall_df, columns=mall_df.iloc[:, 2:].columns)


# In[7]:


scaled_mall_df


# In[8]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_df = pca.fit_transform(scaled_mall_df)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())


# In[9]:


pca_columns=[f'pca{i+1}' for i in range(2)]
pca_df = pd.DataFrame(pca_df, columns=pca_columns)
pca_df


# In[10]:


import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.title("Mall Dendograms")
dend = shc.dendrogram(shc.linkage(pca_df, method='ward'))


# In[11]:


import numpy as np
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
cluster.fit_predict(scaled_mall_df)
np.unique(cluster.labels_)


# In[12]:


pca_df['cluster'] = cluster.labels_


# In[13]:


import seaborn as sns

sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=pca_df)


# In[14]:


mall_df['cluster'] = pca_df.cluster
mall_df


# In[15]:


fit, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(x='Age', hue='cluster', data=mall_df, multiple='dodge', ax=ax[0])
sns.scatterplot(x='Age', y='Annual Income (k$)', hue='cluster', data=mall_df, ax=ax[1])
sns.histplot(x='Spending Score (1-100)', hue='cluster', data=mall_df, multiple='dodge', ax=ax[2])


# ##### 🚩결론: 20대~40대까지 연 소득에 상관없이 쇼핑센터 이용을 많이 하고, 40대부터는 쇼핑센터 이용이 감소한다.

# In[16]:


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
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
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


# In[17]:


visualize_silhouette([2, 3, 4, 5, 6], pca_df)

