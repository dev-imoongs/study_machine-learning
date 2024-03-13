#!/usr/bin/env python
# coding: utf-8

# ### DBSCAN (Density-based spatial clustering of applications with noise), 밀도기반 공간 군집화
# - K-Means와 Hierarchical Clustering은 군집간의 거리를 이용하지만, DBSCAN은 데이터가 몰려 있어서 밀도가 높은 부분을 이용하는 방식이다.
# - 임의의 점 p가 있을 때,  
# 이 점(p)부터 다른 점까지의 거리 e(epsilon)내에 다른 점이 m(minPts)개 있으면 하나의 군집으로 인식하고, 이 때 p를 중심점(core point)이라고 한다.
# - DBSCAN 사용 시, 기준점 부터의 거리 epsilon값과, 이 반경내에 있는 점의 수 minPts를 전달해야 한다.
# - 클러스터의 밀도에 따라서 클러스터를 서로 연결하기 때문에 기하학적인 모양을 갖는 군집도 잘 찾는다.
# <img src="./images/DBSCAN.gif" width="500" style="margin-left: -30px">

# ##### 학생의 수준 분석 데이터
# - STG: 목표물 소재 연구시간 정도
# - SCG: 목표물 재료의 사용자 반복수
# - STR: 목표대상이 있는 관련 대상에 대한 사용자의 학습시간 정도
# - LPR: 목표대상이 있는 관련 대상에 대한 사용자의 시험성적
# - PEG: 목표물에 대한 사용자의 시험성적
# - UNS: 사용자의 지식수준 (매우 낮음, 낮음, 중간, 높음)

# In[2]:


import pandas as pd

# conda install -c anaconda xlrd
student_df = pd.read_excel('./datasets/student.xls')
student_df


# In[3]:


student_df.isna().sum()


# In[4]:


student_df.duplicated().sum()


# In[5]:


student_df.describe().T


# In[6]:


student_df = student_df.rename(columns={' UNS': 'UNS'})
student_df


# In[7]:


student_df.UNS.value_counts()


# In[8]:


student_df.UNS = student_df.UNS.apply(lambda x: 'Very Low' if x == 'very_low' else x)
student_df.UNS.value_counts()


# In[9]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=124)
pca_df = pca.fit_transform(student_df.iloc[:, :-1])

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())


# In[10]:


pca_columns=[f'pca{i+1}' for i in range(2)]
pca_df = pd.DataFrame(pca_df, columns=pca_columns)
pca_df


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x='pca1', y='pca2', data=pca_df)


# In[22]:


import numpy as np

### 클러스터 결과를 담은 DataFrame과 사이킷런의 Cluster 객체등을 인자로 받아 클러스터링 결과를 시각화하는 함수  
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_
        
    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*']
    isNoise=False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name]==label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise=True
        else :
            cluster_legend = 'Cluster '+str(label)
        
        plt.scatter(x=label_cluster['pca1'], y=label_cluster['pca2'], s=70,\
                    edgecolor='k', marker=markers[label], label=cluster_legend)
        
        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',\
                        edgecolor='k', marker='$%d$' % label)
    legend_loc='upper right'
    
    plt.legend(loc=legend_loc)
    plt.show()


# In[13]:


# !pip install mglearn


# In[14]:


import mglearn

mglearn.plots.plot_dbscan()


# In[19]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.07, min_samples=15, metric='euclidean')
dbscan_labels = dbscan.fit_predict(pca_df)

pca_df['cluster'] = dbscan_labels
pca_df


# In[20]:


pca_df.cluster.value_counts()


# In[23]:


visualize_cluster_plot(dbscan, pca_df, 'cluster', iscenter=False)


# In[24]:


student_df['cluster'] = pca_df.cluster
student_df


# In[25]:


student_df.cluster.value_counts()


# In[26]:


fig, ax = plt.subplots(2, 3, figsize=(10, 8))
sns.histplot(x='UNS', hue='cluster', multiple='dodge', data=student_df[student_df.cluster != -1], ax=ax[0][0])
sns.histplot(x='STG', hue='cluster', multiple='dodge', data=student_df[student_df.cluster != -1], ax=ax[0][1])
sns.histplot(x='SCG', hue='cluster', multiple='dodge', data=student_df[student_df.cluster != -1], ax=ax[1][2])
sns.histplot(x='LPR', hue='cluster', multiple='dodge', data=student_df[student_df.cluster != -1], ax=ax[1][0])
sns.histplot(x='PEG', hue='cluster', multiple='dodge', data=student_df[student_df.cluster != -1], ax=ax[1][1])


# - STG: 목표물 소재 연구시간 정도
# - SCG: 목표물 재료의 사용자 반복수
# - STR: 목표대상이 있는 관련 대상에 대한 사용자의 학습시간 정도
# - LPR: 목표대상이 있는 관련 대상에 대한 사용자의 시험성적
# - PEG: 목표물에 대한 사용자의 시험성적
# - UNS: 사용자의 지식수준 (매우 낮음, 낮음, 중간, 높음)

# ##### 🚩결론: 시험 성적이 높아도 모든 부분의 점수를 높게 받지는 않는다.
