#!/usr/bin/env python
# coding: utf-8

# ### Mean Shift (평균 이동)
# - 데이터 포인트의 밀집된 영역을 찾으려고 시도하는 슬라이딩 윈도우 기반 알고리즘이다.
# > 📌 슬라이딩 윈도우 알고리즘이란, 고정 사이즈의 윈도우가 이동하면서 윈도우 내에 있는 데이터를 이용해 문제를 풀이하는 알고리즘이다.
# > - 일정 범위의 값을 비교할 때 사용하면 매우 유용하며, 최소한의 계산으로 다음 배열의 합을 구하는 방식이다.
# > - 이전 배열의 첫 번째 원소를 빼고 다음에 들어올 원소를 더해주는 것이 최소한의 계산으로 다음 배열의 합을 구하는 방법이다.
# - 각 군집(클러스터)의 중심점을 찾는 것이 목표이며, 중심점 후보를 슬라이딩 윈도우 내 점들의 평균으로 업데이트한다.
# <img src="./images/mean_shift01.gif" width="200" style="margin-left:0">
# 
# - K-평균 클러스터링과 달리 자동으로 중심점을 발견하기 때문에 클러스터 수를 정해놓을 필요가 없다.
# <div style="display: flex">
#     <div>
#         <img src="./images/mean_shift02.gif" width="300" style="margin-left:0">
#     </div>
#     <div>
#         <img src="./images/mean_shift03.gif" width="400" style="margin-left:0">
#     </div>
# </div>
# 
# ##### KDE (Kernel Density Estimation), 커널 밀도 추정
# - 커널 함수를 통해 확률 밀도 함수(Probability Density Function)를 추정하는 방법으로서 대표적으로 가우시안 분포 함수(정규 분포 함수)가 사용된다.
# - 데이터 포인트들(중심점)이 데이터 분포가 높은 곳으로 이동하면서 군집화를 수행한다.
# > - K: 커널 함수
# > - x: 확률 변수 데이터
# > - x<sub>i</sub>: 관측 데이터
# > - h: 대역폭(bandwidth)
# <img src="./images/mean_shift04.png" width="400" style="margin-left:10px">
# <img src="./images/mean_shift05.png" width="400" style="margin-left:10px; margin-bottom: 20px">
# > - 대역폭이 클 수록 개별 커널 함수의 영향력이 작어져서 그래프가 곡선형에 가깝고 너무 크게 설정하면 과소적합의 위험이 있다.
# > - 대역폭이 작을 수록 개별 커널 함수의 영향력이 커져서 그래프가 뾰족해지고 너무 작게 설정하면 과대적합의 위험이 있다.
# <img src="./images/mean_shift06.png" width="400" style="margin-left:10px">

# In[1]:


import pandas as pd

super_car_df = pd.read_csv('./datasets/super_car.csv')
super_car_df


# In[2]:


super_car_df.info()


# In[3]:


super_car_df = super_car_df.iloc[super_car_df[['Car Make', 'Car Model', 'Year']].drop_duplicates().index, :]
super_car_df = super_car_df.reset_index(drop=True)
super_car_df


# In[4]:


super_car_df.isna().sum()


# In[5]:


super_car_df['0-60 MPH Time (seconds)'] = super_car_df['0-60 MPH Time (seconds)'].apply(lambda x: x.replace('< ', ''))
super_car_df['0-60 MPH Time (seconds)'] = super_car_df['0-60 MPH Time (seconds)'].astype('float16')


# In[6]:


super_car_df['Price (in USD)'] = super_car_df['Price (in USD)'].apply(lambda x: x.replace(',', ''))
super_car_df['Price (in USD)'] = super_car_df['Price (in USD)'].astype('int32')


# In[7]:


super_car_df['Torque (lb-ft)'] = super_car_df['Torque (lb-ft)'].astype('int32')


# In[9]:


super_car_df['Horsepower'] = super_car_df['Horsepower'].apply(lambda x: x.replace('1000+', '1000'))
super_car_df['Horsepower'] = super_car_df['Horsepower'].apply(lambda x: x.replace(',', ''))
super_car_df['Horsepower'] = super_car_df['Horsepower'].astype('int16')


# In[10]:


super_car_df['Name'] = super_car_df['Car Make'] + ' ' + super_car_df['Car Model']
super_car_df = super_car_df.drop(columns=['Car Make', 'Car Model', 'Year'], axis=1)
super_car_df


# In[11]:


super_car_df = super_car_df.iloc[super_car_df.Name.drop_duplicates().index, :].reset_index(drop=True)
super_car_df


# In[12]:


super_car_df.info()


# In[13]:


super_car_df.isna().sum()


# In[14]:


super_car_df['Engine Size (L)'].value_counts()


# In[15]:


super_car_df['Engine Size (L)'] = super_car_df['Engine Size (L)'].fillna('Electric')
super_car_df.isna().sum()


# In[16]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
super_car_df['Engine Size (L)'] = encoder.fit_transform(super_car_df['Engine Size (L)'])
print(encoder.classes_)


# In[17]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)

np.random.seed(124)
x = np.random.normal(0, 1, size=30)
print(x)
sns.kdeplot(x, bw_method=0.2, label='bw 0.2')
sns.kdeplot(x, label='bw 1')
sns.kdeplot(x, bw_method=2, label='bw 2')
plt.legend()


# In[18]:


super_car_df


# In[19]:


from sklearn.preprocessing import StandardScaler

scaled_super_car_df = StandardScaler().fit_transform(super_car_df.iloc[:, :-1])
scaled_super_car_df = pd.DataFrame(scaled_super_car_df, columns=super_car_df.iloc[:, :-1].columns)
scaled_super_car_df


# In[23]:


from sklearn.cluster import MeanShift

meanshift= MeanShift(bandwidth=1)
cluster_labels = meanshift.fit_predict(scaled_super_car_df)
# np의 unique는 중복없이 선택할 때 사용한다.
print('cluster labels:', np.unique(cluster_labels))


# ##### estimate_bandwidth(input, quantile=0.3)
# - 내부적으로 최적의 kernel bandwidth를 정하기 위해 KNN 기법을 이용한다.
# - KNN을 수행하는 데이터의 건수를 (전체 데이터 * quantile) 로 정하게 되며, 같은 클러스터 내의 데이터간 평균 거리를 기반으로 bandwidth를 정한다.
# - quantile이 크면 bandwidth 값이 커져서 클러스터 개수가 작아지고, quantile이 작으면 bandwidth 값이 작아져서 클러스터 개수가 많아진다.
# - quantile의 범위는 0 ~ 1사이이다.

# In[24]:


from sklearn.cluster import estimate_bandwidth

bandwidth = estimate_bandwidth(scaled_super_car_df, quantile=0.5)
print('bandwidth 값:', round(bandwidth, 3))


# In[25]:


from sklearn.cluster import MeanShift

meanshift= MeanShift(bandwidth=2.246)
cluster_labels = meanshift.fit_predict(scaled_super_car_df)
print('cluster labels:', np.unique(cluster_labels))


# In[26]:


super_car_df['cluster'] = cluster_labels
super_car_df


# In[27]:


super_car_df.cluster.value_counts()


# In[28]:


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


# In[30]:


from sklearn.datasets import load_iris

visualize_silhouette([3, 4, 5, 7], scaled_super_car_df)


# In[29]:


fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.scatterplot(x='Horsepower', y='Engine Size (L)', hue='cluster', data=super_car_df, ax=ax[0])
sns.scatterplot(x='Horsepower', y='Price (in USD)', hue='cluster', data=super_car_df, ax=ax[1])
sns.scatterplot(x='Torque (lb-ft)', y='Price (in USD)', hue='cluster', data=super_car_df, ax=ax[2])

