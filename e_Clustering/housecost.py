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


# In[3]:


path = 'http://scottyoon.cafe24.com/data/data.csv'

df = pd.read_csv(path)


# In[4]:


display(df)


# In[4]:


print(spotify_df.info())
print('='*30)
print(spotify_df.isna().sum())
print('='*30)
print(spotify_df.duplicated().sum())


# In[5]:


spotify_df = spotify_df[spotify_df.released_year > 2000].reset_index(drop=True)


# In[6]:


spotify_df['streams'] = spotify_df['streams'].replace(',', '', regex=True)
spotify_df['in_apple_charts'] = spotify_df['in_apple_charts'].replace(',', '', regex=True)
spotify_df['in_shazam_charts'] = spotify_df['in_shazam_charts'].replace(',', '', regex=True)
spotify_df['in_deezer_playlists'] = spotify_df['in_deezer_playlists'].replace(',', '', regex=True)


# In[7]:


spotify_df.in_shazam_charts.fillna(0, inplace=True)


# In[8]:


spotify_df['streams'] = spotify_df['streams'].astype('int64')
spotify_df['in_apple_charts'] = spotify_df['in_apple_charts'].astype('int64')
spotify_df['in_shazam_charts'] = spotify_df['in_shazam_charts'].astype('int64')
spotify_df['in_deezer_playlists'] = spotify_df['in_deezer_playlists'].astype('int64')


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

columns = ['key','mode']

for column in columns:
    spotify_df[column] = encoder.fit_transform(spotify_df[column])
    print(encoder.classes_)


# In[10]:


spotify_df.describe().T


# In[11]:


spotify_df.info()


# In[ ]:


['song','popularity','danceability','energy','key','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','genre']


# ### **2) k-means 군집 분석**

# In[12]:


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# In[13]:


# 곡 자체의 속성으로만 군집화 하기 위해 아티스트 이름을 drop
X = spotify_df.drop(columns=['track_name','artist(s)_name'], axis=0)


# In[14]:


clusters =[]
for i in range(1, 10):
    km = KMeans(n_clusters=i, random_state=0).fit(X)
    clusters.append(km.inertia_)
    
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(x=list(range(1, 10)), y=clusters, ax=ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
plt.show()


# In[15]:


from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples, silhouette_score 


# In[21]:


new_X = X


# In[22]:


k_means = KMeans(n_clusters=3, random_state=124)
k_means.fit_predict(new_X)
new_X['cluster'] = k_means.labels_
new_X


# In[24]:


# 간단하게 그림을 그릴 수 있는 mglearn 라이브러리 사용 (!pip install mglearn 명령어로 설치)
get_ipython().system('pip install mglearn')
get_ipython().system('pip install --upgrade joblib==1.1.0')
import mglearn

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns           # Seaborn 로딩하기
import matplotlib.pyplot as plt # Matplotlib의 pyplot 로딩하기


# In[29]:


km3 = KMeans(n_clusters=3, random_state=0).fit(X)
labels = km3.labels_

song_ability = ['danceability_%','valence_%','energy_%','acousticness_%','instrumentalness_%','liveness_%','speechiness_%']

for i in song_ability:
    mglearn.discrete_scatter(X[i], X['in_spotify_charts'],labels , markers='o')
    plt.xlabel(i)
    plt.ylabel("in_spotify_charts")
    plt.show()



# In[30]:


km3 = KMeans(n_clusters=2, random_state=0).fit(X)
labels = km3.labels_

song_ability = ['bpm','danceability_%','valence_%','energy_%','acousticness_%','instrumentalness_%','liveness_%','speechiness_%']

for i in song_ability:
    mglearn.discrete_scatter(X[i], X['in_spotify_charts'],labels , markers='o')
    plt.xlabel(i)
    plt.ylabel("in_spotify_charts")
    plt.show()



# In[31]:


km3 = KMeans(n_clusters=4, random_state=0).fit(X)
labels = km3.labels_

song_ability = ['danceability_%','valence_%','energy_%','acousticness_%','instrumentalness_%','liveness_%','speechiness_%']

for i in song_ability:
    mglearn.discrete_scatter(X[i], X['in_spotify_charts'],labels , markers='o')
    plt.xlabel(i)
    plt.ylabel("in_spotify_charts")
    plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




