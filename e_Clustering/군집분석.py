#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/naom99/IT22/blob/main/w15_%EA%B8%B0%EB%A7%90%EB%AF%B8%EB%8B%88%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8_%EA%B5%B0%EC%A7%91%EB%B6%84%EC%84%9D.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


#패키지 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#폰트 설정
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')


# In[9]:


spotify_df = pd.read_excel('./datasets/pro/Spotify 2010 - 2019 Top 100 Songs.xlsx') 
spotify_df


# ### **데이터 전처리0. charts.csv 전처리**
# 
# charts.csv: 1958년 8월 4일부터 2021년 11월 6일까지 매 주차 스포티파이 Top100 차트에 랭크된 곡에 대한 정보를 포함하고 있다. top100 아티스트와 곡 정보를 추출한 뒤 다른 데이터와 연결해 top100 차트인 여부를 판단하는 새로운 변수 top100 만들 때 활용하고자 한다
# 
# *   week: 해당 월 주차 수
# *   date: 날짜
# *   rank: 순위
# *   song: 곡명
# *   artist: 아티스트명
# *   peak_rank: 최고 순위
# *   weeks-on-board: Top100 차트인 기간

# In[5]:


#데이터 불러오기
charts = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Bigmining/charts_new.csv",  encoding='cp949')


# In[6]:


#데이터 탐색하기
display(charts.head())


# In[7]:


#데이터 탐색하기2
charts.describe


# 0-1) 적정한 기간 설정: 2000년 1월 ~ 2019년 12월까지 10년간의 데이터 추출

# In[8]:


#date를 인덱스로 설정
charts = charts.set_index('date')

#기간에 맞는 행만 추출해 charts에 저장
charts = charts['2019-12-31':'2000-01-01']

#artist를 기준으로 중복되는 행 제거
charts.drop_duplicates(['artist'])


# **0-2) 데이터 분석에 사용할 변수만 추출: song, artist,rank**

# In[9]:


#칼럼 추출하기
charts = charts.loc[:,['song','artist','rank']]

display(charts.head())


# ### **데이터 전처리1.spotify_artist 데이터 만들기**
# 
# feature_corel_100.csv는 12904명의 스포티파이 아티스트에 대한 통계적 정보를 보여준다. charts 데이터와 연결해 해당 아티스트가 2000~2019년 스포티파이 top100에 포함된 적이 있는 지를 판단하고자 한다.
# 
# 
# *   name: 아티스트명
# *   popularity: 아티스트 좋아요 수
# *   followers: 아티스트 팔로워 수
# *   num_release: 발매한 노래 수
# *   pop_mean:발매한 노래의 좋아요 수 평균
# *   fol_mean:?
# *   rel_mean:?
# 

# In[10]:


#feature_corel_100.csv파일 불러오기
feature = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Bigmining/feature_corel_100.csv")


# In[11]:


#데이터 확인
print(feature)


# In[12]:


feature = feature.loc[:,['name','popularity','followers','num_release','pop_mean','rel_mean','fol_mean']]


# In[13]:


#name을 artist로 변수명 바꾸기
feature.rename(columns = {'name':'artist'},inplace=True)
display(feature.head())


# 1-3) charts와 feature를 합친 새로운 데이터프레임 spotify_artist 만들기

# In[14]:


#charts 데이터 복사하기
charts_1 = charts.copy()

#artist를 기준으로 데이터 합치기
spotify_artist = pd.merge(feature,charts_1,how='left',on='artist')
display(spotify_artist)


# 1-4) spotify_artist 전처리: 새로운 변수 top-100 만들기

# In[15]:


#새로운 열 추가
spotify_artist["top100"]=spotify_artist["rank"]

#조건에 맞는 값을 칼럼에 추가
spotify_artist['top100']=spotify_artist['top100'].fillna("No")
spotify_artist.loc[(spotify_artist['top100']!="No"),'top100']="Yes"


# 1-5) spotify_artist 전처리: 불필요한 변수와 중복되는 행 제거

# In[17]:


#불필요한 변수 제거
spotify_artist = spotify_artist.drop(columns = ['song','rank'],axis=0)

#artist를 기준으로 중복되는 행 제거
spotify_artist.drop_duplicates(['artist'])


# ### **데이터 전처리2. spotify_song 데이터 만들기**
# 
# songs_normalize.csv는 2000년부터 2019년까지 스포티파이에서 가장 인기있던 2000개의 곡에 대한 정보를 보여준다. charts 데이터와 연결해 해당 곡이 top100 차트에 오른 적이 있었는지를 판단하고자 한다. 
# 
# * popularity: 노래의 좋아요 수
# * danceability
# * energy
# * key
# * loudness
# * speechiness
# * acousticness
# * instrumentalness
# * liveness
# * valence
# * tempo
# * genre
# * top100

# In[18]:


#songs_normalize.csv파일 불러오기
songs = pd.read_csv("/content/drive/My Drive/Colab Notebooks/ITB 2022/songs_normalize.csv")

#데이터 탐색하기
display(songs.head())


# 2-3) charts와 songs를 합친 새로운 데이터프레임 spotify_song 만들기

# In[19]:


#charts 복사하기
charts_2 = charts.copy()

#'song'을 기준으로 데이터 합치기
spotify_song = pd.merge(songs,charts_2,how='left',on='song')
display(spotify_song)


# 2-4) spotify_song 전처리: 새로운 변수 top-100 만들기

# In[20]:


#새로운 열 추가
spotify_song["top100"]=spotify_song["rank"]

#조건에 맞는 값을 칼럼에 추가
spotify_song['top100']=spotify_song['top100'].fillna("No")
spotify_song.loc[(spotify_song['top100']!="No"),'top100']="Yes"

display(spotify_song)


# 2-5) spotify_song 전처리: 불필요한 변수와 중복되는 행 제거

# In[21]:


#불필요한 변수 제거
spotify_song = spotify_song.drop(columns = ['artist_x','artist_y','rank','duration_ms','explicit','year','mode'],axis=0)

#중복되는 행 제거
spotify_song.drop_duplicates(['song'])


# ##**Step 3. 군집분석**

# ### **1) spotify_song 파일 탐색**

# In[ ]:


# 간단하게 그림을 그릴 수 있는 mglearn 라이브러리 사용 (!pip install mglearn 명령어로 설치)
get_ipython().system('pip install mglearn')
get_ipython().system('pip install --upgrade joblib==1.1.0')
import mglearn

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns           # Seaborn 로딩하기
import matplotlib.pyplot as plt # Matplotlib의 pyplot 로딩하기


# In[ ]:


#결측치 확인
spotify_song.isnull()


# In[ ]:


spotify_song.head()


# ### **2) k-means 군집 분석**

# In[ ]:


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# **문자데이터와 중복되는 데이터를 제외하고 X값 지정**

# In[ ]:


X = spotify_song.drop(['song','genre', 'top100'], axis=1)


# **옐보우 방법을 사용하여 최적의 군집 개수 찾기**

# In[ ]:


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


# **k-means 군집 분석**

# In[ ]:


from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_samples, silhouette_score 


# **세 개의 클러스터 중심 사용**

# In[ ]:


km3 = KMeans(n_clusters=3, random_state=0).fit(X)
labels = km3.labels_

mglearn.discrete_scatter(X['tempo'], X['popularity'],labels , markers='o')
plt.xlabel("tempo")
plt.ylabel("popularity")
plt.show()


# In[ ]:


#k=3일 경우, 실루엣 계수값 구하기
average_score = silhouette_score(X, labels)
print('모든 데이터의 평균 실루엣 계수값:', format(average_score)) 


# **네 개의 클러스터 중심 사용**

# In[ ]:


km4 = KMeans(n_clusters=4, random_state=0).fit(X)
labels = km4.labels_

mglearn.discrete_scatter(X['tempo'], X['popularity'],labels , markers='o')
plt.xlabel("tempo")
plt.ylabel("popularity")
plt.show()


# In[ ]:


#k=4일 경우, 실루엣 계수값 구하기
average_score = silhouette_score(X, labels)
print('모든 데이터의 평균 실루엣 계수값:', format(average_score))


# **옐보우 방법과 실루엣 계수를 참고하여 클러스터 개수는 3으로 선택**

# ### **3) 클러스터 유형 파악하기**

# In[ ]:


spotify_song['cluster'] = labels
spotify_song.head()


# In[ ]:


#그룹별 개수
spotify_song.groupby('cluster').count()


# In[ ]:


#그룹별 평균값
spotify_song.groupby('cluster').mean()


# ### **4) 군집분석 인사이트 도출**
# 
# 1. spotify_song 파일은 tempo와 popularity에 근거해 3개의 군집으로 나눌 수 있다.
# 
# *   0번 군집: 인기도가 낮은 그룹
# *   1번 군집: 템포가 낮고 인기도가 높은 그룹. Speechiness(곡 안에서 목소리가 차지하는 비중)가 높은 편
# *   2번 군집: 템포가 높고 인기도가 높은 그룹. Energy(힘)가 높은 편
# 
# 2. 인기도가 높은 군집 1,2는 인기도가 낮은 군집 0보다 모두 key와 liveness(현장감)수치가 높고 instrumentalness(기악성) 수치가 낮다
# 
# 
# 
# **따라서, 음원 제작시**
# 
# 1.   key와 liveness 수치는 높이고 기악성은 낮추는 것을 추천
# 2.   템포가 낮은 음악은 speechiness의 수치를 높이는 것을 추천
# 3.   템포가 높은 음악은 energy가 돋보이게 작곡하는 것을 추천
# 
# 
# 
# 
# 
# 
