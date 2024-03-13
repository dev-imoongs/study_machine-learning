#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔 글꼴 경로 설정
font_path = "C:\Windows\Fonts\malgun.ttf"s

# 한글 폰트 설정
font_name = fm.FontProperties(fname=font_path, size=10).get_name()
plt.rc('font', family=font_name)


# In[3]:


path = './datasets/pro/complete_dataset.csv'
df = pd.read_csv(path)
df


# In[11]:


df['date'] = pd.to_datetime(df['date'],dayfirst=False)


# In[4]:


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# StandardScaler를 사용하여 데이터 스케일링
scaler = StandardScaler().fit_transform(df.iloc[:,1:-2])
scaled_df = pd.DataFrame(data=scaler, columns=df.columns[1:-2])

columns = ['RRP','min_temperature','max_temperature']

for column in columns:
    scaled_df = scaled_df[scaled_df[column].between(-1.96,1.96)]
scaled_df


# In[53]:


# 데이터프레임에서 'date', 'RRP', 'min_temperature' 및 'max_temperature' 컬럼을 추출
date = df.iloc[scaled_df.index,:]['date']




# 그래프 생성
plt.figure(figsize=(12, 6))  # 그래프 크기 설정

# RRP 라인 플롯
plt.plot(date, scaled_df['demand'], label='demand', linestyle='-', markersize=4, alpha=0.5)

# max_temperature 라인 플롯
plt.plot(date, scaled_df['solar_exposure'], label='solar_exposure', linestyle='-', markersize=4, alpha=0.5)

plt.xlabel('Date')  # x-축 레이블
plt.ylabel('Scaled Value')  # y-축 레이블
plt.title('Scaled RRP and Temperature Over Time')  # 그래프 제목
plt.legend()  # 범례 표시
plt.grid(True)  # 그리드 표시

plt.xticks(rotation=45)  # x-축의 날짜 레이블을 45도 회전하여 가독성 향상

plt.tight_layout()  # 그래프가 잘리지 않도록 레이아웃 조정

plt.show()  # 그래프 표시


# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt

# 그래프 생성
plt.figure(figsize=(12, 6))  # 그래프 크기 설정

# lineplot 그리기
plot = sns.lineplot(x='date', y='RRP', data=df)

# y-축 범위 설정
plot.set(ylim=(0, 200))

plt.xlabel('Date')  # x-축 레이블
plt.ylabel('RRP')  # y-축 레이블
plt.title('RRP Over Time')  # 그래프 제목

plt.xticks(rotation=45)  # x-축의 날짜 레이블을 45도 회전하여 가독성 향상

plt.tight_layout()  # 그래프가 잘리지 않도록 레이아웃 조정

plt.show()  # 그래프 표


# In[48]:


plt.figure(figsize=(12, 6))  # 그래프 크기 설정
sns.lineplot(x='date',y='max_temperature',data=df)
plt.show()  # 그래프 표


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = df.iloc[:,:-2].corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
heatmap.set_title("Correlation")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




