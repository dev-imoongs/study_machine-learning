#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔 글꼴 경로 설정
font_path = "C:\Windows\Fonts\malgun.ttf"

# 한글 폰트 설정
font_name = fm.FontProperties(fname=font_path, size=10).get_name()
plt.rc('font', family=font_name)


# In[3]:


path = './datasets/pro/complete_dataset.csv'
df = pd.read_csv(path)
df


# In[4]:


df['date'] = pd.to_datetime(df['date'],dayfirst=False)


# In[5]:


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


# In[6]:


# 데이터프레임에서 'date', 'RRP', 'min_temperature' 및 'max_temperature' 컬럼을 추출
date = df.iloc[scaled_df.index,:]['date']

# 그래프 생성
plt.figure(figsize=(12, 6))  # 그래프 크기 설정

# RRP 라인 플롯
plt.plot(date, scaled_df['demand'], label='demand', linestyle='-', markersize=4, color='#333333')

# max_temperature 라인 플롯
plt.plot(date, scaled_df['solar_exposure'], label='solar_exposure', linestyle='-', markersize=4, color='#999999')

plt.xlabel('Date')  # x-축 레이블
plt.ylabel('Scaled Value')  # y-축 레이블
plt.title('Scaled demand and Solar exposure')  # 그래프 제목
plt.legend()  # 범례 표시
plt.grid(True)  # 그리드 표시

plt.xticks(rotation=45)  # x-축의 날짜 레이블을 45도 회전하여 가독성 향상

plt.tight_layout()  # 그래프가 잘리지 않도록 레이아웃 조정

plt.show()  # 그래프 표시


# In[7]:


# 데이터프레임에서 'date', 'RRP', 'min_temperature' 및 'max_temperature' 컬럼을 추출
date = df.iloc[scaled_df.index,:]['date']

# 그래프 생성
plt.figure(figsize=(12, 6))  # 그래프 크기 설정

# RRP 라인 플롯
plt.plot(date, scaled_df['demand'], label='demand', linestyle='-', markersize=4, alpha=0.6)

# max_temperature 라인 플롯
plt.plot(date, scaled_df['RRP']*2, label='RRP', linestyle='-', markersize=4, alpha=0.6)

plt.xlabel('Date')  # x-축 레이블
plt.ylabel('Scaled Value')  # y-축 레이블
plt.title('Scaled demand and RRP')  # 그래프 제목
plt.legend()  # 범례 표시
plt.grid(True)  # 그리드 표시

plt.xticks(rotation=45)  # x-축의 날짜 레이블을 45도 회전하여 가독성 향상

plt.tight_layout()  # 그래프가 잘리지 않도록 레이아웃 조정

plt.show()  # 그래프 표시


# In[8]:


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


# In[9]:


plt.figure(figsize=(12, 6))  # 그래프 크기 설정
sns.lineplot(x='date',y='max_temperature',data=df)
plt.show()  # 그래프 표


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = df.iloc[:,:-2].corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='Greys')
heatmap.set_title("Correlation")


# In[11]:


print('='*40)
print(df.info())
print('='*40)
print(df.isna().sum())
print('='*40)
print(df.duplicated().sum())
print('='*40)


# In[12]:


df.describe().T


# In[13]:


df


# In[14]:


from sklearn.preprocessing import StandardScaler
scaled_data = StandardScaler().fit_transform(df.drop(columns=['date','school_day','holiday']))
scaled_df = pd.DataFrame(data=scaled_data,columns=df.drop(columns=['date','school_day','holiday']).columns)
scaled_df


# In[15]:


filtered_df = scaled_df[(scaled_df >= -1.96) & (scaled_df <= 1.96)]
new_df = df.iloc[filtered_df.dropna().index,:]
new_df


# In[16]:


filtered_df.describe().T


# In[17]:


df.loc[:,['min_temperature','max_temperature','solar_exposure','rainfall','holiday','demand']].hist(figsize=(10,10),bins=50)


# In[18]:


# 3행 4열로 히스토그램 배치
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, column in enumerate(new_df.loc[:,['min_temperature','max_temperature','solar_exposure','rainfall','holiday','demand']]):
    row = i // 3
    col = i % 3
    new_df[column].hist(bins=50, ax=axes[row, col])
    axes[row, col].set_title(column)

plt.tight_layout()  # 그래프 간 간격 조절
plt.show()


# In[19]:


new_df.reset_index(drop=True,inplace=True)
new_df


# In[20]:


new_df.iloc[:,1:].hist(figsize=(12,10),bins=50)


# In[21]:


df.iloc[:,1:].hist(figsize=(12,10),bins=50)


# In[22]:


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error

def get_evaluation(y_test, prediction):
    MAE =  mean_absolute_error(y_test, prediction)
    MSE = mean_squared_error(y_test, prediction)
    RMSE = np.sqrt(MSE)
    MSLE = mean_squared_log_error(y_test, prediction)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, prediction))
    R2 = r2_score(y_test, prediction)

    print('MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, MSLE: {:.4f}, RMSLE: {:.4f}, R2: {:.4f}'.format(MAE, MSE, RMSE, MSLE, RMSLE, R2))


# In[23]:


from sklearn.preprocessing import LabelEncoder

columns = ['school_day','holiday']

for column in columns:
    encoder = LabelEncoder()
    targets = encoder.fit_transform(df[column])
    df.loc[:, column] = targets
    print(f'{column}_classes: {encoder.classes_}')


# In[24]:


df.dropna(inplace=True)


# In[25]:


df.fillna(0, inplace=True)  # NaN 값을 0으로 대체


# In[26]:


import matplotlib.pyplot as plt
# conda install -c conda-forge seaborn   (0.12.2 이상)
import seaborn as sns

sns.pairplot(df)
plt.show()


# In[27]:


df.school_day = df.school_day.astype('int')
df.holiday = df.holiday.astype('int')


# In[28]:


df.info()


# In[29]:


import statsmodels.api as sm
model = sm.OLS(df[['demand']], df[['min_temperature', 'RRP', 'max_temperature', 'solar_exposure', 'rainfall', 'school_day', 'holiday']])
print(model.fit().summary())


# In[30]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def feature_engineering_VIF(features):
    vif = pd.DataFrame()
    vif['vif_score'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    vif['feature'] = features.columns
    return vif


# In[31]:


print(feature_engineering_VIF(df.drop(columns=['date','demand'])))


# In[32]:


print(feature_engineering_VIF(df[['RRP','RRP_negative', 'min_temperature', 'solar_exposure', 'rainfall', 'school_day','holiday']]))


# In[33]:


model = sm.OLS(df[['demand']], df[['RRP', 'min_temperature', 'solar_exposure', 'rainfall']])
print(model.fit().summary())


# In[34]:


print(feature_engineering_VIF(df[['RRP', 'min_temperature', 'solar_exposure', 'rainfall']]))


# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

features, targets = df[['demand', 'min_temperature', 'max_temperature', 'solar_exposure']], df.RRP
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3)


# In[36]:


y_train[y_train<0] = 0


# In[37]:


y_test[y_test<0] = 0


# In[38]:


import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR



scale = StandardScaler()

X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

y_train = np.log1p(y_train)

#하이퍼파라미터 그리드
dt_param_grid = {
    'max_depth': [4, 6, 8, 10],  # 튜닝하려는 max_depth 후보값
    'min_samples_split': [2, 5, 10],  # 튜닝하려는 min_samples_split 후보값
}

rf_param_grid = {
    'n_estimators': [100, 500, 1000, 3000],  # 튜닝하려는 n_estimators 후보값
    'max_depth': [4, 6, 8, 10],  # 튜닝하려는 max_depth 후보값
}

gb_param_grid = {
    'n_estimators': [100, 500, 1000, 3000],  # 튜닝하려는 n_estimators 후보값
    'learning_rate': [0.01, 0.1, 0.2],  # 튜닝하려는 learning_rate 후보값
    'max_depth': [4, 6, 8, 10],  # 튜닝하려는 max_depth 후보값
}

xgb_param_grid = {
    'n_estimators': [100, 500, 1000, 3000],  # 튜닝하려는 n_estimators 후보값
    'learning_rate': [0.01, 0.1, 0.2],  # 튜닝하려는 learning_rate 후보값
    'max_depth': [4, 6, 8, 10],  # 튜닝하려는 max_depth 후보값
}

lgb_param_grid = {
    'n_estimators': [100, 500, 1000, 3000],  # 튜닝하려는 n_estimators 후보값
    'learning_rate': [0.01, 0.1, 0.2],  # 튜닝하려는 learning_rate 후보값
    'max_depth': [4, 6, 8, 10],  # 튜닝하려는 max_depth 후보값
}

svr_parma_grid = {
    'gamma': [0.1, 1], 
    'C': [0.01, 0.1, 1, 10, 100], 
    'epsilon': [0, 0.01, 0.1]
}


grid_dt = GridSearchCV(DecisionTreeRegressor(), param_grid=dt_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_rf = GridSearchCV(RandomForestRegressor(), param_grid=rf_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_gb = GridSearchCV(GradientBoostingRegressor(), param_grid=gb_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_xgb = GridSearchCV(XGBRegressor(), param_grid=xgb_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_lgb = GridSearchCV(LGBMRegressor(), param_grid=lgb_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_svr = GridSearchCV(SVR(), param_grid=svr_parma_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')



# dt_reg = DecisionTreeRegressor(random_state=124, max_depth=4)
# rf_reg = RandomForestRegressor(random_state=124, n_estimators=3000, max_depth=8)
# gb_reg = GradientBoostingRegressor(random_state=124, n_estimators=3000, max_depth=8)
# xgb_reg = XGBRegressor(n_estimators=3000, max_depth=8)
# lgb_reg = LGBMRegressor(n_estimators=3000, max_depth=8)

# 트리 기반의 회귀 모델을 반복하면서 평가 수행 
i = 0
best_scores = []
model_name = ['DecisionTree','RandomForest','GradientBoosting','XGB','LGBM','SVR']
models = [grid_dt, grid_rf, grid_gb, grid_xgb, grid_lgb, grid_svr]
for model in models:  
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(model_name[i])
    best_scores.append(model.best_score_)
    i += 1
    get_evaluation(np.log1p(y_test), prediction)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# 모델명 및 R2 스코어
model_names = ['DecisionTree', 'RandomForest', 'GradientBoosting', 'XGB', 'LGBM', 'SVR']
r2_scores = [0.4764, 0.5200, 0.5228, 0.5197, 0.5390, 0.5253]

# 막대의 색상을 그레이 톤으로 지정
gray_colors = ['#666666', '#999999', '#CCCCCC', '#BBBBBB', '#DDDDDD', '#888888']

# 막대 그래프 생성
plt.figure(figsize=(8, 6))
bars = plt.bar(model_names, r2_scores, color=gray_colors)
plt.ylabel('R2 Score')
plt.title('R2 Score by Model')
plt.ylim(0, 0.8)  # Y 축 범위 설정
plt.xticks(rotation=45)

plt.show()


# In[ ]:


best_scores


# In[ ]:





# In[ ]:




