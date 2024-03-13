#!/usr/bin/env python
# coding: utf-8

# ### Regularized Linear Regression (정규화된 선형 회귀)
# - 다중 회귀 모델은 복잡도가 높아서 과대적합(overfitting)되는 경향이 있다. 이를 해결하기 위해서는 규제(penalty)를 주어 복잡도를 감소시켜야 한다.
# 
# ##### NORM
# - l<sub>p</sub>-norm 식은 아래와 같다.
# <img src="./images/regularized01.png" style="margin-left: 0">
# 
# - p=1 일 때, l<sub>1</sub>-norm, p=2일 때, l<sub>2</sub>-norm이다.
# - p=1일 때에는 마름모 형태를, p=2일 때는 원의 형태를 가진다.
# <img src="./images/regularized02.png" style="margin-left: 0">
# 
# 
# ##### 라쏘(LASSO, least absolute shrinkage and selection operator)
# - L1 규제를 통한 정규화를 사용하는 방식이다.
# - 파라미터가 2개인 경우 파란색 도형은 L1 규제를 나타내며, 주황색 선은 아래의 식을 선형으로 표현한 적합 회귀선이다.
# <div style="display: flex; margin-top:20px">
#     <div>
#         <img src="./images/regression04.png" width="100" style="margin-top:20px; margin-left: 0">
#     </div>
#     <div>
#         <img src="./images/regularized03.png" style="margin-left: 30px">
#     </div>
# </div>
# 
# - 규제항이 0에 수렴할때 L1 정규화의 경우 주황색 선의 절편은 0이 될 수 있다.
# - L1 노름의 경우 절댓값에 대한 식이므로 미분이 불가능하지만, 특정 방식을 통해 미분하였을 때 가중치가 0이라 말할 수 있기 때문에,  
# 경사하강법을 통해 학습하는 모델에는 적합하지 않다.
# - 중요하지 않은 특성들은 모델에서 제외하여 모델을 단순하게 만들고, 가장 영향력이 큰 특성이 무엇인지 알 수 있기 때문에 모델의 해석력이 좋아진다. 
# > - 📌절대값에 대한 식을 미분할 수 없는 이유는, 기울기가 -1에서 1 사이인 직선 모두가 접선이 되기 때문이다. 접점이 한 개 있을 경우 선을 정확히 그을 수 없으며, 이는 좌극한과 우극한이 다른 것이고 극한의 정의에 의해 어떤 것에 가까워진다고 단정짓기 애매하기 때문에, 해당 지점에서는 미분이 불가능하다고 말한다.
# <img src="./images/regularized06.png" width=500 style="margin-left: 0">
# 
# ##### 릿지(Ridge)
# - L2규제를 통한 정규화를 사용하는 방식이다.
# - 파라미터가 2개인 경우 파란색 도형은 L2 규제를 나타내며, 주황색 선은 아래의 식을 선형으로 표현한 적합 회귀선이다.
# <div style="display: flex; margin-top:20px">
#     <div>
#         <img src="./images/regression04.png" width="100" style="margin-top:20px; margin-left: 0">
#     </div>
#     <div>
#         <img src="./images/regularized04.png" style="margin-left: 30px">
#     </div>
# </div>
# 
# - 규제항이 0에 수렴할때 L2 정규화의 경우 주황색 선의 절편은 0이 될 수 없다.
# - L2 노름의 경우 미분을 했을 때 가중치가 남아있기 때문에, 경사하강법을 통해 학습하는 모델에는 적합하다.
# - 값이 0이 되는(제외하는) 특성은 없지만, 골고루 0에 가까운 값으로 작아지기 때문에 일부로 덜 학습시켜서 장기적으로 더 좋은 모델이 된다.
# 
# ##### λ (Regulation parameter)
# - λ이 커지면 손실 함수를 최소화하는 과정에서 노름(norm)이 작아지므로 규제가 강해졌다고 표현한다.
# - 노름(norm)이 커지면 손실 함수를 최소화하는 과정에서 λ이 작아지므로 규제가 약해졌다고 표현한다.
# <img src="./images/regularized05.png" width="350" style="margin-left: 0">
# 
# ##### 🚩결론: 여러 feature 중 일부분만 중요하면 라쏘, 중요도가 전체적으로 비슷하면 릿지를 사용하자!

# ### Polynomial Regression Task
# 
# ##### 한국인 수익 예측
# - id : 식별 번호
# - year : 조사 년도
# - wave : 2005년 wave 1위부터 2018년 wave 14위까지
# - region: 1)서울 2)경기 3)경남 4)경북 5)충남 6)강원 & 충북 7)전라 & 제주
# - income: 연간 수입 M원(백만원.1100원=1USD)
# - family_member: 가족 구성원 수
# - gender: 1) 남성 2) 여성
# - year_born: 태어난 년도
# - education_level:1)무교육(7세 미만) 2)무교육(7세 이상) 3)초등학교 4)중학교 5)고등학교 6)대학 학위 8)MA 9)박사 학위
# - marriage: 혼인상태. 1)해당없음(18세 미만) 2)혼인중 3)사망으로 별거중 4)별거중 5)미혼 6)기타
# - religion: 1) 종교 있음 2) 종교 없음  
# - occupation: 직종 코드, 별도 첨부
# - company_size: 기업 규모
# - reason_none_worker: 1)능력 없음 2)군 복무 중 3)학교에서 공부 중 4)학교 준비 5)직장인 7)집에서 돌보는 아이들 8)간호 9)경제 활동 포기 10)일할 의사 없음 11)기타

# In[2]:


import pandas as pd

income_df = pd.read_csv('./datasets/korean_income.csv')
income_df.info()


# In[3]:


income_df = income_df[income_df.income > 0.0]
income_df.shape[0]


# In[4]:


income_df.loc[:, 'occupation'] = income_df.occupation.apply(lambda x: x.replace(' ', '0'))
income_df.loc[:, 'company_size'] = income_df.company_size.apply(lambda x: x.replace('99', '0'))
income_df.loc[:, 'company_size'] = income_df.company_size.apply(lambda x: x.replace(' ', '0'))
income_df.loc[:, 'reason_none_worker'] = income_df.reason_none_worker.apply(lambda x: x.replace('99', '12'))
income_df.loc[:, 'reason_none_worker'] = income_df.reason_none_worker.apply(lambda x: x.replace(' ', '12'))


# In[5]:


income_df['target'] = income_df.income
income_df = income_df.drop(columns='income', axis=1)


# In[6]:


income_df[['company_size', 'occupation', 'reason_none_worker']] = income_df[['company_size', 'occupation', 'reason_none_worker']].astype('int16')
income_df.info()


# In[7]:


income_df = income_df.drop(columns='id', axis=1)


# In[8]:


income_df = income_df.reset_index(drop=True)
income_df


# In[9]:


from sklearn.preprocessing import StandardScaler
income_df = income_df[pd.Series(StandardScaler().fit_transform(income_df[['target']]).flatten()).between(-1.96, 1.96)]
income_df


# In[14]:


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


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

features, targets = income_df.iloc[:, :-1], income_df.target

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(poly_features, targets, test_size=0.2, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

prediction = linear_regression.predict(X_test)

get_evaluation(np.log1p(y_test), prediction)


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

features, targets = income_df.iloc[:, :-1], income_df.target

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(poly_features, targets, test_size=0.2, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

lasso = Lasso(max_iter=500)

lasso.fit(X_train, y_train)

prediction = lasso.predict(X_test)
print(lasso.coef_)
# prediction[prediction < 0] = np.log1p(2.0)

# MAE: 0.3556, MSE: 0.2283, RMSE: 0.4778, MSLE: 0.0035, RMSLE: 0.0596, R2: 0.7043

# alpha = 0.001
# MAE: 0.3709, MSE: 0.2420, RMSE: 0.4920, MSLE: 0.0037, RMSLE: 0.0610, R2: 0.6865

# alpha = 0.01
# MAE: 0.3712, MSE: 0.2422, RMSE: 0.4921, MSLE: 0.0037, RMSLE: 0.0610, R2: 0.6863

# alpha = 1
# MAE: 0.3738, MSE: 0.2447, RMSE: 0.4947, MSLE: 0.0038, RMSLE: 0.0613, R2: 0.6830

# alpha = 100
# MAE: 0.3794, MSE: 0.2505, RMSE: 0.5005, MSLE: 0.0038, RMSLE: 0.0618, R2: 0.6754

# alpha = 1000
# MAE: 0.3917, MSE: 0.2647, RMSE: 0.5145, MSLE: 0.0040, RMSLE: 0.0633, R2: 0.6571

get_evaluation(np.log1p(y_test), prediction)


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

features, targets = income_df.iloc[:, :-1], income_df.target

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(poly_features, targets, test_size=0.2, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

ridge = Ridge(max_iter=500, alpha = 1000)

ridge.fit(X_train, y_train)

prediction = ridge.predict(X_test)
print(ridge.coef_)
# prediction[prediction < 0] = np.log1p(2.0)

# MAE: 0.3556, MSE: 0.2283, RMSE: 0.4778, MSLE: 0.0035, RMSLE: 0.0596, R2: 0.7043

# alpha = 0.001
# MAE: 0.3557, MSE: 0.2283, RMSE: 0.4778, MSLE: 0.0035, RMSLE: 0.0596, R2: 0.7043

# alpha = 0.01
# MAE: 0.3557, MSE: 0.2283, RMSE: 0.4778, MSLE: 0.0035, RMSLE: 0.0596, R2: 0.7043

# alpha = 1
# MAE: 0.3557, MSE: 0.2283, RMSE: 0.4778, MSLE: 0.0035, RMSLE: 0.0596, R2: 0.7043

# alpha = 100
# MAE: 0.3557, MSE: 0.2283, RMSE: 0.4778, MSLE: 0.0035, RMSLE: 0.0596, R2: 0.7043

# alpha = 1000
# MAE: 0.3557, MSE: 0.2283, RMSE: 0.4778, MSLE: 0.0035, RMSLE: 0.0596, R2: 0.7043

get_evaluation(np.log1p(y_test), prediction)

