#!/usr/bin/env python
# coding: utf-8

# ### Regularized Linear Regression Task
# 
# ##### 다이아몬드 가격 예측
# 
# - price: 미국 달러로 표시된 가격 (＄326 ~ ＄18,823)
# - carat: 다이아몬드의 무게(0.2 ~ 5.01)
# - cut: 품질(공정, 좋음, 매우 좋음, 프리미엄, 이상적)
# - color: 다이아몬드 색상, J(최악)부터 D(최우수)까지
# - clarity: 다이아몬드가 얼마나 선명한지에 대한 측정값 (I1(최악), SI2, SI1, VS2, VS1, VVS2, VVS1, IF(최우수))
# - x: 길이(mm) (0 ~ 10.74)
# - y: 너비(mm)(0 ~ 58.9)
# - z: 깊이(mm)(0 ~ 31.8)
# - depth: 총 깊이 백분율 = z / 평균(x, y) = 2 * z / (x + y) (43–79)
# - table: 가장 넓은 점에 대한 다이아몬드 상단 폭(43 ~ 95)

# In[1]:


import pandas as pd
diamond_df = pd.read_csv('./datasets/diamond.csv')
diamond_df = diamond_df.drop(columns=diamond_df.columns[0], axis=1)
diamond_df


# In[2]:


diamond_df = diamond_df.drop_duplicates()
diamond_df = diamond_df.reset_index(drop=True)
diamond_df.duplicated().sum()


# In[3]:


from sklearn.preprocessing import LabelEncoder

encoders = []
columns = ['cut', 'color', 'clarity']

for column in columns:
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(diamond_df[column])
    diamond_df[column] = encoded_feature
    encoders.append(encoder)
    print(encoder.classes_)


# In[4]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
diamond_df['scaled_carat'] = scaler.fit_transform(diamond_df[['carat']])
diamond_df[~pd.Series(diamond_df.scaled_carat).between(-1.96, 1.96)]


# In[5]:


diamond_df = diamond_df[pd.Series(diamond_df.scaled_carat).between(-1.96, 1.96)]
diamond_df = diamond_df.drop(columns='scaled_carat')
diamond_df


# In[6]:


diamond_df.reset_index(drop=True, inplace=True)
diamond_df


# In[7]:


diamond_df['target'] = diamond_df.price
diamond_df = diamond_df.drop(columns='price')


# In[8]:


diamond_df


# In[11]:


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error

def get_evaluation(y_test, prediction):
    MAE =  mean_absolute_error(y_test, prediction)
    MSE = mean_squared_error(y_test, prediction)
    RMSE = np.sqrt(MSE)
    MSLE = mean_squared_log_error(y_test, prediction)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, prediction))
    R2 = r2_score(y_test, prediction)

    print('MAE: {:.4f}, MSE: {:.2f}, RMSE: {:.4f}, MSLE: {:.4f}, RMSLE: {:.4f}, R2: {:.4f}'.format(MAE, MSE, RMSE, MSLE, RMSLE, R2))


# In[12]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features, targets = diamond_df.iloc[:, :-1], diamond_df.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=0)

# 로그 변환
y_train = np.log1p(y_train)

linear_regression = LinearRegression()

linear_regression.fit(X_train, y_train)

# 기울기(가중치)
print(linear_regression.coef_)
# 절편(상수)
print(linear_regression.intercept_)

# 지수를 취하여 원래 값으로 복구
prediction = np.expm1(linear_regression.predict(X_test))
get_evaluation(y_test, prediction)


# In[16]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

features, targets = diamond_df.iloc[:, :-1], diamond_df.target

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(poly_features, targets, test_size=0.3, random_state=0)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# 로그 변환
y_train = np.log1p(y_train)

lasso = Lasso(max_iter=10000)
params = {'alpha': [0.001, 0.01, 1, 10]}

# GridSearchCV에서 사용 가능한 점수 지표(scoring)들
# 'explained_variance', 'roc_auc', 'roc_auc_ovr', 'precision_weighted', 
# 'roc_auc_ovr_weighted', 'jaccard_samples', 'rand_score', 'neg_mean_gamma_deviance', 
# 'neg_log_loss', 'jaccard_weighted', 'adjusted_rand_score', 'adjusted_mutual_info_score', 
# 'roc_auc_ovo_weighted', 'positive_likelihood_ratio', 'accuracy', 'neg_median_absolute_error', 
# 'roc_auc_ovo', 'completeness_score', 'f1', 'f1_samples', 'normalized_mutual_info_score', 'r2', 
# 'recall_samples', 'matthews_corrcoef', 'precision_macro', 'v_measure_score', 'fowlkes_mallows_score', 
# 'neg_mean_absolute_error', 'recall_macro', 'precision_samples', 'average_precision', 'jaccard', 
# 'jaccard_micro', 'jaccard_macro', 'neg_root_mean_squared_error', 'f1_weighted', 
# 'homogeneity_score', 'recall', 'precision', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 
# ▶'neg_mean_squared_log_error'◀, 
# 'neg_negative_likelihood_ratio', 'precision_micro', 'recall_micro', 'recall_weighted', 'balanced_accuracy', 
# 'max_error', 'mutual_info_score', 'top_k_accuracy', 'f1_macro', 'neg_brier_score', 'f1_micro', 'neg_mean_absolute_percentage_error'
grid_lasso = GridSearchCV(lasso, param_grid=params, cv=5, refit=True, scoring="r2")
grid_lasso.fit(X_train, y_train)

prediction = grid_lasso.predict(X_test)
get_evaluation(np.log1p(y_test), prediction)


# In[14]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_lasso.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[22]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

features, targets = diamond_df.iloc[:, :-1], diamond_df.target

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(poly_features, targets, test_size=0.3, random_state=0)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# 로그 변환
y_train = np.log1p(y_train)

ridge = Ridge(max_iter=500)
params = {'alpha': [0.001, 0.01, 1, 10]}

# scoring을 neg_mean_squared_log_error로 설정했을 때 음수 연산 오류가 발생하면, r2로 변경해준다.
# grid_ridge = GridSearchCV(ridge, param_grid=params, cv=5, refit=True, scoring="neg_mean_squared_log_error")
grid_ridge = GridSearchCV(ridge, param_grid=params, cv=5, refit=True, scoring="r2")
grid_ridge.fit(X_train, y_train)

prediction = grid_ridge.predict(X_test)
get_evaluation(np.log1p(y_test), prediction)


# In[23]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_lasso.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')

