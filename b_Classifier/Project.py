#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 필요한 라이브러리를 임포트
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

# 의사 결정 트리
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[2]:


holi_df = pd.read_csv('./datasets/pro/holidays_events.csv')
oil_df = pd.read_csv('./datasets/pro/oil.csv')
stores_df = pd.read_csv('./datasets/pro/stores.csv')
train_df = pd.read_csv('./datasets/pro/train.csv')


# dcoilwtico : 원유가격  
# onpromotion : 프로모션의 유무

# In[3]:


display(holi_df)
display(oil_df)
display(stores_df)
display(train_df)


# In[4]:


train_df = train_df.drop(columns='id', axis=1)


# In[5]:


print('='*50)
print(train_df.info())
print('='*50)
print(f'결측치 갯수: \n{train_df.isna().sum()}')
print('='*50)
print(f'중복행 갯수: {train_df.duplicated().sum()}')
print('='*50)


# In[6]:


train_df['date'] = pd.to_datetime(train_df['date'],dayfirst=False)
train_df['Year'] = pd.to_datetime(train_df['date'],dayfirst=False).dt.year
train_df['Month'] = pd.to_datetime(train_df['date'],dayfirst=False).dt.month

oil_df['date'] = pd.to_datetime(oil_df['date'],dayfirst=False)
oil_df['Year'] = pd.to_datetime(oil_df['date'],dayfirst=False).dt.year
oil_df['Month'] = pd.to_datetime(oil_df['date'],dayfirst=False).dt.month

sales_by_month = train_df[train_df['Year']==2016].groupby(by='Month').agg({'sales':'sum','onpromotion':'sum'})
sales_by_month['sales'] = sales_by_month['sales'].round(2)

# sales_by_family = train_df[train_df['Year']==2016].groupby(by='family').agg({'sales':'sum'})
# sales_by_family['sales'] = sales_by_family['sales'].round(2)

# sales_by_store_nbr = train_df[train_df['Year']==2016].groupby(by='store_nbr').agg({'sales':'sum'})
# sales_by_store_nbr['sales'] = sales_by_store_nbr['sales'].round(2)

train_df


# In[7]:


oil_df = oil_df[oil_df['Year']==2016].reset_index(drop=True)
oil_df


# In[8]:


train_df = train_df[train_df['Year']==2016]

# df_list = []
# oil_list = []

# for i in range(12):
#     train = train_df[train_df['Month'] == (i+1)]
#     df_list.append(train)
    
# train_df = pd.concat(df_list,axis=0).reset_index(drop=True)
display(train_df)
display(oil_df)


# In[9]:


merge_df = train_df.drop(columns=['Year','Month','store_nbr','family'],axis=1)
merge_df = merge_df.groupby(by='date').agg({'sales':'sum','onpromotion':'sum'})

merge_df2 = oil_df.drop(columns=['Year','Month'],axis=1)
merge_df2 = merge_df2.groupby(by='date').agg({'dcoilwtico':'sum'})

merge_df = pd.concat([merge_df,merge_df2],axis=1)
merge_df['sales'] = merge_df['sales'].round(2)
merge_df
merge_df = merge_df[~merge_df['dcoilwtico'].isna()]
merge_df = merge_df[~merge_df['onpromotion'].isna()]
merge_df


# In[10]:


print('='*50)
print(merge_df.info())
print('='*50)
print(f'결측치 갯수: \n{merge_df.isna().sum()}')
print('='*50)
print(f'중복행 갯수: {merge_df.duplicated().sum()}')
print('='*50)


# In[11]:


merge_df.hist(figsize=(5,5),bins=100)


# In[12]:


merge_df = merge_df[merge_df['dcoilwtico']!=0]
temp = merge_df['sales']
merge_df = merge_df.drop(columns='sales')
merge_df['sales'] = temp
merge_df.describe().T


# In[13]:


merge_df


# In[14]:


import matplotlib.pyplot as plt

plt.title("Sales by Oil Price")
plt.xlabel('Oil Price')
plt.ylabel('Sales')

# 범례 추가
plt.legend()
plt.scatter(merge_df.iloc[:,1], merge_df.sales, marker='o', c=merge_df.sales, s=25, cmap="rainbow", edgecolors='k')


# In[15]:


import matplotlib.pyplot as plt

plt.title("Sales by OnPromotion")
plt.xlabel('OnPromotion')
plt.ylabel('Sales')

# 범례 추가
plt.legend()
plt.scatter(merge_df.iloc[:,0], merge_df.sales, marker='o', c=merge_df.sales, s=25, cmap="rainbow", edgecolors='k')


# In[16]:


import matplotlib.pyplot as plt

plt.title("OnPromotion by Oil Price")
plt.xlabel('Oil Price')
plt.ylabel('OnPromotion')

# 범례 추가
plt.legend()
plt.scatter(merge_df.iloc[:,0], merge_df.iloc[:,1], marker='o', c=merge_df.iloc[:,1], s=25, cmap="rainbow", edgecolors='k')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = merge_df.corr()
heatmap = sns.heatmap(correlation_matrix, cmap='Oranges', annot=True, fmt='.2f')
heatmap.set_title("Correlation")


# In[17]:


merge_df.loc[merge_df['sales'] < 641895, 'sales'] = 1
merge_df.loc[(merge_df['sales'] < 686115) & (merge_df['sales'] >= 641895), 'sales'] = 2
merge_df.loc[(merge_df['sales'] < 751715) & (merge_df['sales'] >= 686115), 'sales'] = 3
merge_df.loc[merge_df['sales'] >= 751715, 'sales'] = 4

merge_df['sales'].value_counts()


# In[18]:


features, targets = merge_df.iloc[:,:-1], merge_df.sales

scale = StandardScaler()

scaled_features = scale.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(scaled_features, targets, test_size=0.3)


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = merge_df.corr()
heatmap = sns.heatmap(correlation_matrix, cmap='Oranges', annot=True, fmt='.2f')
heatmap.set_title("Correlation")


# In[20]:


correlation_matrix


# In[21]:


# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation(y_test, prediction, classifier=None, X_test=None):
#     오차 행렬
    confusion = confusion_matrix(y_test, prediction)
#     정확도
    accuracy = accuracy_score(y_test , prediction)
#     정밀도
    precision = precision_score(y_test , prediction, average='macro')
#     재현율
    recall = recall_score(y_test , prediction, average='macro')
#     F1 score
    f1 = f1_score(y_test, prediction, average='macro')
#     ROC-AUC : 연구 대상
#     roc_auc = roc_auc_score(y_test, prediction)

    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}'.format(accuracy , precision ,recall, f1))
    print("#" * 75)
    
    if classifier is not None and  X_test is not None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
        titles_options = [("Confusion matrix", None), ("Normalized confusion matrix", "true")]

        for (title, normalize), ax in zip(titles_options, axes.flatten()):
            disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax, cmap=plt.cm.Blues, normalize=normalize)
            disp.ax_.set_title(title)
        plt.show()


# In[22]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# 하이퍼 파라미터 튜닝
dt_params = {'max_depth': [5, 6, 7], 'min_samples_split': [7, 8, 9]}
svm_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
             'kernel': ['linear', 'rbf']}
knn_params = {'n_neighbors': [3, 5, 7, 9, 11]}

grid_dt_classifier = GridSearchCV(DecisionTreeClassifier(), param_grid=dt_params, cv=5, refit=True, return_train_score=True, n_jobs=4, error_score='raise')
# 소프트 보팅에서는 각 결정 클래스별 확률이 필요하기 때문에, SVC에 probability를 True로 하여
# predict_proba()를 사용할 수 있도록 해준다(허은상 도움).
grid_svc_classifier = GridSearchCV(SVC(probability=True), param_grid=svm_params, cv=5, refit=True, return_train_score=True, n_jobs=4, error_score='raise')
# KNN에서 Flag오류 발생
# Series 타입의 훈련 데이터에는 flags 속성이 없기 때문에, numpy로 변경한 뒤 훈련시켜야 한다.
grid_knn_classifier = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, cv=5, refit=True, return_train_score=True, n_jobs=4, error_score='raise')

voting_classifier = VotingClassifier(estimators=[('DTC', grid_dt_classifier)
                                                 , ('SVC', grid_svc_classifier)
                                                 , ('KNN', grid_knn_classifier)]
                                     , voting='soft')

# VotingClassifier 학습/예측/평가
voting_classifier.fit(X_train, y_train)


# In[23]:


prediction = voting_classifier.predict(X_test)
get_evaluation(y_test, prediction, voting_classifier, X_test)


# In[37]:


# 개별 모델의 학습/예측/평가.
classifiers = [grid_dt_classifier, grid_svc_classifier, grid_knn_classifier]
for classifier in classifiers:
    classifier.fit(X_train , y_train)
    prediction = classifier.predict(X_test)
    predict_proba = classifier.predict_proba(X_test)[:, 1].reshape(-1, 1)
    class_name= classifier.best_estimator_.__class__.__name__
    print(f'# {class_name}')
    get_evaluation(y_test, predict_proba, classifier, X_test)


# In[24]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'max_depth': [4, 6, 8, 10, 12],
    'min_samples_split': [6, 12, 18, 24],
    'min_samples_leaf': [4, 8, 16]
}

random_forest_classifier = RandomForestClassifier(n_estimators=100)


grid_random_forest = GridSearchCV(random_forest_classifier, param_grid=param_grid, cv=10, n_jobs=4)

grid_random_forest.fit(X_train, y_train)


# In[25]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_random_forest.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[26]:


prediction = grid_random_forest.predict(X_test)
get_evaluation(y_test, prediction, grid_random_forest, X_test)


# In[ ]:




