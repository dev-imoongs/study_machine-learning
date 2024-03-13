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


# In[3]:


holi_df['type'].value_counts()


# In[4]:


holi_temp_df = holi_df[['date', 'type']]  # 선택한 열로 새로운 데이터프레임 생성
holi_temp_df = holi_temp_df.set_index('date')  # 'date' 열을 인덱스로 설정
holi_temp_df


# dcoilwtico : 원유가격  
# onpromotion : 프로모션의 유무

# In[5]:


oil_df.dropna(inplace=True)
oil_df


# In[6]:


oil_df = oil_df.set_index('date')


# In[7]:


oil_df['oil_price'] = oil_df['dcoilwtico']
oil_df.drop(columns=['dcoilwtico'], inplace=True)


# In[8]:


#price_chg: 원유가격의 변화량
#pct_chg: 원유가격 변화 비율
# oil_df['price_chg'] = oil_df.oil_price - oil_df.oil_price.shift(1)
# oil_df['price_pct_chg'] = oil_df['price_chg']/oil_df.oil_price.shift(-1)


# In[9]:


oil_df


# In[10]:


import matplotlib.pyplot as plt

# Assuming you have a DataFrame named oil_df with 'date' and 'dcoilwtico' columns

fig, ax = plt.subplots(figsize=(15, 5))
plt.plot(oil_df['oil_price'])
plt.title('Oil Price over Time')
plt.xlabel('Date')
plt.ylabel('Oil Price')
# plt.grid(True)  # Add grid lines

plt.show()


# In[11]:


train_df


# In[12]:


train_df.family.value_counts()


# In[13]:


sales_by_family = train_df.groupby(['date','family'])[['sales']].sum()


# In[14]:


# sales_by_family['date'] = pd.to_datetime(sales_by_family['date'],dayfirst=False)
sales_by_family


# In[15]:


sales_by_family['price_change'] = sales_by_family.groupby('family')['sales'].pct_change()
sales_by_family['price_diff'] = sales_by_family.groupby('family')['sales'].diff()
sales_by_family.iloc[70:100,:]


# In[16]:


import numpy as np

# "price_change" 열에서 inf 값을 제외하고 최대 값을 찾는 코드
max_price_change_family_by_date = sales_by_family[~np.isinf(sales_by_family['price_change'])].dropna().groupby('date')['price_change'].idxmax()
max_sales_by_family = sales_by_family.loc[max_price_change_family_by_date]

# 결과 확인
display(max_sales_by_family)


# In[17]:


max_sales_by_family


# In[18]:


max_sales_by_family['family'] = max_sales_by_family.index.get_level_values('family')
max_sales_by_family


# In[19]:


daily_total_sales = train_df.groupby('date')['sales'].sum()

# 결과 출력
display(daily_total_sales)


# In[20]:


merged_df = oil_df.merge(daily_total_sales, on='date', how='inner')

# 결과 출력
display(merged_df)


# In[21]:


merged_df = merged_df.merge(holi_temp_df, on='date', how='outer')
merged_df


# In[22]:


merged_df.loc[merged_df['type'].isna(), 'type'] = 'Work day'
merged_df


# In[23]:


merged_df.info()


# In[24]:


merged_df = merged_df.merge(max_sales_by_family['family'], on='date', how='outer')
merged_df


# In[25]:


merged_df = merged_df.dropna()
merged_df


# In[26]:


# 'type' 열이 'Work day'인 행의 인덱스 추출
work_day_index = merged_df[merged_df['type'] == 'Work day'].index

# 'type' 열이 'Work day'인 행들을 따로 저장
merged_df.loc[work_day_index, 'type'] = 0

# 'type' 열이 'Work day'가 아닌 행들을 따로 저장
merged_df.loc[~merged_df.index.isin(work_day_index), 'type'] = 1

merged_df.type.value_counts()


# In[27]:


merged_df


# In[28]:


merged_df.family.value_counts()


# In[29]:


# temp = merged_df['sales']
# merged_df.drop(columns='sales', inplace=True)
# merged_df['sales'] = temp
# merged_df


# In[30]:


# bar_df = merged_df.groupby(by='type').mean('sales')


# In[31]:


# import pandas as pd
# import matplotlib.pyplot as plt


# colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral', 'lightblue', 'gold']

# # 데이터프레임을 가장 높은 'sales' 값을 가진 순서대로 정렬
# bar_df = bar_df.sort_values(by='sales', ascending=True)

# # 수평 막대그래프 생성
# plt.figure(figsize=(10, 6))  # 그래프 크기 설정

# plt.barh(bar_df.index, bar_df['sales'], color=colors)
# plt.xlabel('Sales')
# plt.title('Sales by Type')
# plt.grid(axis='x', linestyle='--', alpha=0.6)

# plt.show()


# In[32]:


merged_df.info()


# In[33]:


from sklearn.preprocessing import LabelEncoder

encoders = {}

columns = ['type', 'family']

for column in columns:
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(merged_df[column])
    merged_df[column] = encoded_feature
    encoders[column] = encoder
    print(f'Classes for column {column}: {encoder.classes_}')


# In[34]:


merged_df.hist()


# In[35]:


from sklearn.preprocessing import StandardScaler

new_df_columns = ['oil_price','sales']

scale = StandardScaler()
new = scale.fit_transform(merged_df.loc[:,new_df_columns])
new_df = pd.DataFrame(columns=new_df_columns, data=new)
new_df


# In[36]:


new_merged_df = merged_df.reset_index()


# In[37]:


new_df = pd.concat([new_df,new_merged_df['type'],new_merged_df['family']],axis=1)
new_df


# In[38]:


new_df.hist()


# In[39]:


# x = new_df.index
# y1 = new_df['oil_price']
# y2 = new_df['sales']

# # Figure 객체를 생성하고 크기 조정
# fig, ax = plt.subplots(figsize=(10, 5))

# # 두 개의 선 그래프 그리기
# plt.plot(x, y1, label='Oil Price')
# plt.plot(x, y2, label='Sales', alpha=0.6)

# # 범례 추가
# plt.legend(loc='top left', bbox_to_anchor=(1, 0.5))

# # 축 레이블 설정
# plt.ylabel('Oil Price and Sales')
# plt.xlabel('Date')

# # 그래프 표시
# plt.show()


# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = new_df.corr()
heatmap = sns.heatmap(correlation_matrix, cmap='Oranges', annot=True, fmt='.2f')
heatmap.set_title("Correlation")


# In[41]:


# merged_df['sales'] = merged_df['sales'].round(0)
# merged_df['sales'] = merged_df['sales'].astype('int32')


# In[42]:


# merged_df.describe().loc[['min','25%','50%','75%','max'],:]


# In[43]:


# merged_df.dropna(inplace=True)
# merged_df.describe()


# In[44]:


# scaler = StandardScaler()
# scaled_merged = scaler.fit_transform(merged_df)
# scaled_merged_df = pd.DataFrame(data=scaled_merged,columns=merged_df.columns,index=merged_df.index)
# scaled_merged_df


# In[45]:


# scaled_merged_df.describe().T


# In[46]:


merged_df


# In[47]:


# scaled_merged_df.index = merged_df.index
# scaled_merged_df


# In[48]:


# for column in scaled_merged_df.columns:
#     merged_df = merged_df[scaled_merged_df[column].between(-1.96,1.96)]


# In[49]:


merged_df = merged_df.reset_index()


# In[50]:


# merged_df['year'] = pd.to_datetime(merged_df['date'],dayfirst=False).dt.year.astype(int)
# merged_df['month'] = pd.to_datetime(merged_df['date'],dayfirst=False).dt.month.astype(int)
# merged_df['day'] = pd.to_datetime(merged_df['date'],dayfirst=False).dt.day.astype(int)

# merged_df = merged_df.drop(columns=['date'],axis=1)


# In[51]:


merged_df


# In[52]:


merged_df.hist()


# In[53]:


regression_df = merged_df


# In[54]:


# merged_df['target'] = 0


# In[55]:


merged_df.describe().T


# In[56]:


# merged_df.loc[merged_df['sales'] < 400000, 'target'] = 1
# merged_df.loc[(merged_df['sales'] < 600000) & (merged_df['sales'] >= 400000), 'target'] = 2
# merged_df.loc[(merged_df['sales'] < 700000) & (merged_df['sales'] >= 600000), 'target'] = 3
# merged_df.loc[merged_df['sales'] >= 700000, 'target'] = 4

# merged_df['target'].value_counts()


# In[57]:


# new_df = merged_df.drop(columns='sales',axis=1)


# In[58]:


merged_df.family.value_counts()


# In[ ]:


merged_df


# In[59]:


# family 열의 값별로 빈도수를 계산
family_counts = merged_df['family'].value_counts()

# 빈도수가 7 이상인 family 값을 추출
valid_family_values = family_counts[family_counts >= 8].index

# valid_family_values를 이용하여 필터링
filtered_df = merged_df[merged_df['family'].isin(valid_family_values)]


# In[60]:


filtered_df.reset_index(drop=True,inplace=True)


# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = merged_df.drop(columns='date').corr()
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
heatmap.set_title("Correlation")


# In[62]:


filtered_df['type'].value_counts()


# In[63]:


# family 열의 값별로 빈도수를 계산
family_counts = filtered_df['family'].value_counts()

# 빈도수가 7 이상인 family 값을 추출
valid_family_values = family_counts[family_counts >= 7].index

# valid_family_values를 이용하여 필터링
filtered_df = filtered_df[filtered_df['family'].isin(valid_family_values)]


# In[80]:


features, targets = filtered_df.drop(columns=['date','family']), filtered_df.family

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=169)


# In[74]:


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


# In[81]:


from imblearn.over_sampling import SMOTE

# SMOTE 객체 생성
smote = SMOTE(sampling_strategy='auto', random_state=0)  # 'auto'는 자동으로 적절한 비율로 오버샘플링을 수행합니다.

# 독립 변수(X_train) 및 목표 변수(y_train)에 대한 오버샘플링 수행
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)


# In[83]:


X_train_over.value_counts()


# In[84]:


y_train_over.value_counts()


# In[87]:


import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def objective_dt(trial):
    # 의사결정 트리 하이퍼파라미터 튜닝
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2,16)
    
    obj_dtc = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    obj_dtc.fit(X_train_over, y_train_over)
    y_pred = obj_dtc.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')  # weighted f1-score
    return f1

def objective_svm(trial):
    # 서포트 벡터 머신 하이퍼파라미터 튜닝
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)
    gamma = trial.suggest_float('gamma', 1e-4, 1e4, log=True)
    
    obj_svm = SVC(C=C, gamma=gamma)
    obj_svm.fit(X_train_over, y_train_over)
    y_pred = obj_svm.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')  # weighted f1-score
    return f1

def objective_knn(trial):
    # KNN 하이퍼파라미터 튜닝
    n_neighbors = trial.suggest_int('n_neighbors', 1, 10)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    
    obj_knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    obj_knn.fit(X_train_over, y_train_over)
    y_pred = obj_knn.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')  # weighted f1-score
    return f1

# Optuna 스터디 생성 및 실행
study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(objective_dt, n_trials=100)

study_svm = optuna.create_study(direction='maximize')
study_svm.optimize(objective_svm, n_trials=100)

study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(objective_knn, n_trials=100)

# 최적 하이퍼파라미터 출력
print("Best Decision Tree Hyperparameters:")
best_dt_params = study_dt.best_params
print(best_dt_params)

print("Best Support Vector Machine Hyperparameters:")
best_svm_params = study_svm.best_params
print(best_svm_params)

print("Best K-Nearest Neighbors Hyperparameters:")
best_knn_params = study_knn.best_params
print(best_knn_params)


# In[88]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# 하이퍼 파라미터 튜닝
# dt_params = {'max_depth': [5, 6, 7, 10, 15], 'min_samples_split': [2, 3, 4, 7, 8, 9]}
# svm_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#              'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#              'kernel': ['linear', 'rbf']}
# knn_params = {'n_neighbors': [3, 5, 7, 9, 11, 15]}

# grid_dt_classifier = GridSearchCV(DecisionTreeClassifier(), param_grid=dt_params, cv=3, refit=True, return_train_score=True, n_jobs=-1, error_score='raise')
# grid_svc_classifier = GridSearchCV(SVC(probability=True), param_grid=svm_params, cv=3, refit=True, return_train_score=True, n_jobs=-1, error_score='raise')
# grid_knn_classifier = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, cv=3, refit=True, return_train_score=True, n_jobs=-1, error_score='raise')

dt_classifier = DecisionTreeClassifier(max_depth=15, min_samples_split=2)
svc_classifier = SVC(C=63, gamma=0.0001, kernel='rbf', probability=True)
knn_classifier = KNeighborsClassifier(n_neighbors=4, weights='distance')


# voting_classifier = VotingClassifier(estimators=[('DTC', grid_dt_classifier)
#                                                  , ('SVC', grid_svc_classifier)
#                                                  , ('KNN', grid_knn_classifier)]
#                                      , voting='soft')


voting_classifier = VotingClassifier(estimators=[('DTC', dt_classifier)
                                                 , ('SVC', svc_classifier)
                                                 , ('KNN', knn_classifier)]
                                     , voting='soft')

# VotingClassifier 학습/예측/평가
voting_classifier.fit(X_train_over, y_train_over)


# In[89]:


prediction = voting_classifier.predict(X_test)
predict_proba = voting_classifier.predict_proba(X_test)
get_evaluation(y_test, prediction, voting_classifier, X_test)


# In[ ]:


# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt


# # PR 곡선을 그릴 각 클래스의 정밀도, 재현율, 임계값을 계산
# precision = dict()
# recall = dict()
# thresholds = dict()
# n_classes = predict_proba.shape[1]  # 클래스 수

# for i in range(n_classes):
#     precision[i], recall[i], thresholds[i] = precision_recall_curve(y_test, predict_proba[:, i])
#     plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')

# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.legend(loc="best")
# plt.title("Precision-Recall curve")
# plt.show()


# In[79]:


# 개별 모델의 학습/예측/평가.
classifiers = [dt_classifier, svc_classifier, knn_classifier]
# classifiers = [grid_dt_classifier, grid_svc_classifier, grid_knn_classifier]
for classifier in classifiers:
    classifier.fit(X_train_over, y_train_over)
    prediction = classifier.predict(X_test)
    predict_proba = classifier.predict_proba(X_test)[:, 1].reshape(-1, 1)
#     class_name= classifier.best_estimator_.__class__.__name__
#     print(f'# {class_name}')
    get_evaluation(y_test, prediction, classifier, X_test)


# In[ ]:


# from sklearn.preprocessing import Binarizer

# # 임계값을 0.5로 설정
# threshold = 0.7

# custome_proba = prediction_proba[:, 1].reshape(-1, 3)

# binarizer = Binarizer(threshold=threshold).fit(custome_proba)
# custom_prediction = binarizer.transform(custome_proba)

# get_evaluation(y_test, custom_prediction, voting_classifier, X_test)


# In[ ]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'max_depth': [4, 6, 8, 10, 12],
    'min_samples_split': [6, 8, 12, 18, 24],
    'min_samples_leaf': [4, 8, 16]
}

random_forest_classifier = RandomForestClassifier(n_estimators=100)


grid_random_forest = GridSearchCV(random_forest_classifier, param_grid=param_grid, cv=5, n_jobs=-1)

grid_random_forest.fit(X_train_over, y_train_over)


# In[ ]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_random_forest.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[ ]:


prediction = grid_random_forest.predict(X_test)
get_evaluation(y_test, prediction, grid_random_forest, X_test)


# In[ ]:


import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
# conda install -c conda-forge imbalanced-learn


# GridSearchCV 수행 시작 시간 설정.
start_time = time.time()

param_grid = {
    'n_estimators': [16, 32, 50, 100, 500, 1000],
    'learning_rate': [0.001, 0.005, 0.01, 0.1, 0.3, 0.5, 0.7],
}

# boost_from_average가 True일 경우(default: True) 타겟 데이터가 불균형 분포를 이루는 경우 재현률 및 ROC-AUC 성능이 매우 저하됨
# 따라서 boost_from_average를 False로 설정하는 것이 유리하다.
lgbm = LGBMClassifier(boost_from_average=False)


print('SMOTE 적용 전:\n',pd.Series(y_train).value_counts() )
print('SMOTE 적용 후:\n',pd.Series(y_train_over).value_counts() )

# 학습 데이터를 검증 데이터로 분리
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, stratify=y_train, test_size=0.5)

evals = [(X_val_train, y_val_train), (X_val_test, y_val_test)]

grid_lgbm = GridSearchCV(lgbm, param_grid, cv=3, refit=True, return_train_score=True, n_jobs=-1, error_score='raise')

grid_lgbm.fit(X_train_over, y_train_over, early_stopping_rounds=50, eval_set=evals)

print("GridSearchCV 수행 시간: {0:.1f} 초 ".format(time.time() - start_time))


# In[ ]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_lgbm.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[ ]:


prediction = grid_lgbm.predict(X_test)
get_evaluation(y_test, prediction, grid_lgbm, X_test)


# In[ ]:


classifiers = [grid_dt_classifier, grid_svc_classifier, grid_knn_classifier, grid_random_forest, grid_lgbm]


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# 모델과 해당 성능 메트릭을 저장할 리스트 초기화
model_names = ['DTC', 'SVC', 'KNN', 'RandomForest', 'LGBM']
accuracy_scores = [accuracy_score(y_test, classifier.predict(X_test)) for classifier in classifiers]
precision_scores = [precision_score(y_test , classifier.predict(X_test), average='macro') for classifier in classifiers]
recall_scores = [recall_score(y_test , classifier.predict(X_test), average='macro') for classifier in classifiers]
f1_scores = [f1_score(y_test, classifier.predict(X_test), average='macro') for classifier in classifiers]


colors = sns.color_palette("pastel")

# 그래프에 표시할 바 차트의 위치 지정
x = np.arange(len(model_names))

# 막대 그래프 생성
plt.figure(figsize=(10, 6))
plt.bar(x - 0.3, accuracy_scores, width=0.2, label='Accuracy', color='skyblue')
plt.bar(x - 0.1, precision_scores, width=0.2, label='Precision', color='lightsalmon')
plt.bar(x + 0.1, recall_scores, width=0.2, label='Recall', color='lightgreen')
plt.bar(x + 0.3, f1_scores, width=0.2, label='F1 Score', color='lightcoral')

# for i in range(len(model_names)):
#     plt.text(x[i] - bar_width / 2, accuracy_scores[i] + 0.01, f'{accuracy_scores[i]:.2f}', ha='center', va='bottom', fontsize=12)
#     plt.text(x[i] + bar_width / 2, f1_scores[i] + 0.01, f'{f1_scores[i]:.2f}', ha='center', va='bottom', fontsize=12)

# # X 축 레이블 설정
plt.xlabel('Model')
plt.xticks(x, model_names, fontsize=12, ha='center')

# Y 축 레이블 설정
plt.ylabel('Score')

# 범례 추가
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# 그래프 제목 설정
plt.title('Evaluation Score')

# 그래프 표시
plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# 모델과 해당 성능 메트릭을 저장할 리스트 초기화
model_names = ['LGBM']
accuracy_scores = [accuracy_score(y_test, classifier.predict(X_test)) for classifier in classifiers]
precision_scores = [precision_score(y_test , classifier.predict(X_test), average='macro') for classifier in classifiers]
recall_scores = [recall_score(y_test , classifier.predict(X_test), average='macro') for classifier in classifiers]
f1_scores = [f1_score(y_test, classifier.predict(X_test), average='macro') for classifier in classifiers]

# 그래프에 표시할 바 차트의 위치 지정
x = np.arange(len(model_names))

# 막대 그래프 생성
plt.figure()
plt.bar(x - 0.3, accuracy_scores, width=0.15, label='Accuracy', color='skyblue')
plt.bar(x - 0.1, precision_scores, width=0.15, label='Precision', color='lightsalmon')
plt.bar(x + 0.1, recall_scores, width=0.15, label='Recall', color='lightgreen')
plt.bar(x + 0.3, f1_scores, width=0.15, label='F1 Score', color='lightcoral')

# for i in range(len(model_names)):
#     plt.text(x[i] - bar_width / 2, accuracy_scores[i] + 0.01, f'{accuracy_scores[i]:.2f}', ha='center', va='bottom', fontsize=12)
#     plt.text(x[i] + bar_width / 2, f1_scores[i] + 0.01, f'{f1_scores[i]:.2f}', ha='center', va='bottom', fontsize=12)

# # X 축 레이블 설정
plt.xlabel('Model')
plt.xticks(x, model_names, fontsize=12, ha='center')

# Y 축 레이블 설정
plt.ylabel('Score')

# 범례 추가
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# 그래프 제목 설정
plt.title('Evaluation Score')

# 그래프 표시
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.inspection import permutation_importance

importance = permutation_importance(grid_lgbm, X_test, y_test, n_repeats=100, random_state=0)
new_df.columns[importance.importances_mean.argsort()[::-1]]


# In[ ]:


regression_df


# In[ ]:


temp_data = regression_df.sales
regression_df.drop(columns=['sales','target'],axis=1,inplace=True)
regression_df['sales']=temp_data
regression_df


# In[ ]:


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error

def get_evaluation(y_test, prediction):
    MAE =  mean_absolute_error(y_test, prediction)
    MSE = mean_squared_error(y_test, prediction)
    RMSE = np.sqrt(MSE)
    MSLE = mean_squared_log_error(y_test, prediction)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, prediction))
    R2 = r2_score(y_test, prediction)

    print('MAE: {:.4f}, MSE: {:.2f}, RMSE: {:.2f}, MSLE: {:.2f}, RMSLE: {:.2f}, R2: {:.2f}'.format(MAE, MSE, RMSE, MSLE, RMSLE, R2))


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

features, targets = regression_df.iloc[:,:-1], regression_df.sales

poly_features = PolynomialFeatures(degree=3).fit_transform(features)

reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(poly_features, targets, test_size=0.3, random_state=124)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

dt_reg = DecisionTreeRegressor(random_state=124, max_depth=4)
rf_reg = RandomForestRegressor(random_state=124, n_estimators=1000, max_depth=8)
gb_reg = GradientBoostingRegressor(random_state=124, n_estimators=1000, max_depth=8)
xgb_reg = XGBRegressor(n_estimators=1000, max_depth=8)
lgb_reg = LGBMRegressor(n_estimators=1000, max_depth=8)


# 트리 기반의 회귀 모델을 반복하면서 평가 수행 
models = [dt_reg, rf_reg, gb_reg, xgb_reg, lgb_reg]
for model in models:  
    model.fit(reg_X_train, np.log1p(reg_y_train))
    prediction = model.predict(reg_X_test)
    print(model.__class__.__name__)
    get_evaluation(np.log1p(reg_y_test), prediction)


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

scale = StandardScaler()

reg_X_train = scale.fit_transform(reg_X_train)
reg_X_test = scale.fit_transform(reg_X_test)

reg_y_train = np.log1p(reg_y_train)

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


grid_dt = GridSearchCV(DecisionTreeRegressor(random_state=124), param_grid=dt_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_rf = GridSearchCV(RandomForestRegressor(random_state=124), param_grid=rf_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=124), param_grid=gb_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_xgb = GridSearchCV(XGBRegressor(random_state=124), param_grid=xgb_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
grid_lgb = GridSearchCV(LGBMRegressor(random_state=124), param_grid=lgb_param_grid, cv=5, refit=True, return_train_score=True, n_jobs=-1, scoring='r2')
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
    model.fit(reg_X_train, reg_y_train)
    prediction = model.predict(reg_X_test)
    print(model_name[i])
    best_scores.append(model.best_score_)
    i += 1
    get_evaluation(np.log1p(reg_y_train), prediction)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# 모델명 및 R2 스코어
model_names = ['DecisionTree', 'RandomForest', 'GradientBoosting', 'XGB', 'LGBM', 'SVR']
r2_scores = [grid_dt.best_score_, grid_rf.best_score_, grid_gb.best_score_, grid_xgb.best_score_, grid_lgb.best_score_, grid_svr.best_score_]

# 파스텔 컬러 정의
pastel_colors = ['#FF6666', '#FFCC99', '#99FF99', '#66B2FF', '#C2C2F0', '#FF77FF']

# 막대 그래프 생성
plt.figure(figsize=(8, 6))
plt.bar(model_names, best_scores, color=pastel_colors)
plt.ylabel('R2 Score')
plt.title('R2 Score by Model')
plt.ylim(0, 1)  # Y 축 범위 설정
plt.xticks(rotation=45)

plt.show()

