#!/usr/bin/env python
# coding: utf-8

# ### Support Vector Classifier Task
# 
# ##### 심장마비 가능성 예측
# 
# 
# ##### feature
# - age: 환자의 나이  
# - sex: 환자의 성별 (0 = female, 1 = male).  
# - cp: 흉통성 (1 = typical angina(전형 협심증), 2 = atypical angina(비전형 협심증), 3 = non-anginal pain(비협심증), 4 = asymptomatic(무증상))  
# - trestbps: 정지 혈압(mmHg)  
# - chol: 혈청 콜레스테롤 수치(mg/dl)  
# - fbs: 공복혈당 (> 120mg/dl) (1 = true, 0 = false)  
# - restecg: 정지 심전도 결과 (0 = normal, 1 = having ST-T wave abnormality(ST-T파 이상), 2 = probable or definite left ventricular hypertrophy(좌심실 비대 가능성 또는 확실성))  
# - thalach: 최대 심박수  
# - exang: 운동 유발 협심증 (1 = yes, 0 = no)  
# - oldpeak: 휴식과 관련된 운동으로 인한 ST 우울증  
# - slope: 피크 운동 ST 세그먼트의 기울기  
# - ca: 형광 투시법으로 채색된 주요 혈관의 수  
# - thal: 탈라세미아(혈액장애의 일종) results (3 = normal, 6 = fixed defect(고정결함), 7 = reversible defect(가역결함))  
# 
# ##### target
# - output : 1 = 심장마비 존재, 0 = 심장마비 없음

# In[55]:


import pandas as pd

heart_df = pd.read_csv('./datasets/heart.csv')
heart_df


# In[56]:


heart_df.info()


# In[57]:


heart_df.isna().sum()


# In[58]:


heart_df.duplicated().sum()


# In[59]:


heart_df[heart_df.duplicated()]


# In[60]:


heart_df.drop_duplicates(inplace=True)
heart_df.duplicated().sum()


# In[61]:


heart_df.describe().T


# ##### 상관 관계 분석
# - cp, thalachh, slp, restecg

# In[62]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = heart_df.corr()
fig = plt.figure(figsize=(7, 5))
heatmap = sns.heatmap(corr, cmap="Oranges")
heatmap.set_title("Correlation")


# In[63]:


# 1. 상관관계 수치를 내림차순 정렬
# 2. 첫 번째 행 추출(가장 수치가 높은 데이터의 행)
# 3. 내림차순 정렬(상관 관계가 높은 순)
corr.sort_values(by="output", ascending=False).iloc[0].sort_values(ascending=False)


# In[64]:


heart_df.iloc[:, :-1].hist(figsize=(7, 10))


# ##### 표준화를 통한 이상치 제거
# - 표준화된 값이 평균을 기준으로 떨어져 있는 거리이므로, ±1.96 범위를 벗어난다면 이상치에 포함된다.  
# - 상관 관계 비중이 높은 cp, thalachh, slp, restecg에 대해 이상치를 제거한다.

# In[65]:


from sklearn.preprocessing import StandardScaler

features = heart_df.iloc[:, :-1]

standard_scale = StandardScaler()
standard_scale_features = standard_scale.fit_transform(features)
standard_scale_features_df = pd.DataFrame(standard_scale_features, columns=features.columns)

standard_scale_features_df['target'] = heart_df.output
standard_scale_features_df = standard_scale_features_df[~standard_scale_features_df['target'].isna()]
standard_scale_features_df.shape


# In[66]:


columns = ['thalachh', 'slp']

for column in columns:
    print(f'{column}: {standard_scale_features_df[~standard_scale_features_df[column].between(-1.96, 1.96)].shape[0]}건')


# In[67]:


columns = ['thalachh', 'slp']
for column in columns:
    standard_scale_features_df.drop(standard_scale_features_df[~standard_scale_features_df[column].between(-1.96, 1.96)][column].index, axis=0, inplace=True)

for column in columns:
    print(f'{column}: {standard_scale_features_df[~standard_scale_features_df[column].between(-1.96, 1.96)].shape[0]}건')


# In[68]:


heart_df = heart_df.iloc[standard_scale_features_df.index, :].reset_index(drop=True)
heart_df


# ##### 파이프라인 구축
# - 정규화
# - 하이퍼 파라미터 튜닝

# In[69]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf']
}

grid_support_vector = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=5, refit=True, return_train_score=True, n_jobs=4)

features, targets = heart_df.iloc[:, :-1], heart_df.output

X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2)

svc_pipeline = Pipeline([('normalizer', Normalizer()), ('support_vector_classifier', grid_support_vector)])
svc_pipeline.fit(X_train, y_train)


# In[70]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_support_vector.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# ##### 평가
# - 오차 행렬
# - 정확도
# - 정밀도
# - 재현율
# - F1 score
# - ROC-AUC

# In[71]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score

# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation(y_test, prediction, prediction_proba=None, classifier=None, X_test=None):
#     오차 행렬
    confusion = confusion_matrix(y_test, prediction)
#     정확도
    accuracy = accuracy_score(y_test , prediction)
#     정밀도
    precision = precision_score(y_test , prediction)
#     재현율
    recall = recall_score(y_test , prediction)
#     F1 score
    f1 = f1_score(y_test, prediction)
#     ROC-AUC
    roc_auc = roc_auc_score(y_test, prediction_proba)

    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}, AUC:{4:.4f}'.format(accuracy , precision ,recall, f1, roc_auc))
    print("#" * 75)
    
    if classifier is not None and  X_test is not None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
        titles_options = [("Confusion matrix", None), ("Normalized confusion matrix", "true")]

        for (title, normalize), ax in zip(titles_options, axes.flatten()):
            disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax, cmap=plt.cm.Blues, normalize=normalize)
            disp.ax_.set_title(title)
        plt.show()


# In[72]:


# svc_pipeline.fit(X_train, y_train)

prediction = svc_pipeline.predict(X_test)

prediction_proba = svc_pipeline.predict_proba(X_test)[:, 1]
get_evaluation(y_test, prediction, prediction_proba, svc_pipeline, X_test)

