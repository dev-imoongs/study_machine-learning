#!/usr/bin/env python
# coding: utf-8

# ### Cross Validation Task

# ### 약물 A, B, C, X, Y
# ##### 다중 분류(Multiclass Classification)
# - 의학 연구원으로서 동일한 질병을 앓고 있는 일련의 환자에 대한 데이터를 수집했다.
# - 치료 과정 동안 각 환자는 5가지 약물, 즉 약물 A, 약물 B, 약물 c, 약물 x 및 y 중 하나에 반응했다.
# -  미래에 동일한 질병을 앓는 환자에게 어떤 약물이 적합할 수 있는지 알아보기 위한 모델을 구축한다.

# ##### feature
# - Age: 환자의 나이
# - Sex: 환자의 성별
# - BP: 혈압
# - Cholesterol: 콜레스테롤 수치
# - Na_to_K: 나트륨-칼륨
# 
# ##### target
# - Drug: 의약품, 환자에게 효과가 있었던 약

# In[1]:


from sklearn.preprocessing import LabelEncoder


# In[2]:


import pandas as pd

drugs_df = pd.read_csv('./datasets/drugs.csv')
drugs_df


# In[3]:


drugs_encoder = LabelEncoder()

targets = drugs_encoder.fit_transform(drugs_df['Drug'])
drugs_df['Drug'] = targets


# In[4]:


display(drugs_df)
drugs_encoder.classes_[drugs_df.loc[0, 'Drug']]


# In[5]:


gender_encoder = LabelEncoder()

targets = gender_encoder.fit_transform(drugs_df['Sex'])
drugs_df['Sex'] = targets


# In[6]:


display(drugs_df)
gender_encoder.classes_[drugs_df.loc[0, 'Sex']]


# In[7]:


blood_pressure_encoder = LabelEncoder()

targets = blood_pressure_encoder.fit_transform(drugs_df['BP'])
drugs_df['BP'] = targets


# In[8]:


display(drugs_df)
blood_pressure_encoder.classes_[drugs_df.loc[0, 'BP']]


# In[9]:


cholesterol_encoder = LabelEncoder()

targets = cholesterol_encoder.fit_transform(drugs_df['Cholesterol'])
drugs_df['Cholesterol'] = targets


# In[10]:


display(drugs_df)
cholesterol_encoder.classes_[drugs_df.loc[0, 'Cholesterol']]


# In[11]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


# In[12]:


# 데이터를 로딩하고 학습 데이터와 테스트 데이터를 분리한다.

features, targets = drugs_df.iloc[:, :-1], drugs_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=100)
decision_tree_classifier = DecisionTreeClassifier()

# max_depth: 노드가 생성되는 최대 깊이 수 제한
# min_sample_split: 최소 샘플 개수 제한
parameters = {'max_depth': [2, 3, 4], 'min_samples_split': [6, 7]}


# In[13]:


grid_decision_tree_classifier = GridSearchCV(decision_tree_classifier
                                             , param_grid=parameters
                                             , cv=3
                                             , refit=True
                                             , return_train_score=True)

grid_decision_tree_classifier.fit(X_train, y_train)


# In[14]:


grid_decision_tree_classifier.cv_results_


# In[15]:


scores_df = pd.DataFrame(grid_decision_tree_classifier.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']]


# In[16]:


print(f'GridSearchCV 최적 파라미터: {grid_decision_tree_classifier.best_params_}')
print(f'GridSearchCV 최고 정확도: {grid_decision_tree_classifier.best_score_}')

prediction = grid_decision_tree_classifier.predict(X_test)
print(f'테스트 데이터 세트 정확도: {accuracy_score(y_test, prediction)}')

# refit 된 객체는 best_estimator_로 가져올 수 있으며,
# 이미 grid_decision_tree_classifier객체를 GridSearchCV로 작업하여 생성했기 때문에
# 결과는 똑같이 나온다.
estimator = grid_decision_tree_classifier.best_estimator_
prediction = estimator.predict(X_test)
print(f'테스트 데이터 세트 정확도: {accuracy_score(y_test, prediction)}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




