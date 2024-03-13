#!/usr/bin/env python
# coding: utf-8

# ### 👀교차 검증(Cross Validation)
# - 기존 방식에서는 데이터 세트에서 학습 데이터 세트와 테스트 데이터 세트를 분리한 뒤 모델 검증을 진행한다.
# - 교차 검증 시, 학습 데이터를 다시 분할하여 학습 데이터와 모델 성능을 1차 평가하는 검증 데이터로 나눈다.
# <img src="./images/cross_validation01.png" width="500" style="margin-left: -30px">

# ### 교차 검증의 장단점
# - 👍특정 데이터 세트에 대한 과적합 방지
# - 👍데이터 세트 규모가 적을 시 과소적합 방지
# - 👎모델 훈련, 모델 평가에 소요되는 시간 증가  
# ⛳ 과적합을 피하고 하이퍼 파라미터를 튜닝함으로써 모델을 일반화하고 신뢰성을 증가시키기 위해서 사용한다.

# ### 교차 검증의 종류
# ##### K-Fold
# - k개의 데이터 폴드 세트를 만든 뒤 k번만큼 학습과 검증 평가를 반복하여 수행하는 방식.
# - 학습 데이터와 검증 데이터를 정확히 자르기 때문에 타겟 데이터의 비중이 한 곳으로 치중될 수 있다.
# - 예를 들어, 0, 1, 2, 중에서 0, 1, 두 가지만 잘라서 검증하게 되면 다른 하나의 타겟 데이터를 예측할 수 없게 된다.
# - Stratified K-Fold로 해결한다.
# ##### Stratified K-Fold
# - K-Fold와 마찬가지로 k번 수행하지만, 폴드 세트를 만들 때 학습 데이터 세트와 검증 데이터 세트가 가지는 타겟 분포도가 유사하도록 검증한다.
# - 타겟 데이터의 비중을 항상 똑같게 자르기 때문에 데이터가 한 곳으로 치중되는 것을 방지한다.
# <img src="./images/cross_validation02.png" width="500" style="margin-top:20px; margin-bottom:20px; margin-left: -30px">
# ##### GridSearchCV
# - 교차 검증과 최적의 하이퍼 파라미터 튜닝을 한 번에 할 수 있는 객체이다.
# - max_depth와 min_samples_split에 1차원 정수형 list를 전달하면, 2차원으로 결합하여 격자(Grid)를 만들고, 이 중 최적의 점을 찾아낸다.
# - 딥러닝에서는 학습 속도가 머신러닝에 비해 느리고, 레이어(층)가 깊어질 수록 조정해주어야 할 하이퍼 파라미터 값이 많아지기 때문에, RandomSearchCV에서 대략적인 범위를 찾은 다음, GridSearchCV로 디테일을 조정하는 방식을 사용한다.
# <img src="./images/grid_search_cv.png" width="500" style="margin-top: 20px; margin-left: -30px">

# ##### 붓꽃 데이터로 교차 검증

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
features, targets = iris.data, iris.target

target_df = pd.DataFrame(targets, columns=['target'])
target_df.value_counts()


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

decision_tree_classifier = DecisionTreeClassifier(min_samples_leaf=6, random_state=124)
kfold = KFold(n_splits=5)


# ##### KFold.split(feature)
# - features만 전달하고 학습용, 검증용 행 번호를 array로 리턴한다.

# In[3]:


for train_index, test_index in kfold.split(features):
    print(train_index)
    print(test_index)
    print("=" * 80)


# In[4]:


count = 0
cv_accuracy = []

for train_index, test_index in kfold.split(features):
    count += 1
    
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = targets[train_index], targets[test_index]
    
    train_targets = pd.DataFrame(y_train)
    test_targets = pd.DataFrame(y_test)
    
    #학습 및 예측
    decision_tree_classifier.fit(X_train, y_train)
    prediction = decision_tree_classifier.predict(X_test)
    
    # 정확도 측정
    accuracy = np.round(accuracy_score(y_test, prediction), 4)
    
    cv_accuracy.append(accuracy)
    
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f"\n# {count} 교차 검증 정확도: {accuracy}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}")
    print(f"#{count} 학습 타겟 데이터 분포: \n{train_targets.value_counts()}")
    print(f"#{count} 검증 타겟 데이터 분포: \n{test_targets.value_counts()}")
    print(f"#{count} 학습 세트 인덱스: {train_index}")
    print(f"#{count} 검증 세트 인덱스: {test_index}")
    print("=" * 100)

# 폴드 별 검증 정확도를 합하여 평균 정확도 계산
print(f"▶ 평균 검증 정확도: {np.mean(cv_accuracy)}")


# ##### 타겟 데이터의 분포를 동일하게 교차 검증 진행
# 
# ##### StratifiedFold.split(features, targets)
# - features와 targets 모두 전달하고 학습용, 검증용 행 번호를 array로 리턴한다.

# In[5]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

decision_tree_classifier = DecisionTreeClassifier(min_samples_leaf=6, random_state=124)

# 5개의 폴드 세트로 분리
skfold = StratifiedKFold(n_splits=5)


# In[6]:


count = 0
cv_accuracy = []

for train_index, test_index in skfold.split(features, targets):
    count += 1
    
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = targets[train_index], targets[test_index]
    
    train_targets = pd.DataFrame(y_train)
    test_targets = pd.DataFrame(y_test)
    
    #학습 및 예측
    decision_tree_classifier.fit(X_train, y_train)
    prediction = decision_tree_classifier.predict(X_test)
    
    # 정확도 측정
    accuracy = np.round(accuracy_score(y_test, prediction), 4)
    
    cv_accuracy.append(accuracy)
    
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print(f"\n# {count} 교차 검증 정확도: {accuracy}, 학습 데이터 크기: {train_size}, 검증 데이터 크기: {test_size}")
    print(f"#{count} 학습 타겟 데이터 분포: \n{train_targets.value_counts()}")
    print(f"#{count} 검증 타겟 데이터 분포: \n{test_targets.value_counts()}")
    print(f"#{count} 학습 세트 인덱스: {train_index}")
    print(f"#{count} 검증 세트 인덱스: {test_index}")
    print("=" * 100)

# 폴드 별 검증 정확도를 합하여 평균 정확도 계산
print(f"▶ 평균 검증 정확도: {np.mean(cv_accuracy)}")


# ##### 편하게 수행할 수 있는 교차 검증
# 
# ##### cross_val_score(estimator, x, y, cv, scoring)
# - estimator: classifier 종류 모델이면 내부적으로 stratified K-Fold로 진행된다.
# - x: featuers
# - y: targets
# - scoring: 평가 함수, 정확도(accuracy)외에 다른 것은 다른 장에서 배운다.
# - cv: 폴드 세트 개수

# In[7]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
decision_tree_classifier = DecisionTreeClassifier(random_state=124, min_samples_leaf=6)

features, targets = iris.data, iris.target

score = cross_val_score(decision_tree_classifier, features, targets, scoring='accuracy', cv=5)
print('교차 검증별 정확도: {}'.format(score))
print('평균 정확도: {}'.format(np.mean(score)))


# #### GridSearchCV(estimator, param_grid, cv, refit, return_train_score)
# - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한번에 할 수 있다.
# - 머신러닝 알고리즘을 튜닝하기 위한 파라미터가 하이퍼 파라미터.
# - 촘촘하게 파라미터를 입력하면서 테스트 하는 방식이다.
# - 데이터 세트를 cross-validation을 위한 학습/테스트 세트로 자동으로 분할 한 뒤에 하이퍼 파라미터 그리드에 기술된 모든 파라미터를 순차적으로 적용하여 최적의 파라미터를 찾는다.
# > parameter(파라미터)  
#     estimator: 학습할 모델 객체 작성  
#     param_grid: dict형태로 전달해야 하며, 주요 key값은 max_depth, min_samples_split이다.  
#     cv: 폴드 세트 개수  
#     refit: 최적의 하이퍼 파라미터로 전달한 모델 객체를 다시 훈련하고자 할 때 True를 전달한다, 디폴트는 True.  
#     return_train_score: 교차 검증 점수를 가져올 지에 대해 True 또는 False를 전달한다.

# In[8]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# 데이터를 로딩하고 학습 데이터와 테스트 데이터를 분리한다.
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=124)
decision_tree_classifier = DecisionTreeClassifier()

# max_depth: 노드가 생성되는 최대 깊이 수 제한
# min_sample_split: 최소 샘플 개수 제한
parameters = {'max_depth': [2, 3, 4], 'min_samples_split': [6, 7]}


# In[9]:


grid_decision_tree_classifier = GridSearchCV(decision_tree_classifier
                                             , param_grid=parameters
                                             , cv=3
                                             , refit=True
                                             , return_train_score=True)

grid_decision_tree_classifier.fit(X_train, y_train)


# In[10]:


grid_decision_tree_classifier.cv_results_


# In[11]:


import pandas as pd

scores_df = pd.DataFrame(grid_decision_tree_classifier.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']]


# In[12]:


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




