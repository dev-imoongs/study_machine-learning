#!/usr/bin/env python
# coding: utf-8

# ### 분류(Classifier)
# - 대표적인 지도학습 방법 중 하나이며, 다양한 문제와 정답을 학습한 뒤 별도의 테스트에서 정답을 예측한다.
# - 주어진 문제를 먼저 학습한 뒤 새로운 문제에 대한 정답을 예측하는 방식이다.
# - 이진분류(Binary Classification)의 경우 정답은 0(음성, Negative)과 1(양성, Positive)과 같이 True, False값을 가진다.  
# <div style="width:500px; height:200px; display: flex; margin-top: 25px; margin-bottom: 25px; margin-left: 0px;">
#     <div>
#         <img src="./images/classifier01.png" width="200">  
#     </div>
#     <div style="width: 200px; heigth: 100px; margin-top: 25px; margin-left: 30px;">
#         <img src="./images/classifier02.png">  
#     </div>
# </div>
# - 다중 분류(Multiclass Classification)는 정답이 가질 수 있는 값은 3개 이상이다(예: 0, 1, 2).  
# <img src="./images/classifier03.png" width="300" style="margin-top: 25px; margin-left: 0px;"> 

# ###  📌용어 정리
# ##### 피처(Feature)
# - 데이터 세트의 일반 컬럼이며, 2차원 이상의 다차원 데이터까지 통틀어 피처라고 한다.
# - 타겟을 제외한 나머지 속성을 의미한다.
# ##### 레이블(Label), 클래스(Class), 타겟(Target), 결정(Decision)
# - 지도 학습 시 데이터의 학습을 위해 주어지는 정답을 의미한다.
# - 지도 학습 중, 분류의 경우 이를 레이블 또는 클래스라고도 부른다.
# 
# <img src="./images/feature_target.png" width="450" style="margin-left: 0">

# ### 분류 예측 프로세스
# <img src="./images/classifier_flow.png">

# ### 붓꽃 품종 예측
# - 다중 분류(Multiclass Classification)
# ##### Featuer
# - sepal length : 꽃받침의 길이
# - sepal width : 꽃받침의 너비
# - petal length : 꽃잎의 길이
# - petal width: 꽃잎의 너비
# 
# ##### Target(Label)
# - 0 : Setosa
# - 1 : Vesicolor
# - 2 : Virginica

# In[4]:


import sklearn
print(sklearn.__version__)


# In[5]:


# 사이킷런 라이브러리에 내장된 iris(붓꽃) 데이터
from sklearn.datasets import load_iris
# 결정 트리 모델
from sklearn.tree import DecisionTreeClassifier
# 학습 데이터 세트와 테스트 데이터 세트를 분리해주는 라이브러리
from sklearn.model_selection import train_test_split


# ### 붓꽃 데이터 정리
# ##### 키는 보통 data, target, target_name, feature_names, DESCR로 구성된다.
# - data는 feature의 데이터 세트이다.
# - target은 레이블 값이다.
# - target_names는 각 레이블의 이름이다.
# - feature_names는 feature의 이름이다.
# - DESCR은 데이터 세트에 대한 설명과 각 feature의 설명이다.

# In[6]:


import pandas as pd

# 붓꽃 데이터 세트 불러오기
iris = load_iris()

# 붓꽃 데이터 세트의 key값
print(f'붓꽃 데이터 세트의 키 {iris.keys()}')

keys = pd.DataFrame(iris.keys()).rename(columns={0: 'key'}).T
display(keys)

# iris.data는 feature 데이터이며, numpy.ndarray이다.
iris_feature = iris.data
print(f'iris feature: {iris_feature[:5]}')
print(f'iris type: {type(iris_feature[:5])}')
print(f'iris feature name: {iris.feature_names}')

print("=" * 80)

# iris.target은 붓꽃 데이터 세트에서 타겟(레이블, 결정 값) 데이터를 numpy.ndarray로 가지고 있다.
iris_target = iris.target
print(f'iris target: {iris_target[:5]}')
print(f'iris type: {type(iris_target[:5])}')
print(f'iris target name: {iris.target_names}')

# 붓꽃 데이터 세트 DataFrame으로 변환한다.
iris_df = pd.DataFrame(data=iris_feature, columns=iris.feature_names)
iris_df['target'] = iris_target
display(iris_df.head())
print(iris_df.info())


# In[11]:


iris


# ### 데이터 세트 분리
# ##### train_test_split(feature, target, test_size, random_state )
# - 학습 데이터 세트와 테스트 데이터 세트 분리
# - feature: 전체 데이터 세트 중 feature
# - target: 전체 데이터 세트 중 target
# - test_size: 테스트 세트의 비율(0~1)
# - random_state: 매번 동일한 결과를 원할 때, 원하는 Seed값 작성

# In[4]:


import numpy as np

X_train, X_test, y_train, y_test = train_test_split(iris_feature, iris_target, test_size=0.2, random_state=124)

print(type(X_train), type(X_test), type(y_train), type(y_test))
print(X_train[:5], X_test[:5], y_train[:5], y_test[:5], sep="\n======================\n")
print(f'전체 데이터 세트 개수: {iris_feature.__len__()}')
print(f'학습 데이터 세트 개수: {X_train.__len__()}')
print(f'타겟 데이터 세트 개수: {X_test.__len__()}')


# ##### train_test_split(feature, target, test_size, random_state)
# - DataFrame과 Series도 분할이 가능하다.

# In[5]:


import pandas as pd

iris_df = pd.DataFrame(iris_feature, columns=iris.feature_names)
iris_df['target'] = iris_target
iris_df.head()


# In[10]:


feature_df = iris_df.iloc[:, :-1]
target_df = iris_df.loc[:, 'target']

display(feature_df.head())
display(target_df.head())


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(feature_df, target_df, test_size=0.5, random_state=124)

print(X_train[:5], X_test[:5], y_train[:5], y_test[:5], sep="\n")
print(f'전체 데이터 세트 개수: {iris_feature.__len__()}')
print(f'학습 데이터 세트 개수: {X_train.__len__()}')
print(f'타겟 데이터 세트 개수: {X_test.__len__()}')


# ### 모델 학습
# ##### fit(train_feature, train_target)
# - 모델을 학습시킬 때 사용한다.
# - train_feature: 학습 데이터 세트 중 feature
# - train_target: 훈련 데이터 세트 중 target

# In[19]:


# DecisionTreeClassifier 객체 생성
decision_tree_classifier = DecisionTreeClassifier()

# 모델 학습 수행
decision_tree_classifier.fit(X_train, y_train)


# ### 예측 수행
# ##### predict(test_feature)
# - 학습된 모델에 테스트 데이터 세트의 feature를 전달하여 target을 예측한다.
# - test_feature: 테스트 데이터 세트 중 feature

# In[20]:


prediction = decision_tree_classfier.predict(X_test)
print(prediction)


# ### 정확도
# ##### accuracy_score(test_target, prediction)
# - 모델의 예측 정확도(Accuracy)를 사용하여 예측률을 구하여 평가할 수 있다.

# In[21]:


from sklearn.metrics import accuracy_score
print(f"예측 정확도: {accuracy_score(y_test, prediction)}")

