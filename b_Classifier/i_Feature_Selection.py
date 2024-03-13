#!/usr/bin/env python
# coding: utf-8

# ### Feature Selection
# - 결과 예측에 있어서, 불필요한 feature들로 인해 모델 예측 성능을 떨어뜨릴 가능성을 사전 제거할 수 있다.
# - 타겟 데이터와 관련이 없는 변수들을 제거하여, 타겟 데이터를 가장 잘 예측하는 feature들의 조합(상관관계가 높은)을 찾아내는 것이 목적이다.
# #### 📌용어 정리
# <table style="margin-left: 0">
#     <tr>
#         <th style="text-align: center">표현</th>
#         <th style="text-align: center">정의</th>
#     </tr>
#     <tr>
#         <td style="text-align: center">Feature Engineering</td>
#         <td style="text-align: center">도메인(기본) 지식을 사용하여 데이터에서 피처를 변형 및 생성</td>
#     </tr>
#     <tr>
#         <td style="text-align: center">Feature Extraction</td>
#         <td style="text-align: center">차원축소 등 새로운 중요 피처를 추출</td>
#     </tr>
#     <tr>
#         <td style="text-align: center">Feature Selection</td>
#         <td style="text-align: center">기존 피처에서 원하는 피처만 선택하는 과정</td>
#     </tr>
# </table>
# 
# ##### Recursive Feature Elimination(RFE)
# - 모델 최초 학습 이후 feature의 중요도를 선정하는 방식이다.
# - feature의 중요도가 낮은 속성들을 차례로 제거하면서 원하는 feature의 개수가 남을 때 까지 반복적으로 학습 및 평가를 수행한다.
# - 경우의 수로 제거해가며 학습을 재수행하기 때문에 시간이 오래 걸린다.
# - 몇 개의 feature를 추출해야 할 지 직접 정의해야 하는 것이 단점이다.
# <img src="./images/RFE.png" width="400" style="margin-left: 0">
# 
# ##### Recursive Feature Elimination Cross Validation(RFECV)
# - RFE의 단점을 보완하기 위해 만들어졌으며, 최고 성능에서의 feature 개수를 알려주고, 해당 feature를 선택해준다.
# - 각 feature마다 Cross Validation을 진행하여 각기 다른 성능을 도출한다.
# - 도출된 성능 수치를 평균 내어 가장 높은 성능을 발휘하는 feature들을 선택한다.
# <img src="./images/RFECV.png" width="600" style="margin-left: 0">
# 
# ##### Permutation Importance
# - Permutation(순열)이란, 서로 다른 n개의 원소에서 r개를 중복없이 순서에 상관있게 선택하는 혹은 나열하는 것이다.  
# 여기서 원소는 feature이며, 각 feature별로 중복없이 선택하여 feature의 중요도를 검증하는 방식이다.
# - 임의의 feature의 요소 순서를 무작위로 섞은 후 성능 감소에 대한 평균을 구한다. 
# - 중요도를 판단하려는 feature의 요소를 noise로 만들어서 전과 후를 비교한 뒤 중요도를 판단한다.
# - 임의의 feature를 noise로 만들었을 때 성능이 떨어진 정도로 feature importance를 판별할 수 있다.
# 
# <img src="./images/feature_selection01.png" width="500" style="margin-left: 0">
# <img src="./images/feature_selection02.png" width="500" style="margin-left: 0">
# 
# > ##### 📌noise
# > - 찾고자 하는 정보 이외의 정보이며, 머신러닝에서 훈련을 할 때 알고리즘을 방해하는 주된 요인이다.
# > - 데이터의 품질을 저하시킬뿐만 아니라 알고리즘의 성능과 정확도 등에도 영향을 미치게된다.
# <img src="./images/noise.png" width="600" style="margin-left: 0">

# In[8]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=0)

# 학습 데이터로 모델 훈련
model = SVC().fit(X_train, y_train)
# 훈련된 모델에 테스튼 데이터의 특정 feature를 반복하여 noise시킨 뒤 평균 값으로 중요도 산정
importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)
# 성능이 많이 떨어진 순서(중요도 순)
importance.importances_mean.argsort()[::-1]


# In[10]:


for i in importance.importances_mean.argsort()[::-1]:
    print(f'{iris.feature_names[i]} / {round(importance.importances_mean[i], 2)}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




