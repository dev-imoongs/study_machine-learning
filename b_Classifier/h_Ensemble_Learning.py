#!/usr/bin/env python
# coding: utf-8

# ### 👨‍👨‍👧‍👦앙상블 학습(Ensemble Learning)
# ##### 그림 출처: grokking-machine-learning(루이스 세라노), Rosy Park, 메이플스토리
# - 어떤 데이터의 값을 예측한다고 할 때, 하나의 모델만 가지고 결과를 도출할 수도 있지만,  
# 여러 개의 모델을 조화롭게 학습(Ensemble Learning)시켜 그 모델들의 예측 결과들을 이용한다면 더 정확한 예측값을 구할 수 있다.
# - 여러 개의 분류기(Classifier)를 생성하고 그 예측을 결합하여 1개의 분류기를 사용할 때보다 더 정확하고 신뢰성 높은 예측을 도출하는 기법이다.
# - 강력한 하나의 모델을 사용하는 것보다 약한 모델 여러 개를 조합하여 더 정확한 예측에 도움을 주는 방식이다.
# - 앙상블 학습의 주요 방법은 배깅(Bagging)과 부스팅(Boosting)이다. 
# <div style="display: flex;">
#     <div>
#         <img src="./images/ensemble_learning01.png" width="400" style="margin-top:20px; margin-left:0">
#     </div>
#     <div>
#         <img src="./images/ensemble_learning02.png" width="400" style="margin-left:0">
#     </div>
# </div>

# ### 앙상블의 유형
# #### 보팅(Voting)
# - "하나의 데이터 세트"에 대해 서로 다른 알고리즘을 가진 분류기를 결합하는 방식이다.
# - 서로 다른 분류기들에 "동일한 데이터 세트"를 병렬로 학습하여 예측값을 도출하고, 이를 합산하여 최종 예측값을 산출해내는 방식을 말한다.  
# 
# > ##### 1. 하드 보팅(Hard Voting)  
# > -  각 분류기가 만든 예측값을 다수결로 투표해서 가장 많은 표를 얻은 예측값을 최종 예측값으로 결정하는 보팅 방식을 말한다.
# <img src="./images/hard_voting.png" width="420" style="margin-left:0">
#   
# > ##### 2. 소프트 보팅(Soft Voting)
# > - 각 분류기가 예측한 타겟별 확률을 평균내어 가장 높은 확률의 타겟을 최종 예측값으로 도출한다.
# <img src="./images/soft_voting.png" width="440" style="margin-left:-5px">  
# > ##### 🏆 하드보팅과 소프트보팅 중 성능이 더 우수한 모델로 선택하면 된다.
# #### 배깅(Bagging, Bootstrap Aggregation)
# - 하나의 데이터 세트에서 "여러 번 중복을 허용하면서 학습 데이터 세트를 랜덤하게 뽑은 뒤(Bootstrap)"  
# 하나의 예측기 여러 개를 병렬로 학습시켜 결과물을 집계(Aggregration)하는 방법이다.
# - Voting 방식과 달리 같은 알고리즘의 분류기를 사용하고 훈련 세트를 무작위로 구성하여 각기 다르게(독립적으로, 병렬로) 학습시킨다.  
# - 학습 데이터가 충분하지 않더라도 충분한 학습효과를 주어 과적합등의 문제를 해결하는데 도움을 준다.
# - 배깅방식을 사용한 대표적인 알고리즘이 바로 랜덤 포레스트 알고리즘이다.
# 
# 📌부트스트랩(bootstrap)은 통계학에서 사용하는 용어로, random sampling을 적용하는 방법을 일컫는 말이다.
# <img src="./images/voting_bagging.png" width="500" style="margin-top: 20px; margin-left:0">  
# 
# ---
# > 🚩정리.  
# <strong style="color: purple">보팅(Voting)</strong>과 <strong style="color: green">배깅(Bagging)</strong>은  
# <strong>여러 개의 분류기가 하나의 데이터 세트를 훈련</strong>한 뒤 투표를 통해 최종 예측 결과를 결정한다는 공통점이 있지만,  
# 보팅은 각각 <strong style="color: purple">동일한 데이터 세트, 다른 분류기</strong> , 배깅은 <strong style="color: green">각각의 데이터 세트(중복 허용), 같은 분류기</strong>를 사용한다.
# ---
# #### 부스팅(Boosting)
# - 이전 분류기의 학습 결과를 토대로 다음 분류기의 학습 데이터의 샘플 가중치를 조정해 "순차적으로" 학습을 진행하는 방법이다.
# - 이전 분류기를 계속 개선해 나가는 방향으로 학습이 진행되고, 오답에 대해 높은 가중치를 부여하므로 정확도가 높게 나타난다.  
# - 높은 가중치를 부여하기 때문에 이상치(outlier)에 취약할 수 있다.
# <img src="./images/boosting01.png" width="600" style="margin-top: 20px; margin-left:0">  
# 
# > ##### 1. Adaboost(Adaptive boosting)
# > - 부스팅에서 가장 기본 기법이며,  
# 결정 트리와 비슷한 알고리즘을 사용하지만 뻗어나가(tree)지 않고 하나의 조건식만 사용(stump)하여 결정한다.
# > - 여러 개의 stump로 구성되어 있으며, 이를 Forest of stumps라고 한다.  
# 📌 stump란, "나무의 잘리고 남은 부분"이라는 뜻이며, 조건식 하나와 두 갈래의 참, 거짓 리프 노드가 있는 형태이다.
# > - 트리와 다르게, 스텀프는 단 하나의 질문으로 데이터를 분류해야하기 때문에 약한 학습기(weak learner)이다.
# <img src="./images/boosting02.png" width="600" style="margin-top: 20px; margin-left:0">  
# > - 결과에 미치는 영향이 큰 스텀프를 Amount of Say가 높다(가중치가 높다)고 한다. 
# > - 각 스텀프의 error는 다음 스텀프의 결과에 영향을 미치고 줄줄이 마지막 스텀프까지 영향을 미친다.
# > - 각 스텀프의 Amount of Say를 수치로 구한 뒤 여러 스텀프의 Amount of Say를 합치면, Total Amount of Say가 나온다.  
# 이를 통해 최종 분류가 된다.  
# > - 하나의 스텀프는 약한 학습기이지만 여러 스텀프를 모으면 강한 학습기가 된다.
# <img src="./images/stump.jpg" width="350" style="margin: 20px; margin-left:-10px">  
# 
# >> ##### Amount of Say
# >> - Total Error가 0이면 Amount of Say는 굉장히 큰 양수이고, Total Error가 1이면 Amount of Say는 굉장히 작은 음수가 된다.  
# >> - Total Error가 0이면 항상 올바른 분류를 한다는 뜻이고, 1이면 항상 반대로 분류를 한다는 뜻이며, Total Error가 0.5일 때는 Amount of Say는 0이다. 0과 1이 반반이기 때문에, 분류기로서 동전 던지기와 같이 분류 결과를 랜덤으로 판단한다.
# <div style="width: 70%; height: 260px; display: flex; margin: 20px; margin-left: 100px">
#     <div>
#         <img src="./images/amount_of_say01.png" width="300">  
#     </div>
#     <div>
#         <img src="./images/amount_of_say02.png" width="250">  
#     </div>
# </div>  
# 
# > ##### 2. GBM(Gradient Boost Machine)
# > - Adaboost와 유사하지만, 에러를 최소화하기 위해 가중치를 업데이트할 때 경사 하강법(Gradient Descent)을 이용한다.  
# > - GBM은 과적합에도 강한 뛰어난 성능을 보이지만, 병렬 처리가 되지 않아 수행 시간이 오래 걸린다는 단점이 있다.  
# 📌 경사 하강법(Gradient Descent)이란, 오류를 최소화하기 위해 Loss funtion을 최소화할 수 있는 최소값까지 반복해서 점차 하강하며 찾아나가는 기법이다.
# <div style="width: 70%; display: flex; margin-top: 20px; margin-left: 40px">
#     <div>
#         <img src="./images/gradient_boost01.png" width="400">  
#     </div>
#     <div>
#         <img src="./images/gradient_boost02.png" width="380">  
#     </div>
# </div>  
# 
# >> ##### 손실 함수(Loss function) 또는 비용 함수(Cost function)  
# >> - 예측 값이 실제 값과 얼마나 유사한지 판단하는 기준이며, 모델 성능의 좋지 않음을 나타내는 지표이다.
# >> - 실제 값과 예측 값의 벗어난 거리를 종합하여, 이를 최소화 하는 것을 목적으로 한다.
# <img src="./images/loss_function.png" width="400" style="margin-top: 20px; margin-left:0">  
# 
# > - 모델 A를 통해 y를 예측하고 남은 잔차(residual)를 다시 B라는 모델을 통해 예측하고 A+B 모델을 통해 y를 예측하는 방식이다.
# > - 잔차를 계속 줄여나가며, 훈련 데이터 세트를 잘 예측하는 모델을 만들 수 있게 된다.
# > -  잔차를 계속 줄이다보면 복잡도가 증가하여 과적합이 일어날 수도 있다는 단점이 있다.  
# 따라서 실제로 GBM을 사용할 때는 수준 높은 Feature engineering을 해서 더 최적화하는 것이 보편적이다.
# > - 📌잔차란(residual), 실제 타겟값 에서 - A모델의 예측 평균값을 뺀 값이다. 즉, 에러의 비율이다.
# > - 학습률(learning rate)이 높을 수록 빠르게 모델의 치우침(bias, 바이어스)을 줄여나가지만, 학습률이 적으면 디테일한 부분을 놓칠 수 있다.
# <img src="./images/gradient_boost03.png" width="700" style="margin-top: 20px; margin-left:0">  
# 
# > ##### 3. XGBoost(eXtra Gradient Boost)  
# > - 트리 기반의 앙상블 학습에서 가장 각광받고 있는 알고리즘 중 하나이며, 분류에 있어서 일반적으로 다른 머신러닝보다 뛰어난 예측 성능을 나타낸다.
# > - GBM에 기반하고 있지만 병렬 CPU 환경에서 병렬 학습이 가능하기 때문에 기존 GBM보다 빠르게 학습을 완료할 수 있다.
# > - 하이퍼 파라미터를 조정하여 분할 깊이를 변경할 수 있지만, tree pruning(가지치기)으로 더 이상 긍정 이득이 없는 분할을 가지치기 해서 분할 수를 더 줄이는 추가적인 장점을 가지고 있다.
# > - 사이킷런의 기본 Estimator를 그대로 상속하여 만들었기 때문에 fit()과 predict()만으로 학습과 예측이 가능하다.
# <img src="./images/xgboost.png" width="900" style="margin-top: 20px; margin-left:-20px">  
# 
# >> ##### 조기 중단 기능(Early Stopping)
# >> - 특정 반복 횟수 만큼 더 이상 손실함수가 감소하지 않으면 수행을 종료할 수 있다.
# >> - 학습 시간을 단축시킬 수 있으며, 최적화 튜닝 시 적절하게 사용 가능하다.
# >> - 반복 횟수를 너무 낮게 설정하면, 최적화 전에 학습이 종료될 수 있기 때문에 조심해야 한다.
# <img src="./images/early_stopping.png" width="400" style="margin-top: 20px; margin-left:-20px">  
# 
# > ##### 4. LightGBM(Light Gradient Boosting Machine)
# > - XGBoost의 향상된 버전으로서 의사결정 트리 알고리즘을 기반으로 순위 지정, 분류 및 기타 여러 기계 학습 작업에 사용할 수 있다.
# > - 기존 부스팅 방식과 마찬가지로 각각의 새로운 분류기가 이전 트리의 잔차를 조정하여 모델이 향상되는 방식으로 결합 되고,  
# 마지막으로 추가된 트리는 각 단계의 결과를 집계하여 강력한 분류기가 될 수 있다.
# > -  XGBoost와 달리 GOSS 알고리즘을 사용하여 수직으로 트리를 성장시킨다. 즉, 다른 알고리즘은 레벨 단위로 성장시키지만, LightGBM은 트리를 리프 단위로 성장시킨다.  
# > - 인코딩을 따로 할 필요 없이 카테고리형 feature를 최적으로 변환하고 이에 따른 노드 분할을 수행한다. astype('category')로 변환할 수 있으며, 이는 LightGBM에서 전처리 패키지의 다양한 인코딩 방식보다 월등하다.
# > - <strong style="color: orange">XGBoost는 소규모 데이터 세트에 더 적합하며 대량의 데이터에 과적합(overfitting)될 위험이 있는 반면</strong> <strong style="color: purple">LightGBM은 대규모 데이터 세트에 더 적합하며 소량의 데이터로는 학습이 덜 되어 있는 과소적합(underfitting)이 발생할 수 있다.</strong>
# <img src="./images/lightGBM01.png" width="600" style="margin-top: 20px; margin-left:-20px">  
# 
# >> ##### Gradient-based One-Side Sampling (GOSS)
# >> - 데이터 세트의 샘플 수를 줄이는 알고리즘으로서, 덜 학습된 데이터(gradient가 큰 데이터)가 영향력이 크다는 가정으로 출발한다.  
# >> - 데이터 세트의 샘플 수를 줄일 때, gradient가 큰 데이터를 유지하고, gradient가 작은 데이터들을 무작위로 drop한다.
# >> - 균등하게 랜덤으로 학습 데이터를 만드는 것 보다 위 방식이 더 정확한 정보 학습을 유도할 수 있다는 것을 증명한다.
# >> - 전체 데이터를 학습하는 것이 아닌 일부 데이터만 학습하기 때문에 속도가 굉장히 빠르다.
# >> ##### GOSS 논문: https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
# >> ##### GOSS 논문 해석 영상: https://www.youtube.com/watch?v=yZGlt3rGtVs
# <div style="display: flex; margin-left: 80px; margin-top:-30px">
#     <div>
#         <img src="./images/lightGBM02.png" width="400"> 
#     </div>
#     <div>
#         <img src="./images/goss01.png" width="300">  
#     </div>
# </div>
# 
# > 🚩 결과의 정확성을 높이고 강화하는 기능 덕분에 LightGBM은 해커톤 및 머신러닝 대회는 물론 Kaggle 대회에서도   
# 가장 많이 사용되는 알고리즘이다.

# ### 보팅(Voting)
# 
# ##### VotingClassifier(n_estimators, voting)
# 
# ###### n_estimators  
# - 추가할 모델 객체를 list형태로 전달한다.
# - 예시) [('DTC',grid_dt_classifier),('SVC',grid_sv_classifier), ('KNN', grid_knn_classifier)]
# 
# ###### voting
# - soft, hard 중 선택한다.
# - default: 'hard'

# ##### 유방암 예측 - 위스콘신 유방암 데이터 세트

# In[ ]:


import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
cancer_df['target'] = cancer.target
cancer_df


# In[ ]:


cancer_df.info()


# In[ ]:


cancer_df.isna().sum()


# In[ ]:


cancer_df['target'].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 5))
correlation_matrix = cancer_df.corr()
heatmap = sns.heatmap(correlation_matrix, cmap='Oranges')
heatmap.set_title("Correlation")


# In[ ]:


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

grid_dt_classifier = GridSearchCV(DecisionTreeClassifier(), param_grid=dt_params, cv=10, refit=True, return_train_score=True, n_jobs=4, error_score='raise')
# 소프트 보팅에서는 각 결정 클래스별 확률이 필요하기 때문에, SVC에 probability를 True로 하여
# predict_proba()를 사용할 수 있도록 해준다(허은상 도움).
grid_svc_classifier = GridSearchCV(SVC(probability=True), param_grid=svm_params, cv=5, refit=True, return_train_score=True, n_jobs=4, error_score='raise')
# KNN에서 Flag오류 발생
# Series 타입의 훈련 데이터에는 flags 속성이 없기 때문에, numpy로 변경한 뒤 훈련시켜야 한다.
grid_knn_classifier = GridSearchCV(KNeighborsClassifier(), param_grid=knn_params, cv=10, refit=True, return_train_score=True, n_jobs=4, error_score='raise')

# 개별 모델을 "하드" 보팅 기반의 앙상블 모델로 구현한 분류기
# 오차 행렬
# [[40  2]
#  [ 3 69]]
# 정확도: 0.9561, 정밀도: 0.9718, 재현율: 0.9583, F1:0.9650, AUC:0.9554

# voting_classifier = VotingClassifier(estimators=[('DTC', grid_dt_classifier)
#                                                  , ('SVC', grid_svc_classifier)
#                                                  , ('KNN', grid_knn_classifier)]
#                                      , voting='hard')


# 개별 모델을 "소프트" 보팅 기반의 앙상블 모델로 구현한 분류기
# 오차 행렬
# [[37  5]
#  [ 1 71]]
# 정확도: 0.9474, 정밀도: 0.9342, 재현율: 0.9861, F1:0.9595, AUC:0.9335

voting_classifier = VotingClassifier(estimators=[('DTC', grid_dt_classifier)
                                                 , ('SVC', grid_svc_classifier)
                                                 , ('KNN', grid_knn_classifier)]
                                     , voting='soft')

# 데이터 세트 분리
features, targets = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2)

# VotingClassifier 학습/예측/평가
voting_classifier.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score
# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation(y_test, prediction, classifier=None, X_test=None):
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
    roc_auc = roc_auc_score(y_test, prediction)

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


# In[ ]:


prediction = voting_classifier.predict(X_test)
get_evaluation(y_test, prediction, voting_classifier, X_test)


# In[ ]:


# 개별 모델의 학습/예측/평가.
classifiers = [grid_dt_classifier, grid_svc_classifier, grid_knn_classifier]
for classifier in classifiers:
    classifier.fit(X_train , y_train)
    prediction = classifier.predict(X_test)
    class_name= classifier.best_estimator_.__class__.__name__
    print(f'# {class_name}')
    get_evaluation(y_test, prediction, classifier, X_test)


# ### 배깅(Bagging) - 랜덤 포레스트(Random Forest)
# 
# #### RandomForestClassifier(n_estimators, min_samples_split, min_samples_leaf, n_jobs)
# 
# ###### n_estimators  
# - 생성할 tree의 개수를 작성한다.
# - default: 50
# 
# ###### min_samples_split
# - 분할 할 수 있는 샘플 수이다.
# 
# ##### min_samples_leaf
# - 분할했을 때 leaf의 샘플 수이다.

# In[ ]:


import pandas as pd

car_df = pd.read_csv('./datasets/car.csv')
car_df


# In[ ]:


car_df.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

obj_columns = ['Price', 'Main_cost', 'Doors', 'Persons', 'Lug_cap', 'Safety', 'Decision']
encoders = []
for column in obj_columns:
    encoder = LabelEncoder()
    car_df[column] = encoder.fit_transform(car_df[column].tolist())
    encoders.append(encoder)
    print(encoder.classes_)


# In[ ]:


car_df['Price'].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5, 4))
correlation_matrix = car_df.corr()
sns.heatmap(correlation_matrix, cmap="Purples")


# In[ ]:


car_df.iloc[:, 1:].hist(figsize=(10, 8))


# In[ ]:


# car_df['Decision'].value_counts()

# dicision_0 = car_df[car_df['Decision'] == 0]
# dicision_1 = car_df[car_df['Decision'] == 1]
# dicision_2 = car_df[car_df['Decision'] == 2].sample(384)
# dicision_3 = car_df[car_df['Decision'] == 3]

# banlance_car_df = pd.concat([dicision_0, dicision_1, dicision_2, dicision_3])
# banlance_car_df['Decision'].value_counts()


# In[ ]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'max_depth': [4, 6, 8, 10, 12],
    'min_samples_split': [6, 12, 18, 24],
    'min_samples_leaf': [4, 8, 16]
}

random_forest_classifier = RandomForestClassifier(n_estimators=100)

# features, targets = banlance_car_df.iloc[:, 1:], banlance_car_df.Price
features, targets = car_df.iloc[:, 1:], car_df.Price

X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.3)

grid_random_forest = GridSearchCV(random_forest_classifier, param_grid=param_grid, cv=10, n_jobs=4)

grid_random_forest.fit(X_train, y_train)


# In[ ]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_random_forest.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score
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
#     ROC-AUC
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


# In[ ]:


prediction = grid_random_forest.predict(X_test)
get_evaluation(y_test, prediction, grid_random_forest, X_test)


# ##### car.csv 데이터 세트는 전체적으로 분포는 괜찮지만, 데이터의 양이 부족하여 Under fitting이 발생한다.

# ### 부스팅(Boosting) - 에이다부스트(Adaptive Boost)
# 
# #### AdaBoostClassifier(base_estimators, n_estimators, learning_rate)
# 
# ###### base_estimators  
# - 학습에 사용하는 알고리즘 선택한다.  
# - default: DecisionTreeClassifier(max_depth = 1)
# 
# ###### n_estimators
# - 생성할 약한 학습기의 개수를 지정한다.  
# - default : 50
# 
# ##### learning_rate
# - 학습을 진행할 때마다 적용하는 학습률(0~1 사이의 값)이며, 약한 학습기가 순차적으로 오류값을 보정해나갈 때 적용하는 계수이다.  
# - 낮은만큼 최소 손실값을 찾아 예측 성능이 높아질 수 있지만, 그 만큼 많은 수의 트리가 필요하고 시간이 많이 소요된다.  
# - default : 1

# In[1]:


import pandas as pd
water_df = pd.read_csv("./datasets/water_potability.csv")
water_df


# In[2]:


water_df.info()


# In[3]:


water_df.isna().sum()


# In[4]:


water_df.iloc[:, :-1].hist(figsize=(10, 20), bins=100)


# In[5]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
water_scaled = scale.fit_transform(water_df)


# In[6]:


water_scaled_df = pd.DataFrame(water_scaled, columns=water_df.columns)
water_scaled_df[~water_scaled_df['Solids'].between(-1.96, 1.96)]


# In[7]:


water_scaled_df = water_scaled_df[water_scaled_df.Solids.between(-1.96, 1.96)]
water_scaled_df


# In[8]:


water_scaled_df.iloc[:, :-1].hist(figsize=(10, 20), bins=100)


# In[9]:


water_df = water_df.iloc[water_scaled_df.index, :]
water_df


# In[10]:


water_df = water_df.reset_index(drop=True)
water_df


# In[11]:


water_df.isna().sum()
water_df.Sulfate = water_df.Sulfate.fillna(0)
water_df


# In[12]:


water_df.isna().sum()


# In[13]:


water_df.Trihalomethanes = water_df.Trihalomethanes.fillna(0)
water_df.isna().sum()


# In[14]:


water_df.ph = water_df.ph.fillna(water_df.ph.median())
water_df.isna().sum()


# In[15]:


water_df.duplicated().sum()


# In[16]:


from sklearn.preprocessing import MinMaxScaler

features, targets = water_df.iloc[:, :-1], water_df.Potability

water_scaled = MinMaxScaler().fit_transform(features)
water_scaled_df = pd.DataFrame(water_scaled, columns=features.columns)
water_scaled_df


# In[17]:


water_scaled_df['Potability'] = water_df['Potability']
water_scaled_df


# In[18]:


water_scaled_df.Potability.value_counts()


# In[19]:


target_0 = water_scaled_df[water_scaled_df.Potability == 0].sample(1211)
target_1 = water_scaled_df[water_scaled_df.Potability == 1]

balance_water_df = pd.concat([target_0, target_1])
balance_water_df.Potability.value_counts()


# In[20]:


balance_water_df.reset_index(drop=True, inplace=True)


# In[21]:


balance_water_df.isna().sum()
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.3, 0.5, 0.7]
}

features, targets = balance_water_df.iloc[:, :-1], balance_water_df.Potability
X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2)

grid_ada_boost = GridSearchCV(AdaBoostClassifier(), param_grid=param_grid, cv=10, n_jobs=4)
grid_ada_boost.fit(X_train, y_train)


# In[22]:


prediction = grid_ada_boost.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score
import matplotlib.pyplot as plt
# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation(y_test, prediction, classifier=None, X_test=None):
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
    roc_auc = roc_auc_score(y_test, prediction)

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


# In[26]:


get_evaluation(y_test, prediction, grid_ada_boost, X_test)


# In[27]:


def get_evaluation_by_thresholds(y_test, prediction_proba_class1, thresholds):
    
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(prediction_proba_class1) 
        custom_prediction = binarizer.transform(prediction_proba_class1)
        print('임곗값:', threshold)
        get_evaluation(y_test, custom_prediction)


# In[28]:


from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve

prediction_prob = grid_ada_boost.predict_proba(X_test)
prediction_prob_class1 = prediction_prob[:, 1].reshape(-1, 1)

precision, recall, thresolds = precision_recall_curve(y_test, prediction_prob_class1)

get_evaluation_by_thresholds(y_test, prediction_prob_class1, thresolds)


# In[29]:


prediction = Binarizer(threshold=0.5002843602697008).fit_transform(prediction_prob_class1)
get_evaluation(y_test, prediction, grid_ada_boost, X_test)


# ### 부스팅(Boosting) - GBM(Gradient Boosting Machine)
# 
# #### GradientBoostingClassifier(n_estimators, loss, learning_rate, subsample)
# 
# ##### n_estimators
# - 약한 학습기의 개수이며, 개수가 많을 수록 일정 수준까지 좋아질 수 있지만 많을 수록 수행 시간이 오래걸린다.
# - default: 100
# 
# ##### loss
# - 경사 하강법에서 사용할 비용함수를 지정한다.
# - default: 'log_loss'
# > ##### 🚩출처  
# > https://library.virginia.edu/data/articles/understanding-deviance-residuals  
# > https://www.youtube.com/watch?v=lAq96T8FkTw&list=LLypIXWIsUMIMvCa6zQfOZmQ&index=14  
# > 러닝머신
# > 1. log_loss(deviance)
# > - 각 잔차를 계산하여 평균을 내는 알고리즘(로지스틱 회귀 알고리즘 방식),   
# 오래된 데이터의 영향과 최신 데이터의 영향이 비슷해짐.  
# > 2. exponential  
# > - Weight를 부여하는 방식을 도입(AdaBoost 알고리즘 방식)하였으며,   
# 데이터의 시간 흐름에 따라 지수적으로 감쇠하도록 설계한 알고리즘.
# > - 📌지수적 감쇠란, 어떤 양이 그 양에 비례하는 속도로 감소하는 것을 의미하며, 0~1 사이의 값이 제곱이 되면 더 작아지기 때문에 오래된 데이터일수록 현재의 경향을 표현하는 데에 더 적은 영향을 미치게 한다.
# 
# ---
# > - 시계열 데이터: exponential, 일반 데이터: deviance
# > - 📌시계열 데이터란, 일정 시간 간격으로 측정된 데이터의 시간적 순서를 나타내는 데이터를 의미한다.
# 
# ##### learning_rate
# - GBM이 학습을 진행할 때마다 적용하는 학습률이다.
# - 오류를 개선해 나가는 데에 적용하는 계수이며, 0~1사이로 값을 지정한다.
# - 높게 설정하면 최소 오류값을 찾지 못하고 지나쳐버리지만 빠른 수행이 가능하고,  
# 낮게 설정하면 최소 오류 값을 찾아서 성능은 높아지지만, 너무 많은 시간이 소요된다.
# <img src="./images/learning_rate.png" width="600" style="margin-left: 0">  
# 
# ##### subsample
# - 학습에 사용하는 데이터의 샘플링 비율이다.
# - default: 1 (100%)
# - 과적합 방지 시 1보다 작은 값으로 설정한다.

# In[1]:


import pandas as pd

car_df = pd.read_csv('./datasets/car.csv')
car_df


# In[2]:


from sklearn.preprocessing import LabelEncoder

obj_columns = ['Price', 'Main_cost', 'Doors', 'Persons', 'Lug_cap', 'Safety', 'Decision']
encoders = []
for column in obj_columns:
    encoder = LabelEncoder()
    car_df[column] = encoder.fit_transform(car_df[column].tolist())
    encoders.append(encoder)
    print(encoder.classes_)


# In[5]:


import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier

features, targets = car_df.iloc[:, 1:], car_df.Price

X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2)

start_time = time.time()

param_grid = {
    'n_estimators': [50, 100, 500],
    'learning_rate': [0.3, 0.5, 0.7]
}

grid_gradient_boosting = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, cv=3)

grid_gradient_boosting.fit(X_train, y_train)
print(f'GBM 수행 시간: {time.time() - start_time}')


# In[6]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_gradient_boosting.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[9]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score
import matplotlib.pyplot as plt
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
#     ROC-AUC
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


# In[10]:


prediction = grid_gradient_boosting.predict(X_test)
get_evaluation(y_test, prediction, grid_gradient_boosting, X_test)


# ### 부스팅(Boosting) - XGBoost(eXtra Gradient Boost)
# 
# #### XGBClassifier(n_estimators, learning_rate, subsample)
# 
# ##### n_estimators
# - 약한 학습기의 개수이며, 개수가 많을 수록 일정 수준까지 좋아질 수 있지만 많을 수록 수행 시간이 오래걸린다.
# - default: 100
# 
# ##### learning_rate
# - GBM이 학습을 진행할 때마다 적용하는 학습률이다.
# - 오류를 개선해 나가는 데에 적용하는 계수이며, 0~1사이로 값을 지정한다.
# - 높게 설정하면 최소 오류값을 찾지 못하고 지나쳐버리지만 빠른 수행이 가능하고,  
# 낮게 설정하면 최소 오류 값을 찾아서 성능은 높아지지만, 너무 많은 시간이 소요된다.
# <img src="./images/learning_rate.png" width="600" style="margin-left: 0">  
# 
# ##### subsample
# - 학습에 사용하는 데이터의 샘플링 비율이다.
# - default: 1 (100%)
# - 과적합 방지 시 1보다 작은 값으로 설정한다.
# 
# #### fit(X_train, y_train, eval_set, early_stopping_rounds)
# 
# ##### eval_set
# - 예측 오류값을 줄일 수 있도록 반복적하면서 학습이 진행되는데,   
# 이때 학습은 학습 데이터로 하고 예측 오류값 평가는 eval_set로 지정된 검증 세트로 평가한다.
# 
# ##### early_stopping_rounds
# - 지정한 횟수동안 더 이상 오류가 개선되지 않으면 더 이상 학습은 진행하지 않는다.

# In[11]:


import xgboost
print(xgboost.__version__)


# ##### 코로나 바이러스(COVID) 예측

# In[27]:


import pandas as pd

corona_df = pd.read_csv('./datasets/corona.csv', low_memory=False)
corona_df.info()


# In[28]:


corona_df = corona_df[~corona_df['Cough_symptoms'].isna()]
corona_df = corona_df[~corona_df['Fever'].isna()]
corona_df = corona_df[~corona_df['Sore_throat'].isna()]
corona_df = corona_df[~corona_df['Headache'].isna()]
corona_df['Age_60_above'] = corona_df['Age_60_above'].fillna('No')
corona_df['Sex'] = corona_df['Sex'].fillna('unknown')
corona_df.isna().sum()


# In[29]:


corona_df['Target'] = corona_df['Corona']
corona_df.drop(columns='Corona', axis=1, inplace=True)


# In[30]:


print(corona_df['Target'].value_counts())
display(corona_df)


# In[31]:


corona_df = corona_df[corona_df['Target'] != 'other']
print(corona_df['Target'].value_counts())


# In[32]:


corona_df = corona_df.drop(columns=['Ind_ID', 'Test_date', 'Sex', 'Age_60_above', 'Known_contact'], axis=1)
corona_df


# ##### 레이블 인코딩

# In[33]:


from sklearn.preprocessing import LabelEncoder

columns = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Target']

for column in columns:
    encoder = LabelEncoder()
    targets = encoder.fit_transform(corona_df[column])
    corona_df.loc[:, column] = targets
    print(f'{column}_classes: {encoder.classes_}')


# In[34]:


corona_df = corona_df.reset_index(drop=True)
corona_df


# In[35]:


corona_df = corona_df.astype('int16')
corona_df.info()


# ##### 하이퍼 파라미터 튜닝 및 교차 검증

# In[36]:


from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [50, 100, 500],
    'learning_rate': [0.3, 0.5, 0.7]
}

xgb = XGBClassifier()

features, targets = corona_df.iloc[:, :-1], corona_df.Target

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2)

# 학습 데이터를 검증 데이터로 분리
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, stratify=y_train, test_size=0.3)

evals = [(X_val_train, y_val_train), (X_val_test, y_val_test)]

grid_xgb = GridSearchCV(xgb, param_grid, cv=3, refit=True, return_train_score=True, n_jobs=-1, error_score='raise')
grid_xgb.fit(X_train, y_train, early_stopping_rounds=50, eval_set=evals)


# In[37]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_xgb.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[38]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score
import matplotlib.pyplot as plt
# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation(y_test, prediction, classifier=None, X_test=None):
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
    roc_auc = roc_auc_score(y_test, prediction)

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


# In[40]:


prediction = grid_xgb.predict(X_test)
get_evaluation(y_test, prediction, grid_xgb, X_test)


# In[41]:


def get_evaluation_by_thresholds(y_test, prediction_proba_class1, thresholds):
    
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(prediction_proba_class1) 
        custom_prediction = binarizer.transform(prediction_proba_class1)
        print('임곗값:', threshold)
        get_evaluation(y_test, custom_prediction)


# In[43]:


from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve

prediction = grid_xgb.predict(X_test)
prediction_proba_class1 = grid_xgb.predict_proba(X_test)[:, 1].reshape(-1, 1)
precision, recall, thresholds = precision_recall_curve(y_test, prediction_proba_class1)

get_evaluation_by_thresholds(y_test, prediction_proba_class1, thresholds)


# In[45]:


prediction = Binarizer(threshold=0.032460973).fit_transform(prediction_proba_class1)
get_evaluation(y_test, prediction, grid_xgb, X_test)


# ### 부스팅(Boosting) - LightGBM(Light Gradient Boosting Machine)
# 
# #### LGBMClassifier(n_estimators, learning_rate, subsample)
# 
# ##### n_estimators
# - 약한 학습기의 개수이며, 개수가 많을 수록 일정 수준까지 좋아질 수 있지만 많을 수록 수행 시간이 오래걸린다.
# - default: 100
# 
# ##### learning_rate
# - GBM이 학습을 진행할 때마다 적용하는 학습률이다.
# - 오류를 개선해 나가는 데에 적용하는 계수이며, 0~1사이로 값을 지정한다.
# - 높게 설정하면 최소 오류값을 찾지 못하고 지나쳐버리지만 빠른 수행이 가능하고,  
# 낮게 설정하면 최소 오류 값을 찾아서 성능은 높아지지만, 너무 많은 시간이 소요된다.
# <img src="./images/learning_rate.png" width="600" style="margin-left: 0">  
# 
# ##### subsample
# - 학습에 사용하는 데이터의 샘플링 비율이다.
# - default: 1 (100%)
# - 과적합 방지 시 1보다 작은 값으로 설정한다.
# 
# #### fit(X_train, y_train, eval_set, early_stopping_rounds)
# 
# ##### eval_set
# - 예측 오류값을 줄일 수 있도록 반복적하면서 학습이 진행되는데,   
# 이때 학습은 학습 데이터로 하고 예측 오류값 평가는 eval_set로 지정된 검증 세트로 평가한다.
# 
# ##### early_stopping_rounds
# - 지정한 횟수동안 더 이상 오류가 개선되지 않으면 더 이상 학습은 진행하지 않는다.

# In[46]:


import lightgbm

print(lightgbm.__version__)


# ##### 코로나 바이러스(COVID) 예측

# In[79]:


import pandas as pd

corona_df = pd.read_csv('./datasets/corona.csv', low_memory=False)
corona_df.info()


# In[80]:


corona_df = corona_df[~corona_df['Cough_symptoms'].isna()]
corona_df = corona_df[~corona_df['Fever'].isna()]
corona_df = corona_df[~corona_df['Sore_throat'].isna()]
corona_df = corona_df[~corona_df['Headache'].isna()]
corona_df['Age_60_above'] = corona_df['Age_60_above'].fillna('No')
corona_df['Sex'] = corona_df['Sex'].fillna('unknown')
corona_df.isna().sum()


# In[81]:


corona_df['Target'] = corona_df['Corona']
corona_df.drop(columns='Corona', axis=1, inplace=True)


# In[82]:


corona_df = corona_df[corona_df['Target'] != 'other']
print(corona_df['Target'].value_counts())


# In[83]:


corona_df = corona_df.drop(columns=['Ind_ID', 'Test_date', 'Sex', 'Age_60_above', 'Known_contact'], axis=1)
corona_df


# In[84]:


corona_df = corona_df.reset_index(drop=True)
corona_df


# In[85]:


corona_df = corona_df.astype('category')
corona_df.info()


# ##### 하이퍼 파라미터 튜닝 및 검증

# In[86]:


from sklearn.model_selection import GridSearchCV, train_test_split
from lightgbm import LGBMClassifier

param_grid = {
    'n_estimators': [50, 100, 500],
    'learning_rate': [0.3, 0.5, 0.7]
}

lgbm = LGBMClassifier()

features, targets = corona_df.iloc[:, :-1], corona_df.Target

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2)

# 학습 데이터를 검증 데이터로 분리
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, stratify=y_train, test_size=0.3)

evals = [(X_val_train, y_val_train), (X_val_test, y_val_test)]

grid_lgbm = GridSearchCV(lgbm, param_grid, cv=3, refit=True, return_train_score=True, n_jobs=-1, error_score='raise')
grid_lgbm.fit(X_train, y_train, early_stopping_rounds=50, eval_set=evals)


# In[87]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_lgbm.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[88]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation(y_test, prediction, prediction_proba_class1, classifier=None, X_test=None):
#     오차 행렬
    confusion = confusion_matrix(y_test, prediction)
#     정확도
    accuracy = accuracy_score(y_test, prediction)
#     정밀도
    precision = precision_score(y_test, prediction, pos_label="positive")
#     재현율
    recall = recall_score(y_test, prediction, pos_label="positive")
#     F1 score
    f1 = f1_score(y_test, prediction, pos_label="positive")
#     ROC-AUC
    roc_auc = roc_auc_score(y_test, prediction_proba_class1)

    print('오차 행렬')
    print(confusion)
    print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율: {2:.4f}, F1:{3:.4f}, AUC:{4:.4f}'.format(accuracy , precision ,recall, f1, roc_auc))
    print("#" * 75)
    
    if classifier is not None and  X_test is not None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
        titles_options = [("Confusion matrix", None), ("Normalized confusion matrix", "true")]

        for (title, normalize), ax in zip(titles_options, axes.flatten()):
            disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
            disp.ax_.set_title(title)
        plt.show()


# In[89]:


prediction = grid_lgbm.predict(X_test)
prediction_proba_class1 = grid_lgbm.predict_proba(X_test)[:, 1].reshape(-1, 1)
get_evaluation(y_test, prediction, prediction_proba_class1)


# In[90]:


from sklearn.preprocessing import Binarizer

def get_evaluation_by_thresholds(y_test, prediction_proba_class1, thresholds):
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(prediction_proba_class1) 
        custom_prediction = binarizer.transform(prediction_proba_class1)
        custom_prediction = custom_prediction.astype('str')
        custom_prediction[custom_prediction == '0.0'] = 'negative'
        custom_prediction[custom_prediction == '1.0'] = 'positive'
        print('임곗값:', threshold)
        get_evaluation(y_test, custom_prediction, prediction_proba_class1)


# In[91]:


import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve

prediction_proba_class1 = grid_lgbm.predict_proba(X_test)[:, 1].reshape(-1, 1)

# precision, recall, thresholds = precision_recall_curve(y_test, prediction_proba_class1, pos_label="positive")
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
get_evaluation_by_thresholds(y_test, prediction_proba_class1, thresholds)


# In[95]:


from sklearn.preprocessing import Binarizer
prediction = Binarizer(threshold=0.1).fit_transform(prediction_proba_class1)
prediction = prediction.astype('str')
prediction[prediction == '0.0'] = 'negative'
prediction[prediction == '1.0'] = 'positive'
get_evaluation(y_test, prediction, prediction_proba_class1)


# ##### permutation_importance

# In[96]:


from sklearn.inspection import permutation_importance

importance = permutation_importance(grid_lgbm, X_test, y_test, n_repeats=100, random_state=0)
corona_df.columns[importance.importances_mean.argsort()[::-1]]

