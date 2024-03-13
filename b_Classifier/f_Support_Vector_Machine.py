#!/usr/bin/env python
# coding: utf-8

# ### 서포트 벡터 머신(SVM, Support Vector Machine)
# 기존의 분류 방법들은 '오류율 최소화'의 목적으로 설계되었다면,SVM은 두 부류 사이에 존재하는 '여백 최대화'의 목적으로 설계되었다.
# #####  그림 출처: Eunsu Kim, 빅데희터
# - 분류 문제를 해결하는 지도 학습 모델 중 하나이며, 결정 경계라는 데이터 간 경계를 정의함으로써 분류를 할수 있다.
# - 새로운 데이터가 경계를 기준으로 어떤 방향에 잡히는 지를 확인함으로써 해당 데이터의 카테고리를 예측할 수 있다.
# - 데이터가 어느 카테고리에 속할지 판단하기 위해 가장 적절한 경계인 결정 경계를 찾는 선형 모델 이다.
# <img src="./images/support_vector_machine01.png" width="350" style="margin: 20px; margin-left: 0"> 
# 
# ##### 서포트 벡터(Support vector)
# - 결정 경계를 결정하는 데이터(벡터)들을 서포트 벡터(Support vector)라고 부른다.
# - "Support" vector에서 "Support"인 이유는 바로 이 벡터들이 결정 경계(Decision boundary)를 결정하기 때문이다.
# - 서포트 벡터와 결정 경계간의 거리를 마진(margin)이라고 부르고, 마진이 크면 클 수록 좋은 Decision boundary가 된다.
# - 서포트 벡터들이 결정 경계를 결정하기 때문에 다른 학습 데이터들은 무시될 수 있고, 이 덕분에 SVM이 속도가 빠를 수 있다.
# <img src="./images/support_vector01.png" width="450" style="margin: 20px; margin-left: 0">   
# 
# ##### 결정 경계(Decision boundary)
# - 새로운 데이터가 들어오더라도 결정 경계를 중심으로 두 집단이 멀리 떨어져 있어야 두 집단을 잘 구분할 수 있기 때문에 일반화하기 쉬워진다.
# <img src="./images/support_vector02.png" width="450" style="margin: 20px; margin-left: 0"> 
# - 예측변수의 차원보다 한 차원 낮아지며, N차원 공간에서 한 차원 낮은 N-1차원의 결정경계가 생긴다.  
# 즉, 2차원 공간에서는 초평면이 선으로 결정될 것이고, 고차원에서의 결정 경계는 선이 아닌 평면 이상의 도형이며, 이를 "초평면(Hyperplane)"이라고 부른다.
# <img src="./images/support_vector03.png" width="600" style="margin-left: -80px"> 
# 
# > 결정 경계의 정의
# > - 결정 경계(결정 초평면)를 정의하기 위해서는 서포트 벡터와 결정 경계에 수직하는 가중치 벡터가 필요하다.  
# > - 2차원의 경우 법선벡터 w(a, b)와 원점과의 거리가 d인 임의의 벡터에 대한 직선의 방정식은 ax + by + d = 0이다.  
# > 이 때, N차원의 경우 법선벡터와 원점과의 거리가 d인 임의의 벡터에 대한 방정식은 아래와 같이 표현된다.
# <img src="./images/support_vector_machine03.png" width="400" style="margin: 20px; margin-left: 0">   
# > - 결정 경계의 식은 내적을 사용하면 아래와 같다.  
# > 📌법선벡터(Normal vector)란, 어떠한 직선이나 평면의 기울기나 경사각을 표현할 때, 해당 직선이나 평면에 수직인 벡터를 의미한다.
# <img src="./images/support_vector_machine04.png" width="200" style="margin: 20px; margin-left: 20px">  
# > - 임의의 벡터 u가 결정 경계를 기준으로 왼쪽에 속하는지 아니면 오른쪽에 속하는지 구분하기 위해서는(δ, 델타)를 기준으로 큰지 작은지로 판별할 수 있다.  
# <img src="./images/support_vector_machine05.png" width="145" style="margin: 20px; margin-left: -10px"> 
# > - w와 u 모두 알 수 없기 때문에 δ(델타)도 알 수 없다. 이를 정규화하면 아래의 수식이 나온다.
# > - 양수인지 음수인지 알 수 없기 때문에 범위는 -1 부터 1까지로 정규화를 진행한다.
# <img src="./images/support_vector_machine06.png" width="140" style="margin: 20px; margin-left: -15px">  
# > - δ(델타)를 아래와 같이 치환하면 새로운 식이 나온다.
# > - <strong><sup>*</sup>이 때 1 또는 -1과 같다는 조건</strong>으로 가정했을 경우 정확히 임의의 양쪽 벡터 u위에 총 2개의 직선이 정의될 수 있고, 양쪽 벡터 u는 서포트 벡터가 된다.
# <img src="./images/support_vector_machine07.png" width="450" style="margin: 20px; margin-left: -13px">  
# > - 이제 아래의 공식에 대입하여 마진의 너비(Width)를 구할 수 있다.
# <img src="./images/support_vector_machine08.png" width="600" style="margin: 20px; margin-left: -13px">  
# > - 이 때, 마진을 최대화하기 위해서는 서포트 벡터에 그려진 직선을 이동해야 하므로 "1 또는 -1과 같다는 조건"이라는 제약을 없애기 위해 라그랑주 승수법을 사용한다.  
# > 📌라그랑주 승수법은 등식 제약이 있는 문제를 제약이 없는 문제로 바꾸어 문제를 해결하는 방법이다.
# 
# > - 이를 이용해서 법선벡터인 w(가중치)를 구할 수 있다.
# <img src="./images/support_vector_machine10.png" width="150" style="margin: 20px; margin-left: -13px">  
# > 🚩 따라서 결정 경계를 정의하는 식은 아래와 같다.
# <img src="./images/support_vector_machine11.png" width="200" style="margin: 20px; margin-left: -13px">  
# <img src="./images/support_vector_machine02.png" width="400" style="margin: 20px; margin-left: -13px">  
# 
# ### 하드 마진(Hard margin)과 소프트 마진(Soft margin)
# ##### 하드 마진(Hard margin)
# - 매우 엄격하게 집단을 구분하는 방법으로 이상치를 허용해 주지 않는 방법이다.
# - 이상치를 허용하지 않기 때문에 과적합이 발생하기 쉽고, 최적의 결정경계를 잘못 구분하거나 못찾는 경우가 생길 수 있다.
# <img src="./images/hard_margin.png" width="400" style="margin: 20px; margin-left: -13px">
# 
# ##### 소프트 마진(Soft margin)
# - 각 훈련 데이터 샘플 (x<sub>i</sub>, y<sub>i</sub>)마다 잉여 변수(slack variable) ξ(패널티)를 대응시켜서 샘플이 마진의 폭 안으로 ξ<sub>i</sub>만큼 파고드는 상황을 용인해준다.
# - 이상치를 허용해서 일부 데이터를 잘못 분류하더라도 나머지 데이터를 더욱 잘 분류해주는 방법이다.
# - 이상치 허용으로 인해 데이터의 패턴을 잘 감지하지 못하는 문제점이 생길 수 있다.
# <img src="./images/soft_margin.png" width="600" style="margin: 20px; margin-left: -13px">
# 
# > ##### 🚩정리
# 서포트 벡터 머신 알고리즘을 적용한 SVC 모델의 하이퍼 파라미터인 Regularization cost, C에 값을 전달하여 ξ(패널티)를 조절할 수 있다. C가 크면 클수록 목적 함수에서 오차 ξ<sub>i</sub>의 영향력이 커지게 되기 때문에 마진의 크기가 줄어들고(하드 마진), 반대로 C가 작을 수록 마진의 크기가 늘어난다(소프트 마진). 적절히 조절하여 주게 되면 오히려 성능이 좋아질 수 있다.
# 
# ### 커널 트릭(Kernel trick)
# - 선형으로 완전히 분류할 수 없는 데이터 분포가 있을 경우 소프트 마진을 통해 어느정도의 오류는 허용하는 형태로 분류할 수 있지만  
# 더 잘 분류하기 위해서는 차원을 높여야 한다. 이를 고차원 매핑이라하고, 이 때 커널 트릭을 사용한다.
# - 저차원으로 해결하기 어려운 문제들을 고차원으로 변환시켜서 문제를 해결할 때 사용한다.
# <img src="./images/kernel_trick01.png" width="600" style="margin: 20px; margin-left: -13px">
# - 커널트릭의 종류는 아래와 같고 각기 다른 알고리즘으로서 분류의 정확도를 높이는 데에 사용할 수 있다.
# <img src="./images/kernel_trick02.png" width="600" style="margin: 20px; margin-left: -13px">

# ### 📱핸드폰 성능에 따른 가격 예측
# 
# ##### feature
# - battery_power: 배터리가 한 번에 저장할 수 있는 총 에너지(mAh) 
# - blue: 블루투스 지원 여부  
# - clock_speed: 마이크로프로세서가 명령을 실행하는 속도  
# - dual_sim: 듀얼 심 지원 여부  
# - fc: 전면 카메라 메가 픽셀  
# - four_g: 4G 지원 여부  
# - int_memory: 내장 메모리(GB)  
# - m_dep: 핸드폰 깊이(cm)  
# - mobile_wt: 핸드폰 무게  
# - n_cores: 프로세스 코어 수  
# - pc: 기본 카메라 메가 픽셀  
# - px_height: 픽셀 해상도 높이  
# - px_width: 픽셀 해상도 너비  
# - ram: 랜덤 액세스 메모리(MB)  
# - sc_h: 핸드폰 화면 높이(cm)  
# - sc_w: 핸드폰 화면 너비(cm)  
# - talk_time: 배터리 한 번 충전으로 지속되는 가장 긴 시간  
# - three_g: 3G 지원 여부  
# - touch_screen:  터치 스크린 지원 여부  
# - wifi: Wifi 지원 여부  
# 
# ##### target
# -  price_range: 0(저비용), 1(중비용), 2(고비용), 3(초고비용)

# In[3]:


import pandas as pd

mobile_df = pd.read_csv('./datasets/mobile_train.csv')
mobile_df


# In[4]:


mobile_df.info()


# ### feature별 상관관계 분석
# 
# ##### target과 feature
# - 핸드폰 가격(target)은 RAM에 영향을 많이 미친다.
# 
# ##### featcure와 feature
# - 3G와 4G
# - pc(기본 카메라 메가 픽셀)와 fc(전면 카메라 메가 픽셀)
# - px_weight(픽셀 해상도 너비)와 px_heigth(픽셀 해상도 높이)
# - sc_w(핸드폰 화면 너비)와 sc_h(핸드폰 화면 높이)

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = mobile_df.corr()
fig = plt.figure(figsize=(7, 5))
heatmap = sns.heatmap(corr, cmap="Purples")
heatmap.set_title("Correlation")


# #####  핸드폰 가격에 대한 각 feature 상관 관계 수치

# In[23]:


corr.loc['price_range'].sort_values(ascending=False)


# ##### 핸드폰 가격에 영향을 미치는 정도

# In[24]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
sns.scatterplot(x="battery_power", y="ram", hue="price_range",
              palette="Dark2", data=mobile_df, ax=ax1)
sns.swarmplot(x="four_g", y="ram", hue="price_range",
              palette="Dark2", data=mobile_df, ax=ax2)


# ##### SVC(Support Vector Classifier) 하이퍼 파라미터 튜닝
# ##### 출처: https://amueller.github.io/aml/02-supervised-learning/07-support-vector-machines.html
# - SVC는 gamma와 C를 조절하여 튜닝한다.
# - gamma는 하나의 훈련 샘플이 미치는 영향의 범위를 결정한다.
# - C는 패널티를 조절할 수 있고, 값이 커질수록 결정 경계가 데이터에 더 정확하게 맞춰진다.
# <img src="./images/svm_c_gamma.png" width="700" style="margin-left: -40px">

# In[29]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

# C: 하드 마진, 소프트 마진
# gamma: 하나의 훈련 샘플이 미치는 영향의 범위
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf']
}

# 서포트 벡터 분류
support_vector_classifier = SVC(probability=True)

# 하이퍼 파라미터 튜닝
grid_support_vector = GridSearchCV(support_vector_classifier, param_grid=param_grid, cv=3, refit=True, return_train_score=True)

features, targets = mobile_df.iloc[:, :-1], mobile_df.price_range

X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.3)

sv_pipeline = Pipeline([('normalizer', Normalizer()), ('support_vector_classifier', grid_support_vector)])
sv_pipeline.fit(X_train, y_train)


# In[30]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_support_vector.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[31]:


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


# In[32]:


prediction = sv_pipeline.predict(X_test)
get_evaluation(y_test, prediction)


# In[33]:


import pandas as pd

mobile_test_df = pd.read_csv('./datasets/mobile_test.csv')
mobile_test_df


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = mobile_test_df.corr()
fig = plt.figure(figsize=(7, 5))
heatmap = sns.heatmap(corr, cmap="Purples")
heatmap.set_title("Correlation")


# In[37]:


mobile_test_df.drop(columns=['id'], axis=1, inplace=True)
sv_pipeline.predict(mobile_test_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




