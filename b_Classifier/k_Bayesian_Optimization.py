#!/usr/bin/env python
# coding: utf-8

# ### 베이지안 최적화(Bayesian Optimization)
# ##### 참고: https://www.youtube.com/watch?app=desktop&v=w9D8ozS0oC4
# <img src="./images/bayesian_optimization.png" width="700" style="margin-left: -20px">  
# 
# - 최소의 시도로 최적의 답을 찾기 위해서 사용하며, 개별 시도에 있어서 많은 시간 및 자원이 필요할 때도 사용한다.
# - 미지의 함수가 리턴하는 값의 최소 또는 최대값을 만드는 최적의 해를 짧은 반복을 통해 찾아내는 최적화 방식이다.
# - 새로운 데이터를 입력받았을 때 최적 함수를 예측하는 모델을 개선해 나가면서 최적 함수를 도출한다.
# - 대체 모델과 획득 함수로 구성되며, 대체 모델은 획득 함수로부터 최적 입력 값을 추천받은 뒤 이를 기반으로 개선해나간다.  
# 획득 함수는 개선된 대체 모델을 기반으로 다시 최적 입력 값을 계산한다.
# - 함수의 공분산(covariance)이 크다는 것은 곧 불확실성이 크다는 의미이므로 공분산이 최대인 지점을 다음 샘플링 포인트로 선정한다.
# - 📌공분산(Cov)이란, 2개의 확률변수의 선형 관계를 나타내는 값으로서, 서로 다른 변수들 사이에 얼마나 의존하는지를 수치적으로 표현한다.
# 
# ---
# <div style="display: flex">
#     <div>
#         <p style="width: 90%; text-align:center">공분산이 양수일 경우</p>
#         <img src="./images/covariance01.png" width="700" style="margin-left: -30px">
#     </div>
#     <div>
#         <p style="width: 90%; text-align:center">공분산이 음수일 경우</p>
#         <img src="./images/covariance02.png" width="700" style="margin-left: -30px">
#     </div>
#     <div>
#         <p style="width: 90%; text-align:center">공분산이 0일 경우</p>
#         <img src="./images/covariance03.png" width="700" style="margin-left: -30px">
#     </div>
# </div>  
#   
# ##### 🚩공분산이 큰, x = 2인 지점에 샘플링을 하면, 불확실성이 감소하게 된다.
# <div style="display: flex">
#     <div>
#         <img src="./images/bayesian01.png" width="500" style="margin-left: -40px; margin-bottom: 20px">
#     </div>
#     <div>
#         <img src="./images/bayesian02.png" width="465" style="margin-left: -30px">
#     </div>
# </div>  
#   
# ##### 🚩공분산이 큰, x = -0.5인 지점에 샘플링을 하면, 불확실성이 감소하게 된다.
# <div style="display: flex">
#     <div>
#         <img src="./images/bayesian02.png" width="460" style="margin-left: -20px; margin-bottom: 20px">
#     </div>
#     <div>
#         <img src="./images/bayesian03.png" width="455" style="margin-left: -10px">
#     </div>
# </div>  
# 

# In[1]:


import xgboost

print(xgboost.__version__)


# In[2]:


import pandas as pd

corona_df = pd.read_csv('./datasets/corona.csv', low_memory=False)
corona_df.info()


# In[3]:


corona_df = corona_df[~corona_df['Cough_symptoms'].isna()]
corona_df = corona_df[~corona_df['Fever'].isna()]
corona_df = corona_df[~corona_df['Sore_throat'].isna()]
corona_df = corona_df[~corona_df['Headache'].isna()]
corona_df.isna().sum()


# In[4]:


corona_df['Target'] = corona_df['Corona']
corona_df.drop(columns='Corona', axis=1, inplace=True)
corona_df


# In[5]:


corona_df = corona_df[corona_df['Target'] != 'other']
corona_df['Target'].value_counts()


# In[6]:


corona_df = corona_df.drop(columns=['Ind_ID', 'Test_date', 'Sex', 'Known_contact', 'Age_60_above'], axis=1)
corona_df


# In[7]:


from sklearn.preprocessing import LabelEncoder

columns = ['Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Cough_symptoms', 'Target']

for column in columns:
    encoder = LabelEncoder()
    targets = encoder.fit_transform(corona_df[column])
    corona_df.loc[:, column] = targets
    print(f'{column}_classes: {encoder.classes_}')


# In[8]:


# 각 카테고리 값을 정수 타입으로 변환 !
corona_df = corona_df.astype('int16')
corona_df.info()


# In[9]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

features, targets = corona_df.iloc[:, :-1], corona_df.Target

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.3)

# 오버 샘플링
# 검증 데이터나 테스트 데이터가 아닌 학습데이터에서만 오버샘플링 사용할 것
smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

print('SMOTE 적용 전:\n',pd.Series(y_train).value_counts() )
print('SMOTE 적용 후:\n',pd.Series(y_train_over).value_counts() )

# 학습 데이터를 검증 데이터로 분리
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, stratify=y_train, test_size=0.5)

evals = [(X_val_train, y_val_train), (X_val_test, y_val_test)]


# In[10]:


from sklearn.metrics import accuracy_score, precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score
import matplotlib.pyplot as plt
# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation(y_test, prediction, classifier=None, X_test=None):
#     오차 행렬
    confusion = confusion_matrix(y_test, prediction)
#     정확도
    accuracy = accuracy_score(y_test, prediction)
#     정밀도
    precision = precision_score(y_test, prediction)
#     재현율
    recall = recall_score(y_test, prediction)
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
            disp = ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
            disp.ax_.set_title(title)
        plt.show()


# In[11]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
# 목적 함수 정의
def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    max_depth = trial.suggest_int('max_depth', 5, 15)
    
    model = LGBMClassifier(
        boost_from_average=False,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=124
    )
    
    # 모델 학습
    model.fit(X_train_over, y_train_over, early_stopping_rounds=50, eval_set=evals)
    
    # 테스트 데이터에 대한 예측
    prediction = model.predict(X_test)
    
    # 검증 데이터에 대한 평가
    get_evaluation(y_test, prediction)
    auc = roc_auc_score(y_test, prediction)
    return auc


# In[12]:


# conda install -c conda-forge optuna
import optuna
import time

start_time = time.time()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("optuna 수행 시간: {0:.1f} 초 ".format(time.time() - start_time))


# In[13]:


best_params = study.best_params
best_score = study.best_value

print('Best Parameters:', best_params)
print('Best Score:', best_score)


# In[14]:


model = LGBMClassifier(
        boost_from_average=False,
        n_estimators=1000,
        max_depth=12,
        random_state=124
    )

model.fit(X_train_over, y_train_over, early_stopping_rounds=50, eval_set=evals)

prediction = model.predict(X_test)

get_evaluation(y_test, prediction, model, X_test)

