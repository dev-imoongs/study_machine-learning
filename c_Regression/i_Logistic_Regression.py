#!/usr/bin/env python
# coding: utf-8

# ### Logistic Regression (로지스틱 회귀)
# - 선형 회귀 방식을 분류에 적용한 알고리즘이다.
# - 선형 함수의 회귀 최적선을 찾는 것이 아니라 시그모이드(sigmoid) 함수의 최적선을 찾고 이 함수의 반환값을 확률로 간주하여 확률에 따라 분류를 결정한다.
# - 로지스틱 회귀는 다중 분류도 가능하지만, 주로 이진 분류에 활용되며, 예측 값이 예측 확률이다.
# - 독립변수를 input값으로 받아 종속변수가 1이 될 확률을 결과값으로 하는 sigmoid 함수를 찾는 과정이다.
# <img src="./images/sigmoid01.png" width="400" style="margin-left: 0">  
# 
# > 📌 시그모이드(sigmoid) 함수는 입력 값을 넣었을 때 1일 확률은 얼마인지 알아낼 수 있다.  
# > ##### 베이지안 추론을 통한 시그모이드 식 증명  
# > - B가 A<sub>1</sub> 조건에 속할 확률을 구한다.
# <img src="./images/sigmoid02.png" width="250" style="margin-top: -2px; margin-bottom:20px; margin-left: -20px">  
# > - 각 분자와 분모를 분자로 나눠준다.
# > - 아래 로그의 성질을 이용해서 자연상수 e를 대입한다.
# <img src="./images/sigmoid03.png" width="100" style="margin-bottom:20px; margin-left: 0">  
# > - A/B = e<sup>-log(B/A)</sup>
# > - 이를 통해 아래의 식이 나온다.
# <img src="./images/sigmoid04.png" width="250" style="margin-bottom:20px; margin-left: 0">  
# > - likelihood ratio(우도): 어떤 현상이 있을 때 그 현상이 어떤 모집단에 속할 가능성을 의미한다.  
# > 예를 들어, '양성'판정을 받은 모집단이 있고, '암'이라는 현상이 있고, '암이 아님'이라는 현상이 있을 때 '암'이라는 현상일 때 '양성'이라는 모집단에 속할 가능성을 우도라고 한다.  
# 암에 걸린 사람들을 대상으로 암 진단용 시약으로 검사를 했더니 99%가 양성일 경우 우도 99%이다.
# > - P(C<sub>1</sub>|x) : 조건부 확률
# > - P(x|C<sub>1</sub>) : 우도
# > - prior odds ratio: odds를 통해 특정 확률을 역으로 알 수 있다. 즉, 경기에서 지는 확률만 가지고 odds를 사용하여 역확률인 이기는 확률을 구할 수 있다는 의미이다. 두 가지 상황에서의 확률 중 한 가지 상황에서는 0 ~ 1사이가 나오지만, 반대 상황에서는 1 ~ 무한대가 나오므로 균형을 맞추고자 log를 씌워준다(이를 logit이라 부른다).   
# logit함수는 0에서 1까지의 확률값과 -∞에서 ∞ 사이의 확률값을 표현해주는 함수이며, 시그모이드의 역함수이다.  
# 
# - 🚩 y의 범위는 [0, 1]이고, 특성값 x의 범위는 [-∞, ∞]이므로 관계를 말할 수 없지만, logit 변환은 [0, 1]의 범위를 가지는 확률을 [-∞, ∞]로 바꿔주기 때문에, 예측값(y)과 예측값을 만들어내는 특성값(x)의 관계를 선형 관계(y = wx+b)로 만들 수 있게 한다.

# ##### LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')
# - penalty: 원하는 규제를 선택한다.
# - C: 서포트 벡터 머신과 마찬가지로 값이 작을수록 규제가 심해지기 때문에 더 강력한 정규화가 지정된다.
# - solver: {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, default='lbfgs'  
# 최적화 문제에 사용할 알고리즘을 선택할 수 있으며, 데이터 세트가 작을 경우 'liblinear'가 좋고, 큰 경우 'sag'와 'saga'가 더 좋다.  
# 다중 분류는 'newton-cg', 'sag', 'saga' 및 'lbfgs'만 처리 가능하다.  
# 사용 가능한 규제는 아래와 같다.
# > - lbfgs[‘l2’, None]
# > - liblinear - [‘l1’, ‘l2’]
# > - newton-cg - [‘l2’, None]
# > - newton-cholesky - [‘l2’, None]
# > - sag - [‘l2’, None]
# > - saga - [‘elasticnet’, ‘l1’, ‘l2’, None] 

# In[2]:


import pandas as pd

corona_df = pd.read_csv('./datasets/corona.csv', low_memory=False)
corona_df.info()


# In[3]:


corona_df = corona_df[~corona_df['Cough_symptoms'].isna()]
corona_df = corona_df[~corona_df['Fever'].isna()]
corona_df = corona_df[~corona_df['Sore_throat'].isna()]
corona_df = corona_df[~corona_df['Headache'].isna()]
corona_df['Age_60_above'] = corona_df['Age_60_above'].fillna('No')
corona_df['Sex'] = corona_df['Sex'].fillna('unknown')
corona_df.isna().sum()


# In[4]:


corona_df['Target'] = corona_df['Corona']
corona_df.drop(columns='Corona', axis=1, inplace=True)
corona_df


# In[5]:


corona_df['Target'].value_counts()


# In[6]:


corona_df = corona_df[corona_df['Target'] != 'other']
corona_df['Target'].value_counts()


# In[7]:


from sklearn.preprocessing import LabelEncoder

columns = ['Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Age_60_above', 'Target']

for column in columns:
    encoder = LabelEncoder()
    targets = encoder.fit_transform(corona_df[column])
    corona_df.loc[:, column] = targets
    print(f'{column}_classes: {encoder.classes_}')


# In[8]:


corona_df = corona_df.drop(columns=['Ind_ID', 'Test_date', 'Sex', 'Known_contact'], axis=1)
corona_df


# In[9]:


corona_df = corona_df.reset_index(drop=True)
corona_df


# In[10]:


# 각 카테고리 값을 정수 타입으로 변환 !
corona_df = corona_df.astype('int16')
corona_df.info()


# In[11]:


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


# In[12]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

features, targets = corona_df.iloc[:,:-1], corona_df.Target

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, stratify=targets, random_state=124)

smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

lg = LogisticRegression(solver='liblinear', penalty='l2', random_state=124)
lg.fit(X_train_over, y_train_over)
prediction = lg.predict(X_test)


# In[13]:


get_evaluation(y_test, prediction, lg, X_test)


# In[14]:


def get_evaluation_by_thresholds(y_test, prediction_proba_class1, thresholds):
    
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(prediction_proba_class1) 
        custom_prediction = binarizer.transform(prediction_proba_class1)
        print('임계값:', threshold)
        get_evaluation(y_test, custom_prediction)


# In[15]:


from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve

prediction = lg.predict(X_test)
prediction_proba = lg.predict_proba(X_test)[:, 1].reshape(-1, 1)
precision, recall, thresholds = precision_recall_curve(y_test, prediction_proba)

get_evaluation_by_thresholds(y_test, prediction_proba, thresholds)


# In[16]:


prediction = Binarizer(threshold=0.6488399698426035).fit_transform(prediction_proba)
get_evaluation(y_test, prediction, lg, X_test)


# In[17]:


from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

params = {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear', 'saga']}

features, targets = corona_df.iloc[:,:-1], corona_df.Target

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, stratify=targets, random_state=124)

smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

grid_lg = GridSearchCV(LogisticRegression(max_iter=1000, penalty='l2', random_state=124), param_grid=params, cv=3, refit=True)
grid_lg.fit(X_train_over, y_train_over)
prediction = grid_lg.predict(X_test)


# In[18]:


# DataFrame으로 변환
scores_df = pd.DataFrame(grid_lg.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']].sort_values(by='rank_test_score')


# In[19]:


get_evaluation(y_test, prediction, grid_lg, X_test)


# In[20]:


from sklearn.preprocessing import Binarizer
from sklearn.metrics import precision_recall_curve

prediction = grid_lg.predict(X_test)
prediction_proba = grid_lg.predict_proba(X_test)[:, 1].reshape(-1, 1)
precision, recall, thresholds = precision_recall_curve(y_test, prediction_proba)

get_evaluation_by_thresholds(y_test, prediction_proba, thresholds)


# In[21]:


prediction = Binarizer(threshold=0.6541459441734264).fit_transform(prediction_proba)
get_evaluation(y_test, prediction, grid_lg, X_test)

