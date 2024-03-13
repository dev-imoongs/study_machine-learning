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


# Pos : 포지션  
# 3P : 3점  
# 2P : 2점  
# TRB : 리바운드 점유율  
# AST : 어시스트 대비 턴오버 비율  
# STL : 스틸  
# BLK : 블락슛  

# In[2]:


path = 'https://raw.githubusercontent.com/childult-programmer/Tistory/master/data/2020-21_NBA_Player_Stats_Per_Game.csv'
df = pd.read_csv(path)
df


# In[3]:


features , targets = df.drop(columns=['Player','Pos'],axis=1), df.loc[:,'Pos']

X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2)


# ### 의사 결정 트리

# In[4]:


decision_tree_classifier = DecisionTreeClassifier()


# In[5]:


decision_tree_classifier.fit(X_train, y_train)


# In[6]:


target_names = targets.value_counts().keys().to_list()
feature_names = features.columns.to_list()


# In[7]:


export_graphviz(decision_tree_classifier
                , out_file="./images/NBA_Player_tree01.dot"
                , class_names= target_names
                , feature_names= feature_names
                , impurity=True
                , filled=True)


# In[8]:


with open("./images/NBA_Player_tree01.dot") as f:
    dot_graph = f.read()

drugs_tree01_graph = graphviz.Source(dot_graph)
drugs_tree01_graph.render(filename="NBA_Player_tree01", directory='./images', format="png")


# ##### NBA_Player_tree01.png
# <img src='./images/NBA_Player_tree01.png'/>

# ### target에 대한 feature별 중요도 확인

# In[9]:


for name, value in zip(feature_names, decision_tree_classifier.feature_importances_):
    print(f'{name}, {round(value, 4)}')


# In[10]:


import seaborn as sns

sns.barplot(x=decision_tree_classifier.feature_importances_, y=feature_names)


# In[11]:


from sklearn.preprocessing import LabelEncoder
target_encoder = LabelEncoder()

encoded_targets = target_encoder.fit_transform(targets)


# In[12]:


import matplotlib.pyplot as plt

plt.title("2 Targets with 2 Features")
plt.scatter(features.iloc[:, 0], features.iloc[:, 5], marker='o', c=encoded_targets, s=25, cmap="rainbow", edgecolors='k')


# In[13]:


import numpy as np

# Classifier의 Decision Boundary를 시각화 하는 함수
def visualize_boundary(model, X, y):
    fig,ax = plt.subplots()
    
    # 학습 데이타 scatter plot으로 나타내기
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim_start , xlim_end = ax.get_xlim()
    ylim_start , ylim_end = ax.get_ylim()
    
    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행. 
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # contourf() 를 이용하여 class boundary 를 visualization 수행. 
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow',
                           zorder=1)


# In[14]:


visualize_boundary(decision_tree_classifier, features.iloc[:,[0,5]] , encoded_targets)


# In[15]:


modified_decision_tree_classifier = DecisionTreeClassifier(min_samples_leaf=8).fit(X_train, y_train)
visualize_boundary(modified_decision_tree_classifier, features.iloc[:,[0,5]], encoded_targets)


# ### 평가

# In[16]:


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


# In[17]:


df.hist(bins=50)


# In[18]:


scaler = StandardScaler()

features_scaled = scaler.fit_transform(features)

# stratify: 데이터를 나눌 때 타겟 데이터 분호 비율에 맞춰서 나눠준다.
X_train, X_test, y_train, y_test = train_test_split(features_scaled, targets, test_size=0.3, stratify=targets)

# 결정 트리 분류
decision_tree_classifier = DecisionTreeClassifier()
parameters = {'max_depth': [i for i in range(11)], 'min_samples_split': [i for i in range(16)]}

# GridSearchCV 생성자는 훈련이 아니라 하이퍼 파라미터 튜닝이다.
grid_decision_tree = GridSearchCV(decision_tree_classifier, param_grid=parameters, cv=10, refit=True, return_train_score=True)

# 훈련 시 교차 검증으로 진행한다.
grid_decision_tree.fit(X_train, y_train)

print(f'GridSearchCV 최적 파라미터: {grid_decision_tree.best_params_}')
print(f'GridSearchCV 최고 정확도: {grid_decision_tree.best_score_}')

prediction = grid_decision_tree.predict(X_test)
print(f'테스트 데이터 세트 정확도: {accuracy_score(y_test, prediction)}')

scores_df = pd.DataFrame(grid_decision_tree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', 
           'split0_test_score', 'split1_test_score', 'split2_test_score']]

display(scores_df)


# In[19]:


get_evaluation(y_test, prediction, classifier=grid_decision_tree, X_test=X_test)


# In[20]:


prediction_proba = grid_decision_tree.predict_proba(X_test)
encoded_y_test = target_encoder.fit_transform(y_test)
encoded_y_test


# In[21]:


thresholds = [i * 0.01 for i in range(101)]

# 타겟 데이터와 예측 객체를 전달받는다.
def get_evaluation_by_thresholds(y_test, prediction_proba_class1, thresholds):
    
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(prediction_proba_class1) 
        custom_prediction = binarizer.transform(prediction_proba_class1)
        print('임곗값:', threshold)
        get_evaluation(y_test, custom_prediction)
    
get_evaluation_by_thresholds(encoded_y_test, prediction_proba[:,1].reshape(-1,1), thresholds)


# In[22]:


def precision_recall_curve_plot(y_test , prediction_proba_class1):

    precisions, recalls, thresholds = precision_recall_curve(y_test, prediction_proba_class1)
    
    # X축: threshold
    # Y축: 정밀도, 재현율 
    # 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')
    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')
    
    # X축(threshold)의 Scale을 0 ~ 1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    
    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')
    plt.legend()
    plt.grid()
    plt.show()
    
precision_recall_curve_plot(encoded_y_test, grid_decision_tree.predict_proba(X_test)[:, 1] )


# In[23]:


def roc_curve_plot(y_test , prediction_proba_class1):
#     임계값에 따른 FPR, TPR 값
    fprs, tprs, thresholds = roc_curve(y_test ,prediction_proba_class1)

#     ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
#     가운데 대각선 직선을 그림. 
#     TPR과 FPR이 동일한 비율로 떨어진다는 것은 모델이 양성과 음성을 구별하지 못한다는 것을 의미한다.
#     다른 분류기를 판단하기 위한 기준선으로 사용되며, 
#     대각선에 가까울 수록 예측에 사용하기 힘든 모델이라는 뜻이다.
    plt.plot([0, 1], [0, 1], 'k--', label='Standard')
    
    # X축(FPR)의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()
    
roc_curve_plot(encoded_y_test, grid_decision_tree.predict_proba(X_test)[:, 1] )


# In[24]:


prediction_proba_class1 = grid_decision_tree.predict_proba(X_test)[:, 1]
binarizer = Binarizer(threshold=0.5)
prediction = binarizer.fit_transform(prediction_proba_class1.reshape(-1, 1))
roc_score = roc_auc_score(encoded_y_test, prediction)
print(f'ROC AUC 값: {np.round(roc_score, 4)}')


# ### 서포트 벡터 머신

# In[25]:


def svc_param_selection(X, y, nfolds):
    svm_parameters = [
                      {'kernel': ['rbf'],
                       'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
                       'C': [0.01, 0.1, 1, 10, 100, 1000]
                       }]
    clf = GridSearchCV(SVC(probability=True), svm_parameters, cv=10)
    clf.fit(X, y)
    print(clf.best_params_) # 최적의 parameter
    print(clf.best_score_) # 최적의 parameter로 교차 검증된 점수

    return clf


# In[26]:


modified_df = df
modified_df['Pos'] = encoded_targets
modified_df


# In[27]:


corr = modified_df.iloc[:,1:].corr()
fig = plt.figure(figsize=(7, 5))
heatmap = sns.heatmap(corr, cmap="Purples")
heatmap.set_title("Correlation")


# In[28]:


corr.sort_values(by=['Pos'],ascending=False)['Pos']


# In[29]:


clf = svc_param_selection(X_train, y_train, nfolds=10)


# In[30]:


visualize_boundary(clf, features.iloc[:,[0,5]], encoded_targets)


# In[31]:


roc_curve_plot(encoded_y_test, clf.predict_proba(X_test[:,[0,5]])[:, 1] )


# In[32]:


get_evaluation_by_thresholds(encoded_y_test, clf.predict_proba(X_test[:,[0,5]])[:, 1].reshape(-1,1), thresholds)


# In[33]:


prediction_proba_class1 = clf.predict_proba(X_test[:,[0,5]])[:, 1]
binarizer = Binarizer(threshold=0.7)
prediction = binarizer.fit_transform(prediction_proba_class1.reshape(-1, 1))
roc_score = roc_auc_score(encoded_y_test, prediction)
print(f'ROC AUC 값: {np.round(roc_score, 4)}')

