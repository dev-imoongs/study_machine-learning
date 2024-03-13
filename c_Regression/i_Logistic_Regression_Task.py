#!/usr/bin/env python
# coding: utf-8

# ### Logistic Regression Task
# 
# ##### 사람의 체온과 습도를 통한 스트레스 예측
# 
# - Humidity: 스트레스를 느낄 때, 여러분의 체온이 상승하여 땀샘이 활성화됩니다. 이 땀은 습도 수준으로 여겨집니다.
# - Temperature: 스트레스를 받는 동안 사람의 체온입니다.
# - Stepcount: 스트레스를 받는 상황에서 당사자가 적용하는 스텝 수입니다.
# - Stress_Level: 위의 세 가지 요인에 기초하여 스트레스 수준을 높음, 중간 및 낮음으로 예측합니다.

# In[3]:


import pandas as pd

stress_df = pd.read_csv('./datasets/stress.csv')
stress_df


# In[4]:


print('='*40)
print(stress_df.info())
print('='*40)
print(stress_df.isna().sum())
print('='*40)
print(stress_df.duplicated().sum())


# In[15]:


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


# In[51]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

features, targets = stress_df.iloc[:,:-1], stress_df.Stress_Level

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, stratify=targets, random_state=124)

# smote = SMOTE(random_state=0)
# X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

lg = LogisticRegression(solver='liblinear', penalty='l2', random_state=124)
lg.fit(X_train, y_train)


# In[55]:


prediction = lg.predict(X_test)
get_evaluation(y_test, prediction, lg, X_test)


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




