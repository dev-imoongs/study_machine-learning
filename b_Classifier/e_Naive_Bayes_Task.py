#!/usr/bin/env python
# coding: utf-8

# ### Naive Bayes Classifier Task
# ### 문장에서 느껴지는 감정 예측
# ##### 다중 분류(Multiclass Classification)
# - 비대면 심리 상담사로서 메세지를 전달한 환자에 대한 감정 데이터를 수집했다.
# - 각 메세지 별로 감정이 표시되어 있다.
# - 미래에 동일한 메세지를 보내는 환자에게 어떤 심리 치료가 적합할 수 있는지 알아보기 위한 모델을 구축한다.
# 
# ##### 🚩제시된 feature의 target을 예측해보자!
# - 'Sweat deer'  
# - 'The moment I saw her, I realized something was wrong.'

# In[1]:


import pandas as pd

feeling_df = pd.read_csv('./datasets/feeling.csv', sep=";")
feeling_df


# ##### 레이블 인코딩

# In[2]:


from sklearn.preprocessing import LabelEncoder

feeling_encoder = LabelEncoder()
feeling_df['target'] = feeling_encoder.fit_transform(feeling_df.feeling)


# In[3]:


print(feeling_encoder.classes_)
feeling_df


# ##### 타겟 데이터 불균형 해소

# In[4]:


feeling_df.target.value_counts()

# ['anger' 'fear' 'joy' 'love' 'sadness' 'surprise']
anger = feeling_df[feeling_df.target == 0].sample(653)
fear = feeling_df[feeling_df.target == 1].sample(653)
joy = feeling_df[feeling_df.target == 2].sample(653)
love = feeling_df[feeling_df.target == 3].sample(653)
sadness = feeling_df[feeling_df.target == 4].sample(653)
surprise = feeling_df[feeling_df.target == 5]

feeling_df = pd.concat([anger, fear, joy, love, sadness, surprise])


# In[5]:


feeling_df.target.value_counts()


# In[6]:


from sklearn.model_selection import train_test_split

features, targets = feeling_df.message, feeling_df.target

X_train, X_test, y_train, y_test = train_test_split(features, targets, stratify=targets, test_size=0.2)


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

naive_bayes_pipeline = Pipeline([('count_vectorizer', CountVectorizer()), ('multinomialNB', MultinomialNB())])
naive_bayes_pipeline.fit(X_train, y_train)


# In[8]:


prediction = naive_bayes_pipeline.predict(X_test)
prediction


# In[9]:


naive_bayes_pipeline.score(X_test, y_test)


# In[10]:


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


# In[11]:


get_evaluation(y_test, prediction, naive_bayes_pipeline, X_test)

