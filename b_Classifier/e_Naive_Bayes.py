#!/usr/bin/env python
# coding: utf-8

# ### 베이즈 추론, 베이즈 정리, 베이즈 추정(Bayesian Inference)
# - 역확률(inverse probability) 문제를 해결하기 위한 방법으로서, 조건부 확률(P(B|A))을 알고 있을 때 정반대인 조건부 확률(P(A|B))을 구하는 방법이다.
# > 📌조건부 확률(Conditional probability): 두 사건 A, B가 있을 때, 사건 A가 일어났을 때 B가 일어날 확률이다.
# <img src="./images/conditional_probability.png" width="200" style="margin-top:20px; margin-left:0">
# - 추론 대상의 사전 확률과 추가적인 정보를 기반으로 해당 대상의 "사후 확률"을 추론하는 통계적 방법이다.  
# 📌사후 확률이란, 어떤 사건이 발생한 후 앞으로 일어나게 될 다른 사건의 가능성을 구하는 것이다.
# - 어떤 사건이 서로 "배반"하는(독립하는) 원인 둘에 의해 일어난다고 하면, 실제 사건이 일어났을 때 이 사건이 두 원인 중 하나일 확률을 구하는 방식이다.  
# 📌배반하는 원인이란, 하나의 사건이 일어난 원인의 확률이 다른 원인의 확률에 영향을 미치지 않고 각각 독립적이라는 뜻이다.
# - 어떤 상황에서 N개의 원인이 있을 때, 실제 사건이 발생하면 N개 중 한 가지 원인일 확률을 구하는 방법이다.
# - 기존 사건들의 확률을 알 수 없을 때 전혀 사용할 수 없는 방식이다.  
# 하지만 그 간 데이터가 쌓이면서, 기존 사건들의 확률을 대략적으로 뽑아낼 수 있게 되었다.  
# 이로 인해, 사회적 통계나 주식에서 베이즈 정리 활용이 필수로 꼽히고 있다.  
# 
# > ##### 예시
# 질병 A의 양성판정 정확도가 80%인 검사기가 있다. 검사를 시행해서 양성이 나왔다면, 이 사람이 80%의 확률로 병에 걸렸다고 이야기할 수 없다. 왜냐하면 검사기가 알려주는 확률과 양성일 경우 질병을 앓고 있을 확률은 조건부 확률의 의미에서 정반대이기 때문이다.  
# <table style="width:50%; margin-left: 50px">
#     <tr>
#         <th>전제</th>
#         <th>관심 사건</th>
#         <th>확률</th>
#     </tr>
#     <tr>
#         <th>병을 앓고 있다</th>
#         <th>양성이다</th>
#         <th>80%</th>
#     </tr>
#     <tr>
#         <th>양성이다</th>
#         <th>병을 앓고 있다</th>
#         <th>알수 없음</th>
#     </tr>
# </table>  
# 
# > 이런 식의 확률을 구해야 하는 문제를 역확률 문제라고 하고 이를 베이즈 추론을 활용하여 구할 수 있다.  
# 단, 검사 대상인 질병의 유병률(사전 확률, 기존 사건들의 확률)을 알고 있어야 한다.  
# 전세계 인구 중 10%의 사람들이 질병 A를 앓는다고 가정한다.
# <div style="width: 60%; display:flex; margin-top: -20px; margin-left:30px">
#     <div>
#         <img src="./images/bayesian_inference01.png" width="300" style="margin-top:20px; margin-left:0">
#     </div>
#     <div style="margin-top: 28px; margin-left: 20px">
#         <img src="./images/bayesian_inference02.png" width="310" style="margin-top:20px; margin-left:0">
#     </div>
# </div>  
# 
# <div style="width: 60%; display:flex; margin-left:30px">
#     <div>
#         <img src="./images/bayesian_inference03.png" width="800" style="margin-top:20px; margin-left:0">
#     </div>
#     <div style="margin-top: 28px; margin-left: 20px">
#         <img src="./images/bayesian_inference04.png" width="550" style="margin-top:-8px; margin-left:0">
#     </div>
# </div>  
# 
# > 🚩결과: 약 30.8%
# <img src="./images/bayesian_inference05.png" width="200" style="margin-top:20px; margin-left:0">
# 

# ### 나이브 베이즈 분류(Naive Bayes Classifier)
# - 텍스트 분류를 위해 전통적으로 사용되는 분류기로서, 분류에 있어서 준수한 성능을 보인다.
# - 베이즈 정리에 기반한 통계적 분류 기법으로서, 정확성도 높고 대용량 데이터에 대한 속도도 빠르다.
# - 반드시 모든 feature가 서로 독립(independent)적이어야 한다. 즉, 서로 영향을 미치지 않는 feature들로 구성되어야 한다.
# - 감정 분석, 스팸 메일 필터링, 텍스트 분류, 추천 시스템 등 여러 서비스에서 활용되는 분류 기법이다.
# - 빠르고 정확하고 간단한 분류 방법이지만, 실제 데이터에서 모든 feature가 독립적인 경우는 드물기 때문에 실생활에 적용하기 힘들다.
# <img src="./images/naive_bayes_classifier.png" width="400" style="margin-left: 0">

# ### 나이브 베이즈 종류
# ##### BernoulliNB(베르누이 나이브 베이즈)  
# - 가장 기본적인 NB 함수로 이진 분류에 사용한다.
# ##### CategoricalNB  
# - 분류할 카테고리의 종류가 3가지 이상일 때 사용한다.
# ##### MultinomialNB (멀타이노우미얼(다항) 나이브 베이즈) 
# - 텍스트의 등장 횟수처럼 이산적인 값의 수를 예측할 때 사용한다.
# ##### GaussianNB (가우시안 나이브 베이즈) 
# - 예측할 값이 연속적인 값인 경우에 사용한다.
# ##### ComplementNB  
# - target label의 balance가 맞지 않는 불균형한 상황에 사용한다.

# ### 스팸 메일 분류

# In[1]:


import pandas as pd

mail_df = pd.read_csv('./datasets/spam.csv')
mail_df


# In[2]:


mail_df.info()


# ### 레이블 인코딩

# In[3]:


from sklearn.preprocessing import LabelEncoder

mail_encoder = LabelEncoder()
targets = mail_encoder.fit_transform(mail_df.Category)
mail_df['Category'] = targets


# In[4]:


mail_encoder.classes_


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

text = ["햄버거는 맛있어! 정말 맛있어", 
        "아니야 피자가 더 맛있어 햄버거보다 더 맛있어!",
        "아니야 둘 다 먹자!"]
        
count_vec = CountVectorizer()
m = count_vec.fit_transform(text)
print(m.toarray())

# 각 열번호가 뜻하는 단어
print(count_vec.vocabulary_)


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mail_df.Message, mail_df.Category, test_size=0.3, stratify=mail_df.Category)


# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

navie_bayes_pipeline = Pipeline([('count_vectorizer', CountVectorizer()), ('naive_bayes', MultinomialNB())])
navie_bayes_pipeline.fit(X_train, y_train)


# In[8]:


prediction = navie_bayes_pipeline.predict(X_test)

# 스팸 메일로 판단한 이메일을 정렬을 통해 인덱스 가져오기
prediction[prediction == 1].argsort()
# 해당 인덱스의 예측 결과가 1(스팸)인지 검사
print(f'예측 결과: {prediction[216]}')

# feature에서 동일한 인덱스의 메세지 내용을 가져오기
# 실제 정답이 1(스팸)인지 검사
print('실제 정답: {}'.format(mail_df[mail_df['Message'] == X_test.to_list()[216]].Category.to_list()[0]))


# In[9]:


navie_bayes_pipeline.score(X_test, y_test)


# In[10]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, prediction)


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


# In[14]:


get_evaluation(y_test, prediction, navie_bayes_pipeline, X_test)


# In[ ]:




