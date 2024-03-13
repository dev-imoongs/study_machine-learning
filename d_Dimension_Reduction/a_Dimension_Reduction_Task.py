#!/usr/bin/env python
# coding: utf-8

# ### PCA Task
# 
# ##### 뇌암 유전자 발현 예측
# 
# https://drive.google.com/file/d/19ZtDpQoR2Qq0_66xUVOU_NOAJnq251Vg/view?usp=sharing

# In[1]:


import pandas as pd

brain_df = pd.read_csv('./datasets/brain_cancer.csv')
brain_df


# In[2]:


print('='*40)
print(brain_df.info())
print('='*40)
print(brain_df.isna().sum())
print('='*40)
print(brain_df.duplicated().sum())
print('='*40)


# In[3]:


brain_df['type'].value_counts()


# In[4]:


new_df = brain_df


# In[5]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=124)
featrues, targets = brain_df.iloc[:, 2:], brain_df['type']
over_features, over_targets = smote.fit_resample(featrues, targets)

print('SMOTE 적용 전:\n',pd.Series(targets).value_counts() )
print('SMOTE 적용 후:\n',pd.Series(over_targets).value_counts() )


# In[6]:


over_df = pd.concat([over_features, over_targets], axis=1)
over_df


# In[7]:


from sklearn.preprocessing import StandardScaler

over_scaled_data = StandardScaler().fit_transform(over_df.iloc[:,:-1])
over_scaled_df = pd.DataFrame(data=over_scaled_data, columns=over_df.columns[:-1])
over_scaled_df['target'] = over_df['type']
over_scaled_df


# In[10]:


over_df


# In[11]:


from sklearn.preprocessing import LabelEncoder

# 범주형 데이터를 담은 리스트

# LabelEncoder 객체 생성
label_encoder = LabelEncoder()

# 범주형 데이터를 숫자로 변환
encoded_categories = label_encoder.fit_transform(over_df['type'])

over_df['type'] = encoded_categories

# 변환된 데이터 출력
print("변환된 데이터:", encoded_categories)

# 각 숫자에 대응하는 원래 범주 확인
decoded_categories = label_encoder.inverse_transform(encoded_categories)
print("복원된 데이터:", decoded_categories)


# In[16]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

over_pca = pca.fit_transform(over_df.iloc[:, :-1])
print(over_pca.shape)


# In[20]:


# PCA 환된 데이터의 컬럼명을 각각 pca1, pca2, ..., pcan으로 명명
pca_columns=[f'pca{i+1}' for i in range(2)]
over_pca_df = pd.DataFrame(over_pca, columns=pca_columns)
over_pca_df['target']=over_df.type
over_pca_df.head(10)


# In[21]:


print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())


# In[22]:


import seaborn as sns

sns.scatterplot(x="pca1", y="pca2", hue='target', data=over_pca_df, alpha=0.5)


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rfc = RandomForestClassifier(random_state=156, max_depth=3, min_samples_split=30)
scores = cross_val_score(rfc, over_pca_df.iloc[:, 1:], over_pca_df.target, scoring='accuracy', cv=5)
print('원본 데이터 교차 검증 개별 정확도:',scores)
print('원본 데이터 평균 정확도:', np.mean(scores))


# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rfc = RandomForestClassifier(random_state=156, max_depth=3, min_samples_split=30)
scores = cross_val_score(rfc, over_pca_df.iloc[:, :-1], over_pca_df.target, scoring='accuracy', cv=5)
print('PCA 데이터 교차 검증 개별 정확도:',scores)
print('PCA 데이터 평균 정확도:', np.mean(scores))


# In[25]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 타겟 클래스(카테고리) 개수 -n을 전달한다.
lda = LinearDiscriminantAnalysis(n_components=1)

lda.fit(over_df.iloc[:, :-1], over_df.type)
over_lda = lda.transform(over_df.iloc[:, :-1])

print(over_lda.shape)


# In[30]:


over_lda_df = pd.DataFrame()
over_lda_df['lda']=over_lda.flatten()
over_lda_df['target']=over_df.type
over_lda_df.head(10)


# In[31]:


print(lda.explained_variance_ratio_)
print(lda.explained_variance_ratio_.sum())


# In[32]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rfc = RandomForestClassifier(random_state=156, max_depth=3, min_samples_split=30)
scores = cross_val_score(rfc, over_df.iloc[:, 1:], over_df.type, scoring='accuracy', cv=5)
print('원본 데이터 교차 검증 개별 정확도:',scores)
print('원본 데이터 평균 정확도:', np.mean(scores))


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rfc = RandomForestClassifier(random_state=156, max_depth=3, min_samples_split=30)
scores = cross_val_score(rfc, over_lda_df[['lda']], over_lda_df.target, scoring='accuracy', cv=5)
print('LDA 데이터 교차 검증 개별 정확도:',scores)
print('LDA 데이터 평균 정확도:', np.mean(scores))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




