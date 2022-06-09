---  
layout: post  
title: "[Python] 머신러닝 패키지 및 함수 정리"
subtitle: "Python 머신러닝을 위해 설치가 필요한 패키지 및 함수 정리"  
categories: DATA
tags: DATA python Python_Data packages_for_machine_learning 머신러닝 패키지 및 함수
comments: true  
---  

# 패키지 설치
---
**conda prompt**에서 입력

## scickit-learn 설치
```console
conda install scikit-learn
```

## XGBoost, LightGBM 설치

```console
conda install -c anaconda py-xgboost # XGBoost
conda install -c conda-forge lightgbm # LightGBM
```

## Imbalanced Lear(SMOTE)
```console
conda install -c conda-forge imbalanced-learn
```

<br><br>

# 패키지 임포트
---

## 필수 패키지
```py
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # seaborn
import warnings
warnings.filterwarnings("ignore") # 경고메세지 무시
```


## 사이킷런 패키지

### 기본 데이터 패키지
```py
from sklearn.datasets import load_iris # iris
from sklearn.datasets import load_boston # boston
from sklearn.datasets import make_blobs # 군집화용 생성
X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0) # cluster_std로 데이터의 분포도를 조절할 수 있음
from sklearn.datasets import make_classification # 분류용 데이터 생성
```

### 전처리 패키지
```py
from sklearn.model_selection import train_test_split # train test split
from sklearn.model_selection import KFold # KFold
from sklearn.model_selection import cross_val_score # 교차검증
from sklearn.model_selection import GridSearchCV # 그리드서치
from imblearn.oversampling import SMOTE # 오버샘플링
from sklearn.preprocessing import scale
```


### 분류 패키지
```py
# sklearn
from sklearn.tree import DecisionTreeClassifier # 결정트리
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀
from sklearn.ensemble import RandomForestClassifier # RF
from lightgbm import LGBMClassifier #LGBM
from xgboost import XGBClassifier #XGBoost
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score # 분류 평가패키지
```

### 회귀 패키지
```py
from sklearn.linear_model import LinearRegression # 선형회귀
from sklearn.preprocessing import PolynomialFeatures # 다항회귀
from sklearn.linear_model import Ridge, Lasso, ElasticNet # Ridge, Lasso, ElsasticNet
from sklearn.ensemble import RandomForestRegressor # 랜덤포레스트
from sklearn.tree import DecisionTreeRegressor # 결정트리
from sklearn.ensemble import GradientBoostingRegressor # 그래디언트부스트
from xgboost import XGBRegressor # XGBoosdt
from lightgbm import LGBMRegressor # LGBM
from sklearn.metrics import mean_squared_error, r2_score # MSE, R_square
```
### 군집화 패키지
```py
from sklearn.cluster import KMeans # K-means
from sklearn.mixture import GaussianMixture # GMM
from sklearn.metrics import silhouette_samples, silhouette_score # 실루엣 점수
from sklearn.cluster import DBSCAN # DBSCAN
```


### 차원축소 패키지
```py
from sklearn.decomposition import PCA
```

### 텍스트 패키지
```py
from sklearn.feature_extraction.text import CountVectorizer # CountVec
from sklearn.mterics.pairwise import cosine_similarity # 코사인 유사도
from ast import literal_eval # 리스트 내의 딕셔너리 형태의 스트링 값을 객체로 바꿔주기 위해 사용

```

## 기타 패키지
```py

```
<br><br>

# 기타
---
## 마커 목록
markers = ['o','s','^','P','D','H','x'] # 마커목록

## Pandas 데이터 핸들링
```py
df.groupby(["InvoiceNo", "StockCode"])["InvoiceNo"].count().mean() # 두유일한 식별자인지 확인하기
```

<br>

# 유용한 함수
---
## 기타 함수

### 문자열 -> 객체 변환

```py
from ast import literal_eval

# literal_eval을 통해 문자열을 객체로 변환
movies_df["keywords"] = movies_df["keywords"].apply(literal_eval)
```


## 머신러닝

### f1, 재현율, 정밀도, 오차행렬, roc_auc 반환 함수

```python
# f1, 재현율, 정밀도, 오차행렬, roc_auc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score 

# 이진분류 평가용 함수
def get_clf_eval(y_test, pred, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    
    print("오차행렬")
    print(confusion)
    print(f"정확도 : {accuracy:.4f}, 정밀도 : {precision:.4f}, 재현율 : {recall:.4f}, F1: {f1:.4f}, AUC : {roc_auc:.4f}")
```

### 모델 학습/예측/평가 수행
```python
# sklearn의 Estimator 객체와 학습/테스트 데이터 세트를 받아 학습/예측/평가 수행
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]
    get_clf_eval(tgt_test, pred, pred_proba) # 위에서 작성한 평가 함수
```

### 군집 시각화

```py
### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성  
def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")


### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 클러스터링 결과를 시각화 
def visualize_kmeans_plot_multi(cluster_lists, X_features):
    
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 만큼의 sub figures를 가지는 axs 생성 
    n_cols = len(cluster_lists)
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 입력 데이터의 FEATURE가 여러개일 경우 2차원 데이터 시각화가 어려우므로 PCA 변환하여 2차원 시각화
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(X_features)
    dataframe = pd.DataFrame(pca_transformed, columns=['PCA1','PCA2'])
    
     # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 KMeans 클러스터링 수행하고 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링으로 클러스터링 결과를 dataframe에 저장. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(pca_transformed)
        dataframe['cluster']=cluster_labels
        
        unique_labels = np.unique(clusterer.labels_)
        markers=['o', 's', '^', 'x', '*']
       
        # 클러스터링 결과값 별로 scatter plot 으로 시각화
        for label in unique_labels:
            label_df = dataframe[dataframe['cluster']==label]
            if label == -1:
                cluster_legend = 'Noise'
            else :
                cluster_legend = 'Cluster '+str(label)           
            axs[ind].scatter(x=label_df['PCA1'], y=label_df['PCA2'], s=70,\
                        edgecolor='k', marker=markers[label], label=cluster_legend)

        axs[ind].set_title('Number of Cluster : '+ str(n_cluster))    
        axs[ind].legend(loc='upper right')
    
    plt.show()
```