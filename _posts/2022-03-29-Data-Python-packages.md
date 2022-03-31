---  
layout: post  
title: "Python 머신러닝 패키지 및 함수 정리"
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

<br>

# 패키지 임포트
---

## 필수 패키지
```py
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
```

### 전처리 패키지
```py
from sklearn.model_selection import train_test_split # train test split
from sklearn.model_selection import KFold # KFold
from sklearn.model_selection import cross_val_score # 교차검증
from sklearn.model_selection import GridSearchCV # 그리드서치
from imblearn.oversampling import SMOTE # 오버샘플링
```


### 분류 패키지
```py
# sklearn
from sklearn.tree import DecisionTreeClassifier # 결정트리
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score # 분류 평가패키지
```

### 회귀 패키지
```py
from sklearn.linear_model import LinearRegression # 선형회귀
from sklearn.preprocessing import PolynomialFeatures # 다항회귀
from sklearn.metrics import mean_squared_error, r2_score # MSE
from sklearn.linear_model import Ridge # 릿지
from sklearn.linear_model import Lasso, ElasticNet # 라소, ElsasticNet
```

## 기타 패키지
```py

```



<br>

# 유용한 함수
---
## f1, 재현율, 정밀도, 오차행렬, roc_auc 반환 gkatn

```python
# f1, 재현율, 정밀도, 오차행렬, roc_auc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score 

# 평가용 함수
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

## 모델 학습/예측/평가 수행
```python
# sklearn의 Estimator 객체와 학습/테스트 데이터 세트를 받아 학습/예측/평가 수행
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]
    get_clf_eval(tgt_test, pred, pred_proba) # 위에서 작성한 평가 함수
```

