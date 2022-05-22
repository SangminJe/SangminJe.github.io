---  
layout: post  
title: "Python 머신러닝 패키지 및 함수 정리"
subtitle: "Python 머신러닝을 위해 설치가 필요한 패키지 및 함수 정리"  
categories: DATA
tags: DATA python Python_Data packages_for_machine_learning 머신러닝 패키지 및 함수
comments: true  
---  

# 전처리

## 문자열 -> 객체 변환

```py
from ast import literal_eval

# literal_eval을 통해 문자열을 객체로 변환
movies_df["keywords"] = movies_df["keywords"].apply(literal_eval)
```

## 이상치 제거
```py
from scipy.stats import skew

feature_index = all_df.select_dtypes(exclude="object").columns
skew_features = all_df[feature_index].apply(lambda x : skew(x)) # Series로 값이 떨어짐
skew_featrues_top = skew_features[skew_features > 1]
skew_featrues_top.sort_values(ascending=False)

all_df[skew_featrues_top.index] = np.log1p(all_df[skew_featrues_top.index])
```

# 머신러닝
---

## 이진분류

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

## 회귀

### RMSE 산출 함수
```python
def get_rmse(model):
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(f"{model.__class__.__name__} MSE : {mse:.4f}, RMSE : {rmse:.4f}")
    return rmse
    

def get_rmses(models):
    rmses = []
    for model in models:
        rmse = get_rmse(model)
        rmses.append(rmse)
    return rmses


# CV 기반 RMSE 반환
from sklearn.model_selection import cross_val_score

model_list = [lin_reg, rid_reg, lasso_reg]
def get_avg_rmse_cv(models):
    
    for model in models:
        rmse_list = np.sqrt(-cross_val_score(model, X_df, y_df, scoring="neg_mean_squared_error", cv=5))
        rmse_avg = np.mean(rmse_list)
        print(f"{model.__class__.__name__}의 평균 RMSE : {rmse_avg:.4f}")

# GRID SEARCH 기반
from sklearn.model_selection import GridSearchCV

def print_best_params(model, params):
    grid_model = GridSearchCV(model, param_grid=params, scoring="neg_mean_squared_error", cv=5)
    grid_model.fit(X_df, y_df)
    rmse = np.sqrt(-1*grid_model.best_score_)
    print(f"{model.__class__.__name__}의 최적 alpha : {grid_model.best_params_}, 최적 RMSE : {rmse:.4f}")

ridge_params = {"alpha" : [0.05, 0.1,1,5,8,10,12,15,20]}
lasso_params = {"alpha" : [0.001, 0.01, 0.05, 0.1,1,5,8,10,12,15,20]}
print_best_params(rid_reg, ridge_params)
print_best_params(lasso_reg, lasso_params)



```

### 회귀계수 시각화
```py
ef get_top_bot_coef(model, n=10):
    coef = pd.Series(model.coef_, index=X_df.columns)
    
    coef_high = coef.sort_values(ascending=False).head(n)
    coef_low = coef.sort_values(ascending=False).tail(n)
    return coef_high, coef_low

def viz_coef(models):
    # 3개 회귀 모델 시각화
    fig, axs = plt.subplots(figsize=(24,10), nrows=1, ncols=3)
    fig.tight_layout()
    
    for i, model in enumerate(models):
        coef_high, coef_low = get_top_bot_coef(model)
        coef_concat = pd.concat([coef_high, coef_low])
        
        ax_row, ax_col = divmod(i,3) # ax의 위치 (몫과 나머지)
        axs[ax_col].set_title(model.__class__.__name__ + " Coefficient", size=25)
        axs[ax_col].tick_params(axis="y", direction="in", pad=-120)
        for label in (axs[ax_col].get_xticklabels() + axs[ax_col].get_yticklabels()):
            label.set_fontsize(22)
        sns.barplot(x=coef_concat.values, y=coef_concat.index, ax=axs[ax_col])

```



## 군집화
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

# Kaggle 제출
---
```py
submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission["SalePrice"] = pred_t
submission.to_csv("sub.csv",index=False)
```