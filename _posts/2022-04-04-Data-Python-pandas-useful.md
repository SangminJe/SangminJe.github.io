---  
layout: post  
title: "[Python] 유용한 Pandas 문법 정리 (1) - 결손데이터"
subtitle: ""  
categories: DATA
tags: DATA python Python_Data pandas filter null
comments: true  
---  

# 패키지 설치
---

## DataFame에서 Null이 있는 Column만 확인
```py
df.columns[df.isna().any()].tolist()
```

## Null이 있는 컬럼과 그 Null의 개수 확인
```py
df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)
```