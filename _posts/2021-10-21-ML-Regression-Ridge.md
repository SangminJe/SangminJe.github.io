---  
layout: post  
title: "Rigdge와 Lasso"
subtitle: "Ridge와 Lasso 설명"  
categories: DATA
tags: Ridge Lasso regression linear-regression
comments: true  
---  


## Ridge 회귀 (L2 규제)
---

![img](https://sangminje.github.io/assets/img/ridge_lasso/img1.PNG)


 - 선형 모델이며 최소적합법을 사용
 - 일반 회귀와 다른 점은 **가중치(w 또는 β)** 의 절대값을 최대한 작게 함
 - 즉 기울기를 0으로 만듬
 - 이렇게 하는 이유는 **과대적합**이 생기지 않도록 모델을 강제로 제한함을 의미



## Lasso 회귀 (L1 규제)
---

![img](https://sangminje.github.io/assets/img/ridge_lasso/img2.PNG)

- Ridge에서와 마찬가지로 계수를 0에 가깝게 제한하는 것은 동일함
- 하지만 정말로 0인 계수가 생기기도 함. 즉 제외되는 계수도 발생
- 이는 자동으로 특성이 선택된다고 볼 수 있으며 가장 중요한 특성이 무엇인지 드러내 준다고 생각하면 됨
- 간혹 Lasso를 이용하면 과속적합이 발생하기도 하는데, 이는 너무 많은 Feature를 Drop하여 모델을 못 쓰게 만드는 경우이다.


## 무엇을 사용할 것인가?
---

- 일반적으로 Ridge 회귀를 분석에서 선호하는 편
- 다만 분석하기 쉬운 모델을 고를 경우 Lasso가 더 좋은 선택이 될 수 있음
- Lasso와 Ridge의 패널티를 결합한 **ElasticNet**이라는 것도 있음. 최상의 결과를 내기도 하지만 두 개의 매개변수를 조정해야 함.

## 참고 사이트
- [Ridge와 Lasso 쉽게 이해하기](https://rk1993.tistory.com/entry/Ridge-regression%EC%99%80-Lasso-regression-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)
- [Bias vs Variance](https://modulabs-biomedical.github.io/Bias_vs_Variance)