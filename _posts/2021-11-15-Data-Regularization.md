---  
layout: post  
title: "종속변수를 정규화(Regularization)하는 이유"
subtitle: ""  
categories: DATA
tags: Analytics 정규화 Regularization
comments: true  
---  

## 정규화
---

머신러닝을 통해 분석 과제를 해결할 때 종속변수가 정규성 가정을 만족하지 못할 경우 로그를 씌워 정규화를 해주기도 한다. 그런데 몇 가지 의문이 아래와 같이 남았다.

- 로그를 취해서 정규화를 맞춰야 하는 이유가 무엇일까?
- 로그를 씌워 정규화를 해주는 경우, 원데이터의 성질을 제대로 반영하지 못하는 게 아닐까?

이 두가지 질문에 대하여 여러 자료를 찾고 정리했다.



### 정규화를 해주는 이유

**회귀 모델**에서 변수를 대수 변환하는 것은 독립 변수와 종속 변수 사이에 비선형 관계가 존재하는 상황을 처리하는 매우 일반적인 방법이다. 로그를 사용하지 않은 모델 대신 하나 이상의 변수의 로그를 사용하면 선형 모델을 유지하면서 관계는 비선형이 된다.

- 그러니까 따지고 보면 Histogram만 보고 종속변수의 분포만 보고서 이것을 Log를 취해 정규화를 시켜주어야한다는 생각은 굳이 하지 않아도 좋은 것이다.

그 중 로그변환은 심하게 치우친 변수를 변환하는 편리한 방법으로 사용된다.

### 로그를 씌워 정규화를 하면 문제는 없는가?


### 결론
굳이 까놓고 말하자면 로그변환을 해야 하는 상황이 정해져 있는 것은 아니다. 필요에 따라서 로그 변환이 유리하다고 판단되면 그렇게 적용하는 것이고 아니면 말면 그만이다.

<br><br><br>

## 참고자료
---
- <https://evening-ds.tistory.com/31>
- <https://kenbenoit.net/assets/courses/ME104/logmodels2.pdf>
- <https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqhow-do-i-interpret-a-regression-model-when-some-variables-are-log-transformed>
- <https://danbi-ncsoft.github.io/study/2018/08/07/logwithlevel.html>
- <https://stats.stackexchange.com/questions/298/in-linear-regression-when-is-it-appropriate-to-use-the-log-of-an-independent-va>