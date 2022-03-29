---  
layout: post  
title: "RNN(Recurrent Neural Network) 기초"
subtitle: ""  
categories: DATA
tags: RNN DL Deep-Learning 
comments: true  
---  

## 사용하는 상황
---

**Sequential 데이터**일 때 주로 사용한다. 예를 들어 우리가 말을 할 때 하나의 단어만 가지고 문장의 의미를 이해하는 게 아니고 단어의 연결을 통해 완성된 문장으로 소통하게 된다. `나는 지금 배가고프다`라는 문장이 있을 때 `나`라는 단어와 `지금`이라는 단어, 그리고 `배고프다`라는 단어가 서로 영향을 받아 의미있는 하나의 단위를 형성하게 된다. 이처럼 하나의 데이터가 다른 데이터에 영향을 미치는 구조의 데이터를 다룰 때 RNN을 사용한다. 다른 사용 예로는 아래와 같다.

- 주어진 이미지에서 Cation을 생성할 때
- 회사가 파산할지 안할지를 예측할 때(시간에 따른 회사의 재무상태가 연속적으로 주어진다고 가정)
- 똑같은 의미의 문장을 다른 문장으로 전환할 때

<br>


## 사용하는 상황
---

## 참고자료
- [[딥러닝] RNN 기초 (순환신경망 - Vanilla RNN)](https://www.youtube.com/watch?v=PahF2hZM6cs)
- [lec12: NN의 꽃 RNN 이야기](https://www.youtube.com/watch?v=-SHPG_KMUkQ)
- https://www.youtube.com/watch?v=bPRfnlG6dtU