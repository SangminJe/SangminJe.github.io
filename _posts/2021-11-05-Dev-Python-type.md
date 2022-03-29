---  
layout: post  
title: "Python Type"
subtitle: "민감하지 않은(?) 파이썬의 데이터 타입"  
categories: DEV
tags: DEV Python python datatype 
comments: true  
---  


## Python's Dynamic Typing

---

보통 개발에서는 **strongly typed**라는 개념이 언급되는 듯하다. 이 뜻은 데이터가 한 번 data type이 지정되면 Runtime 중에는 바꿀 수 없는 속성을 가리킨다.  
반 면 파이썬에서는 이게 가능하다. 말하자면, 아래와 같은 예시가 가능하다.
```
bob = 1
bob = "bob"
```
Stack Overflow의 설명에난 아래와 같이 설명되어 있다.
- **Strong** typing means that the type of a value doesn't change in unexpected ways. A string containing only digits doesn't magically become a number, as may happen in Perl. Every change of type requires an explicit conversion.
- **Dynamic** typing means that runtime objects (values) have a type, as opposed to static typing where variables have a type.

예전에 **C#** 개발을 억지로 해야 했던 적이 있는데, 그 때도 static이라던지 하는 data type을 지정해야 하는 것을 몰라서 개발에 애를 먹었다. 그만큼 파이썬이 얼마나 쉬운 언어인가를 반대로 보여주는 사례이기도 하다.

## 참고
---
- [Is Python strongly typed?](https://stackoverflow.com/questions/11328920/is-python-strongly-typed)