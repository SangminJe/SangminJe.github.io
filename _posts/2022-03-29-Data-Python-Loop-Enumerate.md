---  
layout: post  
title: "[Python] For Loop와 Enumerate 정리"
subtitle: ""  
categories: DATA
tags: DATA python Python_Data For Loop Enumerate
comments: true  
---  

For Loop와 Enumerate의 기본적인 사용을 정리해놓았다.

# For Loop Practice


```python
var_list = [1,3,5,7]
var_dict = {"a":1 , "b":2}
var_set = {1,3}
var_str = "abc"
var_bytes = b'abcdef'
var_tuple = (1,3,5,7)
var_range = range(0,5)
```


```python
var_bytes
```




    b'abcdef'




```python
for i in var_list:
    print(i)
print("----"*30)
for i in var_dict:
    print(i)
print("----"*30)
for i in var_set:
    print(i)
print("----"*30)   
for i in var_str:
    print(i)
print("----"*30)    
for i in var_bytes:
    print(i)
print("----"*30)    
for i in var_tuple:
    print(i)
print("----"*30)   
for i in var_range:
    print(i)
```

    1
    3
    5
    7
    ------------------------------------------------------------------------------------------------------------------------
    a
    b
    ------------------------------------------------------------------------------------------------------------------------
    1
    3
    ------------------------------------------------------------------------------------------------------------------------
    a
    b
    c
    ------------------------------------------------------------------------------------------------------------------------
    97
    98
    99
    100
    101
    102
    ------------------------------------------------------------------------------------------------------------------------
    1
    3
    5
    7
    ------------------------------------------------------------------------------------------------------------------------
    0
    1
    2
    3
    4
    


```python
# dictionary for loop
for key, value in var_dict.items():
    print(f"{key} is {value}")
```

    a is 1
    b is 2
    

# Enumerate


```python
t = [1,2,3,4,5,6,7,11,25,50]
for i in enumerate(t):
    print(i)
# 튜플 형태로 (순서, 값)을 반환하는 구조
```

    (0, 1)
    (1, 2)
    (2, 3)
    (3, 4)
    (4, 5)
    (5, 6)
    (6, 7)
    (7, 11)
    (8, 25)
    (9, 50)
    
