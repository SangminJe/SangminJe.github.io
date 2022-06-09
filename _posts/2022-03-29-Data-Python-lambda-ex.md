---  
layout: post  
title: "[Python] lambda expression"
subtitle: "lambda 표현식 정리"  
categories: DATA
tags: python Python_Data pandas lmabda_expression
comments: true  
---  

자주 안쓰면 헷갈리는 **lambda_expression**을 정리해두었다.

```python
import pandas as pd

df = pd.read_csv("./data/titanic/train.csv")
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# apply lambda

### 일반 함수와 lambda 식 비교


```python
def get_square(a):
    return a**2

print("3의 제곱 : ", get_square(4))
```

    3의 제곱 :  16
    


```python
lambda_square = lambda x : x**2
print("3의 제곱은 :", lambda_square(4))
```

    3의 제곱은 : 16
    

### Map함수와 사용
**여러개의 인자를 받을 경우 `map()`함수와 함꼐 사용**


```python
a = [1,2,3]
squares = map(lambda x : x**2, a)
list(squares)
```




    [1, 4, 9]




```python
df["Name_len"] = df["Name"].apply(lambda x : len(x))
df[["Name_len", "Name"]].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name_len</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>Braund, Mr. Owen Harris</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>Heikkinen, Miss. Laina</td>
    </tr>
  </tbody>
</table>
</div>



### lambda 조건식
- elif는 지원하지 않음
- 원할 시 내포해서 써야함


```python
df["Child_Adult"] = df["Age"].apply(lambda x : "Child" if x<=15 else "Adult")
df[["Child_Adult", "Age"]].head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Child_Adult</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adult</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adult</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adult</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adult</td>
      <td>35.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["Age_cat"] = df["Age"].apply(lambda x : "child" if x <10 else ("teenager" if x < 20 else "Adult"))
df[["Age_cat", "Age"]].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age_cat</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adult</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adult</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adult</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adult</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adult</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adult</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adult</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>child</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Adult</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>teenager</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>



### 별도의 함수를 구성해서 사용하기


```python
def get_age_cat(age):
    cat = ''
    if age <=5 : cat = "Baby"
    elif age <12 : cat = "Child"
    elif age <20 : cat = "Teenager"
    elif age <60 : cat = "Adult"
    else : cat = "Elderly"
    
    return cat

df["Age_cat"] = df["Age"].apply(lambda x : get_age_cat(x))
df[["Age_cat", "Age"]].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age_cat</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adult</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adult</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adult</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adult</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adult</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Elderly</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Adult</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Baby</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Adult</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Teenager</td>
      <td>14.0</td>
    </tr>
  </tbody>
</table>
</div>


