---  
layout: post  
title: "Flask를 이용한 웹 백엔드 기초 (1)"
subtitle: "flask를 이용한 routing"  
categories: DEV
tags: DEV Backend Python flask 
comments: true  
---  


# Python Backend를 공부하겠다고 생각한 이유

나는 데이터를 다루는 것도 좋아하고 데이터로 뭔가 분석해서 결과물을 내는 것도 좋아하지만, 그 데이터로 소프트웨어의 결과물을 내고 싶다는 생각을 한 적이 많다. 나중에는 내가 가진 아이디어를 서비스로 풀어내보고 싶다는 생각이 들어 가장 간단하다고 소문난 Flask를 배우고 이를 기록하고자 한다.(누굴까 간단하다는 사람..) 


# Routing

---

```python

from flask import Flask # Flask import

app = Flask(__name__)

# 곱셈을 해주는 함수 제작
def multiply(digit):
    return digit*digit

# 기본 라우팅 주소 생성
# 아무것도 없으므로 localhost:8080로 들어갈 경우 바로 나오는 화면
@app.route('/')
def hello():
    return "<h1> Hello Sangmin! </h1>"

# 라우팅을 first로 하고 message를 integer로만 인자로 받음
# 받은 message는 결과로 빠짐
@app.route('/first/<int:message>')
def hi(message):
    result = multiply(message)
    return "<h1>%d</h1>" %result

# Localhost로 서버 구동
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = '8080')
```

- `__main__`는 파이썬 내장함수이고 모듈이름을 나타내는 함수이다.
- 라우팅을 통해서 원하는 URL을 기초로 여러 곳으로 정보를 분산하여 구축할 수 있다는 생각이 들었다.
- message를 인자로 받는 저 부분은 나중에 아이디를 인자로 받을 수 있을 것 같다. 하지만 분명 보안이슈가 있겠지? 현재 생각은 그렇다.
