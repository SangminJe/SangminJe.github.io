---  
layout: post  
title: "[Python] opendartreader Library - 재무제표 가져오기"
subtitle: "Library를 잘쓰자"  
categories: DATA
tags: DATA python Python_Data beautiful-soup post KRX Data 주식데이터
comments: true  
---  

# Python으로 삽질하다
DART Open API를 사용하겠다고 처음에 API 문서를 보면서 삽질을 했다. 비루한 코딩 실력으로 원하는 결과를 반환하지 못하며 절망하고 있을 때, 구글링을 통해 `opendartreader` 라이브러리가 있는 것을 확인했다. 비로소 나는 검색의 중요성을 다시금 인식하였고, 낭만적인 Python 개발자들이 이미 그 어려운 길에 평화를 주고자 API를 만들어놓았다는 사실에 눈물이 났다.

# opendartreader
1. opendartreader 설치
```cmd
pip install opendartreader
```

2. 원하는 기업의 재무제표 가져오기 Test

```py
import OpenDartReader

# ==== 0. 객체 생성 ====
# 객체 생성 (API KEY 지정) 
api_key = "Your API KEY"

dart = OpenDartReader(api_key) 

# 단일기업 전체 재무제표 (삼성전자 2018 전체 재무제표)
dart.finstate_all('005930', 2021)
```

**결론 : 제발 라이브러리가 있으면 그걸 가져다가 쓰자 ㅠㅠ**