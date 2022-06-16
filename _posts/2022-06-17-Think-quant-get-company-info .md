---  
layout: post  
title: "[Quant] Dart API를 이용해서 상장기업의 재무제표 크롤링"
subtitle: ""  
categories: JESSIE
tags: JESSIE python quant crawling dart api 재무제표 크롤링
comments: true  
---  
```python
import requests
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import re
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
```

# 재무제표 및 가치지표

## 1. API 키 받기

- Open Dart를 사용해서 재무제표와 가치지표를 가져오고자 함
- API Key가 있어야 하므로 해당 사이트에서 가입절차를 완료한 후 API 키를 받아야함
- https://opendart.fss.or.kr/mng/apiUsageStatusView.do


![img](https://sangminje.github.io/assets/img/quant/api1png.png)


```python
dart_api = "Your API Key"
```

## 2. 고유번호 받기
- 각 기업에 해당하는 데이터 받기
- https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019018 사이트에 가면 개발정보가 있음
- Zipfile 안에 있는 "CORPCODE.xml"이라는 xml 파일을 읽어내는 프로세스


```python
get_url = "https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key=" + dart_api # Data 요청 URL

resp = urlopen(get_url) # url
zipfile = ZipFile(BytesIO(resp.read())) # Zipfile 읽기
# code_data = zipfile.open("CORPCODE.xml").read().decode("utf-8","replace") # Zipfile 열면서 utf-8로 디코딩
code_data = zipfile.open("CORPCODE.xml").read()
soup = bs(code_data,"xml") # beautifulsoup를 통해 xml 읽기
```

- `corp_code`는 dart에서 사용하게 될 회사코드이며
- `corp_name`은 회사명임


```python
# soup.find_all(["corp_code","corp_name"])
corp_codes = []
for i in soup.find_all("corp_code"):
    corp_codes.append(i.get_text())
corp_codes[:5]
```




    ['00434003', '00434456', '00430964', '00432403', '00388953']




```python
corp_names = []
for i in soup.find_all("corp_name"):
    corp_names.append(i.get_text())
corp_names[:5]
```




    ['다코', '일산약품', '굿앤엘에스', '한라판지', '크레디피아제이십오차유동화전문회사']




```python
stock_codes = []
for i in soup.find_all("stock_code"):
    stock_codes.append(i.get_text())
stock_codes[:5]
```




    [' ', ' ', ' ', ' ', ' ']



- 종목코드(티커)가 없는 기업도 많음
- 우리가 관심있는 기업은 티커가 있는 기업이므로 아래 데이터프레임에서 필터링을 해줌


```python
corp_df = pd.DataFrame({"code" : corp_codes, "name": corp_names, "ticker": stock_codes})
corp_df = corp_df[corp_df["ticker"] != ' '].reset_index().drop("index", axis=1) # ticker가 없는 종목은 drop
corp_df
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
      <th>code</th>
      <th>name</th>
      <th>ticker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00260985</td>
      <td>한빛네트</td>
      <td>036720</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00264529</td>
      <td>엔플렉스</td>
      <td>040130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00358545</td>
      <td>동서정보기술</td>
      <td>055000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00231567</td>
      <td>애드모바일</td>
      <td>032600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00247939</td>
      <td>씨모스</td>
      <td>037600</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3435</th>
      <td>00442455</td>
      <td>코스나인</td>
      <td>082660</td>
    </tr>
    <tr>
      <th>3436</th>
      <td>00567222</td>
      <td>우림피티에스</td>
      <td>101170</td>
    </tr>
    <tr>
      <th>3437</th>
      <td>00361594</td>
      <td>DMS</td>
      <td>068790</td>
    </tr>
    <tr>
      <th>3438</th>
      <td>00141626</td>
      <td>오리엔트바이오</td>
      <td>002630</td>
    </tr>
    <tr>
      <th>3439</th>
      <td>00985686</td>
      <td>큐브엔터</td>
      <td>182360</td>
    </tr>
  </tbody>
</table>
<p>3440 rows × 3 columns</p>
</div>



- 그래도 있지도 않은 기업들이 껴있음
- 해당 기업에서 지금 존재하는 기업들만 솎아내야 함


```python
# KRX에 상장되어있는 기업항목 가져오기
market_list = ["STK", "KSQ"]
ticker = []
name = []

for mk in market_list:
    otp_url = "http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"
    data = {
        "locale": "ko_KR",
        "mktId": mk,
        "trdDd": "20220607", # 가장 최근 영업일 기준으로 조회 필요
        "money": "1",
        "csvxls_isNo": "false",
        "name": "fileDown",
        "url": "dbms/MDC/STAT/standard/MDCSTAT03901"
    }

    response = requests.post(otp_url, data=data)
    otp = bs(response.text, "html.parser")# OTP 생성

    down_url = "http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd"
    down_res = requests.post(down_url, data= {"code" : otp}, headers={"Referer" : otp_url})

    data = bs(down_res.content.decode("euc-kr","replace"), "html.parser") # 인코딩 깨지지 않게
    str_data = data.text.split("\n")

    temp_ticker = [x.replace('"','') for x in [x.split('","')[0] for x in str_data[1:]]]
    temp_name = [x.replace('"','') for x in [x.split('","')[1] for x in str_data[1:]]]
    ticker.extend(temp_ticker)
    name.extend(temp_name)

all_data = list(zip(ticker, name))
ticker_df = pd.DataFrame(all_data, columns = ['ticker','company'])
```


```python
ticker_df.head(5)
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
      <th>ticker</th>
      <th>company</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>095570</td>
      <td>AJ네트웍스</td>
    </tr>
    <tr>
      <th>1</th>
      <td>006840</td>
      <td>AK홀딩스</td>
    </tr>
    <tr>
      <th>2</th>
      <td>027410</td>
      <td>BGF</td>
    </tr>
    <tr>
      <th>3</th>
      <td>282330</td>
      <td>BGF리테일</td>
    </tr>
    <tr>
      <th>4</th>
      <td>138930</td>
      <td>BNK금융지주</td>
    </tr>
  </tbody>
</table>
</div>




```python
intersected_df = pd.merge(corp_df, ticker_df, how='inner') # Inner join
```


```python
intersected_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2374 entries, 0 to 2373
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   code     2374 non-null   object
     1   name     2374 non-null   object
     2   ticker   2374 non-null   object
     3   company  2374 non-null   object
    dtypes: object(4)
    memory usage: 92.7+ KB
    

## 3. 상장기업 재무정보 받기
- xml 데이터 형식으로 받는 것으로 설정
- 삼성전자 데이터를 테스트로 해서 연결재무제표 전체를 가져와보기


```python
# corp_code = "00126380"
corp_code = "00985686"
year="2020"
get_url=f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.xml?crtfc_key={dart_api}&corp_code={corp_code}&bsns_year={year}&reprt_code=11011&fs_div=CFS"

resp = urlopen(get_url) # url 오픈
df = pd.read_xml(resp) # xml 읽기
df2 = df[["sj_div","sj_nm","account_nm", "thstrm_amount"]] # 재무재표구분, 재무제표명, 계정명, 당기금액
df2["year"] = year
```

##  4. 년도 별, 회사 별 정보 크롤링

- all_df 데이터프레임에 데이터가 쌓임
- 일부 재무제표자료가 없는 경우가 있으몰, try except문을 통해 오류를 잡고 다음 code로 넘어가도록 수행


```python
import time

codes = intersected_df["code"].values.tolist()

all_df = pd.DataFrame()
for code in codes:
    try:
        for year in range(2015, 2022): # 2015 ~ 2021년
            get_url=f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.xml?crtfc_key={dart_api}&corp_code={code}&bsns_year={year}&reprt_code=11011&fs_div=CFS"

            resp = urlopen(get_url)
    #         soup = bs(resp, "xml")
    #         if soup.text == "013조회된 데이타가 없습니다.":
    #             continue
    #         else:
            df = pd.read_xml(resp)
            df = df[["sj_div","sj_nm","account_nm", "thstrm_amount"]] # 재무재표구분, 재무제표명, 계정명, 당기금액
            df["year"] = year # 해당년도
            df["ticker"] = intersected_df[intersected_df["code"] == '00956028']["ticker"][0] # 티커
            df["company"] = intersected_df[intersected_df["code"] == '00956028']["name"][0] # 회사명
            
            df = df[["ticker","company","year","sj_div","sj_nm","account_nm","thstrm_amount"]] # 순서 재조합
            all_df = pd.concat([all_df, df])
        
    except Exception as e:
        print(f"Error : at {code}, {year}", str(e))
        continue
    
    time.sleep(1)

all_df.reset_index().drop("index", axis=1, inplace=True) # index Resetting
```
