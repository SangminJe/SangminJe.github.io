---  
layout: post  
title: "[Python] 비트코인 자동매매 로직 수정"
subtitle: "매수 후 로직 추가"  
categories: DATA
tags: DATA python Python_Data 비트코인 자동매매
comments: true  
---  

# 문제 : 로직 구현 후 자꾸 발생하는 손실
---
- 장이 대체적으로 하락장이긴 했으나 손실이 누적됨
- 변동성이 높은 coin 시장임을 감안하고 높은 MDD도 감안했으나 3개월 정도의 테스트 결과 성능이 좋지 않았음

# 해결 : 손실을 최소화하자
---
- 금번 물가 상승과 금리 인상으로 인해 일 단위 변동성 돌파가 잘 효과를 보지 못했음
- 하루의 특정 시간에 매도를 수행하므로, 그 시간을 제외한 하루 안에 일어난 손실이나 이익을 포착할 수 있는 로직이 필요하다고 생각했음
- 그래서 생각한 두 가지 안
    **1. 매수 후 1%라도 손해가 나면 바로 매도 후 보유**
    해당 로직은 손실을 최소화해서 손절만 하더라도 단순 보유보다 높은 효과를 낼 수 있다는 퀀트 방법론에서 영감을 받음
    **2. 매수 후 5% 이상 이익이 나면 바로 매도 후 보유**
    이익을 내고도 일정 시간이 되면 변동성으로 인해 이익이 사라질 수 있으므로 얼른 이득을 내고 보유하는 로직(카지노에서 돈 따고 도망가는 느낌)

# 코드
---

```py
"""2022.08.15 수정내역
1. 매수 후 1% 이상 하락 시 매도 후 잔고 보우
2. 매수 후 5% 이상 먹었으면 매도 후 잔고 보유

위 로직을 구현하기 위해 Buy_Flag 변수를 추가함
"""


#### 변동성 돌파 매수/매도 프로그램 ####
import pyupbit
import time
import datetime
import pandas as pd
import numpy as np
import requests


####### 백테스트 로직 ##########
def get_ror(coin, k=0.5, n=5):
    df = pyupbit.get_ohlcv("KRW-"+coin, count=365)

    # 전략 적용
    df["range"] = (df["high"] - df["low"]) * k # 전일 저가~고가 interval
    df["target"] = df["open"] + df["range"].shift(1) # 목표가
    df["ma5"] = df["close"].rolling(n).mean().shift(1)
    df["bool"] = df["open"] > df["ma5"]
    
    fee = 0.001

    # 수익률
    df["ror"] = np.where((df["high"] > df["target"])&df["bool"],
                         df["close"] / df["target"] - fee,
                         1)

    # 누적 수익률
    ror = df["ror"].cumprod()[-2]
    
    # 낙폭 구하기
    df["hpr"] = df["ror"].cumprod()
    df["dd"] = (df["hpr"].cummax() - df["hpr"]) / df["hpr"].cummax() * 100
    MDD = df["dd"].max()
#     print("MDD(%) : ", df["dd"].max())
    return ror, MDD

# 최적의 k값 구하기
rors = {"n":[], "k":[], "ror":[], "MDD":[]}
for k in np.arange(0.1, 1.0, 0.1):
    for n in [5, 10, 15, 20, 30]:
        ror, MDD = get_ror("ARK", k, n)
        rors["n"].append(n)
        rors["k"].append(k)
        rors["ror"].append(ror)
        rors["MDD"].append(MDD)
#         print(f"k : {k} , n : {n} 일 때, ror : {ror:.3f}")
        
today = datetime.date.today()
max_ror_ind = rors["ror"].index(max(rors["ror"]))

# 최종 n, k 구하기
final_n = rors["n"][max_ror_ind]
final_k = rors["k"][max_ror_ind]
final_ror = rors["ror"][max_ror_ind]

print(f"{today} 기준 Back Test 결과, n : {final_n}, k : {final_k} 일 때, 최고의 ror : {final_ror}, MDD:{MDD}")


################### Slack Bot #########################

myToken = "YOUR TOKEN" # 내 Slack 토큰

def post_message(token, channel, text):
    """ Slack API에 연결하는 함수"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )
    print(response)

def dbgout(myToken, channel, message):
    """인자로 받은 문자열을 파이썬 셸과 슬랙으로 동시에 출력한다."""
    print(datetime.datetime.now().strftime('[%m/%d %H:%M:%S]'), message)
    strbuf = datetime.datetime.now().strftime('[%m/%d %H:%M:%S] ') + message
    post_message(myToken, channel, strbuf)


############# 매수/매도 로직 #################


access = "YOUR_ACESS"          # 본인 값으로 변경
secret = "YOUR_ACESS"          # 본인 값으로 변경
upbit = pyupbit.Upbit(access, secret)

print("autotrade start")

# 시작 메세지 슬랙 전송
post_message(myToken,"#crypto", "autotrade start")

def get_target_price(ticker, k):
    """변동성 돌파 전략으로 매수 목표가 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price

def get_start_time(ticker):
    """시작 시간 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time

def get_current_price(ticker):
    """현재가 조회"""
    return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"] # 가장 낮은 매도호가


def get_ma_n(ticker, n=30):
    """n일 이동 평균선 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=180)
    ma = df['close'].rolling(n).mean().iloc[-1]
    return ma

def get_bal(ticker):
    """잔고조회"""
    bal = upbit.get_balance(ticker) # 현재 원화 잔고
    if bal is not None:
        return float(bal)
    else:
        return 0

def buy_coin_all(coin, krw):
    """매수 후 Slack에 알리기"""
    buy_result = upbit.buy_market_order("KRW-"+coin, krw*0.9995) # 매수
    buy_info_dict = {"코인종류" : buy_result["market"], "내가 산 금액" : buy_result["price"], "예정 수수료" : buy_result["reserved_fee"]}
    dbgout(myToken,"#bit", f"{coin} buy : " +str(buy_info_dict))

def sell_coin_all(coin, coin_bal):
    """매도 후 Slack에 알리기, coin_bal = 코인 잔고"""
    sel_result = upbit.sell_market_order("KRW-"+coin, coin_bal*0.9995) # 전량 매도
    sell_info_dict = {"코인종류" : sel_result["market"], "내가 산 금액" : sel_result["price"]}
    dbgout(myToken,"#bit", f"{coin} sell : {str(sel_result)}")    


#### ---- 프로그램 시작 ---- ####

coin = "ANY_COIN_YOU_WANT"
k = final_k
ma_n = final_n
buy_flag = 0
      
while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-"+coin)
        end_time = start_time + datetime.timedelta(days=1)
        current_price = get_current_price("KRW-"+coin)
        target_price = get_target_price("KRW-"+coin, k)
        ma = get_ma_n("KRW-"+coin, ma_n) # n일 이동평균
        
        # 상승/하락장 표시
        up_down = "하락장" if current_price < ma else "상승장"
        
        # ---- 로직 시작 --- #
        
        # 매수
        if start_time < now < end_time - datetime.timedelta(seconds=10) : # 09:00:00 ~ 08:59:50
            if now.minute == 0 and (now.second == 0 or now.second == 1): # 00:00 ~ 00:01 사이에 메세지
                print(f" 현재가 : {current_price}, 목표가 : {target_price}, 이동평균 : {ma}")
                dbgout(myToken,"#bit", f" 현재가 : {current_price}, 목표가 : {target_price}, 이동평균 : {ma}, 현재 장 상태 : {up_down} ")
            
            else: # 순수 매수 로직 가동 시간
                
                # 순수 변동성 돌파 매수 로직
                if (current_price > target_price) and (current_price > ma):
                    krw = get_bal("KRW") # 현재 원화 잔고
                    if krw > 5000 and buy_flag == 0: # 원화잔고가 5000원 이상이고 Buy_Flag가 0일 때만 동작
                        buy_coin_all(coin, krw)
                
                # 내가 산 가격에서 1% 이상 떨어진다면 전량 매도
                if (target_price - current_price) / target_price >= 0.01:
                    coin_bal = get_bal(coin) # Coin 잔고
                    if (coin_bal*current_price) > 5000: 
                        buy_flag = 1 # flag 변경
                        sell_coin_all(coin, coin_bal)
                
                # 5% 이상 먹었으면 그날 물량 전량 매도
                if (current_price - target_price) / target_price >= 0.05:
                    coin_bal = get_bal(coin) # Coin 잔고
                    if (coin_bal*current_price) > 5000: 
                        buy_flag = 1 # flag 변경
                        sell_coin_all(coin, coin_bal)
    
        # 매도
        else:
            coin_bal = get_bal(coin) # Coin잔고
            if (coin_bal*current_price) > 5000: 
                sell_coin_all(coin, coin_bal)
                
            if buy_flag == 1:
                buy_flag = 0 # flag 변경        
                
        time.sleep(2)

    except Exception as e:
        print(e)
        dbgout(myToken,"#bit", str(e))
        time.sleep(1)

```