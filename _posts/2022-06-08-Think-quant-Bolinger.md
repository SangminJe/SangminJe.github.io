---  
layout: post  
title: "[Quant] 볼린저 밴드 지표"
subtitle: ""  
categories: JESSIE
tags: JESSIE quant Bolinder
comments: true  
---  

# 볼린저 밴드 지표
---
- 주가의 20일 이동평균선을 기준으로 상대적인 고점을 나타내는 상단밴드, 저점을 나타내는 하단 밴드로 구성됨
    - 상단 볼린저 밴드 = 중간 볼린저 밴드 + (2 * 표준편차)
    - 상단 볼린저 밴드 = 종가의 20일 이동평균
    - 하단 볼린저 밴드 = 중간 볼린저 밴드 - (2 * 표준편차)

## %b
- 주가가 볼린저 밴드의 어디에 위치하는지 나타내는 지표
- 상단 밴드에 걸쳐 있으면 1, 중간에 걸쳐있으면 0.5, 하단에 걸쳐있으면 0이 됨
- %b가 1.1이라면 상단 밴드보다 밴드폭의 10%만큼 위에 있다는 의미

$$ \%b = \frac{종가 - 하단 볼린저 밴드}{상단 볼린저 밴드 - 하단 볼린저 밴드} $$

## 밴드폭
- 상단 볼린저 밴드와 하단 볼린저 밴드 사이의 폭
- **스퀴즈**륵 확인하는 데 유용한 지표
- 스퀴즈란 변동성이 극히 낮은 수준까지 떨어져 곧 변동성 증가가 발생할 것으로 예상되는 상황
- 강력한 추세의 시작과 마지막을 포착

$$ 밴드폭 = \frac{상단 볼린저 밴드 - 하단 볼린저 밴드}{중간 볼린저 밴드}$$