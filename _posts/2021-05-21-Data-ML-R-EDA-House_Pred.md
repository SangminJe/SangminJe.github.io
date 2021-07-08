---  
layout: post  
title: "Tidymodels로 시작하는 머신러닝 - House Price Prediction (1)"
subtitle: "Data 확인과 EDA"  
categories: DATA
tags: Data ML R tidymodels EDA 타이디모델 R-machine-learning House-Price-Prediction
comments: true  
---  

## 개요

> 1. Kaggle의 [House Price](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) Competition을 참여합니다.
> 2. 본 문서에서는 Tidyverse를 통한 EDA를 다룹니다.
> 3. 본 분석은 [슬기로운 통계생활](https://www.youtube.com/user/Leesak85) 채널을 운영하시는 **이삭**님의 Kaggle Study 2기에 참여하여 진행하였습니다.

- tidymodels 공부 관련 포스트
  - [1. Build a Model](https://sangminje.github.io/data/2021/04/22/Data-ML-tidymodels-build-a-model/)
  - [2. Recipe](https://sangminje.github.io/data/2021/04/30/Data-ML-tidymodels-Recipe/)
  - [3. Evaluation](https://sangminje.github.io/data/2021/05/03/Data-ML-tidymodels-Evaluation/)
  - [4. Hyperparameter Tuning](https://sangminje.github.io/data/2021/05/08/Data-ML-tidymodels-hyperparameter-tune/)
  - [5. Case Study](https://sangminje.github.io/data/2021/05/13/Data-ML-tidymodels-case-study/)

- 목차
  - [1. Data](#data)
  - [2. Data Load and Check](#data-load-and-check)
  - [3. EDA](#eda)
  - [4. 결론](#결론)


## Data

------------------------------------------------------------------------

이번 주택 데이터의 목록입니다. 미국의 주택을 기준으로 데이터를 만든 것 같은데, 한국 아파트가 더 익숙한 저로서는 익숙하지 않은 항목들이 많았습니다.    
데이터 목록을 보시기 전에 먼저 알아두시면 좋은 점은, **Quality**와 **Condition**을 구분하는 관점입니다. **Quality**는 돈을 들여서 잘 만든 수준을 이야기합니다. 예를 들어서 벽난로가 나무로 만들어진 것과 대리석으로 만들어진 것이 있다면, 대리석으로 만들어진 게 보통 **Quality**가 좋다고 생각하실 겁니다. **Condition**은 이 벽난로의 관리상태입니다. 대리적이지만 깨져있다면 좋은 컨디션이 아니겠죠. 이 두 개념을 한국어로 생각하면 헷갈리기 쉬운 것 같아 먼저 언급을 드리고 넘어가겠습니다.    

-   **SalePrice** - the property’s sale price in dollars. This is the target variable that you’re trying to predict. (**우리의 종속변수입니다.**)
-   MSSubClass: The building class / 주택의 종류
-   MSZoning: The general zoning classification / 주택이 위치한 토지종류
-   LotFrontage: Linear feet of street connected to property / 집과 Street로부터 떨어진 거리를 의미
-   LotArea: Lot size in square feet / 택지의 크기
-   Street: Type of road access / 집까지 도로가 포장되어 있는지 여부
-   Alley: Type of alley access / 골목이 있는지 여부
-   LotShape: General shape of property / 택지의 모양
-   LandContour: Flatness of the property / 땅의 고르기
-   Utilities: Type of utilities available / 수도, 가스, 전기 시설 가능여부
-   LotConfig: Lot configuration / 택지 구성
-   LandSlope: Slope of property / 땅의 경사도
-   Neighborhood: Physical locations within Ames city limits / 인접 도시
-   Condition1: Proximity to main road or railroad / 인접한 시설에 대한 내용인 듯합니다. 예를 들어 철도가 근처에 있다던가, 공원이 있다던가 하는 내용들이네요.
-   Condition2: Proximity to main road or railroad (if a second is present) / Condition1에서 추가적으로 있는 인접 시설을 가리킵니다.
-   BldgType: Type of dwelling / 주거형태
-   HouseStyle: Style of dwelling / 주택의 형태(단층, 복층)
-   OverallQual: Overall material and finish quality / 집 자제와 마감 품질
-   OverallCond: Overall condition rating / 집의 상태
-   YearBuilt: Original construction date / 건축년도
-   YearRemodAdd: Remodel date / 리모델 일자
-   RoofStyle: Type of roof / 지붕의 스타일
-   RoofMatl: Roof material / 지붕의 자재
-   Exterior1st: Exterior covering on house / 주택의 외관(벽돌인지, 시멘트인지 등)
-   Exterior2nd: Exterior covering on house (if more than one material)
-   MasVnrType: Masonry veneer type / 외관의 일종인 것 같습니다.
-   MasVnrArea: Masonry veneer area in square feet /
-   ExterQual: Exterior material quality / 외관의 수준(poor \~ excellent)
-   ExterCond: Present condition of the material on the exterior / 외관의 상태(poor \~ excellent)
-   Foundation: Type of foundation / 집의 기초공사 종류를 의미합니다. 터 라고 생각하시면 편할 듯 하네요.
-   BsmtQual: Height of the basement / 지하실의 퀄리티인데, 높이로 측정하나 봅니다.
-   BsmtCond: General condition of the basement / 지하실 컨디션
-   BsmtExposure: Walkout or garden level basement walls / [garden level basement](https://images.app.goo.gl/zmhWDL5uLm2v9kBL6), 지하실의 형태와 노출 정도를 의미합니다.
-   BsmtFinType1: Quality of basement finished area / 지하실 갖춰진 정도입니다.(전기가 들어오는지, 계단이 잘 설치되었는지,아니면 창고수준인지 등)
-   BsmtFinSF1: Type 1 finished square feet / 갖춰진 지하실의 면적
-   BsmtFinType2: Quality of second finished area (if present)
-   BsmtFinSF2: Type 2 finished square feet
-   BsmtUnfSF: Unfinished square feet of basement area/ 마감이 안된 지하실 면적
-   TotalBsmtSF: Total square feet of basement area / 지하실 전체 면적 Heating: Type of heating / 난방 종류
-   HeatingQC: Heating quality and condition / 난방 퀄리티와 상태(good\~poor)
-   CentralAir: Central air conditioning / 중앙 에어컨 여부
-   Electrical: Electrical system / 전기 시스템의 상태(good\~poor)
-   1stFlrSF: First Floor square feet / 1층의 면적
-   2ndFlrSF: Second floor square feet / 2층의 면적
-   LowQualFinSF: Low quality finished square feet (all floors) / 모든 층의 마감이 안된 면적
-   GrLivArea: Above grade (ground) living area square feet / 실거주가능 면적
-   BsmtFullBath: Basement full bathrooms / 지하공간 욕실 개
-   BsmtHalfBath: Basement half bathrooms / 지하공간 작은 욕실 개수
-   FullBath: Full bathrooms above grade / 지상 욕실 개수
-   HalfBath: Half baths above grade / 지상 작은 욕실 개수
-   Bedroom: Number of bedrooms above basement level / 지상 침실 개수
-   Kitchen: Number of kitchens / 부엌 개수
-   KitchenQual: Kitchen quality / 부엌의 상태
-   TotRmsAbvGrd: Total rooms above grade (does not include bathrooms) / 방 개수
-   Functional: Home functionality rating / 집의 전체적인 상태를 의미하는 듯합니다.
-   Fireplaces: Number of fireplaces / 난로, 벽난로 개수
-   FireplaceQu: Fireplace quality / 난로의 상태
-   GarageType: Garage location / 차고의 형태
-   GarageYrBlt: Year garage was built / 차고의 건축년도
-   GarageFinish: Interior finish of the garage / 차고 인테리어 년도
-   GarageCars: Size of garage in car capacity / 차고의 수용(차의 크기로)
-   GarageArea: Size of garage in square feet / 차고의 수용정도(피트로)
-   GarageQual: Garage quality / 차고의 수준
-   GarageCond: Garage condition / 차고의 상태
-   PavedDrive: Paved driveway / 차고에서부터 길이 포장되어 있는지
-   WoodDeckSF: Wood deck area in square feet / 나무 덱(테라스 같은 공간)의 면적
-   OpenPorchSF: Open porch area in square feet / 포치의 면적, [포치](https://images.app.goo.gl/FzGNrU2DfKcLh6Zo6)
-   EnclosedPorch: Enclosed porch area in square feet / 내부포치의 면적
-   3SsnPorch: Three season porch area in square feet / 3개면이 외부와 닿은 돌출형 포치의 면적
-   ScreenPorch: Screen porch area in square feet / 스크린 설치된 포치의 면적
-   PoolArea: Pool area in square feet / 수영장 면적
-   PoolQC: Pool quality / 수영장 수준
-   Fence: Fence quality / 울타리 수준
-   MiscFeature: Miscellaneous feature not covered in other categories / 기타 특징
-   MiscVal: $Value of miscellaneous feature / 기타 특징의 가격
-   MoSold: Month Sold / 판매된 달
-   YrSold: Year Sold / 판매된 년도
-   SaleType: Type of sale / 판매 방법
-   SaleCondition: Condition of sale / 거래 방법

## Library

------------------------------------------------------------------------

``` r
library(tidyverse)
library(tidymodels)
library(skimr)
```

-   전처리와 시각화를 위해 `tidyverse` 패키지를 주로 사용합니다.

## Data Load and Check

------------------------------------------------------------------------

``` r
train <- read_csv('data/train.csv')
test <- read_csv('data/test.csv')

all_origin <- bind_rows(train, test)
```

``` r
skim(all_origin)
```

|                                                  |            |
|:-------------------------------------------------|:-----------|
| Name                                             | all_origin |
| Number of rows                                   | 2919       |
| Number of columns                                | 81         |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |            |
| Column type frequency:                           |            |
| character                                        | 43         |
| numeric                                          | 38         |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |            |
| Group variables                                  | None       |

Data summary

**Variable type: character**

| skim_variable | n_missing | complete_rate | min | max | empty | n_unique | whitespace |
|:--------------|----------:|--------------:|----:|----:|------:|---------:|-----------:|
| MSZoning      |         4 |          1.00 |   2 |   7 |     0 |        5 |          0 |
| Street        |         0 |          1.00 |   4 |   4 |     0 |        2 |          0 |
| Alley         |      2721 |          0.07 |   4 |   4 |     0 |        2 |          0 |
| LotShape      |         0 |          1.00 |   3 |   3 |     0 |        4 |          0 |
| LandContour   |         0 |          1.00 |   3 |   3 |     0 |        4 |          0 |
| Utilities     |         2 |          1.00 |   6 |   6 |     0 |        2 |          0 |
| LotConfig     |         0 |          1.00 |   3 |   7 |     0 |        5 |          0 |
| LandSlope     |         0 |          1.00 |   3 |   3 |     0 |        3 |          0 |
| Neighborhood  |         0 |          1.00 |   5 |   7 |     0 |       25 |          0 |
| Condition1    |         0 |          1.00 |   4 |   6 |     0 |        9 |          0 |
| Condition2    |         0 |          1.00 |   4 |   6 |     0 |        8 |          0 |
| BldgType      |         0 |          1.00 |   4 |   6 |     0 |        5 |          0 |
| HouseStyle    |         0 |          1.00 |   4 |   6 |     0 |        8 |          0 |
| RoofStyle     |         0 |          1.00 |   3 |   7 |     0 |        6 |          0 |
| RoofMatl      |         0 |          1.00 |   4 |   7 |     0 |        8 |          0 |
| Exterior1st   |         1 |          1.00 |   5 |   7 |     0 |       15 |          0 |
| Exterior2nd   |         1 |          1.00 |   5 |   7 |     0 |       16 |          0 |
| MasVnrType    |        24 |          0.99 |   4 |   7 |     0 |        4 |          0 |
| ExterQual     |         0 |          1.00 |   2 |   2 |     0 |        4 |          0 |
| ExterCond     |         0 |          1.00 |   2 |   2 |     0 |        5 |          0 |
| Foundation    |         0 |          1.00 |   4 |   6 |     0 |        6 |          0 |
| BsmtQual      |        81 |          0.97 |   2 |   2 |     0 |        4 |          0 |
| BsmtCond      |        82 |          0.97 |   2 |   2 |     0 |        4 |          0 |
| BsmtExposure  |        82 |          0.97 |   2 |   2 |     0 |        4 |          0 |
| BsmtFinType1  |        79 |          0.97 |   3 |   3 |     0 |        6 |          0 |
| BsmtFinType2  |        80 |          0.97 |   3 |   3 |     0 |        6 |          0 |
| Heating       |         0 |          1.00 |   4 |   5 |     0 |        6 |          0 |
| HeatingQC     |         0 |          1.00 |   2 |   2 |     0 |        5 |          0 |
| CentralAir    |         0 |          1.00 |   1 |   1 |     0 |        2 |          0 |
| Electrical    |         1 |          1.00 |   3 |   5 |     0 |        5 |          0 |
| KitchenQual   |         1 |          1.00 |   2 |   2 |     0 |        4 |          0 |
| Functional    |         2 |          1.00 |   3 |   4 |     0 |        7 |          0 |
| FireplaceQu   |      1420 |          0.51 |   2 |   2 |     0 |        5 |          0 |
| GarageType    |       157 |          0.95 |   6 |   7 |     0 |        6 |          0 |
| GarageFinish  |       159 |          0.95 |   3 |   3 |     0 |        3 |          0 |
| GarageQual    |       159 |          0.95 |   2 |   2 |     0 |        5 |          0 |
| GarageCond    |       159 |          0.95 |   2 |   2 |     0 |        5 |          0 |
| PavedDrive    |         0 |          1.00 |   1 |   1 |     0 |        3 |          0 |
| PoolQC        |      2909 |          0.00 |   2 |   2 |     0 |        3 |          0 |
| Fence         |      2348 |          0.20 |   4 |   5 |     0 |        4 |          0 |
| MiscFeature   |      2814 |          0.04 |   4 |   4 |     0 |        4 |          0 |
| SaleType      |         1 |          1.00 |   2 |   5 |     0 |        9 |          0 |
| SaleCondition |         0 |          1.00 |   6 |   7 |     0 |        6 |          0 |

**Variable type: numeric**

| skim_variable | n_missing | complete_rate |      mean |       sd |    p0 |      p25 |      p50 |      p75 |   p100 | hist  |
|:--------------|----------:|--------------:|----------:|---------:|------:|---------:|---------:|---------:|-------:|:------|
| Id            |         0 |          1.00 |   1460.00 |   842.79 |     1 |    730.5 |   1460.0 |   2189.5 |   2919 | ▇▇▇▇▇ |
| MSSubClass    |         0 |          1.00 |     57.14 |    42.52 |    20 |     20.0 |     50.0 |     70.0 |    190 | ▇▅▂▁▁ |
| LotFrontage   |       486 |          0.83 |     69.31 |    23.34 |    21 |     59.0 |     68.0 |     80.0 |    313 | ▇▃▁▁▁ |
| LotArea       |         0 |          1.00 |  10168.11 |  7887.00 |  1300 |   7478.0 |   9453.0 |  11570.0 | 215245 | ▇▁▁▁▁ |
| OverallQual   |         0 |          1.00 |      6.09 |     1.41 |     1 |      5.0 |      6.0 |      7.0 |     10 | ▁▂▇▅▁ |
| OverallCond   |         0 |          1.00 |      5.56 |     1.11 |     1 |      5.0 |      5.0 |      6.0 |      9 | ▁▁▇▅▁ |
| YearBuilt     |         0 |          1.00 |   1971.31 |    30.29 |  1872 |   1953.5 |   1973.0 |   2001.0 |   2010 | ▁▂▃▆▇ |
| YearRemodAdd  |         0 |          1.00 |   1984.26 |    20.89 |  1950 |   1965.0 |   1993.0 |   2004.0 |   2010 | ▅▂▂▃▇ |
| MasVnrArea    |        23 |          0.99 |    102.20 |   179.33 |     0 |      0.0 |      0.0 |    164.0 |   1600 | ▇▁▁▁▁ |
| BsmtFinSF1    |         1 |          1.00 |    441.42 |   455.61 |     0 |      0.0 |    368.5 |    733.0 |   5644 | ▇▁▁▁▁ |
| BsmtFinSF2    |         1 |          1.00 |     49.58 |   169.21 |     0 |      0.0 |      0.0 |      0.0 |   1526 | ▇▁▁▁▁ |
| BsmtUnfSF     |         1 |          1.00 |    560.77 |   439.54 |     0 |    220.0 |    467.0 |    805.5 |   2336 | ▇▅▂▁▁ |
| TotalBsmtSF   |         1 |          1.00 |   1051.78 |   440.77 |     0 |    793.0 |    989.5 |   1302.0 |   6110 | ▇▃▁▁▁ |
| 1stFlrSF      |         0 |          1.00 |   1159.58 |   392.36 |   334 |    876.0 |   1082.0 |   1387.5 |   5095 | ▇▃▁▁▁ |
| 2ndFlrSF      |         0 |          1.00 |    336.48 |   428.70 |     0 |      0.0 |      0.0 |    704.0 |   2065 | ▇▃▂▁▁ |
| LowQualFinSF  |         0 |          1.00 |      4.69 |    46.40 |     0 |      0.0 |      0.0 |      0.0 |   1064 | ▇▁▁▁▁ |
| GrLivArea     |         0 |          1.00 |   1500.76 |   506.05 |   334 |   1126.0 |   1444.0 |   1743.5 |   5642 | ▇▇▁▁▁ |
| BsmtFullBath  |         2 |          1.00 |      0.43 |     0.52 |     0 |      0.0 |      0.0 |      1.0 |      3 | ▇▆▁▁▁ |
| BsmtHalfBath  |         2 |          1.00 |      0.06 |     0.25 |     0 |      0.0 |      0.0 |      0.0 |      2 | ▇▁▁▁▁ |
| FullBath      |         0 |          1.00 |      1.57 |     0.55 |     0 |      1.0 |      2.0 |      2.0 |      4 | ▁▇▇▁▁ |
| HalfBath      |         0 |          1.00 |      0.38 |     0.50 |     0 |      0.0 |      0.0 |      1.0 |      2 | ▇▁▅▁▁ |
| BedroomAbvGr  |         0 |          1.00 |      2.86 |     0.82 |     0 |      2.0 |      3.0 |      3.0 |      8 | ▁▇▂▁▁ |
| KitchenAbvGr  |         0 |          1.00 |      1.04 |     0.21 |     0 |      1.0 |      1.0 |      1.0 |      3 | ▁▇▁▁▁ |
| TotRmsAbvGrd  |         0 |          1.00 |      6.45 |     1.57 |     2 |      5.0 |      6.0 |      7.0 |     15 | ▁▇▂▁▁ |
| Fireplaces    |         0 |          1.00 |      0.60 |     0.65 |     0 |      0.0 |      1.0 |      1.0 |      4 | ▇▇▁▁▁ |
| GarageYrBlt   |       159 |          0.95 |   1978.11 |    25.57 |  1895 |   1960.0 |   1979.0 |   2002.0 |   2207 | ▂▇▁▁▁ |
| GarageCars    |         1 |          1.00 |      1.77 |     0.76 |     0 |      1.0 |      2.0 |      2.0 |      5 | ▅▇▂▁▁ |
| GarageArea    |         1 |          1.00 |    472.87 |   215.39 |     0 |    320.0 |    480.0 |    576.0 |   1488 | ▃▇▃▁▁ |
| WoodDeckSF    |         0 |          1.00 |     93.71 |   126.53 |     0 |      0.0 |      0.0 |    168.0 |   1424 | ▇▁▁▁▁ |
| OpenPorchSF   |         0 |          1.00 |     47.49 |    67.58 |     0 |      0.0 |     26.0 |     70.0 |    742 | ▇▁▁▁▁ |
| EnclosedPorch |         0 |          1.00 |     23.10 |    64.24 |     0 |      0.0 |      0.0 |      0.0 |   1012 | ▇▁▁▁▁ |
| 3SsnPorch     |         0 |          1.00 |      2.60 |    25.19 |     0 |      0.0 |      0.0 |      0.0 |    508 | ▇▁▁▁▁ |
| ScreenPorch   |         0 |          1.00 |     16.06 |    56.18 |     0 |      0.0 |      0.0 |      0.0 |    576 | ▇▁▁▁▁ |
| PoolArea      |         0 |          1.00 |      2.25 |    35.66 |     0 |      0.0 |      0.0 |      0.0 |    800 | ▇▁▁▁▁ |
| MiscVal       |         0 |          1.00 |     50.83 |   567.40 |     0 |      0.0 |      0.0 |      0.0 |  17000 | ▇▁▁▁▁ |
| MoSold        |         0 |          1.00 |      6.21 |     2.71 |     1 |      4.0 |      6.0 |      8.0 |     12 | ▅▆▇▃▃ |
| YrSold        |         0 |          1.00 |   2007.79 |     1.31 |  2006 |   2007.0 |   2008.0 |   2009.0 |   2010 | ▇▇▇▇▃ |
| SalePrice     |      1459 |          0.50 | 180921.20 | 79442.50 | 34900 | 129975.0 | 163000.0 | 214000.0 | 755000 | ▇▅▁▁▁ |

-   **PoolQC, Fence, MiscFeature, Alley, SaleType** 변수는 Missing Rate가 높은 편이므로 분석에서 제외하겠습니다.
-   **Id** 역시 분석 데이터로는 불필요 하므로 제거하겠습니다.
-   그 외 NA가 있는 변수들은 나중에 Imputation 해보는 것으로 생각해보겠습니다.

``` r
all_origin %>% 
  select(-c(PoolQC, Fence, MiscFeature, Alley, SaleType, Id)) %>% 
  mutate_if(is.character, as.factor) %>% # Character -> Factor
  mutate(
    MSSubClass = as.factor(MSSubClass), # MSSubClass는 범주형 변수이므로 factor로 변환
    OverallQual = factor(OverallQual, order = T, levels = c(1,2,3,4,5,6,7,8,9,10)),
    OverallCond = factor(OverallCond, order = T, levels = c(1,2,3,4,5,6,7,8,9,10)),
    )-> all
```

-   모든 character는 factor로 변환합니다.
-   일부 ordinal variable들이 있는데, 이게 num variable로 분류되는 걸 막기 위해 따로 처리를 하였습니다.
    -   이게 반드시 필요한 과정인지는 저도 의문입니다. 어차피 num으로 분류되어도 차후에 모델에는 큰 영향도가 없을 것처럼 느껴지기 때문입니다.

``` r
all %>% skim()
```

|                                                  |            |
|:-------------------------------------------------|:-----------|
| Name                                             | Piped data |
| Number of rows                                   | 2919       |
| Number of columns                                | 75         |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |            |
| Column type frequency:                           |            |
| factor                                           | 41         |
| numeric                                          | 34         |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |            |
| Group variables                                  | None       |

Data summary

**Variable type: factor**

| skim_variable | n_missing | complete_rate | ordered | n_unique | top_counts                              |
|:--------------|----------:|--------------:|:--------|---------:|:----------------------------------------|
| MSSubClass    |         0 |          1.00 | FALSE   |       16 | 20: 1079, 60: 575, 50: 287, 120: 182    |
| MSZoning      |         4 |          1.00 | FALSE   |        5 | RL: 2265, RM: 460, FV: 139, RH: 26      |
| Street        |         0 |          1.00 | FALSE   |        2 | Pav: 2907, Grv: 12                      |
| LotShape      |         0 |          1.00 | FALSE   |        4 | Reg: 1859, IR1: 968, IR2: 76, IR3: 16   |
| LandContour   |         0 |          1.00 | FALSE   |        4 | Lvl: 2622, HLS: 120, Bnk: 117, Low: 60  |
| Utilities     |         2 |          1.00 | FALSE   |        2 | All: 2916, NoS: 1                       |
| LotConfig     |         0 |          1.00 | FALSE   |        5 | Ins: 2133, Cor: 511, Cul: 176, FR2: 85  |
| LandSlope     |         0 |          1.00 | FALSE   |        3 | Gtl: 2778, Mod: 125, Sev: 16            |
| Neighborhood  |         0 |          1.00 | FALSE   |       25 | NAm: 443, Col: 267, Old: 239, Edw: 194  |
| Condition1    |         0 |          1.00 | FALSE   |        9 | Nor: 2511, Fee: 164, Art: 92, RRA: 50   |
| Condition2    |         0 |          1.00 | FALSE   |        8 | Nor: 2889, Fee: 13, Art: 5, Pos: 4      |
| BldgType      |         0 |          1.00 | FALSE   |        5 | 1Fa: 2425, Twn: 227, Dup: 109, Twn: 96  |
| HouseStyle    |         0 |          1.00 | FALSE   |        8 | 1St: 1471, 2St: 872, 1.5: 314, SLv: 128 |
| OverallQual   |         0 |          1.00 | TRUE    |       10 | 5: 825, 6: 731, 7: 600, 8: 342          |
| OverallCond   |         0 |          1.00 | TRUE    |        9 | 5: 1645, 6: 531, 7: 390, 8: 144         |
| RoofStyle     |         0 |          1.00 | FALSE   |        6 | Gab: 2310, Hip: 551, Gam: 22, Fla: 20   |
| RoofMatl      |         0 |          1.00 | FALSE   |        8 | Com: 2876, Tar: 23, WdS: 9, WdS: 7      |
| Exterior1st   |         1 |          1.00 | FALSE   |       15 | Vin: 1025, Met: 450, HdB: 442, Wd : 411 |
| Exterior2nd   |         1 |          1.00 | FALSE   |       16 | Vin: 1014, Met: 447, HdB: 406, Wd : 391 |
| MasVnrType    |        24 |          0.99 | FALSE   |        4 | Non: 1742, Brk: 879, Sto: 249, Brk: 25  |
| ExterQual     |         0 |          1.00 | FALSE   |        4 | TA: 1798, Gd: 979, Ex: 107, Fa: 35      |
| ExterCond     |         0 |          1.00 | FALSE   |        5 | TA: 2538, Gd: 299, Fa: 67, Ex: 12       |
| Foundation    |         0 |          1.00 | FALSE   |        6 | PCo: 1308, CBl: 1235, Brk: 311, Sla: 49 |
| BsmtQual      |        81 |          0.97 | FALSE   |        4 | TA: 1283, Gd: 1209, Ex: 258, Fa: 88     |
| BsmtCond      |        82 |          0.97 | FALSE   |        4 | TA: 2606, Gd: 122, Fa: 104, Po: 5       |
| BsmtExposure  |        82 |          0.97 | FALSE   |        4 | No: 1904, Av: 418, Gd: 276, Mn: 239     |
| BsmtFinType1  |        79 |          0.97 | FALSE   |        6 | Unf: 851, GLQ: 849, ALQ: 429, Rec: 288  |
| BsmtFinType2  |        80 |          0.97 | FALSE   |        6 | Unf: 2493, Rec: 105, LwQ: 87, BLQ: 68   |
| Heating       |         0 |          1.00 | FALSE   |        6 | Gas: 2874, Gas: 27, Gra: 9, Wal: 6      |
| HeatingQC     |         0 |          1.00 | FALSE   |        5 | Ex: 1493, TA: 857, Gd: 474, Fa: 92      |
| CentralAir    |         0 |          1.00 | FALSE   |        2 | Y: 2723, N: 196                         |
| Electrical    |         1 |          1.00 | FALSE   |        5 | SBr: 2671, Fus: 188, Fus: 50, Fus: 8    |
| KitchenQual   |         1 |          1.00 | FALSE   |        4 | TA: 1492, Gd: 1151, Ex: 205, Fa: 70     |
| Functional    |         2 |          1.00 | FALSE   |        7 | Typ: 2717, Min: 70, Min: 65, Mod: 35    |
| FireplaceQu   |      1420 |          0.51 | FALSE   |        5 | Gd: 744, TA: 592, Fa: 74, Po: 46        |
| GarageType    |       157 |          0.95 | FALSE   |        6 | Att: 1723, Det: 779, Bui: 186, Bas: 36  |
| GarageFinish  |       159 |          0.95 | FALSE   |        3 | Unf: 1230, RFn: 811, Fin: 719           |
| GarageQual    |       159 |          0.95 | FALSE   |        5 | TA: 2604, Fa: 124, Gd: 24, Po: 5        |
| GarageCond    |       159 |          0.95 | FALSE   |        5 | TA: 2654, Fa: 74, Gd: 15, Po: 14        |
| PavedDrive    |         0 |          1.00 | FALSE   |        3 | Y: 2641, N: 216, P: 62                  |
| SaleCondition |         0 |          1.00 | FALSE   |        6 | Nor: 2402, Par: 245, Abn: 190, Fam: 46  |

**Variable type: numeric**

| skim_variable | n_missing | complete_rate |      mean |       sd |    p0 |      p25 |      p50 |      p75 |   p100 | hist  |
|:--------------|----------:|--------------:|----------:|---------:|------:|---------:|---------:|---------:|-------:|:------|
| LotFrontage   |       486 |          0.83 |     69.31 |    23.34 |    21 |     59.0 |     68.0 |     80.0 |    313 | ▇▃▁▁▁ |
| LotArea       |         0 |          1.00 |  10168.11 |  7887.00 |  1300 |   7478.0 |   9453.0 |  11570.0 | 215245 | ▇▁▁▁▁ |
| YearBuilt     |         0 |          1.00 |   1971.31 |    30.29 |  1872 |   1953.5 |   1973.0 |   2001.0 |   2010 | ▁▂▃▆▇ |
| YearRemodAdd  |         0 |          1.00 |   1984.26 |    20.89 |  1950 |   1965.0 |   1993.0 |   2004.0 |   2010 | ▅▂▂▃▇ |
| MasVnrArea    |        23 |          0.99 |    102.20 |   179.33 |     0 |      0.0 |      0.0 |    164.0 |   1600 | ▇▁▁▁▁ |
| BsmtFinSF1    |         1 |          1.00 |    441.42 |   455.61 |     0 |      0.0 |    368.5 |    733.0 |   5644 | ▇▁▁▁▁ |
| BsmtFinSF2    |         1 |          1.00 |     49.58 |   169.21 |     0 |      0.0 |      0.0 |      0.0 |   1526 | ▇▁▁▁▁ |
| BsmtUnfSF     |         1 |          1.00 |    560.77 |   439.54 |     0 |    220.0 |    467.0 |    805.5 |   2336 | ▇▅▂▁▁ |
| TotalBsmtSF   |         1 |          1.00 |   1051.78 |   440.77 |     0 |    793.0 |    989.5 |   1302.0 |   6110 | ▇▃▁▁▁ |
| 1stFlrSF      |         0 |          1.00 |   1159.58 |   392.36 |   334 |    876.0 |   1082.0 |   1387.5 |   5095 | ▇▃▁▁▁ |
| 2ndFlrSF      |         0 |          1.00 |    336.48 |   428.70 |     0 |      0.0 |      0.0 |    704.0 |   2065 | ▇▃▂▁▁ |
| LowQualFinSF  |         0 |          1.00 |      4.69 |    46.40 |     0 |      0.0 |      0.0 |      0.0 |   1064 | ▇▁▁▁▁ |
| GrLivArea     |         0 |          1.00 |   1500.76 |   506.05 |   334 |   1126.0 |   1444.0 |   1743.5 |   5642 | ▇▇▁▁▁ |
| BsmtFullBath  |         2 |          1.00 |      0.43 |     0.52 |     0 |      0.0 |      0.0 |      1.0 |      3 | ▇▆▁▁▁ |
| BsmtHalfBath  |         2 |          1.00 |      0.06 |     0.25 |     0 |      0.0 |      0.0 |      0.0 |      2 | ▇▁▁▁▁ |
| FullBath      |         0 |          1.00 |      1.57 |     0.55 |     0 |      1.0 |      2.0 |      2.0 |      4 | ▁▇▇▁▁ |
| HalfBath      |         0 |          1.00 |      0.38 |     0.50 |     0 |      0.0 |      0.0 |      1.0 |      2 | ▇▁▅▁▁ |
| BedroomAbvGr  |         0 |          1.00 |      2.86 |     0.82 |     0 |      2.0 |      3.0 |      3.0 |      8 | ▁▇▂▁▁ |
| KitchenAbvGr  |         0 |          1.00 |      1.04 |     0.21 |     0 |      1.0 |      1.0 |      1.0 |      3 | ▁▇▁▁▁ |
| TotRmsAbvGrd  |         0 |          1.00 |      6.45 |     1.57 |     2 |      5.0 |      6.0 |      7.0 |     15 | ▁▇▂▁▁ |
| Fireplaces    |         0 |          1.00 |      0.60 |     0.65 |     0 |      0.0 |      1.0 |      1.0 |      4 | ▇▇▁▁▁ |
| GarageYrBlt   |       159 |          0.95 |   1978.11 |    25.57 |  1895 |   1960.0 |   1979.0 |   2002.0 |   2207 | ▂▇▁▁▁ |
| GarageCars    |         1 |          1.00 |      1.77 |     0.76 |     0 |      1.0 |      2.0 |      2.0 |      5 | ▅▇▂▁▁ |
| GarageArea    |         1 |          1.00 |    472.87 |   215.39 |     0 |    320.0 |    480.0 |    576.0 |   1488 | ▃▇▃▁▁ |
| WoodDeckSF    |         0 |          1.00 |     93.71 |   126.53 |     0 |      0.0 |      0.0 |    168.0 |   1424 | ▇▁▁▁▁ |
| OpenPorchSF   |         0 |          1.00 |     47.49 |    67.58 |     0 |      0.0 |     26.0 |     70.0 |    742 | ▇▁▁▁▁ |
| EnclosedPorch |         0 |          1.00 |     23.10 |    64.24 |     0 |      0.0 |      0.0 |      0.0 |   1012 | ▇▁▁▁▁ |
| 3SsnPorch     |         0 |          1.00 |      2.60 |    25.19 |     0 |      0.0 |      0.0 |      0.0 |    508 | ▇▁▁▁▁ |
| ScreenPorch   |         0 |          1.00 |     16.06 |    56.18 |     0 |      0.0 |      0.0 |      0.0 |    576 | ▇▁▁▁▁ |
| PoolArea      |         0 |          1.00 |      2.25 |    35.66 |     0 |      0.0 |      0.0 |      0.0 |    800 | ▇▁▁▁▁ |
| MiscVal       |         0 |          1.00 |     50.83 |   567.40 |     0 |      0.0 |      0.0 |      0.0 |  17000 | ▇▁▁▁▁ |
| MoSold        |         0 |          1.00 |      6.21 |     2.71 |     1 |      4.0 |      6.0 |      8.0 |     12 | ▅▆▇▃▃ |
| YrSold        |         0 |          1.00 |   2007.79 |     1.31 |  2006 |   2007.0 |   2008.0 |   2009.0 |   2010 | ▇▇▇▇▃ |
| SalePrice     |      1459 |          0.50 | 180921.20 | 79442.50 | 34900 | 129975.0 | 163000.0 | 214000.0 | 755000 | ▇▅▁▁▁ |

## EDA

------------------------------------------------------------------------

이터 EDA를 시작하는 일은 막막합니다. 특히 그 분야에 문외한이라면 더더욱이나 무엇을 시각화해서 어떤 insight를 끌어낼지 더욱 어려운 것 같습니다. 그래서 저는 먼저 IOWA의 Ames City의 특성을 먼저 알아보려고 했습니다.

#### 1. Ames City에 대한 사전정보

Ames City는 대학도시로 유명합니다. 아이오와 주립대가 위치해 있고, 6.5만명 정도가 거주하는 것으로 집계되어있습니다. 더불어서 기후는 겨울엔 한국 이상으로 춥고, 여름에는 한국만큼은 아니지만 꽤 더운 기후입니다. [위키링크](https://ko.wikipedia.org/wiki/%EC%97%90%EC%9E%84%EC%8A%A4_(%EC%95%84%EC%9D%B4%EC%98%A4%EC%99%80%EC%A3%BC))    

대학이 있으니까, 아마 대학가 주변은 조금 더 비싼 가격에 거래되고 있지 않을까 생각했습니다. 그래서 **Neighborhood** 변수로 잡고 집값을 보기 위해 EDA를 시작했습니다.

<br>

#### 2. 대학가 근처라면 더 비쌀까?

집 가격의 분포를 먼저 보고 가시겠습니다.

``` r
all[1:1459, ] %>% 
  ggplot(aes(x = SalePrice))+
  geom_histogram(fill = 'purple')+ 
  scale_x_continuous(labels = comma)+
  geom_vline(aes(xintercept = mean(SalePrice)), lty = 2)
```

    ## `stat_bin()` using `bins = 30`. Pick better value with
    ## `binwidth`.

![a](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-2-1.png)

-   20만 달러 언저리에서 평균이 형성되어 있군요
-   오른쪽으로 Skewed된 데이터이므로 높은 값의 이상치가 꽤 있는 것으로 보입니다. 차후 분석에 참고하겠습니다.

<br>

##### 2-1. 지역에 따른 집가격

``` r
all[1:1459, ] %>% 
  ggplot(aes(x = Neighborhood, y = SalePrice, fill = Neighborhood))+
  geom_boxplot(position = 'dodge')+
  coord_flip() +
  scale_y_continuous(label = comma)
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-3-1.png)

-   **SWISU**는 아이오와 주립대의 남서쪽을 말합니다. 오히려 낮은 가격대이군요
-   그외 지역을 살펴봤을 때도, 딱히 대학교 주변이어서 더 좋은 가격을 형성하고 있지는 않았습니다.
-   만약 지역적인 요소로 좀더 특징을 나눌 수 있다면, 그 특징을 Flag로 사용하여 새로운 변수로 만들면 좋을 것 같습니다.

<br>

##### 2-2. 지역에 따른 방면적

대학가라면 보통 한국에서는 원룸촌이나 고시촌을 떠올리는 것 같습니다. 분명히 집의 전체 넓이와 가격은 상관관계가 있을 겁니다. 그럼 지역별로도 차이가 분명 있을까요?

``` r
all[1:1459, ] %>% 
  mutate(total_feet = `1stFlrSF` + `2ndFlrSF`) %>% 
  ggplot(aes(x = Neighborhood, y = total_feet, fill = Neighborhood))+
  geom_boxplot(position = 'dodge')+
  coord_flip() +
  scale_y_continuous(label = comma)
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-4-1.png)

-   **NoRidge**(Northridge)지역이 평균적으로 높은 넓이의 집을 가진 것
    같습니다. 그래서인지 위에서 확인했듯 가격도 높은 편인 것 같네요.
    아마 Ames City 내에서도 부촌이지 않을까요?

<br>

##### 2-3. 지역에 따른 방면적2

한국에서는 평단가라는 게 있죠? 여기서도 평단가를 한 번 계산해봤습니다.

``` r
all[1:1459, ] %>% 
  mutate(total_feet = `1stFlrSF` + `2ndFlrSF`,
         Price_per_feet = SalePrice/total_feet) %>% 
  ggplot(aes(x = reorder(Neighborhood,Price_per_feet,median ), y = Price_per_feet, fill = Neighborhood))+
  geom_boxplot(position = 'dodge')+
  coord_flip()
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-5-1.png)

-   1, 2층 면적을 모두 더한 걸 가격으로 나눠봤습니다.
-   Northridge Heights가 가장 높은 가격대를 형성하네요. Northridge가 포함된 지역은 비교적 인프라가 잘 되어있는 동네라고 추측해볼 수 있겠습니다.
-   대학 주변(SWISU)은 오히려 단위면적당 가격이 낮은편인 것 같습니다.
-   그렇다면 오히려 대학가에 가까울수록 Sales Price가 낮을 것이라는 예상이 되는데 이유가 있을까요?
    -   <u>아마 집 가격은 대학교 위치와 상관이 없는지도 모릅니다. 한국에서도 오히려 고등학교 학군이 좋다고 소문이 날 수록 집값이 높은 경향이 있습니다. 여기도 비슷하게 오히려 초,중,고가 몰려있는 곳이 집값이 비싸지 않을까요? 그런 데이터를 지금 데이터 셋에서 찾을 수 있다면 좋을 것 같네요.</u>

<br>

#### 3. 집의 여러 시설은 가격에 영향을 어떻게 미칠까?

미국 주택은 차고, 수영장, 덱, 포치 등 다양한 add-on 요소들이 있습니다. 단순히 방 몇 개, 화장실 몇 개 라는 것 보다 고려요소가 더 많아보입니다.

##### 3-1. 상관분석

상관분석을 통해서 여러 요소와 가격과의 관계를 잘 살펴볼 수 있을 것 같습니다.

``` r
library(PerformanceAnalytics)
num_var <- c('LotArea', 'LotFrontage', 'YearBuilt', 'YearRemodAdd',
             'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             '1stFlrSF','2ndFlrSF','LowQualFinSF', 'GrLivArea', 'FullBath',
             'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt',
             'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','OpenPorchSF','EnclosedPorch',
             '3SsnPorch','ScreenPorch','MiscVal','SalePrice')
```

-   `PerformanceAnalytics` 패키지는 상관분석을 위해서 도입한 패키지입니다.
-   num_var는 number type의 변수를 모아놓은 것입니다.

``` r
all[1:1459, c('TotalBsmtSF','GarageArea','YearBuilt','WoodDeckSF','OpenPorchSF', 'SalePrice')] %>% 
  chart.Correlation(histogram = T)
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-7-1.png)

-   지하실, 차고, 건축년도, 덱, 포치와 가격과 유의마한 양의 상관관계를 봅니다.
-   대체로 모두 양의 상관관계를 가지고 있으며 P-value가 높은 것으로 나옵니다.
    -   여기서 p-value가 높다는 것은 Linear Regression의 slope를 의미하는 것인가요?
-   여기서 확인할 수 있는 것은 다양한 요소들이 집값에 분명히 영향을 주고 있다는 점이네요

<br>

##### 3-2. 부대시설 세부

수영장을 한 번 살펴보겠습니다.

``` r
all[1:1459, ] %>% 
  ggplot(aes(x = PoolArea, y = SalePrice))+
  geom_point() +
  geom_smooth(method = 'lm', formula = y ~ x)
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-8-1.png)

-   일단 데이터가 없는지, 아니면 실제로 수영장이 없는 건지 모르겠지만 분석용으로는 적당해 보이지 않습니다.
-   또는 수영장을 가지는 것이 해당 기후에서는 관리비용이 더 많이 들어서일 수도 있을 것 같네요.

<br>

Deck도 한 번 살펴볼게요.

``` r
all[1:1459, ] %>% 
  ggplot(aes(x = WoodDeckSF, y = SalePrice))+
  geom_point() +
  geom_smooth(method = 'lm', formula = y ~ x)
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-9-1.png)

-   **WoodDeck**은 상관분석에서도 봤지만 높은 상관관계를 보이는 듯합니다.
-   하지만 Deck이 없는 집도 상당히 있네요. 이런 경우에는 어떻게 가격 분석을 해야 할까요?

<br>

##### 3-3. 집 면적과 가격

그러면 집 면적과 가격은 어떨까요?

``` r
all[1:1459, c('1stFlrSF','2ndFlrSF','SalePrice')] %>% 
  chart.Correlation(histogram = T)
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-10-1.png)

-   당연히 상관관계가 있군요. 하지만 1층의 넓이가 가격에 더 높은 영향을 주는 것으로 보입니다.
-   1층 넓이와 2층 넓이는 음의 관계에 있군요. 왜일까요?

<br>

#### 4. 리모델링 여부는 집 가격에 영향이 있는가?

YearBuilt 데이터를 보면 1900년 이전에 지어진 집도 있습니다. 한국에서는 아파트를 50년이상 사용하면 이미 철거를 논의하고 있겠지만, 미국 주택은 오래되어도 리모델링이 잘 된다면 여전히 좋은 가격에 팔린다고 하네요. 그래서 리모델 여부와 가격을 한 번 알아보겠습니다.

``` r
all[1:1459, ] %>% 
  mutate(RemodelFlag = as.factor(ifelse(YearBuilt == YearRemodAdd, 'NoRemodel','Remodel')),
         YearBuilt_Class = case_when(
           YearBuilt < 1950 ~ "~1950",
           YearBuilt < 1980 ~ "1950~1980",
           YearBuilt < 2000 ~ "1980~2000",
           T ~ "2000~"
         )) %>%
  ggplot(aes(x = YearBuilt_Class, y = SalePrice, fill = RemodelFlag))+
  geom_boxplot()
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-11-1.png)

-   20년에서 30년을 기준으로 임의로 년도를 분할했습니다.
-   그래도 년도별로 대체로 리모델링 여부가 차이를 보이네요. 그리고 1950년대 이전의 집은 모두 리모델링을 한 것을 알 수 있습니다.
-   그리고 리모델링 한 집이 보통 위로 꼬리가 깁니다. 아마 협상의 여지가 있고, 보통은 높은 가격을 받았다는 의미인 것 같네요.
-   나중에 Feature Engineering에서 Flag로 만들면 분석에 더 도움이 될 것 같습니다.

<br>

``` r
all[1:1459, ] %>% 
  mutate(RemodelFlag = as.factor(ifelse(YearBuilt == YearRemodAdd, 'NoRemodel','Remodel')),
         YearBuilt_Class = case_when(
           YearBuilt < 1950 ~ "~1950",
           YearBuilt < 1980 ~ "1950~1980",
           YearBuilt < 2000 ~ "1980~2000",
           T ~ "2000~"
         )) %>%
  ggplot(aes(x = MSSubClass, y = SalePrice, fill = RemodelFlag))+
  geom_boxplot()
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-12-1.png)

-   집의 종류 별로도 가격을 세워봤습니다.
-   몇몇 종류의 집은 오히려 리모델링을 하지 않았을 때 더 좋은 가격을 보이는 집들도 있네요. 무슨 이유일까요?

<br>

#### 5. 판매 시점에 따른 가격 변화

2008년은 금융위기가 왔었고 집값이 많이 떨어졌습니다. 그래서 2008년에는 집값이 많이 떨어졌는지, 그리고 판매 시점별로 가격의 변화가 있는지 확인해보겠습니다.

``` r
all[1:1459, ] %>% 
  group_by(YrSold) %>% 
  summarise(SalePrice = mean(SalePrice)) %>% 
  ggplot(aes(x = YrSold, y = SalePrice))+
  geom_col(position = 'dodge')+
  scale_y_continuous(labels = comma)
```

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-13-1.png)

-   2008년 금융위기에 한 번 집값이 내렸을 것으로 예상했으나 의외로 많이 내리지는 않았군요.
-   그래도 오르기만 하는 집값의 특성 상 한 번 내린 년도가 있다는 것은 분석 관점에서 의의가 있어 보이네요.

그럼 월별로도 변화가 있는지 살펴볼까요?

``` r
all[1:1459, ] %>% 
  group_by(YrSold, MoSold) %>% 
  summarise(SalePrice = mean(SalePrice)) %>% 
  ggplot(aes(x = MoSold, y = SalePrice))+
  geom_col(position = 'dodge')+
  geom_line(col = 'red', size = 1)+
  scale_y_continuous(labels = comma)+
  facet_grid(YrSold ~ .)
```

    ## `summarise()` has grouped output by 'YrSold'. You can override using the `.groups` argument.

![](https://sangminje.github.io/assets/img/House_Pred_files/figure-markdown_github/unnamed-chunk-14-1.png)

-   딱히 월별로도 특징적인 변화가 있는 것 같지 않습니다. <br>

## 결론

------------------------------------------------------------------------

제가 머신러닝을 위한 EDA를 수행하면서 느낀 점은 아래와 같습니다.

**1. 변수가 많을수록 EDA의 경우의 수가 많아서 힘들다**

변수가 많으면 그만큼 고려해야 할 경우의 수가 많아서 어려운 것 같습니다. 특히 80개 정도 되는 변수 속에서 각 변수의 특성을 파악하고 EDA의 아이디어를 잡는 데 꽤 많은 시각이 걸린 것 같네요. 이럴 때 사용할 수 있는 정형화된 일련의 접근 방식이 있다면 좋을 것 같습니다. 보통은 상관계수로 많이 접근하는 것 같아요. 여러분의 의견은 어떤가요?

**2. 적절한 미국 부동산 지식이 EDA의 접근 가능성을 높임**

저는 미국 부동산 유튜브를 조금 참조했습니다. 그 결과 오래된 집이어도 리모델링이 잘 되어 있다면 높은 가격을 받을 수 있다는 정보를 알게 되었네요. 이게 도메인의 힘인 것 같습니다. 데이터 분석을 위해서는 유효한 질문이 필요하고, 그 질문은 사실 분석가의 아이디어 속에 있는 것 같네요.

**3. EDA의 결과는 Feature Engineering과 이어져야 큰 의미가 있는걸까?**

결국 저희는 머신러닝을 통해서 모델링을 할 예정이잖아요? 시각화로 데이터를 보는 것 자체가 중요하다기 보다는 이를 Feature Engineering으로 새로운 변수를 만들어 모델링에 심을 수 있는 게 더욱 중요하다는 생각이 들었습니다. 그런 의미에서 머신러닝에서의 EDA는 정말 Feature Engineering의 전단계라는 생각이 들었네요.

여기까지 제가 정리한 자료입니다. 앞으로 머신러닝 모델을 진행하면서 더 많이 배우기를 기대해봅니다 :)
