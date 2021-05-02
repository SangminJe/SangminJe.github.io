---  
layout: post  
title: "Tidymodels로 시작하는 머신러닝 (3)"
subtitle: "3. 성능평가(Evaluation)"  
categories: Data  
tags: Data ML tidymodels fold tidymodels-recipe 타이디모델 R-machine-learning R Cross-Validation 교차검증
comments: true  
---  

## 개요

> 1.  모델 평가의 기준과 교차 검증에 대해서 배웁니다.
> 2.  본 문서는 tidymodels [공식 영문서](https://www.tidymodels.org/start/resampling/)를 참고로 만들었습니다.

- 이전 포스트
  - [1. Build a Model](https://sangminje.github.io/_posts/2021-04-22-Data-ML-tidy-build-a-model.md)
  - [2. Recipe](https://sangminje.github.io/_posts/2021-04-30-Data-ML-tidy-Recipe.md)

  
- 목차
  - [1. Cell Image Data](#cell-image-data)
  - [2. Data Splitting](#data-splitting)
  - [3. Modeling](#modeling)
  - [4. Performance 측정](#performance-측정)
  - [5. 교차검증(Cross Validation)](#교차검증cross-validation)
  - [6. 요약](#요약)



``` r
library(tidymodels) # for the rsample package, along with the rest of tidymodels

## -- Attaching packages -------------------------------------- tidymodels 0.1.2 --

## √ broom     0.7.4      √ recipes   0.1.15
## √ dials     0.0.9      √ rsample   0.0.8 
## √ dplyr     1.0.4      √ tibble    3.0.6 
## √ ggplot2   3.3.3      √ tidyr     1.1.2 
## √ infer     0.5.4      √ tune      0.1.2 
## √ modeldata 0.1.0      √ workflows 0.2.1 
## √ parsnip   0.1.5      √ yardstick 0.0.7 
## √ purrr     0.3.4

## -- Conflicts ----------------------------------------- tidymodels_conflicts() --
## x purrr::discard() masks scales::discard()
## x dplyr::filter()  masks stats::filter()
## x dplyr::lag()     masks stats::lag()
## x recipes::step()  masks stats::step()

library(modeldata)  # for the cells data

```

모델을 훈련했다면, 이제 필요한 작업은 훈련한 모델을 평가하는 일입니다. 이전 포스팅에서 잠깐 언급하고 넘어갔던 **Evaluation**을 본격적으로 다루겠습니다.

## Cell Image Data
---

`Cell Image Data`는 `modeldata` 패키지의 데이터로 세포의 분류를 예측하기 위해 만들어진 모델입니다. 
- [Cell Image Data 설명](https://www.tidymodels.org/start/resampling/#data)
- Cell Image가 선명할수록 생물학자들이 분석에 사용하기 용이합니다. 하지만 일부 Cell Data들의 Image가 선명하지 않거나, 서로 뭉쳐있어서 명확하게 세포의 이미지를 분별하기 어려울 때가 있습니다. 
- 이렇게 선명하게 모양이 잘 나온 데이터를 `class` 컬럼에서 **WS(Well-Segmented)**, 선명하지 않고 불명확하게 나온 세포 데이터를 **PS(Poorly-Segmented로 분류합니다.)** 
- 우리가 데이터를 통해 분류할 수 있다면, 큰 DataSet일수록 생물학자들이 효율적으로 활용할 수 있을 겁니다.

``` r 
data(cells, package = 'modeldata')
cells

## # A tibble: 2,019 x 58
##    case  class angle_ch_1 area_ch_1 avg_inten_ch_1 avg_inten_ch_2 avg_inten_ch_3
##    <fct> <fct>      <dbl>     <int>          <dbl>          <dbl>          <dbl>
##  1 Test  PS        143.         185           15.7           4.95           9.55
##  2 Train PS        134.         819           31.9         207.            69.9 
##  3 Train WS        107.         431           28.0         116.            63.9 
##  4 Train PS         69.2        298           19.5         102.            28.2 
##  5 Test  PS          2.89       285           24.3         112.            20.5 
##  6 Test  WS         40.7        172          326.          654.           129.  
##  7 Test  WS        174.         177          260.          596.           124.  
##  8 Test  PS        180.         251           18.3           5.73          17.2 
##  9 Test  WS         18.9        495           16.1          89.5           13.7 
## 10 Test  WS        153.         384           17.7          89.9           20.4 
## # ... with 2,009 more rows, and 51 more variables: avg_inten_ch_4 <dbl>,
## #   convex_hull_area_ratio_ch_1 <dbl>, convex_hull_perim_ratio_ch_1 <dbl>,
## #   diff_inten_density_ch_1 <dbl>, diff_inten_density_ch_3 <dbl>,
## #   diff_inten_density_ch_4 <dbl>, entropy_inten_ch_1 <dbl>,
## #   entropy_inten_ch_3 <dbl>, entropy_inten_ch_4 <dbl>,
## #   eq_circ_diam_ch_1 <dbl>, eq_ellipse_lwr_ch_1 <dbl>,
## #   eq_ellipse_oblate_vol_ch_1 <dbl>, eq_ellipse_prolate_vol_ch_1 <dbl>,
## #   eq_sphere_area_ch_1 <dbl>, eq_sphere_vol_ch_1 <dbl>,
## #   fiber_align_2_ch_3 <dbl>, fiber_align_2_ch_4 <dbl>,
## #   fiber_length_ch_1 <dbl>, fiber_width_ch_1 <dbl>, inten_cooc_asm_ch_3 <dbl>,
## #   inten_cooc_asm_ch_4 <dbl>, inten_cooc_contrast_ch_3 <dbl>,
## #   inten_cooc_contrast_ch_4 <dbl>, inten_cooc_entropy_ch_3 <dbl>,
## #   inten_cooc_entropy_ch_4 <dbl>, inten_cooc_max_ch_3 <dbl>,
## #   inten_cooc_max_ch_4 <dbl>, kurt_inten_ch_1 <dbl>, kurt_inten_ch_3 <dbl>,
## #   kurt_inten_ch_4 <dbl>, length_ch_1 <dbl>, neighbor_avg_dist_ch_1 <dbl>,
## #   neighbor_min_dist_ch_1 <dbl>, neighbor_var_dist_ch_1 <dbl>,
## #   perim_ch_1 <dbl>, shape_bfr_ch_1 <dbl>, shape_lwr_ch_1 <dbl>,
## #   shape_p_2_a_ch_1 <dbl>, skew_inten_ch_1 <dbl>, skew_inten_ch_3 <dbl>,
## #   skew_inten_ch_4 <dbl>, spot_fiber_count_ch_3 <int>,
## #   spot_fiber_count_ch_4 <dbl>, total_inten_ch_1 <int>,
## #   total_inten_ch_2 <dbl>, total_inten_ch_3 <int>, total_inten_ch_4 <int>,
## #   var_inten_ch_1 <dbl>, var_inten_ch_3 <dbl>, var_inten_ch_4 <dbl>,
## #   width_ch_1 <dbl>
```
-   2019개의 세포와 58개의 컬럼이 존재합니다.
-   **Outcome Predictor**는 `class` 변수입니다.

``` r
cells %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

## # A tibble: 2 x 3
##   class     n  prop
## * <fct> <int> <dbl>
## 1 PS     1300 0.644
## 2 WS      719 0.356

cells

## # A tibble: 2,019 x 58
##    case  class angle_ch_1 area_ch_1 avg_inten_ch_1 avg_inten_ch_2 avg_inten_ch_3
##    <fct> <fct>      <dbl>     <int>          <dbl>          <dbl>          <dbl>
##  1 Test  PS        143.         185           15.7           4.95           9.55
##  2 Train PS        134.         819           31.9         207.            69.9 
##  3 Train WS        107.         431           28.0         116.            63.9 
##  4 Train PS         69.2        298           19.5         102.            28.2 
##  5 Test  PS          2.89       285           24.3         112.            20.5 
##  6 Test  WS         40.7        172          326.          654.           129.  
##  7 Test  WS        174.         177          260.          596.           124.  
##  8 Test  PS        180.         251           18.3           5.73          17.2 
##  9 Test  WS         18.9        495           16.1          89.5           13.7 
## 10 Test  WS        153.         384           17.7          89.9           20.4 
## # ... with 2,009 more rows, and 51 more variables: avg_inten_ch_4 <dbl>,
## #   convex_hull_area_ratio_ch_1 <dbl>, convex_hull_perim_ratio_ch_1 <dbl>,
## #   diff_inten_density_ch_1 <dbl>, diff_inten_density_ch_3 <dbl>,
## #   diff_inten_density_ch_4 <dbl>, entropy_inten_ch_1 <dbl>,
## #   entropy_inten_ch_3 <dbl>, entropy_inten_ch_4 <dbl>,
## #   eq_circ_diam_ch_1 <dbl>, eq_ellipse_lwr_ch_1 <dbl>,
## #   eq_ellipse_oblate_vol_ch_1 <dbl>, eq_ellipse_prolate_vol_ch_1 <dbl>,
## #   eq_sphere_area_ch_1 <dbl>, eq_sphere_vol_ch_1 <dbl>,
## #   fiber_align_2_ch_3 <dbl>, fiber_align_2_ch_4 <dbl>,
## #   fiber_length_ch_1 <dbl>, fiber_width_ch_1 <dbl>, inten_cooc_asm_ch_3 <dbl>,
## #   inten_cooc_asm_ch_4 <dbl>, inten_cooc_contrast_ch_3 <dbl>,
## #   inten_cooc_contrast_ch_4 <dbl>, inten_cooc_entropy_ch_3 <dbl>,
## #   inten_cooc_entropy_ch_4 <dbl>, inten_cooc_max_ch_3 <dbl>,
## #   inten_cooc_max_ch_4 <dbl>, kurt_inten_ch_1 <dbl>, kurt_inten_ch_3 <dbl>,
## #   kurt_inten_ch_4 <dbl>, length_ch_1 <dbl>, neighbor_avg_dist_ch_1 <dbl>,
## #   neighbor_min_dist_ch_1 <dbl>, neighbor_var_dist_ch_1 <dbl>,
## #   perim_ch_1 <dbl>, shape_bfr_ch_1 <dbl>, shape_lwr_ch_1 <dbl>,
## #   shape_p_2_a_ch_1 <dbl>, skew_inten_ch_1 <dbl>, skew_inten_ch_3 <dbl>,
## #   skew_inten_ch_4 <dbl>, spot_fiber_count_ch_3 <int>,
## #   spot_fiber_count_ch_4 <dbl>, total_inten_ch_1 <int>,
## #   total_inten_ch_2 <dbl>, total_inten_ch_3 <int>, total_inten_ch_4 <int>,
## #   var_inten_ch_1 <dbl>, var_inten_ch_3 <dbl>, var_inten_ch_4 <dbl>,
## #   width_ch_1 <dbl>
```
-   **PS**가 **64%** 정도로 잘 분류되지 못한 데이터가 많은 것으로 보여지네요.

## Data Splitting
---

데이터를 나누겠습니다. 이전에 `initial split`을 사용해서 Random Sampling을 수행하여 **Train Data**와 **Test Data**를 나누셨던 것을 기억하실 겁니다. 그런데 Random Sampling을 하면 위에서 계산했던 종속변수의 결과가 어떻게 되나요? 아마 **64%**는 아닐겁니다. 이럴 때 사용하는 방법이
[**층화추출법**](https://ko.wikipedia.org/wiki/%EC%B8%B5%ED%99%94%EC%B6%94%EC%B6%9C%EB%B2%95)입니다. **층화추출법**은 표본조사방법에서 실제적으로 가장 많이 이용되는 추출법이며, 단순임의추출(Random Sampling)의 단점을 보완하려는 목적 및 기타 다른 추출법과 비교하여 적은 비용으로 추정치를 다소 정확히 구할 수 있다는 특징이 있습니다.

#### 층화추출
``` r 
set.seed(123) # 차후 동일한 결과확인을 위해 Seed를 정해줍니다.

cells_split <- 
  initial_split(cells %>% select(-case),
                strata = class) # 층화 추출법

cell_train <- training(cells_split)
cell_test <- testing(cells_split)
```
- `strata = class`를 통해서 층화추출법을 이용한 데이터 분할을 수행할 수 있습니다.

``` r
#train
cell_train %>%
  count(class) %>%
  mutate(prop = n / sum(n))

## # A tibble: 2 x 3
##   class     n  prop
## * <fct> <int> <dbl>
## 1 PS      975 0.644
## 2 WS      540 0.356

# test
cell_test %>%
  count(class) %>%
  mutate(prop = n / sum(n))

## # A tibble: 2 x 3
##   class     n  prop
## * <fct> <int> <dbl>
## 1 PS      325 0.645
## 2 WS      179 0.355
```
- 모두 모집단과 동일한 `class`의 비율을 가지고 있음을 확인할 수 있습니다.

## Modeling
---

**Random Forest**를 활용해서 모델링을 수행하겠습니다.

#### Random Forest의 장점

- 모델에 Pre-Preocessing이 적게 사용됩니다.
- 가장 기본적인 parameter 상태에서도 준수한 결과를 나타냅니다.

#### Parsnip을 이용한 Modeling
``` r 
rf_mod <- 
  rand_forest(mtry = 1000) %>% 
  set_engine('ranger') %>% 
  set_mode('classification')
```
- `parsnip`패키지를 활용하여 Random Forest 모델을 만듭니다.
- [`ranger`](https://cran.r-project.org/web/packages/ranger/ranger.pdf)
- `set_mode`는 Classification으로 설정합니다.

#### Fitting
```r
rf_fit <-
  rf_mod %>%
  fit(class ~ ., data = cell_train)

## Warning: 1000 columns were requested but there were 56 predictors in the data.
## 56 will be used.

rf_fit

## parsnip model object
## 
## Fit time:  8.8s 
## Ranger result
## 
## Call:
##  ranger::ranger(x = maybe_data_frame(x), y = y, mtry = min_cols(~1000,      x), num.threads = 1, verbose = FALSE, seed = sample.int(10^5,      1), probability = TRUE) 
## 
## Type:                             Probability estimation 
## Number of trees:                  500 
## Sample size:                      1515 
## Number of independent variables:  56 
## Mtry:                             56 
## Target node size:                 10 
## Variable importance mode:         none 
## Splitrule:                        gini 
## OOB prediction error (Brier s.):  0.1243567
```
-   앞서 만든 모델을 바탕으로 Fitting을 수행합니다.

## Performance 측정
---

모델을 측정하는 두 가지 기준을 소개합니다.
1. [ROC Curve](https://m.blog.naver.com/PostView.nhn?blogId=sw4r&logNo=221015817276&proxyReferer=https:%2F%2Fwww.google.com%2F)
2. Accuracy

`yardstick`이라는 패키지는 이 두 가지를 모두 계산해줍니다.
```r
rf_testing_pred <- 
  predict(rf_fit, cell_test) %>% 
  bind_cols(predict(rf_fit, cell_test, type = 'prob')) %>% 
  bind_cols(cell_test %>% select(class))
rf_testing_pred

## # A tibble: 504 x 4
##    .pred_class .pred_PS .pred_WS class
##    <fct>          <dbl>    <dbl> <fct>
##  1 WS            0.131    0.869  WS   
##  2 PS            1        0      PS   
##  3 WS            0.392    0.608  WS   
##  4 WS            0.444    0.556  PS   
##  5 PS            0.774    0.226  PS   
##  6 WS            0.0387   0.961  WS   
##  7 PS            0.527    0.473  PS   
##  8 PS            0.955    0.0448 PS   
##  9 PS            1        0      PS   
## 10 WS            0.132    0.868  WS   
## # ... with 494 more rows

rf_testing_pred %>% 
  roc_auc(truth = class, .pred_PS)

## # A tibble: 1 x 3
##   .metric .estimator .estimate
##   <chr>   <chr>          <dbl>
## 1 roc_auc binary         0.894

rf_testing_pred %>% 
  accuracy(truth = class, .pred_class)

## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.821
```
-   Train\_Data로 훈련한 모델을 Test Data에 적용했을 때, 나오는 ROC  AUC의 값과 Accuracy의 값을 보여줍니다.
-   그런데 여기서 잠깐, 항상 Testing Data로 모델의 성능을 검증을 수행할 수 있을까요? Testing Data는 분류 결과가 없을 수도 있는데 말입니다. 그럼 어떻게 데이터의 성능을 검증할 수 있을까요?

## 교차검증(Cross Validation)
---

지금까지는 Training Set에서 Testing Set에 적용해서 모델의 기능을 측정해왔습니다. 하지만 바로 Testin Data로 넘어가기 전에, Trainin Set 안에서 모델이 잘 적용이 되었는지 확인이 먼저 진행될 필요가 있습니다.

![img](https://sangminje.github.io/assets/img/tidymodels/resampling.svg)

그림에서처럼 Train Data를 쪼개서 각각을 다시 Trainin Set과 Testing Set으로 나누는 작업을 합니다. 예를 들어 현재 1505 개의 Traing Data의 Row 개수를 10개의 덩어리로 나눈다고 가정하겠습니다. 그럴 경우 90%의 데이터는 `Analysis` 데이터로 두고, 10%의 데이터를 `Assessment` 데이터로 두어서 모델을 적용하고 결과를 확인합니다. 결과를 낸 후 다른 그룹으로 90%와
10%를 각각 묶어 테스트합니다. 이렇게 10번의 과정을 겨치며 테스트 하는 것을 **교차검증**이라고 합니다.

``` r 
set.seed(345)

folds <- vfold_cv(cell_train, v = 10)
folds

## #  10-fold cross-validation 
## # A tibble: 10 x 2
##    splits             id    
##    <list>             <chr> 
##  1 <split [1.4K/152]> Fold01
##  2 <split [1.4K/152]> Fold02
##  3 <split [1.4K/152]> Fold03
##  4 <split [1.4K/152]> Fold04
##  5 <split [1.4K/152]> Fold05
##  6 <split [1.4K/151]> Fold06
##  7 <split [1.4K/151]> Fold07
##  8 <split [1.4K/151]> Fold08
##  9 <split [1.4K/151]> Fold09
## 10 <split [1.4K/151]> Fold10
```
-   `rsample`패키지에 있는 `vfold_cv`함수를 통해 교차검증이 가능하도록 데이터를 10개로 나눕니다.

``` r 
rf_wf <-  
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_formula(class ~ .)

set.seed(456)

rf_fit_rs <- 
  rf_wf %>% 
  fit_resamples(folds)

## 
## Attaching package: 'rlang'

## The following objects are masked from 'package:purrr':
## 
##     %@%, as_function, flatten, flatten_chr, flatten_dbl, flatten_int,
##     flatten_lgl, flatten_raw, invoke, list_along, modify, prepend,
##     splice

## 
## Attaching package: 'vctrs'

## The following object is masked from 'package:tibble':
## 
##     data_frame

## The following object is masked from 'package:dplyr':
## 
##     data_frame

## ! Fold01: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold02: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold03: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold04: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold05: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold06: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold07: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold08: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold09: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

## ! Fold10: preprocessor 1/1, model 1/1: 1000 columns were requested but there were ...

rf_fit_rs

## Warning: This tuning result has notes. Example notes on model fitting include:
## preprocessor 1/1, model 1/1: 1000 columns were requested but there were 56 predictors in the data. 56 will be used.
## preprocessor 1/1, model 1/1: 1000 columns were requested but there were 56 predictors in the data. 56 will be used.
## preprocessor 1/1, model 1/1: 1000 columns were requested but there were 56 predictors in the data. 56 will be used.

## # Resampling results
## # 10-fold cross-validation 
## # A tibble: 10 x 4
##    splits             id     .metrics         .notes          
##    <list>             <chr>  <list>           <list>          
##  1 <split [1.4K/152]> Fold01 <tibble [2 x 4]> <tibble [1 x 1]>
##  2 <split [1.4K/152]> Fold02 <tibble [2 x 4]> <tibble [1 x 1]>
##  3 <split [1.4K/152]> Fold03 <tibble [2 x 4]> <tibble [1 x 1]>
##  4 <split [1.4K/152]> Fold04 <tibble [2 x 4]> <tibble [1 x 1]>
##  5 <split [1.4K/152]> Fold05 <tibble [2 x 4]> <tibble [1 x 1]>
##  6 <split [1.4K/151]> Fold06 <tibble [2 x 4]> <tibble [1 x 1]>
##  7 <split [1.4K/151]> Fold07 <tibble [2 x 4]> <tibble [1 x 1]>
##  8 <split [1.4K/151]> Fold08 <tibble [2 x 4]> <tibble [1 x 1]>
##  9 <split [1.4K/151]> Fold09 <tibble [2 x 4]> <tibble [1 x 1]>
## 10 <split [1.4K/151]> Fold10 <tibble [2 x 4]> <tibble [1 x 1]>
```
-   **Workflow**에 모델과 식을 끼워넣었습니다.
-   `rf_fit_rs` 안에서 `fir_resamples` 함수를 통해 미리 만들어진 `folds`(교차검증용 데이터셋)을 fitting 시킵니다.

``` r
collect_metrics(rf_fit_rs)

## # A tibble: 2 x 6
##   .metric  .estimator  mean     n std_err .config             
##   <chr>    <chr>      <dbl> <int>   <dbl> <chr>               
## 1 accuracy binary     0.827    10 0.00972 Preprocessor1_Model1
## 2 roc_auc  binary     0.896    10 0.00977 Preprocessor1_Model1
```
-   `collect_metrics`는 최종 결과값만을 출력해주는 함수입니다.
-   반환된 **ROC AUC**값과 **Accuracy**값은 10개의 교차검증한 데이터 통계값의 평균치입니다.

## 요약
---

1. Tidymodels 패키지 내에서 모델의 성능을 평가할 수 있다.
2. 교차 검증을 통해 ROC_AUC와 Accuracy를 구할 수 있다.
3. `workflow`를 사용해서 모델을 Fitting하고 그 결과를 교차검증하는 프로세스가 가능하다. 
