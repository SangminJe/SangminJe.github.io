---  
layout: post  
title: "R에서 꼭 필요한 정규표현식"
subtitle: "stringr을 활용한 정규표현식 이해"  
categories: DATA
tags: Data ML R Regex 정규표현식 stringr R-정규표현식
comments: true  
---  

# 개요

---

> 1. 본 문서는 R에서 사용하는 Regex를 다룹니다.
> 2. 본 문서는 엘리님의 드림코딩 유튜브를 참고하였습니다. - [엘리의 드림코딩 - 정규표현식](https://www.youtube.com/watch?v=t3M6toIflyQ)

# REGEX

---

정규표현식은 늘 R에서 만나면 당황하곤 합니다. 뭔가 외계어 같고 특수문자가 가득해서 위화감을 주기 때문인 것 같습니다. 하지만 알고보면 정규표현식은 어렵지 않게 이해하고 사용할 수 있습니다. 아래 예시문을 가지고  정규표현식 연습링크에서 따라하시면 좋습니다. 그리고 위에 언급드린 엘리님께서 유튜브로 너무 잘 정리해주셨습니다.(감사X100) 해당 유튜브 컨텐츠를 보시고 30분만 따라하시면 금방 감이 오실거예요:)

- [정규표현식 연습링크](https://regexr.com/5mhou)


**예시문**

        Hi there, Nice to meet you. And Hello there and hi.
        I love grey(gray) color not a gry, graay and graaay.
        Ya ya YaYaYa Ya
        
        abcdefghijklmnopqrstuvwxyz
        ABSCEFGHIJKLMNOPQRSTUVWZYZ
        1234567890
        
        .[]{}()\^$|?*+
        
        010-898-0893
        010-405-3412
        02-878-8888
        
        dream.coder.ellie@gmail.com
        hello@daum.net
        hello@daum.co.kr
        
        https://www.youtu.be/-ZClicWm0zM
        https://youtu.be/-ZClicWm0zM
        youtu.be/-ZClicWm0zM



## Groups and ranges

-   **\|** : OR
    -   Hi\|Hello : Hi나 Hello에 매칭되는 문자열을 반환
-   **()** : 그룹
    -   (Hi\|Hello)\|(And) :Hi와 Hello는 Group1에 매칭되며, And는
        Group2에 매칭됨
    -   gr(e\|a)y : grey나 gray 모두 매칭되어 반환
    -   gr(?:e\|a)y : 매칭은 되지만 Group1, Group2는 지정이 되지 않음
-   **\[\]** 괄호 안의 어떤 문자든 반환
    -   gr[abcdef]y / gr[a-f]y : 대괄호 안의 모든 문자열을 반환한다
    -   [a-zA-Z0-9] : a~z 까지의 문자와, A~Z까지의 문자와 0~9가
        들어간 모든 숫자를 반환
-   **^** : NOT
    -   [^a-zA-Z0-9] : 영어와 숫자 빼고 모든 문자를 반환

## Quantifier

-   **?** : 없거나 있거나
    -   gra?y : gray, gry 모두 반환
-   **\*** : 없거나 있거나 많거나
    -   gra\*y : gray, graaay, gry 모두 반환
-   **+** : 한개 이상
    -   gra+y : gray, graaay만 추가
-   **{n}** : n개만 특정해서 반환
    -   gra{2}y : graay만 반환
-   **{min,max}** : min \~ max개 사이의 문자열을 반환
    -   gra{2,3}y : graay, graaay를 반환
    -   gra{2,}y : graay, graaay를 반환 (최소값만 지정하여 2개 이상인
        것들을 반환)

## Boundary type

-   **\\b** : 단어 경계
    -   \\bYa : 단어 앞에서 쓰이는 Ya를 반환함
    -   Ya\\b : 단어 뒤에서만 쓰이는 Ya를 반환
-   **\\B** : 단어 경계의 반대
    -   \\BYa : 단어 앞에서 쓰이는 Ya는 빼고 반환함
    -   Ya\\B : 단어 뒤에서만 쓰이는 Ya는 빼고 반환
-   **^,$** : 문장 경계
    -   ^Ya : 문장에서 시작하는 Ya 반환
    -   Ya$ : 문장에서 끝나는 Ya 반환

## Character Classes

-   **.** : 모든 문자열
    -   . : 모든 문자열을 선택 가능
-   **\\** : Escape
-   **\\d** : 숫자를 모두 반환
-   **\\D** : 숫자를 제외하고 모두 반환
-   **\\w** : 모든 word
-   **\\W** : 모든 문자를 제외하고 반환
-   **\\s** : space 반환
-   **\\S** : space를 제외하고 반환

# Exercise

---


``` r
library(tidyverse)
```

    ## -- Attaching packages ------------------------------------------------------------------------------- tidyverse 1.3.1 --

    ## √ ggplot2 3.3.3     √ purrr   0.3.4
    ## √ tibble  3.1.2     √ dplyr   1.0.6
    ## √ tidyr   1.1.3     √ stringr 1.4.0
    ## √ readr   1.4.0     √ forcats 0.5.1

    ## -- Conflicts ---------------------------------------------------------------------------------- tidyverse_conflicts() --
    ## x dplyr::combine() masks gridExtra::combine()
    ## x dplyr::filter()  masks stats::filter()
    ## x dplyr::lag()     masks stats::lag()

**1.  전화번호만 선택하기**

``` r
sentence <- "Hi there, Nice to meet you. And Hello there and hi. I love grey(gray) color not a gry, graay and graaay. Ya ya YaYaYa Ya .[]{}()^$|?*+ 010-898-0893 010-405-3412 02-878-8888 dream.coder.ellie@gmail.com hello@daum.net hello@daum.co.kr  https://www.youtu.be/-ZClicWm0zM https://youtu.be/-ZClicWm0zM youtu.be/-ZClicWm0zM"

print(sentence)
```

    ## [1] "Hi there, Nice to meet you. And Hello there and hi. I love grey(gray) color not a gry, graay and graaay. Ya ya YaYaYa Ya .[]{}()^$|?*+ 010-898-0893 010-405-3412 02-878-8888 dream.coder.ellie@gmail.com hello@daum.net hello@daum.co.kr  https://www.youtu.be/-ZClicWm0zM https://youtu.be/-ZClicWm0zM youtu.be/-ZClicWm0zM"

``` r
str_extract_all(sentence, "[0-9]+[\\-|\\.][0-9]+[\\-|\\.][0-9]+") %>% unlist()
```

    ## [1] "010-898-0893" "010-405-3412" "02-878-8888"

``` r
str_extract_all(sentence, '\\d{2,3}[.-]\\d{3,4}[.-]\\d{3,4}') %>% unlist()
```

    ## [1] "010-898-0893" "010-405-3412" "02-878-8888"

**2.  이메일 주소 가져오기**

``` r
str_extract_all(sentence,
                "[a-zA-Z0-9._+-]+@[a-zA-Z0-9-_]+\\.[a-zA-Z0-9.]+") %>% 
  unlist()
```

    ## [1] "dream.coder.ellie@gmail.com" "hello@daum.net"              "hello@daum.co.kr"

**3.  인터넷 주소 가져오기**

``` r
# 유튜브 아이디 가져오기
str_extract_all(sentence, "(https?:\\/\\/)?(www.)?youtu.be\\/([a-zA-Z0-9-]+)")
```

    ## [[1]]
    ## [1] "https://www.youtu.be/-ZClicWm0zM" "https://youtu.be/-ZClicWm0zM"     "youtu.be/-ZClicWm0zM"

``` r
# 뒤에 유튜브 ID만 추출
str_match(sentence, "(?:https?:\\/\\/)?(?:www.)?youtu.be\\/([a-zA-Z0-9-]+)") %>% .[2]
```

    ## [1] "-ZClicWm0zM"
