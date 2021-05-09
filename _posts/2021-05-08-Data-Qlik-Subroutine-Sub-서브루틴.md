---  
layout: post  
title: "Qlik - SubRoutine(서브루틴)"
subtitle: "QlikView Sub 함수 설명과 용도"  
categories: Data  
tags: data Qlik qlikview qliksense subroutine sub end-sub
comments: true  
---  

## 개요
> SubRoutine (Sub, End Sub에 대한 설명)

서브루틴은 기본적으로 스크립트를 저장하는 함수라고 생각하면 좋습니다. 하나의 SubRoutine 안에 스크립트를 저장하고 필요할 때마다 Call이라는 함수를 통애 저장한 함수를 불러냅니다.

<br/>

아래 예시를 보겠습니다.


```

// List all QV related files on disk

sub DoDir (Root) // 2. Subroutine 시작

For Each Ext in 'qvw', 'qvo', 'qvs', 'qvt', 'qvd', 'qvc' // 3. 확장명 지정

For Each File in filelist (Root&'\*.' &Ext)


// 4. 테이블 로드

LOAD
'$(File)' as Name,
FileSize( '$(File)' ) as Size,
FileTime( '$(File)' ) as FileTime
autogenerate 1;

Next File

Next Ext


For Each Dir in dirlist (Root&'\*' ) // 5. 디렉토리에도 적용
Call DoDir (Dir) // 6. Subroutine 재호출
Next Dir

End Sub


Call DoDir ('C:') // 1. SubRoutine 호출

```

순서는 Call부터 시작한다고 생각하시면 됩니다.


1. Subroutine Call 부터 시작합니다. 인자를 C드라이브로 지정했네요.

2. Subroutine이 시작합니다. Root 안에는 Call에서 지정한 C드라이브가 저장됩니다.

3. 불러올 확장명을 지정합니다. 'qvw', 'qvo', 'qvs', 'qvt', 'qvd', 'qvc' 파일은 모두 불러오겠다는 뜻입니다.

4. 해당 파일의 이름, 파일 사이즈, 파일생성시간을 테이블로 불러옵니다.

5. dirlist는 해당 폴더 안에 있는 폴더(파일이 아님)를 모두 불러옵니다.

6. 그리고 앞서 정의한 Subroutine을 다시 정의하여, 그 폴더 안에 있는 모든 파일 정보를 테이블 안에 담습니다.


## 사용 용도

1. __QVD 파일을 생성할 때__

    - QVD 파일을 생성하는 로직을 Subroutine 안에 지정한 다음, Call 만으로 해당 로드문을 실행할 때 주로 사용할 수 있습니다.

    - [예시링크](https://community.qlik.com/t5/QlikView-App-Dev/Calling-Subroutine/td-p/293494)


2. __파일을 링크할 때__

    - Qlik 내에서 서버 안에 있는 파일을 링크시키고 싶울 경우, 위에서 사용했던 방법으로 폴더 안에 있는 파일과 폴더 정보를 가져옵니다.

    - 링크된 정보를 통해 QlikView '일반표' 차트 등에서 링크를 달아 해당 파일을 오픈할 수 있도록 지원합니다.
