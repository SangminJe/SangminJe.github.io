---  
layout: post  
title: "R에서 Deep Learning을 하기 위한 환경 설정"
subtitle: "Docker를 알아야 한다.."  
categories: DEV
tags: DEV Backend Docker R 
comments: true  
---  

- 이 문서는 Issac님의 Kaggle Study 모임에서 공부한 내용입니다.
- Docker 설치와 관련된 자세한 내용은 [링크](https://www.youtube.com/watch?v=VVxvL4xRPjU)를 참조하세요.

# Docker를 설치하는 이유
---

- R을 활용해서 Deep Learning을 하려고함
    - `Torch`패키지는 R에서 깔리긴 하나,
    - 함수 명령어를 사용하면 Rtudio가 뻗어버림
    - 그래서 Docker로 리눅스 환경을 만들어서 활용하려고 하는 것

</br>

# 사전 준비사항
---
1. Window 10 or 11
2. WSL2 깔려있어야 함
3. Docker Desktop 설치 되어 있어야 함

* 위 사항은 [이삭님의 유튜브](https://www.youtube.com/watch?v=VVxvL4xRPjU)를 보시고 따라오시면 됩니다.

</br>

# 설치과정
---

## 1. Docker Desktop 실행

### 1.1. Virtual Machine Error 해결
---
설치과정에서 바로 복병을 만났다. 그것은 바로 Vritual Machine Error..
![img](https://sangminje.github.io/assets/img/docker/docker_error_virtual.PNG)

- 이 에러의 이유는 사실 BIOS에서 가상화를 사용하지 않기로 설정해서 일어나는 것이다.
- 작업관리자 - 성능에서 확인할 수 있음
![img](https://sangminje.github.io/assets/img/docker/docker_error_virtual2.PNG)

- 이 가상화 사용을 사용으로 바꿔주기 위해서는 각자의 메인보드에서 BIOS 환경으로 진입하여 가상화 환경을 **ENABLED**로 바꿔줄 필요가 있다. 이 과정은 메인보드마다 다르기 때문에 생략!

### 1.2. WSL2 Installation is incomplete
---
이제는 WSL2가 제대로 설치가 되지 않았다고 한다. 분명 Microsocft에서 설치를 했는데..

![img](https://sangminje.github.io/assets/img/docker/docker_error_virtual3.PNG)

이럴 땐 [이 블로그](https://blog.nachal.com/1691)를 참조하여 Docker 리눅스 설치를 마무리해주자.


## 2. Docker에 이미지 불러오기


### 2.1. rocker/tidyverse 이미지 가져오기(Test)
---

![img](https://sangminje.github.io/assets/img/docker/docker_error_virtual4.PNG)

- [Docker Hub](https://hub.docker.com/)에 접속해서 아이디를 만든 후 이미지를 가져올 수 있다.
- 이미지를 가져오는 방법은 **PowerShell**을 통해 명령어를 사용하는 방법이다.
- 예를 들어서 위에서 **rocker/tidyverse**라는 이름의 이미지를 가져다 쓰고 싶은 경우 아래와 같이 명령어를 써주면 된다.

```
docker pull rocker/tidyverse
```

![img](https://sangminje.github.io/assets/img/docker/docker_error_virtual5.PNG)

- 그럼 PowerShell에서 다운로드 과정이 보여지게 되고, 위 그림과 같이 Docker Desktop에 이미지가 생성이 된다.


### 2.2. Rstudio 특정 버전 허브에서 가져오기
---

- `docker pull rocker/rstudio:4.0.5`라고 명령어를 실행
- Rstudio와 R 버전이 4.0.5인 이미지가 다운로드가 된다.

![img](https://sangminje.github.io/assets/img/docker/docker_error_virtual8.PNG)

### 2.3. 다운 받은 이미지(DockerFile)를 컨테이너에서 실행하기
---

아래 명령문을 실행한다. 
```
docker run -d -p 4567:4567 -v ${pwd}:/home/rstudio -e PASSWORD=random --name myfirst-rstudio-docker rocker/rstudio:4.0.5
```
- `docker run ` - docker를 실행하는데,
- `-p 4567:4567` - 포트는 4567로 맞추고
- `-v ${pwd}:/home/rstudio` - 현재 디렉토리에 있는 파일을 home/rstudio에 보내고,
- `PASSWORD=random` - Password는 "random"으로 정한뒤
- `--name myfirst-rstudio-docker` - 그리고 이름은 my-first-rstudio이고
- `rocker/rstudio:4.0.5` - rocker/rstudio:4.0.5에서 실행한다.
- 이로써 pwd, 즉 현재 디렉토리에 있는 로컬 환경이 컨테이너로 이전되어 실행된다.

위 과정은 현재 내 **로컬 폴더**와 **가상 환경(도커)** 를 연결하는 것이다. **이미지**와 **컨테이너**는 다른 개념이다. **이미지**가 CD라고 한다면 **컨테이너**는 그 CD가 깔리는 별도의 공간이라고 생각하면 된다.


## 3. Docker 이미지 만들기

### 3.1. DockerFile 만들기
---

![img](https://sangminje.github.io/assets/img/docker/docker_error_virtual6.PNG)
- R 안에서 **Dockerfile**을 만드는데, Docker의 확장자는 주지 않는다.
- R에서 작성하기보다, 메모장이나 Notepad++와 같은 tool로 작성하길 권장한다.
- 그리고 **test.R** 스크립트 파일에 `print("Hello Docker")`로 입력해놓는다.

### 3.2. DokerFile 명령어 쓰기
---

- `FROM r-base:4.0.3` - R 버전은 4.0.3 버전으로 특정
- `COPY ./test.R /home` - 현재 디렉토리의 test.R을 가상 컴퓨터(도커)의 home이라는 폴더에 저장하라는 뜻
- `WORKDIR /home` 
- `CMD Rscript 'test.R'` - Rscript 안에 있는 명령어를 실행하라는 뜻

### 3.3. PowerShell에서 docker 이미지 만들기
---

- 먼저 PowerShell에서 위치한 디렉토리는 Dockerfile이 위치한 디렉토리여야 한다.
- 해당 디렉토리에서 Powershell을 사용하여 `docker build -t hello-rocker .` 명령어를 쳐준다.
- 이 때 **.** 과 **공백**이 빠지지 않도록 주의한다.
- 그리고 이미지가 생성되면, 생성한 이미지를 `docker run hello-rocker` 명령어를 통해 불러보자. "Hello Docker" 가 생성되면 성공!
- 현재까지의 과정에 대한 스크린샷
![img](https://sangminje.github.io/assets/img/docker/docker_error_virtual7.PNG)

### 3.4. Docker Hub Docker 파일 살펴보기
---

- [Docker Hub](https://hub.docker.com/)에 있는 DockerFile을 보면 명령어들이 많이 있다.
- 해당 명령어를 학습해서 자기만의 Docker 이미지를 만드는 게 가능함


