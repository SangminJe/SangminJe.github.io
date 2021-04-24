---  
layout: post  
title: "Tableau 데이터 모델"
subtitle: "조인과 관계를 통해 알아보는 태블로"  
categories: Data  
tags: Data Tableau tableau 태블로 태블로-모델링 
comments: true  
---  


> 이 장은 태블로의 데이터 모델에 대해서 설명합니다.

## 1. 태블로 조인과 관계
<u>테이블의 조인과 관계는 다릅니다.</u> 현재는 관계(Relation)이 기본설정입니다.
- [테이블 조인과 관계](https://help.tableau.com/v2021.1/public/desktop/ko-kr/datasource_relationships_learnmorepage.htm#WhereAreJoins)
- **조인**은 미리 테이블을 특정 조인으로 합치는 과정이라고 생각하시면 됩니다.
- **관계**는 두 테이블의 관계를 논리적으로 설정하는 것입니다. 즉 양 테이블이 상호작용할 수 있도록 공통 필드를 바탕으로 관계를 맺는 것입니다. 
  - 관계는 테이블 간의 유기적인 `inner join`이라고 생각하시면 편합니다. 즉 테이블을 합치지는 않았지만 각 테이블을 `Key`를 중심으로 참조할 수 있습니다.

그럼 조인은 어떻게 구현할까요?  조인은 아래 그림과 같이 테이블을 누르고 열기버튼을 눌러 직접 구성할 수 있습니다.
![워크시트](https://sangminje.github.io/assets/img/tableau/data_model_singletable_joins.gif)



## 2. 태블로 관계(Relation) 만들기

![워크시트](https://sangminje.github.io/assets/img/tableau/tableau_datamodel1.png)
태블로에서 **데이터 원본**을 클릭하면 엑셀시트 목록이 보입니다. 두 개의 시트를 가져다 놓으면, 태블로에서 자동으로 공통 필드를 찾아내서 조인을 만들어줍니다.
- 여기서는 **ProductCategoryKey**가 공통키로 자동으로 설정되어 있음을 알 수 있습니다.


![워크시트](https://sangminje.github.io/assets/img/tableau/tableau_datamodel3.png)
여기에 추가적으로 **Product**시트도 추가해줍니다. 그러면 ProductSubCategory 테이블을 중심으로 키를 생성해줍니다.

![워크시트](https://sangminje.github.io/assets/img/tableau/tableau_datamodel2.png)
마지막으로 데이터를 중심으로 화면을 봐줍니다.

