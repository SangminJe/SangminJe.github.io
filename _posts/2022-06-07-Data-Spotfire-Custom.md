---  
layout: post  
title: "Spotfire Customization Guide"
subtitle: "Spotfire를 Customization 하기 위한 링크 및 정보 모음"  
categories: DATA
tags: DATA spotfire customization 커스터마이제이션 IronPython Javascript side panel
comments: true  
---  

# Spotfire 참고 사이트
---
- [Spotfire Youtube](https://www.youtube.com/user/TibcoSpotfire/videos) : 공식 유튜브
- [정승렬 교수 Youtube](https://www.youtube.com/c/%EC%A0%95%EC%8A%B9%EB%A0%AC/videos) : Spotfire 한국어 강의 채널, 유용하고 쉬움!
- [Spotfire Enablement Hub](https://community.tibco.com/wiki/spotfire-enablement-hub) : Spotfire의 입문자~숙련자까지 사용정보를 주제 별로 분류해놓은 사이트
- [Spofire Developer Blog](https://spotfired.blogspot.com/2014/02/exporting-to-excel-from-client-only.html) : 
Spotfire 비공식 개발 방법 블로그
- [Making data look good](https://community.tibco.com/feed-items/making-data-look-good) : 동적이고 Interactive한 UI를 구현하는 방법/예시


# Spotfire 디자인
---
1. **디자인 Demos**
    - [RETAIL CUSTOMER  ANALYTICS
 DEMO](https://demo.spotfire.cloud.tibco.com/spotfire/wp/analysis?file=/Public/Retail%20Customer%20Analytics/Retail%20Customer%20Analytics%20%28No%20TDS%29&waid=eKDkpZrTjECl9344tkFad-0514384d80wyMO&wavid=0&options=13-1,10-1,9-1,5-0,6-0,17-0,11-1,12-1,14-1,1-0,3-0,18-0,7-0,15-0,19-0,4-1,2-1)
    - [외국 고객사 DEMOs](https://www.youtube.com/watch?v=OkEms9Tt9nw&list=PLknbq-WaCOiUEJ5esJcxdT2n9KxITH9W6&index=2)
2. **디자인 요소**

    - UI/UX 측면 Customization
        - [Text Areas to Increase Usability of Analytics through HTML, Javascript and CSS](https://community.tibco.com/wiki/using-spotfire-text-areas-increase-usability-analytics-through-html-javascript-and-css) : HTML과 CSS, JS 사용하여 구현 가능한 기능들을 나열, 유용하고 실질적인 예시들이 있어서 가장 먼저 보고 따라해보면 좋음!
        - [Extending TIBCO Spotfire®](https://community.tibco.com/wiki/extending-tibco-spotfire) : Spotfire 확장 기능 전반에 대한 안내
        - [Dr. Spotfire - Data Visualization Best Practices](https://www.youtube.com/watch?v=-JPIY6qltxw) : Best Practice 구현 예시, 불완전한 대시보드를 Customizing해서 새롭게 구현하는 방법에 대해 설명
        - [How To Use The TIBCO Spotfire® JavaScript API](https://community.tibco.com/wiki/how-use-tibco-spotfire-javascript-api) : Javascript API 사용방법
        - [Pretty KPI's in Spotfire using HTML and CSS](https://www.youtube.com/watch?v=VyfzqCu_pxs) : HTML과 CSS로 직접 Text 객체 수정해서 만드는 과정 실습
        - [HTML Geberator](https://www.tablesgenerator.com/html_tables#) : HTML Code를 GUI 기반으로 생성
        - [How to include your own instances of jQuery and jQueryUI in Text Areas](https://community.tibco.com/wiki/how-include-your-own-instances-jquery-and-jqueryui-text-areas)
        - [Color Brewing](https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3), [Adobe color](https://color.adobe.com/ko/create/color-wheel) : 색 파레트 지정 사이트
        - [How to Create Custom Filter Panels in TIBCO Spotfire X](https://www.youtube.com/watch?v=dCyk90WAAFc&list=PLkXZXEEEwIOHS7Z18FGg16q6w08zkizP4&index=9)

    - Spotfire Mod
        - 참고 사이트
            - [TIBCO Spotfire Mods 공식사이트](https://www.tibco.com/ko/products/tibco-spotfire/custom-analytics-apps-mods?page=0)
            - [TIBCO Spotfire® Mods Overview](https://community.tibco.com/wiki/tibco-spotfirer-mods-overview)
            - [Spotfire Mod Github](https://github.com/TIBCOSoftware/spotfire-mods)
            - [Mods 101: The Basics of Custom Visualizations in Spotfire 11](https://www.youtube.com/watch?v=XmeEAIYsYOw&list=PLknbq-WaCOiWeNXYXdc-zgosw9njjAyYe)
            - [Mods Documentation](https://tibcosoftware.github.io/spotfire-mods/docs/)

        - **Spotfire Mods Use Videos**
            - [Spotfire Quick Tips](https://www.youtube.com/watch?v=KbQpUXv335Q&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=2)
            - [Text Card](https://www.youtube.com/watch?v=b2_gU46C4TQ&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=3)
            - [Sankey Diagram](https://www.youtube.com/watch?v=KbQpUXv335Q&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=2)
            - [Rador Chart(Spyder Chart)](https://www.youtube.com/watch?v=hkYVDlKyeMA&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=5)
            - [Bump Chart](https://www.youtube.com/watch?v=jThZXXiW7yg&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=6)
            - [Candle Chart](https://www.youtube.com/watch?v=edRjvrO52L4&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=7)
            - [WordCloud](https://www.youtube.com/watch?v=Norc39iSCu8&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=8)
            - [Area Chart](https://www.youtube.com/watch?v=0wnyr94FcYg&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=9)

    - Iron Python
        - [IronPython Scripting](https://community.tibco.com/wiki/ironpython-scripting-tibco-spotfire)
        - [Spotfire Analyst API](https://docs.tibco.com/pub/doc_remote/sfire_dev/area/doc/api/TIB_sfire-analyst_api/Index.aspx?_ga=2.100716290.2045688560.1654419060-1977636396.1652686758)
        - [Useful IronPython Scripts from TIBCO's COVID19 Dashboard](https://community.tibco.com/wiki/useful-ironpython-scripts-tibcos-covid19-dashboard)



# Spotfire R/Python 연계
---
1. Python, R을 사용한 전처리
    - [Run Python and R in Spotfire Data Canvas Nodes](https://www.youtube.com/watch?v=z3uhOBmraak&list=PLknbq-WaCOiU95NPfj7jUeVOzVGBwnCUW&index=22)
    - [Python Data functions in TIBCO Spotfire](https://community.tibco.com/wiki/python-data-functions-tibco-spotfire)
    - [정승렬교수 강의 - Spotfire 고급 17 Data Function](https://www.youtube.com/watch?v=GNsLx4d803E&list=PLpy_NIroiQ3ZA9agjUH_xlKLzplBWNwEj&index=17)


