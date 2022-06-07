---  
layout: post  
title: "[Spotfire] IronPython 입문"
subtitle: "Spotfire에서 IronPython 사용하기 기본"  
categories: DATA
tags: DATA spotfire customization button IronPython Javascript html css 버튼
comments: true  
--- 

# IronPython 개요
---
- IronPython은 Spotfire 내에서 자동화, 동적인 차트 구성, Customization 등에 사용된다.
    - visualizations, data tables, filters, markings, and bookmark 등 다양한 영역에 개입한다.
- IronPython은 기본적으로 Python 문법을 따르지만, Library Dependent 하다.
    - 파이썬 문법을 몰라도, Scikit-learn 라이브러리를 익히면 통해 대부분의 머신러닝을 처리하듯 Spotfire의 API에 익숙해지고 몇 가지 예시를 수행하면 수행 가능한 영역으로 보인다.
    - 따라서 Python에 대한 선행지식이 반드시 필요하지는 않다.
- IronPython 더 나아가면 C#의 영역까지 넓어질 수 있으나, 일반적인 구축에서 C#까지 필요하진 않을 듯하다.

# 참고 사이트
---
- [Spotfire Analyst API](https://docs.tibco.com/pub/doc_remote/sfire_dev/area/doc/api/) : 해당 사이트는 IronPyothon을 사용하기 위해 Spotfire의 객체 구조(NameSpace) 등을 정리한 사이트이다. IronPython을 개발하면서 해당 사이트의 내요을 참고하면 좋을 것 같다.
- [The Spotfire IronPython Quick Reference](https://www.sf-ref.com/) : IronPython을 예제 별로 하나씩 따라해볼 수 있는 사이트이다. IronPython의 매커니즘을 이해하기 좋다.
