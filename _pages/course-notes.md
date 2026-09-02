---
layout: archive
title: "Course Notes"
permalink: /course-notes/
author_profile: true
toc: false
---

这里整理我在计算机科学与人工智能课程学习过程中的系统笔记。内容以知识理解、算法推导和可运行代码为主，并会持续更新。

{% include base_path %}

{% for post in site.course-notes %}
  {% include archive-single.html %}
{% endfor %}
