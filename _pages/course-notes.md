---
layout: archive
title: "Course Notes"
permalink: /course-notes/
author_profile: false
toc: false
---

这里整理我在计算机科学与人工智能课程学习过程中的系统笔记，共 37 份 PDF。每门课程保留名称与简短介绍，点击课程标题即可直接打开对应 PDF。

{% for section in site.data.course_notes %}
<h2 class="archive__subtitle">{{ section.name }}</h2>
<p>{{ section.description }}</p>

{% for course in section.items %}
<div class="list__item">
  <article class="archive__item" itemscope itemtype="http://schema.org/CreativeWork">
    <h2 class="archive__item-title" itemprop="headline">
      <a href="{{ course.pdf | relative_url }}" target="_blank" rel="noopener" itemprop="url">{{ course.title }}</a>
    </h2>
    <p class="archive__item-excerpt" itemprop="description">{{ course.description }}</p>
  </article>
</div>
{% endfor %}
{% endfor %}
