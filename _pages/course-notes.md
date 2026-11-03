---
layout: archive
title: "Course Notes"
permalink: /course-notes/
author_profile: false
toc: false
---

These are my structured notes from computer science and AI courses, with 37 PDFs in total. Each course includes a short description; click a course title to open its PDF.

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
