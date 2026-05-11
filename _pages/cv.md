---
layout: archive
title: "Curriculum Vitae"
permalink: /cv/
author_profile: true
toc: false
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* **M.S. in Computer Science**, National University of Singapore (NUS), 2026.8 – 2028.7 (expected)
* **B.S. in Business Analytics**, Shanghai University of Finance and Economics (SUFE), 2022.9 – 2026.6
  * TOEFL: 101 · GRE: 327 · JLPT: N2

Research & Industry Experience
======

**Tencent Youtu AI Lab** — Research Intern *(2025.10 – 2026.5)*
* LLM-driven adversarial data synthesis for content-safety evaluation.
* Contributed to a unified multi-label content-moderation model on an XLM-RoBERTa backbone.
* Co-developed a multimodal embedding model for cross-modal retrieval.

**SUFE FinAI Center** — Research Intern, advised by Prof. **Liwen Zhang (张立文)** *(2025.6 – present)*
* Financial reasoning LLM: CoT data curation, SFT + GRPO training, evaluation on FinEval / FinQA / ConvFinQA.
* Co-design of financial evaluation suites and agent-safety benchmarks.

**Shanghai Jiao Tong University (SJTU)** — Research Intern, advised by Prof. **Xiaodong Gu (顾小东)** *(2026.3 – present)*
* Research direction: **LLM for Code** — code generation, repair, and program understanding.

**ICBC Head Office, Private Banking Department** — Research Intern *(2026.1 – 2026.4)*
* Agent + RAG system for scientist discovery, relationship construction, and one-customer-one-strategy marketing.
* Private-banking sparring assistant producing sales scripts and recommendations.

**LanMa Technology (澜码科技)** — Agent Development Intern *(2024.1 – 2024.4)*
* Agent development: workflow orchestration, tool invocation, and prompt engineering for enterprise-grade LLM agents.

Publications
======

  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Selected Projects
======

**ARH — AI Research Helper** *(Open Source · [github.com/LiRunGuo/Arhelper](https://github.com/LiRunGuo/Arhelper))*
* *Background:* A CLI-first research assistant agent for AI researchers — reading papers, searching code, running experiments, writing summaries.
* *Approach:* Plan/Execute safety model with approval-gated writes; three-layer memory (regex + LLM preference extraction, BM25 cross-session recall, end-of-session reflection summary); skill self-learning; multi-LLM fallback (OpenAI / Anthropic / Ollama); automatic context compression; checkpoint & resume.
* *Stack:* Python 3.9+, Pydantic, FastAPI, SQLAlchemy; Telegram & CLI gateways; arXiv / HuggingFace / GitHub / OpenReview crawlers.
* *Status:* 0.2.0-alpha, MIT licensed, all data stored locally in `~/.arh/`.

Skills
======
* **LLM Development** — SFT / GRPO / PPO, distillation, deployment (vLLM, HuggingFace)
* **Multimodal Models** — VLM fine-tuning, model merging (SLERP / TIES / DARE), adversarial data synthesis
* **Agents & RAG** — agent workflow orchestration, retrieval, knowledge bases
* **Frameworks & Tools** — Python, PyTorch, HuggingFace, FAISS, Linux, Docker

Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>

Teaching
======
  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Honors & Awards
======
* National Encouragement Scholarship
* People's Scholarship, 2nd Class
* Silver Medal, FLTRP English Competition (Municipal)

Languages
======
* Chinese (native), English (TOEFL 101 · GRE 327), Japanese (JLPT N2)
