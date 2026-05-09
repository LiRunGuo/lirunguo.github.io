---
layout: archive
title: "Curriculum Vitae"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* **M.S. in Computer Science**, National University of Singapore (NUS), 2026.8 – 2027.7 (expected)
* **B.S. in Business Analytics**, Shanghai University of Finance and Economics (SUFE), 2022.9 – 2026.6
  * GPA: **3.73 / 4.0** (Rank **8 / 128**)
  * TOEFL: 101 · GRE: 327 · JLPT: N2

Research & Industry Experience
======

**Tencent Youtu AI Lab** — Research Intern *(2025.10 – 2026.3)*
* **Adversarial Data Synthesis.** Built an LLM-driven adversarial sample synthesis system with a 5-dimensional evolution strategy (style transfer, scene injection, metaphor substitution, complex logic, multilingual mixing). Generated 100K+ highly stealthy violation samples from seed data with Asyncio-based concurrency and SQLite logging.
* **Medusa Unified Moderation Model.** Designed a "linear-head + Medusa TextCNN multi-head" architecture on XLM-RoBERTa for 26-dimensional atomic detection, integrating 100+ rules. Achieved **92.3%** accuracy on financial violations, **−35%** false-positives, **+60%** moderation efficiency, and hundreds of thousands of QPS.
* **WeMM-Embedding.** Co-developed a multimodal embedding model on a Qwen3-VL backbone with a Deep Fusion module. Reached open-source SOTA on **MMEB-V2** and **UVRB** leaderboards.

**SUFE FinAI Center** — Research Assistant *(2025.6 – present)*
* **ICBC Scientist-Discovery Agent.** Co-developed a RAG + Agent system that mines scientist information, analyzes affiliated enterprises, and predicts sales opportunities. Contributed to FinCorpus (a 100B-token financial corpus).
* **Fin-R1 Financial LLM.** Co-developed China's first open-source financial reasoning LLM (7B, based on Qwen2.5-7B-Instruct). Built 60K financial CoT data, trained with **SFT + GRPO**. Achieved FinEval score **75.2** (**2nd place** overall).
* **Financial Evaluation Suite.** Co-designed FinEval (26K problems), Alibaba Cloud financial multimodal evaluation data, and Ant Group's financial hard-problem evaluation set.

**ICBC Head Office, Private Banking Department** — Research Intern *(2026.1 – present)*
* **Scientist-Discovery Agent.** Built multi-dimensional profiles of high-net-worth scientists from open data (patents, papers, registries). Used Agent + RAG to automate relationship construction, opportunity insight, and personalized "one-customer-one-strategy" marketing.
* **Wealth-Advisor Empowerment.** Developed a Private-Banking sparring assistant that distills financial knowledge and best practices, simulates real-world scenarios, and produces sales scripts and recommendations.

Publications
======
* **FinVault: Benchmarking Financial Agent Safety in Execution-Grounded Environments.** *First Co-Author.* Under review at **ACL 2026**.

  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Selected Projects
======

**Fin-R1 — Financial Reasoning LLM with Reinforcement Learning**
* *Background:* Develop a 7B lightweight model for complex financial reasoning across banking, securities, insurance and trust businesses.
* *Approach:* Distilled 60K financial CoT data from DeepSeek-R1; two-round quality filtering (rules + model scoring + logical consistency); two-stage training (SFT + GRPO); Verifier-augmented reward.
* *Results:* FinQA / ConvFinQA near SOTA, FinEval **75.2** (2nd place); FinQA 76.0, ConvFinQA 85.0; vLLM one-click deployment.

**ICBC — Scientist–Entrepreneur Integrated Agent**
* *Background:* Build an intelligent customer-identification and relationship-construction system for the scientist–entrepreneur community to support tech-transfer business.
* *Approach:* LLM-based parsing of patents, papers and news; unified scientific knowledge graph and enterprise data; RAG-based expert KB; family-business opportunity insight, dialogue sparring, and one-customer-one-strategy marketing.
* *Results:* Multi-criteria scientist discovery, panoramic profiles (person/firm/family/society), affiliated-enterprise analysis, automatic marketing-graph generation.

**Tencent — Medusa Unified Content-Safety Moderation Model**
* *Background:* Internet content evolves quickly; rule-based systems struggle with new violation patterns.
* *Approach:* "Linear-head + Medusa TextCNN multi-head" hybrid on XLM-RoBERTa; partial-label masking for sparse labels; 10 prompt templates for few-shot learning.
* *Results:* 92.3% accuracy, −35% false positives, +60% efficiency; 100+ atomic capabilities online; hundreds of thousands of QPS.

**Tencent — WeMM-Embedding Multimodal Embedding Model**
* *Background:* General multimodal embeddings unifying text, image and video for cross-business retrieval inside the WeChat ecosystem.
* *Approach:* Qwen3-VL backbone; Deep Fusion of multi-level semantics; deduplicated InfoNCE; hierarchical sampler for arbitrary modality mixing.
* *Results:* **MMEB-V2 0.7523** (open-source #1), **UVRB 0.686** (open-source #1), +2.6% on Chinese image-text retrieval.

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
