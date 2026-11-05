---
title: "Do Search Agents Know When They Know? Answer-State and Evidence Dynamics in RL-Trained Search Agents"
collection: publications
published: false
category: preprints
permalink: /publication/2026-11-search-agent-answer-state
excerpt: 'An exact replay protocol tests whether RL-trained search agents know when their evidence is sufficient, across 6,125 questions and 97,841 decision states. Single-author work under review.'
date: 2026-11-02
venue: 'Under review'
paperurl: '/files/search-agent-answer-state.pdf'
bibtexurl: '/files/bibtex/li2026searchagent.bib'
citation: 'Runguo Li. (2026). &quot;Do Search Agents Know When They Know? Answer-State and Evidence Dynamics in RL-Trained Search Agents.&quot; Manuscript under review.'
---

**Status:** Manuscript under review · **PDF:** [Download](/files/search-agent-answer-state.pdf)

### Author

Runguo Li (single author).

### Abstract

Search agents trained with reinforcement learning answer questions by querying a search engine and reading pages until they decide to stop. Methods that make them stop earlier assume that the agent's internal state reveals when its evidence is sufficient. This paper tests that assumption with an exact diagnostic protocol: at every decision point of a frozen agent, it forces the answer the agent would give if stopped, measures how retrieved passages change that answer, and replays stopping policies offline. Across seven QA benchmarks, public RL-trained agents often hold the right answer and then lose it. The paper releases SufficiencyBench with 97,841 decision states, oracle labels, and an exact replay harness.

### Key Findings

- With reasoning allowed, **5–8%** of questions are answered correctly at some point but incorrectly at the end.
- At least two thirds of those questions are already correct before any search, and **3–5%** are lost beyond what sampling at the final state recovers.
- Single retrieved passages rarely decide the current answer on their own.
- Agent confidence, sample agreement, and hidden-state probes separate correct from incorrect answers only moderately; none of the evaluated stopping methods improves significantly.
- Restating the question after every observation raises accuracy by **0.5–1.7 points** for five of six agents, while a placebo reminder does not.

**Full text:** [Download the PDF](/files/search-agent-answer-state.pdf)
