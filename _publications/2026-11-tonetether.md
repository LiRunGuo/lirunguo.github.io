---
title: "ToneTether: Lowering Drift Without Freezing in Long Autoregressive Video Generation"
collection: publications
category: preprints
permalink: /publication/2026-11-tonetether
unlisted: true
excerpt: 'Drift scores reward long videos that stop changing. ToneTether corrects only the predicted global tone of the drift in the KV cache and removes a quarter to a third of the colour drift without a measurable loss of motion or new content.'
date: 2026-11-16
venue: 'Manuscript'
bibtexurl: '/files/bibtex/li2026tonetether.bib'
citation: 'Runguo Li. (2026). &quot;ToneTether: Lowering Drift Without Freezing in Long Autoregressive Video Generation.&quot; Manuscript.'
---

**Status:** Manuscript · **Project page:** [ToneTether](/projects/2026-11-tonetether/)

### Abstract

Autoregressive video diffusion models now stream minutes of video, but errors compound and the video drifts in colour, texture and structure. Training-free corrections anchor the key-value (KV) cache to a trusted reference and are judged by drift scores, which compare a video's ending with its opening. A drift score cannot tell a corrected video from one that has stopped changing, so it rewards freezing. Scoring progression alongside drift, through motion, novelty, prompt following and scripted tone changes, overturns the ranking: the corrections that score best on drift are the ones that freeze. An analysis of the KV cache shows that drift shares its directions with legitimate change, so correcting every direction erases change along with drift. ToneTether removes only the predicted global tone of the estimated drift, the smallest such edit in units of legitimate variation. Added to TetherCache, it removes a quarter to a third of colour drift on two backbones and two horizons with no measurable loss of motion or new content, and, with a short empty-cache rollout as reference, more than doubles the share of a prompted tone change that a long video shows.
