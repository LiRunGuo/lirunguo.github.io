---
title: "WeMM-Embedding — Multimodal Embedding Model"
excerpt: "Open-source SOTA on MMEB-V2 (0.7523) and UVRB video retrieval (0.686).<br/>"
collection: portfolio
---

**WeMM-Embedding** is a general-purpose multimodal embedding model unifying **text, image and video** representations to support cross-business retrieval inside the WeChat ecosystem.

### Approach
- Backbone: **Qwen3-VL**.
- **Deep Fusion** module integrating multi-level semantic representations.
- Deduplicated **InfoNCE** loss to remove false negatives.
- Hierarchical sampler supporting arbitrary modality mixing during training.

### Results
- **MMEB-V2 0.7523** — open-source #1.
- **UVRB 0.686** for video retrieval — open-source #1.
- +2.6% improvement on Chinese image–text retrieval.
