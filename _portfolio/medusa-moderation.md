---
title: "Medusa — Unified Content-Safety Moderation Model"
excerpt: "Production-grade unified moderation model: 92.3% accuracy, −35% false positives, +60% efficiency.<br/>"
collection: portfolio
---

**Medusa** is a unified content-safety moderation model for large-scale internet platforms, designed to keep up with rapidly evolving violation patterns where rule-based systems fail.

### Approach
- Backbone: **XLM-RoBERTa**.
- Hybrid head architecture: **linear head + Medusa TextCNN multi-head**.
- **Partial-label masking** to handle sparse / weakly-labeled data.
- 10 prompt templates enabling few-shot detection of new categories.

### Results
- **92.3%** accuracy on financial violations.
- **−35%** false positives, **+60%** moderation efficiency.
- **100+** atomic capabilities deployed online.
- Hundreds of thousands of QPS in production.
