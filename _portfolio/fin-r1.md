---
title: "Fin-R1 — Open-Source Financial Reasoning LLM"
excerpt: "China's first open-source financial reasoning LLM (7B). FinEval score 75.2 (2nd place overall).<br/>"
collection: portfolio
---

**Fin-R1** is a 7B-parameter open-source financial reasoning model based on Qwen2.5-7B-Instruct, designed for complex reasoning tasks across banking, securities, insurance and trust businesses.

### Approach
- Distilled **60K** financial chain-of-thought (CoT) samples from DeepSeek-R1.
- Two-round quality filtering: rule-based + model scoring + logical-consistency verification.
- Two-stage training: **SFT** for behavior cloning, then **GRPO** for reasoning refinement.
- Verifier-based reward signal to improve reliability.

### Results
- **FinEval 75.2** — 2nd place overall.
- Near-SOTA on **FinQA (76.0)** and **ConvFinQA (85.0)**.
- Outperforms all comparable open models at the 7B scale.
- One-click deployment via vLLM.
