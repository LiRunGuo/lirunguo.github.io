---
title: "DeepSpeed — Distributed Training & Hybrid Engine"
excerpt: "Contributor to DeepSpeed: merged fixes for ZeRO-3 rollout synchronization and Hybrid Engine fallback for unsupported model architectures."
collection: portfolio
category: contribution
order: 2
permalink: /portfolio/deepspeed/
---

I contribute to [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), focusing on the correctness and reliability of distributed training and Hybrid Engine rollouts.

### Selected merged contributions

- **[PR #8264 — Fix ZeRO-3 synchronization during OPSD rollout](https://github.com/deepspeedai/DeepSpeed/pull/8264)** · Merged August 27, 2026. Fixed a distributed deadlock caused by different ranks stopping generation at different times. Kept decoding and parameter-gather collectives aligned across ranks, added a regression test, and validated the fix on NVIDIA H200 and AMD MI250 GPUs.
- **[PR #8265 — Fallback for unsupported Hybrid Engine policies](https://github.com/deepspeedai/DeepSpeed/pull/8265)** · Merged August 29, 2026. Prevented partial inference-policy injection for unsupported model architectures, allowing models such as Qwen2.5 to retain native generation. Added CPU unit coverage and validated distributed OPSD training on AMD MI250 GPUs.

[View my DeepSpeed pull requests](https://github.com/deepspeedai/DeepSpeed/pulls?q=is%3Apr+author%3ALiRunGuo)
