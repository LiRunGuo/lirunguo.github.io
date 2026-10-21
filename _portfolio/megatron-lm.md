---
title: "Megatron-LM — Distributed Checkpointing for MoE"
excerpt: "Contributor to Megatron-LM: merged a distributed-checkpointing change that describes MoE shards with prepended layer and expert axes through the DCP checkpointable path — 165× faster state-dict conversion and a 3.4× shorter blocking phase of async save on 8×H200, with identical checkpoint metadata."
collection: portfolio
category: contribution
order: 12
permalink: /portfolio/megatron-lm/
---

I contribute to [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), focusing on the parts of large-scale MoE training that sit on the critical path without looking like it, such as the synchronous prefix of an asynchronous checkpoint.

### Merged contribution

**[PR #7287 — perf(checkpointing): describe prepended-axis shards with the checkpointable path](https://github.com/NVIDIA/Megatron-LM/pull/7287)** · Merged October 7, 2026 (+247 / −18 across 4 files). Closes [#7275](https://github.com/NVIDIA/Megatron-LM/issues/7275).

`mcore_to_pyt_state_dict` sent every `ShardedTensor` with a prepended axis through the legacy PyTorch `ShardedTensor` conversion, which enumerates the *entire global shard grid*, remote shards included, for each key. Prepended axes are the common case for MoE: a homogeneous `TransformerBlock` adds a layer axis to every parameter, and `SequentialMLP` adds a global-expert axis on top for each routed expert. For a 48-layer model with 128 experts and EP=8, that is 4.7 million grid entries per rank, all processed synchronously before the async write request returns, so asynchronous checkpointing could not hide it from the training loop.

A prepended axis has extent one within a local chunk, and the DCP checkpointable path can describe that directly. `CheckpointableShardedTensor` now reports prepended axes as singleton dimensions of the local chunk and exposes the data through a matching view, with no payload copy. The resulting DCP metadata is identical to what the legacy conversion produced, so old checkpoints still load and new ones are indistinguishable from the old format.

| Phase (8×H200, PyTorch 2.10, `torch_dist`) | Before | After |
| --- | --- | --- |
| `mcore_to_pyt_state_dict` | 11.71 s | **0.07 s** (165×) |
| Synchronous prefix of `async_save()` | 17.75 s | **5.22 s** (3.4×) |

New tests check write-item metadata parity with the legacy conversion, save/load round trips, loading checkpoints written by the legacy path, and MoE round trips for EP=1/2/4 including EP=4 → EP=2 resharding. The existing distributed-checkpointing suites pass on 8 GPUs.

[View my Megatron-LM pull requests](https://github.com/NVIDIA/Megatron-LM/pulls?q=is%3Apr+author%3ALiRunGuo)
