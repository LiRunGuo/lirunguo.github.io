---
title: "colibri — MoE Inference Engine"
excerpt: "Contributor to colibri: three merged pull requests covering multi-drive expert streaming for the 510 GB DeepSeek-V4.1 container, OpenMP team sizing in the four engines that never had it, and a plan/engine mismatch that silently allocated 6.23 GiB of unbudgeted KV cache."
collection: portfolio
category: contribution
order: 1
permalink: /portfolio/colibri/
---

I contribute to [colibri](https://github.com/JustVugg/colibri) — a pure-C, dependency-free engine that runs frontier MoE models on hardware you already own by streaming experts from disk. My work there sits between the machine and the model: how experts are read off storage, how the OpenMP team is sized, and whether the resource plan and the runtime agree about what is going to be built.

### Merged contributions

**[PR #1522 — read the expert stream from more than one drive](https://github.com/JustVugg/colibri/pull/1522)** · Merged September 15, 2026 (+1,377 / −36 across 10 files).

`deepseek_v4.c` had `COLI_MODEL_MIRROR` and direct (`O_DIRECT`) expert reads for many releases; `deepseek_v41.c` — its direct successor, and the project's largest container at 510 GB — had neither. The change adds mirrored read-only replicas spread across drives with a flat-index hash (including the clustering fix that stops an XOR of two small ids from pinning the hot expert subset to one replica), a `COLI_MODEL_DIRS` layout that splits the container across drives as distinct shards with no second copy, and an alignment-aware direct-read primitive in `st.h` that other engines can adopt. Validated end to end on the released 510.3 GB checkpoint: all 88 files verified against the upstream manifest, 36 runs token-exact across both regimes.

**[PR #1517 — size the OpenMP team in the four engines that never did](https://github.com/JustVugg/colibri/pull/1517)** · Merged September 15, 2026 (+415 / −22 across 10 files).

`omp_tune.h` had carried physical-core team sizing for four engines; four engines written afterwards never called it, so they ran one thread per *logical* CPU. Nothing errored — on a 208-logical-CPU host, `deepseek-v41` spent **93% of its cycles inside libgomp** and decoded **18.7× slower** than with a sane team. The pull request calls the existing helper from the four missing engines and replaces a one-engine test with a table-driven gate over every engine, stripping comments before matching so a commented-out call cannot pass (verified by commenting the call out and watching it fail).

**[PR #1526 — make the plan and the engine agree about the engine](https://github.com/JustVugg/colibri/pull/1526)** · Merged September 15, 2026 (+111 / −0 across 3 files).

Two places where the resource planner and the engine disagreed, neither raising an error at the point of divergence. `deepseek_v41` was the only family that declared a context variable and never read it, so the engine always sized its buffers from the checkpoint's 1,048,576-position ceiling — **6.25 GiB of compressed KV cache and index keys against the 0.02 GiB the registry's 4,096 default implies**, unbudgeted on the default path. The second: an engine whose build rule links no accelerator backend still advertised `supports_accelerator`, so the planner proposed a placement the runtime could not honor. Both are now covered by `tests/test_registry_engine_agreement.py`, checked in both directions with each fix reverted to prove the gate can fail.

### Measurement records

- **[#1518 — the OpenMP team is the whole ballgame on a many-core host](https://github.com/JustVugg/colibri/issues/1518)** · 14× on DeepSeek-V4.1 from a call four engines already make, with raw per-run stderr and a validated run manifest.
- **[#1525 — host-RAM residency is worth 31% of a turn](https://github.com/JustVugg/colibri/issues/1525)** · the ceiling a V4.1 GPU expert tier would be chasing.
- **[#1523 — hypothesis #2 holds, and my first attempt said otherwise](https://github.com/JustVugg/colibri/issues/1523)** · the drive-split measurement behind PR #1522, including the negative first result and why it disagreed.

[View my colibri pull requests](https://github.com/JustVugg/colibri/pulls?q=is%3Apr+author%3ALiRunGuo) · [View my colibri reports](https://github.com/JustVugg/colibri/issues?q=is%3Aissue+author%3ALiRunGuo)
