---
title: "SGLang — Import-Time Robustness"
excerpt: "Contributor to SGLang: merged a fix so a failing host check inside the DeepEP import no longer kills every server process at startup, including dense models that never request DeepEP."
collection: portfolio
category: contribution
order: 5
permalink: /portfolio/sglang/
---

I contribute to [SGLang](https://github.com/sgl-project/sglang), focusing on startup and import-time failure modes — the ones that decide whether a server comes up at all, before any model code runs.

### Merged contribution

**[PR #40671 — Don't let a failed `deep_ep` import-time check kill servers that never use DeepEP](https://github.com/sgl-project/sglang/pull/40671)** · Merged October 6, 2026 (+105 / −4 across 3 files).

Since [#33932](https://github.com/sgl-project/sglang/pull/33932), `sgl-deep-ep` is a hard dependency of the `sglang` wheel, so `deep_ep` is always installed and its `__init__.py` runs host checks at import time: `check_nccl_so()` asserts that exactly one libnccl is mapped and that it is byte-identical to the pip NCCL, and `find_cuda_home()` asserts that a CUDA home is found. The guard in `token_dispatcher/deepep.py` dated from when "not installed" was the only failure mode and caught only `ImportError`, so any other failure propagated out of an import that every server process performs, including the launcher:

```text
launch_server -> http_server -> engine -> data_parallel_controller -> scheduler
  -> ... -> layers/moe/token_dispatcher -> deepep -> deep_ep
```

On an H200 running the dense Qwen3-1.7B with no DeepEP flags at all, a second libnccl on the host (`LD_PRELOAD` of a source-built NCCL 2.32.3 against the pip 2.29.7) was enough to take the server down at import:

```text
AssertionError: Duplicate NCCL runtime found in the current system:
  .../nccl/build/lib/libnccl.so.2.32.3 and .../site-packages/nvidia/nccl/lib/libnccl.so.2
```

The same happened on a shared network filesystem without `LD_PRELOAD`, where the detokenizer read the NCCL mapping as `(deleted)` and `filecmp` raised `FileNotFoundError`. Both modules now keep the original exception in their existing `_deepep_import_error` field and chain it into the `ImportError` raised only when a DeepEP dispatcher is actually constructed, and the message says "not available" rather than "not installed" — because the package usually is installed. Users who do request DeepEP still get a hard error, now with the root cause instead of a misleading one.

No forward, kernel or sampling code is touched; the functional check is a dense end-to-end run that previously exited during import and now starts and generates. A new CPU test runs a fresh interpreter with a stand-in `deep_ep` whose import raises `AssertionError`, checking that the dispatcher package imports, that both `use_deepep` flags stay false, and that constructing a DeepEP dispatcher raises an `ImportError` chained to the original error — it fails on `main` and passes with the fix. The existing DeepEP unit tests plus the new one are 39 passed, and `pre-commit` passes on the changed files.

[View my SGLang pull requests](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+author%3ALiRunGuo)
