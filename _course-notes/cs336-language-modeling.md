---
title: "CS336 Language Modeling from Scratch"
excerpt: "斯坦福 CS336 系统学习笔记，从分词、Transformer 与训练系统，到数据、扩展定律、对齐和模型评测。"
collection: course-notes
permalink: /course-notes/cs336-language-modeling
toc: true
toc_sticky: true
---
{% raw %}
*Stanford University, Spring 2026 offering (with material from the Spring 2024/2025 archives). Instructors: Percy Liang and Tatsunori Hashimoto. Compiled from the public course website (cs336.stanford.edu), the executable lecture files (`lecture_XX.py`), the lecture slide PDFs, and the assignment handouts.*

---

## Course Overview

**What is this course about?**

CS336 is a project-based course that walks students through the *entire process of developing a language model from scratch* — data collection and cleaning for pre-training, Transformer construction, model training, and evaluation — deliberately modeled on operating-systems courses that build an entire OS from scratch. There is essentially no scaffolding code: you will write an order of magnitude more code than in typical AI classes.

**The core philosophy — *understanding via building*.** The course argues that researchers have become disconnected from the technology they use: in 2016 researchers implemented and trained their own models; in 2018 they downloaded models (BERT) and fine-tuned them; today they prompt API models. Because these abstractions are "leaky," fundamental research requires tearing up the stack, so the course makes you rebuild it.

**The efficiency mindset.** The unifying lens of the whole course is:

> **accuracy = efficiency × resources**

and the framing question: *what is the best model you can build given a fixed compute and data budget?* Today we are compute-constrained, so design decisions reflect squeezing the most out of given hardware (tokenization, model architecture, data filtering, scaling laws are all justified through this lens). The course notes that "tomorrow, we will become data-constrained."

**Three kinds of knowledge the course can teach:**
- **Mechanics**: how things work (what a Transformer is, how model parallelism works) — these transfer.
- **Mindset**: squeezing the most out of hardware, taking scaling seriously — these transfer.
- **Intuitions**: which data/modeling decisions yield good accuracy — *do not* necessarily transfer across scales (many are just empirical folklore, e.g., Shazeer's paper that introduced SwiGLU includes a "divine benevolence" diagram).

**Prerequisites:**
- **Proficiency in Python and software engineering** (paramount — minimal scaffolding, huge code volume).
- **Deep learning & systems optimization experience** (strong PyTorch familiarity, memory hierarchy concepts).
- **College calculus and linear algebra** (MATH 51 / CME 100 level).
- **Basic probability and statistics** (CS 109 or equivalent).
- **Machine learning** (CS221/CS229/CS230/CS124/CS224N level).

**Logistics (Spring 2026):** Lectures Mon/Wed 3:00–4:20pm in Skilling Auditorium; 5 units; recorded on a private YouTube playlist (not publicly accessible — noted as restricted). All coursework submitted via Gradescope; 6 late days total, max 3 per assignment. Modal sponsors GPU compute for enrolled students.

**Grading / honor-code highlights:**
- Study groups allowed, but you must understand and complete your own work.
- **AI policy (Spring 2025–26)**: AI tools are permitted for *high-level conceptual questions* and *low-level API documentation* but are **not permitted for implementing any part of any assignment** — including coding agents (Cursor Agents, Codex, Claude Code) and AI autocomplete (Cursor Tab, GitHub Copilot). Each assignment repo contains a pedagogical `AGENTS.md` that coding agents auto-read; chat-interface users must paste it into each conversation. Rule of thumb: *"ask whether a CA would comply with your request in office hours."*
- Do not consult existing implementations ("existing code") for things you are asked to build; handouts are self-contained.
- Regrade requests within 3 days of grades.

**The five assignments (details in the Assignments Summary section):**
1. **Basics**: tokenizer, Transformer, cross-entropy loss, AdamW, training loop; train on TinyStories and OpenWebText; leaderboard.
2. **Systems**: benchmarking/profiling, activation checkpointing, a FlashAttention-2 Triton kernel, DDP, optimizer-state sharding, FSDP.
3. **Scaling**: query a training API to fit scaling laws and predict compute-optimal hyperparameters.
4. **Data**: convert Common Crawl HTML to text, filter (quality/toxicity/PII), deduplicate with MinHash.
5. **Alignment & Reasoning RL**: zero/few-shot + CoT prompting, GRPO, policy-gradient estimator variants; optional Part 2: SFT + DPO.

**How to use these notes.** Each lecture is a chapter following the same template: Overview → Core Concepts (with real-world analogies) → Code Examples with detailed commentary (What the Code Does / Implementation Deep Dive / Connection to Assignments) → Key Takeaways → Potential Pitfalls → Review Questions. Code-heavy lectures (1, 2, 6, 7, 10, 14) reverse-engineer the actual lecture code; slide-based lectures (3, 4, 5, 8, 9, 11, 15, 16) distill the slides.

---

## Lecture 1: Overview, Tokenization

*Date: Mon March 30 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_01.py` (executable lecture + code)*

### Overview

This lecture kicks off the course by explaining *why* CS336 exists, mapping the historical landscape of language models, and introducing the first technical unit: **tokenization** — the process of converting raw text (bytes) into the integer sequences a model actually consumes. It implements and compares character-level, byte-level, word-level, and **Byte-Pair Encoding (BPE)** tokenizers in live Python code, including a from-scratch BPE trainer.

### Core Concepts & Definitions

- **Language model**: a probability distribution over sequences of tokens. In 2018 a "language model" was something you fine-tuned (BERT); in 2020 something you prompted (GPT-3); in 2022 something you talked to (ChatGPT); in 2026 something that acts autonomously (agents). The fundamentals (attention, kernels, optimization) are the same; the specs differ (longer context, inference efficiency matters more).
- **Tokenizer**: a class with two methods — `encode(string) -> list[int]` and `decode(list[int]) -> string`. It is the interface between raw inputs (bytes) and the integers the model operates on.
  - *Analogy*: a chef prepping ingredients. You cannot cook with a whole unwashed vegetable; you wash, peel, and chop it into uniform pieces (tokens) that your recipe (model) can easily handle. Different chefs (tokenizers) chop differently, and the chopping determines how well the recipe works.
- **Unicode / code points**: raw text is a sequence of Unicode characters; each character has a code point (e.g., `ord("a") == 97`, `ord("🌍") == 127757`). `chr` goes back.
  - *Analogy*: a code point is like an ISBN — a globally unique number assigned to every "book" (character) in the library of human scripts.
- **UTF-8**: the dominant encoding that maps characters to 1–4 bytes. ASCII characters take 1 byte; `🌍` takes 4 bytes (`\xf0\x9f\x8c\x8d`).
- **Compression ratio**: the number of UTF-8 bytes per token. A higher ratio means shorter sequences — desirable because Transformer attention is quadratic in sequence length.
  - *Analogy*: compression ratio is like packing for a trip. One big suitcase per 10 shirts (high ratio) beats 10 small bags (low ratio) when your airline (attention) charges per bag (token).
- **Byte-Pair Encoding (BPE)**: a data-driven subword algorithm (originally 1994 data compression by Philip Gage; adapted to NLP by Sennrich et al. 2016; used by GPT-2 and almost every modern model). Start with bytes as tokens, then repeatedly merge the most frequent adjacent pair into a new token until you reach a target vocabulary size.
  - *Mechanism*: `count pairs → merge most common pair → repeat`. The result: common sequences of bytes become single tokens; rare sequences stay split into many tokens.
  - *Analogy*: BPE is like how a texting autocomplete learns "brb," "lol," "omw" as shortcuts because they appear constantly, while a rare word like "supercalifragilisticexpialidocious" stays spelled out letter-by-letter.
- **The efficiency lens on tokenization**:
  1. Reduce context length (~1000 bytes → ~250 tokens).
  2. Adaptive computation: allocate more model capacity to "interesting" parts of the input.
- **Tokenizer-free architectures** (the dream): ByteT5, MegaByte, BLT, and others operate directly on bytes — promising but not yet scaled to the frontier.

### Code Example: The `Tokenizer` interface and three naive tokenizers

The lecture defines an abstract interface plus three concrete (suboptimal) tokenizers.

**Code (Python):**
```python
from abc import ABC

class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError

class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))
    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))

class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        indices = list(map(int, string_bytes))
        return indices
    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        string = string_bytes.decode("utf-8")
        return string
```

**What the Code Does:**
1. `CharacterTokenizer.encode` maps each Unicode character to its code point via `ord`, producing one integer per character; `decode` maps back via `chr`.
2. `ByteTokenizer.encode` first encodes the whole string to UTF-8 bytes, then converts each byte (0–255) to an int; `decode` converts ints back to bytes and decodes UTF-8.
3. Both round-trip: `decode(encode(s)) == s`.

**Implementation Deep Dive:**
- **Why these are the "worst of both worlds":** The character tokenizer has a huge vocabulary (≈150K Unicode characters, most rare) *and* a compression ratio of ~1; the byte tokenizer has a tiny vocabulary (256) but a compression ratio of exactly 1 (one token per byte), producing long sequences that blow up attention cost. The lecture demonstrates this numerically: for `"Hello, 🌍! 你好!"`, the byte tokenizer gives a compression ratio of 1.0.
- **Why UTF-8 matters:** not all characters fit in one byte; `bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"`. Handling this correctly is the difference between a working and a broken tokenizer.
- **Why decode must handle invalid bytes:** `bytes.decode("utf-8")` raises on malformed sequences; production tokenizers use `errors="replace"` (as `output_tokenizer` does) to survive arbitrary byte strings.

**Connection to Assignments:** Assignment 1 asks you to implement a full BPE tokenizer. The `Tokenizer` abstract class here is exactly the interface your `BPETokenizer` must satisfy, and the UTF-8 handling is the foundation of the byte-level BPE you'll build (with pre-tokenization, special tokens, and a fast merge implementation).

### Code Example: BPE merge and full tokenizer

**Code (Python):**
```python
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def count_adjacent_pairs(indices: list[int]) -> dict[tuple[int, int], int]:
    """Return a dictionary mapping each adjacent pair of tokens to its count."""
    counts = defaultdict(int)
    for index1, index2 in zip(indices, indices[1:]):
        counts[(index1, index2)] += 1
    return counts

def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    indices = list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}          # pair -> merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        counts = count_adjacent_pairs(indices)       # count adjacent pairs
        pair = max(counts, key=counts.get)           # most frequent pair
        new_index = 256 + i                          # fresh index
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]  # concatenate bytes
        indices = merge(indices, pair, new_index)
    return BPETokenizerParams(vocab=vocab, merges=merges)

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]                # index -> bytes
    merges: dict[tuple[int, int], int]     # (i1, i2) -> new_index

class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        return b"".join(bytes_list).decode("utf-8")
```

**What the Code Does:**
1. `train_bpe` starts with the byte sequence of the training string, then for `num_merges` iterations: counts all adjacent pairs, picks the most frequent, assigns it a brand-new index (`256 + i`), records the merge rule, concatenates the two byte sequences in the vocabulary, and rewrites the sequence via `merge`.
2. `BPETokenizer.encode` applies every learned merge rule in order to a new input's byte sequence.
3. `BPETokenizer.decode` looks up each index's byte sequence in the vocab and concatenates + UTF-8 decodes.

**Implementation Deep Dive:**
- **Why `256 + i`:** the first 256 indices are reserved for single bytes; every new merged token gets the next available index, so the mapping stays injective.
- **Why merges are applied in training order:** BPE is a greedy hierarchical process; applying merges in the order they were learned guarantees the encoded result matches what training implied (a later merge may contain an earlier merged token).
- **Why `decode` works without the merge rules:** the vocabulary stores the full byte string for every token, so decoding is a pure lookup — merge rules are only needed for encoding. This is a clean separation between the *training-time* data structure (merges) and the *inference-time* data structure (vocab).
- **Complexity caveat (flagged in the code):** `encode` loops over *all* merges for every tokenization — O(num_merges × length). Assignment 1 asks you to only apply merges that matter (e.g., skip merges that no longer appear in the sequence), to handle special tokens (e.g., `<|endoftext|>`), to add GPT-2-style pre-tokenization with a regex, and to make everything fast.
- **Why pre-tokenization and end-of-word markers matter:** the classic Sennrich BPE (and the prompt's example) splits text into words first and uses `</w>` to prevent cross-word merges; the lecture's byte-level version sidesteps this by merging raw bytes, which is simpler and matches GPT-style tokenizers.

**Connection to Assignments:** This is the absolute basis of Assignment 1, Section 2 ("Byte-pair encoding tokenizer"). You will extend exactly this logic: learn merges on a large corpus, build a vocabulary with special tokens, pre-tokenize (GPT-2 regex), and implement an efficient encoder. The lecture explicitly lists the four upgrades Assignment 1 requires: (1) only loop over merges that matter, (2) special tokens, (3) pre-tokenization, (4) speed.

### Key Takeaways

1. Tokenization is the critical first step of any LM pipeline: it determines vocabulary size, sequence length, and how rare words are handled, and it must round-trip perfectly (`decode(encode(s)) == s`).
2. Character/byte/word tokenizers are all suboptimal: characters give huge vocabularies and low compression; bytes give tiny vocabularies but compression ratio 1; words give good compression but unbounded vocabularies and out-of-vocabulary (UNK) problems.
3. BPE is a simple, data-driven, widely-used heuristic: start with bytes, repeatedly merge the most frequent adjacent pair. It balances vocabulary size and compression ratio.
4. Everything in this course is about **efficiency**: tokenization is justified because it shortens sequences (quadratic attention!) and allocates capacity adaptively.
5. A good tokenization scheme should (i) let the model operate on meaningful chunks and (ii) make chunks variable so more capacity goes to interesting parts of the input.

### Potential Pitfalls

- **Not handling Unicode properly**: multi-byte characters must be encoded/decoded as UTF-8 bytes; forgetting this breaks round-tripping for non-ASCII text.
- **Forgetting round-trip testing**: always assert `decode(encode(s)) == s`, including for emoji, CJK, and edge-case whitespace.
- **Applying merges out of order**: encoding must apply merges in training order, or the result won't match the learned vocabulary.
- **O(n²)-ish encoding**: naive `encode` (loop over all merges × full sequence) is far too slow for real corpora; Assignment 1 expects a smarter approach.
- **Byte decoding failures**: token sequences may not form valid UTF-8; use `errors="replace"` in production.
- **Ignoring compression ratio**: a tokenizer with ratio 1.0 makes sequences long, and attention is quadratic — sequence length is a first-order cost.

### Review Questions

1. **Q:** Why is the byte tokenizer's compression ratio exactly 1, and why is that a problem?
   - **A:** Every byte maps to exactly one token, so bytes/token = 1. It means a 1000-byte document becomes 1000 tokens; since attention is quadratic in sequence length, the model's cost balloons — you want ~4x compression to ~250 tokens instead.
2. **Q:** In `train_bpe`, why must `decode` be implemented with the vocabulary rather than the merge rules?
   - **A:** The vocabulary maps every token index to the full concatenated byte string it represents, so decoding is a direct lookup. The merge rules describe how tokens were *built*, which is only needed for encoding new text. Also, decoding from the vocab handles arbitrary token sequences without needing to know merge history.
3. **Q:** What happens to BPE if you never merge the pair `(end-of-word marker, first character of next word)` — i.e., if merges are computed over raw bytes with no word boundary information?
   - **A:** The tokenizer may learn tokens that span word boundaries, making tokens less semantically meaningful and potentially mixing contexts; pre-tokenization + boundary markers are used to prevent this.
## Lecture 2: PyTorch, Resource Accounting (FLOPs, Memory, Arithmetic Intensity)

*Date: Wed April 1 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_02.py`*

### Overview

This lecture is the "systems mindset" lecture: before you can train the best model given fixed resources, you must be able to *account* for the resources — memory and compute — that a computation consumes. It covers PyTorch tensor basics (dtypes, memory layout), the `einops` notation for readable tensor manipulation, FLOP counting, and the key concept of **arithmetic intensity / roofline analysis** (are you compute-bound or memory-bound?), then applies this accounting to a training loop, gradient accumulation, and activation checkpointing.

### Core Concepts & Definitions

- **Tensor**: the basic storage unit for everything in deep learning — data, parameters, gradients, optimizer states, activations. Has a *rank* (number of dimensions); Transformers use rank-4 tensors of shape (B=32, S=16, H=16, D=64) for batched multi-head activations.
  - *Analogy*: a tensor is a spreadsheet with a fixed number of axes; a rank-4 tensor is like a warehouse of spreadsheets organized by batch, position, head, and feature.
- **dtypes and their tradeoffs**: fp32 (4 bytes, default), fp16 (2 bytes, but small dynamic range — `torch.tensor([1e-8], dtype=torch.float16)` underflows to 0!), **bf16** (2 bytes, same dynamic range as fp32 but worse resolution — no underflow at 1e-8), fp8 (E4M3/E5M2, used on H100), fp4 (NVFP4, 4 bits, Blackwell).
  - *Analogy*: fp16 is like a ruler marked only in centimeters — cheap, but you can't measure a hair; bf16 is a ruler marked in centimeters *and* with a huge range but only every other tick precise; fp32 is a fine micrometer. Training in fp16/bf16 risks "instability" when small values collapse to zero.
- **Mixed precision training**: use bf16 for parameters/activations/gradients, fp32 for optimizer states (accumulating sums across many steps needs precision). PyTorch's `torch.amp.autocast` automates this.
- **FLOPs vs FLOP/s** (confusing homophones): FLOPs = floating-point operations (a measure of work done, e.g., 3.14e23 for GPT-3); FLOP/s (FLOPS) = operations per second (hardware speed). 
- **MFU (Model FLOPs Utilization)**: actual FLOP/s ÷ promised (peak) FLOP/s. ≥0.5 is quite good; it's never 1 because of memory bandwidth, imperfect kernels, etc.
- **Arithmetic intensity**: FLOPs ÷ bytes moved for a given computation. **Accelerator intensity**: peak FLOP/s ÷ memory bandwidth for the hardware (H100: ~295 FLOP/byte). If a workload's arithmetic intensity < accelerator intensity → **memory-bound**; if greater → **compute-bound**.
  - *Analogy*: a factory (compute unit) fed by a single conveyor belt (memory bandwidth). If the belt can't deliver parts fast enough, the factory idles — you're "belt-bound" (memory-bound). If the belt over-delivers but the factory is slow, you're "factory-bound" (compute-bound).
- **The 6ND rule**: training a model with N parameters on D tokens takes ≈ 6·N·D FLOPs (2ND forward, 4ND backward). Backward is 2× forward per layer.
- **Roofline model**: visualize arithmetic intensity (x) vs achieved FLOP/s (y); the kink is the accelerator intensity; MFU = min(1, arithmetic-intensity / accelerator-intensity).
- **Gradient accumulation**: to use a large logical batch without the memory of a large batch, compute gradients on micro-batches and accumulate (don't zero) before stepping.
- **Activation checkpointing (gradient checkpointing / rematerialization)**: store activations only at a subset of layers; recompute the rest during backward. Memory-compute tradeoff: storing every layer is O(L) memory with no recompute; storing none is O(1) memory but O(L²) compute; storing every √L layers gives O(√L) memory with O(L) recompute.

### Code Example: einops for readable tensor math

**Code (Python):**
```python
from einops import rearrange, einsum, reduce
import torch

x = torch.ones(2, 3, 4)  # batch seq hidden
y = torch.ones(2, 3, 4)  # batch seq hidden

# Old way (easy to mess up -2, -1):
z = x @ y.transpose(-2, -1)  # batch seq seq

# New (einops) way:
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")

# Or use '...' to broadcast over any number of leading dims:
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")

# Reduce (sum over last dim):
y_sum = reduce(x, "... hidden -> ...", "sum")

# Rearrange: split a flattened dim into (heads, hidden1)
w = torch.ones(4, 4)
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

**What the Code Does:**
1. `einsum` performs generalized matrix multiplication with *named dimensions*; dimensions named in both operands but not in the output are summed over (contracted). The first example computes `x @ yᵀ` per batch element (a batched attention-score pattern).
2. `reduce` aggregates one tensor (sum/mean/max/min) over named dims.
3. `rearrange` reshapes without changing data, including splitting/merging parenthesized dimensions (`(heads hidden1)`).

**Implementation Deep Dive:**
- **Why named dimensions:** `x @ y.transpose(-2, -1)` is opaque — which dims are contracted? einops makes the contract explicit in a string and catches shape errors at runtime, which the lecture argues is "easy to mess up" otherwise. In production code, plain `torch.matmul`/`torch.bmm` is often faster than einops' einsum; use einops for clarity in prototypes, and note Assignment 1 explicitly shows einops-style notation for the Transformer forward pass.
- **Why `...` (ellipsis):** it lets the same expression work with or without a batch dim, broadcasting over any number of leading dimensions — handy when writing dimension-agnostic layers.

**Connection to Assignments:** Assignment 1's Transformer forward pass is expressed in exactly this notation (the handout shows einops formulations for attention and RMSNorm). Assignment 2's FlashAttention kernel work involves thinking in named dimensions (B, H, S, D) to tile correctly.

### Code Example: FLOP counting for a linear layer and the training step

**Code (Python):**
```python
B, D, K = 1024, 256, 64   # batch, in-dim, out-dim
x = torch.ones(B, D)
w = torch.randn(D, K)
y = x @ w

# One multiply and one add per (i, j, k) triple:
actual_num_flops = 2 * B * D * K

# 2-layer MLP:
# forward:  h1 = x @ w1          -> 2*B*D*D FLOPs
#           h2 = h1 @ w2         -> 2*B*D*D FLOPs
# backward: h1.grad = h2.grad @ w2^T  -> 2*B*D*D
#           w2.grad = h2.grad^T @ h1   -> 2*B*D*D
# total per layer: 2 (fwd) + 4 (bwd) = 6 * B * D * D
```

**What the Code Does:**
- Counts FLOPs for matmuls (`2 * M * N * K`), then shows that the backward pass for a layer costs exactly 2× the forward (two matmuls: one for the input gradient, one for the weight gradient), giving the famous **6ND** rule.

**Implementation Deep Dive:**
- **Why 2×B×D×K:** each output element is a dot product of length K: K multiplies + (K−1) adds ≈ 2K FLOPs, times B×D... (precisely, B·D·K outputs × 2K FLOPs each → 2·B·D·K... the lecture's convention counts ~2 FLOPs per multiply-accumulate, i.e., 2·B·D·K for an output of size B×K... note in the lecture, actual_num_flops = 2*B*D*K counts one multiply + one add per (i,j,k) triple.) The important takeaway is the *ratio*: backward = 2× forward.
- **Why this is an approximation for Transformers:** the 6ND rule is exact for MLPs and a good approximation for Transformers at short context lengths (attention adds a small term that matters at long context).
- **Why you should benchmark rather than trust the spec:** peak FLOP/s depends on dtype and sparsity (H100: 1979 teraFLOP/s with sparsity, half without); achieved throughput is measured by timing and dividing FLOPs by time, giving MFU.

**Connection to Assignments:** Assignment 1's **resource accounting** section asks you to compute the FLOPs of every Transformer component (embedding, attention QK^T, softmax, attention·V, MLP up/gate/down projections, LM head) for a given config — exactly this 2·M·N·K counting. The memory accounting (2 bytes params + 2 bytes grads + 8 bytes AdamW optimizer state = 12 bytes/param in bf16) is also Assignment 1 material and is the basis for Assignment 2's distributed memory planning.

### Code Example: AdaGrad from scratch and the training loop

**Code (Python):**
```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data
                g2 = state.get("g2", torch.zeros_like(grad))  # sum of squared grads
                g2 += torch.square(grad)
                state["g2"] = g2
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)

# The canonical training loop:
for t in range(num_train_steps):
    x, y = get_batch()
    pred_y = model(x).mean()
    loss = F.mse_loss(pred_y, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

**What the Code Does:**
1. `AdaGrad` maintains a per-parameter running sum of squared gradients `g2`, and divides the gradient by `sqrt(g2 + eps)` — larger past gradients shrink the effective step (adaptive per-coordinate learning rates).
2. The training loop performs the canonical cycle: sample batch → forward → loss → backward → optimizer step → zero gradients.

**Implementation Deep Dive:**
- **Why write your own optimizer:** the lecture walks through AdaGrad as a warm-up for the optimizer-family ladder: momentum = SGD + exponential averaging of grad; AdaGrad = SGD + averaging by grad²; RMSProp = AdaGrad with *exponential* averaging of grad²; Adam = RMSProp + momentum. Assignment 1 requires implementing **AdamW from scratch** — you will write essentially this class with first and second moment estimates plus weight decay.
- **Why fp32 optimizer state:** "It is customary to use fp32 for stability (accumulating averages over powers over many steps)" — hence mixed precision: bf16 params/grads, fp32 optimizer state (8 bytes/param for Adam's two moments; 4 bytes for AdaGrad's one).
- **Why `zero_grad(set_to_none=True)`:** setting to None frees memory faster than zeroing.

**Connection to Assignments:** Assignment 1: implement the cross-entropy loss, AdamW, and the training loop with checkpoint save/load — the exact pattern above. The memory table (params 2B + grads 2B + optimizer state 4–8B + activations) is what you'll report in resource accounting; Assignment 2 builds distributed versions (DDP/FSDP) on top of this loop.

### Code Example: gradient accumulation and activation checkpointing

**Code (Python):**
```python
# Gradient accumulation: use micro-batches of 256 to emulate batch 4096
for micro_step in range(accumulation_steps):
    x, y = get_micro_batch()
    loss = loss_fn(model(x), y) / accumulation_steps  # scale loss
    loss.backward()          # accumulates into .grad (no zero_grad)
optimizer.step()             # one update for the whole logical batch
optimizer.zero_grad(set_to_none=True)

# Activation checkpointing:
for layer in self.layers:
    x = torch.utils.checkpoint.checkpoint(layer, x)  # recompute in backward
```

**What the Code Does:** The first block amortizes one optimizer step over many micro-batches, trading memory (activations for one micro-batch) for the same gradient statistics as a big batch. The second wraps each layer in `torch.utils.checkpoint.checkpoint`, which discards intermediate activations in forward and recomputes them in backward.

**Implementation Deep Dive:**
- **Why gradient accumulation:** activation memory scales with batch size; a logical batch of 64×1024 dims × 16 layers needs 2·64·1024·16 bytes of activations, which can blow up GPU memory. Micro-batches of 256 shrink this 4×.
- **Why checkpointing trades memory for compute:** full storage is O(L) memory with 0 recompute; no storage is O(1) memory but O(L²) compute (recompute from the start for each layer); √L spacing balances both. Note: the loss should be divided by accumulation steps so the mean over the logical batch is correct.
- **Why it matters for assignment scale:** without these tricks you cannot fit the batch sizes that make distributed training (Assignment 2) worthwhile.

**Connection to Assignments:** Assignment 2's **activation checkpointing** task (implement it for TransformerBlocks and verify with memory hooks) is a direct descendant of this example — the handout even shows the same `pack_hook/unpack_hook` instrumentation to measure saved-tensor memory.

### Key Takeaways

1. Everything is tensors: parameters, gradients, activations, optimizer states — and each has a memory cost = numel × element_size (bf16: 2B, fp32: 4B).
2. The 6ND rule: training costs ~6 FLOPs per parameter per token (2 forward + 4 backward); backward is 2× forward.
3. Use the roofline model: matmuls are compute-bound (arithmetic intensity ~n/3), elementwise ops (ReLU/GELU) and dot products are memory-bound — so "ReLU is not faster than GELU" in isolation, and inference (matrix-vector) is memory-bound.
4. Memory tricks matter: mixed precision (bf16 + fp32 optimizer states), gradient accumulation, and activation checkpointing let you fit bigger batches/models.
5. einops makes tensor math readable and debuggable; MFU ≥ 0.5 is good, and always benchmark with `torch.cuda.synchronize()` around timing.

### Potential Pitfalls

- **fp16 underflow**: values like 1e-8 collapse to 0, causing instability; prefer bf16 or fp32 accumulation.
- **Forgetting `torch.cuda.synchronize()` when benchmarking**: CUDA is asynchronous; timings without sync measure launch overhead, not kernel time. Use CUDA events.
- **Confusing FLOPs (work) with FLOP/s (speed)**: they're pronounced the same but are different quantities; also note peak FLOP/s depends on dtype and sparsity.
- **Not scaling loss in gradient accumulation**: forgetting to divide by accumulation steps changes the effective learning rate.
- **Memory accounting that ignores activations**: the "largest trainable model on 8 H100s" napkin math is an *upper bound* — activations depend on batch size and sequence length and can dominate.
- **Using `-2, -1` transposes blindly**: dimension confusion in attention/MLP code is a classic bug source; name your dims (einops) or comment shapes.

### Review Questions

1. **Q:** A GPU has peak 1000 TFLOP/s and 3.35 TB/s bandwidth. A workload moves 1 GB and does 1 TFLOP. Is it memory-bound or compute-bound?
   - **A:** Accelerator intensity ≈ 1000e12 / 3.35e12 ≈ 298 FLOP/byte. Workload intensity = 1e12 / 1e9 = 1000 FLOP/byte > 298 → compute-bound.
2. **Q:** Why does the backward pass cost 2× the forward pass for a linear layer?
   - **A:** You must compute two gradients: one for the input (needed by earlier layers) and one for the weight — each is a matmul of similar FLOPs to the forward matmul, so forward (1 matmul) + backward (2 matmuls) = 3 matmuls ≈ 6·B·D² per layer, i.e., 2ND forward + 4ND backward.
3. **Q:** Why does inference have low arithmetic intensity while training doesn't?
   - **A:** Training processes large batched matmuls (B≫1, compute-bound). Inference decodes one token at a time (B=1): each step reads all parameters (matrix-vector product) with intensity ≈ 1, far below the accelerator intensity — hence memory-bound.
## Lecture 3: Architectures and Hyperparameters

*Date: Mon April 6 (Spring 2026) | Instructor: Tatsu Hashimoto | Materials: `lecture_03.pdf`*

### Overview

This lecture answers: *what do the large LMs have in common, and what varies?* It reviews the choices in the "original" Transformer (post-norm LayerNorm, sinusoidal position embeddings, ReLU FFN) versus the "simple, modern variant" (pre-norm, RoPE, SwiGLU, no biases), then surveys the empirical consensus across dozens of released models on normalization, activations, position embeddings, and hyperparameters (FFN ratio, head dims, aspect ratio, vocabulary size, regularization), plus stability tricks (z-loss, QK-norm, logit soft-capping).

### Core Concepts & Definitions

- **Pre-norm vs post-norm**: put LayerNorm *before* the sublayers (pre-norm) so it doesn't sit in the main residual signal path. Almost all modern LMs are pre-norm (BERT was post-norm); OPT-350M is a funny exception.
  - *Analogy*: pre-norm is like a water filter at the tap (treating water as it's used, leaving the pipes clean); post-norm is like a filter at the reservoir that everything must pass through on the way back.
  - *Why*: better gradient propagation, fewer gradient spikes, stability at scale, enables larger learning rates; the original motivation was removing warmup.
- **LayerNorm vs RMSNorm**: LayerNorm subtracts mean and divides by variance across the hidden dim; RMSNorm only rescales by RMS (`y = x / sqrt(mean(x²)+ε) * γ`), with no mean subtraction or bias. GPT-3/OPT/GPT-J use LayerNorm; LLaMA/PaLM/Chinchilla/T5 use RMSNorm.
  - *Why RMSNorm*: fewer ops (no mean), fewer parameters (no bias); FLOPs are tiny either way but **FLOPs ≠ runtime** — data movement matters, and RMSNorm moves fewer bytes, giving wall-clock gains (Narang et al. 2020).
- **Gated linear units (*GLU)**: replace `FFN(x) = activation(xW1) W2` with a gated variant like `SwiGLU(x) = (swish(xW1) ⊙ (xV)) W2`, adding a gate projection V. GeGLU (gaussian error gating) and SwiGLU (swish gating) are the standard choices; gated FFNs use ~2/3 the FFN dim. Evidence: consistent gains (Shazeer 2020; Narang et al. 2020). Most models post-2023 use SwiGLU.
  - *Analogy*: a gated unit is a bouncer at a club: one projection decides *how much* of the "content" projection gets in, element-wise.
- **Serial vs parallel layers**: standard blocks run attention then MLP serially; "parallel" blocks (GPT-J, PaLM) compute them side-by-side and sum. Parallel layers can share a LayerNorm and fuse matmuls, but most modern models are serial.
- **Position embeddings**: sinusoidal (add sines/cosines; original Transformer), absolute learned (GPT-1/2/3, OPT), relative (T5, Gopher), and **RoPE** (rotary, GPT-J/PaLM/LLaMA and most 2024+ models).
- **RoPE (Rotary Position Embeddings)**: rotate pairs of coordinates of the query/key by an angle proportional to position so that the attention score depends only on the *relative* position: `⟨f(x,i), f(y,j)⟩ = g(x, y, i−j)`.
  - *Analogy*: think of each token's vector as a clock hand; RoPE rotates the hand by an angle that encodes the token's position. The dot product between two rotated hands depends only on the *angle difference* (relative position), not the absolute clock setting.
- **Hyperparameter consensus**: FFN dim ≈ 4× model dim (8/3× for GLU variants — most GLUs use 2.5–2.7×); head_dim × num_heads ≈ model_dim (ratios cluster around 1); aspect ratio model_dim/layers ≈ 100–200; vocab sizes 30–50K monolingual, 100–250K multilingual.
- **Regularization**: newer models drop dropout during pre-training (too much data; single pass; hard to memorize) and rely on weight decay — which in LLMs is less about overfitting and more about interaction with the learning-rate schedule (Andriushchenko et al. 2023).
- **Stability tricks**:
  - **z-loss**: add a small term to the logits to penalize huge log-sum-exp values (prevents logit drift); used by PaLM, Baichuan 2, DCLM, OLMo 2/3.
  - **QK-norm**: normalize query and key vectors (RMSNorm/LayerNorm) before the attention softmax to keep attention logits bounded; used by DCLM, OLMo 2, Gemma 2, Qwen 3, Chameleon.
  - **Logit soft-capping**: `tanh`-cap logits to a max value.
- **GQA/MQA recap**: reduce the number of key/value heads to cut KV-cache memory and improve inference (details in Lecture 10).
- **Interleaved attention**: e.g., Cohere Command A uses full attention every 4th layer with local attention elsewhere; LLaMA 4, Gemma 3/4, OLMo 3 interleave SWA + full attention.

### Code Example: RMSNorm (the Assignment 1 formulation)

The lecture slides define what you implement; the Assignment 1 handout formalizes it. Given an activation vector `a ∈ R^d_model`, RMSNorm rescales each activation:

**Code (Python):**
```python
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learnable per-dim gain γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**What the Code Does:** computes the root-mean-square of the last dimension, divides by it (plus ε), and scales by a learnable gain. No mean subtraction, no bias — exactly the modern recipe.

**Implementation Deep Dive:**
- **Why no mean/bias:** fewer operations and fewer parameters to move; empirically "as good" as LayerNorm while saving wall-clock time (FLOPs are dominated by matmuls, but *data movement* isn't).
- **Why ε:** numerical stability when the RMS is tiny.
- **Why it's part of the "modern variant" checklist:** pre-norm placement + RMSNorm + RoPE + SwiGLU + no biases is what you implement in Assignment 1 — the lecture explicitly asks "Why did we pick these? What should you pick?" (the answer: empirical consensus + stability + efficiency).

**Connection to Assignments:** Assignment 1 Section 3: implement `RMSNorm` with exactly this interface (also used inside your attention and FFN blocks). Assignment 2's **fused RMSNorm Triton kernel** re-implements this operation as a single GPU kernel — the layer-norm is a memory-bound elementwise op that benefits from fusion.

### Code Example: RoPE (conceptual implementation)

**Code (Python):**
```python
def precompute_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    # positions: [seq_len]
    positions = torch.arange(seq_len)
    # frequencies: [head_dim // 2]  (geometric sequence)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))
    # angles: [seq_len, head_dim // 2]
    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin  # precompute once, reuse across layers/heads

def apply_rope(x, cos, sin):
    # x: [..., seq, head_dim]; rotate pairs of coordinates
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # complex-number-style rotation: (x1 + i x2) * (cos + i sin)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

**What the Code Does:** builds a cache of cos/sin angles for every position (using the classic geometric frequency schedule), then rotates each pair of coordinates `(x1, x2)` of the query/key by the position-dependent angle — the complex multiplication `(x1 + i·x2)(cos + i·sin)`.

**Implementation Deep Dive:**
- **Why rotate pairs:** pairing coordinates 2-D at a time is the standard RoPE construction (motivated by complex numbers); Gemma 4's variant only rotates the first 2 coordinates. The key property: `⟨RoPE(x,i), RoPE(y,j)⟩ = g(x,y,i−j)` — the inner product (attention score) depends only on relative position, with *no cross terms* (unlike additive sinusoidal embeddings).
- **Why precompute:** the cos/sin tables depend only on (seq_len, head_dim) — compute once, index during each attention call.
- **Why it matters for GQA/MLA:** RoPE conflicts with MLA-style KV compression (Lecture 4), so DeepSeek adds a few non-rotated "latent" key dims — a subtle implementation detail you should know about even if you only implement plain RoPE.

**Connection to Assignments:** Assignment 1 Section 3: implement RoPE and apply it to Q and K inside multi-head attention (before or as part of the attention score computation). Getting the rotation axes right (per-head, per-position) is a classic debugging spot.

### Key Takeaways

1. Architecture consensus: pre-norm (non-residual), RMSNorm (or LayerNorm), SwiGLU gating, serial layers, no bias terms, RoPE — this "modern variant" is what you build in Assignment 1.
2. FLOPS ≠ runtime: memory movement matters; dropping biases and using RMSNorm saves wall-clock even if FLOPs are dominated by matmuls.
3. Hyperparameter consensus: FFN ≈ 4× model dim (8/3× for GLUs), head-dim×heads ≈ model dim, aspect ratio 100–200, vocab 30–50K (monolingual).
4. Regularization at scale is mostly about optimization dynamics, not overfitting (dropout goes away; weight decay interacts with the LR schedule).
5. Stability tricks (z-loss, QK-norm, soft-capping) exist because softmaxes with exponentials are where training blows up.

### Potential Pitfalls

- **Post-norm with aggressive scaling**: post-norm puts normalization in the residual path and is prone to instability at scale; modern practice is pre-norm (possibly with an extra non-residual post-norm, as in Grok/Gemma 2/OLMo 2).
- **Wrong FFN ratio for GLUs**: a SwiGLU FFN at 4× model dim has ~1.5× too many parameters; use ~8/3×.
- **RoPE applied to the wrong dims / forgetting per-head rotation**: attention silently degrades; verify against a reference.
- **Copying vocab sizes blindly**: 32K vocab works for English but multilingual models need 100–250K.
- **Ignoring softmax stability**: without z-loss/QK-norm, logits can drift and training diverges on large runs.

### Review Questions

1. **Q:** Why do modern models use pre-norm instead of post-norm?
   - **A:** Pre-norm keeps LayerNorm out of the main residual path, improving gradient flow and reducing spikes; this enables stability and larger learning rates at scale (the original stated benefit was removing LR warmup).
2. **Q:** A model has model_dim=4096 and uses SwiGLU. Roughly what FFN dim should it use, and why?
   - **A:** ≈ 8/3 × 4096 ≈ 10,923 (models use ~2.5–2.7×). Gated variants need only 2/3 the FFN dim of a ReLU FFN because the gate adds parameters while improving expressivity.
3. **Q:** What property makes RoPE "relative" even though positions are absolute indices?
   - **A:** Rotation makes the inner product of two position-encoded vectors depend only on the angle *difference*, i.e., `⟨f(x,i), f(y,j)⟩ = g(x,y,i−j)` — the absolute position cancels out.
## Lecture 4: Attention Alternatives and Mixtures of Experts

*Date: Wed April 8 (Spring 2026) | Instructor: Tatsu Hashimoto | Materials: `lecture_04.pdf`*

### Overview

Attention is quadratic in sequence length, which becomes the dominant cost at long context. This lecture covers two families of remedies: (1) **attention alternatives** — linear attention, state-space hybrids (Mamba-2, Gated DeltaNet), and sparse attention (DSA) — and (2) **Mixtures of Experts (MoE)** — replacing the big FFN with many expert FFNs and a router, which decouples parameter count from per-token FLOPs. It walks through routing methods, training objectives (auxiliary balancing losses), systems considerations, upcycling, and the DeepSeek MoE v1→v3 lineage.

### Core Concepts & Definitions

- **The cost of attention**: `Attn(Q,K,V) = softmax(QKᵀ)V` costs O(n²·d_k) for the QKᵀ product and O(n²·d_v) for the attention·V product. As context windows grow, this dominates.
  - *Analogy*: full attention is like every person at a party talking to every other person — n² conversations. Linear attention is like everyone whispering one summary to a scribe who broadcasts it — 2·n·d conversations.
- **Linear attention**: if the softmax is replaced by the identity kernel, `QKᵀV = Q(KᵀV)`, reducing cost from O(n²d) to O(2·n·d_v·d_k) — linear in sequence length.
  - *Recurrent form*: `S_t = S_{t−1} + k_t v_tᵀ`, `y_t = q_tᵀ S_t` — this looks exactly like an RNN! The *duality* lets you train with the parallel (quadratic-form) computation and infer with the serial (linear) form. Weighting `S_{t−1}` by γ gives RetNet.
  - *Analogy*: a student who takes perfect notes each class (state S_t) can answer questions instantly at any time — they don't need to re-read all previous lectures (the whole history) for each new question.
- **Mamba-2**: adds per-position gating: `S_t = γ_t S_{t−1} + k_t v_tᵀ` with `γ_t = f(x_t)` — gating makes linear attention more expressive while keeping the duality.
- **Gated Delta Net (GDN)**: further adds an input gate and selective state erasure: `S_t = γ_t(I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ`. The `β=0` case is "no input operation"; the erasure term forgets anything in the direction of the current key. Related to fast weight programmers / test-time training. Used in 3:1 GDN/attention hybrids (Qwen 3.5/Qwen Next).
- **Hybrids**: instead of all-attention or all-linear, interleave: Minimax M1 (7:1 linear:full), Nemotron 3 (3:1 Mamba:attention), Qwen 3.5 (3:1 GDN:attention). Controlled ablations show low losses at small hybrid ratios, with big inference wins (constant-memory state vs O(n) KV cache).
- **Sparse adaptation (DSA — DeepSeek Sparse Attention)**: instead of attending to all tokens, a lightweight *indexer* selects top-k tokens to attend to; can be applied post-hoc after dense pretraining (DeepSeek v3.2, GLM-5).
- **Mixture of Experts (MoE)**: replace one big FFN with many FFNs ("experts") plus a router that picks top-k experts per token. Total parameters grow with #experts, but FLOPs stay roughly constant (only k experts active per token).
  - *Analogy*: instead of one generalist doctor for every patient, a clinic with 256 specialists and a triage nurse (router) who sends each patient to the 2–8 most relevant specialists. The clinic has lots of staff (parameters) but each patient only sees a few (FLOPs).
  - *Why MoEs are popular*: same FLOPs, more params → better loss; faster to train per unit compute; competitive with dense models; parallelizable across devices (each expert can live on its own device). Examples: Mixtral (8 experts, top-2), Grok, DBRX (16, top-4), Qwen (60, top-4), DeepSeek v3 (256 experts, 8 active + 1 shared), Llama 4 Maverick (128 routed, top-1 + shared).
- **Routing methods**: token-choice top-k (dominant: Switch k=1, GShard/Grok/Mixtral k=2, Qwen/DBRX k=4, DeepSeek k=7-8), hashing (baseline), RL-learned routing (early work, Bengio 2013), linear-assignment routing (Clark '22).
  - *Router flavors*: logistic-regressor gates (DeepSeek v1-2, Grok, Qwen) vs. softmax-over-top-k (Mixtral, DBRX, DeepSeek v3).
  - *Recent variations*: fine-grained experts (many small experts) + a few always-on shared experts (DeepSeek, Qwen; originally DeepSpeed MoE). Ablations generally show fine-grained experts help; shared-experts gains are mixed (OlMoE found none).
- **Training MoEs — the differentiability problem**: sparse routing decisions are not differentiable. Options: (1) RL to optimize gating (works but high variance — not widely used), (2) stochastic perturbations (Gaussian noise, Shazeer 2017; multiplicative jitter, Switch Transformer), (3) **heuristic balancing/auxiliary losses** (the practical standard): e.g., Switch's load-balancing loss that downweights overused experts; DeepSeek v1-2 uses per-expert and per-device balancing; DeepSeek v3 uses **aux-loss-free balancing** via per-expert biases tuned online.
- **MoE systems**: experts parallelize across devices (expert parallelism), each FFN can fit on one device; sparse matmuls need special kernels (MegaBlocks); Nemotron 3 down-projects activations to cut communication.
- **MoE stability/finetuning issues**: router logits benefit from fp32 + z-loss; fine-tuning sparse MoEs overfits on small data (Zoph: fine-tune non-MoE MLPs; DeepSeek: use lots of data, 1.4M SFT examples).
- **Upcycling**: initialize an MoE from a pretrained dense model (split/duplicate expert weights). Examples: MiniCPM-MoE (from MiniCPM), Qwen-MoE (from Qwen 1.8B).
- **DeepSeek MoE lineage**: v1 (16B/2.8B active: shared 2 + fine-grained 64, aux-loss balancing), v2 (236B/21B active: shared 2 + fine-grained 160, 6 active, communication-balancing loss, top-M device routing), v3 (671B/37B active: shared 1 + fine-grained 256, 8 active, sigmoid+softmax top-k + top-M, aux-loss-free + seq-wise aux).
- **Supporting tricks in DeepSeek v3**: MLA (multi-head latent attention — compress KV into a low-dim latent c, see Lecture 10) and MTP (multi-token prediction with a small head that predicts one token ahead).

### Code Example: linear attention's recurrent form (conceptual)

**Code (Python):**
```python
# Parallel (training) form:  Y = Q(K^T V)   -- O(n^2 d) but only 2 matmuls
# Q: [n, dk], K: [n, dk], V: [n, dv]
KV = K.transpose(-2, -1) @ V          # [dk, dv]  -- key/value outer-product sum
Y = Q @ KV                           # [n, dv]

# Recurrent (inference) form:        -- O(n) per token, O(1) state
S = zeros(dk, dv)                    # state
for t in range(n):
    S = S + K[t][:, None] * V[t][None, :]   # S_t = S_{t-1} + k_t v_t^T
    y[t] = Q[t] @ S                        # y_t = q_t^T S_t
```

**What the Code Does:** shows the two equivalent computations of linear attention: a parallel matrix form for training and a sequential state-update form for inference that never revisits old tokens.

**Implementation Deep Dive:**
- **Why the duality matters:** training with the parallel form exploits GPU matmuls; inference with the recurrent form has O(1) memory per token (no KV cache growth). This is the same "train parallel, infer sequential" trick that makes state-space models practical.
- **Why gating (Mamba-2/GDN) is added:** the vanilla recurrence can't forget; per-position gates `γ_t`, input gates `β_t`, and erasure `(I − β k kᵀ)` add selective memory, which empirically matters a lot for language.

**Connection to Assignments:** This material is conceptual for Assignments 1–5 (you implement *full* attention with FlashAttention-2 in Assignment 2), but the arithmetic-intensity reasoning you develop here (Lecture 2 + 10) is exactly what justifies GQA/MLA choices and FlashAttention's tiling in Assignment 2.

### Code Example: top-k routing (conceptual MoE router)

**Code (Python):**
```python
import torch
import torch.nn.functional as F

def topk_route(x, router_weight, num_experts, k):
    # x: [num_tokens, d_model]
    logits = x @ router_weight                # [num_tokens, num_experts]
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)   # which experts
    # Normalize routing weights over the chosen experts (Mixtral-style softmax)
    probs = F.softmax(topk_logits, dim=-1)
    return topk_idx, probs                    # dispatch tokens to experts
```

**What the Code Does:** computes per-token expert logits, picks the top-k experts, and normalizes weights over the selected experts only (the Mixtral/DBRX/DeepSeek-v3 flavor).

**Implementation Deep Dive:**
- **Why top-k and not argmax-k=1:** k>1 gives the router a graded choice and smooths gradients; Switch (k=1) is simpler but brittle. DeepSeek v3 uses 8 active experts out of 256.
- **Why shared experts:** always-on experts capture common patterns so routed experts specialize; DeepSeek/Qwen include 1–4 shared experts.
- **Why softmax-after-topk:** normalizing only over chosen experts makes routing weights independent of the non-selected logits — a deliberate design choice that differs from the DeepSeek v1-2 logistic-gate router.

**Connection to Assignments:** You do not implement MoE in the required assignments (it's covered in Lecture 8/expert parallelism conceptually), but Assignment 2's distributed training (all-to-all token dispatch) is the systems foundation MoEs need; the leaderboard mindset (minimize loss under a budget) is the same as DeepSeek's ablation-driven choices.

### Key Takeaways

1. Attention's O(n²) cost can be attacked by linear/state-space alternatives (linear attention, Mamba-2, Gated DeltaNet) — often used as *hybrids* (e.g., 3:1) that trade a little accuracy for much cheaper long-context inference.
2. Linear attention has a train-parallel/infer-recurrent duality; gating is what makes the recurrence expressive.
3. MoEs decouple parameters from FLOPs: more capacity at the same compute, at the price of complex routing, balancing losses, and systems overhead.
4. Top-k token-choice routing with heuristic balancing losses is the practical consensus; RL routing is principled but too high-variance.
5. Modern MoEs use fine-grained experts + shared experts, aux-loss-free balancing, and fp32/z-loss-stabilized routers; upcycling from dense checkpoints is a cheap way to get MoE models.

### Potential Pitfalls

- **Load imbalance**: without balancing losses, a few experts get all the tokens — wasted capacity and (worse) devices idling.
- **Router instability**: fp16 router logits can blow up; use fp32 + z-loss (Zoph et al. 2022).
- **Batch-level token dropping**: routing drops tokens per batch, so *other users'/examples' queries can drop your tokens* — an extra source of stochasticity.
- **MoE overfitting in fine-tuning**: sparse models overfit small SFT sets; use more data or freeze/down-weight routed experts.
- **KV-cache blind spots with MLA/RoPE**: RoPE and MLA caching conflict; you need non-rotated key dims (DeepSeek's 64-dim trick).

### Review Questions

1. **Q:** Why does `Q(KᵀV)` change the cost of attention, and what's the catch?
   - **A:** Matrix multiplication is associative: `(QKᵀ)V = Q(KᵀV)`. The left form is O(n²·d), the right is O(n·d²). The catch: this is only valid when the attention kernel is the *identity* (no softmax), which is exactly what linear attention assumes — softmax attention can't be factored this way.
2. **Q:** What is the "duality" of linear attention, and why is it practically important?
   - **A:** The same computation can be written in parallel form (Q(KᵀV), great for GPU training) or recurrent form (S_t = S_{t−1} + k_t v_tᵀ, great for O(1)-state inference). You train with one and run inference with the other.
3. **Q:** How does an MoE with 256 experts and 8 active experts get "more parameters but the same FLOPs"?
   - **A:** Only k=8 experts process each token, so per-token matmul FLOPs ≈ 8/256 of the total expert FLOPs — roughly constant vs a dense model of the *active* size. The other 248 experts' parameters still occupy memory (and contribute capacity/knowledge) but do no compute for that token.
## Lecture 5: GPUs

*Date: Mon April 13 (Spring 2026) | Instructor: Tatsu Hashimoto | Materials: `lecture_05.pdf`*

### Overview

This lecture demystifies GPUs: how they differ from CPUs, their anatomy (SMs, warps, memory hierarchy), and — most importantly — *why they get slow* and *how to make fast algorithms*: low-precision computation, operator fusion, recomputation, memory coalescing, and tiling. It concludes by unpacking FlashAttention as the canonical example: tiling for the KQV matmuls plus the online (telescoping) softmax trick.

### Core Concepts & Definitions

- **GPU vs CPU**: CPUs optimize for a few fast threads (latency); GPUs optimize for many, many threads (throughput). GPUs have many tiny ALUs, less branching support, and a deep memory hierarchy.
  - *Analogy*: a CPU is a few expert craftsmen who each finish one item very quickly; a GPU is an assembly line of thousands of simple workers who together produce enormous total output — but they all must follow the same instruction each step (SIMT).
- **Anatomy**: SMs (streaming multiprocessors) each contain many SPs (streaming processors) that execute threads; a **thread block** runs on one SM with its own shared memory; threads execute in **warps** of 32 consecutive threads in lockstep (SIMT — single instruction, multiple threads).
- **Memory hierarchy**: registers (fastest) → shared memory/L1 (inside the SM, ~8× faster than DRAM but ~100× more expensive per byte) → L2 (on die) → HBM/global memory (the DRAM chips next to the GPU). Compute (FLOPs) has scaled faster than memory bandwidth — the "memory wall" — so keeping compute units fed is the central problem.
- **Tensor cores**: specialized matrix-multiply circuits (since Volta) that make matmuls >10× faster than other floating-point ops. TPUs are similar in spirit: lightweight control + fast (big) matmul unit + fast memory, but with fewer, bigger cores and no warps (block-only model).
- **The roofline model**: plot arithmetic intensity (FLOPs/byte, x) vs achieved performance (FLOP/s, y). The kink = accelerator intensity = peak FLOP/s ÷ bandwidth. Left of the kink you're memory-bound (performance grows linearly with intensity); right of it you're compute-bound (flat at peak).
- **Six ways to make GPUs fast** (from the slides):
  1. **Low precision computation**: fewer bits = fewer bytes to move; improves arithmetic intensity (fp32 ReLU: 8 bytes/FLOP → fp16: 4 bytes/FLOP). Tensor cores accelerate low/mixed precision. Frontier: FP8 (E4M3/E5M2), MXFP8 (block scaling, E8M0 scale factors), NVFP4.
  2. **Operator fusion**: combine many elementwise ops into one kernel so data isn't shipped back and forth to HBM (e.g., `sin²x + cos²x` as 5 kernels → 1 kernel). *Analogy*: a factory fed by a warehouse conveyor belt — don't send each partially-processed part back to the warehouse between steps; process it fully in one pass.
  3. **Recomputation (activation checkpointing)**: don't store every activation; recompute them in backward. Often optimal: 3 stacked sigmoids with recomputation use 5/8th the memory accesses.
  4. **Memory coalescing**: DRAM reads come in bursts (128-byte transactions); a warp's accesses are coalesced when all 32 threads hit the same burst. For row-major matrices, threads moving along rows are *not* coalesced — the classic matmul performance trap.
  5. **Tiling** (the big one): cut the output matrix into tiles; load A/B tiles into shared memory once, reuse them for many output elements, and make accesses coalesced. Non-tiled matmul reads each input N times from global memory; tiled reads each input N/T times from global memory + T times from shared memory — a factor-of-T reduction in HBM traffic.
  6. **Control divergence avoidance** (not a memory issue): threads in a warp execute the same instruction; conditionals serialize (A-path then B-path) — a hidden cost of data-dependent branches.
- **Wave quantization**: if the number of thread blocks doesn't divide the number of SMs, the last wave is partially idle (e.g., A100 has 108 SMs; 120 tiles → 108 + 12). Explains mysterious periodic performance dips (the 1792→1793 matmul mystery).
- **FlashAttention**, decomposed:
  - *Part 1*: tiling for the KQV matmul (blocked GEMM) — move A/B tiles through shared memory.
  - *Part 2*: incremental (online) softmax — to normalize tile-by-tile you track the running max and use a telescoping correction, so the softmax denominator is exact without materializing the full S = QKᵀ matrix.
  - Backward pass: recompute tile-by-tile (no storing of attention matrices).

### Code Example: online softmax (the heart of FlashAttention)

**Code (Python):**
```python
import torch, math

def online_softmax_attention(Q, K, V, block_size=2):
    # Q,K,V: [n, d]  (single head); process rows of the score matrix in tiles
    n = Q.shape[0]
    acc = torch.zeros(n, V.shape[1])          # weighted sum accumulator
    m = torch.full((n,), -float("inf"))       # running row max
    l = torch.zeros(n)                        # running row sum of exp
    for j in range(0, n, block_size):
        S = Q @ K[j:j+block_size].T           # scores tile: [n, block]
        m_new = torch.maximum(m, S.max(dim=1).values)
        alpha = torch.exp(m - m_new)          # rescale old accumulator
        P = torch.exp(S - m_new[:, None])     # unnormalized tile probs
        acc = acc * alpha[:, None] + P @ V[j:j+block_size]
        l = l * alpha + P.sum(dim=1)
        m = m_new
    return acc / l[:, None]                   # final normalization

# Verify against the naive version:
def naive(Q, K, V):
    S = Q @ K.T
    P = torch.softmax(S, dim=-1)
    return P @ V
```

**What the Code Does:** computes softmax attention without ever materializing the full `[n, n]` score matrix. It walks over column tiles of K/V, maintaining three running quantities per row: the max `m`, the sum of exponentials `l`, and the weighted accumulator `acc`. When a new tile arrives with a larger max, the old accumulator is rescaled by `exp(m_old − m_new)` (the telescoping correction), so the final division `acc / l` is exact.

**Implementation Deep Dive:**
- **Why the running max + rescale:** the standard softmax `exp(S − max(S))` requires the full row before normalizing. The online trick lets you *stream* tiles: rescaling the accumulator by `exp(m_old − m_new)` keeps every contribution weighted correctly relative to the current max. This is exactly FlashAttention's forward pass, fused with the KQV matmuls.
- **Why this enables O(1)-block memory:** only the accumulator (size [n, d]) and tile of P (size [n, block]) live in registers/shared memory; the full S and P matrices are never stored in HBM.
- **Why the backward pass recomputes:** recomputing S and P tiles in backward avoids storing them, at the cost of one extra QKᵀ-style pass — the memory-for-compute tradeoff from the recomputation slide.

**Connection to Assignments:** Assignment 2's core task is implementing **FlashAttention-2 in Triton** (forward *and* backward), including this online-softmax + tiling logic, plus masking and bias handling. The lecture's "tiling + online softmax" story is the conceptual blueprint; the assignment's Triton tips (tl.dot, block pointers, recomputation of the backward) are the mechanical implementation.

### Key Takeaways

1. GPUs are massively parallel SIMT machines: warps of 32 lockstep threads, thread blocks on SMs with shared memory, and a memory hierarchy where bandwidth (not just FLOPs) is the scarce resource.
2. Compute scales faster than memory → you must minimize data movement: fuse operators, coalesce accesses, tile through shared memory, and (sometimes) recompute instead of store.
3. Low precision (fp16/bf16/fp8) improves arithmetic intensity and unlocks tensor cores; the roofline model tells you whether you're memory- or compute-bound.
4. Tiling + online softmax = FlashAttention: the canonical example of turning an "unavoidably quadratic" op into a memory-efficient, fused kernel.
5. Performance is full of quantization effects (wave quantization, alignment, bank conflicts) — benchmarking and profiling are essential, and small changes (1792→1793) can cause big, non-obvious swings.

### Potential Pitfalls

- **Bank conflicts in shared memory**: 32 banks, one access per bank per cycle; strided access patterns (e.g., reading a matrix column) cause 32-way serialization. Mitigate with padding/swizzling.
- **Uncoalesced HBM access**: thread indices must map to consecutive addresses (128-byte transactions); row-major column-walking is a classic killer.
- **Wave quantization**: choose grid sizes that divide the SM count to avoid a partially-idle last wave.
- **Warp divergence**: data-dependent branches (e.g., `if x < 0` in ReLU kernels) serialize both paths — harmless in theory, costly in practice.
- **Low occupancy from register bloat**: each thread using >160 registers reduces how many warps an SM can schedule; thread coarsening (one thread handling multiple elements) is sometimes the fix, sometimes the problem.
- **Believing FLOPs equal runtime**: the same operation can take very different wall-clock time depending on fusion and data movement (RMSNorm example from Lecture 3).

### Review Questions

1. **Q:** Why is a matrix-vector product memory-bound while a matrix-matrix product is compute-bound, on the same hardware?
   - **A:** Both read O(n²) bytes for the matrix, but the matvec does O(n²) FLOPs (intensity ~1) while the matmul does O(n³) FLOPs (intensity ~n/3). For n=1024, matmul intensity ~341 ≫ H100's ~295 accelerator intensity (compute-bound), matvec intensity ~1 ≪ 295 (memory-bound).
2. **Q:** How does online softmax keep the result *exact* while streaming tiles?
   - **A:** It maintains a running max m and rescales previously-accumulated terms by exp(m_old − m_new) whenever the max increases. This is algebraically identical to subtracting the final max once — a telescoping correction — so the final accumulator equals the true softmax-weighted sum.
3. **Q:** Why does tiling reduce HBM traffic in matmul by a factor of T (the tile size)?
   - **A:** Each input element is loaded into shared memory once per tile it participates in (N/T times instead of N times), and within a tile it's read from fast shared memory T times. Global-memory traffic drops from O(N) reads per element to O(N/T).
## Lecture 6: Kernels, Triton

*Date: Wed April 15 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_06.py` | Deadlines: Assignment 1 due, Assignment 2 out*

### Overview

This lecture is the hands-on kernel-writing lecture: benchmark and profile to find bottlenecks, then write custom **Triton** kernels to eliminate them. It develops four kernels of increasing difficulty — GeLU (elementwise), softmax (row reduction), row-sum (reduction where the row doesn't fit in a block), and matmul+ReLU (tiling with shared memory) — and explains the GPU programming model (threads → thread blocks → grid), occupancy, bank conflicts, coalescing, and wave quantization from Lecture 5 in concrete code.

### Core Concepts & Definitions

- **Kernel**: a function that runs on the GPU. In PyTorch every primitive op launches a standard kernel; writing custom kernels (CUDA/Triton/CUTLASS/ThunderKittens) lets you fuse and tile to "make GPUs go brrr."
- **GPU hardware table (per-SM, from the lecture)**: A100: 108 SMs, 192KB L1+shared, 40MB L2, 80GB HBM, 2TB/s HBM BW; H100: 132 SMs, 256KB, 50MB L2, 80GB, 3.35TB/s; B200: 148 SMs, 256KB, 96-126MB L2, 192GB, 8TB/s. Register BW is 4–20× HBM BW — hence "keep data in registers."
- **Programming model**: *thread* (executes on part of the data) → *thread block/CTA* (group of threads sharing shared memory, scheduled on one SM) → *grid* (collection of thread blocks). Elementwise ops map naturally to threads; reductions/matmuls need thread blocks because threads must communicate via shared memory.
- **Triton's model**: specify what each *thread block* does (vs CUDA's per-thread control): load tiles from global memory into shared memory, compute, write back. Triton compiles to PTX (GPU assembly).
- **Warps**: 32 threads in lockstep; control divergence (if/else across threads in a warp) executes sequentially; SMs switch warps at zero cost when one is blocked on memory.
- **Occupancy**: #concurrent warps an SM can run, limited by registers (0–255/thread), shared memory, etc. Low occupancy isn't inherently bad if each thread does more (thread coarsening). Example: 128 threads × 160 regs = 20480 regs/block → 65536/20480 = 3 blocks on an SM.
- **Bank conflicts** (shared memory): 32 banks × 4 bytes; each cycle one access per bank. 32 threads hitting the same bank (e.g., reading a matrix column) → 32-way serialization. Swizzling (row XOR col) rearranges addresses to avoid conflicts.
- **Memory coalescing** (HBM): a warp's 32 accesses merge into 128-byte transactions when consecutive; full coalescing = 32 threads × 4 bytes in one transaction.
- **Wave quantization**: thread blocks fill SMs in waves; 160 blocks on 148 SMs → 148 + 12 (second wave mostly idle). Solution: make #blocks divide #SMs.
- **Benchmarking vs profiling**: benchmarking measures end-to-end wall time (compare implementations, study scaling); profiling shows *which kernels* run and for how long (PyTorch profiler, nsight). The kernel name itself leaks implementation details: `cutlass3x_sm100_simt_sgemm_f32_..._64x64x16` = CUTLASS library, Blackwell (sm100), float32, 64×64×16 tile.
- **Kernel fusion**: naive GeLU launches multiple kernels (many HBM round-trips); fused/builtin/torch.compile versions run one kernel (one read, one write) — a huge speedup for memory-bound elementwise ops.

### Code Example: benchmarking and profiling harness

**Code (Python):**
```python
def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()          # critical: flush async CUDA work

    times: list[float] = []
    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()          # GPU-side timestamp
        run()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    return mean(times)

def profile(run: Callable, num_warmups: int = 1):
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
```

**What the Code Does:** the benchmark wrapper warms up (compilation/JIT), uses CUDA events for GPU-accurate timing (avoids CPU launch overhead), and averages trials. The profiler captures a per-kernel table sorted by CUDA time.

**Implementation Deep Dive:**
- **Why warmup + synchronize:** the first launches may trigger compilation; CUDA is asynchronous, so without `synchronize()` you'd measure launch latency, not kernel time. CUDA events timestamp on the GPU itself, excluding CPU overhead.
- **Why multiple trials:** kernel time has variance (clocks, memory state); averaging reduces noise. For scaling studies, sweep dimension (256→8192): small matmuls are launch-overhead-bound (flat time), large ones show cubic scaling.
- **Why profiling matters:** the naive vs builtin vs compiled GeLU comparison shows *why* one is faster: the profiler reveals many kernel launches (no fusion) vs one.

**Connection to Assignments:** Assignment 2 Part 1 is exactly this: build a benchmarking + profiling harness (with Nsight Compute and NVTX ranges) for your Assignment 1 model, report per-kernel runtime, and answer questions like "which kernel dominates forward+backward?" The lecture's `run_operation1/2`, warmup, and CUDA-event patterns are the reference implementation.

### Code Example: Triton GeLU (elementwise)

**Code (Python):**
```python
import triton
import triton.language as tl

def triton_gelu(x: torch.Tensor):
    assert x.is_cuda and x.is_contiguous()
    y = torch.empty_like(x)
    num_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)   # ceil division
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)          # which block
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)   # this block's element indices
    mask = offsets < num_elements                # don't read/write past the end
    x = tl.load(x_ptr + offsets, mask=mask)      # read from HBM
    # tanh(a) = (exp(2a) - 1) / (exp(2a) + 1) since tl.tanh doesn't exist
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    tl.store(y_ptr + offsets, y, mask=mask)      # write to HBM
```

**What the Code Does:** one thread block per 1024 elements; each block computes its index range, loads with a boundary mask, computes the tanh-approximation of GeLU (reimplementing tanh from exp because Triton's `tl.tanh` doesn't exist), and stores back — all in one kernel: one HBM read, one HBM write.

**Implementation Deep Dive:**
- **Why masks:** `num_elements` may not divide BLOCK_SIZE; the mask prevents out-of-bounds accesses (a silent-corruption bug otherwise).
- **Why `tl.constexpr` for BLOCK_SIZE:** compile-time constant → Triton unrolls/specializes; grid size is runtime. `triton.cdiv` = ceil division so the last block covers the tail.
- **Why reimplement tanh:** Triton's language has a limited op set; `(e^{2a}−1)/(e^{2a}+1)` is the standard workaround. This is a great example of the "write kernels in Triton" tradeoff: less control than CUDA, but far less boilerplate.
- **Why one thread processes 8 elements:** the generated PTX shows thread coarsening — Triton/compilers vectorize so one thread handles several elements, improving ILP and memory throughput.

**Connection to Assignments:** Assignment 2 requires implementing a **fused RMSNorm Triton kernel** using precisely this pattern (blocked elementwise + mask + single read/write). The PTX-reading exercise (ld.global/st.global, %ctaid.x, %tid.x) is how you verify what your kernel actually does.

### Code Example: Triton softmax (row reduction) and row sum (looping tiles)

**Code (Python):**
```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE
    row_idx = tl.program_id(0)                    # one block per row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + row_idx * x_row_stride + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))
    x_row = x_row - tl.max(x_row, axis=0)         # subtract row max (stability)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator
    tl.store(y_ptr + row_idx * y_row_stride + col_offsets, y_row, mask=col_offsets < num_cols)

@triton.jit
def row_sum_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)   # per-thread accumulators
    for start in range(0, N, BLOCK_SIZE):            # loop over column tiles
        cols = start + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + row * N + cols, mask=cols < N, other=0.0)
        acc += x
    result = tl.sum(acc, axis=0)                     # final reduction across threads
    tl.store(out_ptr + row, result)
```

**What the Code Does:** the softmax kernel assigns one thread block per row: load the row (masked, padded with `-inf` so padded columns contribute exp(-inf)=0), subtract the max, exponentiate, sum, divide, store — a single-kernel row softmax. The row-sum kernel handles rows *longer than a block*: each thread accumulates over column tiles and the final `tl.sum` reduces the per-thread accumulators.

**Implementation Deep Dive:**
- **Why `other=float("-inf")` in softmax:** padded positions must behave as exp(−∞) = 0 in the sum — and the max subtraction must not be skewed by padding. For row-sum, `other=0.0` is right.
- **Why the loop:** `assert num_cols <= BLOCK_SIZE` is the "row fits in a block" assumption; when a row is 4096 columns and BLOCK_SIZE is 1024, you iterate tiles and accumulate — "baby tiling" (tiling a reduction) that foreshadows matmul tiling.
- **Why `tl.sum(acc, axis=0)`:** after the loop, each of the BLOCK_SIZE threads holds a partial sum over its strided columns; the block-level reduction (shared memory / warp shuffles inside Triton) produces the scalar row sum.
- **Cost accounting:** the naive PyTorch softmax does ~5MN reads + 3MN writes (max, subtract, exp, sum, divide); the fused Triton kernel does MN reads + MN writes — up to ~4× fewer memory transactions, which matters because softmax is memory-bound.

**Connection to Assignments:** This is the exact structural template for Assignment 2's FlashAttention-2: per-block tiling + masked loads + online softmax accumulation (Lecture 5's trick, here with `other=-inf` masking). Understanding the reduction loop is what makes the attention `O` accumulator (a per-block [BLOCK_M, head_dim] running weighted sum) make sense.

### Code Example: Triton matmul + ReLU (tiling with shared memory)

**Code (Python):**
```python
@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    indices_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    indices_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    indices_k = tl.arange(0, BLOCK_K)

    # Pointer grids for A and B tiles
    a_ptrs = a_ptr + indices_m[:, None] * stride_am + indices_k[None, :] * stride_ak
    b_ptrs = b_ptr + indices_k[:, None] * stride_bk + indices_n[None, :] * stride_bn

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):                       # loop over K tiles
        a = tl.load(a_ptrs, mask=(indices_m[:, None] < M) & (indices_k[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(indices_k[:, None] + k < K) & (indices_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)                              # tensor-core matmul on tiles
        a_ptrs += BLOCK_K * stride_ak                    # advance tiles
        b_ptrs += BLOCK_K * stride_bk

    acc = tl.maximum(acc, 0.0)                           # fused ReLU!
    c_ptrs = c_ptr + indices_m[:, None] * stride_cm + indices_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(indices_m[:, None] < M) & (indices_n[None, :] < N))
```

**What the Code Does:** partitions C into BLOCK_M × BLOCK_N output tiles (one thread block per tile, a 2D grid). Each block loops over the K dimension in BLOCK_K chunks: load an A tile and a B tile (with boundary masks), accumulate `tl.dot(a, b)` into the output accumulator, and finally applies ReLU *inside the kernel* before storing.

**Implementation Deep Dive:**
- **Why tiling:** naive matmul reads A[m,k] and B[k,n] from HBM for every (m,n) — O(M·K·N) reads, arithmetic intensity O(1). Loading tiles into shared memory once per K-chunk and reusing them for the whole output tile gives intensity O(tile_size); the lecture's idealized (all-in-shared-memory) version gives O(N) intensity.
- **Why masks on every load/store:** the grid is sized by `triton.cdiv(M, BLOCK_M) × triton.cdiv(N, BLOCK_N)`, so edge tiles overrun M/N/K; masked loads with `other=0.0` make overrun contributions zero — mathematically correct padding for matmul.
- **Why `tl.dot`:** it lowers to tensor-core instructions (the "matmuls are >10× faster than other float ops" hardware from Lecture 5). Triton chooses the MMA layout; in CUDA you'd manage fragments manually.
- **Why fuse ReLU:** applying `tl.maximum(acc, 0.0)` before the store avoids a second HBM round-trip (read C, activate, write C) — the fusion principle applied to a matmul. This is exactly how FlashAttention fuses softmax into the attention matmuls.
- **Why strided pointers:** matrices are linearized; `index = row*stride_row + col*stride_col`. Using strides (not hardcoded shapes) makes the kernel work for non-contiguous layouts and is what allows the same kernel to serve transposed operands.

**Connection to Assignments:** Assignment 2's FlashAttention-2 Triton kernel is this same structure (BLOCK_M × BLOCK_N output tiles, K-loop with `tl.dot`, masks) extended with online softmax and masking/bias handling, plus a backward pass that recomputes the attention scores. The FlashAttention forward "is literally just tiling for a KQV matrix multiply" (Lecture 5) — the softmax is the extra part.

### Key Takeaways

1. The recipe: benchmark → profile → change → benchmark again. Benchmarking gives end-to-end time; profiling shows which kernels dominate and what they're named.
2. Understand the hardware (SMs, warps, occupancy, bank conflicts, coalescing, wave quantization) — it determines performance even for correct code.
3. Triton: think in thread blocks, not threads: load tile → HBM, compute → shared memory/registers, store → HBM. Fuse elementwise ops into kernels to avoid round-trips.
4. The four canonical kernel patterns: elementwise (GeLU), row-reduction (softmax), tiled reduction (row sum), and tiled matmul (matmul+ReLU) — every real kernel is a combination.
5. Kernel fusion and tiling are the two levers that turn memory-bound code into compute-bound code; MFU climbs when HBM traffic drops.

### Potential Pitfalls

- **Forgetting masks**: out-of-bounds loads/stores cause silent corruption or crashes; always mask with `other=` padding values that are mathematically neutral for your op.
- **Missing `torch.cuda.synchronize()` in benchmarks**: you'll measure async launch overhead instead of kernel time.
- **Wrong padding value**: `-inf` for softmax (so exp → 0), `0.0` for matmul/sum — using the wrong one produces wrong results at boundaries.
- **Bank conflicts from tile layout**: naive shared-memory layouts serialize on the same banks; use swizzling (row XOR col) as in real FlashAttention implementations.
- **Grid sizes that don't divide SM count**: wave quantization leaves SMs idle; tune block counts.
- **Non-contiguous inputs**: kernels assert `is_contiguous()`; passing transposed tensors without strides handling breaks coalescing or correctness.
- **Precision drift**: accumulate in fp32 (`acc = tl.zeros(..., dtype=tl.float32)`) even for bf16 inputs; assignment guidance emphasizes matching PyTorch references within tolerance.

### Review Questions

1. **Q:** Why does the fused Triton GeLU run so much faster than the naive PyTorch expression `0.5*x*(1+tanh(...))` even though they compute the same math?
   - **A:** The naive version launches many kernels (mul, add, tanh, add, mul), each reading and writing the whole tensor from/to HBM. The fused kernel does one read, computes everything in registers, and one write — HBM traffic (the bottleneck for elementwise ops) drops by ~4–5×.
2. **Q:** In the matmul kernel, why does masking the A-load with `other=0.0` preserve correctness when M is not divisible by BLOCK_M?
   - **A:** Padded rows contribute 0 to the dot products, so the accumulator for out-of-bounds rows is 0; the final masked store simply doesn't write those rows. Zeros are the additive identity, making the padding mathematically harmless.
3. **Q:** What is "wave quantization," and how would you fix a kernel whose performance drops at N=1793?
   - **A:** Thread blocks are scheduled on SMs in waves; if the tile count doesn't divide the SM count, the last wave is partially idle. Fixes: adjust tile size/block count so the grid divides the SM count, or pad the problem so tiles align.
## Lecture 7: Parallelism (Across GPUs)

*Date: Mon April 20 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_07.py`*

### Overview

Last week was parallelism *within* one GPU (fusion, tiling). This week is parallelism *across* GPUs: collective operations (broadcast, scatter, gather, reduce, all-gather, reduce-scatter, all-reduce, all-to-all), the hardware (NVLink/NVSwitch vs Infiniband vs Ethernet, RDMA), and bare-bones implementations of the three classic strategies — **data parallelism** (cut the batch), **tensor parallelism** (cut the width), and **pipeline parallelism** (cut the depth) — on deep MLPs using `torch.distributed`.

### Core Concepts & Definitions

- **The unifying theme**: compute (ALUs) is far from data (memory). Within a GPU, minimize HBM accesses via fusion/tiling; across GPUs, minimize network traffic via replication/sharding. The hierarchy: L1/shared (fastest) → HBM → NVLink/NVSwitch (intra-node) → Infiniband/Ethernet (inter-node, slowest).
- **Why multi-GPU?** (1) Parameters + optimizer state + gradients + activations don't fit on one GPU; (2) more GPUs = more FLOPs = faster training.
- **Collective operations** (conceptual primitives since the 1980s; specify a *pattern* across devices, not point-to-point messages):
  - **Rank** = a device id (0..world_size−1); **world size** = number of devices.
  - **Broadcast**: copy from rank 0 to all (e.g., rank 0 loads the checkpoint and broadcasts).
  - **Scatter**: split one tensor on rank 0 across ranks (stepping stone to reduce-scatter).
  - **Gather**: collect pieces from all ranks onto rank 0 (stepping stone to all-gather).
  - **Reduce**: combine pieces from all ranks with an op (sum/min/max) onto rank 0.
  - **All-gather**: gather to *all* ranks (use: each rank holds a parameter shard, gather full params for forward).
  - **Reduce-scatter**: reduce each dimension, then scatter (use: after backward, sum gradients from data shards but distribute storage).
  - **All-reduce** = reduce-scatter + all-gather (use: sum gradients while replicating full params — plain DDP).
  - **All-to-all**: every rank sends every rank a piece (use: MoE token routing; balanced splits look like a transpose).
  - *Memory trick*: "reduce" = associative/commutative op; "scatter" = inverse of gather; "all" = destination is all devices.
- **Hardware**: PCIe (home: 242 GB/s), Ethernet (~200 MB/s, via CPU), NVLink→NVSwitch (B200: 1.8 TB/s), Infiniband (~0.05 TB/s, via HCA/NIC). **RDMA** lets a GPU read/write another GPU's memory without CPU involvement (Infiniband supports it; standard Ethernet doesn't; RoCE is Ethernet+RDMA, used by Meta). NCCL translates collectives into optimized low-level packets, detects topology, and launches GPU kernels for send/receive. GB200 NVL72: 72 GPUs in one NVLink domain.
- **Data parallelism (DDP)**: each rank gets a slice of the batch; each rank has a *full copy* of parameters; after local backward, all-reduce the gradients (AVG) so all ranks stay in sync.
- **Tensor parallelism**: each rank holds a *slice of each layer's weights* (e.g., column slice W_i of the MLP); all ranks process the full batch; after each layer, all-gather the partial activations and concatenate. Needs very fast interconnects (NVLink) because it communicates every layer.
- **Pipeline parallelism**: each rank holds a *subset of layers* (a stage); activations flow rank 0 → 1 → … with micro-batches to fill the pipeline (reduce the "bubble"). Works on slow interconnects (point-to-point, activation-sized), but bubbles hurt unless batch is big.
- **Pipeline bubble**: with n_stages and m_micro_batches, idle fraction ≈ (n_stages−1)/m — "so we need a big batch size!"
- **What's missing (flagged)**: comm/compute overlap, attention-specific parallelism, sequence/expert parallelism, and the "next lecture" FSDP/ZeRO (all-gather + reduce-scatter to avoid holding all parameters).

### Code Example: collective operations in `torch.distributed`

**Code (Python):**
```python
import torch.distributed as dist

def collective_operations_main(rank: int, world_size: int):
    setup(rank, world_size)  # init_process_group("nccl"/"gloo", ...)

    data = tensor([0., 1, 2, 3], device=f"cuda:{rank}") + rank
    dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)  # in-place!
    # after: every rank has sum of all ranks' vectors

    input = torch.arange(world_size, dtype=torch.float32) + rank   # [world_size] per rank
    output = torch.empty(1)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM)
    # output[rank] = sum of column rank across all ranks

    input = output
    output = torch.empty(world_size)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input)
    # output = [column 0 sums, column 1 sums, ...] on every rank

    cleanup()  # destroy_process_group()
```

**What the Code Does:** runs the same function in `world_size` separate processes (via `mp.spawn`); each performs all-reduce, reduce-scatter, and all-gather in sequence, demonstrating that all-reduce = reduce-scatter + all-gather (the outputs match).

**Implementation Deep Dive:**
- **Why in-place:** `dist.all_reduce(tensor=data, ...)` mutates `data` (both input and output) to avoid extra allocation — a common surprise.
- **Why NCCL vs gloo:** NCCL is the GPU backend (uses NVLink/Infiniband optimally); gloo works on CPU. The lecture's `setup` picks based on `torch.cuda.is_available()`.
- **Why `async_op=False`:** synchronous collectives block until complete; for overlap you'd use `async_op=True` and collect handles — Assignment 2 explicitly builds async overlap.
- **Why spawn with world_size=4:** each rank is a separate OS process with its own GPU; `MASTER_ADDR/MASTER_PORT` coordinate via rank 0.

**Connection to Assignments:** Assignment 2's distributed parts — DDP, optimizer-state sharding, and FSDP — are built on exactly these primitives: gradient all-reduce (DDP), reduce-scatter (state sharding), and all-gather + reduce-scatter (FSDP). The bandwidth-measurement code (`sent_bytes = size_bytes * 2 * (world_size-1)` for all-reduce) is the pattern you'll use to benchmark your implementation.

### Code Example: data parallelism (bare-bones DDP)

**Code (Python):**
```python
def data_parallelism_main(rank, world_size, data, num_layers, num_steps):
    setup(rank, world_size)
    batch_size = data.size(0)
    local_batch_size = int_divide(batch_size, world_size)     # slice the batch
    data = data[rank*local_batch_size:(rank+1)*local_batch_size].to(f"cuda:{rank}")

    params = [get_init_params(num_dim, num_dim, rank) for _ in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)            # each rank own copy

    for step in range(num_steps):
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()

        loss.backward()

        # The ONLY difference from single-GPU training:
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        optimizer.step()
    cleanup()
```

**What the Code Does:** each rank computes the loss on its own slice of the batch (different losses across ranks), then averages the gradients across ranks via all-reduce(AVG) before stepping — so all ranks' parameters evolve identically.

**Implementation Deep Dive:**
- **Why AVG not SUM:** each rank's gradient is a mean over its local batch; averaging preserves the global-batch gradient estimate.
- **Why this is "naive" DDP:** communication is a blocking all-reduce per step, 2×#params per step. The lecture notes "next time: FSDP/ZeRO — use all-gather and reduce-scatter to avoid holding all parameters in memory." Memory per rank: params + grads + optimizer states (12+ bytes/param in bf16/AdamW).
- **Why MLPs are representative:** "MLPs are the compute bottleneck in Transformers" — the patterns transfer.

**Connection to Assignments:** Assignment 2: implement distributed data parallel training with backward hooks + async communication (the lecture's blocking all-reduce is the correctness baseline; the assignment wants overlap). This is the reference "what DDP must do" — everything else is engineering.

### Code Example: tensor parallelism (forward pass)

**Code (Python):**
```python
def tensor_parallelism_main(rank, world_size, data, num_layers):
    setup(rank, world_size)
    data = data.to(f"cuda:{rank}")            # ALL ranks have the full batch
    batch_size, num_dim = data.shape
    local_num_dim = int_divide(num_dim, world_size)   # shard the width

    params = [get_init_params(num_dim, local_num_dim, rank) for _ in range(num_layers)]

    x = data
    for layer in range(num_layers):
        x = x @ params[layer]                 # only this rank's column slice
        x = F.gelu(x)

        activations = [torch.empty(batch_size, local_num_dim, device=f"cuda:{rank}")
                       for _ in range(world_size)]
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)  # gather slices
        x = torch.cat(activations, dim=1)     # reconstruct full-width activations
    cleanup()
```

**What the Code Does:** every rank holds a column-slice of each layer's weight; the full batch passes through each rank's slice, and after each layer the partial activations are all-gathered and concatenated to reconstruct the full-width tensor for the next layer.

**Implementation Deep Dive:**
- **Why all-gather per layer:** the next layer's matmul needs the *full* activation; tensor parallelism trades a per-layer all-reduce (8·b·s·h·(n−1)/n per layer) for the ability to shard weights — that's why it demands NVLink-class bandwidth and is used intra-node (≤8 GPUs).
- **Why backward is the mirror image:** f (identity in forward) becomes all-reduce in backward; g (all-reduce in forward) becomes identity — "homework exercise" in the lecture, but the pattern (Lecture 8 slides) is: column-parallel QKV/up-projection, row-parallel output/down-projection.
- **Why memory scales:** each rank stores 1/world_size of each weight matrix, so total parameter memory scales linearly with devices.

**Connection to Assignments:** Assignment 2's FSDP part is the memory-sharding cousin of this; tensor parallelism itself is not required in the assignments (it's a Lecture 8 topic), but the all-gather machinery you write for FSDP is the same primitive.

### Code Example: pipeline parallelism (with micro-batches)

**Code (Python):**
```python
def pipeline_parallelism_main(rank, world_size, data, num_layers, num_micro_batches):
    setup(rank, world_size)
    data = data.to(f"cuda:{rank}")
    batch_size, num_dim = data.shape

    local_num_layers = int_divide(num_layers, world_size)   # shard the depth
    local_params = [get_init_params(num_dim, num_dim, rank) for _ in range(local_num_layers)]

    micro_batch_size = int_divide(batch_size, num_micro_batches)
    if rank == 0:
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)   # source of data
    else:
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=f"cuda:{rank}")
                         for _ in range(num_micro_batches)]

    for x in micro_batches:
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)              # get activations from prev stage
        for param in local_params:                          # compute my layers
            x = x @ param
            x = F.gelu(x)
        if rank + 1 < world_size:
            dist.send(tensor=x, dst=rank + 1)               # pass to next stage
    cleanup()
```

**What the Code Does:** rank 0 owns the data; each rank computes its stage's layers on each micro-batch and sends activations to the next rank. Micro-batches let rank 1 start computing micro-batch 0's second-half while rank 0 still works on micro-batch 1 — filling the pipeline bubble.

**Implementation Deep Dive:**
- **Why micro-batches:** without them, only one rank works at a time (utilization 1/n). With m micro-batches the bubble fraction is ≈ (n_stages−1)/m — hence "we need a big batch size!"
- **Why point-to-point (send/recv):** pipeline communication is activation-sized and only between adjacent stages — cheap enough for slow interconnects (inter-node), which is why pipeline parallelism is used across machines while tensor parallelism stays intra-node.
- **Why not overlap here:** the lecture explicitly does *not* overlap comm/compute ("Not handled: overlapping communication/computation to eliminate pipeline bubbles") — the assignment-level pattern is 1F1B scheduling with async sends.

**Connection to Assignments:** Pipeline parallelism is not implemented in the required assignments (Assignment 2 covers DDP → FSDP), but understanding the bubble math explains why Assignment 2's FSDP overlaps communication, and why the lecture notes "pipeline: can work with slow interconnects, but need to work to reduce pipeline bubbles."

### Key Takeaways

1. There are many ways to cut up a model: data (batch), tensor/expert (width), pipeline (depth), sequence (length) — each with different communication patterns and hardware requirements.
2. The primitive vocabulary — broadcast/scatter/gather/reduce and their "all-" variants — is the shared language of distributed training; all-reduce = reduce-scatter + all-gather, and that decomposition is what makes ZeRO/FSDP possible.
3. Data parallelism (DDP): all-reduce gradients, replicate parameters — simple, but 2×#params comm and no memory scaling.
4. Tensor parallelism: shard weights, all-gather activations per layer — needs NVLink; pipeline parallelism: shard layers, send activations point-to-point — tolerates slow networks but pays pipeline bubbles.
5. Communication bandwidth is the resource to minimize (same principle as HBM bandwidth in Lecture 5/6); overlap communication with computation to hide it.

### Potential Pitfalls

- **Blocking collectives serialize the pipeline**: `async_op=False` everywhere kills throughput; overlap with computation and use handles.
- **Slicing data unevenly**: `int_divide` asserts `a % b == 0` — uneven splits break DDP's gradient averaging semantics.
- **Forgetting `dist.barrier()` in benchmarks**: ranks race ahead; timing includes straggler skew. Always barrier + synchronize around timed collectives.
- **All-gather shape mismatches**: tensor lists must pre-allocate exact output shapes; `all_gather_into_tensor` avoids list management.
- **In-place all-reduce surprise**: the input tensor is overwritten — copy if you need the pre-reduce value.
- **Using slow collectives on slow links**: e.g., per-layer all-reduce (tensor parallel) across nodes over Ethernet will dominate runtime; match the strategy to the interconnect.
- **No seed control across ranks**: `get_init_params` seeds manually; without care, ranks get different initializations (actually desirable for TP? No — TP needs identical init per weight slice; DDP needs identical init across ranks). Reproducibility matters.

### Review Questions

1. **Q:** Why does pipeline parallelism tolerate slow interconnects while tensor parallelism doesn't?
   - **A:** Pipeline sends only activations between adjacent stages — O(b·s·h) point-to-point per micro-batch, independent of model width. Tensor parallelism all-reduces activation-sized tensors *every layer* — 8·b·s·h·(n−1)/n per layer, roughly 8× pipeline's traffic — so it needs NVLink-level bandwidth to avoid becoming the bottleneck.
2. **Q:** In DDP, why must gradients be averaged (AVG) rather than summed?
   - **A:** Each rank computes its gradient as a mean over its local data slice; summing across ranks would multiply the effective learning rate by world_size. Averaging reproduces the global-batch gradient.
3. **Q:** What is the pipeline bubble, and how do micro-batches shrink it?
   - **A:** At the start and end of a pipeline pass, stages are idle while the first/last micro-batch drains — idle fraction ≈ (n_stages−1)/m for m micro-batches. More micro-batches (bigger logical batches) fill the pipeline and amortize the bubble.
## Lecture 8: Parallelism Basics (Systems Details)

*Date: Wed April 22 (Spring 2026) | Instructor: Tatsu Hashimoto | Materials: `lecture_08.pdf` (Lecture 7's parallel-PDF is on a private repo — noted as restricted)*

### Overview

Where Lecture 7 built bare-bones distributed code, this lecture gives the systems deep-dive: networking basics (why we can't connect everything), the full zoo of parallelization strategies — naive DDP, ZeRO stages 1–3 (FSDP), pipeline, tensor, sequence/context, expert parallelism — with memory and communication accounting for each, and how real large-scale runs (DeepSeek, Llama 3 405B, Gemma 2, Mixtral, Qwen 3, Nemotron) combine them ("3D/4D parallelism").

### Core Concepts & Definitions

- **Why multi-GPU**: single-GPU scaling has compute limits (even exascale supercomputers) and memory limits (large models don't fit). Parallelism splits memory *and* compute across GPUs/machines. Intra-node uses high-speed interconnects; inter-node uses the network.
- **Networking**: TPUs use a toroidal mesh (cheap, great for tensor parallel); GPUs use all-to-all/tree topologies (better for less-structured comms like expert parallel). TPU8i/8t are moving toward tree/switched networks (for MoEs). **Why not connect everything?** Cost — domain sizes and physical limits.
- **Memory accounting for naive DDP**: per parameter you need ~16 bytes: 2 (bf16 params) + 2 (bf16 grads) + 4 (fp32 master weights) + 4+4 (Adam first/second moments) — the "5 copies of weights" problem. This is why DDP's memory doesn't scale.
- **ZeRO (Zero Redundancy Optimizer) stages** — shard the redundant copies:
  - **Stage 1 — optimizer state sharding**: split first+second moments across GPUs. Steps: full local gradient → **reduce-scatter** gradients (each rank gets the slice it owns) → update only your param slice → **all-gather** updated params. Communication = 2×#params (same as DDP's all-reduce!), memory = (4+K/N_gpu)×#params. *"ZeRO stage 1 is free (in the bandwidth-limited regime) memory wins."*
  - **Stage 2 — + gradient sharding**: also shard gradients; free them as soon as they're reduced during backward (never instantiate a full gradient vector).
  - **Stage 3 = FSDP — shard everything, including parameters**: parameters are all-gathered on demand for forward/backward, then freed. Communication = 3×#params (1.5× DDP) but memory per rank = 12/8 bytes/param for pure-bf16 training. The trick: **incremental communication/computation** — overlap the all-gathers with the forward compute (e.g., gather W1, W2 while computing with W0), so comm cost is hidden.
  - *Analogy*: DDP is every library having a full copy of every book; ZeRO-3 is a library system where each branch keeps one shelf, and a book is only fetched (all-gather) when someone needs it, then returned (freed).
- **Model parallelism** (split parameters, communicate *activations* — vs ZeRO-3 which communicates params):
  - **Pipeline parallel (layer-wise)**: naive layer-wise parallelism has terrible utilization (each GPU active 1/n of the time). Micro-batches fix this; bubble ≈ (n_stages−1)/n_micro. Good communication properties (point-to-point, activation-sized), used inter-node; performance highly dependent on batch size. "Zero-bubble" variants split backward into activation-backprop vs weight-gradient computation.
  - **Tensor parallel**: split matmuls along the width. Forward: f = identity, g = all-reduce (sum partial sums); backward: f = all-reduce, g = identity. Column-wise split for QKV/up-projection; row-wise split for attention-output/down-projection; norms/routers replicated. Communication: 8·b·s·h·(n−1)/n per layer (all-reduce) vs pipeline's b·s·h point-to-point. Use TP where interconnects are fast (intra-node, ≤8 GPUs). Pros: no bubble, low complexity, no big batches needed.
  - **Sequence/context parallel**: shard the *sequence* dimension for pointwise ops (LayerNorm, dropout) and long contexts (ring attention), so activation memory scales with machines.
  - **Expert parallel**: split experts across devices, route tokens via all-to-all (only for MoE MLPs). Needs enough tokens per expert for efficiency.
- **Activation memory**: a hidden driver — even with perfect parameter sharding, activations dominate (e.g., ~5·a·s·h terms from quadratic attention including dropout, reducible by recomputation; the 10·s·b·h LayerNorm/dropout terms reducible via sequence parallel).
- **Combining strategies — "3D parallelism" rules of thumb**: (1) until the model fits: tensor/expert parallel within a machine, pipeline across machines (or ZeRO-3 depending on bandwidth); (2) then scale the rest with data parallel; if batch is small, gradient-accumulate to trade batch size for communication efficiency. Example (Narayanan et al. 2021): TP=8 first, then PP to fit, DP shrinks as models grow (DP: 32→32→32→24→15→9→6).
- **Real-world recipes**: DeepSeek v3: ZeRO-1 + TP=1 + EP=64 + PP=16; Llama 3 405B: DP=128, TP=8, PP=16; Gemma 2: ZeRO-3 + MP(TP+SP) + DP=768; Mixtral 8x22B (Megatron): TP/PP/CP/EP = 4/4/1/8; Nemotron 3 120B: TP=2, EP=64, CP=64; Qwen 3: EP=32, TP=2, PP=8.

### Code Example: ZeRO-1 (optimizer state sharding) — conceptual

**Code (Python):**
```python
# Per training step, with world_size devices, each owning params[my_slice]:
# Step 1: everyone computes a FULL gradient on their local batch
loss.backward()                       # param.grad is full-size on every rank

# Step 2: reduce-scatter the gradients -> each rank keeps only its slice
grad_slices = [torch.empty_like(grad_chunk) for ...]
dist.reduce_scatter_tensor(output=my_grad_slice, input=full_grad,
                           op=dist.ReduceOp.SUM)

# Step 3: each rank updates only ITS parameters using its slice of grad+state
for i in my_param_indices:
    state[i].m1 += ...                # fp32 moments live only on this rank
    state[i].m2 += ...
    params[i] -= lr * update(my_grad_slice_i, state[i])

# Step 4: all-gather the updated parameters so every rank has the full model
dist.all_gather_into_tensor(output=full_params, input=my_param_slice)
```

**What the Code Does:** sketches the ZeRO-1 cycle: full local gradient → reduce-scatter → local update of a parameter subset → all-gather the fresh parameters.

**Implementation Deep Dive:**
- **Why the comm cost stays 2×#params:** reduce-scatter sends #params, all-gather sends #params — exactly the volume of a single all-reduce. In the bandwidth-limited regime, ZeRO-1 is "free" vs DDP while cutting optimizer-state memory by N_gpu.
- **Why update-then-gather instead of gather-then-update:** each rank only needs its own slice's gradients and moments, so the update is embarrassingly parallel before the gather.
- **Why this is the foundation of Assignment 2's "optimizer state sharding":** the assignment asks you to implement exactly reduce-scatter for gradients, local AdamW updates, and all-gather of parameters — this is the identical algorithm.

**Connection to Assignments:** Assignment 2 tasks: (4) distributed data parallel training (all-reduce), (5) optimizer state sharding (reduce-scatter + all-gather, as above), (6) FSDP (shard params too, all-gather on demand + overlap). The lecture's "ZeRO stage 3 is 3×#param — 1.5× comm cost, but that's not bad" analysis is what the assignment's FSDP implementation and write-up should reproduce.

### Code Example: FSDP-style sharded forward (conceptual)

**Code (Python):**
```python
# Each rank stores only its shard of every weight W_l.
# To compute with layer l, first gather the full weight, then free it:
def fsdp_forward(x, layers, rank, world_size):
    for layer in layers:
        # 1. all-gather this layer's weight shards -> full W on every rank
        full_W = all_gather(layer.weight_shard[rank])
        # 2. compute (this can be overlapped with the next all-gather)
        x = x @ full_W
        x = F.gelu(x)
        # 3. free full_W (only shards stay resident)
    return x
```

**What the Code Does:** shows FSDP's on-demand parameter materialization: gather a layer's weight only when needed, compute, and drop it — so peak memory holds one full layer's weights, not the whole model.

**Implementation Deep Dive:**
- **Why overlap matters:** if gathers were blocking, FSDP would be slower than DDP; by issuing the next all-gather *while* the current matmul runs (incremental communication/computation), the comm cost is masked — e.g., `(W1W0 + W2W0)x = y` gathers W1, W2 during the W0 matmul.
- **Why communication is 3×#params:** 2 all-gathers (forward + backward parameter materialization) + 1 reduce-scatter (gradients). The lecture notes this is 1.5× DDP's traffic but with linear memory scaling.
- **Why it's "conceptually very simple — write an FSDP block wrapper":** the magic is wrapping each module so its parameters are sharded/gathered transparently; Assignment 2 asks for exactly such a wrapper (`FSDP` class around `torch.nn.Module`).

**Connection to Assignments:** Assignment 2's FSDP task (fully sharded data parallel training, with forward/backward gather and gradient reduce-scatter, plus benchmarks vs DDP) is this design. The "will it fit?" table (6.67B baseline → 16B ZeRO-1 → 24.6B ZeRO-2 → 53.3B ZeRO-3 on 8×A100-80G with 12 bytes/param) is the memory math your write-up should reproduce.

### Key Takeaways

1. Naive DDP is memory-inefficient (≈16 bytes/param); ZeRO stages 1→2→3 shard optimizer state, then gradients, then parameters — with only 1.5× DDP's communication at stage 3, but linear memory scaling.
2. ZeRO-1 is "free": same comm as DDP, strictly better memory — "you might as well always do it."
3. Model parallelism splits parameters and communicates activations: pipeline (depth, point-to-point, bubbles), tensor (width, all-reduce per layer, needs NVLink), sequence (length), expert (MoE routing, all-to-all).
4. Real training runs compose all of these: TP ≤ 8 intra-node, PP across nodes, DP for the rest, EP for MoE layers, CP for long context — with communication/computation overlap everywhere.
5. Memory is dynamic: activations often dominate over parameters; recomputation and sequence parallel are the levers.

### Potential Pitfalls

- **Using DDP when memory-bound**: every rank replicates the full model; move to ZeRO-1/2 (nearly free) or FSDP (1.5× comm).
- **Not overlapping comm and compute**: FSDP without overlap is strictly worse than DDP in wall-clock; the whole point is masking gather latency.
- **Blocking pipeline sends**: without 1F1B-style scheduling and async ops, pipelines stall.
- **TP across slow links**: per-layer all-reduces over Ethernet destroy throughput; keep TP ≤ node size.
- **Ignoring activation memory**: you can shard parameters perfectly and still OOM on activations; use sequence parallel + recomputation.
- **Naively composing DP and EP**: DP usually shares replicas with EP splits (so EP < DP), and DP + TP can interact badly (utilization drops).

### Review Questions

1. **Q:** Why is ZeRO-1 called "free" compared to DDP?
   - **A:** Its communication volume is 2×#params — the same as DDP's single all-reduce — because reduce-scatter + all-gather together move the same bytes as all-reduce. But memory drops from (4+K)×#params to (4+K/N_gpu)×#params. Same bandwidth cost, strictly better memory.
2. **Q:** What does FSDP gather and when, and why is 3×#params "not bad"?
   - **A:** FSDP all-gathers parameter shards on demand for each layer in forward and backward (2×#params) and reduce-scatters gradients (1×#params): 3×#params total, 1.5× DDP's 2×#params — while sharding *all* memory (params, grads, optimizer state, and with sequence parallel, activations).
3. **Q:** Why does tensor parallelism require faster interconnects than pipeline parallelism?
   - **A:** TP communicates every layer (all-reduce of activation-sized tensors, 8·b·s·h·(n−1)/n per layer); pipeline communicates only between stages (b·s·h point-to-point per micro-batch). TP's per-layer, all-to-all-ish traffic needs NVLink; pipeline's sparse point-to-point works over Infiniband.
## Lecture 9: Scaling Laws — Basics

*Date: Mon April 27 (Spring 2026) | Instructor: Tatsu Hashimoto | Materials: `lecture_09.pdf` | Deadline: Assignment 2 due*

### Overview

Given 10,000 B200s for a month, *which* model do you train? This lecture introduces **scaling laws** — simple, predictive power-law rules relating loss to data size, model size, and compute — so you can tune hyperparameters on small models and extrapolate to large ones. It covers the history (from 1993 sample-complexity work to Hestness 2017), the theory of *why* power laws appear (estimation error, intrinsic dimensionality), the classic Kaplan and Chinchilla results (including the famous N vs D tradeoff and why they disagree), critical batch size, muP, and the practical "scaling-law design procedure."

### Core Concepts & Definitions

- **Scaling law**: a simple formula mapping a resource (dataset size n, parameters N, compute C) to loss/error, e.g., `Loss ≈ C·n^(−α)`. Log-log plots become straight lines: "scale-free" or power-law behavior.
  - *Analogy*: like measuring how much faster a road trip gets with more lanes — a clean power-law trend (each doubling of lanes gives a fixed fractional speedup) that lets you predict the effect of 16 lanes from measurements at 2/4/8 lanes.
- **Why power laws? (theory)**: estimation error decays polynomially. Toy example: estimating a mean from n i.i.d. samples gives E[(μ̂−μ)²] = σ²/n — a scaling law with slope −1. Nonparametric regression in d dimensions gives error ~ n^(−1/d): *dimension-dependent* slopes. So scaling-law exponents relate to the (intrinsic) dimensionality of the data — an active research area (Bahri 2021) with sketchy estimators.
- **Empirical facts**: loss vs dataset size is linear in log-log across LM/MT/speech; data *composition* affects the intercept (offset) but not the slope (distribution-shift scaling laws, Hashimoto 2021) — meaning diverse data matters. Slopes differ from classical 1/n predictions — a "mystery."
- **Data repetition**: repeating finite data reduces its value; effective data D' < unique tokens — so data selection should be *adaptive to scale*.
- **Model engineering with scaling laws**: choose architecture (Transformers beat LSTMs at scale), optimizer (Adam vs SGD), depth/width, batch size, learning rate — all from small-model experiments. **Important caveat**: downstream-task scaling is often *less* predictable than pretraining-loss scaling.
- **Critical batch size**: the minimum batch size before diminishing returns; defined by fitting S_min/E_min on the steps-vs-examples curve (~2× the steps/passes of the naive optimum; claimed to relate to the trace of the gradient covariance over the squared gradient norm). The smaller the loss target, the bigger the batch.
- **muP (maximal update parametrization)**: a width-aware initialization + learning-rate scaling so that optimal hyperparameters transfer across model sizes (details in Lecture 11).
- **The N vs D question**: given compute C = 6ND, train a bigger model (N) or on more tokens (D)?
  - **Kaplan et al. 2020**: N_opt ∝ C^0.73, D_opt ∝ C^0.27 — *tokens per parameter decreases* with compute (bigger models, relatively less data).
  - **Chinchilla (Hoffmann et al. 2022)**: N_opt ∝ C^0.5, D_opt ∝ C^0.5 — *compute-optimal* is ≈ D = 20N (70B params ↔ ~1.4T tokens).
  - *Why the disagreement*: Kaplan's counting quirks (excluded the last layer, warmup too high at small budgets); non-embedding vs total parameters; small nonlinearities. A forensic re-analysis (Besiroglu et al. 2024) found Chinchilla method 3 itself was flawed, and re-fitting the raw data matched methods 1/2.
- **Chinchilla's three fitting methods**: (1) minimum over runs (lower envelope of training curves), (2) **IsoFLOPs** (fix compute C_i, sweep model sizes, take the min loss; the ⟨C_i, N_opt⟩ pairs form a power law), (3) joint least-squares fit of a parametric form over a size×data grid.
- **Train-optimal ≠ deployment-optimal**: Chinchilla optimizes for *training* compute, but most real compute is *inference* — so models are increasingly *over-trained*: GPT-3: 2 tokens/param, Chinchilla: 20, LLaMA-65B: 22, Llama 2 70B: 29, Mistral 7B: 110, Llama 3 70B: 215. The more the model is used, the more it's worth overtraining.
- **The scaling-law design procedure**: (1) train a few smaller models; (2) establish a scaling law (e.g., Adam-vs-SGD); (3) pick hyperparameters from the law's prediction — *"the effect of hyperparameters on big LMs can be predicted before training!"*
- **IsoFLOPs everywhere**: the method transfers to diffusion models, MoEs (sparsity scaling), etc.

### Code Example: IsoFLOPs scaling-law fitting (Assignment 3's core)

**Code (Python):**
```python
import numpy as np

# Data: for each compute budget C_i, train models of varying sizes N_ij
# (with D_ij = C_i / (6 * N_ij) tokens), record final loss L_ij.
# runs = [(C_i, N_ij, L_ij), ...]

def isoflop_optima(runs):
    optima = []  # (C_i, N_opt(C_i), D_opt(C_i))
    for C_i in sorted(set(r for r, _, _ in runs)):
        subset = [(n, l) for r, n, l in runs if r == C_i]
        N_opt, L_min = min(subset, key=lambda nl: nl[1])   # min loss on this isoflop curve
        D_opt = C_i / (6 * N_opt)                          # 6ND rule
        optima.append((C_i, N_opt, D_opt))
    return optima

def fit_power_law(xs, ys):
    # log y = log a + b log x  ->  linear regression in log space
    logx, logy = np.log(np.array(xs, dtype=float)), np.log(np.array(ys, dtype=float))
    b, loga = np.polyfit(logx, logy, 1)
    return np.exp(loga), b

# Example: fit N_opt = a * C^b  and  D_opt = c * C^d
optima = isoflop_optima(runs)
a, b = fit_power_law([o[0] for o in optima], [o[1] for o in optima])
c, d = fit_power_law([o[0] for o in optima], [o[2] for o in optima])
# Predict for a big budget C_target:
N_target = a * C_target ** b
D_target = c * C_target ** d
```

**What the Code Does:** for each fixed FLOPs budget, finds the model size with minimum loss (the IsoFLOPs optimum), derives the token count via C = 6ND, then fits power laws N_opt(C) and D_opt(C) in log space to extrapolate to a large target budget.

**Implementation Deep Dive:**
- **Why min-over-isoflop-curves:** for fixed compute, tiny models can't fit the data and huge models can't take enough steps — the loss curve is convex, and its minimum is the compute-optimal configuration for that budget.
- **Why log-log regression:** power laws are lines in log space; `np.polyfit(logx, logy, 1)` gives the exponent as the slope. Watch out: this is fitting in log space, which weights small losses differently than fitting in linear space (a known subtlety).
- **Why C = 6ND:** the 6ND rule from Lecture 2 converts (N, D) to compute — the backbone of every scaling-law computation.

**Connection to Assignments:** Assignment 3 is *exactly* this: you get a training API (hyperparameters → validation loss) with a 12 B200-hour budget to fit scaling laws, then submit predicted compute-optimal hyperparameters and predicted loss for a 48 B200-hour run. The write-up must describe your IsoFLOPs (or joint-fit) methodology; the leaderboard grades your predicted model's loss. The lecture's "Chinchilla method 2 = IsoFLOPs" is the recommended starting point; the assignment also allows Kaplan-style and muP ideas.

### Code Example: critical batch size (conceptual)

**Code (Python):**
```python
# For a target loss, sweep batch sizes; record steps-to-target S and examples E.
# Theory (McCandlish et al.): the curve follows roughly
#   S(B) = S_min * (1 + B_crit / B)          # steps needed vs batch size
#   E(B) = S(B) * B                           # examples consumed
# Fit S_min, B_crit; pick B* that balances steps and examples.
def fit_critical_batch(B_sweep, S_measured):
    # Solve for S_min, B_crit: S(B) = S_min * (1 + B_crit / B)
    # (least squares in B vs S, or in 1/B vs S)
    import numpy as np
    invB = np.array([1.0 / b for b in B_sweep])
    S = np.array(S_measured, dtype=float)
    S_min, S_min_Bcrit = np.polyfit(invB, S, 1)   # S = S_min + (S_min*B_crit) * (1/B)
    B_crit = S_min_Bcrit / S_min
    return S_min, B_crit
```

**What the Code Does:** fits the batch-size→steps curve to extract the critical batch size B_crit, the point beyond which larger batches give diminishing returns (roughly 2× the steps/passes optimum).

**Implementation Deep Dive:**
- **Why diminishing returns:** beyond B_crit, doubling the batch doesn't halve the steps to target loss; you waste examples. The critical batch grows as the loss target shrinks.
- **Why it matters for Assignment 1/3:** batch size is a first-order training hyperparameter; scaling analyses (DeepSeek, StepFun — Lecture 11) are built on fitting optimal batch vs scale.

**Connection to Assignments:** Assignment 1's training loop must support configurable batch size (and gradient accumulation); Assignment 3's API takes batch size as an input hyperparameter — the scaling-law fits you build should treat batch/LR as tunable knobs, exactly as the lecture's critical-batch analysis suggests.

### Key Takeaways

1. Loss vs data/model/compute follows clean power laws in log-log space — predictable enough to tune small and extrapolate large ("scaling as prediction").
2. The theory: estimation error decays polynomially (mean-estimation slope −1; nonparametric regression slope −1/d), tying scaling exponents to data dimensionality — but the observed LM slopes remain a partial mystery.
3. Compute-optimal training (Chinchilla) ≈ 20 tokens per parameter; but deployment reality (inference-dominated compute) pushes toward heavily over-trained models (up to 200+ tokens/param).
4. IsoFLOPs is the workhorse method: fix compute, sweep size, take the minimum, fit power laws — it transfers to MoEs, diffusion, and your Assignment 3.
5. Hyperparameter choices (optimizer, depth, architecture, batch) can be *predicted* from small-model scaling laws before spending big-compute — the whole point of the course's efficiency mindset.

### Potential Pitfalls

- **Blindly extrapolating**: scaling laws are lower bounds; they "break" if you apply them outside the fitted regime (e.g., data repetition, different architectures).
- **Parameter-counting inconsistencies**: embedding vs non-embedding parameters, last-layer exclusion — these shifted Kaplan vs Chinchilla by a lot. Be explicit about what counts as N.
- **Treating train-optimal as deployment-optimal**: Chinchilla's 20 tokens/param is for training compute; if inference dominates, overtrain.
- **Fitting in log space carelessly**: log-space least squares weights points unevenly; report both and check residuals.
- **Ignoring batch/LR co-tuning**: scaling laws fitted with a fixed batch size may not transfer; batch and LR are scale-sensitive.
- **Downstream ≠ pretraining**: capability scaling (MMLU etc.) is much less predictable than loss scaling — beware overclaiming.

### Review Questions

1. **Q:** Why do Kaplan and Chinchilla disagree on the optimal N/D ratio, and what's the practical takeaway?
   - **A:** Kaplan found N_opt ∝ C^0.73 (fewer tokens per param as compute grows); Chinchilla found N_opt ∝ C^0.5 with ≈20 tokens/param. Disagreement sources: parameter-counting (non-embedding vs total, last layer), warmup at small budgets, and small nonlinearities in the fits (plus a flawed method 3). Practical takeaway: modern practice over-trains relative to Kaplan and often relative to Chinchilla because inference compute dominates.
2. **Q:** Walk through the IsoFLOPs procedure in 4 steps.
   - **A:** (1) Pick a set of compute budgets C_i; (2) for each, train models of several sizes N_ij with D_ij = C_i/6N_ij tokens; (3) for each budget, take the model size with minimum loss → (C_i, N_opt); (4) fit power laws N_opt ∝ C^a, D_opt ∝ C^b in log space and extrapolate to the target budget.
3. **Q:** Why might a model's scaling-law slope differ from the classical 1/n?
   - **A:** Classical parametric estimation gives 1/n (slope −1); nonparametric learning in d dimensions gives n^(−1/d), so the slope reflects the data's intrinsic dimensionality — neural LMs exhibit slopes corresponding to neither cleanly, which is the "mystery" motivating intrinsic-dimension theory (Bahri 2021).
## Lecture 10: Inference

*Date: Wed April 29 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_10.py` | Deadlines: Assignment 2 due, Assignment 3 out*

### Overview

Inference is how models are actually used — and it has very different characteristics from training: it's **memory-bound** and **dynamic** (requests arrive and finish at different times). This lecture derives the arithmetic intensity of the MLP and attention layers during prefill vs generation, computes theoretical latency/throughput for Llama 2 13B on an H100, and surveys the techniques to speed inference up: reducing the KV cache (GQA, MLA, CLA, local/sliding-window attention, DeepSeek v4's CSA/DSA/HCA), quantization (QAT/PTQ, AWQ), model pruning + distillation, speculative sampling (lossless!), and dynamic-workload systems (continuous batching, PagedAttention).

### Core Concepts & Definitions

- **Why inference efficiency matters**: training is a one-time cost; inference is repeated endlessly (OpenAI processes ~8.6T tokens/day). Agents make it worse: internal traces can grow unboundedly. Tokens generated = compute spent.
- **Metrics**: **TTFT** (time-to-first-token, driven by prefill), **latency** (seconds/token for one query, interactive), **throughput** (tokens/sec for many queries, batch).
- **Two stages**: **prefill** (process the whole prompt in parallel, like training — compute-bound) and **generation/decode** (one token at a time — memory-bound). Key asymmetry: *checking is faster than generation* (prefill computes all positions at once).
- **KV cache**: avoid recomputing key/value vectors for the whole history at every generation step. For every sequence (B), token (S), layer (L), head (K), store an H-dim vector. Naive inference is O(T³) FLOPs for T generated tokens; KV caching makes it O(T²).
- **Arithmetic intensity accounting** (bf16, 2 bytes/val):
  - MLP per step: FLOPs = 6·B·T·D·F; bytes = 4·B·T·D + 4·B·T·F + 6·D·F → intensity ≈ B·T. Prefill (large B·T): compute-bound; generation (T=1): intensity ≈ B — needs many concurrent requests (batching) to stay compute-bound.
  - Attention per step: FLOPs = 4·B·S·T·D; bytes = 4·B·S·D + 4·B·T·D → intensity = S·T/(S+T). Prefill (T=S): S/2 (good); generation (T=1): <1 — **impossible to fix with batching**, because every sequence has its own KV cache (Q,K,V all depend on B), unlike MLP weights which are shared.
  - Summary: prefill is compute-bound; generation is memory-bound (read all parameters + KV cache each step).
- **Latency/throughput model**: latency ≈ memory/bandwidth (read all params + KV cache), throughput = B/latency. Bigger batches: worse latency (larger KV cache to read/write), better throughput (amortize parameter reads) — a fundamental tradeoff. Also: model replication gives linear throughput scaling; TTFT is a prefill phenomenon (small batches for TTFT, big batches for generation throughput).
- **Reducing the KV cache** (memory-bound ⇒ smaller cache ⇒ faster):
  - **GQA (grouped-query attention)**: N query heads but K key/value heads (K < N); MHA = K=N, MQA = K=1. Cuts KV cache by N/K with little accuracy loss (Ainslie 2023). Llama 2 13B with K:40→8: worse latency per batch but better throughput and fits in memory.
  - **MLA (multi-head latent attention, DeepSeek v2)**: store a compressed latent c_t = W_c h_t (C dims) instead of K,V; project up to K=W_K c, V=W_V c on the fly. DeepSeek v2: N·H=16384 → C=512 (+64 RoPE dims = 576). MLA slightly *beats* MHA at lower cost. RoPE incompatibility: add non-latent rotated key dims.
  - **CLA (cross-layer attention)**: share KV across *layers* (like GQA shares across heads); improves the accuracy/KV-size Pareto frontier.
  - **Local (sliding-window) attention**: attend only to a window (Longformer, Mistral); KV cache independent of sequence length; effective context grows linearly with layers; hurts accuracy sometimes → interleave local with global attention (hybrid layers).
  - **DeepSeek v4 attention** (1M context): Compressed Sparse Attention (CSA — compress every m tokens into 1), DeepSeek Sparse Attention (DSA — select top-k), Heavily Compressed Attention (HCA).
  - Other: linear attention / state-space models (Mamba-2, GatedDeltaNet), diffusion LMs.
- **Quantization**: fewer bits = fewer bytes = faster (memory-bound). fp32 (training) → bf16 (default inference) → fp8/int8 → int4/nvfp4. **QAT** (quantize during training, expensive); **PTQ** (post-training, cheap: calibrate scale/zero-point on sample data; GPTQ uses Hessian information); **AWQ** (activation-aware: keep 0.1–1% of important weights high-precision based on activation magnitudes; fp16→int3 gives 4× lower memory, 3.2× speedup).
- **Pruning + distillation**: (1) identify important {layer, head, hidden dim} on ~1024 calibration samples; (2) remove unimportant parts → smaller model; (3) distill the original into the pruned model (NVIDIA's pruning-KD loop).
- **Speculative sampling (lossless)**: a cheap **draft model** p proposes γ tokens; the **target model** q scores them all in parallel (prefill-speed); accept/reject via modified rejection sampling. Key properties: (1) always generates at least one token (rejection sampling would loop forever); (2) **guaranteed exact sample from q** — proof by example: with vocab {A,B}, p(A)>q(A), residual max(q−p,0) correction makes P[sample A]=q(A), P[sample B]=q(B). Extensions: Medusa (parallel heads), EAGLE (draft from target's features).
  - *Analogy*: a fast intern drafts a paragraph; the professor reads it once (fast — checking is faster than writing) and either approves or corrects; the final text is statistically identical to the professor writing it alone.
- **Continuous batching (Orca)**: iteration-level scheduling — add new requests to the batch as they arrive instead of waiting for all sequences in a static batch to finish. **Selective batching**: attention processes each sequence separately (ragged), non-attention ops concatenate all sequences into one [Σs, H] tensor.
- **PagedAttention (vLLM)**: virtual-memory paging for the KV cache — divide each sequence's KV into non-contiguous fixed-size blocks; eliminate internal/external fragmentation; share prefix blocks across sequences (system prompts, multiple samples) with copy-on-write. Other vLLM tricks: fused block-attention kernels, FlashAttention/FlashDecoding, CUDA graphs.
  - *Analogy*: static KV allocation is like assigning each process a contiguous chunk of RAM sized for its worst case (fragmentation); paging is virtual memory — blocks mapped on demand, shared read-only pages, copy-on-write.

### Code Example: arithmetic intensity of attention (the key derivation)

**Code (Python):**
```python
# B batch, S past tokens, T next tokens, D model dim; bf16 => 2 bytes/value
flops = 4*B*S*T*D                       # QK^T: 2*B*S*T*D  +  softmax@V: 2*B*S*T*D
bytes = 4*B*S*D + 4*B*T*D               # read Q,K,V; write Y

intensity = (S*T) / (S + T)             # after simplification

# Prefill: T = S  =>  intensity = S/2        (good — compute-bound)
prefill_intensity = S / 2

# Generation: T = 1  =>  intensity = S / (S + 1) < 1   (bad — memory-bound)
generate_intensity = S / (S + 1)
```

**What the Code Does:** counts FLOPs and HBM bytes for the attention matmuls and shows the arithmetic intensity is S·T/(S+T) — S/2 for prefill, <1 for generation — with **no dependence on B**.

**Implementation Deep Dive:**
- **Why no B dependence:** in attention, Q/K/V are per-sequence (B multiplies both FLOPs and bytes, canceling); in MLPs, weights are shared across the batch, so B improves intensity. This is *why* batching can't save generation-time attention — the KV cache is unique per sequence.
- **Why <1 intensity is fatal:** the H100 accelerator intensity is ~295 FLOP/byte; generation attention moves ~1 byte per FLOP — the GPU's tensor cores are almost completely idle during decode.
- **Why this justifies GQA/MLA:** every KV-cache byte removed directly reduces generation latency (latency ∝ memory read per step).

**Connection to Assignments:** Assignment 1's resource accounting asks for these exact per-component FLOP/byte counts; Assignment 2's FlashAttention and benchmarking work targets the training side of the same equations. The theoretical latency/throughput model (`compute_transformer_performance_stats` with `num_params = 2VD + 3LDF + 2L·(2DNH + 2DKH)`, `kv_cache_size = 4·S·K·H·L` bytes) is what you'd reproduce to sanity-check Assignment 2's measured numbers.

### Code Example: speculative sampling (lossless decoding)

**Code (Python):**
```python
def speculative_sample(draft_logits, target_logits, draft_next, rng):
    # draft_logits / target_logits: [vocab] for the NEXT position
    p = softmax(draft_logits)             # draft distribution
    q = softmax(target_logits)            # target distribution
    x = draft_next                        # candidate token drawn from draft
    u = rng.uniform(0, 1)
    if u < min(1, q[x] / p[x]):           # accept with probability q(x)/p(x)
        return x, True
    # Reject: resample from the residual distribution, normalized
    residual = torch.clamp(q - p, min=0)
    x2 = sample(residual / residual.sum())
    return x2, False
```

**What the Code Does:** draws a candidate from the draft model; accepts it with probability q(x)/p(x) (importance-weighting); on rejection, resamples from the normalized residual max(q−p, 0). This is rejection sampling modified to always terminate with a valid sample from q.

**Implementation Deep Dive:**
- **Why it's exact:** the mixture of accept (with prob min(1, q/p)) and residual-resample reproduces q exactly. The lecture's two-symbol proof: P[sample A] = p(A)·(q(A)/p(A)) + p(B)·1·0 = q(A); P[sample B] = p(B)·1 + p(A)·(1−q(A)/p(A))·1 = q(B).
- **Why it's fast:** the draft (e.g., 8B) generates γ tokens at memory-bound speed; the target (e.g., 70B) scores γ tokens *in parallel* (prefill-style, compute-bound) — the asymmetry between checking and generation.
- **Why "always generate at least one":** vanilla rejection sampling can reject forever; the modification guarantees progress, and the residual correction keeps the distribution exact.
- **How to make drafts better:** distill the draft toward the target (higher acceptance), Medusa (parallel draft heads), EAGLE (draft conditioned on target features).

**Connection to Assignments:** Not implemented in the required assignments, but Assignment 5's RL training loop uses the same "rollout via fast inference server (vLLM)" machinery — the vLLM interface in `cs336_alignment/vllm_utils.py` is the practical instantiation of the inference stack this lecture describes.

### Code Example: latency/throughput model (Llama 2 13B on H100)

**Code (Python):**
```python
def compute_transformer_performance_stats(config):
    # Number of parameters (embedding + 3 MLP mats + attention QKV/O, per layer)
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L
    parameter_size = 2 * num_params                       # bf16

    # KV cache per sequence: S tokens * K heads * H dim * L layers * (K+V) * 2 bytes
    kv_cache_size_per_seq = S * (K*H) * L * 2 * 2

    memory = B * kv_cache_size_per_seq + parameter_size   # total to read per step

    latency = memory / memory_bandwidth                   # seconds/token
    throughput = B / latency                              # tokens/sec
    return num_params, memory, latency, throughput

# Llama 2 13B config: S=1024, D=5120, F=13824, N=40, K=40, H=128, L=40, V=32000
# B=1:  latency ~ (26GB params + small cache) / 3.35TB/s  (~7.8ms/token)
# B=64: better throughput, worse latency (bigger KV cache to read)
# B=256: throughput gains diminish AND it doesn't fit in 80GB H100!
```

**What the Code Does:** computes parameter count, memory (params + KV cache), and the memory-bandwidth-limited latency/throughput, then instantiates it for Llama 2 13B at batch sizes 1/64/256 — showing the latency/throughput tradeoff and the memory ceiling.

**Implementation Deep Dive:**
- **Why latency = memory/bandwidth:** generation is memory-bound; every step must read all parameters plus the whole KV cache from HBM. This is a theoretical upper bound (perfect overlap assumed) — real systems do worse.
- **Why throughput = B/latency:** B sequences are generated in parallel per step, so tokens/sec = B × (1 step/sec).
- **Why KV-cache reduction helps twice:** less memory → shorter read time per step (lower latency) *and* room for larger batches (higher throughput).

**Connection to Assignments:** The `TransformerPerformanceStats` pattern (symbolic accounting with sympy in the lecture) is a reusable template for Assignment 1's resource accounting and for sanity-checking Assignment 2's profiling numbers (e.g., "why is generation memory-bound?" — your benchmark of the attention kernel should reflect it).

### Key Takeaways

1. Inference comes in two flavors: prefill (compute-bound, like training) and generation (memory-bound, one token at a time) — and generation's attention is memory-bound in a way *batching can't fix* (intensity <1, independent of B).
2. The KV cache is the key resource: latency ∝ (params + KV cache) read per step; reduce it with GQA, MLA, CLA, local attention, or state-space hybrids.
3. Lossless speedups exist: speculative sampling is provably exact (modified rejection sampling) and exploits "checking is faster than generating."
4. Lossy speedups: quantization (QAT/PTQ/AWQ) and pruning+distillation trade accuracy for memory and speed.
5. Dynamic workloads need systems tricks: continuous batching (iteration-level scheduling, selective batching) and PagedAttention (paging, prefix sharing, copy-on-write) — ideas borrowed straight from operating systems.

### Potential Pitfalls

- **Ignoring the KV cache in memory budgets**: with long contexts and large batches, the KV cache (not parameters) is what OOMs — and it's per-sequence, so it grows with concurrency.
- **Using MHA when memory-bound**: at high batch, GQA/MLA are nearly free wins; MHA's accuracy edge is small.
- **Quantizing naively**: per-tensor scales lose outlier channels; use per-block scales (AWQ, MXFP8) and calibrate on real activation ranges.
- **Believing speculative decoding changes the distribution**: it must be exact — if your implementation modifies sampling, it's no longer "the target model."
- **Static batching**: waiting for all requests to finish wastes GPU (one slow sequence blocks everyone); use continuous batching.
- **Fragmented KV allocation**: reserving max-length blocks per request causes internal/external fragmentation and wasted HBM — page it.
- **Forgetting TTFT vs throughput**: optimizing throughput for interactive traffic hurts user-visible latency; tune batch size per phase (small for prefill, big for generation).

### Review Questions

1. **Q:** Why can't batching fix the memory-boundness of generation-time attention?
   - **A:** Attention's arithmetic intensity S·T/(S+T) has no B term: Q, K, V are all per-sequence, so B multiplies FLOPs and bytes equally and cancels. MLPs are different — weights are shared across B, so batching raises intensity to ≈B·T. Generation attention stays <1 intensity no matter the batch size.
2. **Q:** How does speculative sampling guarantee exact samples from the target model q?
   - **A:** It's modified rejection sampling: accept draft x with probability min(1, q(x)/p(x)); on rejection, resample from the normalized residual max(q−p, 0). The mixture algebra (shown for the two-symbol case) reproduces q exactly, and the "always accept at least one token" modification keeps it terminating.
3. **Q:** GQA with K=8 instead of K=40 heads on Llama 2 13B — what changes and what doesn't?
   - **A:** The KV cache shrinks 5× (latency per step drops, larger batches fit, throughput rises). Query heads stay at 40 (expressivity of attention largely preserved); accuracy drops are small or negligible per Ainslie et al. 2023.
## Lecture 11: Scaling Laws — Case Studies and Details

*Date: Mon May 4 (Spring 2026) | Instructor: Tatsu Hashimoto | Materials: `lecture_11.pdf` | Deadline: Assignment 3 due*

### Overview

This lecture turns the theory into practice by walking through *public, detailed scaling recipes*: **MiniCPM** (small high-perf models using muP + WSD learning rates + Chinchilla analysis) and **DeepSeek** (batch/LR scaling via small runs + IsoFLOPs model sizing), plus StepFun's large-scale empirical study of LR/batch scaling, optimizer scaling (including Muon), and a deeper dive into **muP** — what it is, how it's derived, and what it's robust (and not robust) to.

### Core Concepts & Definitions

- **Scaling in practice**: since 2022, few models publish scaling details; MiniCPM and DeepSeek are the notable exceptions with careful, public analyses.
- **MiniCPM recipe (2024)**: 1–2.5B models that beat most 2B and match many 7B models.
  1. **muP to stabilize scaling**: `scale_emb=12, scale_depth=1.4, init_std=0.1, lr=0.01` — with muP, the optimal LR is (roughly) constant across widths.
  2. **Fix the aspect ratio** and scale up overall size (gap between largest scaling run and actual model ≈ 5×).
  3. **Optimal batch**: from 3 model sizes (9m/30m/170m), plot loss vs (batch, data); optimal batch grows polynomially as loss decreases (Kaplan 2020-style analysis).
  4. **WSD learning rate** (warmup–stable–decay): enables cheap Chinchilla analysis — you can *restart* a run at the end of the stable phase with a different token budget instead of training from scratch. This turns the O(n²) cost of fitting scaling laws into something affordable.
  5. Chinchilla methods 1 (lower envelope) and 3 (joint fit) → very high data:model ratios.
- **WSD (warmup-stable-decay) LR schedule**: split the schedule into warmup, stable, and decay phases; the loss drops rapidly during decay (~10% of training). Matches cosine performance while enabling restarts.
- **DeepSeek recipe (2024)**: no muP — directly estimate optimal batch/LR from small-scale runs ("near-optimal" runs within 0.25% of min); WSD-style LR with two 10% decay steps; Chinchilla method 2 (straightforward IsoFLOPs) for model sizing; fitted scaling models accurately predict the final model's loss.
- **Recent recipes** (less detailed): Qwen (LR/batch fits), Kimi K2 (sparsity scaling law for MoEs), Hunyuan (IsoFLOPs for MoE — optimal 96:1 data:active-param), LLaMA 3 (IsoFLOPs, 39:1 ratio, compute-to-downstream scaling), MiniMax-01 (architecture scaling + Chinchilla method 1).
- **StepFun scaling-law study**: purely empirical grid search over (LR, batch) at multiple scales:
  1. Loss over batch/LR is **convex** — minimizers cleanly identifiable.
  2. Scaling trends: batch depends primarily on dataset size; higher optimal LR with D for fixed M (fragile if switching to WSD).
  3. Generalizes to MoEs and other datasets (with caveats).
- **Optimizer scaling problems**: (1) different optimizers need different hyperparameters and possibly different scaling rules; (2) significant scale dependence — "always check scaling with respect to compute and Chinchilla ratios — these are often major confounders to performance"; (3) establishing scaling is nontrivial — nice-looking scaling can blow up (AdamC + sqrt-batch LR example).
- **Muon**: an optimizer for *matrix-valued* parameters that (approximately) orthogonalizes the update via Newton–Schultz iteration: B_t = UΣVᵀ → UVᵀ. Works well at scale (nanoGPT speedrun, Kimi K2); gains are tricky to measure.
- **muP (maximum update parametrization) in depth**:
  - *Assertions*: as width n_l grows, (A1) activations at init stay Θ(1); (A2) after one gradient step, the activation change is Θ(1).
  - *Derivation sketch*: for a deep linear net h_l = W_l h_{l−1} with W ~ N(0, σ²I), choosing σ = Θ(1/√n_{l−1}·min(1, n_l/n_{l−1})) keeps ‖h_l‖² = Θ(n_l). For updates, ΔW_l = −η∇_{h_l}ℓ·h_{l−1}ᵀ (rank-one outer product); enforcing Δℓ = O(1) gives the LR scaling η = Θ(n_l/n_{l−1}) for SGD and η = Θ(1/√n_{l−1})... (with Adam, ΔW·√n_{l−1} = Θ(η_l)).
  - *Standard parametrization (SP)* vs muP: SP uses init 1/√n_{l−1}, LR Θ(1); muP adjusts both init and LR (and uses η·n_{l−1} for Adam's "base" scaling) so optimal LR is *width-invariant*.
  - *What muP is robust to*: SwiGLU/squared-ReLU activations, large/small batches, zero-attention init, exotic optimizers (Lion) — mostly.
  - *What breaks muP*: **RMSNorm gains** (learnable gains break the theory — removable with little loss), **exotic optimizers based on gradient signs**, and **strong weight decay (0.1)** — "maybe the only significant muP failure."
  - *Bottom line*: muP is generally useful — SP is quite a bit more unstable; muP init/parametrization is easier to tune.

### Code Example: WSD learning-rate schedule (why it makes scaling analysis cheap)

**Code (Python):**
```python
def lr_schedule(step, total_steps, warmup_steps, max_lr, decay_frac=0.1):
    # Phase 1: warmup (linear)
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # Phase 2: stable (constant)
    decay_start = int(total_steps * (1 - decay_frac))
    if step < decay_start:
        return max_lr
    # Phase 3: decay (e.g., cosine or linear to ~10% of max_lr)
    t = (step - decay_start) / max(total_steps - decay_start, 1)
    return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

# The key trick for Chinchilla analysis: restart from the stable phase.
# If a run at 200B tokens ends its stable phase at step S, you can continue
# (with a new decay) for 400B, 800B... without retraining from scratch.
```

**What the Code Does:** implements the three-phase WSD schedule and highlights the restart property — the stable phase lets you fork a run into multiple token budgets cheaply.

**Implementation Deep Dive:**
- **Why WSD matches cosine:** empirically the decay phase concentrates most of the loss drop; the stable phase keeps the model "ready to decay." DeepSeek uses two 10% decay steps; MiniCPM a single decay.
- **Why this matters for scaling laws:** fitting a Chinchilla joint fit needs losses at many (N, D) points; each used to require a from-scratch training run. WSD restarts turn one run into several data points — the difference between O(n²) and O(n) cost. This is the exact economics Assignment 3's budget (12 B200 hours) forces you to respect.

**Connection to Assignments:** Assignment 3's training API may not expose WSD, but the *methodology* — fitting scaling laws from a limited budget of runs, extrapolating to 48 B200 hours — mirrors MiniCPM/DeepSeek. Your write-up should describe your search-space exploration strategy (IsoFLOPs vs joint fit vs muP-flavored assumptions), exactly as these papers document theirs.

### Code Example: muP-style initialization and LR scaling (conceptual)

**Code (Python):**
```python
import math

def init_and_lr_scale(fan_in, fan_out, scheme="muP"):
    if scheme == "SP":                       # standard parametrization
        init_std = 1.0 / math.sqrt(fan_in)
        lr_scale = 1.0
    else:                                    # muP (baby version)
        # init: Theta(1/sqrt(fan_in) * min(1, fan_out/fan_in))
        init_std = (1.0 / math.sqrt(fan_in)) * min(1.0, fan_out / fan_in)
        lr_scale = fan_out / fan_in          # for SGD; Adam uses lr * fan_in
    return init_std, lr_scale
```

**What the Code Does:** contrasts standard parametrization (init 1/√fan_in, LR 1) with muP (init 1/√fan_in·min(1, fan_out/fan_in), LR fan_out/fan_in) — the two changes that make optimal hyperparameters width-invariant.

**Implementation Deep Dive:**
- **Why the init rule:** keeps activations Θ(1) regardless of width — derived from matrix concentration (‖W_l‖ → σ·(√n_{l−1} + √n_l)).
- **Why the LR rule:** keeps the *update* Θ(1) per step (condition A2): with ΔW = −η·grad-outer-product, enforcing Δℓ = O(1) gives η ∝ n_l/n_{l−1} (SGD) — and with Adam's per-coordinate normalization, the LR scaling changes (η·√n_{l−1}).
- **Why it's useful:** hyperparameter *transfer* across model sizes is what makes small-scale scaling experiments (Assignment 3!) trustworthy. The lecture's miniCPM numbers (scale_emb=12, scale_depth=1.4) are the practical knob values.

**Connection to Assignments:** Assignment 3's handout explicitly allows incorporating muP ideas (G. Yang et al.) in your scaling methodology; Assignment 1's initialization (the handout's "some scaling factor" for init) should at least be width-aware. If you ever train models at multiple scales, muP is the difference between "LR transfers" and "LR must be re-tuned."

### Key Takeaways

1. Two public recipes: MiniCPM (muP + WSD + Chinchilla 1&3) and DeepSeek (small-run LR/batch fits + IsoFLOPs + WSD) — both accurately predicted their final models' losses.
2. WSD learning rates are the enabling trick: they match cosine, and their restart property makes Chinchilla-style fits affordable (n vs n² training runs).
3. LR and batch are scale-sensitive: loss is convex in (LR, batch) per scale, but the optimal values shift with compute/data — always check scaling against Chinchilla ratios; it's a major confounder in optimizer/architecture comparisons.
4. muP makes optimal LR (roughly) width-invariant via width-aware init + LR scaling; it's robust to many modern components but breaks on RMSNorm gains and strong weight decay.
5. Recent large labs (Qwen, Kimi K2, Hunyuan, LLaMA 3, MiniMax) all do *some* form of IsoFLOPs/scaling analysis — the field treats scaling as the standard design tool.

### Potential Pitfalls

- **O(n²) scaling-law fitting**: training from scratch for every (N, D) point is unaffordable; use WSD restarts / stable-phase forks.
- **Trusting LR transfer without muP**: naive width scaling shifts the optimal LR; either use muP or re-fit (DeepSeek's approach).
- **Confusing train-optimal with inference-optimal**: recall Lecture 9 — deployment compute is mostly inference, so overtrain (tokens/param ≫ 20).
- **Forgetting batch-size scaling**: batch has strong diminishing returns past the critical batch size; sweep it jointly with LR.
- **muP claims over-applied**: RMSNorm gains and strong weight decay break the transfer — validate rather than assume.
- **Fitting "nice" curves that blow up**: a good-looking scaling fit can hide instability (AdamC example); always validate at one larger scale.

### Review Questions

1. **Q:** How does the WSD schedule make Chinchilla-style analysis cheaper?
   - **A:** Cosine schedules require a full from-scratch run per (N, D) point. WSD's stable phase lets you fork a run: continue with different decay timings/token budgets to get multiple loss points from one pretraining run — reducing the cost from quadratic to (near) linear in the number of fitted points.
2. **Q:** What are muP's two assertions and the resulting knobs?
   - **A:** (A1) activations stay Θ(1) at init; (A2) a gradient step changes activations by Θ(1). This fixes the init scale σ = Θ(1/√fan_in·min(1, fan_out/fan_in)) and the LR scale (fan_out/fan_in for SGD; Adam's base LR scaled by √fan_in).
3. **Q:** Why did DeepSeek skip muP and still succeed?
   - **A:** Instead of assuming LR transfer, they *measured* it: run many small-scale models, keep "near-optimal" (within 0.25% of min loss) runs, fit the LR/batch scaling empirically, then used IsoFLOPs for model size. Empirically-derived scaling can substitute for parametrization-based transfer — at the cost of more small-scale compute.
## Lecture 12: Evaluation

*Date: Wed May 6 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_12.py` | Deadlines: Assignment 3 due, Assignment 4 out*

### Overview

Before deciding *what data* to train on, you must decide *what behavior you want* — and how to measure it. This lecture surveys the evaluation landscape: perplexity (in-distribution and zero-shot), exam benchmarks (MMLU, MMLU-Pro, GPQA, HLE), chat benchmarks (Chatbot Arena, AlpacaEval, WildBench), agentic benchmarks (SWE-bench, Terminal-Bench, CyBench, MLE-Bench), pure-reasoning benchmarks (ARC-AGI), and safety benchmarks (HarmBench, AIR-Bench) — then discusses realism/ecological validity (GDPVal, MedHELM, Clio), validity (train-test contamination, dataset quality), and how to think about evaluation (methods vs models vs agents).

### Core Concepts & Definitions

- **The core challenge**: converting an *abstract construct* ("good model") into a *concrete metric*. Different stakeholders define "good" differently: benchmark scores, cost-adjusted scores, human preference (Chatbot Arena), or actual user adoption (OpenRouter).
  - *Analogy*: evaluating an LM is like rating a restaurant — Michelin stars (benchmarks), price/quality (cost-adjusted), Yelp reviews (human preference), and repeat customers (actual adoption) all measure different things.
- **Perplexity**: for a model p and dataset D: `PPL = (1/p(D))^(1/|D|)`. It measures whether p assigns high probability to D. Training minimizes it; the obvious evaluation is test-set perplexity.
  - *History*: PTB, WikiText-103, One Billion Word benchmark (in-distribution era; CNNs+LSTMs improved 51.3→30.0). GPT-2 introduced *zero-shot* (out-of-distribution) evaluation: trained on WebText, evaluated on standard datasets.
  - *"Perplexity is all you need"*: if p = true distribution t, then p(solution|problem) solves every task — pushing down perplexity eventually "reaches AGI." *"Perplexity is more than you need"*: it penalizes every token, including uninformative ones ("Stanford was founded in **1885**" vs "founded"); use conditional perplexity p(response|prompt)^(1/|response|). Some benchmarks are perplexity in disguise: LAMBADA (cloze), HellaSwag (sentence completion).
  - *Warning for perplexity leaderboards*: you must trust that submitted probabilities are valid (sum to 1).
- **Exam benchmarks**: controlled subject/difficulty, unambiguous answers, easy grading.
  - **MMLU**: 57 subjects, multiple-choice, few-shot; really tests *knowledge*, not "language understanding." **MMLU-Pro**: removes noisy questions, 4→10 choices, CoT evaluation; drops accuracy 16–33% (less saturated).
  - **GPQA**: graduate-level, Google-proof questions written by PhD contractors; PhD experts 65%, non-experts with Google 34%, GPT-4 39%.
  - **HLE (Humanity's Last Exam)**: 2500 multimodal questions, $500K prize pool for question creators, filtered by frontier LLMs through multiple review stages.
  - *Limit*: doesn't capture real usage (open-ended, no single correct answer).
- **Chat benchmarks** (open-ended evaluation):
  - **Chatbot Arena**: random users prompt two anonymized models and vote; compute **Elo** rankings via p(A beats B) = 1/(1+10^((Elo_B−Elo_A)/400)). Properties: real prompts, dynamic, no need to feed the same prompts to all models (humans rate), but biased populations, style-vs-correctness conflation, sycophancy.
  - **AlpacaEval**: 805 instructions; win rate vs GPT-4 as judged by GPT-4; had a length bias (LLM judges favor longer responses) → AlpacaEval 2.0 debiased via regression; correlates well with human Arena.
  - **WildBench**: 1024 examples from 1M real human-chatbot conversations; GPT-4-turbo judge with a checklist (like CoT for judging).
  - Key lesson: pairwise comparisons between similar responses give higher signal; beware human and LLM-judge biases; checklists/rubrics improve reliability.
- **Agentic benchmarks** (evaluate what LMs *do*, not say): Agent = LM + scaffold (planning, delegation, memory, context engineering).
  - **SWE-bench**: 2294 tasks over 12 Python repos; given codebase + issue, submit a PR; graded by unit tests.
  - **Terminal-Bench**: terminal environments (simple, universal); 229 crowdsourced tasks.
  - **CyBench**: 40 CTF tasks; first-solve time as difficulty.
  - **MLE-Bench**: 75 Kaggle competitions (train models, process data).
- **Pure-reasoning benchmarks**: **ARC-AGI** — tasks 100% solvable by humans but hard for AI; each task unique so memorization doesn't help. ARC-AGI-1 (2019), ARC-AGI-2 (2025, more multi-step), ARC-AGI-3 (2026, interactive). Pretrained LMs didn't move the needle; reasoning models (o1, o3) did.
- **Safety benchmarks**: **HarmBench** (510 harmful behaviors violating laws/norms); **AIR-Bench** (regulatory frameworks: 314 risk categories, 5694 prompts); **jailbreaking** — GCG auto-optimizes adversarial prompts that transfer from open to closed models. Safety is contextual (politics, law, norms vary by country) and varied (hallucination, sycophancy, crime abetting, inequality).
- **Realism / ecological validity**: how well does the eval capture real use? **GDPVal** (OpenAI): 44 occupations from top-9 GDP sectors, tasks from ~14-year-experience professionals; **MedHELM**: 121 clinical tasks from 29 clinicians (vs standardized exams); **Clio** (Anthropic): LM analysis of real user data. Realism and privacy are sometimes at odds.
- **Validity / contamination**: 
  - Route 1: infer train-test overlap from the model (exchangeability tests).
  - Route 2: encourage reporting norms (report train-test overlap, confidence intervals).
  - Route 3: fresh evals (LiveCodeBench, UncheatableEval — scrape new pages).
  - Route 4: private evals (internal codebases, personal writings) — easiest for perplexity.
  - Dataset quality: SWE-bench → SWE-bench Verified; "Platinum" versions of benchmarks; agentic benchmarks often have insufficient test cases (trivial agents solve them); Docent inspects agent traces with LLMs.
- **How to think about evaluation**: no one true evaluation — depends on the question (purchase decision, raw capability, benefits+harms, development feedback). Pre-foundation models evaluated *methods* (standardized splits); today we mostly evaluate *models/systems* (anything goes) — except method-style setups like nanogpt speedrun (fixed data, time-to-loss). Evaluating methods encourages algorithmic innovation; evaluating systems serves downstream users. **Define the rules of the game.**

### Code Example: perplexity and conditional perplexity

**Code (Python):**
```python
import torch

def perplexity(log_probs: torch.Tensor) -> torch.Tensor:
    # log_probs: [num_tokens] (log p(token_i | past))
    nll = -log_probs.sum()
    return torch.exp(nll / log_probs.numel())       # (1/p(D))^(1/|D|)

def conditional_perplexity(prompt_log_probs, response_log_probs):
    # Only score the response tokens (p(response | prompt)^(1/|response|))
    nll = -(prompt_log_probs.sum() + response_log_probs.sum())  # if needed
    return torch.exp(-response_log_probs.sum() / response_log_probs.numel())
```

**What the Code Does:** computes perplexity from per-token log-probs (the standard formula), and shows the conditional variant that only scores the response — avoiding dilution by predictable prompt tokens.

**Implementation Deep Dive:**
- **Why exp of mean NLL:** PPL = exp(−(1/T)Σ log p(x_t|x_<t)) — a geometric-mean inverse probability. Lower is better; random guessing over a vocab of size V gives PPL ≈ V.
- **Why conditional PPL:** for tasks like "Stanford was founded in ___", tokens like "founded" and "in" are nearly deterministic and cheap; scoring only the answer isolates the model's actual predictive quality on the *relevant* part.
- **Why this matters for leaderboards:** Assignment-style leaderboards that minimize OpenWebText perplexity must trust the submitted probability model — the lecture's warning about validating probabilities (sum-to-1) applies.

**Connection to Assignments:** Assignment 1's leaderboard is a **perplexity leaderboard** (minimize OpenWebText perplexity given 45 minutes on a B200) — exactly the metric defined here. Assignment 4's leaderboard minimizes perplexity given a token budget (evaluation of data curation via downstream PPL). The "evaluate methods vs models" distinction is why the course can run these leaderboards at all.

### Code Example: Elo rating (Chatbot Arena-style)

**Code (Python):**
```python
import numpy as np

def p_win(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def update_elo(elo, winner_idx, loser_idx, k=32):
    e = p_win(elo[winner_idx], elo[loser_idx])
    elo[winner_idx] += k * (1 - e)      # winner gained more if upset
    elo[loser_idx] += k * (0 - (1 - e)) # loser lost more if upset
    return elo

# Fit: iterate over all pairwise comparisons, updating both models' Elo.
# (In practice: fit by maximum likelihood over the comparison matrix.)
```

**What the Code Does:** implements the Elo update: the winner takes points proportional to how surprising the win was (1 − expected), the loser loses the same amount.

**Implementation Deep Dive:**
- **Why Elo:** it converts pairwise preferences into a scalar rating whose difference predicts win probability via the logistic curve. The Arena fits Elo by maximum likelihood over all comparisons rather than the online update (shown here for intuition).
- **Why pairwise is powerful:** comparing two similar responses yields higher signal than absolute scoring — the basis for both human (Arena) and LLM-judge (AlpacaEval/WildBench) evaluation.

**Connection to Assignments:** Evaluation methodology is relevant to Assignment 5's optional DPO supplement (AlpacaEval is one of the eval datasets used there) and to Assignment 4's "train classifiers to filter quality" (whose training data mirrors DCLM's positive/negative example construction — see Lecture 13/14).

### Key Takeaways

1. There is no one true evaluation — pick metrics that answer *your* question, and clearly state whether you're evaluating a method, a model, or an agent.
2. Perplexity is still the workhorse for model *development* (smooth scaling laws), but benchmarks must capture real-world situations for the non-believers.
3. The benchmark landscape trends toward harder exams (MMLU→MMLU-Pro→GPQA→HLE), real-user chat preference, agentic task performance, pure reasoning (ARC-AGI), and safety.
4. Watch for validity threats: train-test contamination, dataset quality issues, judge biases (length!), and overfitting to the eval.
5. Evaluation shapes AI development — choosing what to measure is a consequential design decision (rules of the game matter).

### Potential Pitfalls

- **Benchmark saturation**: models hit ceiling on MMLU; use MMLU-Pro/GPQA/HLE which are designed to stay hard.
- **LLM-judge length bias**: GPT-4-based judges favor longer responses; debias (AlpacaEval 2.0 regression) or use checklists.
- **Contamination**: if your pretraining data (Assignment 4!) contains benchmark examples, perplexity/accuracy numbers are invalid — filter, or use fresh/private evals.
- **Sycophancy and style conflation in human ratings**: Arena votes measure preference, not correctness.
- **Evaluating the agent ≠ evaluating the model**: agent scaffolds change results dramatically; report the whole system.
- **Trusting unvalidated probabilities in leaderboards**: ensure submitted models are proper distributions.

### Review Questions

1. **Q:** Why might a model with better test-set perplexity still be worse for users?
   - **A:** Perplexity rewards probability on all tokens (including trivial ones), is sensitive to tokenization, and doesn't capture preference, safety, style, or task success — the things users actually care about. Chat benchmarks and agentic benchmarks measure different constructs.
2. **Q:** What makes GPQA "Google-proof" and why does that matter?
   - **A:** The questions are so specialized that non-experts can't answer them even with web access (34% with Google vs 65% for PhD experts) — so the benchmark isolates genuine expert-level knowledge rather than retrieval.
3. **Q:** How do you evaluate an open-ended response reliably?
   - **A:** Prefer pairwise comparisons between similar responses (higher signal); use checklists/rubrics with LLM judges (WildBench-style CoT judging); validate the judge against human (Arena) correlations; beware length/sycophancy biases.
## Lecture 13: Data I — Sources and Datasets

*Date: Mon May 11 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_13.py`*

### Overview

Data is "the most important thing to get right" in training language models — and it "does not fall from the sky." This lecture covers where data comes from (crawlers, live services), the legal landscape (copyright, licenses, fair use, lawsuits), the canonical sources (Common Crawl, Wikipedia, GitHub, arXiv), and a tour of famous datasets: BooksCorpus, WebText/OpenWebText, CCNet, C4, GPT-3's mix, The Pile, Gopher's MassiveText, LLaMA's mix, RefinedWeb/FineWeb, Dolma, DCLM, Nemotron-CC, The Stack, and CommonPile.

### Core Concepts & Definitions

- **Why data is secret**: open-weight models (Llama 3) publish architecture and training procedures but essentially *no data details* — reasons: competitive dynamics and copyright liability. "Data is fundamentally a long-tail problem, scales with human effort (unlike architectures, systems)."
- **Stages of training data**: **pre-training** (large amounts of lower-quality raw text) → **mid-training** (high-quality data for capabilities) → **post-training** (chat transcripts / RL). The trend: from large-low-quality to small-high-quality. Terminology: base model (pre+mid), instruct/chat model (post). (OLMo 2 example: pretraining → Dolmino mid-training → Tülu post-training.)
- **Raw sources**: the web = live servers; you need a **crawler** (discover pages from a seed set, download, respect policies). What you *can't* get: dynamic content (apps, Discord), authenticated content (Facebook, X, NYT), robots.txt-blocked, Cloudflare-blocked, rate-limited, ToS-prohibited. The "decline of consent": restrictions on common datasets' URLs have increased over time. **Shadow libraries** (LibGen, Sci-Hub, Z-Library): technically on the web but legally piracy — ~4M books (LibGen 2019), ~88M papers (Sci-Hub 2022).
  - *Analogy*: crawling the web is like fishing in a huge lake — the lake is "the Internet," but much of it is private property (walled gardens), protected areas (robots.txt), or legally off-limits (copyrighted waters).
- **Copyright basics**: protects "original works of authorship fixed in any tangible medium" — i.e., *basically everything on the Internet is copyrighted*. Protection applies to *expression*, not *ideas* (quicksort isn't copyrightable; your website is). Lasts ~75 years; registration required before suing (not for protection). To use a work: (1) **license** it (Creative Commons, commercial deals — Google×Reddit, OpenAI×Shutterstock/StackExchange) or (2) **fair use** (four factors: purpose/character, nature of work, amount used, market effect). Examples of fair use: summarizing a movie, reimplementing an algorithm, Google Books snippets (Authors Guild v. Google).
  - *LM-specific considerations*: copying the data (the first step of training) is already a technical violation; training should be transformative; models should capture ideas (wizards), not expressions (Harry Potter); LMs can affect markets regardless of copyright. **ToS can add restrictions** beyond copyright (YouTube prohibits downloading even CC videos).
  - *Lawsuits*: NYT v. OpenAI (2023); Authors v. Anthropic (2024) — summary judgment (2025): training on books is fair use, *pirating* copies is not; Anthropic settled for $1.5B. Authors v. Meta (2025): training on books (this instance) is fair use; torrenting still pending. Verdict so far: *training has been deemed fair use in specific instances; pirating is clearly illegal; still evolving*.
- **Common Crawl**: non-profit (2007); monthly crawls adding 3–5B pages; ~300B pages total; April 2026 crawl = 2.19B pages (372.2 TB). Built on Apache Nutch: seed URLs → queue → download → enqueue hyperlinks, with selection/politeness/re-visit policies. Two formats: **WARC** (raw HTTP responses, e.g., HTML) and **WET** (converted text — a lossy process). HTML→text tools: trafilatura, resiliparse — and the conversion choice measurably affects downstream task accuracy (DCLM).
- **Wikipedia**: 67M articles, 361 languages; no original thought, notability-based; anyone can edit (vandalism reverted); periodic dumps (no crawling needed). *Data poisoning risk*: malicious edits can be injected right before dumps.
- **GitHub**: 420M+ repos (28M public); repos (via git protocol) + metadata (issues/PRs/comments via API, GitHub Archive). Lots of duplicates (forks/copies); permissive licenses only. Software Heritage aggregates repos from GitHub/GitLab/Bitbucket/PyPI (28.8M source files).
- **arXiv**: ~3M papers since 1991; metadata (title/abstract, CC0) + PDF + optional LaTeX; bulk download from S3.
- **Dataset lineage tour** (each with its filtering recipe):
  - **BERT**: Wikipedia + BooksCorpus (7K self-published books from Smashwords, 985M words — since taken down for ToS violations); documents (not sentences) — vs the 1B-word benchmark.
  - **GPT-2 WebText**: pages linked from Reddit posts with ≥3 karma (a quality surrogate); 8M pages, 40GB. **OpenWebTextCorpus**: open replication using Reddit submission URLs + fastText language filter + near-dup removal.
  - **CCNet**: Common Crawl + paragraph dedup (light normalization) + fastText language ID + **KenLM 5-gram Wikipedia-likeness filter**; CCNet(CC) beat Wikipedia-trained BERT.
  - **C4** (Colossal Clean Crawled Corpus): one April-2019 snapshot (1.4T tokens); *manual heuristics*: keep lines ending in punctuation with ≥5 words, drop pages with <3 sentences, bad words, '{', 'lorem ipsum', 'terms of use'; langdetect English p≥0.99 → 806 GB (156B tokens). WebText-like variant from OpenWebText-linked pages improved GLUE/SQuAD.
  - **GPT-3**: Common Crawl (processed) + WebText2 + Books1/2 + Wikipedia → 570GB (400B tokens). CC processing: *quality classifier* distinguishing {WebText, Wikipedia, Books} from the rest + fuzzy dedup.
  - **The Pile** (EleutherAI): 22 curated domains, 825GB (~275B tokens): Pile-CC (WARC + jusText), PubMed Central, arXiv (LaTeX), Enron emails, Project Gutenberg, **Books3** (196K books from shadow library Bibliotik — since taken down; contained Stephen King etc.), StackExchange (Q&A, XML dumps with metadata).
  - **Gopher MassiveText**: MassiveWeb (English, dedup, rule-based quality — "80% words contain an alphabetic char", Google SafeSearch toxicity) + C4 + Books/News/GitHub/Wikipedia → 10.5TB (Gopher trained on 300B tokens = 12%).
  - **LLaMA**: CommonCrawl via CCNet (classify *references* of Wikipedia), C4, GitHub (permissive licenses + manual rules), Wikipedia (20 languages), Gutenberg + Books3, arXiv (stripped comments/macros/bibliography), Stack Exchange (28 largest sites by score) → 1.2T tokens. Replicated by RedPajama v1; SlimPajama = 627B deduplicated (MinHashLSH).
  - **RefinedWeb** (Falcon): "web data is all you need" — trafilatura on WARC (not WET), Gopher rules, no ML-based filtering (avoid biases), MinHash fuzzy dedup → 600B released (of 5T). **FineWeb**: 95 CC dumps, URL filtering, lang ID p(en)>0.65, Gopher+C4 rules, MinHash, PII anonymization → 15T tokens.
  - **Dolma** (AI2): Reddit (Pushshift), PeS2o (40M papers), C4, Gutenberg, Wikipedia; CC processed with fastText lang ID, Gopher+C4 rules, Jigsaw toxicity, Bloom-filter dedup → 3T tokens.
  - **DCLM** (DataComp-LM): DCLM-pool (240T tokens, processed CC); DCLM-baseline filtered by a **quality classifier** trained on positives (OpenHermes-2.5, ELI5 — 200K each) vs negatives (RefinedWeb — 200K); fastText classifier beats other filtering methods → 3.8T tokens. Model-based filtering "becoming the norm."
  - **Nemotron-CC**: FineWebEdu/DCLM filter too aggressively (remove 90%); ensembled classifiers (distilled Nemotron-340B educational-value scorer + DCLM); **synthetic rephrasing** of low-quality data and task generation for high-quality data → 6.3T tokens (1.1T HQ subset). For reference: Llama 3 trained on 15T, Qwen3 on 36T.
  - **The Stack**: 137M repos (git clone via GitHub Archive names, 2015–2022), 51B files (5B unique); permissive licenses only (go-license-detector); MinHash near-dup removal → 3.1TB. **Stack v2**: adds issues/PRs/comments, Software Heritage, docs sites; removes binaries/malware/bots; pairs low-resource languages with LLVM IR; linearizes PR diffs for training.
  - **CommonPile**: 8TB of *permissively licensed* data — can you train a good model legally? Decent results, but "tough to compete without more tokens." Subtleties: license laundering, collection licenses don't extend to individual works, synthetic-data copyright unclear.

### Code Example: quality filtering via a classifier (the DCLM/GPT-3 pattern)

**Code (Python):**
```python
import numpy as np

# Given: target data T (what "good" looks like) and huge raw data R.
# 1. Train a fastText-style classifier: positives from T, negatives from R.
#    score(x) = p(good | x)
# 2. Keep documents stochastically based on their score.

def keep_document_gpt3_style(score: float) -> bool:
    # GPT-3 kept docs with probability based on score (Pareto(9) draw > 1 - score)
    return np.random.pareto(9) > 1 - score

def keep_document_threshold(score: float, thresh: float = 0.5) -> bool:
    return score >= thresh  # Dolma-style: keep pages with p(English) >= 0.5
```

**What the Code Does:** shows the two keep-decisions: stochastic retention (GPT-3: high-score docs almost always kept, low-score docs occasionally kept — a soft filter) vs hard thresholding (Dolma).

**Implementation Deep Dive:**
- **Why stochastic:** a hard cutoff can be brittle; GPT-3's Pareto(9) draw implements "keep with probability ~score" smoothly. Modern pipelines (DCLM) threshold the classifier score directly.
- **Why fastText:** filtering must run over hundreds of TB — speed is a hard requirement ("extremely fast" is a desideratum); fastText linear classifiers run orders of magnitude faster than neural scorers at inference time.
- **Why the recipe generalizes:** the same "target data T vs raw R" framework covers language ID (T = English pages), quality (T = curated/instruction data), toxicity (T = clean comments) — Lecture 14's filtering lecture builds on exactly this.

**Connection to Assignments:** Assignment 4's core task is this recipe: convert Common Crawl HTML→text (trafilatura), then filter with (a) the provided NSFW/hate-speech classifiers, (b) Gopher/C4-style rules, and (c) PII removal — then deduplicate with MinHash (Lecture 14). The DCLM quality-classifier idea motivates why the assignment gives you pretrained classifiers rather than raw heuristics.

### Key Takeaways

1. Data does not fall from the sky: live services → crawls/dumps → processed data, with technical (crawling, dynamic content, auth) and legal (ToS, copyright) constraints at every step.
2. Almost everything on the Internet is copyrighted; you either license it or argue fair use (four factors). Current legal state: training has been found fair use in specific instances; pirating copies is not.
3. The dataset lineage shows a steady march toward *more processing*: rule-based heuristics (C4, Gopher) → ML quality classifiers (GPT-3, DCLM, Nemotron-CC) → synthetic rephrasing (Nemotron-CC) → legal-only corpora (CommonPile).
4. Key sources: Common Crawl (web, WARC/WET), Wikipedia (dumps), GitHub (git protocol + archive), arXiv (S3 dumps) — each with its own access mechanics.
5. Data is the key differentiator between LMs: companies guard it fiercely; open models publish everything *except* data.

### Potential Pitfalls

- **Training on WET when WARC is available**: HTML→text conversion quality (trafilatura vs WET) measurably changes downstream accuracy (DCLM).
- **Ignoring ToS**: even Creative Commons content can be off-limits if the platform's ToS prohibits downloading (YouTube example).
- **Deduplicating before quality filtering (or vice versa) without care**: order and item granularity (3-sentence spans in C4) matter; removing mid-document spans can break coherence.
- **Using copyrighted shadow-library data (Books3)**: taken down, litigation risk — prefer licensed/PD sources.
- **Trusting any single source**: even Wikipedia can be poisoned right before dumps; filtering and dedup are always needed.
- **Forgetting that dumps ≠ live services**: Common Crawl/GitHub Archive are snapshots with their own biases and gaps.

### Review Questions

1. **Q:** What are the four fair-use factors, and which direction does each push for LLM training?
   - **A:** (1) Purpose/character — transformative/educational favored (training is arguably transformative); (2) nature of the work — factual over creative (books are creative, pushing against); (3) amount used — snippets over whole works (training uses whole works); (4) market effect — LMs may substitute for writers. Courts have so far found training fair use in specific instances.
2. **Q:** Why did GPT-3's Common Crawl processing outperform C4's despite both starting from the same raw crawl?
   - **A:** GPT-3 used a *learned quality classifier* (trained to distinguish WebText/Wikipedia/Books from the rest) plus fuzzy dedup, whereas C4 used fixed manual heuristics (line/sentence/punctuation rules, bad-word lists). Classifier-based filtering generalizes better to what "good" looks like — the theme DCLM later formalized.
3. **Q:** What's the difference between WARC and WET, and why does it matter?
   - **A:** WARC stores raw HTTP responses (HTML); WET stores the lossy HTML→text conversion. Downstream accuracy depends on the conversion (trafilatura > WET in DCLM's comparison), so many modern pipelines re-extract text from WARC with better tools (The Pile's jusText, RefinedWeb's trafilatura).
## Lecture 14: Data II — Transformation, Filtering, Deduplication, Mixing, Synthetic Data

*Date: Wed May 13 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_14.py`*

### Overview

This lecture is the *algorithmic* heart of data engineering. It covers the four pipeline stages — **transformation** (HTML/PDF → text), **filtering** (language ID, quality, toxicity via classifiers), **deduplication** (exact and near-dup via hashing, MinHash, LSH), and **data mixing** (how to weight sources, epoching traps, UniMax caps, regression-based mixing, simulated epoching) — plus **post-training / synthetic data** (OpenThoughts, SWE-smith, SWE-Zero, etc.).

### Core Concepts & Definitions

- **Transformation**: raw data is HTML, PDF, or directories. HTML→text: remove boilerplate, extract content, linearize tables/images (lossy). Tools: trafilatura, resiliparse, jusText, lynx. Accuracy matters (DCLM). FinePDFs: recrawl + OCR (RolmOCR/Docling) + cleanup for PDFs.
- **Filtering — the algorithmic building block**: given **target data T** and **raw data R**, find subset T′ ⊂ R similar to T. Two-step framework: (1) estimate a model from R and T → scoring function; (2) keep examples by score. Types: **generative model of T** (KenLM: score(x) = p_T(x)) or **classifier** (fastText: score(x) = p(T|x)); keep if score ≥ threshold (stochastically).
  - *Desiderata*: generalize from target (T′ ≠ T) and be extremely fast (R is huge).
  - *Applications*: language ID (fastText lid.176, 176 languages; Dolma keeps p(en) ≥ 0.5), quality filtering, toxicity filtering (Jigsaw Toxic Comments, 6 labels).
  - *Model-based vs rule-based*: C4/Gopher/RefinedWeb/FineWeb/Dolma deliberately avoid model-based filtering; GPT-3/LLaMA/DCLM use it — "becoming the norm."
  - *Case studies*: OpenMathText (rules + KenLM perplexity < 15000 + fastText math classifier → 14.7B tokens beating 20×-larger corpora); GPT-3 (linear classifier on word features, Pareto-9 stochastic retention); LLaMA/RedPajama (positives = Wikipedia-*referenced* pages); phi-1 (GPT-4 "educational value" labels on The Stack Python subset → random forest on codegen embeddings → HumanEval 12.19%→17.68% with 1/3 the steps); Dolma toxicity (Jigsaw classifier).
  - *Scale-dependent filtering*: no single optimal threshold — long training wants more (lower-quality) data; short training wants less (higher-quality) data.
- **Deduplication**: exact dups (mirrors, forks) and near dups (ToS pages, templates — a product description repeated 61,036 times in C4). *Why dedup*: trains faster (fewer tokens) and avoids memorization (copyright/privacy).
  - *Design space*: (1) what is an item (sentence/paragraph/document); (2) how to match (exact, common subitem, fraction of common subitems); (3) what action (remove all / keep one).
  - *The key challenge*: comparing items to items needs **linear-time** algorithms at scale.
- **Hashing**: map items to small hash values. Cryptographic (SHA-256): collision-resistant, slow. Non-crypto (MurmurHash, DJB2, CityHash): fast, used in hash tables. Dedup uses MurmurHash.
- **Exact dedup**: group by hash, keep one per group (MapReduce-style — parallelizes). C4: item = 3-sentence span, exact match, keep one — but removing mid-document spans can break coherence.
- **Jaccard similarity**: J(A,B) = |A∩B| / |A∪B|; near-duplicates = Jaccard ≥ threshold.
- **MinHash**: a hash scheme where **Pr[h(A) = h(B)] = Jaccard(A,B)** — you *want* collisions to track similarity (opposite of normal hashing!). `minhash(S, seed) = min(mmh3.hash(x, seed) for x in S)`; with many seeds, the fraction of matching min-hashes estimates Jaccard.
  - *Why it works*: a random hash induces a random permutation; the minimum element of a set is uniform among its elements, so A and B share the min iff the global-min item among A∪B is in both — probability |A∩B|/|A∪B|.
- **Locality-Sensitive Hashing (LSH)**: sharpen the collision probability into a threshold. Use n = b·r hash functions in b bands of r: A,B collide iff *some* band has *all* r hashes equal. Collision probability: P = 1 − (1 − s^r)^b — an S-curve in similarity s; the phase transition sits at s* = (1/b)^(1/r). Increasing r sharpens/moves the threshold right (harder to match); increasing b moves it left (easier). Real setting (Lee et al. 2021): n=9000, b=20, r=450 → threshold ≈ (1/20)^(1/450) ≈ 0.993. At the threshold, P(collision) ≈ 1 − 1/e.
- **Data mixing**: what distribution p(s) over sources? Baselines: vibes (manual), uniform, proportional to token counts. Two intuitions conflict: upweight high-quality sources, but each source is finite — over-epoching a small high-quality source causes overfitting (example: 10B-token source at p=0.5 over 1T training tokens = 50 epochs!).
  - **UniMax**: sample uniformly with a hard **cap C** on epochs per source: p(s)·train_tokens ≤ C — balances languages in multilingual models.
  - **Regression-based mixing (RegMix)**: define a distribution over mixtures (Dirichlet), train small models, regress mixture → loss (linear/GBT), optimize; hopes: (1) regression accurate at the minimizer, (2) optimal mixtures transfer to large scale.
  - **Simulated epoching**: make small-scale look like large-scale by downsampling *all* sources proportionally — so small-scale runs experience the same epoching regime as the big run, and the fitted optimum transfers.
- **Post-training / synthetic data recipe**: (1) define environments, (2) define tasks/prompts, (3) collect responses from a strong teacher model. Examples: OpenThoughts (1.2M examples from QwQ-32B; sampling 16 responses/prompt helps; better models aren't necessarily better teachers — QwQ-32B > DeepSeek-R1; answer filtering didn't help; small high-quality sources beat large diverse ones); SWE-smith (generate tasks by injecting bugs into repos with an LM; 128 repos → 50K tasks); SWE-Zero (300K agent trajectories without repo-specific execution — strong models have an internal "world model" of code semantics; 150K GitHub PRs; Qwen3-Coder-480B distilled); SWE-rebench (21K interactive Python SWE tasks); SWE-ZERO-12M-trajectories (scale-up with a tiny 1.7B agent).

### Code Example: exact deduplication with hashing

**Code (Python):**
```python
import itertools, mmh3

items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]

# Group by hash, keep one per group (MapReduce-style, parallelizable)
hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)
deduped_items = [next(group) for h, group in hash_items]
# -> "hello" appears only once; "Hello!" is distinct (different bytes)
```

**What the Code Does:** sorts items by their MurmurHash value, groups equal hashes, and keeps the first of each group — exact dedup in linear time.

**Implementation Deep Dive:**
- **Why MurmurHash:** fast non-cryptographic hash — collisions are acceptable here (we're deduplicating, not securing); cryptographic hashing would be needlessly slow at TB scale.
- **Why sort-then-groupby:** identical to hash-bucket grouping but expressed in a MapReduce-friendly way — the pattern parallelizes across workers on sharded data (Assignment 4 processes WET files with `concurrent.futures` the same way).
- **Why "Hello!" ≠ "hello":** hashing operates on bytes — case and punctuation differences make near-identical text hash differently; that's the limitation exact dedup can't fix (hence MinHash).

**Connection to Assignments:** Assignment 4's deduplication task uses exact hashing of paragraphs/documents as a baseline and then MinHash for near-dup removal — this snippet is the starting point, extended to operate on tokenized documents at scale.

### Code Example: MinHash and LSH (the Assignment 4 core)

**Code (Python):**
```python
import mmh3

def jaccard(A, B):
    return len(A & B) / len(A | B)

def minhash(S: set[str], seed: int) -> int:
    """MinHash: Pr[minhash(A) == minhash(B)] = Jaccard(A, B)."""
    return min(mmh3.hash(x, seed) for x in S)

def get_prob_collision(sim, b, r):
    prob_match = sim ** r                     # one band of r hashes all match
    return 1 - (1 - prob_match) ** b          # some band matches

# Verify the estimator:
A = {"1", "2", "3", "4"}; B = {"1", "2", "3", "5"}
true_j = jaccard(A, B)
n = 100
matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
assert abs(sum(matches)/n - true_j) < 0.01   # estimate ≈ true Jaccard

# LSH: b bands of r hashes; collision probability is a sharp S-curve
p80 = get_prob_collision(sim=0.8, b=10, r=10)   # near 1
p20 = get_prob_collision(sim=0.2, b=10, r=10)   # near 0
threshold = (1 / b) ** (1 / r)                  # phase transition location
```

**What the Code Does:** implements MinHash (per-seed minimum hash over the set), verifies the Jaccard estimator empirically, and computes LSH band-collision probabilities — showing the S-curve that turns similarity into a near-threshold decision.

**Implementation Deep Dive:**
- **Why the minimum:** for a random hash, each element of A∪B is equally likely to be the minimum; the min of A equals the min of B exactly when the global minimum lies in A∩B → probability |A∩B|/|A∪B| = Jaccard.
- **Why bands (b·r):** a single hash's collision probability is Jaccard itself — too soft. The "and" within a band (all r must match: s^r) plus the "or" across bands (any of b: 1−(1−s^r)^b) sharpens it into a step function centered at (1/b)^(1/r).
- **Why the parameters matter:** n=9000, b=20, r=450 (a real config) targets near-duplicates at similarity ≥ ~0.993 — you're looking for *nearly identical* documents, not loosely related ones. Tuning (b, r) trades false positives vs false negatives.
- **Why this is Assignment 4's bottleneck:** dedup over hundreds of GB requires linear-time, approximate, parallelizable matching — MinHash + LSH is exactly that (no O(n²) pairwise comparisons).

**Connection to Assignments:** Assignment 4: implement MinHash deduplication over your filtered corpus (item = document/paragraph, match by Jaccard of shingled tokens, keep one per near-dup cluster), and measure the perplexity effect of dedup. The lecture's `get_prob_collision` and threshold math is what you'll cite to justify your (b, r) choice in the write-up.

### Code Example: data mixing and the epoching trap

**Code (Python):**
```python
def num_epochs(source_tokens: float, weight: float, train_tokens: float) -> float:
    return (weight * train_tokens) / source_tokens

# The trap: a small high-quality source gets re-read many times
sources = {"low": 10e12, "high": 10e9}        # 10T vs 10B tokens
p = {"low": 0.5, "high": 0.5}                  # naive 50/50 mixture
train = 1e12                                   # 1T training tokens
epochs = {s: num_epochs(sources[s], p[s], train) for s in sources}
# epochs["high"] == 50  -> 50x epochs on the scarce source: overfitting!

# UniMax: cap the epochs per source (hard constraint)
C = 2
p = {s: min(p[s], C * sources[s] / train) for s in sources}   # then renormalize

# Simulated epoching: downsample all sources proportionally
ratio = 10e9 / 1e12                            # small run / large run
downsampled = {s: sources[s] * ratio for s in sources}   # small-scale analog
```

**What the Code Does:** computes per-source epochs for a naive mixture (exposing the 50-epoch overfitting trap), applies a UniMax-style epoch cap, and shows simulated epoching's proportional downsampling.

**Implementation Deep Dive:**
- **Why epoching matters:** a source seen 50× gets memorized; the "optimal mixture" at small scale (favoring scarce high-quality data) is *not* the optimal mixture at large scale — a scale-dependent effect that breaks naive mixture transfer.
- **Why simulated epoching fixes transfer:** by downsampling all sources to the small-run token budget, small-scale experiments see the same *epoch structure* as the big run; the fitted optimum then transfers (Lecture 9's "make small scale look like large scale" theme).
- **Why caps are the pragmatic solution (UniMax):** hard-capping epochs per source prevents pathological over-epoching without re-fitting anything — the standard trick in multilingual mixing.

**Connection to Assignments:** Assignment 4's final stage mixes sources under a *token budget* for the leaderboard (minimize perplexity given N tokens): the epoching trap and UniMax caps are exactly the considerations for choosing what fraction of your filtered data to keep per source. Assignment 3's scaling-law reasoning applies the same "transfer from small to large" logic.

### Key Takeaways

1. The data pipeline: transform (HTML→text) → filter (classifiers for language/quality/toxicity) → deduplicate (exact + MinHash/LSH) → mix (weights, caps, simulated epoching).
2. Filtering = "find the subset of raw data similar to a target": score by a generative model (KenLM) or classifier (fastText); keep by threshold/stochastically; must generalize and be extremely fast.
3. Deduplication at scale requires linear-time approximate matching: MurmurHash for exact dups; MinHash (Pr[collision] = Jaccard) + LSH bands (S-curve threshold at (1/b)^(1/r)) for near dups.
4. Mixing is scale-dependent: naive weights over-epoch scarce sources (50-epoch trap); fix with UniMax caps or simulated epoching so small-scale optima transfer.
5. Post-training data is increasingly synthetic: teacher models + environments + filtering (OpenThoughts, SWE-Zero) — and smaller high-quality sources often beat larger diverse ones.

### Potential Pitfalls

- **O(n²) dedup**: comparing every document pair is infeasible; always hash-based (exact or MinHash).
- **Wrong LSH parameters**: (b, r) set the threshold — b too small/r too big misses near-dups; the phase transition is at (1/b)^(1/r), so pick for your target similarity.
- **Hash collisions treated as ground truth**: MurmurHash collisions exist; exact dedup is "high precision" but not perfect; MinHash estimates Jaccard with variance — use enough hashes (n).
- **Over-epoching scarce sources**: naive proportional/quality-weighted mixes memorize small sources; apply caps or simulated epoching.
- **Filtering with a fixed threshold across scales**: optimal thresholds shift with training budget (longer training → keep more data).
- **Breaking document coherence in dedup**: C4-style mid-document span removal can produce incoherent text — check what your dedup unit does to downstream quality.
- **Biased positives/negatives in classifier training**: your quality filter inherits the biases of the target data you picked (e.g., RefinedWeb negatives).

### Review Questions

1. **Q:** Why is MinHash's collision probability equal to Jaccard similarity?
   - **A:** A random hash permutation makes every element of A∪B equally likely to be the minimum. The minima of A and B coincide exactly when the global minimum element lies in A∩B — probability |A∩B|/|A∪B| = Jaccard.
2. **Q:** How does LSH turn "collide with probability = similarity" into a hard threshold?
   - **A:** With b bands of r hashes, a pair collides if some band has all r min-hashes equal: P = 1−(1−s^r)^b. This is an S-curve in similarity s whose steep transition sits at s* = (1/b)^(1/r) — near-duplicates (above s*) collide almost surely, unrelated pairs (below) almost never.
3. **Q:** Why does a mixture that's optimal at small scale fail at large scale, and what's the fix?
   - **A:** Scarce high-quality sources get over-epoch'd as training tokens grow (50 epochs in the example) — the small-scale optimum overweights them. Fixes: UniMax hard caps on per-source epochs, or simulated epoching (downsample all sources proportionally) so the small run reproduces the big run's epoch structure.
## Lecture 15: Mid/Post-Training (SFT and RLHF)

*Date: Mon May 18 (Spring 2026) | Instructor: Tatsu Hashimoto | Materials: `lecture_15.pdf`*

### Overview

Pretraining gets you to GPT-3; this lecture covers the path to instructGPT: **supervised fine-tuning (SFT)** on instruction data, then **RLHF** (reinforcement learning from human feedback) with PPO or DPO. It examines what instruction data actually looks like (FLAN → Alpaca → OpenAssistant → Nemotron agentic data), the subtle effects of style/length/knowledge/safety in SFT data, the shift from imitation to reward optimization, pairwise-feedback data collection, and the pitfalls of RLHF: overoptimization and mode collapse.

### Core Concepts & Definitions

- **The G-V gap**: "people don't always write the thing that they prefer" — imitation (SFT) fits p*(y|x) from demonstrations, but what users *prefer* is a reward to *optimize*, not a distribution to imitate. This is the core motivation for RLHF.
- **SFT data lineage**: FLAN (benchmark-style tasks, terse) → Self-Instruct / Alpaca (LLM-generated instruction-following; GPT-3.5/4-generated data; 52K examples) → ShareGPT/Vicuna (real user-assistant conversations) → OpenAssistant (crowdsourced, detailed, knowledge-heavy) → WizardLM → Tulu 3 → Nemotron (agentic: tool calls, multi-turn, AGENTS.md-aware).
  - *What varies*: chattiness/length, detail, tool use, scale, safety. *What it affects*: style (strong length effects in human *and* GPT-judge preferences), benchmark performance (mostly unaffected by style), and factuality.
- **Knowledge extraction vs alignment**: fine-tuning on facts the model doesn't know ("tail knowledge") makes it *hallucinate* (Schulman 2023; Gekhman et al.) — "you may not want to fine-tune on tail knowledge, even if that's the LM use case." SFT is best at *extracting* pretrained behaviors, not adding new ones; adding factually-correct data can sometimes hurt.
- **Safety SFT**: few thousand examples suffice to teach refusal behavior (Llama 2); ~500 safety samples + 500 Alpaca-style samples make models follow safety guidelines. Safety data = scenarios extracted from users.
- **SFT as "pretraining continuation"**: mix instruction data into pretraining (mid-training / two-phase training) then do a short SFT round — scales instruction tuning without catastrophic forgetting (miniCPM, jetMoE; "common knowledge among LLM companies but not documented").
- **RLHF data**: pairwise feedback (chosen vs rejected completions). Sources: humans (crowdsourcing — hard to verify correctness, ethics, demographic biases shift behavior), expert annotators (expensive, growing), and **LM-generated feedback** (GPT-4's agreement is near human inter-annotator level; near-perfect system-level rank correlation) — used by Zephyr (UltraFeedback), Tulu 3, OLMo. Also self-training (Constitutional AI: critique + revise).
  - *Length effects*: RLHF systematically produces longer responses (both human and AI feedback reward it) — a significant, often-unwanted outcome.
- **PPO (Proximal Policy Optimization)** — the original RLHF algorithm (InstructGPT):
  - *Lineage*: policy gradients (too high variance: ∇E[R] = E[R∇log p]) → TRPO (linearize around current policy, trust region) → PPO (clip the importance ratios at ±ε).
  - *In LMs*: actions = tokens; sparse reward at the end (sequence-level); needs a value model + reward model; per-token KL penalty to the reference policy; generalized advantage estimation (GAE) — in the bandit setting γ=λ=1 works, i.e., reward-to-go minus value.
  - *Practice*: outer rollout loop + inner optimization loop; reward shaping = last-token reward + KL; cliprange 0.2; clip the KL when the new policy's logprob < reference's (stability).
  - *Costs*: complicated implementation, memory-hungry value model, extra tuning.
- **DPO (Direct Preference Optimization)** — RLHF without tears:
  - *Idea*: under a *nonparametric assumption* (policy can be any distribution), the RLHF objective's closed-form optimum links the reward to the policy: r(x,y) = β·log(π(y|x)/π_ref(y|x)) + β·log Z(x). Substitute this "implied reward" into the preference (Bradley-Terry/Stiennon) loss → a supervised loss on preference pairs, no reward model, no rollouts.
  - *Interpretation*: "positive gradient on good stuff, negative gradient on bad stuff," scaled by the implied reward model's prediction error.
  - *Variants*: SimPO (no reference), length-normalized DPO (Tulu 3), IPO, etc.
- **RLHF pitfalls**:
  - **Overoptimization**: optimizing the reward past a point degrades true quality — holds for human preferences and noisy LM preferences, but not noiseless LM preferences (overfitting to the reward model).
  - **Mode collapse / entropy loss**: RLHF models stop being proper probabilistic models — no calibration by default.
  - *Contingency*: "lots of results are highly contingent on the specifics of the experiment setup" — PPO sometimes beats DPO and vice versa.

### Code Example: DPO loss (the Assignment 5 optional supplement core)

**Code (Python):**
```python
import torch
import torch.nn.functional as F

def dpo_loss(log_pi_w, log_pi_l, log_ref_w, log_ref_l, beta=0.1):
    """DPO: implicit-reward Bradley-Terry loss on preference pairs.
    log_pi_w/l: log-probs of chosen/rejected under the trained policy.
    log_ref_w/l: log-probs under the frozen reference policy.
    """
    log_ratio_w = log_pi_w - log_ref_w          # implied reward for chosen
    log_ratio_l = log_pi_l - log_ref_l          # implied reward for rejected
    logits = beta * (log_ratio_w - log_ratio_l) # Bradley-Terry logit
    return -F.logsigmoid(logits).mean()          # maximize P(chosen > rejected)
```

**What the Code Does:** implements the DPO objective: the loss is −log σ(β·(r_w − r_l)) where the rewards are implied by the policy/reference log-ratio — a simple binary-classification loss over preference pairs.

**Implementation Deep Dive:**
- **Why no reward model:** DPO reparameterizes the reward as β·log(π/π_ref) (plus a partition constant that cancels in the pair difference), so the "reward model" is the policy itself. This is the nonparametric-optimum trick from the lecture: solve the RLHF constrained optimization in closed form, then plug the implied reward into the preference loss.
- **Why β:** the KL-regularization strength; larger β = closer to the reference policy. It also scales the gradient: the update is "positive on chosen, negative on rejected," weighted by how wrong the implied reward is.
- **Why the reference is frozen:** π_ref anchors the update; without it (SimPO) you need other normalizations (length normalization) to avoid reward hacking.

**Connection to Assignments:** The optional Assignment 5 supplement (SFT + DPO on Llama 3.1 8B with Anthropic HH preference data) implements exactly this loss. The required Assignment 5 (GRPO/RLVR) is the *on-policy* cousin — the lecture's PPO→DPO contrast explains why the course chose GRPO for the required part (no value model, no reward model, verifiable rewards).

### Code Example: PPO-style clipped surrogate objective (conceptual)

**Code (Python):**
```python
def ppo_loss(log_pi_new, log_pi_old, advantages, clip_eps=0.2):
    ratio = torch.exp(log_pi_new - log_pi_old)      # importance ratio
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()    # pessimistic bound

# LM flavor: per-token KL penalty to reference + final-token reward
# advantages = (reward - value) or group-normalized rewards (GRPO, Lecture 16)
```

**What the Code Does:** computes the PPO clipped objective: take the minimum of the unclipped and clipped surrogate, which prevents the policy from moving too far in one update.

**Implementation Deep Dive:**
- **Why clipping:** the importance ratio can explode when the new policy diverges from the old; clipping at [1−ε, 1+ε] bounds the update — TRPO's trust region made practical.
- **Why it's still needed in the LM setting:** rewards are sparse and sequence-level; the per-token KL penalty (to the reference policy) is the main guard against drift, and GAE with γ=λ=1 reduces to reward-minus-value in the bandit setting.
- **Why the course prefers GRPO (next lecture):** PPO's value model is memory-hungry and finicky; with verifiable rewards (math), group-normalized advantages remove the value model entirely.

**Connection to Assignments:** Assignment 5's required part uses **GRPO** (a PPO variant without the value function) — this PPO background is the baseline the assignment's write-up asks you to understand, and the assignment's "policy gradient estimation variants" task (importance-weight clipping) is a direct generalization of this objective.

### Key Takeaways

1. Imitation (SFT) extracts pretrained behaviors; optimization (RLHF) targets preferences — the G-V gap means demonstrations ≠ preferences.
2. SFT data quality beats quantity in subtle ways: style/length drive preference judgments, tail knowledge causes hallucination when fine-tuned, and small amounts of safety/instruction data go a long way.
3. RLHF = reward optimization with KL control: PPO (value model + clipping) is the classic finicky approach; DPO removes the reward model via the nonparametric implied-reward trick.
4. Pairwise feedback is the currency: humans are noisy, demographics shift behavior, LM judges are surprisingly good (near-human agreement) — but all feedback biases length.
5. Watch for overoptimization (reward overfitting) and mode collapse (no longer a calibrated distribution).

### Potential Pitfalls

- **SFT on tail knowledge**: fine-tuning facts the model doesn't know induces hallucination; SFT should mostly extract, not inject.
- **Ignoring length bias**: both human and LLM-judge feedback reward longer responses — length-normalize or debias your metrics (and DPO variants).
- **Skipping KL control in RLHF**: without a reference-policy penalty, the model drifts and collapses.
- **Reward overfitting**: "optimizing for reward overfits past a point" — monitor true quality on held-out evals, early-stop.
- **Trusting noisy annotators**: crowdsourced correctness is hard to verify; expert/LM feedback with checklists is more reliable.
- **Post-norm... no — post-hoc data mixing without mid-training**: forgetting the two-phase recipe causes catastrophic forgetting of general abilities.

### Review Questions

1. **Q:** Why does DPO need no reward model or rollouts, and what assumption makes that possible?
   - **A:** The nonparametric assumption (the optimal policy can be any distribution) lets the KL-regularized RLHF objective be solved in closed form, giving r(x,y) = β·log(π/π_ref) + const. Substituting this "implied reward" into the Bradley-Terry preference loss turns RLHF into a supervised classification loss on preference pairs.
2. **Q:** What is the G-V gap and why does it motivate RLHF over pure SFT?
   - **A:** People don't necessarily write what they prefer (generation ≠ valuation). SFT fits a distribution over demonstrated responses; RLHF maximizes a reward reflecting actual preferences — the two objectives differ even with the same data.
3. **Q:** What are the two main failure modes of RLHF, and how do practitioners mitigate them?
   - **A:** (1) Overoptimization — reward keeps rising while true quality falls; mitigate with KL regularization, held-out evals, early stopping. (2) Mode collapse/entropy loss — the model stops being a calibrated distribution; mitigate with entropy bonuses/KL bounds and length normalization.
## Lecture 16: Post-Training II — RL from Verifiable Rewards (RLVR)

*Date: Wed May 20 (Spring 2026) | Instructor: Tatsu Hashimoto | Materials: `lecture_16.pdf` | Deadlines: Assignment 4 due, Assignment 5 out*

### Overview

RLHF can't scale cleanly (overoptimization); RL from **verifiable rewards** (RLVR) can — optimizing exactly what you want (correct math answers, passing tests) in domains where rewards are exact. This lecture covers PPO→GRPO, GRPO's variants and flaws (baseline validity, length bias), and three case studies: **DeepSeek-R1** (GRPO, R1-zero, SFT+RL recipe, distillation), **Kimi K1.5** (length control, curriculum, RL infra), and **Qwen 3** (low-data RLVR, thinking-mode fusion, agentic RL).

### Core Concepts & Definitions

- **RLVR**: reinforcement learning with rewards that are *verifiable* (answer matches ground truth, tests pass) rather than learned from human preference. This sidesteps reward-model overoptimization and enables clean scaling (the path to o1/r1).
- **Policy gradient lineage**: ∇E[R] = E[R·∇log p] (high variance) → TRPO (trust region) → PPO (clip ratios) → **GRPO** (drop the value model, use group-normalized rewards).
- **PPO in LMs, recap**: actions = tokens; big dense reward at the end; per-token KL penalty; GAE with γ=λ=1 (bandit setting: advantage = reward-to-go − value). Implementation complexity: outer rollout loop, inner optimization, value model (memory-hungry, extra tuning), reward shaping (last-token reward + KL), clipping.
- **Why not PPO / why not DPO for reasoning**: PPO is complicated with a memory-hungry value model; DPO needs pairwise (Bradley-Terry) data and is offline. GRPO: no value function, no pairwise data, online — "you can (and people do) write tiny GRPO implementations."
- **GRPO**: 
  - *Advantage*: for each prompt, sample a *group* of G responses; advantage = (reward − group mean)/group std — a "z-score within group." In the online setting this is just policy gradient with group-normalized rewards.
  - *Objective*: loss = −(1/G)Σ Σ_t [advantage·min(ratio, clip(ratio)) − β·KL] (with per-token KL to the reference and length normalization).
  - *Algorithm*: compute reward per rollout → mean/var normalize per group → compute KL term → gradient update.
- **GRPO's theoretical flaws** (a "minor RL detour"): subtracting the *group mean* is a valid baseline (unbiased), but **dividing by the group standard deviation is NOT a valid baseline** — it biases the gradient. An unbiased variant (Liu et al. 2025) is close to REINFORCE with leave-one-out. Also the standard GRPO objective has a **length bias**: stdev upweights easy/hard questions; length normalization interacts with it; fixes reweight the length-normalizer term.
- **DeepSeek-R1** — the landmark open RLVR recipe:
  - *R1-zero* (controlled setting): base DeepSeek-V3, GRPO with **accuracy reward + format reward** (use thinking tags). Emergent: longer CoTs, "aha moments" — though follow-up analysis (Dr. GRPO) suggests length growth partly comes from the biased objective and the base model already had "aha" behaviors.
  - *R1* adds: **SFT initialization** (long-CoT cold start: ~1K math/science questions with long CoTs from Gemini/R1 — "even a small number of samples is effective for bootstrapping reasoning"), a **language-consistency reward** (RL naturally mixes languages), non-verifiable rewards in stage 2 (V3 as judge, 600K examples), then the usual SFT/RLHF post-training (200K non-reasoning SFT + R1-zero-style RLHF).
  - *No PRMs / no MCTS*: R1 "ended speculations on the necessity of MCTS/PRMs" (process reward models and Monte-Carlo tree search were tried and not needed).
  - *Distillation*: R1 generates 800K CoT traces → distill into Qwen 2.5 — small models can reason.
- **Kimi K1.5**:
  - *Data curation*: math-style curation balancing topics; exclude multiple-choice/true-false (false positives); select only examples the model fails on best-of-8 (difficulty filtering).
  - *RL*: reference-based reward model; DPO-style derivation (nonparametric assumption + solve for r); squared-loss surrogate; baselined policy gradient with regularization.
  - *Length control*: per-batch length reward — λ ∈ [−0.5, 0.5]; correct answers incentivized short; incorrect incentivized shorter than the group's center; enabled late in training.
  - *Curriculum*: difficulty labels, easy→hard; sample problems ∝ (1 − success_rate) to avoid repeating solved ones.
  - *Rewards*: code — ground-truth solutions + generated new test cases; math — 800K samples to train a CoT reward model for answer-equivalence checks.
  - *RL infra*: on-policy rollouts = slow inference; framework switching; uneven long CoT batches — utilization is a first-class problem.
- **Qwen 3**: SFT + reasoning RL with GRPO on **only 3995 examples** (low-data RLVR!); difficulty filtering (best-of-n, remove things the model gets right without CoT, remove validation-similar); **thinking-mode fusion** — mix non-thinking and thinking data with tags, early-stop via a special string; then general RLHF (math/stem abilities dip slightly). **Qwen 3 Coder Next**: mid-training (GitHub, 600B long-context repo-level tokens, PR+RAG, synthetic code QA, agent trajectories) + expert models (web-dev, UX, QA, SWE) + agent RL (800K automated SWE-bench-style environments).
- **Overall picture**: SFT + reasoning RL (GRPO) → (optional distillation) → general RLHF — RLHF comes *after* reasoning RL.

### Code Example: GRPO loss (the Assignment 5 required core)

**Code (Python):**
```python
import torch
import torch.nn.functional as F

def grpo_loss(log_probs, old_log_probs, rewards, kl, beta=0.01, clip_eps=0.2):
    """GRPO with group-normalized advantages.
    log_probs/old_log_probs: [G, T] per prompt group (G responses, T tokens)
    rewards: [G] scalar rewards per response
    kl: [G, T] per-token KL to the reference policy
    """
    # Advantage: z-score within the group (NOT unbiased — see Lecture 16)
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    ratio = torch.exp(log_probs - old_log_probs)          # [G, T]
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    # Token-level loss: min(ratio, clip(ratio)) * adv, plus KL penalty
    per_token = -torch.min(ratio, clipped) * adv.unsqueeze(1) + beta * kl
    return per_token.mean()                                # + length normalization
```

**What the Code Does:** implements the GRPO objective: group-normalized advantage (mean/std), clipped importance ratios per token, and a per-token KL penalty to the reference policy — the complete "tiny GRPO" from the lecture (nano-aha-moment style).

**Implementation Deep Dive:**
- **Why group normalization:** with G rollouts per prompt, the group mean is a valid baseline (reduces variance, unbiased); the std division normalizes reward scales across prompts but *biases* the gradient (the lecture's RL detour — an unbiased version is leave-one-out REINFORCE). Assignment 5 asks you to implement both and compare!
- **Why clip:** same reason as PPO — bound the importance ratio; the assignment explores *unclipped* and *clipped* variants.
- **Why the KL term:** keeps the policy near the reference (the SFT model), preventing reward hacking and collapse; the assignment adds a length-normalization term (reweighting by 1/|y|) because vanilla GRPO biases toward longer responses.
- **Why 1e-4 in the std:** a stability factor when all rewards in a group are equal (std = 0) — a real gotcha in practice.

**Connection to Assignments:** Assignment 5's required part: (1) zero/few-shot + CoT prompting baselines on math data, (2) implement **GRPO** end-to-end (vLLM rollouts + policy-gradient update + metrics), and (3) implement **policy-gradient estimation variants** — baseline choices (group mean vs other), importance-weight clipping, and length normalization. This code block is the mathematical core of all three.

### Code Example: on-policy rollout loop (the RL infrastructure pattern)

**Code (Python):**
```python
def train_step(policy, ref_policy, vllm_server, prompts, tokenizer, g=8):
    # 1. Rollout: sample G responses per prompt from the CURRENT policy (on-policy!)
    responses, response_masks = vllm_server.sample(prompts, n=g)
    # 2. Score: verifiable rewards (exact match / unit tests)
    rewards = score(responses, answers)               # [num_prompts, G]
    # 3. Compute log-probs under current and reference policies
    log_probs = policy.log_probs(prompts, responses)          # [P, G, T]
    old_log_probs = policy.log_probs.detach().clone()          # stored before update
    ref_log_probs = ref_policy.log_probs(prompts, responses)   # KL anchor
    kl = log_probs - ref_log_probs
    # 4. One (or several) optimizer steps on the GRPO loss
    loss = grpo_loss(log_probs, old_log_probs, rewards, kl)
    loss.backward(); optimizer.step()
```

**What the Code Does:** sketches the RLVR training step: sample from the live policy (on-policy), score with a verifiable reward, compute current/old/reference log-probs, and optimize the GRPO loss — with gradient accumulation over micro-batches for memory.

**Implementation Deep Dive:**
- **Why on-policy:** the rollouts must come from the *current* policy; this is what makes RL "on-policy" and why the assignment runs 32 training steps per inference batch (32× off-policy — importance ratios correct the drift).
- **Why vLLM:** rollouts are inference at scale; the assignment's `VLLMServer` (with stop-strings for CoT) is the practical server. Batch unevenness from long CoTs is a real infra problem (Kimi's section).
- **Why reference log-probs:** the KL anchor; computing them once per batch (frozen reference) is the standard efficiency trick.

**Connection to Assignments:** Assignment 5's full GRPO training loop = this sketch: `VLLMCompletion`/`VLLMServer` interfaces, `response_mask` handling, checkpoint save/load (`get_model_and_tokenizer`), metrics (reward mean, KL, response length), and gradient accumulation. The "GRPO variants" task (unclipped/clipped, baseline, length-normalized) is the assignment's estimator section.

### Key Takeaways

1. RLVR is the scalable successor to RLHF for domains with verifiable rewards — it optimizes exactly what you want and avoids reward-model overoptimization.
2. GRPO = PPO minus the value function: group-normalized advantages + clipped ratios + KL penalty; simple enough to write in ~20 lines — but its std-normalization is theoretically biased (unbiased ≈ leave-one-out REINFORCE) and length-biased.
3. DeepSeek-R1's recipe: cold-start SFT (~1K long-CoT examples) → GRPO with accuracy + format (+ language-consistency) rewards → general RLHF; no PRMs/MCTS needed; distillation transfers reasoning to small models.
4. Kimi K1.5 shows the engineering: difficulty filtering, curriculum, length-control rewards, and heavy RL infrastructure (rollout efficiency, uneven batches).
5. Qwen 3 shows low-data RLVR works (GRPO on ~4K examples) and the modern recipe: SFT → reasoning RL → general RLHF, with thinking-mode control and agentic mid-training.

### Potential Pitfalls

- **Using std-normalized advantages as if unbiased**: the group-mean baseline is fine; dividing by std biases the estimator — the assignment's "GRPO vs unbiased variants" comparison exists precisely because of this.
- **Length bias**: vanilla GRPO (and RLHF generally) inflates response length; use length-normalized objectives (and length rewards à la Kimi, enabled late).
- **Reward hacking with format rewards**: R1's format reward (thinking tags) is gameable — verify content, not just tags.
- **Rollouts from stale policies**: going too far off-policy without correction (importance ratios/clipping) degrades the estimator; the assignment's 32× off-policy steps need the clipping variants.
- **Infrastructure neglect**: RL at scale is inference-bound; batch unevenness (long CoTs) wastes GPUs — pad/schedule carefully.
- **Skipping SFT cold-start**: R1-zero works but SFT initialization (small long-CoT set) substantially improves stability and quality.

### Review Questions

1. **Q:** What exactly does GRPO remove from PPO, and what replaces the removed component?
   - **A:** GRPO removes the value (critic) network and its advantage estimation. In its place, advantages are computed per prompt group as (reward − group mean)/group std — a z-score across the G sampled responses — so no value model training is needed.
2. **Q:** Why is GRPO's std normalization theoretically problematic, and what's the unbiased alternative?
   - **A:** Subtracting the group mean is a valid (unbiased) baseline, but dividing by the group standard deviation changes the estimator's expectation — it's not a valid baseline. An unbiased variant (Liu et al. 2025) is close to REINFORCE with leave-one-out baselines (and a modified length-normalization term).
3. **Q:** Why did R1 not need MCTS or process reward models (PRMs)?
   - **A:** R1's GRPO with outcome-level verifiable rewards (accuracy + format) produced strong reasoning (beating o1) without search or per-step supervision; R1's ablations/unsuccessful-attempts section documents that PRMs and MCTS were tried and not required — simplifying the recipe considerably.
## Lecture 17: Alignment — Multimodality

*Date: Wed May 27 (Spring 2026) | Instructor: Percy Liang | Materials: `lecture_17.py` (Guest lectures 18–19: Daniel Selsam, Dan Fu — no public materials)*

### Overview

The world is multimodal; the ultimate goal is an **omni model** that inputs and outputs any combination of modalities. Since Transformers "speak tokens," everything must be converted into tokens. This lecture covers how to *input* non-text data (CLIP/SigLIP contrastive encoders, ViTs), how to *inject* image encodings into LLMs (LLaVA and successors, Qwen-VL 1/2/3), and a step toward *generation* — Chameleon's fully-discrete (VQ-VAE) token approach, plus the training-stability challenges of mixed text/image autoregressive modeling.

### Core Concepts & Definitions

- **The two questions**: (1) how to *input* non-text data (understand images/audio/video), (2) how to *output* non-text data (generate images/audio). Comprehension and generation may demand different things (semantics vs fine-grained detail).
  - *Analogy*: describing a photo (understanding) requires gist; re-drawing it (generation) requires pixel-perfect fidelity. One encoding rarely serves both.
- **CLIP (Contrastive Language-Image Pretraining)**: train an image encoder and a text encoder so that aligned (image, text) pairs have high similarity and misaligned pairs low similarity — a batch-level ranking objective over ~32K pairs.
  - *Data*: 400M (image, caption) pairs from the web (not released; OpenCLIP reproduces with LAION-5B).
  - *Vision encoder*: ViT (or ResNet); best = ViT-L/14@336px; attention pooling; images resized (shorter side 336) + center-cropped.
  - *Text encoder*: GPT-2-style Transformer (63M); encode [BOS]…[EOS], take the [EOS] activation.
  - *Headline result*: zero-shot CLIP beats ResNet-50 trained on ImageNet. Ablation: predicting text from images directly is much less compute-efficient than ranking.
  - *Analogy*: CLIP is like a student who learns "what a cat is" by matching thousands of photos to their captions — never told "this is a cat," but the caption co-occurrence teaches the concept.
- **SigLIP**: same idea but *binary* classification per pair (aligned or not) instead of batch softmax — decouples batch size from the loss, works well at <16K batch sizes, ~5 days on 32 TPUv4 vs CLIP's 10 days on 256 TPUv3. Data: WebLI (billion-scale web pairs, OCR-filtered, 100 languages).
- **LLaVA (Large Language and Vision Assistant)**: the standard VLM template — **vision encoder (CLIP) + projector + LLM (Vicuna)**.
  - *Data*: 158K GPT-4-generated conversations from MS-COCO images (captions/detected objects → questions/conversations).
  - *Training*: Stage 1 (alignment): freeze vision encoder + LM, train only the projector W; Stage 2 (fine-tune): freeze vision encoder, train W + LM.
  - *AnyRes* (LLaVA 1.5+): preserve high resolution by splitting the image into a×b patches at the encoder's native resolution, encode each, concatenate (crucial for OCR).
- **LLaVA-OneVision**: SigLIP encoder (grid features before + after last Transformer layer), Qwen-2-72B decoder, 2-layer MLP projector; handles single image / multiple images / video by tuning resolution per modality so all produce ~equal token counts ("quality over quantity," "easier to harder" training); cross-modal transfer (OCR → GUI agents, visual prompting → video).
- **Qwen-VL**: OpenCLIP ViT-bigC + 1-layer cross-attention adaptor (2D positional encodings, fixed 256 length) + special tokens (<img>, <box>, <ref>); 3-stage training (frozen-LM alignment → all-params task data at higher resolution → instruction tuning with frozen encoder).
- **Qwen2-VL**: larger ViT (675M); **dynamic resolution** (224×224 patches, 2×2 → 66 tokens); video 2 fps, max 16K tokens; **MRoPE** (multimodal RoPE: temporal/width/height axes); 3-stage training.
- **Qwen3-VL**: SigLIP-2 encoder with **interleaved MRoPE** ([t w h t w h…] instead of [t t t w w w h h h]) + explicit video timestamps; **square-root-normalized per-token loss** to balance text vs long video sequences; **DeepStack** cross-layer adapter (inject visual info into multiple layers); 4-stage pretraining (adapter → all-params at 8K/32K/256K lengths) + SFT on long CoT + distillation + RL. SOTA, but "lots of data work, not many details."
- **Chameleon (toward omni)**: map *everything* into discrete tokens: images via a **VQ-VAE** (encode 512×512 image → 1024 tokens from an 8192-codebook; decode back, minimize reconstruction loss), then train one autoregressive Transformer on mixed text/image token streams — analyze *and* generate in a uniform way.
  - *Training*: Stage 1: 80% unsupervised (2.9T text tokens + 1.5T text/image + 400B interleaved); Stage 2: 20% high-quality mix.
  - *Stability*: text tokens have low entropy, image tokens high entropy → norm growth and **logit drift**; fixes: **QK-norm** and **z-loss** (from Lecture 3).
  - *Tradeoff*: elegant (pure next-token prediction) but less performant — discretization loses information (OCR suffers).
- **Training stability with mixed modalities**: balance images/video (lower information density) vs text; per-token loss normalization (Qwen3-VL's sqrt trick) prevents video from dominating.

### Code Example: contrastive (CLIP-style) loss — conceptual

**Code (Python):**
```python
import torch
import torch.nn.functional as F

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """Contrastive loss: align each image with its caption, in both directions.
    image_embeds, text_embeds: [batch, d] (L2-normalized)
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.T / temperature      # [B, B]
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, labels)                 # image -> text
    loss_t = F.cross_entropy(logits.T, labels)               # text -> image
    return (loss_i + loss_t) / 2
```

**What the Code Does:** builds the B×B similarity matrix between images and captions in a batch, and trains each image to prefer its own caption over all others (and vice versa) via symmetric cross-entropy — the CLIP objective (SigLIP replaces this with per-pair binary sigmoid losses).

**Implementation Deep Dive:**
- **Why symmetric (both directions):** matching must be bidirectional — images retrieve text and text retrieves images; this also doubles the training signal.
- **Why a large batch:** the contrastive loss's difficulty (negatives) scales with batch size — CLIP used ~32K; SigLIP's binary reformulation decouples batch size from the loss (its key efficiency win).
- **Why temperature:** scales logits into a usable range for cross-entropy (learnable in practice).
- **Why normalization:** cosine similarity (L2-normalized) prevents embedding-magnitude artifacts.

**Connection to Assignments:** Not directly implemented in assignments, but the *architecture pattern* (encoder + projection + LLM) and the *evaluation* mindset (Lecture 12's benchmarks incl. multimodal HLE) apply. Assignment 4's quality classifiers are also trained with contrastive-adjacent ideas (positive/negative example pairs → fastText).

### Code Example: LLaVA-style vision-language model (architecture sketch)

**Code (Python):**
```python
import torch
from torch import nn

class LLaVA(nn.Module):
    def __init__(self, vision_encoder, projector, llm):
        super().__init__()
        self.vision_encoder = vision_encoder   # frozen CLIP/SigLIP ViT
        self.projector = projector             # e.g., 2-layer MLP (or linear W)
        self.llm = llm                          # the language model

    def forward(self, images, input_ids, pixel_values=None):
        # 1. Encode images -> patch embeddings (or grid features)
        img_feats = self.vision_encoder(images)          # [B, num_patches, d_vit]
        # 2. Project into the LLM's embedding space
        img_tokens = self.projector(img_feats)           # [B, num_patches, d_model]
        # 3. Interleave image tokens with text tokens, then standard LM forward
        #    (e.g., <image> tokens placed in the sequence, causal masking)
        return self.llm(input_ids=input_ids, img_tokens=img_tokens)
```

**What the Code Does:** sketches the canonical VLM: a frozen vision encoder produces patch features, a projector maps them into the LLM embedding space, and the LM consumes the interleaved image/text token stream.

**Implementation Deep Dive:**
- **Why freeze the vision encoder:** LLaVA stage 1 (alignment) freezes encoder + LM and trains only the projector — cheap and stable; stage 2 unfreezes the LM. Qwen-VL stage 1 instead freezes the LM and trains encoder+adaptor — the "which part to freeze" choice is a recurring design decision.
- **Why the projector matters:** a linear projection (LLaVA) is minimal; 2-layer MLPs (OneVision) and cross-attention adaptors (Qwen-VL) trade capacity vs tokens. The projector maps d_vit → d_model.
- **Why resolution handling (AnyRes) is essential:** CLIP-style resize+crop destroys fine detail (OCR, charts); splitting into patches at native resolution preserves it — at the cost of more tokens (hence OneVision's per-modality resolution tuning).

**Connection to Assignments:** This is the architecture lineage behind modern open VLMs; while Assignment 1–5 are text-only, the *tokenization* lesson (Lecture 1) — "convert everything to tokens" — is the same principle extended to images (VQ-VAE in Chameleon) and the training-stability tricks (QK-norm, z-loss) are ones you'll meet again in Assignment 1's model implementation.

### Key Takeaways

1. The omni-model goal: any input → any output; today's answer is "convert everything to tokens" (discrete for text/Chameleon images, continuous embeddings for CLIP-style injection).
2. CLIP/SigLIP give text-aligned image encoders via contrastive learning; SigLIP's binary loss is far more efficient (batch-size decoupled).
3. The standard VLM recipe: frozen/semi-frozen vision encoder + projector + LLM, with staged training (alignment → fine-tuning → instruction tuning) and resolution-preserving tricks (AnyRes, dynamic resolution, MRoPE).
4. Generation demands discrete or continuous *output* tokens: Chameleon's VQ-VAE approach is elegant but loses fine detail; diffusion is the alternative for high-fidelity generation.
5. Mixed-modality training is unstable (logit drift, norm growth): QK-norm, z-loss, and per-token loss normalization are the practical fixes.

### Potential Pitfalls

- **Resolution destruction**: resize+crop to 336×336 kills OCR/reading tasks; use AnyRes/dynamic-resolution pipelines.
- **Modality imbalance**: long video sequences dominate the loss; normalize per-token (sqrt trick) or balance data.
- **Unstable mixed training**: image tokens have high entropy; without QK-norm/z-loss, logits drift and norms grow.
- **Frozen-encoder mistakes**: forgetting which stage freezes what (encoder vs LM vs projector) breaks the staged-training recipe.
- **Discrete-token information loss**: VQ-VAE discretization hurts fine-grained tasks (OCR); know the tradeoff before choosing Chameleon-style modeling.
- **Positional encoding conflicts**: RoPE + video/multi-image axes needs MRoPE/interleaving; naive positional embeddings don't handle 3D (time, height, width) token grids.

### Review Questions

1. **Q:** Why is SigLIP so much cheaper than CLIP at the same quality?
   - **A:** CLIP's loss is a B×B softmax over the batch (contrastive over all pairs) — it needs huge batches to have enough negatives. SigLIP treats each (image, text) pair as an independent binary classification (sigmoid), decoupling batch size from the loss — so it trains well with far fewer TPUs (5 days on 32 TPUv4 vs CLIP's 10 days on 256 TPUv3).
2. **Q:** What are the three stages of LLaVA training, and what's frozen in each?
   - **A:** Stage 1 (feature alignment): freeze vision encoder and LLM, train only the projector W. Stage 2 (end-to-end fine-tuning): freeze vision encoder, train projector + LLM on instruction data. (Qwen-VL's variant trains encoder+adaptor with frozen LM first.)
3. **Q:** Why does Chameleon need a VQ-VAE, and what's its main weakness?
   - **A:** To make images *generatable* by the same autoregressive Transformer as text, images must become discrete tokens — the VQ-VAE maps image patches to codebook indices (512×512 → 1024 tokens) with a reconstruction loss. Weakness: discretization loses fine-grained information (e.g., OCR detail), making it less performant than continuous-encoder approaches for understanding tasks.
## Assignments Summary

All five assignments are completed in **PyTorch with minimal scaffolding**: the repos give you unit tests and adapter interfaces for correctness checking, but you implement everything from scratch. Work locally on CPU for correctness, then run on GPU (Modal-sponsored for enrolled students) for training/benchmarking. Deadlines are listed in the schedule. Submissions via Gradescope; some assignments have leaderboards (minimize perplexity given a fixed budget). The **AI policy** (Spring 2025–26) prohibits AI tools from implementing any part of the assignments — use them only for conceptual questions and API documentation, with the repo's `AGENTS.md` pasted into chat conversations.

---

### Assignment 1: Basics (Tokenization, Model, Training) — `assignment1-basics`

**Objective:** Implement all components needed to train a standard Transformer language model, and actually train a minimal one (TinyStories, then OpenWebText).

**Key implementation tasks (from the handout):**
1. **BPE tokenizer** (Section 2): learn merges on a corpus (vocabulary initialization, pre-tokenization with the GPT-2-style regex, merge computation, special tokens like `<|endoftext|>`, parallel pre-tokenization, optimized merging); encode/decode with round-trip guarantees.
2. **Transformer LM** (Section 3): token embeddings, **RMSNorm**, **RoPE**, **causal multi-head self-attention** (QKV projections, causal masking, softmax), position-wise **SwiGLU feed-forward**, pre-norm blocks, LM head → next-token probabilities; dimension conventions documented (B, S, D, etc.).
3. **Cross-entropy loss + AdamW optimizer** (Section 4): the standard NLL loss, and an AdamW implemented as a `torch.optim.Optimizer` subclass with first/second moments, weight decay, and LR schedule (with a worked SGD example in the handout).
4. **Training loop** (Section 5): checkpoint save/load (serialize model + optimizer state), training config (batch size, LR, etc.), decoding support (greedy/sampling), and perplexity evaluation.
5. **Resource accounting**: FLOPs and memory for every Transformer component at a given config.
6. **Run**: train a small model on TinyStories (see sample output in handout), then on OpenWebText.
7. **Leaderboard**: minimize OpenWebText perplexity given 45 minutes on a B200 (see last year's leaderboard in the lecture).

**Most critical lecture concepts:** Lecture 1 (tokenization/BPE — direct blueprint), Lecture 2 (resource accounting, Adam-family optimizers, training loop), Lecture 3 (the modern architecture variant: pre-norm, RMSNorm, RoPE, SwiGLU, no biases).

---

### Assignment 2: Systems (Profiling, Kernels, Distributed) — `assignment2-systems`

**Objective:** Profile and benchmark the Assignment 1 model with advanced tools, optimize attention with your own Triton implementation of FlashAttention-2, and build memory-efficient distributed training.

**Key implementation tasks (from the handout):**
1. **Benchmarking + profiling harness**: measure per-operation runtime and memory; use Nsight Compute and NVTX ranges; answer questions like "which kernel takes the most runtime in forward+backward?"; mixed-precision (autocast) datatype questions.
2. **Activation checkpointing**: wrap TransformerBlocks with recomputation; verify memory savings with `saved_tensors_hooks` (pack/unpack hooks).
3. **FlashAttention-2 Triton kernel**: forward (tiled QKᵀ, online softmax, masked store) and backward (recompute scores), with `torch.autograd.Function`; a WeightedSum autograd.Function tutorial is included; precision/tolerance guidelines.
4. **Distributed data parallel training**: gradient all-reduce (backward hooks, async communication), benchmark vs single-GPU.
5. **Optimizer state sharding**: reduce-scatter gradients, local AdamW update, all-gather parameters (ZeRO-1).
6. **Fully sharded data parallel (FSDP)**: shard parameters too, all-gather on demand in forward/backward, reduce-scatter gradients; verify memory scaling; train a model that wouldn't fit otherwise.
7. **Leaderboard**: submit your best result.

**Most critical lecture concepts:** Lecture 5 (GPU model, roofline), Lecture 6 (Triton kernels: tiling, online softmax, fusion — the four kernel patterns), Lecture 7 (collectives, DDP, bandwidth measurement), Lecture 8 (ZeRO 1–3/FSDP memory + communication accounting, overlap).

---

### Assignment 3: Scaling (Scaling Laws) — `assignment3-scaling`

**Objective:** Fit scaling laws to predict the compute-optimal model configuration for a large training run (48 B200 hours) using a limited budget (12 B200 hours) of experiments.

**Key implementation tasks (from the handout):**
1. Query the **training API** (HTTP endpoints, X-API-Key header): submit experiments with hyperparameters (layers, embedding size, heads, batch size, LR, training tokens) + max wall-clock time; poll status; receive validation loss.
2. Design a search-space strategy: IsoFLOPs profiles (Chinchilla method 2) — for each compute budget, sweep model sizes with D = C/6N; or joint fits (method 3), Kaplan-style, or muP-informed approaches (the write-up welcomes them).
3. Fit power laws N_opt(C) and D_opt(C); **extrapolate** to the 48 B200-hour budget; submit predicted optimal hyperparameters and predicted final validation loss.
4. Write-up: complete methodology detailed enough to reproduce; part of the grade is the *actual performance* of your predicted model.

**Most critical lecture concepts:** Lecture 9 (scaling-law foundations, IsoFLOPs procedure, 6ND rule, power-law fitting, Chinchilla vs Kaplan), Lecture 11 (case studies: WSD restarts to make fitting cheap, DeepSeek/MiniCPM recipes, muP, critical batch/LR scaling).

---

### Assignment 4: Data (Curation Pipeline) — `assignment4-data`

**Objective:** Convert raw Common Crawl dumps into usable pretraining data via transformation, filtering, and deduplication; measure the impact on model quality.

**Key implementation tasks (from the handout):**
1. **Convert Common Crawl HTML to text** (WARC/WET files): parse with trafilatura/resiliparse; inspect WET vs WARC; process in parallel (`concurrent.futures`).
2. **Filter**: (a) rule-based filters (Gopher/C4 criteria — e.g., remove documents with too few words, too little alphabetic content, boilerplate, non-English); (b) **harmful content classifiers** (provided NSFW and hate-speech classifiers); (c) **PII removal** (emails, IP addresses); (d) optional additional filters.
3. **Deduplicate**: exact hashing, then **MinHash** near-duplicate detection (with Jaccard similarity and LSH banding); tokenize with your Assignment 1 BPE tokenizer where needed.
4. **Train**: build a tokenized dataset, train a small LM on filtered vs unfiltered data (starter code included for tokenization + training with saved token IDs).
5. **Leaderboard**: minimize perplexity given a token budget.

**Most critical lecture concepts:** Lecture 13 (sources: Common Crawl WARC/WET, copyright, dataset lineages, quality classifiers), Lecture 14 (the full pipeline: transformation, filtering framework — target vs raw data, KenLM/fastText classifiers, toxicity; deduplication — MurmurHash, MinHash, LSH thresholds, (b, r) math; mixing — epoching trap, UniMax, simulated epoching), Lecture 12 (perplexity as the evaluation metric).

---

### Assignment 5: Alignment and Reasoning RL (GRPO) — `assignment5-alignment`

**Objective:** Train LMs to reason when solving math problems via supervised prompting baselines and reinforcement learning (GRPO), and explore policy-gradient estimator variants. *Optional Part 2:* SFT + DPO for instruction following and safety.

**Key implementation tasks (required, from the handout):**
1. **Zero-shot, few-shot, and chain-of-thought prompting** baselines on math evaluation data.
2. **GRPO implementation**: vLLM rollout server (`VLLMCompletion`/`VLLMServer`, stop-strings), tokenization with response masks, log-prob computation under policy/old/reference, group-normalized advantages, clipped ratio loss with KL penalty + length normalization, gradient accumulation, full training loop with metrics (reward, KL, length).
3. **Policy gradient estimation variants**: baselines (group mean vs others), importance-weight clipping (unclipped vs clipped), length-normalization reweighting — analyze variance/expectation; the handout derives the "unclipped token-level reweighting" estimator and its clipped version.
4. Run on a small model (e.g., OLMo-2-0425-1B) with ~32× off-policy steps per inference batch; measure accuracy gains.

**Optional Part 2 (supplement — SFT + DPO):**
1. Zero-shot baselines on MMLU, GSM8K, AlpacaEval, SimpleSafetyTests, Anthropic HH.
2. **Supervised fine-tuning** of Llama 3.1 8B on instruction-response data.
3. **Direct Preference Optimization (DPO)** on pairwise preference data (Anthropic HH), with AlpacaEval/safety evaluation via a Llama 3.3 70B judge.

**Most critical lecture concepts:** Lecture 15 (SFT data, RLHF objectives, PPO vs DPO, overoptimization/mode collapse), Lecture 16 (GRPO — the exact algorithm you implement, its biases and variants; R1/Kimi/Qwen3 recipes; RL infra), Lecture 10 (inference infrastructure — vLLM, KV cache, generation), Lecture 12 (evaluation benchmarks used in the supplement).

---

### Assignment-to-Lecture map

| Assignment | Primary lectures | Core skills |
|---|---|---|
| 1 — Basics | 1, 2, 3 | BPE, Transformer blocks, AdamW, training loop, resource accounting |
| 2 — Systems | 5, 6, 7, 8 | profiling, Triton kernels, FlashAttention-2, DDP/ZeRO/FSDP |
| 3 — Scaling | 9, 11 | scaling-law fitting, IsoFLOPs, extrapolation, prediction |
| 4 — Data | 12, 13, 14 | HTML→text, classifiers, MinHash/LSH dedup, mixing, PPL evals |
| 5 — Alignment | 10, 15, 16 | prompting, GRPO, policy-gradient variants, (DPO/SFT optional) |

---

## Glossary

- **Activation checkpointing** — recomputing forward activations during backward instead of storing them; trades memory for compute (also: gradient checkpointing / rematerialization).
- **Adam / AdamW** — adaptive optimizers maintaining per-parameter first/second moments (AdamW adds decoupled weight decay); the standard LLM optimizer (8 bytes/param of fp32 state).
- **All-reduce / reduce-scatter / all-gather** — collective operations: sum across devices and replicate; sum and shard; gather shards to all. All-reduce = reduce-scatter + all-gather.
- **Arithmetic intensity** — FLOPs per byte moved; compared against the hardware's accelerator intensity (peak FLOP/s ÷ bandwidth) to determine memory-bound vs compute-bound.
- **AWQ (Activation-aware Weight Quantization)** — PTQ that keeps a small fraction of weights (those hit by large activation channels) in higher precision.
- **Bank conflict** — shared-memory access serialization when multiple threads in a warp hit the same bank; mitigated by swizzling/padding.
- **bf16 / fp16 / fp8 / fp4** — float formats: bf16 = fp32-range at 2 bytes; fp16 = smaller range (underflow risk); fp8 (E4M3/E5M2) and NVFP4 for lower-precision compute/inference.
- **BPE (Byte-Pair Encoding)** — subword tokenizer: start with bytes, repeatedly merge the most frequent adjacent pair until the target vocab size.
- **Chinchilla scaling laws** — compute-optimal scaling: N_opt ∝ C^0.5, D_opt ∝ C^0.5, ≈20 tokens/param (vs Kaplan's 0.73/0.27).
- **Collective operation** — a communication pattern across all devices (broadcast, scatter, gather, reduce, and "all-" variants).
- **Compute-bound vs memory-bound** — whether computation time or memory-transfer time dominates (arithmetic intensity vs accelerator intensity).
- **Continuous batching** — iteration-level scheduling that adds/removes requests per decode step instead of static batches.
- **DDP (Distributed Data Parallel)** — replicate parameters, shard the batch, all-reduce gradients.
- **DPO (Direct Preference Optimization)** — RLHF without a reward model: optimize the implied reward β·log(π/π_ref) via a preference-pair classification loss.
- **Deduplication** — removing exact and near-duplicate documents (exact hashing; MinHash + LSH for near-dups) to save compute and avoid memorization.
- **Fair use** — copyright doctrine (four factors: purpose, nature, amount, market effect) under which some LLM training has been judged legal in specific instances.
- **FlashAttention** — fused, tiled attention with online (telescoping) softmax; avoids materializing the S/P matrices; O(1) block memory.
- **FLOPs vs FLOP/s** — floating-point operations (work) vs operations per second (speed); MFU = achieved/promised.
- **FSDP / ZeRO-3** — shard parameters, gradients, and optimizer state; all-gather params on demand, reduce-scatter grads; ~3×#params communication.
- **GQA (Grouped-Query Attention)** — fewer KV heads than query heads to shrink the KV cache; MQA = 1 KV head.
- **GRPO (Group Relative Policy Optimization)** — PPO without the value model: advantages = (reward − group mean)/group std over G rollouts per prompt; biased std-normalization and length bias are known flaws.
- **IsoFLOPs** — scaling-law method: fix compute C_i, sweep model sizes, take min loss; fit power laws for N_opt(C), D_opt(C).
- **KV cache** — stored key/value vectors enabling O(1) per-token generation cost (vs O(T) recomputation); its size drives generation memory-boundedness.
- **Linear attention** — attention with identity kernel: Q(KᵀV) = (QKᵀ)V, O(n) in sequence length; recurrent form S_t = S_{t−1} + k_t v_tᵀ (duality).
- **LSH (Locality-Sensitive Hashing)** — b bands of r min-hashes; collision probability 1−(1−s^r)^b is an S-curve with threshold (1/b)^(1/r).
- **Memory coalescing** — warp accesses merging into 128-byte transactions; needed for fast HBM reads.
- **MFU (Model FLOPs Utilization)** — achieved FLOP/s ÷ peak FLOP/s; ≥0.5 is good.
- **MinHash** — hash scheme where Pr[h(A)=h(B)] = Jaccard(A,B); the min over the set under a random hash.
- **MLA (Multi-head Latent Attention)** — compress K/V into a low-dim latent c (DeepSeek v2); shrinks KV cache; needs extra non-rotated dims for RoPE.
- **MoE (Mixture of Experts)** — many expert FFNs + a router; more parameters at constant per-token FLOPs; trained with top-k routing + balancing losses.
- **muP (Maximal Update Parametrization)** — width-aware init + LR scaling so optimal hyperparameters transfer across model sizes; breaks with RMSNorm gains / strong weight decay.
- **PagedAttention** — virtual-memory-style paging of the KV cache (vLLM): non-contiguous blocks, prefix sharing, copy-on-write.
- **Perplexity** — exp(mean NLL); (1/p(D))^(1/|D|); lower is better; ~vocab size for random guessing.
- **Prefill vs decode** — inference phases: process prompt in parallel (compute-bound) vs generate one token at a time (memory-bound).
- **PPO (Proximal Policy Optimization)** — RL algorithm with clipped importance ratios, value model, and KL control; the classic RLHF optimizer.
- **QK-norm / z-loss** — stability tricks: normalize Q,K before softmax; penalize log-sum-exp to prevent logit drift.
- **RLHF** — reinforcement learning from human (or AI) pairwise feedback to optimize preferences with KL control.
- **RLVR** — RL with verifiable rewards (exact answers, tests): scalable, avoids reward-model overoptimization.
- **RMSNorm / LayerNorm** — normalization layers: RMSNorm rescales by RMS without mean/bias; LayerNorm centers and scales.
- **Roofline model** — plot of achievable FLOP/s vs arithmetic intensity; kink = accelerator intensity.
- **RoPE (Rotary Position Embeddings)** — rotate Q/K pairs by position so attention scores depend only on relative position.
- **Scaling law** — power-law relation between loss and data/model/compute; enables small-scale → large-scale prediction.
- **Speculative sampling** — draft model proposes tokens, target scores in parallel; modified rejection sampling is *exact* for the target.
- **SFT (Supervised Fine-Tuning)** — training on instruction-response demonstrations; best at extracting pretrained behaviors.
- **SwiGLU / GeGLU** — gated FFN activations (swish/gaussian-error gate ⊙ linear); standard in post-2023 models; FFN dim ≈ 8/3× model dim.
- **Tensor parallel** — shard weight matrices across devices, all-gather activations per layer; needs NVLink-class interconnects.
- **Tiling** — computing in shared-memory tiles to cut HBM traffic (matmul, FlashAttention); factor-T reduction in global reads.
- **Tokenizer** — encode(text)→token IDs and decode(IDs)→text; BPE is the standard; must round-trip and handle UTF-8 + special tokens.
- **Upcycling** — initializing an MoE from a pretrained dense model.
- **WSD learning rate** — warmup–stable–decay schedule; restartable for cheap scaling-law fitting.
- **ZeRO-1 / ZeRO-2** — shard optimizer state only / + gradients; same communication as DDP with better memory.

---

*Notes compiled from the public CS336 website (Spring 2026 offering; Spring 2024/2025 archives for scheduling cross-reference), the executable lecture files, lecture slide PDFs, and assignment handouts. Video recordings (private YouTube playlist), Slack, and the Lecture 7 parallel-PDF (private GitHub repo) are restricted and noted in-place.*

{% endraw %}
