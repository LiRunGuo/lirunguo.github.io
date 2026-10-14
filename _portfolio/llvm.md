---
title: "LLVM — Interprocedural Optimization"
excerpt: "Contributor to LLVM: merged an Attributor fix so AAPointerInfo follows only the casts that preserve a pointer, instead of tracking integers from ptrtoaddr and byte-type bitcasts as pointers until an assertion fires."
collection: portfolio
category: contribution
order: 5
permalink: /portfolio/llvm/
---

I contribute to [LLVM](https://github.com/llvm/llvm-project), focusing on the analyses that GPU and accelerator compilers build on, where a wrong assumption about what a value is surfaces far from where it was made.

### Merged contribution

**[PR #225290 — [Attributor] Only follow addrspacecasts in AAPointerInfo](https://github.com/llvm/llvm-project/pull/225290)** · Merged October 13, 2026 (+164 / −18 across 3 files). Fixes [#221852](https://github.com/llvm/llvm-project/issues/221852).

`AAPointerInfo` tracks how a pointer flows through a function so the Attributor can reason about the memory it reaches. It followed every cast of a pointer except a `ptrtoint` instruction. A `ptrtoaddr`, a `ptrtoint`/`ptrtoaddr` constant expression, or a bitcast to a byte type therefore produced an integer that was still tracked as a pointer. Once that value reached a PHI, `getPointerAddressSpace()` cast its type to `PointerType` and asserted:

```llvm
; opt -passes=attributor
define void @foo() {
entry:
  %0 = alloca i8
  %1 = ptrtoaddr ptr %0 to i64
  br label %exit
exit:
  phi i64 [ %1, %entry ]
  ret void
}
```

```text
Assertion `isa<To>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
```

Rather than add the missing cast kinds to the exclusion list one at a time, the fix inverts the rule: the analysis now follows only `addrspacecast`, the one cast that keeps the value a pointer. New regression tests in `pointer-info.ll` send each cast kind that used to leak through (`ptrtoaddr`, `ptrtoint` and `ptrtoaddr` constant expressions, and a bitcast to a byte type) into a PHI, and an existing test that relied on a no-op pointer-to-pointer bitcast now loads through the pointer directly.

[View my LLVM pull requests](https://github.com/llvm/llvm-project/pulls?q=is%3Apr+author%3ALiRunGuo)
