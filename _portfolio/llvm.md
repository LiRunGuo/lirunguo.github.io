---
title: "LLVM — Value Tracking & Interprocedural Analysis"
excerpt: "Contributor to LLVM: two merged changes — ValueTracking now proves x != y from any condition on x - y that excludes zero, and the Attributor's AAPointerInfo follows only the casts that preserve a pointer instead of asserting on integers it mistook for pointers."
collection: portfolio
category: contribution
order: 5
permalink: /portfolio/llvm/
---

I contribute to [LLVM](https://github.com/llvm/llvm-project), focusing on the analyses that GPU and accelerator compilers build on, where a wrong assumption about what a value is surfaces far from where it was made.

### Merged contributions

**[PR #225200 — [ValueTracking] Infer x != y from (x - y) pred C when 0 pred C is false](https://github.com/llvm/llvm-project/pull/225200)** · Merged October 17, 2026 (+206 / −9 across 2 files). Fixes [#223345](https://github.com/llvm/llvm-project/issues/223345).

LLVM already folded `x == y` to false when an assumption said `x - y` equals a nonzero constant, but a range on the same subtraction was ignored even when it excludes zero. `opt -O3` kept the comparison below and dropped the assumption instead:

```llvm
%diff = sub i64 %x, %y
%cond = icmp ugt i64 %diff, 1
call void @llvm.assume(i1 %cond)
%r = icmp eq i64 %x, %y      ; should fold to false
```

Since `x - y == 0` exactly when `x == y`, with no `nsw` or `nuw` needed, the implication generalizes to any `(x - y) pred C` for which `0 pred C` is false. The change replaces the equality-only check in `isImpliedCondICmps` with the existing `cmpExcludesZero` helper, and lets `findValuesAffectedByCondition` register the operands of the `sub` in ordered comparisons with a constant, so both `llvm.assume` calls and dominating branch conditions benefit. The rule is checked with Alive2, and the InstCombine tests cover unsigned and signed predicates, commuted operands, pointer differences, dominating conditions, and negative cases where `0 pred C` can hold and nothing must fold.

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
