---
title: "附录 A · CSAPP 核心概念与命令速查表"
collection: course-notes
chapter: true
permalink: /course-notes/cmu-15213-csapp/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/cmu-15213-csapp/) · [← l25](/course-notes/cmu-15213-csapp/l25)

{% raw %}
## 附录 A · CSAPP 核心概念与命令速查表

> 本附录是全书 24 讲的压缩索引：按主题列成表格，供随时查阅。
> 表中每条都可以在正文对应章节找到展开讲解。

### A.1 数据表示速查

#### A.1.1 C 类型 → 字节数 → 汇编后缀（x86-64）

| C 类型 | 字节 | 位宽 $w$ | 汇编后缀 | 典型指令 | 说明 |
|---|---|---|---|---|---|
| `char` | 1 | 8 | `b` (byte) | `movb` | 有符号性由实现定义（x86-64 Linux 上为 **signed**） |
| `short` | 2 | 16 | `w` (word) | `movw` | — |
| `int` | 4 | 32 | `l` (long word) | `movl` | 默认有符号 |
| `long` | 8 | 64 | `q` (quad word) | `movq` | LP64：Linux/macOS 为 8 字节，Windows 为 4 字节 |
| `long long` | 8 | 64 | `q` | `movq` | 与 `long` 同宽 |
| `char *`（任何指针） | 8 | 64 | `q` | `movq` | 指针即 64 位地址 |
| `float` | 4 | 32 | `ss` (scalar single) | `movss` | 存于 `%xmm` 低 32 位 |
| `double` | 8 | 64 | `sd` (scalar double) | `movsd` | 存于 `%xmm` 低 64 位 |
| `long double` | 16 | 80(有效) | — | `fldt` 等 | x87 扩展精度，ABI 规定按 16 字节对齐，**不走 `%xmm`** |

> 记忆钩子：`b`=1、`w`=2、`l`=4、`q`=8；浮点后缀 `ss`/`sd` = **s**calar **s**ingle / **s**calar **d**ouble。
> AT&T 语法下 8 位寄存器称 `%al` 一类（`b` 后缀），16 位为 `%ax`（`w`），32 位为 `%eax`（`l`），64 位为 `%rax`（`q`）。

#### A.1.2 取值范围表（补码与无符号）

| 类型 | $w$ | 最小值 | 最大值的十进制 | 最大值的十六进制 |
|---|---|---|---|---|
| `int` | 32 | $-2^{31}$ = **-2,147,483,648** | **2,147,483,647** | `0x7FFFFFFF` |
| `unsigned` | 32 | 0 | **4,294,967,295** | `0xFFFFFFFF` |
| `long` | 64 | $-2^{63}$ = **-9,223,372,036,854,775,808** | **9,223,372,036,854,775,807** | `0x7FFFFFFFFFFFFFFF` |
| `unsigned long` | 64 | 0 | **18,446,744,073,709,551,615** | `0xFFFFFFFFFFFFFFFF` |

| 记号 | 值 | 位模式 |
|---|---|---|
| `UMin` | 0 | `000…0` |
| `UMax` | $2^w - 1$ | `111…1` |
| `TMin` | $-2^{w-1}$ | `100…0` |
| `TMax` | $2^{w-1} - 1$ | `011…1` |
| `-1` | $-1$ | `111…1`（与 `UMax` 位模式相同） |

常用速记：`INT_MAX` = `0x7FFFFFFF`，`INT_MIN` = `0x80000000`，`UINT_MAX` = `0xFFFFFFFF`，
`LONG_MAX` = `0x7FFFFFFFFFFFFFFF`，`ULONG_MAX` = `0xFFFFFFFFFFFFFFFF`。
在 C 里更稳妥的写法是 `#include <limits.h>` 后用 `INT_MAX` / `INT_MIN` / `UINT_MAX`，或用 `stdint.h` 的 `INT32_MAX`、`INT64_MIN`、`UINT64_MAX`。

#### A.1.3 补码 ↔ 无符号：四个转换函数

| 函数 | 定义 | 公式 |
|---|---|---|
| $B2T_w$ | 位模式 → 补码值 | $B2T_w(X) = -x_{w-1}2^{w-1} + \\sum_{i=0}^{w-2} x_i 2^i$ |
| $B2U_w$ | 位模式 → 无符号值 | $B2U_w(X) = \\sum_{i=0}^{w-1} x_i 2^i$ |
| $T2U_w$ | 补码 → 无符号（**位不变**） | $T2U_w(x) = x + x_{w-1}\\cdot 2^w = \\begin{cases} x & x \\ge 0\\\\ x + 2^w & x < 0\\end{cases}$ |
| $U2T_w$ | 无符号 → 补码（**位不变**） | $U2T_w(u) = u - u_{w-1}\\cdot 2^w = \\begin{cases} u & u < 2^{w-1}\\\\ u - 2^w & u \\ge 2^{w-1}\\end{cases}$ |

- **核心恒等式**：$T2U_w(x) = x + 2^w$（当 $x<0$）；$U2T_w(u) = u - 2^w$（当 $u \\ge 2^{w-1}$）。二者都**只改解释、不改位**。
- **溢出环**：$T2U$ 把 $TMin \\to 2^{w-1}$、$-1 \\to UMax$；$U2T$ 把 $UMax \\to -1$、$2^{w-1} \\to TMin$。
- **C 语言中的触发点**：显式强制转换 `(unsigned)ty`、赋值 `ux = tx`、以及**函数调用传参**（原型为 `unsigned` 而实参为 `int`）都会做 $T2U$。
- **排序反转**：$x<0 \\Rightarrow T2U(x) > TMax$，所以 `-1 > 0U` 为真。混合有/无符号比较时，**有符号操作数被隐式转成无符号**。

#### A.1.4 溢出检测规则

| 运算 | 类型 | 溢出条件（数学判定） | 条件码判定 | 结果性质 |
|---|---|---|---|---|
| 加法 | 无符号 | `s = x + y`，若 $s < x$（或 $s < y$）则溢出 | `CF = 1`（进位出） | 真值 $= s + 2^w$ |
| 减法 | 无符号 | `s = x - y`，若 $x < y$ 则借位 | `CF = 1`（借位） | 真值 $= s + 2^w$ |
| 加法 | 补码 | $x,y$ 同号且 $s$ 异号 | `OF = 1` | `((x^s) & (y^s)) < 0` |
| 减法 | 补码 | $x,y$ 异号且 $s$ 与 $x$ 异号 | `OF = 1` | `((x^y) & (x^s)) < 0` |

- **正溢出**（positive overflow）：$x>0, y>0, s<0$，真值 $= x+y-2^w$（结果为 $TMin$ 附近）。
- **负溢出**（negative overflow）：$x<0, y<0, s \\ge 0$，真值 $= x+y+2^w$（结果为 $TMax$ 附近）。
- **补码构成阿贝尔群**：加法满足交换律、结合律，`0` 是单位元，每个数有加法逆元（$TMin$ 的逆元是它自己）；由此推出"平凡"恒等式 `-x == ~x + 1`、`x - y == x + (-y)` 恒成立。
- **乘法**：$w$ 位乘积只保留低 $w$ 位，$x \\cdot y$ 的无符号值与补码值**位模式相同**（乘法的低 $w$ 位不区分有/无符号）。溢出判定用 `imulq` 的 `OF`（判断 128 位乘积是否等于符号扩展后的低 64 位）。
- **取反**：`-x` 即 `~x + 1`；对 $TMin$ 取反得 $TMin$（溢出）。
- **C 语言注意**：**有符号溢出是未定义行为（UB）**，编译器可据此优化；无符号溢出是**有定义的模运算**。检测有符号溢出的可移植写法是先转成无符号算，再检查。

#### A.1.5 移位规则

| 表达式 | 类型 | 补位 | 数学等价 | 备注 |
|---|---|---|---|---|
| `x << k` | 任意 | 低位补 0 | $x \\cdot 2^k \\bmod 2^w$ | 移出的高位丢弃；有符号左移溢出在 C 中是 UB |
| `x >> k`（无符号） | `unsigned` | 高位补 0 | $\\lfloor u / 2^k \\rfloor$ | **逻辑右移（logical shift）** |
| `x >> k`（有符号，非负） | 补码 | 高位补符号位 = 0 | $\\lfloor x / 2^k \\rfloor$ | 与逻辑右移结果相同 |
| `x >> k`（有符号，负数） | 补码 | 高位补符号位 = 1 | $\\lfloor x / 2^k \\rfloor$（向 $-\\infty$ 取整） | **算术右移（arithmetic shift）** |

- **负数右移在 C 中是实现定义行为（implementation-defined）**：标准未规定补 0 还是补符号位。x86-64 的 `sar` 族补符号位，GCC/Clang 一贯如此，但**不可移植**。
- **移位量非法**：C 标准规定 $k < 0$ 或 $k \\ge w$ 是 **UB**（x86 硬件实际只取 $k \\bmod w$，例如 `1 << 32` 在硬件上等价于 `1 << 0`）。
- **右移 ≠ 除法**：负数算术右移向 $-\\infty$ 取整，`/` 向 0 取整。例：`-7 >> 1 == -4`，`-7 / 2 == -3`。
  向 0 取整的补偿写法：`(x < 0 ? x + (1<<k) - 1 : x) >> k`。
- **`>>` vs `>>>`**：**C 语言只有 `>>`**，没有 `>>>`；要逻辑右移就先转 `unsigned`（`(unsigned)x >> k`）。`>>>` 是 **Java** 的逻辑右移运算符（Java 的 `>>` 恒为算术右移），`>>>` 也存在于 JavaScript。阅读语言特性时勿混淆。
- 指令对应：`shl`/`sal` 等价（左移）、`shr` 逻辑右移、`sar` 算术右移；移位量放在 `%cl` 或立即数（`salq $2, %rax`）。

#### A.1.6 IEEE 754 位域与数值分类

| 格式 | 总位 | 符号 `s` | 阶码位 $k$ | 尾数位 $n$ | 偏置 $\\text{bias}=2^{k-1}-1$ | 十进制有效数字 |
|---|---|---|---|---|---|---|
| `float`（单精度） | 32 | 1（bit 31） | 8（bits 30–23） | 23（bits 22–0） | **127** | 约 **7** 位 |
| `double`（双精度） | 64 | 1（bit 63） | 11（bits 62–52） | 52（bits 51–0） | **1023** | 约 **15–16** 位 |

```
double —— 64 位                               float —— 32 位
 63  62        52 51                    0      31  30     23 22           0
+---+-----------+-----------------------+     +---+-------+--------------+
| s |    exp    |         frac          |     | s |  exp  |     frac     |
| 1 |    11     |          52           |     | 1 |   8   |      23      |
+---+-----------+-----------------------+     +---+-------+--------------+
```

| `exp` | `frac` | 类别 | 值 $V$ |
|---|---|---|---|
| $0$ | $0$ | $\\pm 0$ | $\\pm 0$（**有两个零**，`+0 == -0` 为真） |
| $0$ | $\\neq 0$ | **非规格化数**（denormalized） | $V = (-1)^s \\times 0.\\texttt{frac} \\times 2^{1-\\text{bias}}$ |
| $1 \\sim 2^{k}-2$ | 任意 | **规格化数**（normalized） | $V = (-1)^s \\times M \\times 2^{E}$，$M = 1.\\texttt{frac}$，$E = \\texttt{exp} - \\text{bias}$ |
| 全 1 | $0$ | $\\pm\\infty$ | $1.0/0.0 \\to +\\infty$，$-1.0/0.0 \\to -\\infty$ |
| 全 1 | $\\neq 0$ | **NaN** | `0.0/0.0`、$\\infty-\\infty$、$\\sqrt{-1}$；`NaN != NaN` 为真 |

- **规格化数隐含前导 1**（$M = 1.\\texttt{frac}$）：这一位不存储，白赚一个有效位，代价是 `0` 无法表示——于是 `exp=0` 专用于非规格化数。
- **非规格化数**：$M = 0.\\texttt{frac}$，阶码固定为 $1-\\text{bias}$（不是 $0-\\text{bias}$），使 $0$ 与最小规格化数之间的间距**均匀**，称为**渐进下溢**（gradual underflow）。
- **最大规格化数**：`float` 约 $3.4\\times10^{38}$（`exp=0xFE`，`frac` 全 1）；**最小正规格化数**：$2^{-126}\\approx1.18\\times10^{-38}$。
- **舍入**：默认**向偶数舍入**（round-to-nearest-even）；另可设向零、向下、向上。
- **浮点特殊事实**：
  - `float` 有 24 位有效位（隐含 1 + 23），$\\log_{10}2^{24}\\approx 7.22$，故约 **7 位**十进制有效数字；`double` 有 53 位，$\\log_{10}2^{53}\\approx 15.95$，故约 **15–16 位**。
  - **`0.1 + 0.2 != 0.3`**：0.1、0.2、0.3 都不能被二进制有限表示，相加后的舍入结果与 0.3 的最近可表示数不同（`0.1+0.2` 打印约 `0.30000000000000004441`）。**浮点比较必须用容差**：`fabs(a-b) < 1e-9 * fmax(1.0, fmax(fabs(a), fabs(b)))`。
  - 浮点**不满足结合律**：`(x+y)+z != x+(y+z)`（大数吞小数）；只有 `-ffast-math` 才允许编译器重排。
  - `float → int → float` 不是恒等（`(int)3.14f == 3`，精度与截断都丢）。
  - `int → float` 对 $>2^{24}$ 的整数会丢低位；`int → double` 对所有 32 位整数精确。

---

### A.2 x86-64 寄存器与调用约定速查

#### A.2.1 16 个通用寄存器：64/32/16/8 位名称对照

| 64 位 | 32 位 | 16 位 | 8 位 | 角色 / 约定 |
|---|---|---|---|---|
| `%rax` | `%eax` | `%ax` | `%al` | **返回值**；caller-saved |
| `%rbx` | `%ebx` | `%bx` | `%bl` | **callee-saved** 临时量 |
| `%rcx` | `%ecx` | `%cx` | `%cl` | **第 4 个参数**；移位量寄存器（`%cl`） |
| `%rdx` | `%edx` | `%dx` | `%dl` | **第 3 个参数**（旧称数据寄存器） |
| `%rsi` | `%esi` | `%si` | `%sil` | **第 2 个参数** |
| `%rdi` | `%edi` | `%di` | `%dil` | **第 1 个参数** |
| `%rbp` | `%ebp` | `%bp` | `%bpl` | **callee-saved**；可选帧指针（frame pointer） |
| `%rsp` | `%esp` | `%sp` | `%spl` | **栈指针**（stack pointer），特殊 callee-saved |
| `%r8` | `%r8d` | `%r8w` | `%r8b` | **第 5 个参数** |
| `%r9` | `%r9d` | `%r9w` | `%r9b` | **第 6 个参数** |
| `%r10` | `%r10d` | `%r10w` | `%r10b` | caller-saved 临时量 |
| `%r11` | `%r11d` | `%r11w` | `%r11b` | caller-saved 临时量 |
| `%r12` | `%r12d` | `%r12w` | `%r12b` | **callee-saved** 临时量 |
| `%r13` | `%r13d` | `%r13w` | `%r13b` | **callee-saved** 临时量 |
| `%r14` | `%r14d` | `%r14w` | `%r14b` | **callee-saved** 临时量 |
| `%r15` | `%r15d` | `%r15w` | `%r15b` | **callee-saved** 临时量 |

- `%r8`–`%r15` 的 8 位形式是 `%r8b`–`%r15b`（**不是** `%r8l`），且需要 REX 前缀编码。
- `%sil`、`%dil`、`%bpl`、`%spl` 也是 REX 时代新增的低字节名（历史上 `%esi`/`%edi`/`%ebp`/`%esp` 的低字节不可寻址）。
- **写 32 位寄存器会清零高 32 位**：`movl $1, %eax` 之后 `%rax == 1`；而 `movw`/`movb` 只改写低 16/8 位，高位保留。这是 x86-64 特有的优化点（`movl` 省一条 `movz`）。
- `%rip` 不是通用寄存器：**程序计数器**（instruction pointer），只能用于 RIP 相对寻址（如 `0x2fda(%rip)`）和 `leaq`，不能作为算术目标。

#### A.2.2 参数传递与返回值

| 位置 | 内容 |
|---|---|
| `%rdi` | 第 1 个整型/指针参数 |
| `%rsi` | 第 2 个整型/指针参数 |
| `%rdx` | 第 3 个整型/指针参数 |
| `%rcx` | 第 4 个整型/指针参数 |
| `%r8` | 第 5 个整型/指针参数 |
| `%r9` | 第 6 个整型/指针参数 |
| 栈 | **第 7 个及以后的参数**，从右往左压栈，调用者负责清理（`call` 后 `%rsp+8` 处即第 7 个参数） |
| `%xmm0`–`%xmm7` | 浮点/向量参数（**独立计数**，与整型参数各数各的） |
| `%rax` | 整型/指针返回值 |
| `%xmm0` | 浮点返回值 |
| `%rdx:%rax` | **128 位返回值**（高 64 位在 `%rdx`） |

- 整型与浮点参数**混排时各自计数**：`void f(int a, double b, int c, double d)` → `a`→`%rdi`，`b`→`%xmm0`，`c`→`%rsi`，`d`→`%xmm1`。
- 调用者不需要为栈传参预留"影子空间"——这与 Windows x64 的 32 字节 shadow space 不同。

#### A.2.3 caller-saved vs callee-saved

| 类别 | 别名 | 寄存器 | 责任方 |
|---|---|---|---|
| **caller-saved** | call-clobbered | `%rax`、`%rcx`、`%rdx`、`%rsi`、`%rdi`、`%r8`、`%r9`、`%r10`、`%r11` | **调用者**在 `call` 前把要保留的值存进自己的栈帧 |
| **callee-saved** | call-preserved | `%rbx`、`%rbp`、`%r12`、`%r13`、`%r14`、`%r15` | **被调用者**若使用，必须保存并在 `ret` 前恢复 |
| **特殊** | — | `%rsp` | 函数退出时必须恢复为进入时的值（`ret` 依赖它） |

- 规律：**所有参数寄存器 + 返回值寄存器 + `%r10/%r11` 都是 caller-saved**；`%rbx/%rbp/%r12–%r15` 是 callee-saved。
- 被调用者保存 callee-saved 寄存器的两种典型做法：`pushq %rbx … popq %rbx`，或先 `subq $N, %rsp` 再 `movq %rbx, k(%rsp)`。
- 递归能正常工作，正是因为"只在 callee-saved 寄存器与自己的栈帧里保存值"这条纪律。

#### A.2.4 栈帧布局与 16 字节对齐

```
        高地址
        +---------------------------+
        |   调用者的栈帧 (caller)     |
        +---------------------------+
        |  第 7+ 个参数 (argument build) |
        +---------------------------+
        |   返回地址 (return address)  |  <-- call 压入；%rsp 在 callee 入口指向它
        +---------------------------+
 %rbp ->|   保存的 %rbp (old %rbp)    |  <-- 可选：被调用者压入
        +---------------------------+
        |   被保存的 callee-saved 寄存器 |
        +---------------------------+
        |   局部变量 / 数组 / 临时量     |
        +---------------------------+  <-- %rsp（函数体内随 push/sub 移动）
        低地址（栈向低地址增长）
```

| 要点 | 说明 |
|---|---|
| `call` 的行为 | 把返回地址（下一条指令地址）**压栈**，再跳到目标；`%rsp -= 8` |
| `ret` 的行为 | 从栈顶**弹出返回地址**到 `%rip`；`%rsp += 8` |
| **16 字节对齐** | ABI 要求 `call` 执行**之前** `%rsp % 16 == 0`。因而 callee 入口处 `%rsp % 16 == 8`（返回地址占了 8 字节）。函数内常用 `subq $8, %rsp` 补齐，因为 `%xmm` 的 `movaps` 要求 16 字节对齐 |
| 局部变量位置 | 位于 `%rsp` **之上**（高地址方向）到 `%rbp` 之间；函数真正在用的是 `%rsp` 到 `%rbp` 这段区间 |
| 帧指针可选 | `-O1` 以上通常省略 `%rbp`（`-fomit-frame-pointer`），改用 `%rsp` 相对寻址；`-O0` 与 `-fno-omit-frame-pointer` 保留 `%rbp` 便于调试 |
| 红色区域 | 叶子函数可用 `%rsp` 以下的 128 字节（red zone）而不必先减 `%rsp`；信号处理会破坏它，故编译器只在叶子函数使用 |
| 对齐陷阱 | 手写汇编忘记 `subq $8` 补齐对齐，会在 `movaps` 上触发 `SIGSEGV`——这是 Attack Lab / Bomb Lab 的常见翻车点 |

#### A.2.5 浮点/向量寄存器 `%xmm0`–`%xmm15`

| 寄存器 | 作用 |
|---|---|
| `%xmm0`–`%xmm7` | **浮点参数寄存器**（第 1–8 个浮点参数） |
| `%xmm0` | **浮点返回值** |
| `%xmm8`–`%xmm15` | 临时量，**全部 caller-saved** |

- 每个 `%xmm` 是 **128 位**；标量 `float` 用低 32 位，`double` 用低 64 位，写入时**高位置零**。
- `%xmm` 寄存器**没有 callee-saved 的**：调用者若在 `%xmm` 里存了跨调用要用的值，必须自己溢出到栈上。
- `%ymm0`–`%ymm15`（AVX，256 位）与 `%zmm0`–`%zmm31`（AVX-512，512 位）复用同一编号的寄存器组，低 128 位即 `%xmm`。

---

### A.3 x86-64 指令速查（AT&T 语法）

> **AT&T 语法三条铁律**：① 操作数顺序是 `源, 目的`（`movq src, dst`）；② 寄存器加 `%`（`%rax`）；③ 立即数加 `$`（`$15213`）。
> 内存寻址写作 `D(Rb, Ri, S)` $= \\text{Mem}[R_b + R_i \\cdot S + D]$，$S \\in \\{1,2,4,8\\}$，`Ri` **不能是** `%rsp`。

#### A.3.1 数据传送

| 指令 | 语义 | 说明 |
|---|---|---|
| `mov  S, D` | `D = S` | `movb/movw/movl/movq` 按宽度选择 |
| `movz  S, D` | 零扩展传送 | `movzbw`、`movzbl`、`movzbq`、`movzwl`、`movzwq` |
| `movs  S, D` | 符号扩展传送 | `movsbl`、`movsbw`、`movswl`、`movswq`、`movslq` |
| `movabsq $imm64, %reg` | 装入 64 位立即数 | **唯一**能直接装 64 位立即数的指令；普通 `movq $imm, %reg` 只支持 32 位符号扩展立即数 |
| `leaq  D(Rb,Ri,S), D` | `D = 有效地址` | **不访问内存**，纯粹是地址计算器；编译器用它做 `x + k*y`（$k \\in \\{1,2,4,8\\}$）的快速算术 |
| `pushq S` | `%rsp -= 8; (%rsp) = S` | 压栈 |
| `popq  D` | `D = (%rsp); %rsp += 8` | 出栈 |

- **`movl` 的清零副作用**：`movl $0, %eax` 使整个 `%rax` 为 0；而 `movw`/`movb` 保留高位。
- **`leaq` 的经典用法**：`leaq (%rdi,%rdi,2), %rax` 得 $3x$；`leaq 4(%rdi,%rdx), %rcx` 得 $x + y + 4$；配合 `salq` 实现任意常数乘法（如 `x*12` = `leaq (%rdi,%rdi,2),%rax` + `salq $2,%rax`）。
- 传送指令**只有 `movabsq` 例外**：`movq $0x123456789, %rax` 汇编器会报错或截断，必须 `movabsq`。

#### A.3.2 算术与逻辑

| 指令 | 语义 | 备注 |
|---|---|---|
| `add  S, D` | `D += S` | 设置条件码 |
| `sub  S, D` | `D -= S` | 设置条件码 |
| `imul S, D` | `D *= S`（有符号） | 两操作数形式；也支持 `imulq $c, S, D` 三操作数 |
| `imulq S` | `%rdx:%rax = %rax * S` | 一操作数形式，产生 128 位乘积 |
| `mulq  S` | 同上，**无符号** | 与 `imulq` 单操作数形式配对 `divq` |
| `idivq S` / `divq S` | `%rax = 商, %rdx = 余` | 被除数在 `%rdx:%rax`；有符号除法前须 `cqto` |
| `shl  k, D` / `sal k, D` | `D <<= k` | **两者等价**（shift left / shift arithmetic left） |
| `shr  k, D` | `D >>= k`（**逻辑**，补 0） | 用于无符号 |
| `sar  k, D` | `D >>= k`（**算术**，补符号位） | 用于有符号 |
| `xor  S, D` | `D ^= S` | `xorq %rax,%rax` 是清零惯用法（比 `movq $0` 短） |
| `and  S, D` | `D &= S` | 常用于掩码 |
| `or   S, D` | `D \|= S` | — |
| `inc  D` | `D += 1` | **不设 `CF`**，只设 `OF/SF/ZF` |
| `dec  D` | `D -= 1` | 同上 |
| `neg  D` | `D = -D` | 即 `~D + 1`；置 `CF` |
| `not  D` | `D = ~D` | **不影响任何条件码** |

- 移位量：`k` 为立即数，或放在 **`%cl`** 中（`salq %cl, %rax`）。
- 乘法常被编译器强度削减为 `leaq` + `sal`（如 `x*48` → `leaq (%rsi,%rsi,2),%rdx` + `salq $4,%rdx`）——**看汇编时不要以为一定有 `imul`**。

#### A.3.3 比较与测试

| 指令 | 语义 | 说明 |
|---|---|---|
| `cmp  S1, S2` | 计算 `S2 - S1`，**只设条件码，不写回** | ATT 顺序是"减数在前"：`cmpq %rsi, %rdi` 算 `%rdi - %rsi` |
| `test S1, S2` | 计算 `S1 & S2`，**只设条件码，不写回** | `testq %rax, %rax` 用于判断 `%rax` 是否为 0/负 |

#### A.3.4 条件码表

| 标志 | 含义 | 由谁设置 |
|---|---|---|
| `CF` | **进位/借位**（Carry） | `add`/`sub` 的无符号溢出；移位指令（最后移出的位）；`neg`。**`inc`/`dec` 不设 `CF`** |
| `ZF` | **零标志**（Zero） | 结果为 0 时置 1；`add`/`sub`/`and`/`or`/`xor`/`cmp`/`test`/`inc`/`dec` 均设置 |
| `SF` | **符号标志**（Sign） | 结果最高位为 1（即"负"）时置 1 |
| `OF` | **溢出标志**（Overflow） | 补码加/减溢出；`imul` 溢出。**`inc`/`dec` 会设 `OF`** |

- **`leaq` 不设置任何条件码**——这是它在编译器中被当作"纯算术"使用的关键原因。
- 四种组合判定有符号比较：`cmp` 后看 `SF` 与 `OF` 是否相等（`SF == OF` 表示"确实不小于"）。

#### A.3.5 `set` 指令：按条件写单字节

| 指令 | 同义 | 条件码组合 | 语义 |
|---|---|---|---|
| `sete` | `setz` | `ZF` | 相等 / 为零 |
| `setne` | `setnz` | `~ZF` | 不等 / 非零 |
| `sets` | — | `SF` | 负数 |
| `setns` | — | `~SF` | 非负 |
| `setg` | `setnle` | `~ZF & (SF == OF)` | **有符号** $>$ |
| `setge` | `setnl` | `SF == OF` | 有符号 $\\ge$ |
| `setl` | `setnge` | `SF != OF` | 有符号 $<$ |
| `setle` | `setng` | `ZF \| (SF != OF)` | 有符号 $\\le$ |
| `seta` | `setnbe` | `~CF & ~ZF` | **无符号** $>$ |
| `setae` | `setnb` | `~CF` | 无符号 $\\ge$ |
| `setb` | `setnae` | `CF` | 无符号 $<$ |
| `setbe` | `setna` | `CF \| ZF` | 无符号 $\\le$ |

- `set` 类指令的**目的操作数必须是单字节位置**（`%al`、`%r10b` 等，或内存），写成 `setg %rax` 会报错。
- 常与 `movzbl` 配合得到 32 位 0/1 结果：`cmpq %rsi,%rdi; setg %al; movzbl %al,%eax`。

#### A.3.6 `j` 指令：按条件跳转

| 指令 | 同义 | 条件码组合 | 用途 |
|---|---|---|---|
| `jmp` | — | 无条件 | 直接/间接跳转（`jmp *%rax`、`jmp *(%rax)`） |
| `je` | `jz` | `ZF` | 相等 |
| `jne` | `jnz` | `~ZF` | 不等 |
| `js` | — | `SF` | 负数 |
| `jns` | — | `~SF` | 非负 |
| `jg` | `jnle` | `~ZF & (SF == OF)` | **有符号** $>$ |
| `jge` | `jnl` | `SF == OF` | 有符号 $\\ge$ |
| `jl` | `jnge` | `SF != OF` | 有符号 $<$ |
| `jle` | `jng` | `ZF \| (SF != OF)` | 有符号 $\\le$ |
| `ja` | `jnbe` | `~CF & ~ZF` | **无符号** $>$ |
| `jae` | `jnb` | `~CF` | 无符号 $\\ge$ |
| `jb` | `jnae` | `CF` | 无符号 $<$ |
| `jbe` | `jna` | `CF \| ZF` | 无符号 $\\le$ |

- **选择口诀**：**`g`/`l`（greater/less）用于有符号**；**`a`/`b`（above/below）用于无符号**。写错类别是汇编题最常见错误——`int` 用 `jg`，`unsigned` 用 `ja`。
- C 的 `<`、`>` 编译成 `cmp` + `jl`/`jg`；`unsigned` 的比较编译成 `cmp` + `jb`/`ja`。

#### A.3.7 `cmov` 指令：条件传送（无分支）

| 指令 | 同义 | 条件码组合 | 语义 |
|---|---|---|---|
| `cmove` | `cmovz` | `ZF` | 条件成立时 `dst = src` |
| `cmovne` | `cmovnz` | `~ZF` | — |
| `cmovs` | — | `SF` | — |
| `cmovns` | — | `~SF` | — |
| `cmovg` | `cmovnle` | `~ZF & (SF == OF)` | 有符号 $>$ |
| `cmovge` | `cmovnl` | `SF == OF` | 有符号 $\\ge$ |
| `cmovl` | `cmovnge` | `SF != OF` | 有符号 $<$ |
| `cmovle` | `cmovng` | `ZF \| (SF != OF)` | 有符号 $\\le$ |
| `cmova` | `cmovnbe` | `~CF & ~ZF` | 无符号 $>$ |
| `cmovae` | `cmovnb` | `~CF` | 无符号 $\\ge$ |
| `cmovb` | `cmovnae` | `CF` | 无符号 $<$ |
| `cmovbe` | `cmovna` | `CF \| ZF` | 无符号 $\\le$ |

- **源操作数不能是立即数**：必须先 `movq $c, %reg` 再 `cmovXX %reg, %dst`。
- **内存源无论如何都会被读取**：即使条件不成立，CPU 也会发起该访存（可能触发缺页/段错误）。因此编译器只在"两个分支都安全可算"时才用 `cmov`——这也是 CS:APP 讲的"分支不可预测时 `cmov` 更快，但语义不等价"的原因。
- 浮点比较用 `ucomisd`/`comisd`，它设置的是 **`ZF`/`PF`/`CF`**（不设 `SF`/`OF`），也无序（NaN）时置 `ZF=PF=CF=1`；因此浮点的条件传送/跳转要用 **无符号** 形式（`cmova`/`cmovae`/`cmovb`/`cmovbe`），不能用 `cmovg`/`cmovl`。

#### A.3.8 过程调用

| 指令 | 语义 | 备注 |
|---|---|---|
| `call Label` | 压返回地址（`%rsp -= 8`），跳转 | 直接调用 |
| `call *Operand` | 间接调用（函数指针、虚表、PLT） | `call *%rax`、`call *(%rdx)` |
| `ret` | 从栈顶弹出返回地址到 `%rip` | 与 `call` 配对 |
| `leave` | `movq %rbp,%rsp; popq %rbp` | 撤销帧指针的惯用两条合体 |

#### A.3.9 浮点/SSE 指令

| 类别 | 指令 | 语义 |
|---|---|---|
| 传送 | `movss` / `movsd` | 单/双精度**标量**传送（只动低 32/64 位） |
| 传送 | `movaps` / `movapd` | **对齐**（16 B）打包传送，未对齐会 `SIGSEGV` |
| 传送 | `movups` / `movupd` | **非对齐**打包传送，无对齐要求 |
| 算术 | `addss`/`addsd`、`subss`/`subsd`、`mulss`/`mulsd`、`divss`/`divsd` | 标量加/减/乘/除（`ss`=float，`sd`=double） |
| 算术 | `addps`/`addpd` 等 | **打包**（4×float 或 2×double）算术 |
| 比较 | `ucomiss`/`ucomisd` | 比较并设置 `ZF/PF/CF`（供 `seta`/`cmova` 用） |
| 转换 | `cvtss2sd` | `float → double` |
| 转换 | `cvtsd2ss` | `double → float` |
| 转换 | `cvtsi2sd` / `cvtsi2ss` | 整数 → `double` / `float` |
| 转换 | `cvttsd2si` / `cvttss2si` | `double`/`float` → 整数，**截断**（truncate，向 0） |
| 转换 | `cvtsd2si` | 按当前舍入模式转整数（默认向偶数） |

- 浮点参数 `%xmm0`–`%xmm7`，返回值 `%xmm0`；`%xmm` 全为 caller-saved。
- `%xmm` 与 `%rdi` 等整型寄存器的计数**互相独立**（见 A.2.2）。

#### A.3.10 易混淆指令对照表

| 指令 | 方向 | 语义 | 陷阱 |
|---|---|---|---|
| `movzbl` | byte → long | **零扩展**：`%dil` → `%edi`/`%rax` | 高位补 **0**，用于 `unsigned char` |
| `movsbl` | byte → long | **符号扩展**：`%dil` → `%edi`/`%rax` | 高位补 **符号位**，用于 `char`/`signed char` |
| `movslq` | long(32) → quad(64) | **符号扩展**：`movslq %eax, %rax` | 名字里的 `s` 是 sign，`l` 是**源**宽 4 字节，`q` 是**目的** 8 字节——**后缀先后是"源→目的"** |
| `movswl` | word → long | 符号扩展 2 → 4 字节 | 同上规则 |
| `movzwl` | word → long | 零扩展 2 → 4 字节 | `unsigned short` → `int` |
| `cltq` | `%eax` → `%rax` | 等价于 `movslq %eax,%rax` | **无操作数**；"convert long to quad" |
| `cqto` | `%rax` → `%rdx:%rax` | 把 `%rax` 的符号位扩展到 `%rdx`，为 `idivq` 准备 128 位被除数 | **无操作数**；无符号版是 `xorq %rdx,%rdx` |
| `shr` | 右移 | **逻辑**右移，高位补 0 | 用于 `unsigned` |
| `sar` | 右移 | **算术**右移，高位补符号位 | 用于有符号；负数除法场景 |
| `shl` vs `sal` | 左移 | **完全等价** | 没有"算术左移"与"逻辑左移"之分 |
| `movl` vs `movq` | 传送 | `movl` 写 32 位会**清零高 32 位**；`movq` 不会涉及该规则 | `movl $-1, %eax` 得 `%rax == 0x00000000FFFFFFFF` |
| `movq $imm` vs `movabsq` | 立即数 | `movq` 只支持 32 位符号扩展立即数；`movabsq` 支持 64 位 | 装 `0x123456789ABCDEF0` 必须用 `movabsq` |
| `leaq` vs `movq` | 地址 | `leaq 8(%rsp), %rax` 得到**地址**；`movq 8(%rsp), %rax` 得到**内容** | `leaq` **不访存**，可安全用于无效地址表达式 |
| `imul` vs `mul` | 乘法 | `imul` 有符号，`mul` 无符号 | 低 64 位结果**位模式相同**，只用 `imul` 也无妨 |
| `inc`/`dec` | 自增/减 | **不修改 `CF`** | 循环计数若依赖 `CF` 会出错；`add $1` 才改 `CF` |

---

### A.4 GDB 速查

> 启动前用 `gcc -g -O0 -Wall` 编译，调试信息才完整。`-Og` 兼顾优化与可调试性。

#### A.4.1 启动

| 命令 | 作用 |
|---|---|
| `gdb ./prog` | 加载可执行文件的符号 |
| `gdb --args ./prog a b` | 连命令行参数一起加载（等价于 `run a b`） |
| `gdb ./prog core` | 用 core dump 事后调试（先 `ulimit -c unlimited`） |
| `gdb -p PID` | 附加到正在运行的进程（attach） |
| `gdb -tui` / `gdb -tui ./prog` | 启动 TUI 分屏界面 |
| `gdb ./prog -ex 'b main' -ex run` | 非交互式批处理（脚本化） |

#### A.4.2 断点

| 命令 | 作用 |
|---|---|
| `break func` / `b func` | 在函数入口下断点 |
| `b file.c:42` / `b file:line` | 在指定源文件行下断点 |
| `b *0x4005d6` | 在**指令地址**下断点（无符号信息时的救命招） |
| `b *main+16` | 在函数相对偏移处下断点 |
| `tbreak ...` | **临时断点**，命中一次后自动删除 |
| `rbreak regex` | 对所有匹配正则的函数下断点 |
| `delete [N]` / `d` | 删除第 N 个（或全部）断点 |
| `disable N` / `enable N` | 禁用 / 启用断点（保留不删） |
| `info breakpoints` / `info b` | 列出所有断点及其命中次数 |
| `watch var` | 监视变量：**被写入**时停下（硬件/软件 watchpoint） |
| `rwatch var` / `awatch var` | 被读 / 被读或写时停下 |
| `condition N expr` | 给断点 N 加条件，如 `condition 1 i==5` |
| `ignore N 100` | 忽略前 100 次命中 |
| `catch syscall write` / `catch fork` | 捕获系统调用/信号等事件 |

#### A.4.3 执行

| 命令 | 缩写 | 作用 |
|---|---|---|
| `run [args]` | `r` | 从头运行程序 |
| `start` | — | 在 `main` 处停下（相当于 `b main; run`） |
| `step` | `s` | **步入**：进入被调函数（源码级） |
| `next` | `n` | **步过**：不进入被调函数（源码级） |
| `stepi [n]` | `si` | **单条指令**步入（汇编级） |
| `nexti [n]` | `ni` | 单条指令步过（汇编级） |
| `finish` | `fin` | 运行到当前函数返回，并打印返回值 |
| `continue` | `c` | 继续到下一个断点 |
| `until [line]` | `u` | 运行到当前行之后/指定行（跳出循环常用） |
| `advance func` | — | 运行到函数（不重复进入当前函数） |
| `kill` | `k` | 终止被调试程序 |
| `quit` | `q` | 退出 GDB |

- **卡在 `input` 里怎么办**：`finish` 可能因程序等待输入而挂住；先 `kill` 再重启。
- Bomb Lab 惯用组合：`b explode_bomb` → `run` → `bt` 找到调用点 → `frame N` → `info locals`。

#### A.4.4 查看数据

| 命令 | 作用 |
|---|---|
| `print expr` / `p expr` | 按自然类型打印表达式的值 |
| `p/x expr` | 以**十六进制**打印（也支持 `/d` 十进制、`/u` 无符号、`/t` 二进制、`/o` 八进制、`/c` 字符、`/f` 浮点） |
| `p/d expr` | 以**有符号十进制**打印 |
| `p/u expr` | 以**无符号十进制**打印 |
| `p/s ptr` | 以**字符串**打印（把 `char *` 按 C 字符串解释） |
| `p *ptr` | **解引用**，打印指针指向的对象 |
| `p **ptr` / `p ptr->field` | 多级解引用 / 打印结构体成员 |
| `p arr[3]@5` | 从 `arr[3]` 开始打印 **5 个**元素（`@` 是"连续 n 个"运算符） |
| `p (char*)ptr` / `p *((int*)p+2)` | 强制类型转换后再打印 |
| `p $rax`、`p $rip`、`p $eflags` | 打印**寄存器**（寄存器名前加 `$`） |
| `p/x $rax - $rbx` | 寄存器参与算术 |
| `display expr` | 每次停下**自动打印**该表达式（`undisplay N` 取消，`info display` 查看） |
| `set var x = 1` | 修改变量值（如 `set var i=0` 跳过循环） |
| `set $rax = 0` | 修改寄存器值（Bomb/Attack Lab 常用） |
| `set {int}0x7fffffffe000 = 5` | 直接改内存 |
| `whatis expr` / `ptype type` | 查看表达式的类型 / 类型定义 |

#### A.4.5 `x/nfu addr` 格式详解

语法 `x/[n][f][u] addr`：`n` 重复次数（十进制，默认 1）、`f` 显示格式、`u` 单元大小。

| 字段 | 取值 | 含义 |
|---|---|---|
| `n` | 十进制数字，如 `8`、`16` | 打印多少个单元（默认 1） |
| `f` | `x` | 十六进制 |
| `f` | `d` | 有符号十进制 |
| `f` | `u` | 无符号十进制 |
| `f` | `o` | 八进制 |
| `f` | `t` | 二进制（two's complement 显示） |
| `f` | `a` | 地址（同时给出最近的符号名偏移） |
| `f` | `c` | 字符 |
| `f` | `s` | **以 0 结尾的字符串** |
| `f` | `i` | **反汇编指令** |
| `f` | `f` | 浮点数 |
| `u` | `b` | **byte**，1 字节 |
| `u` | `h` | **halfword**，2 字节 |
| `u` | `w` | **word**，4 字节 |
| `u` | `g` | **giant**，8 字节 |

| 示例 | 含义 |
|---|---|
| `x/8xb &arr` | 从 `arr` 起打印 **8 个字节**的十六进制 |
| `x/8dw arr` | 打印 8 个 4 字节有符号整数 |
| `x/s 0x402008` | 把该地址当作 C 字符串打印（Bomb Lab 必备） |
| `x/4i $rip` | 反汇编从当前指令开始的 4 条指令 |
| `x/16gx $rsp` | 以 8 字节为单位打印栈顶 16 个单元（即 128 字节的栈） |
| `x/2gx $rdx` | 打印 `%rdx` 指向的 2 个 8 字节（如 128 位返回值） |
| `x/20wx $rsp-40` | 查看当前栈帧上下各 20 个 4 字节字 |

- 省略 `f`/`u` 时沿用**上一次**的设置（所以 `x/8xb` 之后再 `x/4` 仍是字节）；默认单元是 `w`、默认格式是 `x`。
- 常用组合速记：**`nfu` = "多少 / 怎么显示 / 多大一块"**。

#### A.4.6 栈与帧

| 命令 | 作用 |
|---|---|
| `backtrace` / `bt` | 打印调用栈：当前帧 + 各层调用者 |
| `bt full` | 附带每帧的局部变量 |
| `bt 5` | 只打印最内 5 层 |
| `frame N` / `f N` | 切换到第 N 号帧（0 = 最内层） |
| `up [N]` / `down [N]` | 上移/下移帧（沿调用链往外/往内） |
| `info frame` | 当前帧的详细信息（帧地址、返回地址、保存的寄存器位置） |
| `info registers` / `info reg` | 打印所有通用寄存器的值 |
| `info registers rip` | 只打印 `%rip` |
| `info all-registers` | 含 `%xmm`、`%st` 等全部寄存器 |
| `info locals` | 当前函数的**局部变量** |
| `info args` | 当前函数的**参数** |
| `info variables` / `info functions` | 全局变量 / 全部函数（支持正则过滤） |
| `info symbol 0x4005d6` | 某个地址落在哪个符号的哪个偏移 |
| `x/8gx $rbp` | 直接按内存查看栈帧内容 |

- **读栈帧的经典套路**：`bt` 找到帧号 → `f N` → `info frame` 看到 saved rip 的位置 → `x/8gx $rbp` 核对局部变量与返回地址。
- 编译加 `-fno-omit-frame-pointer`（或 `-O0`）时 `%rbp` 链才完整，`bt` 才可靠。

#### A.4.7 反汇编与源码

| 命令 | 作用 |
|---|---|
| `disassemble` / `disas` | 反汇编**当前函数** |
| `disas func` | 反汇编指定函数 |
| `disas 0x400540, 0x400560` | 反汇编地址区间 |
| `disas /r` | 附上**机器码字节**（raw bytes） |
| `disas /m` | 与源码行交错显示 |
| `set disassembly-flavor intel` | 切换到 **Intel 语法**（无 `%`、操作数顺序相反） |
| `set disassembly-flavor att` | 切回 AT&T 语法（默认） |
| `list` / `l` | 显示源码（`l func`、`l 42`、`l file.c:42`） |
| `info line *0x4005d6` | 该地址对应的源码行 |
| `directory DIR` | 添加源码搜索路径 |

#### A.4.8 TUI 与布局

| 命令 | 作用 |
|---|---|
| `layout regs` | 分屏：源码/汇编 + **寄存器**窗口（寄存器的改动会高亮） |
| `layout asm` | 只显示汇编窗口 |
| `layout src` | 只显示源码窗口 |
| `layout split` | 源码 + 汇编同时显示 |
| `tui reg general` / `tui reg float` / `tui reg all` | 切换寄存器窗口显示哪一组 |
| `focus cmd` / `focus src` / `focus asm` | 切换键盘焦点到某个窗口（上下箭头滚动） |
| `Ctrl-x a` / `Ctrl-x 2` | 进入/退出 TUI / 切换布局 |
| `Ctrl-l` | 刷新屏幕（TUI 花屏时） |
| `update` | 刷新 TUI 窗口 |

- **Bomb Lab 推荐姿势**：`gdb -tui ./bomb` → `layout regs` → `b explode_bomb` / 单步 `si`，一边看 `%rax`、`%rdi` 一边看反汇编。
- TUI 花屏或按键失灵时退出重进通常最快；`Ctrl-x a` 可临时回到普通模式。

#### A.4.9 信号与线程

| 命令 | 作用 |
|---|---|
| `handle SIGINT stop print` | 收到 `SIGINT` 时停下来并打印（默认 `stop print`） |
| `handle SIGINT nostop noprint pass` | 让 GDB **不拦** `SIGINT`，直接传给程序 |
| `handle SIGSEGV stop print pass` | 段错误时停下（默认行为），便于定位越界 |
| `handle SIGALRM nostop noprint pass` | 常见于带定时器的程序 |
| `info signals` | 列出所有信号的当前处理策略 |
| `info threads` | 列出所有线程及其当前帧（`*` 表示当前线程） |
| `thread N` | 切换到第 N 号线程 |
| `thread apply all bt` | 对**所有线程**打印调用栈（排查死锁/挂起第一招） |
| `thread apply all bt full` | 加上局部变量 |
| `set scheduler-locking on` | 单步时锁定调度，只让当前线程跑（调试并发时很有用） |
| `set follow-fork-mode child` | `fork` 后跟踪子进程 |

- 多线程下断点会在所有线程命中；配合 `scheduler-locking` 可让调试结果可复现。
- 死锁排查顺序：`thread apply all bt` → 找到卡在 `pthread_mutex_lock`/`sem_wait`/`read` 的线程 → `thread N` + `bt full` 看谁持锁。

---

### A.5 二进制工具速查

#### A.5.1 `gcc` 常用开关

| 开关 | 作用 |
|---|---|
| `-O0` | 不优化（默认）。**Bomb/Attack Lab 调试首选**，指令与源码一一对应 |
| `-O1` | 基础优化，兼顾可调试（`-Og` 是专为调试设计的优化级别） |
| `-O2` | 标准优化（多数发行版的默认发布级别） |
| `-O3` | 激进优化：更狠的内联、向量化、循环变换，代码体积与调试难度上升 |
| `-g` | 生成**调试信息**（DWARF），GDB 才能按源码行/变量名工作；`-g3` 额外含宏信息 |
| `-Wall` | 打开常用警告 |
| `-Wextra` | 打开更多警告（含未使用参数、符号比较等） |
| `-Werror` | 把警告当错误（CI 常用） |
| `-std=c11` / `-std=c17` / `-std=gnu11` | 选择 C 标准（`gnu*` 允许 GNU 扩展） |
| `-S` | 只到**汇编**（产出 `.s`，不汇编不链接） |
| `-c` | 只到**目标文件**（产出 `.o`） |
| `-E` | 只做**预处理**（把 `#include`/`#define` 展开） |
| `-o FILE` | 指定输出文件名 |
| `-lNAME` | 链接库，如 `-lm`（数学库）、`-lpthread`；**顺序重要**，库要放在引用它的 `.o` 之后 |
| `-L DIR` | 增加库搜索路径 |
| `-I DIR` | 增加头文件搜索路径 |
| `-shared` | 生成共享库（`.so`） |
| `-fPIC` | 生成**位置无关代码**（共享库必须用） |
| `-fno-pic` / `-no-pie` | 生成非 PIC、非 PIE 的可执行文件（Attack Lab 需要固定地址时常用） |
| `-static` | 静态链接，把库代码全部塞进可执行文件（体积大、无 `.so` 依赖） |
| `-pthread` | 启用 POSIX 线程：定义宏并链接线程库 |
| `-mavx2` / `-mavx512f` / `-msse4.2` | 允许生成对应 ISA 的指令（在无该指令的机器上运行会 `SIGILL`） |
| `-fsanitize=address,undefined` | 插入 ASan/UBSan 运行时检查（见 A.6） |
| `-fno-omit-frame-pointer` | 强制保留 `%rbp` 帧指针，`bt`/`perf` 更可靠 |
| `-save-temps` | 保留 `.i`/`.s`/`.o` 等中间产物，便于逐阶段检查 |
| `-v` | 打印实际调用的编译/汇编/链接命令与搜索路径（查"为什么找不到头文件"神器） |
| `-D NAME[=VALUE]` | 命令行定义宏，等价于源码里 `#define NAME VALUE` |
| `-M` / `-MM` | 输出依赖关系（给 Makefile 用） |
| `-Wl,OPT` | 把 `OPT` 透传给链接器，如 `-Wl,-z,now`（立即绑定）、`-Wl,-rpath,/path` |

- 典型组合：调试 `gcc -g -O0 -Wall -Wextra -std=c11 prog.c -o prog`；优化对比 `gcc -O0/-O1/-O2/-O3 -S`。
- `-O2` 及以上会让 `%rbp` 消失、变量被提升进寄存器、循环被重排——**看汇编时先确认编译级别**。

#### A.5.2 `objdump`：反汇编目标文件

| 命令 | 作用 |
|---|---|
| `objdump -d prog` | 反汇编所有含代码的节（AT&T 语法，默认） |
| `objdump -d -M intel prog` | 反汇编并输出 **Intel 语法**（无 `%`、操作数顺序反转） |
| `objdump -t prog.o` | 打印**符号表**（含节、地址、大小、local/global） |
| `objdump -s prog` | 打印各节的**完整十六进制内容** |
| `objdump -r prog.o` | 打印**重定位表**（链接前才能看到未解析的引用） |
| `objdump -j .text -d prog` | 只反汇编指定节 |
| `objdump -d --start-address=0x400540 --stop-address=0x400560 prog` | 反汇编地址区间 |
| `objdump -h prog` | 节头摘要（大小、VMA、文件偏移） |
| `objdump -x prog` | 打印全部头信息（节 + 符号 + 动态段 + 重定位） |
| `objdump -R prog` | 打印动态重定位表（对应 GOT 槽） |
| `objdump -S prog` | 反汇编与源码交错（需 `-g`） |

- **`-d` 反汇编 `.text` 中所有函数，但依赖节头里的符号信息**；被剥离符号的二进制要靠地址区间或 `nm`。
- `-M intel` 输出的地址格式与 GDB `set disassembly-flavor intel` 一致，两者可交叉核对。
- 链接**前**的 `.o` 里，未解析引用的目标字段是 0，必须看 `-r` 的重定位条目才能知道它要填什么。

#### A.5.3 `readelf`：ELF 结构

| 命令 | 作用 |
|---|---|
| `readelf -a prog` | 打印**全部** ELF 信息（头 + 节 + 段 + 符号 + 重定位 + 动态段），最全但冗余 |
| `readelf -h prog` | **ELF 头**：类型（REL/EXEC/DYN）、入口点、机器、节头表位置 |
| `readelf -S prog` | **节头表**（section headers）：`.text`/`.data`/`.bss`/`.symtab`/`.rela.text` 等的大小与地址 |
| `readelf -l prog` | **段头表**（program headers / segments）：`LOAD` 段的权限（R E）、对齐、`INTERP`、`GNU_STACK` |
| `readelf -s prog` | **符号表**（`.symtab` 与 `.dynsym`）：名字、值、大小、绑定（LOCAL/GLOBAL/WEAK）、类型（FUNC/OBJECT） |
| `readelf -r prog` | **重定位表**：偏移、类型（`R_X86_64_PC32` 等）、符号 |
| `readelf -d prog` | **动态段**：`NEEDED`（依赖的 `.so`）、`RPATH`、`INIT`/`FINI` |
| `readelf -x .rodata prog` | 十六进制 dump 指定节（`-x` / `--hex-dump`） |
| `readelf -p .comment prog` | 以字符串形式打印节的 ASCII 内容 |
| `readelf --dyn-syms prog` | 只打印动态符号表 |

- 记忆钩子：**小写 `-h/-S/-l/-s/-r/-d` 分别对应 header / Sections / segments(program headers) / symbols / relocations / dynamic**。
- 对比：`readelf` 面向 ELF 结构，`objdump` 面向机器码与反汇编；两者都要会。

#### A.5.4 其它二进制工具

| 工具 | 命令示例 | 作用 |
|---|---|---|
| `nm` | `nm prog` | 列出符号及地址（`T` 代码段、`D` 已初始化数据、`B` 未初始化数据 `.bss`、`U` **未定义/外部引用**、`w` 弱符号、小写=local） |
| `nm` | `nm -a prog` | 显示**全部**符号（含调试/局部符号） |
| `nm` | `nm -u prog` | 只列**未定义**符号（看程序依赖哪些外部函数，如 `printf`、`malloc`） |
| `nm` | `nm --defined-only prog` | 只列本文件定义的符号 |
| `size` | `size prog` | 打印 `text`/`data`/`bss` 三节字节数（`dec` 为总和） |
| `strings` | `strings -a prog \| head` | 抽取可打印字符串（找格式串、错误信息、隐藏提示；`-n 4` 改最短长度） |
| `ldd` | `ldd ./prog` | 列出**运行时**依赖的共享库及其解析到的路径（"not found" 即缺失） |
| `ar` | `ar rs lib.a a.o b.o` | 创建/更新**静态库**归档（`r` 插入、`s` 写索引）；`ar t lib.a` 列出成员 |
| `ld` | `ld -o prog a.o b.o -lc` | 直接调用链接器（`gcc` 会代你调用它） |
| `ldconfig` | `sudo ldconfig -p \| grep libm` | 重建/查询共享库缓存 `/etc/ld.so.cache` |
| `file` | `file prog prog.o lib.a` | 识别文件类型（`ELF 64-bit LSB executable, x86-64, dynamically linked, ...`） |
| `xxd` | `xxd prog \| head` | 十六进制 + ASCII 双栏 dump（`xxd -l 64 -s 0x1000 prog` 限定长度与偏移） |
| `hexdump` | `hexdump -C prog` | 十六进制 dump（`-C` 为规范格式，`-n 32` 限定字节数） |
| `od` | `od -A x -t x1z prog` | 八进制/十六进制 dump（`-t x1` 一字节十六进制、`-t d4` 四字节十进制、`-A x` 偏移用十六进制显示） |
| `strip` | `strip prog` | 删除符号表（体积变小，但反汇编更难读） |
| `gdb` | `gdb -batch -ex 'disas main' ./prog` | 批处理式反汇编（无需交互） |

#### A.5.5 链接关键机制（一句话版）

| 机制 | 一句话说明 |
|---|---|
| **符号解析**（symbol resolution） | 把每个**引用**关联到符号表里的一个**定义**；找不到定义 → `undefined reference`，定义重名（两个强符号）→ `multiple definition` |
| **重定位**（relocation） | 合并各 `.o` 的节、确定运行时地址后，用重定位条目**改写指令/数据中的地址字段** |
| `R_X86_64_PC32` | **PC 相对**寻址：填入"符号地址 − 重定位处地址"的 32 位差值，用于 `call`/`jmp`/`lea` 到全局变量；x86-64 默认偏移量要减 4（故反汇编常见 `sum-0x4`） |
| `R_X86_64_32` | **绝对寻址**：把符号的 32 位绝对地址写进字段（受限于低 2 GB 地址空间） |
| `R_X86_64_PLT32` | 调用外部函数时的 PC 相对条目，经 PLT 中转（PIC 世界里的 `call`） |
| `R_X86_64_JUMP_SLOT` | GOT 中的**延迟绑定槽**：加载后由动态链接器填入真正的函数地址 |
| `R_X86_64_GLOB_DAT` | GOT 中全局数据的槽 |
| **强符号 vs 弱符号** | 函数与已初始化的全局变量是**强符号**；未初始化全局变量、`extern` 声明、被 `__attribute__((weak))` 修饰的是**弱符号** |
| 强弱符号三规则 | ① 不允许两个强符号重名；② 强 + 多个弱 → 选强；③ 多个弱 → 任选其一（**危险**，是链接类 bug 的温床） |
| **静态库** | `.a` 是 `.o` 的归档；链接器只把**被引用到**的成员拉进可执行文件，因此**顺序敏感**（引用者在前，被引用者在后） |
| **共享库** | `.so` 在加载/运行时映射；多进程共享同一份物理代码页，靠 VM 实现 |
| **PIC / GOT** | 位置无关代码不写死绝对地址：访问全局变量经 **GOT**（Global Offset Table）间接寻址，访问外部函数经 **PLT** |
| **PLT 与延迟绑定** | 首次调用 `printf`：PLT 表项 → GOT 表项（初始指向 PLT 表项内部的下一条指令）→ push 重定位索引 → `PLT[0]` → `_dl_runtime_resolve` 解析真实地址并**回填 GOT**；此后每次调用只多一次间接跳转 |
| **立即绑定** | `-Wl,-z,now` 或环境变量 `LD_BIND_NOW=1` 让加载时就解析全部符号（更安全，GNU_RELRO 可把 GOT 设为只读） |
| `LD_PRELOAD` | 指定**优先加载**的共享库，其符号覆盖后续库的同名符号——用于打桩拦截 `malloc`/`open` 等 |
| `LD_LIBRARY_PATH` | 运行时**优先**搜索的共享库目录列表（先于系统默认路径） |
| `LD_DEBUG=libs,bindings` | 打印动态链接器的搜索与绑定过程（排查"加载了哪个 `.so`"） |

---

### A.6 调试与动态分析工具速查

#### A.6.1 `valgrind` 工具族

| 工具 | 命令 | 检测什么 |
|---|---|---|
| **memcheck**（默认） | `valgrind ./prog` | 非法读写、未初始化值使用、内存泄漏、重复 `free`、`free` 非堆指针、越界 |
| **callgrind** | `valgrind --tool=callgrind ./prog` | 函数级**调用次数与指令计数**（配合 `callgrind_annotate` / KCachegrind 看热点） |
| **cachegrind** | `valgrind --tool=cachegrind ./prog` | 模拟 L1/L2 缓存的命中与未命中、分支预测失败 |
| **helgrind** | `valgrind --tool=helgrind ./prog` | **线程错误**：数据竞争、锁顺序不一致、锁使用错误 |
| **drd** | `valgrind --tool=drd ./prog` | 数据竞争检测的另一实现（侧重 happens-before 分析与竞争报告） |
| **massif** | `valgrind --tool=massif ./prog` | **堆内存使用随时间的变化**（峰值为谁分配），`ms_print` 可视化 |
| **memcheck 变体** | `--tool=exp-sgcheck` | 栈/全局数组越界（已废弃，现代用 ASan） |

**memcheck 关键选项**

| 选项 | 作用 |
|---|---|
| `--leak-check=full` | 完整泄漏报告：给出每个泄漏块的**分配调用栈** |
| `--show-leak-kinds=all` | 显示全部泄漏类别（默认只报 definite/possible） |
| `--track-origins=yes` | 追踪**未初始化值**的来源（更慢，但能直接指出"哪个变量没初始化"） |
| `--error-exitcode=1` | 发现错误时以退出码 1 结束——**CI 里必加** |
| `-s` / `--stack-traces=yes` | 打印调用栈 |
| `--num-callers=30` | 调用栈深度（默认 12，太浅看不到根因） |
| `--track-fds=yes` | 报告**未关闭的文件描述符** |
| `--vgdb=yes` | 允许用 GDB 通过 vgdb 控制 valgrind |
| `--log-file=vg.log` | 输出写入文件（结果太长时） |
| `--gen-suppressions=all` | 为每个错误生成抑制条目（`--suppressions=file` 复用） |

**memcheck 报告分类速查**

| 报告 | 含义 | 严重度 |
|---|---|---|
| `Invalid read of size N` / `Invalid write of size N` | 读写**未分配/已释放/越界**的地址 | 高，必查 |
| `Use of uninitialised value` | 使用了未初始化内存（分支/输出/系统调用都算） | 高 |
| `Conditional jump or move depends on uninitialised value(s)` | 用未初始化值做了分支判断 | 高，常是漏初始化 |
| `definitely lost` | **确定泄漏**：没有指针指向该块，无法释放 | 高 |
| `indirectly lost` | 泄漏块**内部的指针**所指的块也泄漏（根因是外层块丢失） | 中（修好外层即可） |
| `possibly lost` | 仍有指针指向块**内部**（如指向结构体中间），可能泄漏 | 中，需人工判断 |
| `still reachable` | 程序结束时**仍有指针指向**、未释放但可达 | 低（常见于未 `free` 的全局缓存/库内部结构） |
| `suppressed` | 被抑制规则屏蔽的错误 | 忽略 |
| `ERROR SUMMARY: N errors from M contexts` | 汇总：N 个错误、M 个不同调用点 | — |
| `HEAP SUMMARY: in use at exit` | 退出时仍占用的字节数与块数 | — |

- **Valgrind 不是万能的**：它跟踪的是真实执行的路径；`-O3` 下越界可能被优化掉从而"查不出错"——**务必用 `-O0 -g` 编译后再跑 valgrind**。
- Valgrind 通过动态二进制插桩运行，程序会**慢 10–50 倍**，不适合作性能测量（用 `perf`）。

#### A.6.2 Sanitizer 编译开关

| 开关 | 检测内容 | 典型用途 |
|---|---|---|
| `-fsanitize=address`（ASan） | **越界/释放后使用（UAF）/重复释放/栈溢出/全局越界**，含分配与释放调用栈 | 替代 valgrind，快得多（约 2× 减速） |
| `-fsanitize=undefined`（UBSan） | **未定义行为**：有符号溢出、移位越界、空指针解引用、对齐错误、越界数组索引 | 抓"看起来能跑其实是 UB"的代码 |
| `-fsanitize=thread`（TSan） | **数据竞争**（与 ASan **互斥**，不能同时开） | SFS / Proxy Lab 的并发 bug |
| `-fsanitize=leak`（LSan） | **内存泄漏**（只查泄漏，代价低） | 大程序只关心泄漏时 |
| `-fsanitize=memory`（MSan） | **未初始化读**（需所有代码含库都用 MSan 编译） | 与 valgrind memcheck 类似 |
| `-fsanitize=address,undefined` | 组合使用（最常用） | 日常开发 |
| `-fsanitize-recover=all` | UBSan 报错后**继续**执行（默认对部分检查是继续） | 一次跑完收集全部问题 |
| `-fno-sanitize-recover=all` | 任何 sanitizer 报错立即终止 | CI 严格模式 |
| `-fsanitize=undefined -fno-sanitize=alignment` | 关闭对齐检查 | 处理故意非对齐访问的代码 |
| `-g -O1` | **Sanitizer 推荐编译级别**（`-O1` 保证行号与内联都可用） | — |

| 环境变量 | 作用 |
|---|---|
| `ASAN_OPTIONS=detect_leaks=1` | 打开/关闭泄漏检测（LSan） |
| `ASAN_OPTIONS=halt_on_error=0` | 出错后继续运行 |
| `ASAN_OPTIONS=abort_on_error=1` | 出错时 `abort()`（便于拿 core） |
| `ASAN_OPTIONS=symbolize=1` | 符号化栈帧（需要 `llvm-symbolizer` 或 `addr2line`） |
| `ASAN_OPTIONS=log_path=asan.log` | 报告写入文件 |
| `ASAN_OPTIONS=detect_stack_use_after_return=1` | 额外检测**返回后使用栈**（更慢） |
| `UBSAN_OPTIONS=print_stacktrace=1` | UBSan 报告附调用栈 |
| `UBSAN_OPTIONS=halt_on_error=1` | UBSan 首个错误即停 |
| `TSAN_OPTIONS=second_deadlock_stack=1` | TSan 死锁报告附第二个栈 |
| `LSAN_OPTIONS=suppressions=lsan.supp` | 泄漏抑制文件 |

#### A.6.3 `strace` 与 `ltrace`

| 命令 | 作用 |
|---|---|
| `strace ./prog` | 打印程序的所有**系统调用**及返回值 |
| `strace -f ./prog` | **跟随子进程/线程**（`fork`、`clone` 产生的新进程也跟） |
| `strace -e trace=process ./prog` | 只跟踪进程类调用（`fork`/`execve`/`wait4`/`exit_group`） |
| `strace -e trace=file ./prog` | 只跟踪文件类调用（`openat`/`stat`/`read`/`write`/`close`） |
| `strace -e trace=network ./prog` | 只跟踪网络类调用（`socket`/`connect`/`bind`/`sendto`/`recvfrom`） |
| `strace -e trace=openat,read,write ./prog` | 精确指定调用名（可逗号分隔） |
| `strace -c ./prog` | **统计汇总**：每个系统调用的次数、耗时、错误数 |
| `strace -T ./prog` | 每条调用后显示**耗时**（秒） |
| `strace -t` / `-tt` / `-r` | 显示时间戳 / 微秒时间戳 / 相对时间 |
| `strace -o out.txt ./prog` | 输出重定向到文件 |
| `strace -p PID` | 附加到运行中的进程 |
| `strace -s 128 ./prog` | 字符串最大打印长度（默认 32，看路径常需加大） |
| `strace -k ./prog` | 附上用户态调用栈（需内核支持） |
| `strace -y ./prog` | 把 fd 打印成对应路径（"这个 3 到底是哪个文件"） |

| 命令 | 作用 |
|---|---|
| `ltrace ./prog` | 跟踪**库函数调用**（`malloc`、`printf`、`strcmp`…），看参数与返回值 |
| `ltrace -e malloc+free ./prog` | 只看指定库函数 |
| `ltrace -c ./prog` | 库函数调用统计 |

- **典型用法**：Shell Lab 调试 `fork`/`execve`/`waitpid` 用 `strace -f -e trace=process ./tsh`；Proxy Lab 看 socket 用 `strace -e trace=network`；"文件打不开"用 `strace -e trace=file -y`。

#### A.6.4 `perf`：硬件性能计数器

| 命令 | 作用 |
|---|---|
| `perf stat ./prog` | 汇总统计：`cycles`、`instructions`、`IPC`、缓存未命中、分支预测失败 |
| `perf stat -e cache-misses,LLC-load-misses ./prog` | 指定事件计数 |
| `perf stat -r 5 ./prog` | 重复 5 次并给方差 |
| `perf record -g ./prog` | **采样**并记录，`-g` 采集调用图（`perf.data`） |
| `perf record -e LLC-load-misses -c 1000 ./prog` | 指定事件与采样周期 |
| `perf report` | 交互式查看 `perf.data`（按热点排序、展开调用图） |
| `perf report --stdio` | 文本方式输出报告 |
| `perf top` | 实时查看**系统范围**热点函数 |
| `perf annotate` | 把热点精确到**指令级**（哪个汇编指令最费时） |
| `perf list` | 列出本机支持的全部事件名 |
| `perf stat -a -I 1000` | 系统级每秒采样（`-a` 全体 CPU） |

| 常用事件 | 含义 |
|---|---|
| `cycles` | CPU 周期数（与频率结合可算时间） |
| `instructions` | 退休指令数；`instructions/cycles` 即 **IPC** |
| `cache-references` | 缓存访问次数 |
| `cache-misses` | 缓存未命中次数 |
| `L1-dcache-loads` / `L1-dcache-load-misses` | L1 数据缓存读次数/未命中 |
| `LLC-loads` / `LLC-load-misses` | **最后一级缓存**（L3）读次数/未命中——访存瓶颈的核心指标 |
| `branch-instructions` / `branch-misses` | 分支数/预测失败数（>5% 通常意味着分支代价高） |
| `dTLB-loads` / `dTLB-load-misses` | 数据 TLB 访问/未命中 |
| `page-faults` / `minor-faults` / `major-faults` | 缺页总数 / 次缺页 / 主缺页（含磁盘 I/O） |
| `context-switches` / `cpu-migrations` | 上下文切换 / CPU 迁移次数 |
| `task-clock` | 占用的 CPU 时间（毫秒） |

- `perf` 需要内核权限：`sudo perf ...` 或把 `perf_event_paranoid` 调低。
- 性能优化流程：`perf stat` 定位瓶颈类别（IPC 低？缓存未命中高？）→ `perf record -g` 找热点函数 → `perf annotate` 找热点指令。

#### A.6.5 `gprof`、`time`、`ulimit`

| 工具 | 命令 | 作用 |
|---|---|---|
| `gprof` | `gcc -pg -g prog.c -o prog` 然后 `./prog` | 运行后在 `gmon.out` 记录采样，`gprof ./prog gmon.out` 给出**平坦剖面**（每个函数自身耗时）与**调用图**（谁调谁、调用次数） |
| `time` | `time ./prog` | 打印 `real`（墙钟）/`user`（用户态 CPU）/`sys`（内核态 CPU） |
| `time` | `/usr/bin/time -v ./prog` | 详细版：峰值内存（Maximum resident set size）、缺页、上下文切换 |
| `ulimit` | `ulimit -c unlimited` | 允许生成 **core dump**（配合 `gdb ./prog core` 事后调试） |
| `ulimit` | `ulimit -v 1000000` | 限制**虚拟内存**（KB）；测试 malloc 失败路径、制造 OOM |
| `ulimit` | `ulimit -s 8192` | 限制**栈大小**（KB）；制造栈溢出实验 |
| `ulimit` | `ulimit -a` | 查看全部限制 |

- `gprof` 只统计**用户态**且需要 `-pg` 重编译，对库调用与内联函数不敏感；现代更推荐 `perf`。
- `-pg` 与 `-O2` 可同用，但内联会让调用图失真。

#### A.6.6 `/proc/<pid>/` 常用文件

| 路径 | 内容 |
|---|---|
| `/proc/<pid>/maps` | **虚拟内存区域（VMA）清单**：地址区间、权限（`rwxp`）、偏移、设备、inode、路径 |
| `/proc/<pid>/smaps` | 每个 VMA 的**详细统计**：`Rss`（驻留物理页）、`Pss`、`Shared_Clean/Dirty`、`Private_*`、`Swap` |
| `/proc/<pid>/smaps_rollup` | 上述统计的汇总（Linux 4.14+） |
| `/proc/<pid>/status` | 人类可读的进程状态：`VmPeak`/`VmSize`（虚拟）、`VmRSS`（驻留）、`VmData`/`VmStk`/`VmExe`、`Threads`、`SigQ` |
| `/proc/<pid>/stat` | 机器可读的单行状态（utime/stime、缺页数、优先级等） |
| `/proc/<pid>/statm` | 以**页**为单位的极简内存摘要（size/resident/shared/text/lib/data/dirty） |
| `/proc/<pid>/fd/` | 打开的文件描述符目录，`ls -l` 可见每个 fd 指向的文件/socket/pipe |
| `/proc/<pid>/fdinfo/<n>` | 某个 fd 的偏移、flags、锁信息 |
| `/proc/<pid>/task/` | 每个**线程**一个子目录（内含各自的 `stat`、`maps`、`status`） |
| `/proc/<pid>/cmdline` | 命令行参数（以 `\0` 分隔） |
| `/proc/<pid>/environ` | 环境变量（以 `\0` 分隔） |
| `/proc/<pid>/cwd`、`/proc/<pid>/exe`、`/proc/<pid>/root` | 当前目录、可执行文件、根目录的符号链接 |
| `/proc/<pid>/stack` | 内核态调用栈（需权限） |
| `/proc/<pid>/io` | 读写字节数、syscall 次数 |
| `/proc/meminfo` | 系统级内存概况：`MemTotal`/`MemFree`/`MemAvailable`/`Buffers`/`Cached`/`SwapTotal` |
| `/proc/self/maps` | 读自己的映射表（进程内 `cat /proc/self/maps` 即可观察自己的地址空间） |

- 排查"内存涨了但不知道是谁"：对比 `/proc/<pid>/smaps` 里各 VMA 的 `Rss`；排查"到底是哪个文件"：`ls -l /proc/<pid>/fd/`。

---

### A.7 存储与缓存速查

#### A.7.1 缓存组织的基本公式

| 符号 | 含义 | 关系 |
|---|---|---|
| $C$ | 缓存总容量（字节，不含标记/有效位等元数据） | $C = S \\times E \\times B$ |
| $B$ | 块大小（block size，字节） | $B = 2^b$，$b = \\log_2 B$ |
| $S$ | 组数（number of sets） | $S = 2^s$，$s = \\log_2 S$ |
| $E$ | 每组的行数（相联度，associativity） | $E=1$ 直接映射；$E=S$ 全相联；否则组相联 |
| $m$ | 地址位数 | — |
| $t$ | 标记位数（tag） | $t = m - s - b$ |
| $e$ | 行号位数 | $E = 2^e$ |

**地址切分（从高位到低位）**

```
 m-1            m-s-b   m-s-b-1        b   0
+----------------+-------+---------------+
|   tag (t 位)    | 组索引 |  块偏移 (b 位)  |
|                | (s 位) |               |
+----------------+-------+---------------+
```

| 字段 | 作用 |
|---|---|
| **块偏移**（block offset，低 $b$ 位） | 选块内第几个字节 |
| **组索引**（set index，中间 $s$ 位） | 选第几个组——**取的是中间位**，这样相邻块（低 $b$ 位不同）和跨组的大步长（高位不同）都能被区分 |
| **标记**（tag，高 $t$ 位） | 与行中存的标记比较，判定是否命中；**取高位** |

- **为什么组索引取中间位**：若用高位做索引，连续地址会全部落在同一组，造成严重冲突；用低位做索引则大步长访问（跨块）会全撞一组。中间位同时打散这两类模式。
- 一条缓存行（line）的元数据：**有效位**（valid bit）、**标记**（tag）、**脏位**（dirty bit，仅写回策略需要），外加 $B$ 字节的数据块。
- **组相联查找**：组索引选中一组 → 组内 $E$ 行**并行**比较标记与有效位 → 命中则按块偏移取字节，未命中则从下一级取块并替换组内某行。

#### A.7.2 命中 / 未命中 / 替换 / 有效位 / 脏位

| 术语 | 含义 |
|---|---|
| **命中**（hit） | 组内某行的有效位为 1 且标记与地址的 tag 相等 |
| **未命中**（miss） | 组内没有任何行同时满足"有效且标记相等" |
| **有效位 = 0** | 该行**从未被加载**过（冷启动）或已被作废；不能参与命中判定 |
| **脏位 = 1** | 该行被写过且尚未写回下一级；**被替换时必须先写回**（写回策略下） |
| **替换策略** | 直接映射：别无选择，覆盖唯一那行；组相联/全相联：LRU（最近最少使用）、近似 LRU、随机 |
| **写回**（write-back）与**写直达**（write-through） | 见 A.7.3 |
| **驱逐 / 牺牲行**（eviction / victim） | 被新块覆盖的那一行 |

#### A.7.3 写策略四格表

| 写命中时 | **写回（write-back）** | **写直达（write-through）** |
|---|---|---|
| 含义 | 只改缓存行，置脏位；**被驱逐时**才写回下一级 | 同时写缓存**和**下一级（可加写缓冲 write buffer 缓解） |
| 优点 | 减少写流量（同一行的多次写只下传一次） | 实现简单，下一级始终最新，易做多核一致性 |
| 缺点 | 需要脏位；驱逐路径变慢；一致性协议更复杂 | 写流量大（每次写都下传） |
| 典型使用 | L1/L2/L3 缓存（现代 CPU 全部用写回） | 早期机器、某些设备的帧缓冲路径 |

| 写未命中时 | **写分配（write-allocate）** | **非写分配（no-write-allocate）** |
|---|---|---|
| 含义 | 先把块**调入缓存**，再在缓存里写（"为写而读"） | 直接写下一级，**不调入**该块 |
| 优点 | 后续对同一块的写/读都能命中，利于局部性 | 避免"写一次就不再用"的数据污染缓存 |
| 缺点 | 每次写未命中都多一次读（fetch-on-write） | 后续再访问仍会未命中 |
| 典型搭配 | 与**写回**配套（最常见组合） | 与**写直达**配套 |

- **记忆**：真实的 L1/L2/L3 = **写回 + 写分配**；简单的写直达缓存常配**非写分配**。
- 写回策略下，**丢弃脏行是数据丢失**；`msync`/`fsync` 就是把"缓存里的脏数据"强制推到下一级。

#### A.7.4 3C 未命中分类与削减手段

| 类别 | 英文 | 成因 | 削减手段 |
|---|---|---|---|
| **强制/冷未命中** | compulsory (cold) miss | 该块**第一次**被访问，缓存里必然没有 | 增大块大小（利用空间局部性）、**预取**（prefetching）、软件预取指令 |
| **冲突未命中** | conflict miss | 组相联度不够：多个**活跃**块映射到**同一个组**，互相驱逐 | **提高相联度** $E$、增大 $B$（减少组数但每行更大）、**分块/填充**（padding）错开地址、编译器数组重排 |
| **容量未命中** | capacity miss | 工作集（working set）**大于**缓存容量，躲不开 | 增大缓存容量、**分块（blocking）**提高时间局部性、降低工作集、循环变换 |

- 判定顺序：先看是不是第一次访问（冷）→ 再看"若缓存无限大是否还会未命中"（容量）→ 否则是冲突。
- 冲突未命中是**唯一可以不改硬件、只改代码就消除**的一类（用 padding 或重排访问顺序）。
- 3C 之外常补充第四类：**一致性未命中（coherence miss）**——其他核写入了共享块导致本核的副本作废。

#### A.7.5 AMAT：平均访存时间

$$\text{AMAT} = \text{命中时间}(\text{Hit Time}) + \text{未命中率}(\text{Miss Rate}) \times \text{未命中惩罚}(\text{Miss Penalty})$$

**多级缓存的 AMAT 递推**（把下一级整体看成"上一级的未命中惩罚"）：

$$\text{AMAT}_{L1} = t_{L1} + m_{L1}\times\bigl(t_{L2} + m_{L2}\times(t_{L3} + m_{L3}\times t_{\text{Mem}})\bigr)$$

| 量 | 定义 | 讲义典型值 |
|---|---|---|
| 未命中率（miss rate） | 未命中次数 / 访问次数 $= 1 - $ 命中率 | L1 约 3–10%；L2 < 1%；L3 更小 |
| 命中时间（hit time） | 判定命中并把数据送到处理器的时间 | L1 约 4 周期；L2 约 10 周期；L3 约 40–75 周期 |
| 未命中惩罚（miss penalty） | 因未命中额外付出的时间 | 主存约 50–200 周期（趋势：**越来越大**） |
| **局部未命中率**（local miss rate） | 该级未命中数 / **到达该级**的访问数 | 用于递推 |
| **全局未命中率**（global miss rate） | 该级未命中数 / **L1 的总访问数** | 衡量该级对整体性能的贡献 |

- **算例（讲义原例）**：命中 1 周期、未命中惩罚 100 周期。
  97% 命中 → $1 + 0.03\\times100 = 4$ 周期；99% 命中 → $1 + 0.01\\times100 = 2$ 周期。
  **即"99% 命中率是 97% 的两倍好"**——命中率的微小提升在 AMAT 上被放大。
- 优化方向只有两条：**降低未命中率**（局部性、分块）或**降低未命中惩罚**（多级缓存、预取、更宽的带宽）。

#### A.7.6 Intel Core i7 缓存层次参数

| 层级 | 容量 | 相联度 $E$ | 组数 $S$ | 访问延迟 | 备注 |
|---|---|---|---|---|---|
| L1 d-cache（数据） | 32 KB | 8 路 | 64 | 4 周期 | 每核私有 |
| L1 i-cache（指令） | 32 KB | 8 路 | 64 | 4 周期 | 每核私有 |
| L2 unified（统一） | 256 KB | 8 路 | 512 | 10 周期 | 每核私有，指令+数据 |
| L3 unified（统一） | 8 MB | 16 路 | 8192 | 40–75 周期 | **所有核共享**（core 0–3） |
| 主存 | — | — | — | ~200 周期 | — |

- **块大小对所有层级统一为 64 字节**（$b=6$）。
- **L1 d-cache 的地址位域（47 位物理地址，Core i7）**：$C = 64 \\times 8 \\times 64 = 32768$ B ；
  $S = C/(E \\times B) = 32768/(8\\times64) = 64 \\Rightarrow s = 6$ ；$b = 6$ ；$t = 47 - 6 - 6 = 35$ 位。

```
物理地址 47 位 → L1 d-cache 划分
 46            12 11        6 5          0
+----------------+-----------+------------+
|  tag: 35 位     | set: 6 位 | off: 6 位   |
+----------------+-----------+------------+
例：地址 0x00007f7262a1e010
    tag  = 0x7f7262a1e
    set  = 0x0        (位 11..6)
    off  = 0x10       (位 5..0，落在块内第 16 字节)
```

| 层级 | 组索引位数 $s$ | 标记位数 $t$（47 位地址） |
|---|---|---|
| L1（32 KB，8 路，64 B 块） | 6 | 35 |
| L2（256 KB，8 路，64 B 块） | 9 | 32 |
| L3（8 MB，16 路，64 B 块） | 13 | 28 |

- 术语对照：**L1/L2/L3 = 缓存层级**；**i-cache = 指令缓存、d-cache = 数据缓存、unified = 二者合一**；**块偏移/组索引/标记**三段切分对每一层都成立，只是 $s$、$t$ 不同。
- 页大小 4 KB = 64 块/页，因此**同页内的相邻块组索引不同、标记相同**——这是"TLB 与 L1 可并行查找"的硬件基础（决定组索引的位在 VA 与 PA 中相同）。

#### A.7.7 局部性速查

| 类型 | 定义 | 例子 | 利用方式 |
|---|---|---|---|
| **时间局部性**（temporal locality） | 被访问过的数据/指令**很快会被再次访问** | 循环变量 `i`、循环体内的指令、被反复读的数组元素 | 把数据留在缓存里；分块提高复用次数 |
| **空间局部性**（spatial locality） | 被访问的地址**附近的地址很快会被访问** | 顺序遍历数组、顺序执行的指令流 | 增大块大小；顺序访问而非跳跃访问 |

| 访问模式 | 结论 |
|---|---|
| 步长 stride = 1（`a[i]` 逐个访问） | 最好：每个块只付一次未命中，块内 8 个 `double` 全用上 |
| 步长较小（stride < $B$） | 仍有空间局部性，但每块只用部分字节 |
| 步长较大（stride ≥ $B$） | 空间局部性基本消失，每个元素都可能是一块的首字节 → 未命中率趋近 100% |
| 循环嵌套顺序 | **按行访问（row-major，内层走列索引）** 对 C 数组最友好；按列访问（`a[j][i]` 外层变 i）会跨行跳转，未命中剧增 |
| 同一行被重复访问 | 时间局部性：内层循环复用同一行 $n$ 次，远好于每次换行 |

- 验算口诀：**"每 $B/\\text{sizeof(elem)}$ 个连续元素共享一次未命中"**。`double` 数组、64 B 块 → 每 8 个元素一次未命中。
- 矩阵 `sumA += a[i][j]` 按行 vs 按列，实测可差 **10 倍以上**（Core i7 上讲义实测从约 20 GB/s 掉到不足 2 GB/s）。

#### A.7.8 分块（Blocking）速查

**原则**：把大矩阵切成 $B \\times B$ 的小块，让**一个块在被换出之前被尽可能多次复用**——把"对全矩阵的时间局部性"压缩到"对一个小块的时间局部性"，使工作集装得进缓存。

| 要点 | 内容 |
|---|---|
| 目标 | 提高**时间局部性**，把容量未命中降下来 |
| 工作集约束 | 同时活跃的块要能全部驻留：对矩阵乘，需要"两个输入块 + 一个输出块" |
| **块大小选择** | 取**满足 $3B^2 < C$ 的最大 $B$**（$C$ 为缓存容量，单位与 $B^2$ 一致）——三个 $B\\times B$ 块（`a`、`b`、`c`）必须同时装进缓存 |
| 未命中数对比 | 不分块：$\\frac{9}{8}n^3$ 次未命中；分块：$\\frac{n^3}{4B}$ 次（以"块 = 8 个 double"为例） |
| 收益来源 | 矩阵乘本身有 $O(n)$ 的时间局部性（3$n^2$ 个输入元素被用 $2n^3$ 次），**但必须写对循环顺序**才能兑现 |
| 常见陷阱 | $B$ 太大 → 三块装不下，退化为不分块；$B$ 太小 → 块内空间局部性用不满；索引变量写错 → 结果正确但没提速 |
| 推广 | 分块思路可用于转置、矩阵向量乘、卷积、稀疏矩阵；本质是"**循环变换 + 提高局部性**"，不改变算术量 |

**64×64 转置的 4×4 子块 + 8 局部变量技巧（Cache Lab 经典）**

| 步骤 | 做法 | 为什么 |
|---|---|---|
| ① 不直接逐元素转置 | 朴素 `B[j][i] = A[i][j]` 使 A 按行读、B 按列写，两者都跨块跳跃 | 一次访问只用一个元素，空间局部性浪费 |
| ② 划 4×4 子块 | 外层循环以 4 为步长遍历块，块内先把 A 的 4 行读进局部变量 | 块内 4 行×4 列 = 16 个元素分布在 2 个块内（8 个 `int`/块） |
| ③ 用 **8 个局部变量**（或 `int` 寄存器变量）作中转 | 把 4×4 的 16 个元素分批搬到 B 的 4×4 区 | 减少对 B 的重复读写：B 的 4 行各只在最终写入时被触碰一次 |
| ④ 对角线块特判 | 4×4 块落在对角线上时，A 块与 B 块重叠，逐元素处理更省 | 避免同一块的读写互相驱逐 |
| ⑤ 更激进：8×8 块 + 变体 | 把命中率从 4×4 方案进一步提升到 ~95%+ | Cache Lab 满分需要更细的分块与临时变量 |

- **测什么**：Cache Lab 用 `csim` 模拟器统计 `hits/misses/evictions`，再对真实 `trans` 用 `valgrind --tool=cachegrind` 看 `D1 miss rate`。
- 判断是否有效的实操：先 `perf stat -e L1-dcache-load-misses,LLC-load-misses ./prog` 看基线，再改分块后对比。

#### A.7.9 存储器山（Memory Mountain）读法

**定义**：以**读取吞吐率（read throughput，MB/s）**为高度，横轴为**工作集大小**（size，从 32 KB 到 128 MB，对数刻度），另一轴为**步长**（stride，1–11 个 `long`，即 8–88 字节）的三维地形图。它把内存系统的空间/时间局部性**一次性画成一张图**。

| 图上要素 | 含义 |
|---|---|
| **山脊（ridges of temporal locality）** | 固定步长、扫过工作集大小时看到的高处：工作集落进 L1/L2/L3 时吞吐高，超出后陡降——每一级缓存是一道"平台"，平台边缘即该级容量 |
| **山坡（slopes of spatial locality）** | 固定工作集、增大步长时看到的下降：步长超过块大小（64 B = 8 个 `long`）后吞吐骤降，因为每块只用 1 个元素 |
| 最高峰（约 14000 MB/s） | 小工作集 + 步长 1：全在 L1，且空间局部性拉满 |
| 最低谷（约 2000 MB/s 以下） | 工作集远大于 L3（128 MB）+ 大步长：几乎每次访问都到主存 |
| 台阶状下降的位置 | 32 KB（L1）、256 KB（L2）、8 MB（L3）三处——**台阶边界就是各级缓存的容量** |
| 尾部"翘起"（aggressive prefetching） | 大步长、大工作集时硬件预取器仍能拉起一部分吞吐 |

- **读法三步**：① 沿某根"脊"（固定 stride）走，定位三级台阶 → 读出 L1/L2/L3 容量；② 沿某个"坡"（固定 size）走，看 stride 从 1 增到 11 的跌幅 → 读出块大小的影响；③ 比较不同机器的山形，即可判断其缓存层次与预取策略的强弱。
- 测量方法（讲义 `mountain.c`）：`test(elems, stride)` 用 **4×4 循环展开**累加，先跑一次**预热缓存**，再跑一次计时；对多种 `elems`/`stride` 组合重复，绘制成图。
- 实践意义：**"为什么我的程序忽快忽慢"**——把访问模式（size × stride）落在这张图上，就知道它在吃哪一级缓存。

---

### A.8 虚拟内存速查

#### A.8.1 术语表

| 术语 | 英文 | 含义 |
|---|---|---|
| **VA** | Virtual Address | 虚拟地址，CPU 发出的地址，每进程独立 |
| **PA** | Physical Address | 物理地址，DRAM 上的真实地址 |
| **页** | page（VP, virtual page） | 虚拟地址空间被切成的定长块，大小 $P = 2^p$（x86-64 典型 4 KB，$p=12$） |
| **页帧 / 物理页** | page frame（PP, physical page） | 物理内存被切成的同样大小的块 |
| **页表** | page table | 把 VPN 映射到 PPN 的**数组**，每项是一个 PTE；每个进程一棵（或一套） |
| **PTE** | Page Table Entry | 页表项：PPN + 有效位 + 权限位（R/W/X/U）+ 脏位 + 访问位 |
| **VPN / VPO** | Virtual Page Number / Offset | 虚拟地址的高位（页号）与低位（页内偏移） |
| **PPN / PPO** | Physical Page Number / Offset | 物理地址的高位与低位 |
| **MMU** | Memory Management Unit | CPU 内完成地址翻译的硬件 |
| **TLB** | Translation Lookaside Buffer | MMU 内缓存 **VPN → PPN 完整映射**的小型组相联缓存 |
| **缺页** | page fault | 访问的页不在内存（PTE 有效位为 0）时触发的异常，由内核处理 |
| **VMA** | Virtual Memory Area | Linux 用一个结构描述一段连续、同权限的虚拟地址区间 |
| **按需分页** | demand paging | 页只在**第一次被访问**时才换入，`mmap`/`malloc` 本身不分配物理页 |
| **抖动** | thrashing | 工作集超过物理内存，页面反复换入换出，性能崩溃 |
| **CR3** | — | 存放**顶级页表**物理基址的寄存器；切换进程即切换 CR3 |

#### A.8.2 关键恒等式与翻译流程

| 恒等式 / 规则 | 内容 |
|---|---|
| **偏移不变** | $\\text{VPO} = \\text{PPO}$（页内偏移在翻译前后**完全相同**，因此可并行取 TLB 与 L1） |
| 位数划分 | VA 共 $m$ 位：$p$ 位页偏移 + $(m-p)$ 位 VPN；PA 共 $m^{\\prime}$ 位：$p$ 位偏移 + $(m^{\\prime}-p)$ 位 PPN |
| x86-64 数值 | $m = 48$（规范地址），$p = 12$ → **VPN 36 位**、页偏移 12 位；PPN 最多 40 位（52 位物理地址） |
| 页大小 | $P = 2^p$，4 KB = $2^{12}$；大页 2 MB（$2^{21}$）、1 GB（$2^{30}$） |
| 页内地址 | $\\text{VA} = \\text{VPN} \\times P + \\text{VPO}$，$\\text{PA} = \\text{PPN} \\times P + \\text{VPO}$ |
| PTE 大小 | x86-64 每项 **8 字节**；每级页表 **512 项** × 8 B = 4096 B = **恰好 1 页** |
| 单级页表为何不可行 | $2^{36}$ 个 VPN × 8 B = $2^{39}$ B = **512 GiB**（每进程一张） |

**地址翻译数据通路**

```
 CPU 发 VA
   │
   ├─ VPO (12 位) ────────────────────────────────┐
   │                                              │
   └─ VPN (36 位) → [ TLB 查找 ]                  │
                     │命中                        │
                     │   PPN ──────────┐          │
                     │未命中           │          │
                     ↓                 ↓          ↓
              查页表（最多 4 级访存）→ PPN ──► 拼接 PA = PPN|VPO
                                                 │
                                                 ↓
                                         查 L1/L2/L3 → 取数据
```

| 步骤 | 内容 |
|---|---|
| 1 | MMU 用 VPN 查 TLB（TLBI 选组，TLBT 比对标记） |
| 2 | **命中** → 直接得 PPN，省掉访存；**未命中** → 走页表（PTEA = 页表基址 + VPN×8） |
| 3 | 逐级查页表；若某级 PTE 有效位为 0 → **缺页异常**，陷入内核 |
| 4 | 内核处理缺页后更新 PTE，重新执行**原指令**（此时 TLB 已填充） |
| 5 | 拼出 PA，与 VPO 一起访问缓存/主存 |

- **TLB 命中省掉一次内存访问**；TLB 未命中只是"多一次访存"，与缺页（可能含磁盘 I/O）**量级完全不同**。
- 无论页表有多少级，**TLB 缓存的始终是完整的 VPN → PPN 映射**。

#### A.8.3 x86-64 四级页表位划分

| 字段 | 位区间 | 位数 | 名称 |
|---|---|---|---|
| PGD 索引（L4） | 47–39 | 9 | Page Global Directory（顶级，CR3 指向它） |
| PUD 索引（L3） | 38–30 | 9 | Page Upper Directory |
| PMD 索引（L2） | 29–21 | 9 | Page Middle Directory |
| PTE 索引（L1） | 20–12 | 9 | Page Table Entry |
| 页偏移 | 11–0 | 12 | VPO = PPO |

```
 47        39 38       30 29       21 20       12 11          0
+------------+-----------+-----------+-----------+-------------+
|  PGD (9)   |  PUD (9)  |  PMD (9)  |  PTE (9)  |  offset(12) |
+------------+-----------+-----------+-----------+-------------+
     │            │            │            │
   CR3 ──►L4表──►L3表────────►L2表────────►L1表────────► PPN | offset = PA
```

| 要点 | 说明 |
|---|---|
| 每级表大小 | 512 项 × 8 B = 4 KB = 1 页，天然页对齐（**这就是为什么每级恰好 9 位**：$2^9 = 512$） |
| 覆盖范围 | 顶级每项覆盖 512 GB；每下一级依次覆盖 1 GB、2 MB、4 KB |
| 为什么要多级 | 只为**真正用到的地址路径**分配下级页表：散布在地址空间各处的 5 个区域最多占 5×4 = 20 张表 ≈ 80 KB，而非 512 GiB |
| 代价 | TLB 未命中时最多 **4 次**访存（可被页表缓存/大页缓解） |
| PTE 关键位 | `P` 有效位（bit 0）、`R/W` 可写（bit 1）、`U/S` 用户/内核（bit 2）、`A` 访问位（bit 5）、`D` 脏位（bit 6）、`XD` 不可执行（最高位，即 NX/No-eXecute） |
| 规范地址 | x86-64 的 48 位 VA 必须符号扩展到 64 位（高位全 0 或全 1）；非规范地址立即 `#GP` |
| CR3 切换 | 进程切换 = 写 CR3 = 换一棵页表树；内核页表映射在所有进程地址空间的高半部 |

#### A.8.4 缺页处理流程

| 步骤 | 动作 | 判定 |
|---|---|---|
| 1 | CPU 访问 VA，MMU 查到某级 PTE 的**有效位为 0** | 硬件触发缺页异常（`#PF`），压入出错地址（`CR2`）与 error code |
| 2 | 内核取 `mmap_lock`，用出错地址查 **VMA**（`find_vma`） | 若**没有**包含该地址的 VMA → **段错误（SIGSEGV）**，进程终止 |
| 3 | 若找到 VMA，检查**权限**是否与本次访问匹配（读/写/执行、用户/内核） | 写只读页（如 COW 页、`.text`）→ 保护异常：**合法则做 COW 复制**，非法则 `SIGSEGV` |
| 4 | 判定缺页类型 | **次缺页（minor）**：页已在内存（page cache 命中、COW、零页）→ 直接建映射；**主缺页（major）**：需从磁盘/交换区读入 → 阻塞 I/O |
| 5 | 选出牺牲页（若物理内存满）：脏页先**写回**，干净页直接丢弃；更新被牺牲页的 PTE 有效位为 0 | 替换策略（近似 LRU / CLOCK） |
| 6 | 把页换入内存，**更新 PTE**：填 PPN、置有效位、设置权限位（清脏位） | — |
| 7 | **重新执行触发缺页的那条指令** | 这一次翻译成功，程序继续，用户态完全察觉不到 |
| 8 | 若配置了预取/`MAP_POPULATE`，可一并换入相邻页 | 减少后续缺页次数 |

- **关键洞察**："合法的地址 + 无效的 PTE" 是**正常流程**（按需分页），不是错误；只有"地址不在任何 VMA"或"权限不符"才是真正的段错误。
- **error code 读法**（`dmesg` 里 `segfault at 0 error 4`）：bit 0 = 保护违规（1）还是页不存在（0）；bit 1 = 写（1）还是读（0）；bit 2 = 用户态（1）还是内核态（0）。
- 观察命令：`perf stat -e page-faults,minor-faults,major-faults ./prog`；`/usr/bin/time -v ./prog` 看缺页总数。

#### A.8.5 TLB 参数（Intel Core i7）

| TLB | 项数 | 相联度 | 组数 | 覆盖内容 |
|---|---|---|---|---|
| **L1 d-TLB**（数据） | 64 | 4 路 | 16 | 4 KB 页的 VPN → PPN |
| **L1 i-TLB**（指令） | 128 | 4 路 | 32 | 同上（指令侧） |
| **L2 TLB**（统一） | 512 | 4 路 | 128 | 统一，容量更大、延迟更高 |
| 大页 TLB | 32 项 | 4 路 | 8 | 2 MB / 4 MB 大页 |

- **位域计算（L1 d-TLB）**：VPN 共 36 位，组数 $= 64/4 = 16 \\Rightarrow$ **TLBI = 4 位**；**TLBT = 36 − 4 = 32 位**。
- **性能含义**：工作集（活跃页数）≤ TLB 项数时命中率接近 100%；一旦超出，每轮都要换入换出，命中率崩盘——**"工作集小于 TLB 容量则快"**。
- 大页（2 MB）把"每页 4 KB 一个 TLB 项"变成"每页 2 MB 一个 TLB 项"，同样项数覆盖 512 倍的内存，是数据库/大内存程序的常用优化。
- 观察命令：`perf stat -e dTLB-loads,dTLB-load-misses,page-faults ./prog`。
- **TLB 与缓存的协同**：VPO = PPO，且决定 L1 组索引的位在 VA 与 PA 中相同，因此 **TLB 查找与 L1 组索引可以并行**；页命中时两者同时完成，进一步掩盖翻译延迟。

#### A.8.6 Linux VMA 标志

| 标志 | 含义 |
|---|---|
| `VM_READ` | 可读（对应 `PROT_READ`，`maps` 里的 `r`） |
| `VM_WRITE` | 可写（`PROT_WRITE`，`w`） |
| `VM_EXEC` | 可执行（`PROT_EXEC`，`x`） |
| `VM_SHARED` | **共享**映射：写入对同一映射的其他进程可见（`MAP_SHARED`，`maps` 里的 `s`）；否则为私有 `p` |
| `VM_MAYREAD` / `VM_MAYWRITE` / `VM_MAYEXEC` / `VM_MAYSHARE` | "**允许**被 `mprotect` 改成"对应权限的标志位 |
| `VM_GROWSDOWN` | 该 VMA 是**向下增长**的栈区域（缺页地址紧邻 `vm_start` 之下时自动扩展栈） |
| `VM_GROWSUP` | 向上增长（某些体系结构的栈） |
| `VM_IO` / `VM_PFNMAP` | 内存映射 I/O（设备寄存器），**不参与分页** |
| `VM_DONTCOPY` / `VM_DONTEXPAND` / `VM_LOCKED` | `fork` 时不复制 / 不允许 `mremap` 扩展 / 锁定在内存中不换出（`mlock`） |
| `VM_ACCOUNT` / `VM_NORESERVE` | 计入/不计入地址空间记账（overcommit 相关） |
| `VM_HUGETLB` | 使用大页 |

- `/proc/<pid>/maps` 每行的权限串 `rwxp` 就是 `VM_READ\|VM_WRITE\|VM_EXEC` 加 `VM_SHARED`（`s` 而非 `p`）的组合；`---p` 表示保留但不可访问（如 guard page）。
- 权限位在 **PTE 里**（硬件实际执行），VMA 标志是**内核记账**；`mprotect` 同时改两者中受影响页的 PTE。

#### A.8.7 `mmap` 用法与三大用途

```c
#include <sys/mman.h>
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int munmap(void *addr, size_t length);
int mprotect(void *addr, size_t len, int prot);
int msync(void *addr, size_t len, int flags);
```

| 参数 | 取值 | 说明 |
|---|---|---|
| `addr` | `NULL`（推荐）或指定地址 | `NULL` 让内核选地址 |
| `length` | 字节数 | 内核按页向上取整 |
| `prot` | `PROT_READ` / `PROT_WRITE` / `PROT_EXEC` / `PROT_NONE` | 可或组合；`PROT_NONE` 用作 guard |
| `flags` | **必须**含 `MAP_SHARED` 或 `MAP_PRIVATE` 之一 | 见下表 |
| `fd` / `offset` | 文件描述符与偏移 | `offset` **必须是页大小整数倍**；匿名映射传 `-1, 0` |
| 返回值 | 页对齐地址；失败返回 `MAP_FAILED`（`(void*)-1`）并置 `errno` | 必须检查 |

| flag | 含义 |
|---|---|
| `MAP_SHARED` | 共享：对映射的写入**对其他进程可见**，文件映射还会写回文件 |
| `MAP_PRIVATE` | 私有：**写时复制**（COW），改动不影响文件与其他进程 |
| `MAP_ANONYMOUS` / `MAP_ANON` | 匿名映射：不对应文件（`fd` 忽略），内容是**零页**；`malloc` 的大块走这条路 |
| `MAP_FIXED` | 强制映射到 `addr`（**危险**：会覆盖已有映射） |
| `MAP_FIXED_NOREPLACE` | 要求地址精确且不覆盖已有映射（较安全） |
| `MAP_POPULATE` | 预先建立页表并换入（把缺页代价提前，适合已知会被全访问的区域） |
| `MAP_NORESERVE` | 不为交换空间预留（overcommit） |
| `MAP_LOCKED` / `mlock` | 锁定不换出 |
| `MAP_HUGETLB` | 使用大页 |
| `MAP_GROWSDOWN` | 向下增长（`mmap` 出的栈） |
| `MAP_STACK` | 提示内核这是栈区域 |

| 三大用途 | 说明 |
|---|---|
| ① **映射文件**（file mapping） | `fd` 指向文件，读写映射区即读写文件；`MAP_SHARED` 写入经 page cache 反映到文件（`msync` 强制刷盘），`MAP_PRIVATE` 得到写时复制的私有副本。可用于"把大文件当数组随机访问"而无需 `read` |
| ② **匿名内存**（anonymous memory） | `MAP_ANONYMOUS`：不涉及文件，初始为零页；`malloc` 大块、线程栈、`brk` 之外的所有内存需求都可由它满足；`free` 时 `munmap` 直接归还内核 |
| ③ **共享内存 / 进程间通信**（IPC） | 两个进程对同一文件（或 `shm_open` 对象）做 `MAP_SHARED` 映射，读写同一批物理页即完成共享；共享库的代码段也是这种方式被多个进程共享 |

- 惯用法：`char *p = mmap(NULL, len, PROT_READ\|PROT_WRITE, MAP_PRIVATE\|MAP_ANONYMOUS, -1, 0);`
- **`mmap` 之后并不分配物理页**（实测缺页增量为 0）——这正是按需分页：只有第一次触碰某页才产生次缺页。
- `munmap` 归还整个区间；`madvise(MADV_DONTNEED)` 归还**物理页但保留 VMA**（适合"释放但留着以后再分配"）。

#### A.8.8 `fork` 写时复制（COW）与 `execve`

| 机制 | 要点 |
|---|---|
| `fork` 的语义 | 复制整个地址空间（代码、数据、堆、栈、文件描述符表） |
| `fork` 的实现 | **只复制页表，不复制物理页**：父子 PTE 指向同一批物理页，且都标记为**只读 + COW** |
| 写时复制 | 任一方**写入**触发保护异常 → 内核复制**那一页**，把写方的 PTE 指向新副本并恢复可写，另一方的 PTE 保持不变（引用计数减 1） |
| 为什么快 | `fork` 只花"复制页表 + 建 VMA"的时间（微秒级），与地址空间大小**基本无关** |
| 引用计数 | 物理页有引用计数；最后一个使用者退出或 `munmap` 时才真正释放 |
| 与 `execve` 的关系 | `fork` 后立刻 `execve` 时，子进程往往一页都没写过 → 全部页面直接丢弃，**零拷贝浪费**；这正是 `posix_spawn`/`vfork` 想优化的场景 |
| 陷阱 | `fork` 后父子共享文件偏移（同一个打开文件表项）；`fork` 在内核缓冲区/锁状态下调用会有可重入问题 |

| `execve` 的内存映射步骤 | 内容 |
|---|---|
| 1 | 校验可执行文件（ELF 头、体系结构） |
| 2 | 丢弃**旧**地址空间的用户态映射（内核映射保留） |
| 3 | 映射 `.text` 等只读可执行段：**文件映射 + `PROT_EXEC`，`MAP_PRIVATE`**，PTE 初始无效 |
| 4 | 映射 `.data`：`MAP_PRIVATE`（写时复制）、可读写；初始内容来自文件 |
| 5 | `.bss` 等零初始化区域：**匿名映射**，靠"零页 + 缺页"实现"初始化为 0"而**不占文件空间** |
| 6 | 映射共享库（`ld.so` 先启动，按 `NEEDED` 加载），代码段多进程共享同一份物理页 |
| 7 | 建立初始栈：压入 `argv`、`envp`、辅助向量（auxv） |
| 8 | 把 `%rip` 指向**入口点或动态链接器**，跳入用户态开始执行 |

- **核心结论**：`execve` 之后**所有页都是按需换入的**（PTE 无效），所以"启动新程序"并不需要先把整个程序读进内存。
- 观察：`cat /proc/<pid>/maps` 能看到 `r-xp` 的 `.text` 文件映射、`rw-p` 的匿名堆栈、`r--p` 的只读共享库段。

#### A.8.9 `malloc` 的 `brk` / `mmap` 双路径

| 路径 | 触发条件 | 机制 | 释放行为 |
|---|---|---|---|
| **`brk` 堆** | 较小的请求（默认 < 128 KB） | 分配器用 `sbrk`/`brk` 在**堆顶**推进 `brk` 指针；`[heap]` 区间就是这段虚拟区间 | `free` 通常只是把块**放回空闲链表**，不归还内核（堆顶收缩只在末尾大块被释放时发生） |
| **`mmap` 区** | 较大的请求（默认 ≥ `MMAP_THRESHOLD` = **128 KB**） | 直接用 `mmap(NULL, size, PROT_READ\|WRITE, MAP_PRIVATE\|MAP_ANONYMOUS, -1, 0)` 建独立映射 | `free` 可直接 `munmap` **立即归还内核** |

| 参数 / 命令 | 说明 |
|---|---|
| `M_MMAP_THRESHOLD`（默认 **128 KB**） | 超过它就走 `mmap`；glibc 还会**动态调整**该阈值（观察到 `mmap` 块被 `free` 后可能上调） |
| `M_TRIM_THRESHOLD`（默认 128 KB） | 堆顶空闲超过该值就收缩堆（`brk` 回退） |
| `M_TOP_PAD` | 每次 `sbrk` 额外多要的余量，减少系统调用次数 |
| `mallopt(M_MMAP_THRESHOLD, n)` | 运行时调整阈值 |
| `MALLOC_MMAP_THRESHOLD_` 环境变量 | 同上（也支持 `MALLOC_TRIM_THRESHOLD_`、`MALLOC_TOP_PAD_`、`MALLOC_CHECK_`） |
| 为什么分两条路 | `brk` 路径分配快、碎片可控，但**归还困难**；`mmap` 路径**归还容易**，但每次分配/释放都是一次系统调用与一次缺页，且地址不连续、易造成 TLB/缓存压力 |
| 实测特征 | `[heap]` 中的地址（如 `0x1351000`）来自 `brk`；`0x7f...` 附近来自 `mmap` |
| 观察 | `malloc(1<<20)` 落在 `mmap` 区；`strace -e trace=brk,mmap,munmap` 可看到分配器的系统调用轨迹 |

- **容易混淆**：`brk` 推的是**进程级**的堆顶；`mmap` 建的是**独立 VMA**。两者都只申请**虚拟**空间，物理页仍按需分配（因此"`malloc` 了 1 GB 却没爆内存"是正常的，直到你逐页触碰）。

#### A.8.10 常用命令（VM 相关）

| 命令 | 作用 |
|---|---|
| `getconf PAGE_SIZE` | 打印系统页大小（x86-64 上通常 `4096`）；`getconf _PHYS_PAGES` 得物理页数 |
| `cat /proc/self/maps` | 读**当前 shell 执行的 cat 自己**的 VMA 清单（地址区间、权限、偏移、路径） |
| `cat /proc/<pid>/maps` | 观察指定进程的地址空间（`r-xp` 代码、`rw-p` 数据、`[heap]`、`[stack]`、`[vdso]`） |
| `pmap -x <pid>` | 以表格列出每个映射的 `Kbytes`、`RSS`、`Dirty`、`Mode`、`Mapping`（比 `maps` 更直观） |
| `pmap <pid>` | 简表（`-q` 只列映射、`-d` 显示设备格式） |
| `ulimit -v` | 查看/设置**虚拟内存**上限（KB） |
| `ulimit -s` | 栈大小上限（KB，默认 8192） |
| `ulimit -c unlimited` | 允许 core dump（配合 `gdb ./prog core`） |
| `ulimit -a` | 查看全部限制 |
| `cat /proc/<pid>/statm` | 以页为单位的极简内存摘要 |
| `cat /proc/<pid>/smaps` | 各 VMA 的详细内存统计（RSS / PSS / Shared / Private / Swap） |
| `cat /proc/meminfo` | 系统内存与交换区概况 |
| `sysctl vm.overcommit_memory` | 查看 overcommit 策略（0 启发式、1 总是允许、2 严格） |
| `perf stat -e page-faults,minor-faults,major-faults,dTLB-load-misses ./prog` | 统计缺页与 TLB 行为 |
| `valgrind --tool=cachegrind ./prog` | 模拟各级缓存的命中率（`cg_annotate` 按行/函数展开） |
| `getconf LEVEL1_DCACHE_SIZE` / `LEVEL1_DCACHE_ASSOC` / `LEVEL1_DCACHE_LINESIZE` | 直接问系统 L1 数据缓存的容量/相联度/块大小 |
| `lscpu` | 汇总 CPU 与缓存层次信息（`-C` 可看缓存详情） |
| `dmesg \| tail` | 查看 `segfault at ... error N` 的出错地址与 error code |

### A.9 动态内存分配器速查

> 对应讲次：Lecture 13（`F25-13-malloc-basic.txt`）、Lecture 14（`F25-14-malloc-advanced.txt`）；
> 教材 CS:APP3e 第 9.9–9.11 节；关联 **L5a/L5b Malloc Lab**。
> 表中未出现在 F25 讲义里的实现细节（glibc 调参、错误信息文本）已标注**「补充说明」**。

#### A.9.1 `malloc` 家族接口语义

| 接口 | 原型（`<stdlib.h>`） | 语义要点 |
|---|---|---|
| `malloc` | `void *malloc(size_t size)` | 成功：返回指向**至少 `size` 字节**的块的指针，且在 x86-64 上按 **16 字节边界对齐**；`size == 0` 时**返回 `NULL`**（讲义口径）；失败：返回 `NULL` 并置 `errno` |
| `calloc` | `void *calloc(size_t nmemb, size_t size)` | `malloc` 的**清零**版本：分配 `nmemb * size` 字节并**逐字节初始化为 0**；不做溢出检查时 `nmemb*size` 可能回绕 |
| `realloc` | `void *realloc(void *p, size_t size)` | 改变**已分配块**的大小；返回新块起始地址（**可能与 `p` 不同**），旧内容保留下 `min(旧大小, 新大小)` 字节 |
| `free` | `void free(void *p)` | 把 `p` 指向的块还给分配器；`p` **必须**来自之前的 `malloc`/`calloc`/`realloc`；`free(NULL)` 是合法的空操作 |
| `sbrk` | `void *sbrk(intptr_t incr)` | **分配器内部**用来增长/收缩堆顶（`brk` 指针）；应用代码不应直接调用 |

**`realloc` 的三个边界情形**（讲义只给了"改变大小"一句，下表为**补充说明**，按 C 标准与 glibc 实际行为）：

| 调用 | 等价于 | 注意 |
|---|---|---|
| `realloc(NULL, n)` | `malloc(n)` | 合法，返回新块 |
| `realloc(p, 0)` | glibc：`free(p)` 并返回 `NULL` | C17 起为**实现定义**，C23 起为**未定义行为**；**绝不要写 `p = realloc(p, 0)`**，否则 `p` 被覆盖成 `NULL`，旧块已释放 |
| `realloc(p, n)` **失败** | 返回 `NULL`，**原块 `p` 保持不变** | 故必须写成 `tmp = realloc(p, n); if (!tmp) { /* p 仍有效 */ } else p = tmp;` |

- **`malloc(0)` 的两套口径**：讲义规定返回 `NULL`；glibc 实际返回一个**唯一的最小可 `free` 块**（非 `NULL`）。写代码时应把 `NULL` 当"分配失败或零长度"处理，不要依赖 glibc 行为。
- **成功/失败判定**：`malloc` 失败时返回 `NULL` 并**设置 `errno`**，所以 `perror("malloc")` 有意义；但 `size == 0` 也返回 `NULL` 且**不设置 `errno`**——这是 `malloc` 返回值语义最容易踩的歧义。

#### A.9.2 分配器硬性约束与两个目标

| # | 硬性约束 | 含义 | 为什么不能违反 |
|---|---|---|---|
| 1 | **不假设请求模式** | 应用可发出**任意** `malloc`/`free` 序列；分配器不能假设大小会重复、不能假设"先全分配后全释放"；`free` 的参数必须是 malloc 出来的块 | 分配器是库，无法预知调用者行为 |
| 2 | **立即响应** | 必须在 `malloc` 返回前给出块，**不能重排或缓冲请求**（不能攒一批再统一处理） | 调用者拿到指针后会立刻写入 |
| 3 | **只能用堆里的空闲内存** | 已分配的块必须落在**堆的空闲内存**中；**不能移动已分配块**（禁止压缩 compaction）；只能读写空闲块 | 移动块会使所有已发出的指针失效——C 没有 GC 的重定位能力 |

- **附加要求：对齐**。必须满足所有对齐约束：讲义口径是 x86-64 上 **16 字节**；**Malloc Lab 放宽到 8 字节**（与 libc `malloc` 一致），`mdriver` 会强制检查。
- **两个目标（常互相冲突）**：

| 目标 | 定义 | 度量 |
|---|---|---|
| **吞吐率**（throughput） | 单位时间内**完成的请求数**（`malloc` 与 `free` 都算） | 例：10 秒内 5000 次 `malloc` + 5000 次 `free` → 1000 ops/s |
| **峰值内存利用率**（peak memory utilization） | 让堆尽量小地装下程序的峰值载荷 | 见 A.9.3 |

#### A.9.3 碎片与利用率

**利用率与开销（讲义定义）**：设第 $k$ 次请求后

$$P_k = \sum_{\text{当前已分配块}} \text{payload},\qquad
H_k = \text{当前堆大小（只增不减）},\qquad
\max_{i\le k} P_i = \text{峰值聚合载荷}$$

$$U_k = \frac{\sum \text{已分配载荷}}{\text{堆大小}} = \frac{P_k}{H_k},
\qquad
O_k = \frac{H_k}{\max_{i\le k} P_i} - 1.0$$

*   $U_k$ 越接近 1 越好；$O_k$ 是"堆里没被程序数据用掉的比例"，完美分配器趋近于 0。
*   讲义 benchmark `syn-array-short`（10 个块的分配/释放序列）峰值载荷为 **90036 字节**，用于画出 $P_k$ 与 $\\max_{i\\le k}P_i$ 的曲线。

| 碎片类型 | 定义 | 公式 | 可测性 |
|---|---|---|---|
| **内部碎片**（internal fragmentation） | **某个块内**载荷小于块大小 | $\\text{frag}_{\\text{int}} = asize - p$，其中 $p$ 为请求载荷、`asize` 为实际块大小（含 header/footer 与对齐 padding） | **容易**：只依赖**过去**的请求模式 |
| **外部碎片**（external fragmentation） | 堆里**总空闲量足够**，但**没有单个空闲块**够大 | 无解析公式；用"为了满足请求而额外扩大的堆"来度量 | **困难**：依赖**未来**请求模式 |

- 内部碎片的三个来源：① 维护堆数据结构（header/footer）的开销；② 对齐 padding；③ **显式策略**（例如"小请求也给一个大块"）。
- 讲义 benchmark 数值：完美适配内部碎片开销 **1.6%**；外部碎片叠加后最佳适配 **8.3%**、首次适配 **11.9%**、下次适配 **21.6%**。

#### A.9.4 块格式与边界标记（boundary tag）

**块布局**（`asize` = 块总大小，含 header 与 footer）：

```
   low addr                                                      high addr
   +---------------------+--------------------------+---------------------+
   | header: size | a    | payload (app data)       | footer: size | a    |
   | 1 word = 8 bytes    | asize - 16 bytes         | 1 word = 8 bytes    |
   +---------------------+--------------------------+---------------------+
   ^                     ^                          ^
  hdr                 bp = hdr + 8                hdr + asize - 8
```

**size 字的位域**：

```
   63                        3   2   1   0
   +--------------------------+---+---+---+
   |       block size         | 0 | P | A |
   +--------------------------+---+---+---+
                                  A = 本块已分配（cur allocated）
                                  P = 前一块已分配（prev allocated）
```

| 字段 | 编码内容 | 说明 |
|---|---|---|
| header（1 word） | 块大小 $\\vert $ 分配位 | 大小**含 header 与 footer**；读取时必须**先屏蔽低位**（`size & ~0x7`） |
| footer（1 word） | **header 的副本**（边界标记） | 让分配器能**反向遍历**；仅"需要合并"的实现才加，代价是每块多 1 word |
| payload | 应用数据 | 仅已分配块有；返回给应用的指针是 `hdr + 8`（跳过 header） |

**为什么低 3–4 位可以复用为标志位**：

*   块大小**必然是 8 的倍数**（因为所有块至少 8 字节对齐、且大小按 word 向上取整），所以 size 的**低 3 位恒为 0**；当块按 **16 字节对齐**时**低 4 位也恒为 0**。
*   既然这些位永远是 0，就不必"存一个恒为 0 的位"，直接**借用来当标志**。读取大小前 `& ~0x7` 即可。
*   讲义在"合并四情况"里实际用**低 2 位**：`P<<1 \| A`（bit1 = 前块是否已分配，bit0 = 本块是否已分配）——有了 bit1，判断前块状态**不用**再去读前块 header，合并变成真正的 O(1)。
*   Malloc Lab 用 16 字节对齐时可用低 4 位，多出的位可放"块是否来自 `mmap`"等标记。

**边界标记为何给出 O(1) 合并**：

```
   prev_hdr          hdr (= cur_hdr)                next_hdr = hdr + asize
   +--------+--------+--------+-------------------+--------+--------+
   |  ...   | footer | header |   payload ...     | header |  ...   |
   +--------+--------+--------+-------------------+--------+--------+
            ^                                   ^
      prev_footer = hdr - 8                读 next_hdr 的 size 得下一块大小
      -> 读出 prev_size -> prev_hdr = hdr - prev_size
```

1. **找后块**：`next_hdr = hdr + asize`（当前块大小从自己的 header 读）。
2. **找前块**：`prev_size = *(word_t *)(hdr - 8)`（前块的 **footer** 就是当前块 header 前 8 字节），于是 `prev_hdr = hdr - prev_size`。
3. 前后块的大小、分配状态全部**O(1)** 拿到，不需遍历链表——这是"边界标记 + 立即合并"能做到常数时间释放的全部秘密。
4. 讲义另加两个哨兵：**堆起始处的 dummy footer**（标记为已分配，防止释放首块时误合并）与**堆结束前的 dummy header**（防止释放末块时越界合并）。

**堆布局示例（隐式空闲链表，标注"大小/分配位"，单位 = word）**：

```
  heap_start                                                   heap_end
  +--------+--------+--------+--------+--------+--------+-------------+
  | 16/0   | 32/1   | 32/1   | 64/0   |  ...   |  8/1   | dummy hdr   |
  | free   | alloc  | alloc  | free   |        | epilog | (allocated) |
  +--------+--------+--------+--------+--------+--------+-------------+
    16     32      32      64      ...      8       8  (bytes)
```

*   **header 落在非对齐位置，payload 才是对齐的**——这正是"低 3 位可当标志"的物理来源。

#### A.9.5 四种分配器对比

| 分配器 | 分配时间 | 释放时间 | 合并时间 | 额外内存开销 | 备注 |
|---|---|---|---|---|---|
| **隐式空闲链表**（implicit free list） | $O(n)$，$n$ = **全部**块数（线性最坏） | $O(1)$（**含**合并，靠边界标记） | $O(1)$ | 1 word/块（+1 word footer 若需反向合并） | 实现最简单；因线性分配**实践中不用于 `malloc`/`free`**；"分割 + 边界标记合并"是通用技术 |
| **显式空闲链表**（explicit free list） | $O(n_{\\text{free}})$，线性于**空闲块**数 | $O(1)$（LIFO/FIFO 插入）或 $O(n_{\\text{free}})$（地址序插入） | $O(1)$ | **2 word/块**（`next`/`prev` 指针）+ header/footer | 内存越满越快；分配/释放需在链表中 splice 摘挂；指针只能存在**空闲块的 payload 区** |
| **分离适配**（segregated free list） | 先查对应大小类，命中则 $O(1)$；否则逐类扩大搜索，约 $O(\\log)$（2 的幂类） | $O(1)$（合并后插入对应类） | $O(1)$ | 指针 + **每个大小类一个链表头**的数组 | 首次适配某个类 ≈ 全局**最佳适配**的近似（极端情形：每块一个类 = 精确最佳适配）；吞吐率与利用率双赢 |
| **简单分离存储**（simple segregated storage） | $O(1)$（从非空类直接取**整块**，**不分割**） | $O(1)$（整块插回原类） | **不做合并** | 仅每类一个链表头；块内**无 header/footer** | 时间与开销最优，但**内部碎片严重**、外部碎片不存在；只适合"块大小单一化"的场景 |

- 显式链表的**插入策略**：`LIFO`（插到表头）与 `FIFO`（插到表尾）都简单且常数时间，但研究显示**碎片比地址序更差**；**地址序**（保持 `addr(prev) < addr(curr) < addr(next)`）碎片更低，代价是插入需**搜索**。
- 分离适配的分配流程：① 在合适类中做首次适配，找到就**分割**并把余料放入合适类；② 该类找不到就**试下一个更大的类**，直到找到；③ 全都找不到才向 OS 申请新堆内存（`sbrk`），从新内存切出 `n` 字节，余料作为一个空闲块放进合适类。

#### A.9.6 合并的四种情况（立即合并，含边界标记）

设被释放块大小为 $n$，前块大小 $m_1$、后块大小 $m_2$：

| 情况 | 前块状态 | 后块状态 | 操作 | 新块大小 | header 低 2 位处理 |
|---|---|---|---|---|---|
| **Case 1** | 已分配 | 已分配 | 仅把当前块标记为空闲，不合并 | $n$ | 当前块 `A=0`；后块 `P=0` |
| **Case 2** | 已分配 | **空闲** | 与**后块**合并 | $n + m_2$ | 写新 header/footer；**再后一块的 `P` 位清 0** |
| **Case 3** | **空闲** | 已分配 | 与**前块**合并 | $m_1 + n$ | 改写前块 header/footer；后块 `P` 保持 0 |
| **Case 4** | **空闲** | **空闲** | **三块**全部合并 | $m_1 + n + m_2$ | 改写前块 header/footer；再后一块 `P` 清 0 |

- 讲义记法：Case 2 为 `n+m2`，Case 3 为 `m1+n`，Case 4 为 `m1+m2+n`；四种情况**都只需常数时间**。
- 讲义给出的位变化记法：`?0`（前块状态未知/空闲）、`n 01`（本块空闲且前块已分配）、`m2 10`（后块空闲且前块已分配）等，本质就是维护 `P<<1 \| A`。
- **合并策略**：**立即合并**（每次 `free` 就合并，简单、碎片少）vs **延迟合并**（攒着直到 `malloc` 找不到合适块时才合并，提升 `free` 吞吐率）。

#### A.9.7 适配策略对比

| 策略 | 搜索方式 | 复杂度 | 碎片倾向（讲义 benchmark 总开销） | 备注 |
|---|---|---|---|---|
| **首次适配**（first fit） | 从表头开始，选**第一个够大**的块 | $O(n)$ | 11.9% | 实现最简单；会在表头附近产生"**splinters**"（小残渣） |
| **下次适配**（next fit） | 像首次适配，但**从上一次搜索结束处**继续 | $O(n)$ 但常更快 | **21.6%（最差）** | 避免反复扫描无用的头部块；但研究显示**碎片比首次适配更差** |
| **最佳适配**（best fit） | **全表扫描**，选剩余空隙**最小**的块 | $O(n)$，常数更大 | **8.3%（最好）** | 保持碎片小、利用率高；通常比首次适配**更慢**；仍是**贪心**，**不保证最优** |
| （参照）完美适配 | 不存在的理想分配器 | — | 1.6% | 只作基准线 |

- **记忆口诀**：**首次快、下次更差、最佳更省**。分离适配的"类内首次适配"可以**同时**拿到接近最佳适配的利用率与常数级吞吐率，这是它成为工业实现主流的原因。
- 三类策略选择之外还有两个正交策略：**分割策略**（何时把空闲块切开——直接决定能容忍多少内部碎片）与**合并策略**（立即/延迟）。

#### A.9.8 分离适配的大小类划分

**2 的幂划分示例**（类 $k$ 覆盖 $[2^{k-2}+1,\\ 2^{k-1}]$，类内统一按上界取整）：

| 类号 | 覆盖范围（字节） | 类内块大小 | 最大内部碎片 |
|---|---|---|---|
| 1 | $\\{1\\}$ | 1 | 0 |
| 2 | $\\{2\\}$ | 2 | 0 |
| 3 | $\\{3\\text{–}4\\}$ | 4 | 1 |
| 4 | $\\{5\\text{–}8\\}$ | 8 | 3 |
| 5 | $\\{9\\text{–}16\\}$ | 16 | 7 |
| … | … | … | … |
| 11 | $\\{1025\\text{–}2048\\}$ | 2048 | 1023 |
| 12 | $\\{2049\\text{–}4096\\}$ | 4096 | 2047 |
| 13 | $\\{4097\\text{–}\\infty\\}$ | **无上限**（实际受 $2^{64}$ 约束） | — |

- 讲义的两条常见选择：① **小尺寸一档一块**（16、32、48、64…，类内无内部碎片）；② 到某个点后**切换到 2 的幂** $[2^{i}+1,\\ 2^{i+1}]$。
- **最大块那一类必须没有上界**，否则超大请求无处安放。
- 大小类如何划分"对利用率和吞吐率都有重大影响"，是**设计决策**而非定式。

#### A.9.9 `realloc` 的实现路径

| 路径 | 触发条件 | 操作 | 代价 |
|---|---|---|---|
| **① 原地扩展** | 当前块后面的邻居**空闲**且"当前块 + 后块"足够大（可选再合并更远的空闲块） | 吞并后块，改 header/footer 的 size，**返回原指针** | $O(1)$，**没有数据搬运**——最优路径 |
| **② 搬移拷贝** | 后面不够，或需要缩小但选择重排 | `newp = mm_malloc(size)` → `memcpy(newp, oldp, min(oldsize, size))` → `mm_free(oldp)` → 返回 `newp` | $O(\\text{size})$ 拷贝 + 分配/释放开销 |
| **③ 原地缩小** | 新尺寸**小于**当前块，且切出来的余料 ≥ 最小块 | 原地改 size，把余料作为空闲块插入相应大小类（**可立即合并**） | $O(1)$，**指针不变** |

- Malloc Lab 的关键提分点：先能用 `mm_malloc`+`mm_free` **搭出** `realloc`（保证正确性），**再**改写成独立的 `realloc`（原地扩展 + 原地缩小），因为最后 2 个 trace 全是 `realloc`，吞吐率差距明显。
- 语义红线：`realloc` 失败时**必须**返回 `NULL` 且**不动原块**；返回的新块同样要满足**8 字节对齐**。

#### A.9.10 glibc `malloc` 要点

> 以下均为**补充说明**（F25 讲义未涉及 glibc 实现细节），用于对照 Malloc Lab 与真实调试。

| 主题 | 要点 |
|---|---|
| **双路径** | 小请求走**主堆（main arena）**，用 `brk`/`sbrk` 推高堆顶；大请求走 **`mmap`**（`MAP_PRIVATE \| MAP_ANONYMOUS`）建立独立映射，`free` 时用 `munmap` 直接还给内核 |
| **`MMAP_THRESHOLD`** | 默认 **128 KB = 131072 字节**：单次请求 **≥ 该值**时走 `mmap`。glibc 还会在释放 mmap 块时**动态上调**该阈值（64 位上限 `DEFAULT_MMAP_THRESHOLD_MAX = 32 MB`） |
| **`M_TRIM_THRESHOLD`** | 默认 **128 KB**：堆顶（top chunk）空闲量超过它时用 `brk` **收缩堆**、把内存还给 OS |
| **`mallopt(param, value)`** | 运行时调参。常用参数：`M_TRIM_THRESHOLD`、`M_MMAP_THRESHOLD`、`M_MMAP_MAX`、`M_TOP_PAD`（每次 `sbrk` 多要的余量）、`M_ARENA_MAX`（多线程 arena 数上限） |
| **`mallinfo2()`** | 返回 `struct mallinfo2`，字段：`arena`（主堆总字节）、`ordblks`（空闲块数）、`hblks`/`hblkhd`（mmap 块数与字节）、`uordblks`（**已分配**字节）、`fordblks`（**空闲**字节）、`keepcost`（可 trim 的堆顶空闲）。**旧版 `mallinfo()` 用 `int`，已废弃** |
| **`malloc_stats()`** | 直接把上述统计打到 **`stderr`**（无需返回值） |
| **`malloc_usable_size(p)`** | 返回 `p` 实际可安全使用的字节数，**≥ 请求值**（原因：向上取整到 16 字节、且"下一块的 `prev_size` 字段"可被本块复用）——这正是内部碎片的一部分 |
| **`malloc_trim(0)`** | 主动触发一次堆顶归还，常用于"长期运行进程做完一批大任务后释放内存" |

- **arena 与竞争**：多线程下 glibc 会为每个线程分配独立 **arena**，减少锁竞争；这也是为什么"多线程 malloc"看起来并没有一个全局锁那么慢。
- **`tcache`**：glibc 2.26+ 每个线程有一组 **tcache**（每类最多 7 个块）的快速缓存，`malloc`/`free` 的**绝大多数**操作在此完成、**几乎无锁**——这解释了下面那些错误信息里的 `in tcache 2`。

#### A.9.11 glibc 常见错误信息含义

> **补充说明**（讲义未涉及；文本随 glibc 版本略有差异）。这些都是**堆元数据被破坏**的信号，不是"分配器不可靠"。

| 错误信息 | 直接含义 | 典型根因 | 定位手段 |
|---|---|---|---|
| `free(): double free detected in tcache 2` | 同一指针被 `free` **两次**（tcache 层检测到该块已在缓存中） | 双重释放；两条释放路径（如错误处理与正常路径）都 `free` 了同一块；或指针被复制后各自 `free` | `valgrind`（报 `Invalid free()` / `double free`）、或用 `MALLOC_CHECK_=3` 使程序立即 abort 并保留 core |
| `free(): double free detected in fasttop` | 同一指针**连续两次**释放（fastbin 的 `fasttop` 检测） | 同上，且两次 `free` 紧邻 | 同上 |
| `malloc(): corrupted top size` | **top chunk（堆顶块）的 size 字段被改写** | **写越界**：往某块 payload 里写了超过块大小的字节，覆盖了**下一块的 header**；把 top chunk 的 size 写成非法值 | `valgrind --leak-check=full`、`gdb` + 在 `malloc` 上设断点看 `top` 与实际请求大小；`x/16gx <p-16>` 观察相邻 header |
| `munmap_chunk(): invalid pointer` | 该指针被判定为 **mmap 块**，但**不是 mmap 块的首地址**（或压根不是 mmap 块） | `free` 了栈/全局数组地址、`free` 了 `p + k`（指向块中间）、或 `free` 了已经交给别处的指针 | `valgrind`、检查所有 `free` 的参数是否都来自 `malloc` 的**原始返回值** |
| `free(): invalid pointer` | `free` 的地址**不是**合法块首（未对齐、或不属于堆） | `free` 栈地址 / 全局数组地址 / 结构体字段地址 | `valgrind`、`gdb` 打在 `free` 上看调用栈 |
| `free(): invalid next size (normal/fast)` | 相邻块的 `size` 字段被破坏 | 越界写、`memcpy` 长度算错（经典的 `strlen` 忘加 `'\0'`） | `valgrind`、仔细核对每个 `memcpy`/`strcpy` 的长度 |
| `corrupted double-linked list` | unsorted/small bin 的**双向链表指针**被破坏 | 写越界覆盖了某个**空闲块**的 `fd`/`bk` 指针 | `valgrind`、heap checker |

- **最实用的一招**：编译时不要"绕过"分配器检查——`MALLOC_CHECK_=3 ./prog`、`MALLOC_PERTURB_=165 ./prog`（每次分配填垃圾值，暴露"读了未初始化内存"）、以及 `valgrind`。Malloc Lab 里则完全靠**自己写的 `mm_check` 堆一致性检查器**。

#### A.9.12 Malloc Lab 提分点清单与 `mdriver` 用法

**评分公式**（官方 writeup：Correctness 20 + Performance 35 + Style 10）：

$$P = wU + (1-w)\min\!\left(1,\ \frac{T}{T_{\text{libc}}}\right),\qquad w = 0.6,\quad T_{\text{libc}} = 600\ \text{Kops/s}$$

| 项 | 权重 | 关键点 |
|---|---|---|
| Correctness | 20 分 | 逐 trace 给分；**崩溃或违规得 0** |
| Space utilization $U$ | 占 $w=0.6$ | 峰值"驱动器已分配字节 / 堆大小"，最优为 1；**利用率权重高于吞吐率** |
| Throughput $T$ | 占 $1-w=0.4$ | 平均每秒完成的操作数；超过 libc 后**封顶**（`min(1, T/T_libc)`），所以**不要**为吞吐率牺牲利用率 |
| Style | 10 分 | 5 分给**堆一致性检查器 `mm_check`**，5 分给**函数分解与注释**（顶部必须写明块结构、空闲链表组织、链表操作方式） |

**评分规则红线**：

*   不得修改 `mm.c` 里的四个接口签名（`mm_init`/`mm_malloc`/`mm_free`/`mm_realloc`）。
*   不得调用任何内存管理相关库函数/系统调用：**禁止 `malloc`、`calloc`、`free`、`realloc`、`sbrk`、`brk`** 及其变体。
*   不得定义任何**全局或静态复合数据结构**（数组、结构体、树、链表）；只允许全局**标量**（整数、浮点、指针）。空闲链表必须**构造在堆内存里**。
*   `mm_init` 出错返回 `-1`，否则 `0`；`mm_malloc` 必须返回 **8 字节对齐**的指针（`mdriver` 强制检查）。

**提分路线（按投入产出排序）**：

| 阶段 | 做法 | 收益 |
|---|---|---|
| 1 | **隐式空闲链表 + 边界标记 + 立即合并 + 首次适配** | 正确性达标，利用率一般，吞吐率差 |
| 2 | 改为 **显式空闲链表**（`next`/`prev` 放空闲块 payload 里），插入用 LIFO 或地址序 | 吞吐率大幅提升（只遍历空闲块） |
| 3 | 改为 **分离空闲链表**（2 的幂大小类 + 类内首次适配），升级为常数时间 | 利用率提升到接近最佳适配，吞吐率再上一档 |
| 4 | **独立 `mm_realloc`**：原地扩展 + 原地缩小 + 只在必要时搬移 | 最后 2 个（`realloc`）trace 的吞吐率 |
| 5 | 细节优化：**合并时用小块优先/`prev` 位判断**、分割阈值、类内按大小**有序插入**、`mm_check` 写扎实 | 利用率尾数 + Style 10 分 |
| 6 | **主动降低每块开销**：16 字节对齐可把低 4 位当标志；小块省掉 footer（只给空闲块加 footer） | 直接换利用率 |

**`mdriver` 命令行**（`-h` 可查全部）：

| 选项 | 作用 |
|---|---|
| `./mdriver -V` | **最重要**：最详细输出，处理每个 trace 文件时打印诊断信息，便于定位**哪个 trace 出错** |
| `./mdriver -v` | 每个 trace 打印一行紧凑的性能明细表（利用率、吞吐率） |
| `./mdriver -f <tracefile>` | 只跑**指定单个 trace**（开发期用 `short1-bal.rep`、`short2-bal.rep` 这类微型 trace 极省时间） |
| `./mdriver -t <tracedir>` | 到指定目录找 trace 文件（默认目录在 `config.h` 里） |
| `./mdriver -l` | **额外**测量并打印 libc `malloc` 的性能，作为对照 |
| `./mdriver -h` | 打印所有命令行参数 |
| `gprof ./mdriver` | 官方提示可用它定位热点函数 |

- **官方 hint 摘要**：先用 `-f` 跑小 trace 建立正确性；**前 9 个 trace 只含 `malloc`/`free`，最后 2 个含 `realloc`**——先把前者做对再做 `realloc`；**用宏封装指针运算**（分配器里的指针算术充满强制类型转换，是 bug 温床）；**先完全读懂教材的隐式链表例子再动手**；用 `gcc -g` + 调试器定位越界引用。

---

### A.10 进程与信号速查

> 对应讲次：Lecture 16（`F25-16-processes.txt`）、Lecture 17（`F25-17-ecf.txt`）；
> 教材 CS:APP3e 第 8 章（8.1–8.8）；关联 **L6 Shell Lab**。

#### A.10.1 异常四分类

| 类别 | 原因 | 同步/异步 | 返回行为 | 典型例子 |
|---|---|---|---|---|
| **中断**（Interrupt） | 来自**处理器外部**的事件，通过置位**中断引脚**通知 CPU | **异步** | 返回到 **$I_{\\text{next}}$**（下一条指令） | 定时器中断（每几毫秒一次）、I/O 中断、Ctrl-C 引起的信号、网络包到达 |
| **陷阱**（Trap） | **有意为之**的指令（程序主动陷入内核） | **同步** | 返回到 **$I_{\\text{next}}$** | 系统调用（`syscall`/`int 0x80`）、`gdb` 断点 |
| **故障**（Fault） | 指令执行引起的**意外**，可能可恢复 | **同步** | **重新执行 $I_{\\text{current}}$**，或无法恢复时终止进程 | **缺页**（page fault，可恢复）、保护故障、浮点异常、除法错误 |
| **终止**（Abort） | 意外且**不可恢复**的致命错误 | **同步** | **不返回**，直接终止当前程序 | 非法指令、奇偶校验错、机器检查（machine check） |

- **机械记忆**：**中断异步并跳过、陷阱同步并跳过、故障同步并重来、终止同步并结束**。
- **异常号（讲义 Figure 8.9 口径）**：0 = 除法错误，13 = 一般保护故障，14 = 缺页，18 = 机器检查；32–255 由操作系统定义（Linux 用 128 = `0x80` 作系统调用入口）。
- **故障两种结局的典范**：缺页 → 内核调页 → **重新执行同一条 `movl`**；非法地址 → 无法恢复 → 内核向进程发 `SIGSEGV` → "segmentation fault" 退出。**同一机制，两种归宿。**
- **异常表 / 中断向量**：内核启动时把每个异常号对应的处理程序地址填进**异常表（exception table，又称中断向量 interrupt vector）**，硬件按异常号查表跳转。

#### A.10.2 进程控制接口

| 接口 | 原型 | 语义要点 |
|---|---|---|
| `fork` | `pid_t fork(void)` | **调用一次、返回两次**：父进程得到**子进程 PID**（$>0$），子进程得到 **0**；出错返回 $-1$（并置 `errno`）。子进程获得父进程地址空间的**副本**（实现上用**写时复制 COW**）、文件描述符表的**副本**（指向**同一**打开文件表项 → **共享文件偏移量**）、但**独立的 PID**。父子**并发**执行，输出顺序不确定 |
| `exit` | `void exit(int status)` | 终止进程；`status` 的低 8 位可由父进程用 `WEXITSTATUS` 读出。会执行 `atexit` 注册的函数与 stdio 清理（**非异步信号安全**） |
| `_exit` | `void _exit(int status)` | 立即终止，**跳过**清理——**唯一**信号处理程序中可用的终止方式（异步信号安全） |
| `wait` | `pid_t wait(int *statusp)` | 等价于 `waitpid(-1, statusp, 0)`；**已废弃**，新代码用 `waitpid` |
| `waitpid` | `pid_t waitpid(pid_t pid, int *statusp, int options)` | 回收子进程并取回状态；支持非阻塞与作业控制（详见 A.10.3） |
| `execve` | `int execve(const char *filename, char *const argv[], char *const envp[])` | **加载并运行**新程序：覆盖当前进程的代码/数据/栈，**PID 不变**，**保留打开的文件描述符**（除非标了 `O_CLOEXEC`）。**成功不返回**；失败返回 $-1$ 并置 `errno`——**只有失败路径会执行 `execve` 之后的代码** |
| `getpid` | `pid_t getpid(void)` | 返回调用进程的 PID |
| `getppid` | `pid_t getppid(void)` | 返回调用进程**父进程**的 PID |
| `sleep` | `unsigned int sleep(unsigned int secs)` | 挂起 `secs` 秒；**返回尚未睡满的剩余秒数**（被信号打断时非 0） |
| `pause` | `int pause(void)` | 挂起直到收到信号；**总是**返回 $-1$ 且 `errno = EINTR` |

- **`fork` 的经典坑**：`fork` 后父子共享终端输出，若在 `fork` **之前**有未 `fflush` 的 stdio 缓冲，父子各会输出一份 → 同一行打印两遍。**修复**：`fork` 前 `fflush(stdout)`，或改用 `write`。
- **僵尸进程**（zombie）：子进程已终止但父进程尚未 `wait`，内核保留其退出状态 → 父进程必须回收。父进程先死则子进程被 `init`/`pid 1` 收养。

#### A.10.3 `waitpid` 选项与 `status` 宏完整表

```c
#include <sys/wait.h>
pid_t waitpid(pid_t pid, int *statusp, int options);
```

**`pid` 参数取值**：

| `pid` | 含义 |
|---|---|
| `pid > 0` | 只等待**进程 ID 为 `pid`** 的那**一个**子进程 |
| `pid == -1` | 等待**任意一个**子进程（最常用） |
| `pid == 0` | 等待与调用者**同一进程组**中的任意子进程 |
| `pid < -1` | 等待**进程组 `\|pid\|`** 中的任意子进程（作业控制里配合 `setpgid` 使用） |

**`options` 参数**（可用 `\|` 组合）：

| 选项 | 含义 |
|---|---|
| `0` | 默认：**阻塞**，直到 `pid` 指定的某个子进程**终止** |
| `WNOHANG` | **非阻塞**：若无已终止的子进程可回收，**立即返回 0**（返回 `0` 不是错误） |
| `WUNTRACED` | 除终止外，还报告**已停止（stopped）**的子进程（`SIGTSTP`/`SIGSTOP` 之后） |
| `WCONTINUED` | 还报告收到 `SIGCONT` **恢复执行**（continued）的子进程 |

**`statusp` 判定宏**（**必须先判类别、再取具体值**，顺序不可颠倒）：

| 宏 | 判定/取值 | 返回值语义 |
|---|---|---|
| `WIFEXITED(status)` | **是否正常终止** | 非 0 表示通过 `exit`/`return` 正常结束 |
| `WEXITSTATUS(status)` | 正常终止时的**退出状态** | `exit` 参数的**低 8 位**；**仅在** `WIFEXITED` 为真时有效 |
| `WIFSIGNALED(status)` | **是否因未捕获的信号而终止** | 非 0 表示为信号所杀 |
| `WTERMSIG(status)` | **导致终止的信号编号** | 如 `2`（`SIGINT`）、`9`（`SIGKILL`）、`11`（`SIGSEGV`）；仅在 `WIFSIGNALED` 为真时有效 |
| `WIFSTOPPED(status)` | **是否处于停止状态** | 非 0 表示已停止（需 `WUNTRACED` 才会被报告） |
| `WSTOPSIG(status)` | **导致停止的信号编号** | 如 `20`（`SIGTSTP`）、`19`（`SIGSTOP`）；仅在 `WIFSTOPPED` 为真时有效 |
| `WIFCONTINUED(status)` | **是否已由 `SIGCONT` 恢复** | 需 `WCONTINUED` 才会被报告 |

- **返回值**：成功返回被回收子进程的 **PID**；`WNOHANG` 且无子进程可回收时返回 **0**；出错返回 **$-1$**（常见 `errno = ECHILD`，表示没有可等待的子进程——SIGCHLD 处理程序里用 `while ((pid = waitpid(-1, &st, WNOHANG\|WUNTRACED)) > 0)`，靠返回 `-1`/`0` 退出循环）。
- **Shell Lab 的官方推荐用法**：`waitfg` 里用围绕 `sleep` 的**忙循环**，`sigchld` 处理程序里**恰好调用一次** `waitpid`；不要在 `waitfg` 里也调 `waitpid`（"虽然可行，但极易混乱"）。

#### A.10.4 `execve` 家族表

| 函数 | 参数形式 | 搜索 `PATH` | 环境变量 | 备注 |
|---|---|---|---|---|
| `execl` | **列表**：`execl(path, arg0, ..., NULL)` | 否 | 继承 `environ` | `l` = list |
| `execv` | **数组**：`execv(path, argv[])` | 否 | 继承 `environ` | `v` = vector |
| `execlp` | 列表 | **是** | 继承 `environ` | `p` = 搜索 `PATH` |
| **`execvp`** | 数组：`execvp(file, argv[])` | **是** | 继承 `environ` | 最常用：`execvp(argv[0], argv)` |
| `execle` | 列表：`execle(path, arg0, ..., NULL, envp[])` | 否 | **显式 `envp`** | `e` = environment |
| **`execve`** | 数组：`execve(path, argv[], envp[])` | 否 | **显式 `envp`** | **唯一的系统调用**，其余都是 libc 包装 |

- **`execvp` 如何搜索 `PATH`**：若 `file` **不含 `/`**，则按 `PATH` 环境变量（以 `:` 分隔的目录列表）**依次**在每个目录下查找可执行文件；找到即执行，全都找不到才返回 $-1$（`errno = ENOENT`）。若 `file` **含 `/`**，则当作**路径**直接使用，**不搜索 `PATH`**（这正是 Shell Lab 里 `./myspin` 与 `myspin` 行为不同的原因）。
- **`execve` 与 `argv` 约定**：`argv[0]` 按惯例是可执行文件名（**可任意**，程序无法强制校验），`argv` 必须以 `NULL` 结尾。
- **`execve` 之后的代码只在失败时执行**——Shell 中的标准写法是 `execve(...); fprintf(stderr, "%s: Command not found.\n", argv[0]); exit(0);`。

#### A.10.5 进程组与作业控制

| 接口/概念 | 说明 |
|---|---|
| `getpgrp()` | 返回**当前进程所属进程组**的 ID（PGID） |
| `setpgid(pid, pgid)` | 改变进程的进程组。**`setpgid(0, 0)`** 表示"把**我自己**放进一个**以我的 PID 为组 ID** 的新进程组" |
| **进程组**（process group） | 一组进程的集合，**组 ID = 组长进程的 PID**；同一组的进程可用 **一个信号一起通知** |
| **前台进程组** | 当前拥有终端的进程组——键盘信号只发给它 |
| **后台进程组** | 不拥有终端；读终端会收到 `SIGTTIN`（默认停止），写终端可能收到 `SIGTTOU` |
| `Ctrl-C` | 内核向**前台进程组中的每一个进程**发送 **`SIGINT`(2)**，默认动作 = **终止** |
| `Ctrl-Z` | 内核向**前台进程组中的每一个进程**发送 **`SIGTSTP`(20)**，默认动作 = **停止（挂起）**，直到收到 `SIGCONT` 才继续 |
| `/bin/kill` 程序 | `kill -9 24818` 发给单个进程；**`kill -9 -24817`** 发给**进程组 24817 的所有进程**（注意负号） |
| Shell 的 `fg` / `bg` | `bg` 给作业发 `SIGCONT` 并让它在**后台**运行；`fg` 给作业发 `SIGCONT`（若已停止）并把它放到**前台**，等待其终止或停止 |

**`kill(pid, sig)` 的 `pid` 取值含义**：

| `pid` 取值 | 信号发往 |
|---|---|
| `pid > 0` | 只发给**进程 `pid`** |
| `pid == 0` | 发给**调用者所在进程组**的每个进程（**含调用者自己**） |
| `pid < 0` | 发给**进程组 `\|pid\|`** 中的每个进程 |
| `pid == -1` | 发给调用者**有权发送信号的所有进程**（危险） |

**Shell Lab 的作业控制要点**：

*   子进程在 `fork` 之后、`execve` 之前必须调用 **`setpgid(0, 0)`**，否则子进程默认留在 Shell 的进程组里——于是 `Ctrl-C` 会**同时**打到 Shell 自己和它创建的所有进程，这显然是错的。
*   Shell 捕获 `SIGINT`/`SIGTSTP` 后，必须用 **`kill(-pid, SIGINT)`**（**负的 `pid`**）转发给**整个前台进程组**；`sdriver.pl` 会专门测试你有没有用 `-pid`。
*   **必须**在 `fork` 前用 `sigprocmask` 屏蔽 `SIGCHLD`，在 `addjob` 之后解除屏蔽，否则会出现"子进程已被 `sigchld` 处理程序回收并从作业表删除，父进程却还没 `addjob`"的**竞态**。由于子进程继承父进程的屏蔽字，**子进程在 `execve` 前必须解除 `SIGCHLD` 屏蔽**。

#### A.10.6 常用信号表（编号 / 默认动作 / 可否捕获或阻塞）

| ID | 名称 | 默认动作 | 对应事件 | 可否捕获/忽略/阻塞 |
|---|---|---|---|---|
| 1 | `SIGHUP` | Terminate | 终端挂断 | 可 |
| **2** | **`SIGINT`** | Terminate | 用户按 `Ctrl-C` | 可 |
| **3** | **`SIGQUIT`** | Terminate + core | 用户按 `Ctrl-\` | 可 |
| **4** | **`SIGILL`** | Terminate + core | 非法指令 | 可* |
| 5 | `SIGTRAP` | Terminate + core | 断点 / 陷阱 | 可* |
| **6** | **`SIGABRT`** | Terminate + core | `abort()` | 可 |
| 7 | `SIGBUS` | Terminate + core | 总线错误（未对齐/映射错） | 可* |
| **8** | **`SIGFPE`** | Terminate + core | 除零 / 算术错误 | 可* |
| **9** | **`SIGKILL`** | Terminate | 强杀（管理员最后手段） | **不可**（`sigaction` 返回 `EINVAL`） |
| 10 | `SIGUSR1` | Terminate | 用户自定义 | 可 |
| **11** | **`SIGSEGV`** | Terminate + core | 段错误（非法内存访问） | 可* |
| 12 | `SIGUSR2` | Terminate | 用户自定义 | 可 |
| **13** | **`SIGPIPE`** | Terminate | 向**已关闭的管道/socket 写入** | 可 |
| **14** | **`SIGALRM`** | Terminate | `alarm()` 定时器到期 | 可 |
| **15** | **`SIGTERM`** | Terminate | 温和的终止请求（`kill` 默认） | 可 |
| 16 | `SIGSTKFLT` | Terminate | 协处理器栈错误 | 可 |
| **17** | **`SIGCHLD`** | **Ignore** | 子进程**停止或终止** | 可 |
| **18** | **`SIGCONT`** | Continue | 恢复被停止的进程 | 可 |
| **19** | **`SIGSTOP`** | Stop | 强制停止 | **不可**（`EINVAL`） |
| **20** | **`SIGTSTP`** | Stop | 用户按 `Ctrl-Z` | 可 |
| **21** | **`SIGTTIN`** | Stop | **后台**进程读终端 | 可 |
| **22** | **`SIGTTOU`** | Stop | **后台**进程写终端 | 可 |
| 23 | `SIGURG` | Ignore | socket 上有紧急数据 | 可 |
| 24 | `SIGXCPU` | Terminate + core | CPU 时间超限 | 可 |
| 25 | `SIGXFSZ` | Terminate + core | 文件大小超限 | 可 |
| 26 | `SIGVTALRM` | Terminate | 虚拟定时器 | 可 |
| 27 | `SIGPROF` | Terminate | 性能剖析定时器 | 可 |
| 28 | `SIGWINCH` | Ignore | 终端窗口大小改变 | 可 |
| 29 | `SIGIO` | Terminate | 异步 I/O 就绪 | 可 |
| 30 | `SIGPWR` | Terminate | 电源故障 | 可 |

\* 带 `*` 的信号**只能被"其他进程发来的"实例阻塞**；由当前指令直接触发的同步异常（如自己产生 `SIGSEGV`/`SIGILL`）无法靠屏蔽躲开。

- 信号类型用**小整数 ID（1–30）**标识；信号里**唯一的载体就是 ID** 和"它到了"这个事实，**不携带数据**。
- **`SIGKILL`(9) 与 `SIGSTOP`(19) 是唯二**不可捕获、不可忽略、不可阻塞的信号。

#### A.10.7 `sigaction` 结构与 `sa_flags`

```c
#include <signal.h>
int sigaction(int signum, struct sigaction *act, struct sigaction *oldact);
/* 结构字段：sa_handler / sa_sigaction、sa_mask、sa_flags */
```

| 字段 | 作用 |
|---|---|
| `sa_handler` | 单参数处理程序 `void (*)(int)`；或 `SIG_IGN`（忽略）、`SIG_DFL`（默认） |
| `sa_sigaction` | 三参数处理程序 `void (*)(int, siginfo_t *, void *)`，当 `sa_flags & SA_SIGINFO` 时使用（与 `sa_handler` 是**联合体**，只能设一个） |
| `sa_mask` | 处理程序**运行期间额外屏蔽**的信号集合（**隐式**地，正在处理的信号自身总会被屏蔽） |
| `sa_flags` | 行为开关，见下表 |

| `sa_flags` 标志 | 含义 |
|---|---|
| `SA_RESTART` | 自动**重启**被该信号打断的**慢速系统调用**（`read`、`wait` 等）——不设它就要自己处理 `EINTR` |
| `SA_NODEFER` | 处理程序运行期间**不**自动屏蔽自身（默认是屏蔽的） |
| `SA_RESETHAND` | 进入处理程序时把该信号的动作**重置为 `SIG_DFL`**（只生效一次） |
| `SA_SIGINFO` | 改用三参数 `sa_sigaction`，可取得 `siginfo_t` 与被打断的上下文 |
| `SA_ONSTACK` | 在备用信号栈（`sigaltstack`）上运行处理程序 |

- **`signal()` 的教训**：老接口 `signal()` 语义在不同系统上不一致（是否重启系统调用、是否重置），**Posix 标准推荐一律用 `sigaction`**。
- **隐式屏蔽**：内核在进入处理程序前，自动把"正在处理的信号类型"加入屏蔽集——所以 `SIGINT` 处理程序**不会被另一个 `SIGINT` 打断**（这正是"同种信号不会嵌套"的机制来源）。

#### A.10.8 阻塞与待处理信号

| 概念/接口 | 说明 |
|---|---|
| **待处理**（pending） | 信号**已发送但尚未被接收**。内核为每个进程维护 **pending 位向量**：发送时置位 $k$，接收时清位 $k$ |
| **阻塞**（blocked） | 一个进程可以**阻塞**某些信号的接收。被阻塞的信号**可以发出**，但会**一直待处理**，直到解除阻塞才被接收 |
| **信号掩码**（signal mask） | 被阻塞信号的集合；用 `sigprocmask` 读写 |
| **关键事实：信号不排队** | 每种信号**最多只有一个待处理信号**。若进程已有类型 $k$ 的待处理信号，则**后续同类型信号直接被丢弃**。故"待处理信号最多被接收一次"——**不要用信号来计数事件**（Shell Lab 里数"有几个子进程结束"必须用 `waitpid` 循环，而不是数信号次数） |
| `sigprocmask(how, set, oldset)` | 读/改当前信号掩码。`how`：`SIG_BLOCK`（加入）、`SIG_UNBLOCK`（移除）、`SIG_SETMASK`（替换） |
| `sigemptyset(&set)` | 把集合清空 |
| `sigfillset(&set)` | 把**所有**信号加入集合 |
| `sigaddset(&set, sig)` | 加入指定信号 |
| `sigdelset(&set, sig)` | 删除指定信号 |
| `sigismember(&set, sig)` | 判断某信号是否在集合中 |
| `sigpending(&set)` | 取回**当前待处理**（且被阻塞）的信号集合 |

**暂时阻塞信号的惯用法**（保护不可被打断的代码区）：

```c
sigset_t mask, prev_mask;
sigemptyset(&mask);
sigaddset(&mask, SIGINT);
sigprocmask(SIG_BLOCK, &mask, &prev_mask);   /* 阻塞 SIGINT，并保存原掩码 */
    /* 这段区域不会被 SIGINT 打断 */
sigprocmask(SIG_SETMASK, &prev_mask, NULL);  /* 恢复原掩码 = 解除阻塞 */
```

- **恢复必须用 `SIG_SETMASK` + 保存的旧掩码**，而不是 `SIG_UNBLOCK`——否则会把调用者原本就屏蔽的信号**误解除**。

#### A.10.9 异步信号安全（async-signal-safe）函数

**定义**：函数是**异步信号安全**的，当且仅当它**要么可重入**（所有变量都在栈帧上），**要么不可被信号打断**。POSIX 保证 **117 个**函数满足该性质（`man 7 signal-safety`）。

| ✅ 安全（可在处理程序中调用） | ❌ 不安全（**严禁**在处理程序中调用） |
|---|---|
| `write`（**唯一安全的输出函数**） | `printf` / `fprintf`（stdio 缓冲区非可重入） |
| `_exit` / `_Exit` | `exit`（会跑 `atexit` 与 stdio 清理） |
| `read`、`open`、`close`、`dup2`、`lseek`、`fcntl` | `malloc` / `free` / `calloc` / `realloc`（堆状态可能被撕裂） |
| `wait`、`waitpid`、`sleep`、`alarm`、`pause` | `sprintf` / `snprintf`（依赖 locale 与缓冲） |
| `kill`、`raise`、`sigqueue` | `strerror`、`getpwnam`、`gethostbyname`（静态缓冲区） |
| `sigprocmask`、`sigpending`、`sigsuspend` | `readdir`、`localtime`、`asctime`、`ctime`（静态缓冲区） |
| `sigaction`、`signal` 与 `sigemptyset`/`sigfillset`/`sigaddset`/`sigdelset`/`sigismember` | `setenv` / `putenv`（修改全局环境） |
| `fork`、`execve`、`setpgid`、`getpid`、`getpgrp`、`getuid` | `rand` / `srand`（全局种子状态） |
| `strlen`、`strcmp`、`strcpy`、`memcpy`、`memset`、`memmove` | stdio 的任何 `f*` 系列（`fgets`/`fputs`/`fflush`…） |
| **`sem_post`**、`sem_wait`（POSIX 列入清单） | `pthread_*` 中未列入清单者、`dlopen`、`syslog` |
| `longjmp`/`siglongjmp`（见注） | `setjmp`（**不在**清单里） |

- **`write` 是唯一的异步信号安全输出函数**——所以信号处理程序里要打印信息，只能用 `write(STDOUT_FILENO, buf, n)` 或 CS:APP 提供的 **SIO 库**（`sio_puts`、`sio_putl`、`sio_error`、支持 `%c %s %d %u %x %%` 的 `sio_printf`，内部只用 `write`）。
- **`longjmp` 的限定**：自 POSIX.1-2008 TC2 起列入安全清单，但若处理程序打断的正是某个**非**异步信号安全函数，从处理程序中 `longjmp` 出去行为未定义。

#### A.10.10 信号处理五条准则

| 准则 | 内容 | 反面教材 |
|---|---|---|
| **G0**（讲义补充） | **让处理程序尽可能简单**：最理想是"置一个全局标志然后返回" | 在处理程序里做业务逻辑、加锁、遍历链表 |
| **G1** | 处理程序中**只调用异步信号安全函数** | `printf`、`sprintf`、`malloc`、`exit` **都不安全**；`free`、`strtok`、`gethostbyname`、`ctime` 同样不安全 |
| **G2** | **进场保存、退场恢复 `errno`** | 处理程序覆盖了主程序正要检查的 `errno`，导致主程序按错误原因走错分支 |
| **G3** | 访问**共享全局数据结构**时，**临时阻塞所有信号**保护 | 处理程序与主程序同时改同一个链表 → 结构损坏或丢失更新 |
| **G4** | **全局变量声明为 `volatile`** | 编译器把变量缓存在**寄存器**里，处理程序的修改"看不见"→ 死循环 |
| **G5** | **全局标志声明为 `volatile sig_atomic_t`** | 这类标志只做**读/写**（如 `flag = 1`，**不是** `flag++`），因此**不需要**像其他全局量那样加 G3 的保护 |

**G1–G3 代码骨架**：

```c
volatile sig_atomic_t flag = 0;              /* G4 + G5 */

void handler(int sig)                        /* G0：尽量简单 */
{
    int olderrno = errno;                    /* G2：进场保存 errno */
    flag = 1;                                /* G5：只做赋值，无需加锁 */
    write(STDOUT_FILENO, "caught\n", 7);     /* G1：只用异步信号安全的 write */
    errno = olderrno;                        /* G2：退场恢复 errno */
}
```

#### A.10.11 `setjmp` / `longjmp` 与信号版本

| 接口 | 原型/语义 |
|---|---|
| `setjmp` | `int setjmp(jmp_buf env)`：把当前**寄存器上下文（含栈指针）**存入 `env`。**直接调用时返回 0**；从 `longjmp` 返回时返回 `longjmp` 的第二个参数 |
| `longjmp` | `void longjmp(jmp_buf env, int retval)`：恢复 `env` 保存的上下文，效果是**掀掉中间所有栈帧**跳回 `setjmp` 处。`retval` **必须非 0**（传 0 会被实现改成 1） |
| `sigsetjmp` | `int sigsetjmp(sigjmp_buf env, int savesigs)`：`savesigs` 非 0 时**同时保存信号掩码** |
| `siglongjmp` | `void siglongjmp(sigjmp_buf env, int retval)`：跳回时**一并恢复信号掩码**——**从信号处理程序里跳出必须用这一对**，否则掩码会停留在"处理程序运行期间"的状态 |

**两个必须记住的陷阱**：

1. **`setjmp` 只能出现在 `if`、`switch`、循环条件或比较表达式里**，**不能**写成 `x = setjmp(env)`（标准把它限定在这几种"控制表达式"位置）。
2. **`volatile` 陷阱**：在 `setjmp` 与 `longjmp` 之间被修改的**局部变量**，若被编译器分配到**寄存器**，`longjmp` 恢复寄存器旧值后，这些变量会"回到过去"。**标准只保证 `volatile` 变量与全局变量正确**。因此处理程序里改过的标志、`longjmp` 之后要读的局部量，**一律加 `volatile`**。
3. **`longjmp` 不做清理**：中间栈帧里的 `malloc`、打开的 `fd` **不会被释放**（类比"撕掉书页，夹在里面的借书卡还在"）。C++ 的 `throw`/`catch` 会逐个调用析构函数，这是两者最本质的差别。

#### A.10.12 `sigsuspend` 与 `pause` 的区别

**四种"等待一个信号"的写法对照（讲义原例）**：

| 写法 | 评价 | 原因 |
|---|---|---|
| `while (!pid) pause();` | ❌ **有竞态** | 在"检查 `pid`"与"进入 `pause`"之间信号可能到达，于是 `pause` 永远等不到 → **永久挂起** |
| `while (!pid) sleep(1);` | ⚠️ **安全但慢** | 最坏要多等 **1 秒**才响应 |
| `while (!pid) ;` | ❌ **忙等** | 白烧 CPU |
| `sigprocmask(SIG_BLOCK, ...)` + `while (!pid) sigsuspend(&prev);` | ✅ **正确** | 改掩码与进入睡眠是**原子**的 |

```c
int sigsuspend(const sigset_t *mask);
/* 语义：原子地（不可打断地）执行以下三步——
 *   1) 把信号掩码设为 *mask
 *   2) 挂起进程直到收到一个信号（且该信号的处理程序返回后）
 *   3) 把信号掩码恢复为调用前的值
 */
```

- **`pause()` 的问题**是"改状态"与"睡下去"之间有窗口；**`sigsuspend` 把这两步合并成一个不可分割的原子操作**，窗口消失。
- **典型用法**：先 `sigprocmask(SIG_BLOCK, &mask, &prev)` 屏蔽 `SIGCHLD`（防止信号在 `fork` 与"准备等待"之间丢失），`fork` 后 `while (!pid) sigsuspend(&prev);`——`sigsuspend` 用 `prev`（即**未屏蔽 `SIGCHLD`** 的掩码）替换当前掩码，从而"在等待期间允许 `SIGCHLD` 递达"。

#### A.10.13 进程与信号常用命令

| 命令 | 作用 |
|---|---|
| `ps -o pid,ppid,pgid,stat,cmd` | 看 PID / 父 PID / **进程组** / 状态（`Z` = 僵尸、`T` = 停止）/ 命令 |
| `ps -efj` | 含 PGID、SID 的全表 |
| `kill -9 <pid>` / `kill -9 -<pgid>` | 给单个进程 / **整个进程组**发信号 |
| `pstree -p` | 以树形显示父子关系 |
| `strace -f -e trace=process,signal ./tsh` | 跟踪 `fork`/`execve`/`wait4`/信号递达——**Shell Lab 的头号调试利器** |
| `strace -f -e trace=signal -p <pid>` | 附着到运行中的进程看信号流 |
| `gdb -p <pid>` + `handle SIGINT nostop print pass` | 让 gdb 不拦截某些信号，观察程序自身行为 |
| `trap -l`（shell 内建） | 列出信号编号与名称对照 |
| `ulimit -c unlimited` | 允许 core dump（配合 `gdb ./prog core` 定位 `SIGSEGV`/`SIGABRT`） |
| `timeout -s INT 5 ./prog` | 5 秒后发 `SIGINT`，实测处理程序行为 |
---

### A.11 系统级 I/O 速查

> 对应讲次：Lecture 18（`F25-18-system-io.txt`）；教材 CS:APP3e 第 10 章；
> 关联 **L6 Shell Lab**（`dup2` 重定向）与 **L7 Proxy Lab**（`rio` + `stat`）。

#### A.11.1 `open` 的 flags 与 `mode`

**两种调用形式**：

```c
int open(const char *pathname, int flags);                  /* 打开已存在的文件 */
int open(const char *pathname, int flags, mode_t mode);     /* 打开或创建 */
/* 成功返回文件描述符（最小的未用非负整数）；出错返回 -1 并置 errno */
```

**访问模式（`flags` 中必须恰好包含其中之一）**：

| 标志 | 含义 |
|---|---|
| `O_RDONLY` | 只读 |
| `O_WRONLY` | 只写 |
| `O_RDWR` | 可读可写 |

**可选标志（用 `\|` 组合）**：

| 标志 | 含义 | 典型用途 |
|---|---|---|
| `O_CREAT` | 文件不存在则**创建**（**必须**同时提供第三个参数 `mode`） | 新建输出文件 |
| `O_TRUNC` | 若文件已存在，**先删除其全部内容**（截断为 0） | "覆盖写" |
| `O_APPEND` | 每次 `write` **都写到文件末尾**（内核保证偏移量更新与写入是原子的） | 日志追加 |
| `O_EXCL` | **与 `O_CREAT` 合用**：若文件**已存在则失败**（返回 `-1`，`errno = EEXIST`） | 原子地"创建独占文件"、锁文件 |
| `O_CLOEXEC` | **执行 `execve()` 时自动关闭**该描述符（FD_CLOEXEC 的原子版） | 防止 fd 泄漏给子进程 |
| `O_NONBLOCK` | 非阻塞模式（对管道/FIFO/socket 有意义） | 事件驱动 I/O |
| `O_SYNC` / `O_DSYNC` | 写入同步落盘（绕过页缓存延迟） | 数据库日志 |

**第三个参数 `mode` 与 `umask`**：

| 要点 | 说明 |
|---|---|
| **何时必须给** | **只有** `flags` 中出现 `O_CREAT` 时才有意义；否则**被忽略**（C 里 `open` 是**变参函数**，靠这个技巧实现"两个或三个参数"） |
| **`mode` 的含义** | 新建文件的**默认访问权限**位（如 `0644` = `rw-r--r--`） |
| **实际权限** | $\\text{实际权限} = mode\\ \\&\\ \\sim umask$，即**被 `umask` 掩掉一些位**。典型 `umask` 为 `022`，故 `0666 & ~022 = 0644` |
| **建议取值** | 没有特殊理由时用 `DEFFILEMODE`（来自 `<sys/stat.h>`，典型值 `0666`），让 `umask` 决定最终权限 |
| **查看** | `umask`（shell 内建命令，打印当前掩码） |

**实践模板**（讲义原例 + 健壮性补充）：

```c
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
    int fd;
    if ((fd = open("/etc/hosts", O_RDONLY)) < 0) {   /* 打开已有文件 */
        perror("open");
        exit(1);
    }
    /* ... 用 fd ... */
    close(fd);

    /* 创建（可读可写、截断、不存在则建） */
    int fd2 = open("out.txt", O_WRONLY | O_CREAT | O_TRUNC, DEFFILEMODE);
    if (fd2 < 0) { perror("open"); exit(1); }
    close(fd2);
    return 0;
}
```

#### A.11.2 `read` / `write` 返回值语义

```c
ssize_t read(int fd, void *buf, size_t n);
ssize_t write(int fd, const void *buf, size_t n);
```

| 返回值 | `read` 的含义 | `write` 的含义 |
|---|---|---|
| $n^{\\prime}$，$0 < n^{\\prime} < n$ | **短计数（short count）**——正常现象，**不是错误**（见下） | **短计数**——通常表示写满了（管道/socket 缓冲满）或磁盘满，**必须继续写剩余部分** |
| $n^{\\prime} = n$ | 完全成功 | 完全成功 |
| $0$ | **EOF**（文件结束）：**仅 `read` 可能出现**，且**只有**在 `read` 返回 0 时才能判定 EOF | **`write` 返回 0 不是 EOF**——`write` 不会返回 0（除非 `n == 0`），返回 0 或短计数都要当作异常/继续 |
| $-1$ | 出错，`errno` 说明原因 | 出错，`errno` 说明原因 |
| $-1$ 且 `errno == EINTR` | 被信号打断，**应重试**（不重试会**静默丢数据**） | 同左 |

**短计数在何时发生（讲义权威结论）**：

| **会发生**短计数 | **不会发生**短计数 |
|---|---|
| 读时遇到 **EOF** | 从**磁盘文件**读（除 EOF 外） |
| 从**终端**读（一次只能读到一行） | 向**磁盘文件**写 |
| 从**网络 socket** 读 | — |
| 读/写 **Linux 管道（pipe）** | — |

**最佳实践**：**永远为短计数做好准备**。讲义明确批评用 `read` 包一个 `while (n > 0)` 就以为万事大吉的做法——真正的解法就是 A.11.3 的 `rio` 库。

**逐字节 `cp` 的两个版本对比（讲义原例）**：

```c
/* ❌ 有 bug 的版本：没处理短计数，会静默丢数据 */
while (read(STDIN_FILENO, &c, 1) != 0)
    write(STDOUT_FILENO, &c, 1);

/* ✅ 正确版本：检查 write 的返回值并重试 */
while (Read(STDIN_FILENO, &c, 1) != 0)
    Write(STDOUT_FILENO, &c, 1);
```

#### A.11.3 `rio` 四函数对比与"为什么必须用 rio"

| 函数 | 原型 | 语义 | EOF 行为 | 可否混用 |
|---|---|---|---|---|
| **`rio_readn`** | `ssize_t rio_readn(int fd, void *usrbuf, size_t n)` | **无缓冲**的健壮读：自动处理短计数与 `EINTR`，**只在 EOF 时**返回短计数 | 返回**已读字节数**（可能 $< n$），再读返回 **0** | 与 `rio_writen` **可任意交错** |
| **`rio_writen`** | `ssize_t rio_writen(int fd, void *usrbuf, size_t n)` | **无缓冲**的健壮写：自动重试短计数与 `EINTR` | **永不返回短计数** | 同上 |
| **`rio_readinitb`** | `void rio_readinitb(rio_t *rp, int fd)` | 把 fd 绑定到一个带缓冲的 `rio_t`（**必须先调用**） | — | — |
| **`rio_readlineb`** | `ssize_t rio_readlineb(rio_t *rp, void *usrbuf, size_t maxlen)` | **带缓冲**地读**一行文本**：最多 `maxlen` 字节，读到 `'\n'` 为止并**保留** `'\n'`，末尾补 `'\0'` | 返回**已读字节数**；`0` 表示 EOF | 与 `rio_readnb` **可任意交错**；**⚠️ 不可与 `rio_readn` 交错** |
| **`rio_readnb`** | `ssize_t rio_readnb(rio_t *rp, void *usrbuf, size_t n)` | **带缓冲**地读 `n` 字节（不足只可能因 EOF） | 返回已读字节数；`0` 表示 EOF | 同上 |

**返回值总览**：`rio_readn`/`rio_writen`：成功返回传输字节数，`rio_readn` 在 EOF 返回 **0**，出错返回 **$-1$**。
`rio_readlineb`/`rio_readnb`：成功返回读到的字节数，**0 = EOF**，**$-1$ = 错误**。

**"为什么必须用 rio"——三层理由**：

1. **`read`/`write` 的短计数会让朴素代码静默出错**：网络 socket 与管道上短计数是常态，手写重试循环容易漏掉 `EINTR`、漏掉部分写、或在 `write` 里把"部分成功"当成功。
2. **文本协议需要按行读**：HTTP 请求头、Tiny 服务器的请求行、Shell 的命令行都以 `'\n'` 为界；`rio_readlineb` 把"逐字节读 + 找换行 + 补 `'\0'`"封装成一次调用，且**带缓冲**（一次系统调用喂满整个缓冲区），比逐字节 `read` 快几个数量级。
3. **线程/多协议下的正确性**：`rio_readlineb` 与 `rio_readnb` 共享同一个 `rio_t` 缓冲区，允许"读一行头部、再读 `Content-Length` 字节正文"这种混合模式；但 `rio_readn` **不带缓冲**、会绕过缓冲区，与缓冲版本混用会导致**数据错位**——这是 Proxy Lab 里最经典的诡异 bug。

**使用模板**：

```c
#include "csapp.h"
int main(void)
{
    rio_t rio;
    char buf[MAXLINE];
    Rio_readinitb(&rio, STDIN_FILENO);
    ssize_t n;
    while ((n = Rio_readlineb(&rio, buf, MAXLINE)) != 0)   /* 0 = EOF */
        Rio_writen(STDOUT_FILENO, buf, n);                 /* 永不短计数 */
    return 0;
}
```

#### A.11.4 `struct stat` 关键字段与类型判定宏

```c
#include <sys/stat.h>
int stat(const char *filename, struct stat *buf);   /* 按路径查询元数据 */
int lstat(const char *filename, struct stat *buf);  /* 不跟踪符号链接（查的是链接本身） */
int fstat(int fd, struct stat *buf);                /* 按已打开的描述符查询 */
/* 成功返回 0，出错返回 -1 并置 errno */
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `st_dev` | `dev_t` | 设备（含该文件的磁盘设备编号） |
| **`st_ino`** | `ino_t` | **i-node 号**（文件在磁盘上的真实身份标识） |
| **`st_mode`** | `mode_t` | **保护位与文件类型**（`S_IS*` 宏与权限位的来源） |
| **`st_nlink`** | `nlink_t` | **硬链接数**（= 0 时才真正删除文件） |
| `st_uid` / `st_gid` | `uid_t` / `gid_t` | 属主 / 属组的用户 ID |
| `st_rdev` | `dev_t` | 若为设备文件，则是设备类型 |
| **`st_size`** | `off_t` | **文件总大小（字节）**——Tiny 服务器发 `Content-Length` 就靠它 |
| `st_blksize` | `unsigned long` | 文件系统 I/O 的理想块大小 |
| `st_blocks` | `unsigned long` | 已分配的**块数**（512 字节为单位） |
| `st_atime` / `st_mtime` / `st_ctime` | `time_t` | 最后访问 / 最后修改 / 最后状态变更时间 |

**文件类型判定宏（都作用于 `st_mode`）**：

| 宏 | 判定类型 | 典型用途 |
|---|---|---|
| `S_ISREG(m)` | **普通文件**（regular file） | Tiny 服务器"只服务普通文件"的检查（拒绝目录、FIFO、设备） |
| `S_ISDIR(m)` | **目录** | `ls`、`find` 遍历 |
| `S_ISSOCK(m)` | **socket** | 识别 Unix 域套接字 |
| `S_ISLNK(m)` | **符号链接** | 与 `lstat` 配合检测链接（`stat` 会跟随链接，故看不到） |
| `S_ISCHR(m)` / `S_ISBLK(m)` | 字符设备 / 块设备 | `/dev` 下的文件 |
| `S_ISFIFO(m)` | **命名管道（FIFO）** | 防止 `read` 被 FIFO 永久阻塞 |
| `S_ISUID` / `S_ISGID` / `S_ISVTX` | set-uid / set-gid / sticky 位 | 权限审计 |

- **`stat` vs `lstat` 的唯一区别**：遇到**符号链接**时，`stat` **跟随**（返回目标文件的元数据），`lstat` **不跟随**（返回链接自身的元数据，`S_ISLNK` 为真）。
- **Tiny Web 服务器的核心安全检查**（Proxy Lab 的参考实现）：

```c
struct stat sbuf;
if (stat(filename, &sbuf) < 0) {           /* 文件不存在 */
    clienterror(cfd, filename, "404", "Not found", "Tiny couldn't find this file");
    return;
}
if (!(S_ISREG(sbuf.st_mode)) || !(S_IRUSR & sbuf.st_mode)) {   /* 不是普通文件或不可读 */
    clienterror(cfd, filename, "403", "Forbidden", "Tiny couldn't read the file");
    return;
}
```

#### A.11.5 三层结构与 `fork` 后的共享

**Unix 内核用三张表表示打开的文件**：

```
   进程 A 的描述符表            打开文件表（全系统共享）        v-node 表（全系统共享）
   [每进程一张]                [每打开一次一个表项]            [每个文件一个]
   +------+                    +----------------------+        +----------------------+
   | fd 0 |---+                | File pos (偏移量)     |        | st_mode / st_size    |
   | fd 1 |---+                | refcnt = 1           |        | ...（stat 结构信息）  |
   | fd 2 |---+                | File access mode     |   +--->| v-node 号 = i-node 号 |
   | fd 3 |   |                +----------------------+   |    +----------------------+
   | fd 4 |---+                        ^                  |              ^
   +------+   |                        |                  |              |
              +------------------------+------------------+--------------+
                                       |                  |
   进程 B 的描述符表（fork 后）         |                  |
   +------+                            |                  |
   | fd 0 |----------------------------+（共享同一表项！）|
   | fd 4 |-----------------------------------------------+
   +------+
```

| 层 | 表 | 共享范围 | 存放内容 | 关键点 |
|---|---|---|---|---|
| 第 1 层 | **描述符表**（descriptor table） | **每进程一张** | `fd → 打开文件表项指针` | `open`/`dup` 都改这一层；`fd` 只是**本进程的小整数下标** |
| 第 2 层 | **打开文件表**（open file table） | **全系统共享** | **文件偏移量 `File pos`**、`refcnt`、访问模式 | **`File pos` 是"每次 `open` 一份"的**——这就是下面两条规则的全部原因 |
| 第 3 层 | **v-node 表**（v-node table） | **全系统共享** | `stat` 结构信息（大小、类型、权限、i-node 号） | 无论多少进程打开同一文件，都指向**同一个 v-node** |

**两条必须记住的推论**：

| 场景 | 偏移量行为 | 图示要点 |
|---|---|---|
| **`fork` 之后** | 父子进程**共享**同一个打开文件表项 → **共享文件偏移量**。父进程读 5 字节后，子进程从第 6 字节继续读 | 两个描述符表项指向**同一个** File pos |
| **`open` 两次同一文件** | 内核创建**两个不同的打开文件表项** → **各自独立的偏移量**，互不干扰 | 两个 File pos，`refcnt` 各为 1 |
| `dup(fd)` / `dup2(fd, n)` | 新描述符与旧描述符指向**同一个**表项 → **共享偏移量** | 同 `fork` |

- **`refcnt`（引用计数）**：每多一个描述符指向该表项就加一；`close` 只减一，**归零时**才真正释放打开文件表项。
- **这一层结构解释了 Shell 重定向与 Proxy 的多线程共享行为，也是"`fork` 之后父子输出交错"的根本原因。**

#### A.11.6 `dup2` 重定向与 `fileno` / `fdopen`

```c
int dup2(int oldfd, int newfd);
/* 把（本进程）描述符表项 oldfd 复制到 newfd 位置。
 * 若 newfd 已打开，先关闭它（且是原子的）。
 * 成功返回 newfd；出错返回 -1。 */
```

**Shell 重定向 `ls > foo.txt` 的三步写法**：

```c
/* ① 打开目标文件，拿到 fd（通常是 3，因为 0/1/2 已被占用） */
int fd = open("foo.txt", O_WRONLY | O_CREAT | O_TRUNC, DEFFILEMODE);

/* ② 把 fd 复制到 stdout（fd 1）的位置 */
dup2(fd, STDOUT_FILENO);      /* 此后写 fd 1 就是写 foo.txt */

/* ③ 关闭多余的 fd（否则 foo.txt 的表项 refcnt 永远不为 0，泄漏） */
close(fd);

/* ④ 现在 execve：子进程继承"fd 1 指向 foo.txt"这一事实 */
execve("/bin/ls", argv, environ);
```

**重定向前后的描述符表**：

```
   before dup2(4,1)                        after dup2(4,1)
   +------+                                +------+
   | fd 0 |--> a (terminal)                | fd 0 |--> a (terminal)
   | fd 1 |--> a (stdout)                  | fd 1 |--> b (fd 4 的表项)
   | fd 2 |--> a (stderr)                  | fd 2 |--> a (stderr)
   | fd 3 |   (closed)                     | fd 3 |   (closed)
   | fd 4 |--> b (foo.txt)                 | fd 4 |--> b (foo.txt)
   +------+                                +------+
   a、b 为打开文件表项
```

- **`dup2` 的原子性**：`dup2` 先关 `newfd`（若已打开）再复制，这两步在**内核里原子完成**，不会出现"关掉了 `newfd` 但还没复制"的窗口。若 `oldfd == newfd`，`dup2` **直接返回 `newfd` 且不做任何事**（不会误关）。
- **`open` 后必须 `close` 原 fd**：`execve` 只保留 fd 1，多出的 fd 会让文件表项 `refcnt` 不为 0；而且长驻进程会**泄漏描述符**。

**标准 I/O 与文件描述符的双向转换**：

| 函数 | 原型 | 作用 |
|---|---|---|
| `fileno` | `int fileno(FILE *stream)` | 从 `FILE *` 取出底层的**文件描述符**（如 `fileno(stdout)` = 1） |
| `fdopen` | `FILE *fdopen(int fd, const char *mode)` | 把一个**已有的 fd** 包成 `FILE *`（例如把 socket fd 包成 `FILE *` 以便用 `fprintf`/`fgets`） |

```c
/* 把 socket 包成 FILE* —— 方便用 fprintf 写 HTTP 响应 */
int fd = open_clientfd(host, port);
FILE *fp = fdopen(fd, "w");
fprintf(fp, "GET / HTTP/1.0\r\n\r\n");
fflush(fp);                          /* 别忘了 flush！否则请求还压在缓冲区里 */
```

#### A.11.7 标准 I/O 缓冲类型与 `fflush`；`fork` 后输出重复的陷阱

| 缓冲类型 | 触发条件 | 何时真正写出 |
|---|---|---|
| **全缓冲**（fully buffered） | 流关联到**磁盘文件**（非常规文件） | 缓冲区满、`fflush`、`fclose`、或进程正常 `exit` 时 |
| **行缓冲**（line buffered） | 流关联到**终端**（`isatty` 为真） | 遇到 `'\n'`、缓冲区满、`fflush`、读操作、`fclose` |
| **无缓冲**（unbuffered） | **`stderr` 总是无缓冲** | 每次写立即发出（保证错误信息不丢） |

| 接口 | 作用 |
|---|---|
| `fflush(FILE *fp)` | **强制把用户空间缓冲区里的数据写出**到内核。`fflush(NULL)` 刷新**所有**输出流 |
| `setvbuf(FILE *fp, char *buf, int mode, size_t size)` | `mode` 取 `_IOFBF`（全缓冲）/`_IOLBF`（行缓冲）/`_IONBF`（无缓冲）；**必须在任何 I/O 之前调用** |
| `fclose(FILE *fp)` | 刷新并释放（**内含 `fflush`**） |

**`fork` 之后输出重复的陷阱（讲义与笔记中的经典 bug）**：

| 阶段 | 发生了什么 |
|---|---|
| ① 父进程 `printf("hello\n")`，**输出到磁盘文件**（全缓冲） | 数据只在**用户空间缓冲区**里，尚未 `write` 给内核 |
| ② 父进程 `fork()` | 子进程获得父进程**用户空间缓冲区的完整副本**（含那份未刷出的数据） |
| ③ 两个进程各自 `exit()` | 各自的 stdio 清理**分别把同一份数据写出一次** → **屏幕上出现两遍** |

**三种修复方式**：

1. **`fork` 之前 `fflush(stdout);`**（最直接）。
2. **输出到终端时改行缓冲**（但重定向到文件就失效，不能依赖）。
3. **干脆用 `write(STDOUT_FILENO, ...)`**（无用户空间缓冲，天然安全；也是信号处理程序中唯一可用的输出方式）。

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
    /* ⚠️ 把 stdout 重定向到文件后运行，可见 "hello" 打印两遍 */
    printf("hello\n");        /* 全缓冲：此时还没真正写出 */
    if (fork() == 0) {
        exit(0);              /* 子进程 exit -> 刷出缓冲区里那份 hello */
    }
    exit(0);                  /* 父进程 exit -> 再刷出一次 */
    /* 修复：在 fork 之前加一句 fflush(stdout); */
}
```

> ⚠️ **仅供演示，请勿模仿**：上面代码刻意制造"重复输出"。在**终端**上运行时因为行缓冲（遇 `\n` 立即刷出）看不到问题，**只有重定向到文件**才会暴露——这正是它难查的原因。复现命令：`./forkdup > out.txt; cat out.txt`。

#### A.11.8 目录操作

| 接口 | 原型 | 语义要点 |
|---|---|---|
| `opendir` | `DIR *opendir(const char *name)` | 打开目录，返回**目录流**；失败返回 `NULL` |
| `readdir` | `struct dirent *readdir(DIR *dirp)` | 返回**下一个目录项**的指针；**读完返回 `NULL`**（这既表示结束也表示出错，需配合 `errno`） |
| `closedir` | `int closedir(DIR *dirp)` | 关闭目录流 |
| `mkdir` | `int mkdir(const char *pathname, mode_t mode)` | 创建目录（`mode` 同样受 `umask` 影响） |
| `rmdir` | `int rmdir(const char *pathname)` | 删除**空**目录（非空则失败，`errno = ENOTEMPTY`） |
| `chdir` | `int chdir(const char *pathname)` | 改变**当前工作目录**（影响后续相对路径解析） |
| `getcwd` | `char *getcwd(char *buf, size_t size)` | 把**绝对路径**写入 `buf`（`size` 不够则失败，`errno = ERANGE`） |

**`struct dirent` 的关键字段**：

| 字段 | 含义 |
|---|---|
| `d_ino` | i-node 号（与 `stat` 的 `st_ino` 对应） |
| `d_off` | 目录流中的偏移量（供 `seekdir`/`telldir` 使用） |
| `d_reclen` | 本条记录的长度 |
| **`d_type`** | 文件类型（`DT_REG`/`DT_DIR`/`DT_LNK`…，**不是所有文件系统都填**，不可靠时要用 `stat`/`lstat` 再判） |
| **`d_name[]`** | 文件名（**不以 `'\0'` 保证结尾**以外的长度保证，用 `strncpy` 时注意） |

**遍历目录模板**：

```c
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : ".";
    DIR *dirp = opendir(path);
    if (!dirp) { perror("opendir"); return 1; }

    struct dirent *de;
    while ((de = readdir(dirp)) != NULL) {          /* NULL = 读完 */
        struct stat sbuf;
        if (stat(de->d_name, &sbuf) < 0) continue;  /* 相对当前工作目录 */
        const char *kind = S_ISDIR(sbuf.st_mode) ? "dir " :
                           S_ISREG(sbuf.st_mode) ? "file" : "othr";
        printf("%s  %10ld  %s\n", kind, (long)sbuf.st_size, de->d_name);
    }
    closedir(dirp);
    return 0;
}
```

- **必须跳过 `.` 与 `..`**（`readdir` 会返回它们），否则递归遍历会无限循环。
- **`readdir` 不是线程安全的**（返回内部静态缓冲区）；多线程下要用 `readdir_r`（已废弃）或在外部加锁。**因此它不在异步信号安全清单里。**
- **`d_name` 是相对名字**：`stat(de->d_name)` 只在"当前工作目录 == 被遍历目录"时才正确，严谨写法是把目录前缀拼上。

#### A.11.9 i-node、硬链接与符号链接

**i-node（索引节点）**：磁盘上每个文件（**每个真实文件对象**，而非每个名字）对应一个 i-node，保存**除文件名以外**的一切：文件类型与权限、属主、大小、时间戳、链接计数、**指向数据块的指针**。

| 对比项 | **硬链接**（hard link） | **符号链接**（symbolic link / soft link） |
|---|---|---|
| 本质 | 目录项里"**同一个 i-node 号**"的另一个名字 | 一个**独立文件**，内容是**目标路径字符串** |
| i-node | 与目标**共享同一个** i-node | 有**自己的** i-node，类型是 `S_ISLNK` |
| `st_nlink` | **+1**（这就是"硬链接数"） | **不变**（不影响目标的链接数） |
| 跨文件系统 | **不可以**（i-node 号只在同一文件系统内唯一） | **可以** |
| 指向目录 | **不允许**（会造成目录环） | **允许**（但可能产生循环，需靠最大深度限制） |
| 目标被删除后 | 只要 `st_nlink > 0`，**数据仍在**，其他名字照常访问 | **变成悬空链接（dangling）**，访问报 `ENOENT` |
| `stat` | 返回**目标**的元数据 | 返回**链接自身**的元数据（用 `lstat` 才看得到） |
| 创建命令 | `ln target linkname` | `ln -s target linkname` |
| 删除命令 | `unlink(name)` / `rm name` | `unlink(name)` / `rm name`（**删的是链接本身，不是目标**） |

| 命令 / 接口 | 作用 |
|---|---|
| `ln <target> <name>` | 创建**硬链接** |
| `ln -s <target> <name>` | 创建**符号链接**（`target` 可以是相对路径——**相对的是链接所在目录**，这是常见的困惑点） |
| `unlink(<pathname>)` | **删除一个目录项**：i-node 的 `st_nlink` 减 1，**减到 0 且无进程打开**时才真正释放数据块。系统调用名就叫 `unlink`（不是 `delete`），正体现了"删名字不删 inode"的语义 |
| `remove(<pathname>)` | libc 封装：对文件等价于 `unlink`，对目录等价于 `rmdir` |
| `stat` / `lstat` | 跟随 / 不跟随符号链接 |
| `readlink(<path>, buf, size)` | 读出符号链接**指向的路径字符串**（**不加** `'\0'`，且**不跟随**多级链接） |
| `ls -l` 显示 `lrwxrwxrwx ... link -> target` | 一眼识别符号链接；硬链接则表现为 `st_nlink` 大于 1，用 `ls -li` 比对 i-node 号 |

- **`st_nlink == 0` 才是文件的真正死亡时刻**：这解释了"为什么 `rm` 一个被进程打开的文件，进程仍能继续读写"——i-node 与数据块要等最后一个打开的 fd 关闭才释放。
- **`ln -s` 的相对路径陷阱**：`ln -s ../a/b link` 中 `../a/b` 是**相对 `link` 所在目录**解析的，不是相对你敲命令时的目录。用 `readlink` 可以看清链接里到底存了什么。

#### A.11.10 系统级 I/O 常用命令

| 命令 | 作用 |
|---|---|
| `strace -e trace=openat,read,write,close ./prog` | 看真实的系统调用序列与返回值（含短计数与 `EINTR`） |
| `strace -e trace=file -f ./prog` | 只跟踪与文件相关的调用（含 `stat`、`openat`） |
| `lsof -p <pid>` | 列出进程打开的所有 fd（**排查 fd 泄漏**） |
| `ls -l /proc/<pid>/fd` | 直接看描述符表（`0 -> /dev/pts/0`、`3 -> socket:[...]`） |
| `ls -li <file>` | 显示 i-node 号——**验证两个名字是否为硬链接**（同号即硬链接） |
| `stat <file>` | 格式化查看 `st_mode`、`st_ino`、`st_nlink`、`st_size`、三个时间戳 |
| `readlink -f <path>` | 逐级解析符号链接并输出绝对路径 |
| `cat /proc/<pid>/fdinfo/<fd>` | 查看单个 fd 的 `pos`（**文件偏移量**）、`flags`——验证"共享偏移量"的绝佳手段 |
| `mkfifo /tmp/f` | 创建命名管道，实测 FIFO 上的短计数行为 |
| `valgrind --track-fds=yes ./prog` | 报告 fd 泄漏与重复关闭 |

---

### A.12 网络编程速查

> 对应讲次：Lecture 19（`F25-19-fs_netprog1.txt`）、Lecture 20（`F25-20-netprog2.txt`）；
> 教材 CS:APP3e 第 11.1–11.6 节；关联 **L7 Proxy Lab**。

#### A.12.1 协议分层

| 层次 | 协议 | 提供的服务 | 数据单元 | 编址 |
|---|---|---|---|---|
| **应用层** | HTTP、DNS、SMTP、FTP | 具体应用语义 | 报文（message） | URL / 域名 |
| **传输层** | **TCP** / **UDP** | **进程到进程**的通信 | 段（TCP segment）/ 数据报（UDP datagram） | **端口号**（16 位） |
| **网络层** | **IP**（IPv4 / IPv6） | **主机到主机**的不可靠数据报投递 + 统一编址 | 数据报（datagram）/ 分组（packet） | **IP 地址**（IPv4 32 位 / IPv6 128 位） |
| **链路层** | Ethernet、802.11、Fibre Channel、T1、DSL | **同一局域网内**相邻节点间的帧传输 | 帧（frame） | MAC 地址 |
| **物理层** | 双绞线、光纤、无线电 | 比特流传输 | 比特 | — |

- **层次的意义**：每一层只依赖下一层提供的服务，并用自己的**封装（encapsulation）**把上层数据包起来——讲义的分层图明确画出 `LAN1 frame` 里套着 `internet packet`，包里再套 `data`，路由器**拆掉旧的帧头、换上新的帧头**（PH 保持不变，FH1 → FH2）。
- **一个"internet"（小写）** = 用**路由器**把若干互不兼容的 LAN 互连起来的网络；**"Internet"（大写）** 特指基于 **TCP/IP 协议族**的全球互联网。
- 协议要做两件事：**命名方案**（统一的地址格式）与**投递机制**（标准的传输单元 = 头部 + 载荷）。

#### A.12.2 IP / UDP / TCP 特性对比

| 特性 | **IP** | **UDP** | **TCP** |
|---|---|---|---|
| 英文 | Internet Protocol | User Datagram Protocol | Transmission Control Protocol |
| 服务对象 | **主机到主机**（host-to-host） | **进程到进程**（process-to-process） | **进程到进程**（process-to-process） |
| 可靠性 | **不可靠**：数据报可能丢失、重复、乱序 | **不可靠**（继承 IP） | **可靠**：不丢、不重、按序 |
| 连接性 | **无连接** | **无连接** | **面向连接**（三次握手建立、四次挥手关闭） |
| 数据抽象 | 数据报（datagram） | 数据报（datagram） | **字节流**（byte stream）：无记录边界 |
| 编址 | IP 地址 | IP 地址 + **端口** | IP 地址 + **端口** |
| 是否保证边界 | — | **保证**（一个数据报一个边界） | **不保证**（可能被合并/切分，必须自己定界） |
| 典型用途 | 所有上层协议的基础 | DNS、视频流、游戏、`traceroute` | HTTP、SSH、SMTP、几乎所有需要正确的场景 |
| 讲义原话 | "基本命名方案 + 不可靠的**分组投递**" | "用 IP 提供**不可靠的数据报**投递" | "用 IP 提供**可靠字节流**，运行在**连接**之上" |

- **TCP 字节流的推论的实战意义**：TCP **不保留应用层的记录边界**，所以 HTTP 必须用 `Content-Length` 或连接关闭（`Connection: close`）来定界——这正是 Proxy Lab 里"精确转发 `Content-Length`"如此关键的原因。**绝不能用"一次 `read` = 一个报文"来读 socket。**

#### A.12.3 字节序转换函数与网络字节序

**网络字节序 = 大端（big-endian）**：最高有效字节放在最低地址。任何**在包头中被搬运的整数**（IP 地址、端口号、长度字段）都必须用网络字节序，否则不同主机间会互相误解。

| 函数 | 原型 | 方向 | 用途 |
|---|---|---|---|
| `htonl` | `uint32_t htonl(uint32_t hostlong)` | 主机 → 网络，**32 位** | 转换 `in_addr.s_addr`（IPv4 地址） |
| `htons` | `uint16_t htons(uint16_t hostshort)` | 主机 → 网络，**16 位** | 转换端口号（`sin_port`） |
| `ntohl` | `uint32_t ntohl(uint32_t netlong)` | 网络 → 主机，**32 位** | 从包头读回地址 |
| `ntohs` | `uint16_t ntohs(uint16_t netshort)` | 网络 → 主机，**16 位** | 从包头读回端口 |

- **记忆**：`h` = host，`n` = network，`s` = short（16 位），`l` = long（**在这些函数里指 32 位**，不是 C 的 `long`）。
- **在**小端（x86-64 就是小端）机器上，这四个函数**真的会翻转字节**；在大端机器上它们是**空操作**。所以**必须调用**，不能因为"我的机器恰好匹配"而省略。
- 讲义强调："这对**任何**从一个机器传到另一个机器的包头整数都成立"，例如端口号。
- **`in_addr` 结构与点分十进制**：

```c
struct in_addr {
    uint32_t s_addr;   /* 网络字节序（big-endian） */
};
/* 0x8002C2F2 == 128.2.194.242 */
```

| 转换工具 | 作用 |
|---|---|
| `getaddrinfo` | 字符串（主机名或点分十进制 / IPv6 字面量）→ **socket 地址结构**（**首选**） |
| `getnameinfo` | socket 地址结构 → 字符串（点分十进制或主机名） |
| `inet_pton` / `inet_ntop` | 仅做"字符串 ↔ 二进制地址"转换（**不含 DNS 查询**，旧接口） |
| `inet_aton` / `inet_ntoa` | **废弃**接口（不支持 IPv6、`inet_ntoa` 非线程安全） |
| `gethostbyname` / `getservbyname` | **过时且非可重入**，已被 `getaddrinfo` 取代 |

- **IPv4 vs IPv6**：IPv4 是 **32 位**地址（1981 年制定，1990 年前后就已知不够用）；IPv6 是 **128 位**地址（1996 年制定，如 `2001:0db8:0:0:0:0:cafe:1a7e`），因需替换路由器而**推广极慢**。**现代 sockets API 让应用代码几乎无需关心**——用 `getaddrinfo` + `sockaddr_storage` 就自动兼容两者。

#### A.12.4 socket 地址结构

```c
/* 通用结构：只用于"指针类型转换"，从不直接使用其字段 */
struct sockaddr {
    uint16_t  sa_family;    /* 地址族：AF_INET / AF_INET6 */
    char      sa_data[14];  /* 地址数据（长度随协议族变化） */
};
typedef struct sockaddr SA;   /* CS:APP 的惯用简写 */

/* IPv4 专用 */
struct sockaddr_in {
    uint16_t       sin_family;  /* AF_INET */
    uint16_t       sin_port;    /* 端口号，网络字节序（htons） */
    struct in_addr sin_addr;    /* IP 地址，网络字节序（in_addr.s_addr） */
    unsigned char  sin_zero[8]; /* 填充，使结构与 sockaddr 等长（16 字节） */
};

/* IPv6 专用 */
struct sockaddr_in6 {
    uint16_t       sin6_family;   /* AF_INET6 */
    uint16_t       sin6_port;     /* 端口号，网络字节序 */
    uint32_t       sin6_flowinfo; /* 流信息 */
    struct in6_addr sin6_addr;    /* 128 位地址 */
    uint32_t       sin6_scope_id; /* 范围 ID */
};

/* 足够容纳任何地址族的"万能"结构（客户端地址就用它） */
struct sockaddr_storage {
    uint16_t ss_family;   /* 地址族；其余字节足够大以容纳任何协议族地址 */
    /* ... 填充到足够大 ... */
};
```

| 结构 | 何时用 | 关键点 |
|---|---|---|
| `struct sockaddr` | 仅作为**函数参数类型**（`SA *`） | 类型转换的"公共接口"，**不要**直接填字段 |
| `struct sockaddr_in` | 手写 IPv4 地址时 | `sin_port` 必须 `htons`；`sin_addr.s_addr` 必须网络字节序（用 `inet_pton`） |
| `struct sockaddr_in6` | 手写 IPv6 地址时 | — |
| **`struct sockaddr_storage`** | **`accept` 的客户端地址缓冲区**（唯一正确选择） | 大小**足够容纳任何**地址族；用 `sockaddr_in` 接收 IPv6 连接会**缓冲区溢出** |

- **`accept` 的标准写法**（讲义原例，注意 `clientlen` 的初始化）：

```c
struct sockaddr_storage clientaddr;              /* 够大，能装任何地址 */
socklen_t clientlen = sizeof(struct sockaddr_storage);   /* ⚠️ 必须初始化！ */
int connfd = Accept(listenfd, (SA *)&clientaddr, &clientlen);
```

- **类型转换约定**：`sockaddr_storage` → `sockaddr_in` 时用 `(struct sockaddr_in *)&clientaddr`，读 `sin_family` 判断是哪一族，再取 `sin_addr`/`sin_port`。
- **`socklen_t` 是"值-结果"参数**：`accept` 进入时你要告诉内核缓冲区**有多大**，返回时内核写入**实际用了多少**；不初始化就是 UB。

#### A.12.5 套接字函数表

| 函数 | 作用 | 客户端 / 服务器 | 关键参数 | 常见错误 |
|---|---|---|---|---|
| `getaddrinfo` | 主机名+服务名 → **`addrinfo` 链表**（每个含一个 ready-to-use 的 socket 地址与参数） | **两者都用** | `host`、`service`、`hints`（输入约束）、`result`（输出链表） | 忘记 `freeaddrinfo` 泄漏；`hints` 未 `memset` 清零导致垃圾字段；不知道要**遍历链表**逐个尝试 |
| `freeaddrinfo` | 释放 `getaddrinfo` 返回的**整个链表** | 两者 | `result` | 只 `free` 首个节点；用 `free()` 逐个释放（错，链表是一次分配的，必须用这个函数） |
| `socket` | 创建**套接字描述符** | 两者 | `domain`（`AF_INET`）、`type`（`SOCK_STREAM`）、`protocol`（`0` = 自动选择） | 用 `getaddrinfo` 时**不要硬编码** `AF_INET`/`SOCK_STREAM`，应写 `ai->ai_family`/`ai->ai_socktype` |
| `connect` | 客户端**发起连接**（三次握手） | **仅客户端** | `clientfd`、服务器地址 `addr`、`addrlen` | 忘了对链表中每个候选地址重试；未检查返回 $-1$ |
| `bind` | 把**本地地址**与 `sockfd` 关联 | **仅服务器** | `sockfd`、`addr`（含端口）、`addrlen` | 端口被占用（`EADDRINUSE`，需 `SO_REUSEADDR`）；`addr` 未用 `htons` 转端口 |
| `listen` | 把**主动套接字**转为**监听套接字** | **仅服务器** | `sockfd`、`backlog`（内核排队上限，**约 128**） | 忘了 `listen` 直接 `accept`；把 `listenfd` 当数据通道读写 |
| `accept` | **等待并接受**一个连接请求，返回**已连接描述符 `connfd`** | **仅服务器** | `listenfd`、`addr`（客户端地址，用 `sockaddr_storage`）、`addrlen`（**必须初始化**） | **在循环外只 `accept` 一次**（只能服务一个客户端）；用 `sockaddr_in` 接 IPv6 |
| `close` | 关闭描述符 | 两者 | `fd` | 忘记 `close(connfd)` → **描述符泄漏**（长驻服务器最终 `EMFILE`） |
| `setsockopt` | 设置套接字选项 | 通常服务器 | `SOL_SOCKET`、`SO_REUSEADDR`、`&optval`、`sizeof(optval)` | Proxy/Tiny 调试时端口 `TIME_WAIT` 导致"地址已占用"，加 `SO_REUSEADDR` 解决 |
| `getnameinfo` | 地址结构 → 主机名/服务名的可读字符串 | 两者 | 地址、长度、缓冲区 | 用于日志打印（把客户端 IP 转成点分十进制） |

**错误处理要点**：`getaddrinfo` **不设置 `errno`**，要用 **`gai_strerror(errcode)`** 取错误信息；其余函数返回 $-1$ 时用 `errno` + `strerror`/`perror`。

- **CS:APP 封装**：`open_clientfd(host, port)` 内部 = `getaddrinfo` + 循环 `socket`/`connect` + `freeaddrinfo`；`open_listenfd(port)` 内部 = `getaddrinfo(hints.ai_flags = AI_PASSIVE)` + `socket`/`setsockopt(SO_REUSEADDR)`/`bind`/`listen`/`freeaddrinfo`。

#### A.12.6 客户端与服务器的调用顺序

**客户端（`open_clientfd` 展开）**：

```
  getaddrinfo(host, port, &hints, &listp)      <-- 解析主机+服务
        |
        v
  for (p = listp; p; p = p->ai_next) {         <-- 逐个候选地址尝试
        |
        v
     socket(p->ai_family, p->ai_socktype, p->ai_protocol)   -> clientfd
        |
        v
     connect(clientfd, p->ai_addr, p->ai_addrlen)   <-- 成功则跳出循环
  }
        |
        v
  freeaddrinfo(listp)
        |
        v
  rio_readinitb(&rio, clientfd)  ->  rio_writen(请求)  <->  rio_readlineb/rio_readnb(响应)
        |
        v
  close(clientfd)
```

**服务器（迭代式 `open_listenfd` 展开）**：

```
  getaddrinfo(NULL, port, &hints{AI_PASSIVE}, &listp)      <-- 通配本地地址
        |
        v
  for (p = listp; p; p = p->ai_next) {
     socket(p->ai_family, p->ai_socktype, p->ai_protocol)  -> listenfd
     setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, ...)
     bind(listenfd, p->ai_addr, p->ai_addrlen)             <-- 关联端口，失败则 close 重试
     listen(listenfd, LISTENQ)                             <-- 转为监听套接字
  }
  freeaddrinfo(listp)
        |
        v
  while (1) {                                              <-- 服务器是无限循环！
     clientlen = sizeof(struct sockaddr_storage);
     connfd = accept(listenfd, (SA *)&clientaddr, &clientlen)   <-- 阻塞等待
        |
        v
     do_work(connfd)                                       <-- 读写请求/响应（rio）
        |
        v
     close(connfd)                                         <-- ⚠️ 关的是 connfd，不是 listenfd
  }
```

- **一句话记住**：**客户端 = `getaddrinfo` → `socket` → `connect`；服务器 = `getaddrinfo` → `socket` → `bind` → `listen` → 循环(`accept` → 服务 → `close`)**
- **`backlog` 的含义**：内核在**开始拒绝**连接请求之前，允许排队的**未完成连接请求**数量（约 128）。它**不是**能同时服务的客户端数上限。

#### A.12.7 `listenfd` vs `connfd`

| 对比项 | **`listenfd`**（监听描述符） | **`connfd`**（已连接描述符） |
|---|---|---|
| 创建次数 | **整个服务器生命周期只创建一次**（在 `open_listenfd` 里） | **每接受一个客户端连接就创建一个** |
| 创建者 | `socket()` + `bind()` + `listen()` | `accept()` 返回 |
| 生命周期 | 与服务器进程同寿 | 只覆盖**这一次**请求/响应，处理完立即 `close(connfd)` |
| 用途 | 是"**接线员**"：只负责**接收连接请求**，**绝不用它读写数据** | 是"**通话线路**"：真正的读写通道，交给 `rio_read*`/`rio_writen` 或工作线程 |
| 并发含义 | 唯一，被所有请求共享 | 每连接一个，是**并发的天然分界点**——Proxy Lab 里 `connfd` 就是传给每个工作线程的参数 |

- **最常见的初学者 bug**：把 `listenfd` 传给 `rio_readlineb` 去读 HTTP 请求，或者只 `accept` 一次然后处理所有客户端——两者都源自"分不清这两个描述符"。
- **`close(connfd)` 必须做**：`listenfd` 永远不关，`connfd` 每次都要关，否则描述符泄漏。

#### A.12.8 `getaddrinfo` 的 `hints` 关键字段

```c
#include <netdb.h>
int getaddrinfo(const char *host,             /* 主机名或地址字符串（可为 NULL） */
                const char *service,          /* 端口号或服务名 */
                const struct addrinfo *hints, /* 输入：对结果的约束 */
                struct addrinfo **result);    /* 输出：addrinfo 链表 */
void freeaddrinfo(struct addrinfo *result);
const char *gai_strerror(int errcode);
```

| `hints` 字段 | 取值 | 含义 |
|---|---|---|
| `ai_family` | `AF_INET` / `AF_INET6` / `AF_UNSPEC` | 只接受 IPv4 / 只接受 IPv6 / **两者都接受**（**可移植代码应留 `AF_UNSPEC`**，或一开始就用 `AF_INET` 也不要紧，取决于需求） |
| `ai_socktype` | `SOCK_STREAM` / `SOCK_DGRAM` | 只要 TCP / 只要 UDP |
| `ai_protocol` | `0`（或 `IPPROTO_TCP`） | 通常给 0 让系统推断 |
| **`ai_flags`** | **`AI_PASSIVE`** | **服务器专用**：表示"我要 `bind` 一个**监听**套接字，`host` 传 `NULL` 时返回**通配地址**（`INADDR_ANY`/`in6addr_any`）"。**不设它而 `host = NULL`，得到的地址无法用于 `bind`** |
| `ai_flags` | `AI_NUMERICHOST` / `AI_NUMERICSERV` | 禁止 DNS 查询 / 只接受数字端口（加速与调试） |
| `ai_flags` | `AI_CANONNAME` | 在 `ai_canonname` 里返回规范主机名 |
| `ai_flags` | `AI_ADDRCONFIG` | 只返回本机**已配置**的地址族（避免拿到用不了的 IPv6） |

- **`hints` 必须清零**：`memset(&hints, 0, sizeof(hints));` 后再填字段——未清零的栈垃圾会让 `ai_flags` 里带上随机标志位。
- **`ai_addr` / `ai_addrlen` / `ai_family` / `ai_socktype` 是"可直接用"的**：讲义明确说"用 `getaddrinfo` 就**不必**硬编码协议细节"，`socket()` 直接传 `ai->ai_family, ai->ai_socktype, ai->ai_protocol`。
- **`ai_next`**：链表指针，**必须遍历**——一个主机名可能解析出多个 IP（如 `twitter.com` 返回 4 个 IPv4），一个地址族失败要试下一个。
- **`getaddrinfo` 的三个优势**：**可重入**（线程安全）、**协议无关**（同一份代码兼容 IPv4/IPv6）、**取代了 `gethostbyname`/`getservbyname`**（后者非可重入）。缺点：**略复杂**，但"少数几个用法模式足以覆盖绝大多数场景"。
- **`AI_PASSIVE` 的对照写法**：

```c
struct addrinfo hints, *listp;
memset(&hints, 0, sizeof(hints));
hints.ai_socktype = SOCK_STREAM;
hints.ai_flags    = AI_PASSIVE | AI_ADDRCONFIG | AI_NUMERICSERV;
getaddrinfo(NULL, port, &hints, &listp);   /* 服务器：NULL host + AI_PASSIVE */
```

#### A.12.9 HTTP 请求/响应报文格式模板

**HTTP 请求**：请求行 + 零个或多个请求头 + **空行**（+ 可选请求体）

```
<method> <uri> <version>\r\n
<header name>: <header data>\r\n
<header name>: <header data>\r\n
\r\n
```

```
GET /index.html HTTP/1.1\r\n        <- 请求行：方法 + URI + 版本
Host: www.cmu.edu\r\n               <- 必需头（HTTP/1.1）
User-Agent: Mozilla/5.0 ...\r\n     <- 可选头
\r\n                                <- 空行：终止头部
```

**HTTP 响应**：响应行 + 零个或多个响应头 + **空行** + 可选内容

```
<version> <status code> <status msg>\r\n
<header name>: <header data>\r\n
\r\n
<content bytes>
```

```
HTTP/1.1 200 OK\r\n                 <- 响应行：版本 + 状态码 + 状态消息
Date: Wed, 05 Nov 2014 17:37:26 GMT\r\n
Server: Apache/1.3.42 (Unix)\r\n
Content-Type: text/html; charset=UTF-8\r\n
Content-Length: 4096\r\n            <- 正文长度
\r\n                                <- 空行：头部与正文的分界
<html> ... </html>                  <- 正文
```

| 关键规则 | 说明 |
|---|---|
| **每一行都以 `\r\n` 结尾** | 不是 `\n`！这是 HTTP 规范（RFC 1945/7230）的硬性要求 |
| **空行 = `\r\n`（一个孤立的 CRLF）** | **同时**终止**请求**头部与**响应**头部；读写时必须靠它切分头部与正文 |
| **请求行三字段** | `<method> <uri> <version>`；`<uri>` 对**代理**是**完整 URL**，对**源服务器**是 **URL 后缀**（路径） |
| **响应行三字段** | `<version> <status code> <status msg>` |
| 头部字段 | `<header name>: <header data>`，**冒号后有一个空格** |
| 版本 | `<version>` ∈ {`HTTP/1.0`, `HTTP/1.1`} |
| **文本行可以是任意长度** | 所以读头部**必须用 `rio_readlineb` 配足够大的缓冲**，并检查溢出 |
| 正文长度 | 靠 `Content-Length` **或** 连接关闭（`Connection: close`）定界；**分块传输（chunked）** 用十六进制长度行（如 `15c`）开头、以 `0` 行结束（讲义 telnet 实例中出现） |

- **代理与源服务器的 URI 差异（Proxy Lab 的核心）**：
  浏览器发给代理的是 **`GET http://www.cmu.edu/hub/index.html HTTP/1.1`**（**完整 URL**）；
  代理转发给源服务器的必须是 **`GET /hub/index.html HTTP/1.0`**（**相对路径**）。
  即：代理必须**拆出 hostname 与 path**，把 `HTTP/1.1` **降级为 `HTTP/1.0`**。

#### A.12.10 HTTP 方法与状态码

**请求方法**（讲义列出 7 个）：

| 方法 | 语义 | 有请求体 | 幂等/安全 | 备注 |
|---|---|---|---|---|
| **`GET`** | 请求指定资源 | 无 | 安全、幂等 | 最常见；Proxy Lab **只要求支持 `GET`** |
| `POST` | 向服务器提交数据 | **有** | 非幂等 | 表单提交；Proxy Lab **可选** |
| `HEAD` | 只要**响应头**，不要正文 | 无 | 安全、幂等 | 用于检查资源是否存在/是否改动；**Proxy Lab 不必支持** |
| `OPTIONS` | 查询服务器/资源**支持的方法** | 无 | 安全、幂等 | CORS 预检也用它 |
| `PUT` | 上传/替换资源 | 有 | 幂等 | REST 风格 |
| `DELETE` | 删除资源 | 可无 | 幂等 | REST 风格 |
| `TRACE` | 回显收到的请求（诊断） | 无 | 安全 | 常因安全原因被禁用 |

**状态码**（讲义直接给出的：200、301、404；下表其余为**补充说明**，均为 RFC 7231 定义的标准码）：

| 码 | 消息 | 含义 | 何时返回 |
|---|---|---|---|
| **200** | `OK` | 请求被**无错误地处理** | 正常响应 |
| **301** | `Moved Permanently` | **永久**重定向，提供**备用 URL**（`Location` 头） | 资源永久搬家（讲义 telnet 实例中 `www.cmu.edu/` → `/index.shtml` 就是 301） |
| **302** | `Found` | **临时**重定向（`Location` 头） | 临时跳转（旧称 `Moved Temporarily`） |
| **400** | `Bad Request` | 请求**语法错误**、代理无法解析 | 畸形请求行/头部 |
| **403** | `Forbidden` | 服务器**拒绝**提供该资源（有权限问题或非普通文件） | Tiny 服务器的"文件不可读"分支 |
| **404** | `Not Found` | **服务器找不到该文件** | 资源不存在 |
| **500** | `Internal Server Error` | 服务器**内部错误** | 服务器代码抛异常 |
| **501** | `Not Implemented` | 服务器**不支持**该请求方法 | 收到未实现的方法（如只支持 `GET` 却收到 `POST`） |
| **502** | `Bad Gateway` | 作为网关/代理时，**从上游收到无效响应** | **Proxy Lab 中 DNS 解析失败或服务器不可达时的常用码** |
| **505** | `HTTP Version Not Supported` | **不支持的 HTTP 版本** | 请求行版本字段无法识别 |

- **状态码分段记忆**：**1xx 信息、2xx 成功、3xx 重定向、4xx 客户端错、5xx 服务器错**。
- **Proxy Lab 的错误码映射**：客户端请求无法解析 → `400`；方法不支持 → `501`；版本不支持 → `505`；DNS/连接上游失败 → `502`；且**必须给客户端一个合法的 HTTP 错误响应**（不能直接关连接，也不能让代理退出）。

#### A.12.11 HTTP 常见头部

| 头部 | 方向 | 作用 | 代理的处理要点 |
|---|---|---|---|
| **`Host`** | 请求 | **端服务器的主机名**（HTTP/1.1 必需） | 代理**必须发送**（用于虚拟主机）；若浏览器已带，就**沿用浏览器的值**；`Host` 不含端口时用默认 **80** |
| **`User-Agent`** | 请求 | 标识客户端（操作系统、浏览器、引擎版本） | 可选；Proxy Lab 给了现成字符串 `User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:10.0.3) Gecko/20120305 Firefox/10.0.3`（讲义中分两行显示，**实际必须作为单独一行发送**） |
| **`Connection`** | 请求/响应 | 本次交换后**是否保持连接** | 代理**总是发送 `Connection: close`** |
| **`Proxy-Connection`** | 请求 | **非标准**头，用于和代理协商连接保持 | 代理**总是发送 `Proxy-Connection: close`**；对端发来的这个头**应丢弃**（不转发给源服务器） |
| **`Content-Length`** | 响应（也可用于请求） | **正文的精确字节数** | 代理**必须精确转发**——少转发会截断对象、多转发会把后续数据当正文；**缓存对象大小也依赖它** |
| **`Content-Type`** | 响应 | 正文的 **MIME 类型** + 可选参数（如 `charset=UTF-8`） | 原样转发（缓存命中时也必须带） |
| `Location` | 响应 | 重定向目标 URL | 通常**不修改**（注意：重定向目标对代理是新的完整 URL） |
| `Date` / `Server` | 响应 | 时间戳 / 服务器软件标识 | 原样转发 |
| `Transfer-Encoding: chunked` | 响应 | 正文采用**分块传输**，无 `Content-Length` | 代理如不支持分块，仍可**逐字节透传**直到连接关闭 |

- **Proxy Lab 的头部规则总结**：① **总是**发送 `Host`；② **总是**发送 `Connection: close` 和 `Proxy-Connection: close`；③ `User-Agent` 可选（提供了常量字符串）；④ **其他浏览器发来的头一律原样转发**；⑤ 必须**丢弃** `Proxy-Connection`（它是给代理看的，不是给源服务器看的）。

#### A.12.12 MIME 类型表

| MIME 类型 | 含义 |
|---|---|
| **`text/html`** | HTML 文档（网页） |
| **`text/plain`** | 未格式化的纯文本 |
| **`image/gif`** | GIF 格式的二进制图像 |
| **`image/png`** | PNG 格式的二进制图像 |
| `image/jpeg` | JPEG 格式的二进制图像 |
| `application/pdf` | PDF 文档 |
| `application/octet-stream` | 未知/通用二进制流（Tiny 服务器对不认识的扩展名的默认值） |

- **MIME** = **Multipurpose Internet Mail Extensions**：Web 服务器返回的 content 是"一段字节流 + 一个关联的 MIME 类型"。
- 完整列表见 IANA 的 media-types 注册表（`http://www.iana.org/assignments/media-types/media-types.xhtml`）。
- **静态内容 vs 动态内容**：静态 = 存在文件里按请求取回（HTML/图片/音频/JS）；动态 = **按请求现场生成**（服务器为客户端执行程序，约定俗成放在 `cgi-bin` 目录）。二者由服务器**根据 URL 后缀自行判断**。
- **URL 的解剖**：`http://www.cmu.edu:80/index.html` —— 客户端用**前缀** `http://www.cmu.edu:80` 推断**协议（HTTP）、主机、端口**；服务器用**后缀** `/index.html` 判断静态/动态、定位文件。**最小后缀是 `/`**，服务器把它扩展为配置的默认文件名（通常是 `index.html`）。

#### A.12.13 Proxy Lab 要点

**总体流程**：迭代式代理 → 多线程（每连接一线程）→ 加缓存（读者-写者锁 + LRU 淘汰）。**Gradescope/Autolab 评分 70 分**：

| 项 | 分值 | 内容 |
|---|---|---|
| **BasicCorrectness** | 40 | 基本代理功能（自动评分） |
| **Concurrency** | 15 | 处理**并发请求** |
| **Cache** | 15 | 可工作的**缓存** |

**① 全 URL vs 相对路径**：

| 位置 | 请求行里的 URI |
|---|---|
| 浏览器 → 代理 | **完整 URL**：`GET http://www.cmu.edu/hub/index.html HTTP/1.1` |
| 代理 → 源服务器 | **相对路径**：`GET /hub/index.html HTTP/1.0`（**版本降到 1.0**） |

**必须**从 URL 中解析出 **hostname**（`www.cmu.edu`）、**可选端口**（`http://host:**8080**/path`，缺省 **80**）、**路径与查询串**（`/hub/index.html`）三部分。**URL 带不带端口号都必须能正常工作**。

**② 必须保留 / 丢弃的头部**：

| 动作 | 头部 |
|---|---|
| **必须发送** | `Host: <hostname>`（虚拟主机必需；浏览器已带则沿用其值） |
| **必须发送（固定值）** | `Connection: close`、`Proxy-Connection: close` |
| **可选发送（固定值）** | `User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:10.0.3) Gecko/20120305 Firefox/10.0.3` |
| **必须原样转发** | 浏览器发来的**其他所有**请求头 |
| **必须丢弃** | `Proxy-Connection`（给代理看的，不应转发给源服务器） |

**③ 错误码映射表**：

| 情况 | 响应 |
|---|---|
| 请求行/头部**畸形**、无法解析 | `400 Bad Request` |
| 请求方法不是 `GET`（而你只实现 GET） | `501 Not Implemented` |
| HTTP 版本不受支持 | `505 HTTP Version Not Supported` |
| **DNS 解析失败 / 无法连接上游服务器** | `502 Bad Gateway`（并**继续服务**其他请求，**绝不退出**） |
| 客户端请求的资源不存在（由上游返回） | 直接**透传上游的 `404`** |

- **健壮性要求（讲义明确）**：代理是**长驻进程**，"对很多错误而言，代理立刻退出显然是不合适的"；必须**对畸形甚至恶意输入鲁棒**、不段错误、**不泄漏内存与文件描述符**；还要能优雅处理 `write` 返回 **`EPIPE`** 的情况（客户端提前断开）。

**④ `Content-Length` 精确转发**：

*   正文是**二进制**的，长度必须靠 `Content-Length` 判定，**不能靠 `rio_readlineb` 读行**（正文里可能出现 `\n`，也不一定是文本）。
*   标准做法：`Content-Length: N` → 用 **`rio_readnb(connfd, buf, N)`** 精确读 N 字节，再 `rio_writen` 精确写出 N 字节。
*   转发给客户端时，`Content-Length` 必须与**实际写出的字节数一致**；缓存命中时也要用缓存里记录的长度，而不是重新 `strlen`。

**⑤ 缓存约束（讲义明确规定）**：

| 常量 | 值 | 说明 |
|---|---|---|
| `MAX_CACHE_SIZE` | **1 MiB** | 整个缓存上限；**只统计存 web 对象的字节**，元数据（链表节点、键等）不计入 |
| `MAX_OBJECT_SIZE` | **100 KiB** | 单个对象上限；超过则不缓存 |

*   **边转发边缓存**：从服务器读数据的同时发给客户端，**同时**往缓冲区里累积；**若缓冲超过 `MAX_OBJECT_SIZE` 就丢弃缓冲**（不再尝试缓存这个对象）。
*   这个方案下，代理用于 web 对象的**最大内存**是 `MAX_CACHE_SIZE + T * MAX_OBJECT_SIZE`（$T$ = 最大活跃连接数）= 缓存本身 + 每个活跃连接各一份累积缓冲。
*   **淘汰策略**：**近似 LRU** 即可（讲义原文："不必严格 LRU，但应当合理地接近"）；**注意**"读取对象"和"写入对象"**都算使用**。
*   **同步（本 lab 最难点）**：缓存访问必须线程安全，但**有硬性要求——多个线程必须能同时读缓存**，只有**写**才允许独占。因此"**用一把大独占锁保护整个缓存是不可接受的方案**"。可选做法：**分区缓存**、**`pthread_rwlock_t` 读者-写者锁**、或**用信号量自己实现读者-写者**。不必严格 LRU 这一点，正好为支持多读者留出腾挪空间。
*   **多线程注意**：工作线程必须 **detached**（否则内存泄漏）；`open_clientfd`/`open_listenfd` 基于**可重入的 `getaddrinfo`**，因此**线程安全**。

**⑥ 调试工具**：

| 工具 | 用途 |
|---|---|
| `./driver.sh` | **官方自动评分器**（BasicCorrectness + Concurrency + Cache），必须在 **Linux 机器**上运行 |
| `tiny`（CS:APP Tiny Web 服务器） | handout 自带源码；**driver 抓页面就用它**；也是写代理的起点 |
| `telnet localhost <proxy_port>` | 手写 HTTP 请求，逐字观察代理转发与响应 |
| `curl -v --proxy http://localhost:<proxy_port> http://localhost:<tiny_port>/home.html` | 端到端验证代理链路 |
| `netcat`（`nc -l 12345`） | 伪装成服务器，观察代理**实际发出去的字节**（验证头部与请求行） |
| `./port-for-user.pl <userid>` | 生成**个人专属端口号**（**偶数**，如 `45806`）；可用 `p` 与 `p+1` 两个端口；**别自己随便挑端口**，会干扰他人 |
| `valgrind --leak-check=full` / `--track-fds=yes` | 查内存与 fd 泄漏（讲义要求"无泄漏"） |
| `ps -o pid,cmd -L` | 观察多线程是否真的创建了线程 |

- **测试注意**：用**浏览器**测缓存**极不可靠**——现代浏览器自己会缓存、会加条件请求头，所以**必须用 `curl`/`telnet` 或 driver** 来验证缓存命中。
- **两个"环境"常识**：HTTP 请求端口是 URL 里的可选字段（缺省 **80**）；代理的**监听端口**由命令行参数给出（`./proxy 15213`），必须是 **1024 < p < 65536** 的非特权端口。

---

### A.13 并发与同步速查

> 对应讲次：Lecture 21（`F25-21-concprog.txt`）、Lecture 22（`F25-22-sync-basic.txt`）、
> Lecture 23（`F25-23-sync-advanced.txt`）、Lecture 24（`F25-24-parallelism.txt`）；
> 教材 CS:APP3e 第 12 章；关联 **L7 Proxy Lab**、**L8 SFS Lab**。

#### A.13.1 三种并发模型对比

| 维度 | **基于进程**（process-based） | **基于 I/O 多路复用/事件驱动**（event-based） | **基于线程**（thread-based） |
|---|---|---|---|
| 核心机制 | 每个连接 `fork` 一个**子进程** | 单进程单线程，用 `select`/`poll`/`epoll` 检测就绪 fd，**复用**一个控制流 | 每个连接一个**线程**（`pthread_create`） |
| 创建开销 | **最高**（`fork` 要复制地址空间结构，代价大） | **最低**（无新执行流，只需一个 fd 表项） | **居中**（线程创建远低于进程，但非零） |
| 上下文切换开销 | **最高**（切换页表/`%cr3`、TLB 失效） | **极低**（同一进程内，仅函数调用与状态机跳转） | **居中**（同进程换线程栈与寄存器，**不换页表**） |
| 数据共享难度 | **最难**：无共享状态；要共享必须显式 IPC（管道/共享内存）。讲义标注：描述符 **不**共享、**文件表共享**、全局变量 **不**共享 | **最容易**：一个地址空间，所有连接数据就是普通全局/堆变量 | **最容易**：共享代码、数据、堆、内核上下文；**但必须加同步** |
| 控制流复杂度 | **低**：每个连接一条顺序控制流，代码直观 | **很高**：必须把逻辑写成状态机（"部分 HTTP 头到达怎么处理？"是典型难题）；难做细粒度并发 | **中**：每连接一条顺序控制流，但并发交错使调试变难 |
| 多核利用 | **可以**（多进程天然并行） | **不能**：只有**一个**逻辑控制流，**无法利用多核** | **可以**（多线程并行，最直接） |
| 适用连接数 | 中（受进程数上限与切换成本约束） | **极大**（C10K 场景的设计选择） | 大（受线程栈内存与切换成本约束） |
| 调试友好度 | 中（可单独 attach 某子进程） | **最好**：单控制流，可**单步调试** | 较差（竞态、时序依赖） |
| 典型缺点 | 需要**回收僵尸子进程**（否则致命内存泄漏）；父子都要**关掉对方那份 fd**（`refcnt(connfd) = 2`） | **编码显著更复杂**；难以提供细粒度并发；单核 | 需要显式同步（锁、信号量），否则数据竞态 |
| 现实代表 | 传统 Apache prefork、CGI | **nginx**、**Node.js**、Tornado | 现代 Apache worker、大多数 Java/C++ 服务 |

- **进程模型的必做两件事**：① 监听进程必须 **reap 僵尸子进程**；② **父进程 `close(connfd)`**、**子进程 `close(listenfd)`**——因为内核为每个 socket 维护**引用计数**，`fork` 后 `refcnt(connfd) = 2`，**只有计数归零连接才真正关闭**，不关就会一直占着。
- **线程模型 = 进程模型 + 共享地址空间**：讲义原文说线程模型"与进程模型非常相似，只是用线程代替进程"。
- **混合方案**：**预线程化服务器**（prethreaded，教材 12.5.5）——主线程预先创建线程池 + 共享的有界缓冲区（生产者-消费者），兼得线程的并行与创建开销的摊薄。Proxy Lab 允许用，也可以用"每连接一线程"（**最简单，推荐起步**）。
- **线程模型的进程视图**：进程 = **线程** + 代码/数据/内核上下文。每个线程**独占**：数据寄存器、条件码、栈指针 SP、程序计数器 PC、**自己的栈**（局部变量）；所有线程**共享**：只读代码、读写数据、运行时堆、共享库、**内核上下文（VM 结构、描述符表、`brk` 指针）**。**关键警告：线程栈"不protected"**——一个线程可以（错误地）读写另一个线程的栈，编译器不会拦你。

#### A.13.2 `pthread_*` 接口表

| 接口 | 原型（`<pthread.h>`） | 语义要点 |
|---|---|---|
| `pthread_create` | `int pthread_create(pthread_t *tid, const pthread_attr_t *attr, void *(*f)(void *), void *arg)` | 创建线程运行 `f(arg)`；`attr = NULL` 用默认属性；**成功返回 0，出错返回错误码（不设 `errno`）** |
| `pthread_join` | `int pthread_join(pthread_t tid, void **retval)` | **等待** `tid` 结束（类似 `waitpid`），并可取回线程返回值；**只有 joinable 线程可以 join** |
| `pthread_detach` | `int pthread_detach(pthread_t tid)` | 把线程设为**分离态**：结束后资源**自动回收**，**不能被 `join`**。也可用属性 `PTHREAD_CREATE_DETACHED` 在创建时设定。**服务器工作线程必须 detach，否则内存泄漏** |
| `pthread_exit` | `void pthread_exit(void *retval)` | 终止**当前线程**（类同 `exit` 对进程）；`retval` 可被 `join` 取到 |
| `pthread_self` | `pthread_t pthread_self(void)` | 返回当前线程 ID（TID），对应 `getpid` |
| `pthread_once` | `int pthread_once(pthread_once_t *once_control, void (*init)(void))` | **保证 `init` 在多线程下恰好执行一次**（用于一次性的全局初始化）；`once_control` 必须初始化为 `PTHREAD_ONCE_INIT` |
| `pthread_mutex_init` / `_destroy` | `int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a)` | 动态初始化 / 销毁互斥量 |
| `pthread_mutex_lock` / `_unlock` | `int pthread_mutex_lock(pthread_mutex_t *m)` | 加锁 / 解锁；**`unlock` 只能由持锁者调用**，否则未定义行为 |
| `pthread_mutex_trylock` | `int pthread_mutex_trylock(pthread_mutex_t *m)` | **非阻塞**加锁：已被占用则立即返回 `EBUSY`（**死锁避免手段之一**） |
| `pthread_cond_init` / `_destroy` | `int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a)` | 初始化 / 销毁条件变量（销毁时**不应有等待者**） |
| `pthread_cond_wait` | `int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m)` | **原子地**释放 `m` 并挂起；被唤醒时**重新获取 `m` 再返回** |
| `pthread_cond_signal` | `int pthread_cond_signal(pthread_cond_t *c)` | 唤醒**至少一个**等待者 |
| `pthread_cond_broadcast` | `int pthread_cond_broadcast(pthread_cond_t *c)` | 唤醒**所有**等待者（屏障/多资源场景必须用这个） |
| `pthread_rwlock_rdlock` / `_wrlock` / `_unlock` | `pthread_rwlock_t` | **读者-写者锁**：`rdlock` 可多个读者共享，`wrlock` 独占；`unlock` 两者通用。**必须由程序员自己判断"哪段代码需要读权限、哪段需要写权限"** |
| `pthread_cancel` | `int pthread_cancel(pthread_t tid)` | 请求取消线程（异步且难以正确使用，**不推荐用于服务器**） |

- **线程 API 与进程 API 的对应（讲义对照表）**：`pthread_create` ↔ `fork`；`pthread_join` ↔ `waitpid`；`pthread_self` ↔ `getpid`；`pthread_exit` ↔ `exit`；`return from thread proc` ↔ `return from main`；`pthread_mutex_lock/unlock` **无精确对应物**。
- **返回值约定**：Pthreads 函数**成功返回 0，失败返回非 0 错误码**（**不使用 `errno`**）。所以 CS:APP 的封装写着 `if ((ret = pthread_create(...)) != 0) unix_error(...)`。
- **`pthread_join` 与 detach 的竞态**：对一个已 detach 的线程 `join`，返回 **`EINVAL(22)`**，且结果**依赖时序**；所以"要么全部可 join、要么全部 detach"，不要混用。
- **`Pthreads` 是标准接口**：约 **60 个**函数，`pthread_*` 前缀，是 C 程序操纵线程的事实标准。

#### A.13.3 `sem_*` 与 `pthread_mutex_*` 对照表

| 语义 | **POSIX 信号量**（`<semaphore.h>`） | **互斥量**（`<pthread.h>`） |
|---|---|---|
| 声明类型 | `sem_t s;` | `pthread_mutex_t m;` |
| 初始化（动态） | `sem_init(&s, pshared, val)`；`pshared = 0` 表示**线程间**共享 | `pthread_mutex_init(&m, NULL)` |
| 初始化（静态） | **无**（必须先 `sem_init`，否则是垃圾字节） | `pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;` |
| 加锁 / P 操作 | `sem_wait(&s)`：$s > 0$ 则 $s \\mathrel{-}= 1$，否则**阻塞** | `pthread_mutex_lock(&m)`：空闲则锁住，否则阻塞重试 |
| 解锁 / V 操作 | `sem_post(&s)`：$s \\mathrel{+}= 1$，唤醒一个等待者 | `pthread_mutex_unlock(&m)`：**只能由持锁者调用** |
| 取值 | `sem_getvalue(&s, &v)`（非标准语义，仅参考） | 无（不透明对象） |
| 销毁 | `sem_destroy(&s)` | `pthread_mutex_destroy(&m)` |
| **初值范围** | **任意 $val \\ge 0$**（计数信号量） | 只能 0/1（**二元**） |
| **调用约束** | P/V **可以乱序**、**可跨线程配对** | `unlock` 必须与 `lock` **同线程配对** |
| 语义差异（核心） | `sem_wait` **可在任何时刻**被任何线程调用（初值为 0 时立即阻塞） | `lock` 只在"空闲"时成功；**持锁者之外的人不能解锁** |
| 典型用途 | 互斥、**计数（资源数）**、**顺序/调度（wait on event）** | **仅**互斥 |
| 相对开销 | **更慢**（讲义实测 27.6 s vs mutex 15 s） | 较快 |

- **关键区别一句话**：**`sem_wait` 无需"持有"关系，`unlock` 必须由持锁者调用。** 这就是信号量能做"计数 + 顺序"而互斥量不能的原因。
- **"信号量的真正价值是 wait on event"**：初值为 0 时它就是一个"等待某事件发生"的原语——`sem_post` 由事件产生方调用，`sem_wait` 由等待方调用，**两者可以是不同线程**。生产者-消费者的 `slots`/`items` 就是这一用法的教科书范例。
- **互斥量语义（讲义定义）**：不透明对象，**要么锁定要么未锁定**；**初始为未锁定**；`lock(m)` 空闲则锁上并返回，否则**等待其变为未锁定后重试**；`unlock(m)` **只能由加锁的代码调用**。它的作用是"对共享变量提供**互斥访问**"，用一条不可违反的不变量把**不安全区域**从状态空间中物理隔离。
- **性能提醒**：讲义明确"**能用 mutex 就别用 semaphore**"——因为信号量更贵，且互斥场景下 `sem_wait/sem_post` 的计数器语义是多余的。反之，**需要计数或顺序时 mutex 做不到**，必须用信号量。

#### A.13.4 `select` / `poll` / `epoll` 与 `fd_set` 宏

| 维度 | **`select`** | **`poll`** | **`epoll`** |
|---|---|---|---|
| 接口 | `int select(int nfds, fd_set *r, fd_set *w, fd_set *e, struct timeval *t)` | `int poll(struct pollfd *fds, nfds_t n, int timeout)` | `epoll_create1` + `epoll_ctl` + `epoll_wait` |
| 描述符上限 | **`FD_SETSIZE`（通常 1024）**——**硬上限** | **无上限**（数组由调用者提供） | 无上限 |
| 数据结构 | 位图 `fd_set`（**固定大小**） | `struct pollfd` **数组**（`fd` + `events` + `revents`） | **内核中的红黑树 + 就绪链表** |
| 就绪事件的表达 | **就地修改** `fd_set`（见下） | 写回 `revents` 字段（**输入/输出分离**，不破坏 `events`） | 返回就绪数组，**不修改注册表** |
| 每次调用开销 | $O(\\text{nfds})$，每次都要把整个集合**从用户态拷到内核态** | $O(n)$，同样要拷贝整个数组 | $O(1)$ 级：`epoll_wait` 只返回就绪项，**无需重复注册** |
| 时间复杂度 | $O(n)$ | $O(n)$ | $O(\\text{就绪数})$（边沿触发需注意处理完） |
| 可扩展性 | **差**（大并发下退化严重） | 中 | **好**（现代高性能服务器的选择） |
| 可移植性 | **最好**（POSIX，无处不在） | 好（POSIX） | **仅 Linux**（BSD 用 `kqueue`，Windows 用 IOCP） |
| 典型用途 | 教材/讲义的事件驱动服务器、教学与小型服务 | 需要突破 1024 上限且要可移植时 | nginx、Redis 等生产级事件驱动服务 |

**`fd_set` 操作宏**：

| 宏 | 作用 |
|---|---|
| `FD_ZERO(&set)` | **清空**集合（**每次调用 `select` 前必须重新执行**） |
| `FD_SET(fd, &set)` | 把 `fd`（及 `fd + 0` 到 `fd + 7` 共 8 个描述符位）加入集合 |
| `FD_CLR(fd, &set)` | 从集合中**移除** `fd` |
| `FD_ISSET(fd, &set)` | **测试** `fd` 是否在（就绪的）集合中——**返回值非 0 表示该 fd 有事件** |
| `FD_SETSIZE` | `fd_set` 能表示的**最大描述符数**（Linux 上 1024） |

**`nfds` 参数 = `maxfd + 1`**：`select` 的 `nfds` 是"要检查的**描述符个数**"，等于**集合中最大 fd 加一**（因为描述符从 0 开始编号）。传小了会漏掉大号 fd，传大了浪费扫描时间。

**⚠️ `select` 会就地修改 `fd_set`，每轮必须重建**：

```
   用户设置：   rset = {listenfd, connfd1, connfd2}       ← 我想监视这三个
        |
        v
   select(nfds, &rset, NULL, NULL, NULL)                    ← 等待
        |
        v
   返回后：     rset = {connfd2}                            ← ⚠️ 只剩下"就绪"的那些！
        |                                                      原来的 listenfd/connfd1 位被清掉了
        v
   必须：FD_ZERO / FD_SET 重新把三个都放回去，才能再调用 select   ← 这就是"每轮重建"
```

- **原因**：`select` 用**同一个** `fd_set` 做"输入 = 监视集合"和"输出 = 就绪集合"（返回时只保留就绪的位）。因此**正确的循环模式**是：每轮先 `FD_ZERO` 再 `FD_SET` 所有关心的 fd（含 `listenfd` 与所有活跃 `connfd`），然后调用 `select`，再用 `FD_ISSET` 逐个检查。
- **对比 `poll`/`epoll` 的优势**：`poll` 的 `revents` 是**独立的输出字段**，所以**不必重建**；`epoll` 更彻底——注册表持久保存在内核里，`epoll_wait` 只是取就绪列表。**这就是"`select` 每轮重建"这件事在现代接口里消失的原因。**
- **事件驱动服务器的主循环（讲义要点）**：① 维护一个**活跃 `connfd` 数组**（含 `listenfd`）；② 每轮确定哪些描述符**有 pending 输入**（`select` 或 `poll`）；③ 若 `listenfd` 有输入 → `accept` 并把新 `connfd` 加入数组；④ 对所有有输入的 `connfd` 逐个服务。**"就绪输入的到来"就是一个事件。**
- **教材实现细节**：讲义明确"基于 `select` 的服务器细节在教材里"——包括"**`listenfd` 和 `connfd` 必须分别用不同的 `fd_set`**"（否则 `listenfd` 就绪后会被随后的 `FD_ISSET` 检查污染）以及"**先处理 `listenfd` 的接受，再处理已连接描述符**"等要点。

#### A.13.5 生产者-消费者（有界缓冲区）

**结构（`sbuf_t`）与三个信号量**：

```
   +-----------------------------+        信号量初值与职责
   |  int *buf;                  |        +--------------------------+
   |  int n;         (槽位数 N)   |        | sem_t mutex;  init = 1   |  互斥：保护 buf/front/rear 的访问
   |  int front;                 |        | sem_t slots;  init = N   |  计数：当前"空槽位"数
   |  int rear;                  |        | sem_t items;  init = 0   |  计数：当前"已填 item"数
   +-----------------------------+        +--------------------------+
          不变量：slots + items = N（= 空槽数 + 已填数 = 总槽数）
```

**正确的操作顺序**：

| 角色 | 顺序 | 代码 |
|---|---|---|
| **生产者** | ① 等空位 ② 拿锁 ③ 存 ④ 放锁 ⑤ **通知"有 item 了"** | `sem_wait(&sp->slots); sem_wait(&sp->mutex); sp->buf[(++sp->rear) % n] = item; sem_post(&sp->mutex); sem_post(&sp->items);` |
| **消费者** | ① 等 item ② 拿锁 ③ 取 ④ 放锁 ⑤ **通知"有空位了"** | `sem_wait(&sp->items); sem_wait(&sp->mutex); item = sp->buf[(++sp->front) % n]; sem_post(&sp->mutex); sem_post(&sp->slots);` |

**铁律（两句话）**：

1. **先等资源（`slots`/`items`），后拿锁（`mutex`）**；
2. **先放锁（`mutex`），后通知（`items`/`slots`）**。

**❌ 死锁写法与机制**：

```c
/* ❌ 错误：先拿 mutex 再等 slots —— 必然死锁 */
sem_wait(&sp->mutex);      /* ① 先拿锁 */
sem_wait(&sp->slots);      /* ② 再等空位 —— 卡在这里！ */
sp->buf[(++sp->rear) % n] = item;
sem_post(&sp->mutex);
sem_post(&sp->items);
```

```
   生产者:  [已持有 mutex] ──等待──> (slots > 0)
                                        ^
                                        | 只有"消费者取出 item"才能增加 slots
                                        |
   消费者:  [等待 mutex] <──需要──── (sem_post(&slots))
                 |
                 +── mutex 被生产者攥着 => 谁也动不了 => 死锁（循环等待 + 持有并等待）
```

- **为什么 `mutex` 不可省**：`slots`/`items` 只保证**计数语义**（"能不能进"），**不保证对 `buf`/`front`/`rear` 的访问互斥**。两个生产者都成功等待到空位后，都会执行 `++sp->rear`——这是**非原子的读-改-写**，可能算出**同一个下标**，导致一个 item 被覆盖、另一个凭空消失。**`slots`/`items` 管"能不能进"，`mutex` 管"进去以后别撞车"——职责正交，缺一不可。**
- **为什么不变量是 `slots + items = N`**：每次生产 `slots` 减 1、`items` 加 1；每次消费反之。故任意时刻"空槽数 + 已填数 = 总槽数"。这个不变量是**验证实现正确性的利器**（可以在每次操作后断言它）。
- **`sem_init` 的第三个参数就是初值**：`sem_init(&sp->mutex, 0, 1)`、`sem_init(&sp->slots, 0, n)`、`sem_init(&sp->items, 0, 0)`——`pshared = 0` 表示**线程间**共享。
- **一个易漏的细节**：忘记 `sem_init` 会让 `sem_t` 是**垃圾字节**，通常表现为立刻死锁或随机唤醒（UB）。
- **条件变量版**：用 `count` + `not_empty`/`not_full` 两个条件变量替代 `items`/`slots`，**必须用 `while` 包裹条件判断**（见 A.13.7）。

#### A.13.6 读者-写者问题

**问题定义**：读者线程**只读**对象，写者线程**修改**对象。约束是：**写者必须独占访问**；**无限多个读者可以同时访问**。现实中频繁出现——**在线订票系统**、**多线程缓存 Web 代理**（Proxy Lab 的缓存就是它）。

**第一类：读者优先（readers preference）**——写者可能**饥饿**。

| 元素 | 作用 |
|---|---|
| `sem_t mutex`（初值 1） | **保护 `readcnt`** 这个共享计数器 |
| `sem_t w`（初值 1） | **写者锁**，也代表"读权限" |
| `int readcnt = 0` | 当前正在读的读者数 |

```c
/* 读者进入 */                              /* 写者进入 */
sem_wait(&mutex);                           sem_wait(&w);
readcnt++;                                  /* 临界区：写 */
if (readcnt == 1)                           sem_post(&w);
    sem_wait(&w);   /* 第一个读者拦住写者 */
sem_post(&mutex);
/* 临界区：读 */
sem_wait(&mutex);
readcnt--;
if (readcnt == 0)
    sem_post(&w);   /* 最后一个读者放行写者 */
sem_post(&mutex);
```

*   **`readcnt` 必须用 `mutex` 保护**：否则两个读者同时 `readcnt++` 会丢失更新，可能出现"`readcnt` 从 1 变 2 但只有一次 `sem_wait(&w)`"或反之——后果是**写者与读者同时进入临界区**（正确性崩溃）或**写者锁永久不放**（死锁）。
*   **饥饿的机制**：只要读者源源不断到达，`readcnt` 永远大于 0，`w` 永远不被释放 → 写者**永远等不到**。这是"读者优先"这个名字的由来，也是它的缺陷。
*   **Proxy Lab 的注意点**：`pthread_rwlock_t` 的具体策略由实现决定，故讲义把"不必严格 LRU"当作实现多读者并发的腾挪空间。

**第二类：写者优先（writers preference）**——读者可能**饥饿**。经典做法是加一个**旋转门（turnstile）**信号量：

| 元素 | 作用 |
|---|---|
| `sem_t w`（初值 1） | 真正的写者锁 |
| `sem_t turnstile`（初值 1） | **旋转门**：写者一到就"关门"挡住后续读者，写完再"开门" |
| `sem_t mutex` + `readcnt` | 同读者优先 |

```c
/* 写者 */                                   /* 读者 */
sem_wait(&turnstile);   /* 关门：挡住新读者 */  sem_wait(&turnstile);   /* 通过旋转门 */
sem_wait(&w);           /* 拿写者锁 */          sem_post(&turnstile);   /* 立刻放行（不长期持有）*/
/* 临界区：写 */                                sem_wait(&mutex);
sem_post(&w);                                   readcnt++;
sem_post(&turnstile);   /* 开门 */              if (readcnt == 1) sem_wait(&w);
                                                sem_post(&mutex);
                                                /* 临界区：读 */
                                                sem_wait(&mutex);
                                                readcnt--;
                                                if (readcnt == 0) sem_post(&w);
                                                sem_post(&mutex);
```

*   **`turnstile` 的关键**：读者**只短暂持有**它（进门后立刻 `sem_post`），所以**读者之间仍然可以并发**——这正是它优于"在读者入口直接 `sem_wait(&w)`"的地方。
*   **⚠️ 常见的错误写法**：在读者入口直接 `sem_wait(&w)`、出口 `sem_post(&w)`。它**能挡住新读者**，但**把读者之间也串行化了**（每个读者都独占 `w`），彻底丢掉了"多读者并发"这一核心优势。**能跑，但对性能有害。**

**饥饿（starvation）与公平（fairness）**：

| 概念 | 说明 |
|---|---|
| **饥饿** | 一个线程"在**不可接受地长**的时间内没有取得任何进展" |
| 与死锁的区别 | **饥饿可能最终解脱**；死锁则**永远**卡住 |
| "不可接受地长" | **取决于应用**——没有客观阈值 |
| **公平（fair）** | 保证**无饥饿**的算法。公平读写锁 = **每个等待者按先来先服务（FCFS）顺序拿到锁**（多个读者可同时拿到） |
| 公平的代价 | 实现更复杂；且**可能让所有线程都比不公平系统更慢**（典型是 **lock convoy problem / 锁 convoy**） |

#### A.13.7 条件变量要点

| 要点 | 说明 |
|---|---|
| **本质** | 条件变量让线程**在某个条件为假时挂起**，并在条件可能变真时被唤醒——解决"轮询忙等浪费 CPU"与"用信号量硬凑、语义不匹配"的问题 |
| **自身不提供互斥** | 它**总是与一把互斥锁配对使用**。锁保护**条件本身**（谓词），条件变量负责**等待/唤醒** |
| **`pthread_cond_wait(c, m)` 的原子语义** | **必须持有 `m` 才能调用**。它会**原子地**：① **释放 `m`**；② 把线程加入 `c` 的等待队列并挂起；③ 被唤醒后**重新获取 `m`**，然后才返回。这三步**不可分割**——如果"释放锁"和"挂起"之间有窗口，就会丢失唤醒（lost wakeup），这正是它存在的全部理由 |
| **必须用 `while` 而非 `if`** | 被唤醒后**必须重新检查条件**。原因有三：① **虚假唤醒（spurious wakeup）**——POSIX 允许 `cond_wait` 无理由返回；② **唤醒被别的线程抢走**（"惊群"后竞争者先拿到锁并消耗了资源）；③ `broadcast` 唤醒所有等待者但只有部分能继续。**`if` 版本会在条件实际为假时继续执行，导致缓冲区溢出/下溢** |
| `signal` vs `broadcast` | `signal` 只唤醒**至少一个**等待者（单一资源）；`broadcast` 唤醒**全部**（条件对多个等待者同时为真时，如**屏障**、状态整体翻转） |
| 使用模板 | 等待方：`pthread_mutex_lock(&m); while (!cond) pthread_cond_wait(&c, &m); /* 用资源 */ pthread_mutex_unlock(&m);` 通知方：`pthread_mutex_lock(&m); /* 改状态使 cond 为真 */ pthread_cond_signal(&c); pthread_mutex_unlock(&m);` |
| **`while` 的正确写法** | `while (count == 0) pthread_cond_wait(&not_empty, &mutex);` ——谓词写在 `while` 条件里，而非 `if`，也**不是** `while(1) { if (...) break; wait(); }` 之外的错误变体 |

- **讲义补充（L21 末尾）**：条件变量在 L21/L22 讲义里只作为"**Bonus: 自行查阅**"出现；教材（CS:APP3e 12.5.4 与 12.7）才正式展开。**SFS Lab 与 Proxy Lab 中条件变量与信号量都可用**，但语义不同，不要混用。
- **屏障（barrier）的条件变量实现**：每线程到达时 `count++`；若 `count < N` 则 `wait`，否则 `count = 0` 并 **`broadcast`**——这里必须用 `broadcast`，因为**所有**等待者都该被放行。

#### A.13.8 线程安全四类函数分类表

| 类别 | 根本原因 | 典型例子 | 修复方法 | 代价 |
|---|---|---|---|---|
| **第 1 类：不保护共享变量** | 函数访问了**共享全局/静态变量**却没同步 | 自己写的 `cnt++` 计数器；**自实现的 `malloc`/`free`**（无内部锁） | 在函数**首尾**加 `pthread_mutex_lock/unlock` | 加锁**降性能**；多把锁可能引入**死锁** |
| **第 2 类：依赖跨调用保持的状态** | 函数**跨多次调用**保持状态（静态变量） | `rand`（依赖全局 `next`）、`srand`、`strtok` | **把状态改为参数**由调用者传入（`rand_r(int *nextp)`） | **必须改 API**；调用者负责分配状态空间 |
| **第 3 类：返回指向静态变量的指针** | 函数把结果写进**共享的静态缓冲区**并返回其指针 | **`ctime`**、`asctime`、`localtime`、`gethostbyname`、`inet_ntoa`、`itoa` | **lock-and-copy**：锁内调用并**把结果拷贝到调用者私有存储**；或改用 `strftime`/`getaddrinfo` 这类安全替代 | 需改签名；调用者要知道缓冲多大 |
| **第 4 类：调用了上面三类中任一个** | **传递性**——函数自身没毛病，但它调用的函数不安全 | 任何内部用了 `rand`/`strtok` 的函数 | 只调用**线程安全的版本**（`_r` 后缀） | 可能触发**连锁 API 改动** |

- **加锁只能修好第 1 类**。第 2、3 类**加锁治不好**：给 `rand` 内部加锁，两个线程各调 100 次的序列仍然**不同于**单线程调 200 次——锁只保证"不崩"，不保证"可重现"。第 3 类更微妙：`char *s = ctime(&t);` 拿到指针后**锁已释放**，从解锁到使用 `s` 之间存在窗口，别的线程任何一次 `ctime` 都会**覆盖**这块静态缓冲区。
- **可重入（reentrant）与线程安全的关系**：函数**可重入**，当且仅当它被多线程调用时**不访问任何共享变量**。因此**可重入 ⇒ 线程安全**（它压根没有共享状态，**不需要任何同步操作**）；**反之不成立**——加了互斥锁的 `rand` 是线程安全的，却仍然修改全局 `next`，**不可重入**。
- **显式 vs 隐式可重入**：**显式可重入**（explicitly reentrant）= 函数内部**完全不碰**共享变量；**隐式可重入**（implicitly reentrant）= 把共享变量全改成局部变量后，**再通过指针参数由调用者传入**（如 `rand_r(&seed)`）——它的可重入性**依赖调用者是否传入了共享指针**。
- **ISO C 的现状**：绝大多数标准库函数**是线程安全的**（`malloc`、`free`、`printf`、`scanf` 内部有锁）；例外是 `strtok`、`rand`、`asctime`。`printf`/`fprintf`/`puts` 内部用 **lock-and-copy** 保护输出流，避免多线程输出交错；`snprintf` 内部还会调用 `malloc` 取暂存空间。
- **老式 Unix C 库**则大量不安全，通常有 `_r` 后缀的安全替代品。

#### A.13.9 死锁与避免规则

**死锁定义**：程序**死锁**，当一组线程中**每一个都在等待一个只能由组内其他线程触发的事件**——最典型是"**相互等待对方持有的锁**"。

```
   线程 1                     线程 2
   lock(mA)                   lock(mB)          <-- 各自先拿一把
   lock(mB)  <-- 等线程2        lock(mA)  <-- 等线程1
   ...                        ...
   unlock(mB)                 unlock(mA)
   unlock(mA)                 unlock(mB)

   => 循环等待：T1 等 mB（T2 持有），T2 等 mA（T1 持有）=> 永久卡住
```

**进度图（progress graph）几何解释（讲义核心工具）**：

*   横轴 = 线程 1 的进度，纵轴 = 线程 2 的进度；每个**指令**是一个状态，轨迹是一串**状态转换**。
*   **禁止区域（forbidden region）**：由**互斥不变量**定义的矩形——"两个线程的临界区不能重叠"。
*   **不安全区域（unsafe region）**：禁止区域的并集；**轨迹一旦进入，就必然违反互斥**。
*   **死锁区域（deadlock region）**：不安全区域的**子集**，形状像"**左上角的矩形**"——一旦轨迹进入，**既不能向右也不能向上移动**（两个方向都被对方阻塞）。
*   轨迹可以**绕过**禁止区域的**上边或右边**（"always possible to move up or move right"）——这就是**正确的加锁顺序**能让轨迹贴着不安全区域边缘"溜过去"的几何含义。
*   **死锁状态**：存在一组线程，每个都在等待一个只能由组内另一个线程触发的事件。
*   **⚠️ 重要细节**：**加锁顺序不一致会导致死锁，但解锁顺序不一致完全无所谓**（讲义原文："Inconsistent unlock order does not matter"）——因为解锁不会阻塞。

| 死锁避免规则 | 说明 |
|---|---|
| **① 加锁顺序全局一致** | 若多个线程都要拿多把锁，**所有线程必须按同一个全序**获取（如按锁的地址排序）。这是**最根本**的一条：它把"循环等待"从状态空间中**彻底删除** |
| **② 用 `pthread_mutex_trylock`** | 需要多把锁时改用非阻塞版：拿到一把后，若第二把拿不到就**释放已持有的锁并重试**（打破"持有并等待"），避免永久阻塞 |
| **③ 用粗粒度锁** | 把多把锁合并成**一把大锁**（减少锁数量 → 减少能形成的等待环）。**代价是并发度下降** |
| **④ 避免嵌套加锁** | 尽量让临界区里**只持有一把锁**，从根本上消除"持有多锁"这一死锁的必要条件 |
| ⑤ 固定"资源 → 锁"顺序 | 生产者-消费者里的铁律"**先 `sem_wait(&slots)` 再 `sem_wait(&mutex)`**"就是这个规则的具体化（见 A.13.5） |
| ⑥ 别在持锁时做慢操作 | 持锁期间不做 I/O、不调 `malloc`、不睡眠——否则加长"持有并等待"的窗口，把**饥饿**放大成**死锁** |

- **死锁的四个必要条件（经典理论，讲义未列，补充）**：互斥、持有并等待、非抢占、循环等待。**打破任一条即可防死锁**，实践中主要打第 3、4 条。
- **饥饿 vs 死锁**：**死锁**永不解脱；**饥饿**可能最终解脱（见 A.13.6）。
- **lock convoy problem**：追求公平（FCFS）时，锁在线程间"传递"导致频繁上下文切换，**所有线程都比不公平系统更慢**。

#### A.13.10 伪共享（false sharing）与填充

**定义**：**不同线程访问同一个缓存块中的"不同字节"**——虽然逻辑上不共享数据，但**缓存一致性维护是以缓存块为单位**的，所以这些线程仍会相互冲突。

```
   psum 数组（每个元素 8 字节，缓存块 = 64 字节 = 8 个元素）
   +--------------+--------------+--------------+
   |  块 m        |  块 m+1      |  ...
   | psum[0..7]   | psum[8..15]  |              |
   +--------------+--------------+--------------+
     ^      ^
     |      |
   线程 0  线程 1     <-- 逻辑上互不相干，物理上却在同一块里"打架"
   写 psum[0] 写 psum[8]
```

| 概念 | 定义 |
|---|---|
| **真共享**（true sharing） | **同一个字节**被多个线程读写——必须用同步（锁/原子）保证正确性 |
| **伪共享**（false sharing） | **不同字节、同一缓存块**被不同线程写。逻辑上无害，**性能上极坏**——"线程们会为了拿到这个块而互相争抢" |
| 后果 | 缓存块在核间**反复弹跳（ping-pong）**：每次写都要**独占**该块，其他核的副本作废 → 大量一致性流量，加速比崩塌 |

**讲义实测数据（必记）**：

| 实验 | 相邻累加器 | 间隔开的累加器 | 结论 |
|---|---|---|---|
| 内存累加（`psum[i]` 写在共享数组里） | 加速比 **5.0×** | 加速比 **13.3×**（**唯一超过 8 的观测值**） | 间隔放置提速 **2.8×**（13.3 / 5.0 ≈ 2.66，讲义表述为"最好间隔版本比最好相邻版本好 2.8 倍"） |
| 寄存器累加（每线程局部 `sum`，最后写一次） | — | 加速比 **7.5×**，比最快的内存累加**快 2 倍** | **用寄存器永远最快** |

*   **为什么间隔能解决**：`spacing` 参数以**缓存块大小**为单位，让 `psum[myid*spacing]` 落在**不同缓存块**里，于是不再伪共享。
*   **"证明缓存块大小 = 64"**：讲义由实验推得——**8 字节的值**，"间隔再增大到超过 8 个元素就没有额外收益"，说明块边界在 **8 个 8 字节值 = 64 字节**处。这就是用性能实验**反推硬件参数**的经典手法。
*   **填充（padding）到 64 字节**：把需要每线程独占的计数器/状态**补齐或对齐到 64 字节**（`__attribute__((aligned(64)))` 或手工填 `char pad[64 - sizeof(x)]`），确保不同线程的变量落在**不同缓存块**。

**三条教训（讲义原话）**：

1. **共享内存很昂贵**——要同时警惕**真共享**与**伪共享**；
2. **尽量用寄存器**（记得 Cache Lab 的教训）；
3. **尽量用本地缓存**（local cache）。

- **`Beware the speedup metric!`（警惕加速比指标！）**：讲义特意提示，**加速比容易被误读**——同一个程序改一下数据布局（相邻 → 间隔），加速比就从 5× 变成 13.3×；而换成一维局部累加又能到 7.5×。所以**报加速比必须同时说明测量条件与基线**。

#### A.13.11 竞态、进度图与不安全区域

**竞态（race）定义**：**当程序的正确性依赖于一个或多个线程的"不可控的相对执行顺序（时序）"时，就发生了竞态**。它的本质是"**共享 + 未同步**"。

| 竞态形态 | 例子 | 修复手段 |
|---|---|---|
| **数据竞态**（多个线程读写同一变量） | `cnt++` 在两个线程里各跑 N 次，结果小于 $2N$（**丢失更新**）；讲义"数到 20000"实验 | 互斥量/信号量保护（第 1 类线程不安全） |
| **传递参数竞态**（`pthread_create` 传 `&i` 而非值） | 主线程在循环里 `pthread_create(&tid, NULL, f, &i)`，工作线程读到的 `i` 已经是后来的值 | **传值**（`(void *)(long)i`）或为每个线程单独分配参数 |
| **跨调用的状态竞态** | `rand`/`strtok` 的静态状态 | 改用 `_r` 版本（第 2 类） |
| **返回静态缓冲区竞态** | `ctime` 返回指针，锁已释放 | lock-and-copy（第 3 类） |
| **生命周期竞态** | `pthread_detach` 与 `pthread_join` 竞争 → `join` 返回 `EINVAL(22)`；"**join WINS / detach WINS**"两种结局都能复现 | 统一约定：**要么全部可 join、要么全部 detach** |
| **`fork` 与信号处理程序竞态** | Shell 中"子进程被 `sigchld` 回收并从作业表删除，父进程却还没 `addjob`" | 用 `sigprocmask` **屏蔽 `SIGCHLD`**，`addjob` 后再解除（讲义 `procmask2.c`） |
| **非线程导致的竞态** | 讲义明确指出"**并非所有竞态都涉及线程**"——信号处理程序与主程序之间就有竞态 | 屏蔽信号 / `sigsuspend` |
| **无法用互斥解决的竞态** | 有些竞态"**不能靠互斥解决**"，需要用**拷贝数据**或**信号量**来消除（讲义举了具体例子） | 副本/信号量 |

**进度图（progress graph）术语表**：

| 术语 | 含义 |
|---|---|
| **轨迹（trajectory）** | 状态空间中的一条路径，表示两个线程**实际**的相对推进序列 |
| **临界区（critical section）** | 访问共享变量的代码段，必须互斥执行 |
| **禁止区域（forbidden region）** | 由互斥不变量定义的、**轨迹绝不可进入**的状态集合（"两个线程同时在临界区"） |
| **不安全区域（unsafe region）** | 禁止区域的并集；进入即违反互斥 |
| **安全轨迹（safe trajectory）** | **完全绕开**不安全区域的轨迹——正确的同步应该**只允许**这种轨迹 |
| **死锁区域（deadlock region）** | 不安全区域中"进得去出不来"的子集（左上角的矩形） |
| **互斥不变量（mutex invariant）** | 定义禁止区域的数学条件（如 $s \\ge 0$ 或"最多一个线程在临界区"） |

- **不变量（invariant）的力量**：加锁不是"防君子"的礼貌，而是**用一条不可违反的不变量，把不安全区域从状态空间中物理隔离**。这是 Lecture 22 的核心结论，也是理解一切同步原语的统一视角。
- **"任意交错"的正确心智模型**：不要试图枚举时序，**要画进度图**——用几何把"所有可能的交错"一次性覆盖，然后证明你的同步**把不安全区域变成了不可达**。

#### A.13.12 并发与同步常用命令

| 命令 | 作用 |
|---|---|
| `valgrind --tool=helgrind ./prog` | **检测数据竞态与锁顺序问题**（讲义实测用它发现了 `ctime` 内部的竞态：`Possible data race during read of size 1 ... __tz_convert ← ctime`） |
| `valgrind --tool=drd ./prog` | 另一款竞态检测器（与 helgrind 侧重不同，可互为补充） |
| `gcc -fsanitize=thread -g ./prog` | **ThreadSanitizer（TSan）**：编译期插桩的竞态检测，比 valgrind 快得多 |
| `gcc -fsanitize=address -g ./prog` | **AddressSanitizer**：检测越界与 use-after-free（多线程共享数据被提前释放时尤其有用） |
| `perf stat -e cache-misses,LLC-load-misses ./prog` | 量化**伪共享**造成的缓存一致性流量 |
| `perf c2c record ./prog` && `perf c2c report` | **直接定位伪共享的热点缓存行**（哪两个变量/线程在抢同一行） |
| `ps -o pid,nlwp,cmd -p <pid>` | `nlwp` = 线程数——确认线程真的创建了 |
| `ps -L -p <pid>` | 逐线程列出（LWP、状态、CPU 占用）——**排查"某个线程卡在锁上"** |
| `top -H -p <pid>` | 按**线程**显示 CPU 占用 |
| `gdb -p <pid>` + `info threads` / `thread apply all bt` | **死锁排查利器**：一次看清所有线程都卡在哪个 `pthread_mutex_lock` / `sem_wait` 上 |
| `gdb` + `set scheduler-locking on` | 单步调试时锁定调度，避免线程乱跑 |
| `strace -f -e trace=clone,futex ./prog` | 观察线程创建与 `futex` 系统调用（锁的底层实现） |
| `ltrace -f ./prog` | 跟踪库函数调用（可看到 `pthread_mutex_lock` 的进出与返回值） |
| `taskset -c 0,1 ./perf_test` | **限制可用核心数**——复现"核心数变化导致加速比异常"的问题 |

---

### A.14 性能度量速查

> 对应讲次：Lecture 15（`F25-15-optimization.txt`）、Lecture 24（`F25-24-parallelism.txt`）；
> 教材 CS:APP3e 第 5 章、第 12.6 节；关联 **L4 Cache Lab Part B**、**L5 Malloc Lab**、**Arch Lab**。

#### A.14.1 CPE 定义与测量要点

**定义**：**CPE（Cycles Per Element，每元素周期数）** = 平均每处理一个元素所花的**时钟周期数**。

$$\text{Cycles} = \text{CPE} \times n + \text{Overhead}$$

*   **CPE 是拟合直线的斜率**，`Overhead` 是截距。讲义给的示意图里 `psum1` 斜率 **9.0**、`psum2` 斜率 **6.0**。
*   **为什么不用秒/毫秒**：绝对时间会被**时钟频率**污染——同一份代码在 2.0 GHz 老机器上跑 10 ms、在 4.0 GHz 新机器上跑 5 ms，**不代表代码变好了**。CPE 把频率因子除掉，剩下的才是代码与微架构的真实效率。
*   **CPE 的倒数就是硬件吞吐量**：CPE = 1 表示"每周期做一个元素"；CPE = 0.5 表示"**每周期做两个**"——这只有在处理器有**多个功能单元并行工作**时才可能。

**测量三条纪律**：

| # | 纪律 | 原因 |
|---|---|---|
| ① | **$n$ 必须足够大** | 让截距 `Overhead` 可忽略，斜率才测得准（否则测到的主要是启动成本） |
| ② | **每个版本重复多次，取最小值（或中位数）** | 调度、中断、频率波动**只会让结果变慢、绝不会变快**，所以**最小值最接近真实能力** |
| ③ | **用 `clock_gettime(CLOCK_MONOTONIC, ...)` 计时**（或 `clock()`），并**扣除循环外的固定开销** | `CLOCK_MONOTONIC` 不受系统时间调整影响；讲义的 `fsecs` 工具把这些纪律固化成可复用的计时框架 |
| ④ | （建议）**标定实际主频** | 标称 **3.7 GHz** 是最大加速频率，**持续负载下只有 3.07 GHz**；不标定会让 CPE **整体偏差 20%** |

```c
/* 计时骨架：讲义 fsecs 的思路 */
#include <time.h>
#include <stdio.h>

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);          /* 单调时钟，不受校时影响 */
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(void)
{
    const long n = 1L << 20;                       /* n 足够大，压过 Overhead */
    double best = 1e30;
    for (int rep = 0; rep < 15; rep++) {           /* 重复多次 */
        double t0 = now_sec();
        volatile long s = 0;
        for (long i = 0; i < n; i++) s += i;       /* 被测内核 */
        double dt = now_sec() - t0;
        if (dt < best) best = dt;                  /* 取最小值 */
    }
    /* CPE = 周期数 / n；周期数 = 时间 * 实际主频（需自己标定） */
    printf("best time = %.6f s for n = %ld\n", best, n);
    return 0;
}
```

- **报告规范**：讲义/笔记的实践是**每版本重复 15 次，报告 `CPE_min` 与 `CPE_med`**。
- **`-O` 级别必须写明**：同一份代码在 `-O1`/`-O2`/`-O3 -mavx2` 下 CPE 差异巨大（见 A.14.2 表格）。

#### A.14.2 优化阶梯表

**讲义原始数据（CPE，Haswell）**：

| 方法 | int add | int mult | double add | double mult |
|---|---|---|---|---|
| `combine1` 未优化 | 22.68 | 20.02 | 19.98 | 20.18 |
| `combine1 -O1` | 10.12 | 10.12 | 10.17 | 11.14 |
| `combine1 -O3` | 4.5 | 4.5 | 6 | 7.8 |
| **`combine4`**（局部变量累加） | **1.27** | 3.01 | 3.01 | 5.01 |
| **循环展开 + 多累加器（unroll）** | **0.81** | **1.51** | **1.51** | **2.51** |

**优化阶梯（每一步消除什么、收益多少）**：

| 步 | 优化 | 消除的开销 | 典型收益 |
|---|---|---|---|
| **0** | 基线 `combine1`（每轮调 `vec_length` + `get_vec_element`） | — | CPE ≈ **20**（最慢） |
| **1** | **消除循环低效**：把 `vec_length(v)` **提到循环外**；消除每次迭代的**边界检查** | 每次迭代重复的**过程调用**与长度计算 | `-O1` 后 CPE ≈ **10**（**减半**） |
| **2** | **减少过程调用**：不要用 `get_vec_element` 取值，直接访问数组（或让编译器内联） | 每次迭代的 `call`/`ret`、参数传递、栈帧开销 | 与第 1 步合并见效 |
| **3** | **消除内存引用**：**用局部变量累积**（`data_t val = IDENT; ... val = val OP d[i]; ... *dest = val;`） | 每轮对 `*dest` 的**读-改-写内存访问**（延迟远高于寄存器） | CPE 从 10 降到 **1.27**（int add）——**十倍跃降** |
| **4** | **循环展开（loop unrolling）**：一次迭代处理 $k$ 个元素（如 4 个），摊薄分支与循环控制开销 | 循环控制（比较、分支）的**每元素摊分** | CPE 再降（int add 1.27 → **0.81**） |
| **5** | **多个累加器（multiple accumulators）**：用 $k$ 个独立的局部累加器打破**串行依赖链** | 关键路径上的**延迟串行化**（见 A.14.4） | 突破**延迟界限**，逼近**吞吐量界限** |
| **6** | **SIMD / 向量化**（`-mavx2` 等）与**重新结合（reassociation）** | 每指令处理更多元素 | 讲义"**每周期 4–16 个元素**" |

- **讲义总结**："编译器优化是**容易获得的收益**：**20 CPE 降到 3–5 CPE**；而配合仔细的手工调优与计算机体系结构知识，可以做到**每周期 4–16 个元素**——最新的编译器正在缩小这个差距。"

**为什么第 3 步能降十倍（机制透视）**：逐条数一数：v1 每轮 **2 次函数调用 + 1 次栈访问 + 1 次内存读改写**；v2 每轮 **1 条 `addq` 内存操作数 + 指针递增 + 比较 + 分支**。函数调用要压栈、跳转、返回，而**内存读改写的延迟远高于寄存器操作**。注意 **`-O1` 已足够让 GCC 把局部变量放进寄存器**——因为它的地址**从未逃逸（never escapes）**。

- **"做什么"与"不做什么"**：讲义强调优化要有**优先级**——① 好的编译器与正确的编译选项；② **不做任何次优的事**（警惕**隐藏的算法级低效**、写**编译器友好**的代码、警惕**优化阻碍**）；③ **盯紧最内层循环**（绝大部分工作在这里）；④ **针对机器调优**（利用指令级并行、**避免不可预测的分支**）。

#### A.14.3 内存别名（aliasing）与 `restrict`

**问题**：编译器无法判断两个**指针**是否指向同一块内存，于是不敢做某些优化（不敢把"先写 A 再读 B"重排、不敢把跨迭代的读提到循环外）。

```c
/* 讲义原例：求和矩阵每一行 */
void sum_rows2(double *a, double *b, long n) {
    long i, j;
    for (i = 0; i < n; i++) {
        double val = 0;
        for (j = 0; j < n; j++)
            val += a[i*n + j];        /* 累加在寄存器里 */
        b[i] = val;                   /* 写 b —— 编译器担心 b 与 a 别名！ */
    }
}
```

| 手段 | 做法 | 效果 |
|---|---|---|
| **`restrict` 关键字** | `void sum_rows2(double *restrict a, double *restrict b, long n)` | **告诉编译器"这是指向该内存位置的唯一指针"**，于是可以放心优化。讲义明确把 `restrict` 列为"消除别名惩罚"的手段 |
| **局部变量累积** | 用 `val` 累加而不是直接写 `b[i]` | 把 n 次内存写变成 **1 次**，同时降低别名风险 |
| **`-fno-strict-aliasing` / `-O3`** | 编译选项 | 反过来：若代码确实依赖别名行为，需要**关掉**基于 strict aliasing 的优化 |

- **`restrict` 是"程序员的承诺"**：一旦你**撒谎**（两块内存实际重叠），行为**未定义**（UB）。这是典型的"以正确性换性能"的契约。
- **优化阻碍（optimization blockers）两大类**：**过程调用** 与 **内存引用**。很多情况下二者**互相纠缠**——`get_vec_element` 是过程调用，`*dest` 是内存引用；`strlen(s)` 在循环条件里则同时是"过程调用**加上**无法外提的内存依赖"，导致**算法级**的二次复杂度（见下）。

**过程调用阻碍优化的经典案例（讲义原例）**：

```c
/* O(n^2)：每轮迭代都重新调 strlen(s) */
void lower_quadratic(char *s) {
    size_t i;
    for (i = 0; i < strlen(s); i++)        /* ⚠️ 每轮都重算字符串长度 */
        if (s[i] >= 'A' && s[i] <= 'Z')
            s[i] += 'a' - 'A';
}

/* O(n)：把长度提到循环外 */
void lower_linear(char *s) {
    size_t i, n = strlen(s);
    for (i = 0; i < n; i++)
        if (s[i] >= 'A' && s[i] <= 'Z')
            s[i] += 'a' - 'A';
}
```

- **编译器为什么不敢自动外提 `strlen(s)`**：循环体内 `s[i] = ...` **修改了字符串**，编译器无法证明"长度不会变"，因此**不能**把 `strlen` 当成循环不变量移出去。**这种"算法级低效"编译器救不了你，只能靠人**。

#### A.14.4 延迟界限 vs 吞吐量界限

**两个界限的公式**：

$$\text{CPE}_{\text{lat}} = \frac{L}{k}
\qquad\qquad
\text{CPE}_{\text{thr}} = T = \frac{1/C}
\qquad\qquad
\text{CPE} \approx \max\!\left(\frac{L}{k},\ T\right)$$

| 符号 | 含义 |
|---|---|
| $L$ | 该运算的**延迟（latency）**——一条指令从操作数就绪到结果可用所需周期数 |
| $k$ | **累加器个数**（独立的功能单元通道数） |
| $T$ | **发射时间（issue time / throughput）**——连续两条同类指令之间至少要隔的周期数 |
| $C$ | 该类功能单元的**个数**（并发度），故 $T = 1/C$ |
| $\\text{CPE}_{\\text{lat}} = L/k$ | **延迟界限**：受**关键路径串行延迟**限制 |
| $\\text{CPE}_{\\text{thr}} = T$ | **吞吐量界限**：受**功能单元数量**限制 |

**为什么只有一个累加器时 CPE = L**：

```
   x = x OP d[i];   x = x OP d[i+1];   x = x OP d[i+2];   ...
   |___________|         ^
        必须等上一步算完    |
        ==> 纯串行依赖链 ==> CPE = L（与功能单元个数无关！）
```

**多累加器如何突破延迟界限**：

```
   累加器 1:  x0 = x0 OP d[i]    x0 = x0 OP d[i+2]   ...   \
   累加器 2:  x1 = x1 OP d[i+1]  x1 = x1 OP d[i+3]   ...    } k 条独立的依赖链
                                                            /   ==> 关键路径缩短 k 倍
   最后:      *dest = x0 OP x1  (再串行合并 k 个累加器)
```

**讲义给出的具体数值分析（$L = 5$，$T = 1$）**：

| 累加器个数 $k$ | $\\text{CPE} = \\max(L/k, T)$ | 处于哪个界限 |
|---|---|---|
| 1 | $\\max(5/1, 1) = 5$ | **纯延迟受限** |
| 2 | $\\max(5/2, 1) = 2.5$ | 延迟受限（改善中） |
| 4 | $\\max(5/4, 1) = 1.25$ | 接近吞吐界限 |
| **5** | $\\max(5/5, 1) = \\mathbf{1}$ | **到达吞吐量界限** |
| ≥ 5 | $\\max(5/k, 1) = 1$ | **再加累加器毫无收益** |

*   **核心结论**：**累加器个数到达 $k_{\\text{opt}} = L/T = L \\cdot C$ 之后就封顶了**——此时受限的不再是延迟，而是**功能单元的数量**。继续增加累加器只会造成**寄存器压力**（见 A.14.5），得不偿失。
*   **循环展开（unrolling）与多累加器的分工**：**循环展开**主要摊薄**循环控制开销**（分支、比较、指针递增）并**为调度器提供更多可重排指令**；**多累加器**专门用于**打破依赖链、降低关键路径**。二者常常一起用，且**必须先展开才能塞进多个累加器**。
*   **调度（scheduling）的作用**：讲义明确说"重排指令让 CPU 更容易**让所有功能单元保持忙碌**"，例如"**把所有 load 提到展开循环的顶部**"——讲义还顺带点出"现在也许更容易理解**为什么我们需要很多寄存器**"（load 提前需要更多寄存器同时存活跃值）。
*   **展开后的真实汇编（讲义原例，整数 2 路展开）**：`addq (%rdx), %rcx; addq $16, %rdx; addq -8(%rdx), %rdi; cmpq %r8, %rdx; jne .L3` —— **两条 `addq` 之间没有依赖**（分别累积到 `%rcx` 与 `%rdi`），所以可以每周期发射多条，这就是"**每周期多条指令（Multiple instructions every cycle!）**"的来源。

#### A.14.5 寄存器溢出（register spilling）

| 项目 | 说明 |
|---|---|
| **定义** | 当活跃变量（live values）的数量**超过可用的物理寄存器数**时，编译器只能把一些变量**溢出（spill）到栈上**，用时再重新载入 |
| **触发原因** | **过度展开 + 过多累加器**：展开 $k$ 路并配 $k$ 个累加器，就需要 $k$ 个累加器寄存器 + $k$ 个载入暂存寄存器 + 指针 + 边界，**寄存器很快用完** |
| **代价** | 每次溢出/重载都是**内存访问**——正是优化阶梯第 3 步刚消除的东西。结果：**CPE 不降反升**，可能出现"展开越多越慢"的**反常现象** |
| **观测方法** | ① 看 `gcc -S` 的汇编里是否出现额外的 `movq ... -N(%rbp)` / `movq -N(%rbp), ...` 进出栈帧；② 看 `perf stat -e ...`；③ **实测 CPE 随展开因子 $k$ 的曲线**——通常先降后升，拐点就是寄存器上限 |
| **应对** | **找到最优展开因子**（经验值：x86-64 上 4–8 路较常见，但要实测）；减少同时活跃的变量；**让编译器自己决定**（写清晰的代码 + `-O2`/`-O3`，现代编译器往往比手工展开更懂寄存器分配） |
| **与多累加器的关系** | 多累加器的收益**有上限**（A.14.4 的 $k_{\\text{opt}} = L/T$）；超过这个上限再"多"就纯属增加寄存器压力，**必然引发溢出** |

- **一句话**：**展开与多累加器是"用寄存器换并行度"的交易**，而寄存器是**有限的**——交易存在**最优点**，过了最优点就开始亏。

#### A.14.6 Haswell 关键功能单元的延迟与发射时间表

**本节从略，见 Lecture 15。**

> **说明**：Fall 2025 的 Lecture 15 讲义**未包含** Haswell 各功能单元延迟/发射时间的表格（教材 CS:APP3e 第 5.7 与 5.12 节有完整的"Latency, Issue time, Capacity"表，本讲义的幻灯片只保留了从该表**推导出的结论**）。为避免编造数据，此处不列具体数值。讲义中**确实给出**并被引用的参数只有一处：**双精度浮点加的延迟 $L = 5$、发射时间 $T = 1$**（用于 A.14.4 的 $\\max(L/k, T)$ 计算）。完整表格请查教材 Figure 5.12 / 5.16。

#### A.14.7 Amdahl 定律与强/弱扩展

**Amdahl 定律（讲义口径）**：

$$T_k = \frac{pT}{k} + (1-p)T
\qquad\Longrightarrow\qquad
S_k = \frac{T}{T_k} = \frac{1}{(1-p) + \dfrac{p}{k}}$$

$$T_\infty = (1-p)T
\qquad\Longrightarrow\qquad
S_\infty = \frac{1}{1-p}$$

| 符号 | 含义 |
|---|---|
| $T$ | 问题的**总串行时间** |
| $p$ | **可被加速的部分占总时间的比例**（$0 \\le p \\le 1$） |
| $k$ | **加速因子**（并行度/核心数） |
| $T_k$ | 加速后的时间：**可加速部分快 $k$ 倍 + 不可加速部分不变** |
| $T_\\infty$ | $k \\to \\infty$ 的极限时间 |
| $S_\\infty = 1/(1-p)$ | **最大可能加速比**——由**不可并行部分**（串行瓶颈）决定 |

**讲义的两个算例**：

| 算例 | 数据 | 计算 | 结论 |
|---|---|---|---|
| **旅行类比** | $T = 7.5$ 小时；$p = 6/7.5 = 0.8$（飞行部分）；$k \\to \\infty$ | $T_\\infty = (1-0.8)\\times 7.5 = 1.5$ 小时 | **最大加速比 5×**——"即使有 FTL，也必须先到纽约"（必须先飞 PIT→JFK 的 1.5 小时） |
| **数值例** | $T = 10$，$p = 0.9$，$k = 9$ | $T_9 = 0.9\\times\\frac{10}{9} + 0.1\\times 10 = 1.0 + 1.0 = 2.0$ | **5× 加速比**（讲义原文如此标注；实际 $10/2 = 5$，与 $T_9$ 的数值一致） |
| 同上，极限 | $k \\to \\infty$ | $T_\\infty = 0.1 \\times 10 = 1.0$ | **10× 上限**——**"即便有无限的并行计算资源"** |

- **Amdahl 的本质**：**"极限加速比揭示的是算法层面的限制（algorithmic limitation），而不是硬件的"** ——加再多核也救不了串行部分。讲义用**并行快速排序**做例证：**顶层划分无法加速**，第二层最多 2×，第 $k$ 层最多 $2^{k-1}$×——**串行瓶颈在递归树的顶端**。
- **对并行程序的实践含义**：优化并行程序时，**先减少串行部分**（$1-p$）往往比增加并行度更有效；此外还要注意**真共享/伪共享全局数据**、**同步开销**、**负载不均**（讲义列的"Beware of Amdahl's Law"提醒）。

**Gustafson 定律**（**补充说明**：F25 Lecture 24 讲义**未涉及** Gustafson 定律，教材第 12.6 节有）：

$$S_p = (1-\alpha) + \alpha \cdot p
\qquad\text{（在"问题规模随核心数增长"的前提下）}$$

| 定律 | 隐含假设 | 视角 | 结论 |
|---|---|---|---|
| **Amdahl** | **问题规模固定**，**串行部分固定** | 强扩展（strong scaling）：**同一个问题**加更多核 | 加速比有**上限** $1/(1-p)$ |
| **Gustafson** | **单核上的执行时间固定**，**问题规模随 $p$ 增长** | 弱扩展（weak scaling）：**更大的问题**用更多核在同一时间内解决 | 加速比可**随 $p$ 线性增长**（因为串行部分占比被摊薄） |

- **两者的分歧不是数学矛盾，而是假设不同**：Amdahl 说"**固定的工作量**加核有上限"，Gustafson 说"**核多了就做更大的工作量**，则几乎线性加速"。这正是**强扩展 vs 弱扩展**的经典分野——**报加速比必须说明是哪种**。

#### A.14.8 加速比、效率与扩展性

$$S_p = \frac{T_1}{T_p}
\qquad\qquad
E_p = \frac{S_p}{p} = \frac{T_1}{p \cdot T_p}
\qquad\qquad
0 < E_p \le 1$$

| 量 | 定义 | 理想值 | 说明 |
|---|---|---|---|
| **加速比** $S_p$ | $T_1 / T_p$（**单核时间 / $p$ 核时间**） | $p$（线性加速） | $S_p > p$ 称为**超线性加速（superlinear）**，通常来自**缓存效应**（$p$ 核时总缓存更大，工作集能装下）而非并行本身 |
| **效率** $E_p$ | $S_p / p$ | 1（100%） | 效率下降的原因：串行部分、同步开销、负载不均、通信/一致性流量 |
| **强扩展**（strong scaling） | **问题规模固定**，增加核心数 | — | 受 **Amdahl 定律**约束，$S_p$ 很快饱和 |
| **弱扩展**（weak scaling） | **每核的问题规模固定**，总规模随核心数**同步增长** | — | 更接近 **Gustafson 定律**，$S_p$ 可近似线性 |

**讲义实测数据（用于校验你的理解）**：

| 实验 | 配置 | 加速比 |
|---|---|---|
| 讲义某并行实验的**理论值** | 16 核 | **16×** |
| 实测**最佳**（某内核） | 16 核 | **2.86×**（远低于理论值——同步/串行瓶颈） |
| 另一实验 | 16 核 | **6.84×** |
| **内存累加（相邻 `psum[i]`）** | — | **5.0×** |
| **内存累加（间隔开的 `psum[i]`）** | — | **13.3×**（**唯一 > 8 的观测值**） |
| **寄存器累加（每线程局部变量）** | — | **7.5×**（比最快的内存累加**快 2 倍**） |

- **"Beware the speedup metric!"**：同一程序只改**数据布局**，加速比就从 5× 变 13.3×；换个**累加方式**又能到 7.5×。所以**加速比必须与测量条件、基线、核心数、数据规模一起报告**，否则毫无意义。
- **超线性加速的一个真实来源**：Cache Lab 的教训——**用寄存器/本地缓存永远最快**（讲义 "Lessons learned" 第 2、3 条）。多核时每个核有独立的 L1/L2，工作集被切小后命中率上升，可能出现 $S_p > p$。

#### A.14.9 常用性能工具与命令

| 工具/命令 | 作用 |
|---|---|
| `gcc -O2 -S -masm=att prog.c` | 生成汇编，**逐行核对编译器到底做了什么**（有没有把循环不变量外提、有没有向量化、有没有溢出） |
| `gcc -O3 -march=native -mavx2 -S` | 看**向量化**后的汇编（`vmovsd`/`vaddsd`/`vfmadd` 等） |
| `gcc -O2 -fopt-info-vec-optimized prog.c` | 让 GCC **报告哪些循环被向量化**（`-fopt-info-vec-missed` 报告没被向量化的原因，**极其实用**） |
| `perf stat -e cycles,instructions,cache-misses,branch-misses ./prog` | 一站式硬件计数器：周期数、指令数、缓存未命中、分支预测失败 |
| `perf stat -e cycles,instructions` → **IPC = instructions/cycles** | 判断"是否受限于指令发射"还是"受限于延迟/访存"（IPC < 1 通常意味着停顿） |
| `perf record -g ./prog` && `perf report` | 采样剖析，定位**热点函数**（比 `gprof` 更准，因为包含库调用） |
| `perf c2c record ./prog` && `perf c2c report` | **直接定位伪共享的缓存行**（A.13.10） |
| `gprof ./prog gmon.out` | 传统**插桩**剖析（讲义提示 Malloc Lab 可用）；只统计主程序，且对短函数有偏差 |
| `valgrind --tool=cachegrind ./prog` + `cg_annotate` | **模拟**各级缓存的命中率，按行/函数展开 |
| `getconf LEVEL1_DCACHE_LINESIZE` | 直接问系统**缓存行大小**（验证 A.13.10 的"64 字节"结论） |
| `lscpu` / `lscpu -C` | 查看 CPU 型号、频率、各级缓存容量与相联度 |
| `taskset -c 0 ./prog` | **固定到单核**——测 $T_1$、做"单核基线"时的必需操作 |
| `time ./prog`（bash 内建） | 粗看 `real`/`user`/`sys`；`user < real` 说明**没有充分利用多核** |
| `/usr/bin/time -v ./prog` | 输出 `Maximum resident set size`、`Page faults`、`Context switches` 等 |
| `likwid-perfctr -g MEM ./prog`（若安装） | 更细的内存带宽/缓存计数器（教学环境不一定有） |
| `numactl --hardware` / `numactl --cpunodebind` | 查看与绑定 NUMA 节点（多路机器上跨节点访存会显著拖慢性能） |

- **测量的黄金法则**：**先量化，再优化**。用 `perf stat` 判断瓶颈类别（**计算受限 / 延迟受限 / 访存受限 / 分支受限**），再选择对应手段——**不要凭直觉改代码**。

---

### A.15 Lab 速查

> 来源：教材官网 <https://csapp.cs.cmu.edu/3e/labs.html> 的 11 套官方 handout 与 writeup
> （`F25-.../LAB-*.txt`）+ CMU 15-213 Fall 2026 课程 Labs 页面。
> 权重与日期取自 Fall 2026 课程页面；handout 内部的分值分布取自各 lab 的官方 writeup。

#### A.15.1 11 个 lab 总表

| # | Lab | 对应讲次 | 官方工具 | **最容易踩的坑** |
|---|---|---|---|---|
| **0** | **C Programming**（L0，热身） | Lecture 1–2 | `make`、课程自带测试 | 改动**不该改的文件**；不读 README 就动手；C 语言基础（指针、字符串、`malloc`）不牢 |
| **1** | **Data Lab** | Lecture 2 | **`btest`**、**`dlc`**、**`driver.pl`**、`ishow`、`fshow` | 用了**被禁的运算符/控制流**（整数题只准**直线代码**，不准循环/条件）；常量**超过 8 位**；`dlc` 报的**运算符计数超标** |
| **2** | **Bomb Lab** | Lecture 3–5（+ GDB） | `objdump -d`、`gdb`、（评分服务器） | 用 `strings` 直接找答案（**学不到东西**）；**触雷扣分**（每次爆炸 **−1/2 分，最多扣 20 分**）；不先定位 `phase_1`…`phase_6` 就乱试 |
| **3** | **Attack Lab** | Lecture 5–6 | `hex2raw`、`ctarget`、`rtarget`、`farm.c` | **忘了先画栈帧**（`getbuf` 帧、`buf` 起始地址、返回地址偏移）；ROP 阶段没在 `.rodata`/`farm` 里用 `objdump` 找 **gadget**；在**与生成 target 不同的机器**上做题（地址对不上） |
| **4**（遗留） | **Buffer Lab**（IA32） | Lecture 5–6 | `hex2raw`、`bufbomb` | 与 Attack Lab 同类；**已被 64 位 Attack Lab 取代**，自学**优先做 64 位** |
| **5** | **Arch Lab** | Lecture 3–4 + 教材 Ch.4 | `make`、**`benchmark.pl`**、**`correctness.pl`**、`yis`/`ssim`/`psim` | 直接**用现成汇编**而不先想清楚数据通路；Part C 的 **`ncopy.ys`** 只对**部分块长度**正确；改 `pipe-full.hcl` 时**破坏回归测试** |
| **6** | **Cache Lab** | Lecture 9–10、15 | **`csim-ref`**、**`test-csim`**、**`test-trans`**、**`tracegen`** | Part A：**LRU 用链表导致 trace 大时过慢**（time 测量超时）；Part B：**越界使用数组**（只准**至多 12 个局部 int**、**不准定义数组**）；**只对 32×32 有效、64×64 崩** |
| **7** | **Performance Lab** | Lecture 9–10、15 | `driver.c`、`fcyc.c`、`clock.c`、`kernels.c` | 只改 `kernels.c` 的限制下**改错文件**；**只对测试尺寸正确**（尺寸是 32 的倍数但**其他尺寸也要对**）；忘了"**无 credit 于 buggy 代码**" |
| **8** | **Shell Lab** | Lecture 16–18 | **`tsh`**（骨架）、**`tshref`**（参考）、**`sdriver.pl`**、`trace01-16.txt`、`myspin`/`mysplit`/`myint` | **没用 `setpgid(0,0)`**；信号转发**忘了用 `-pid`**（`sdriver.pl` **专门测这个**）；`fork` 前**没屏蔽 `SIGCHLD`**（竞态）；**僵尸进程没回收**；在 `waitfg` 里也调 `waitpid` |
| **9** | **Malloc Lab** | Lecture 11–14 | **`mdriver`**（`-V`/`-v`/`-f`/`-t`/`-l`/`-h`）、`gprof` | **违反三条禁令**（改接口、调 `malloc`/`sbrk`、定义全局复合结构）；**利用率与吞吐率一头偏**（记得 $w=0.6$ 偏向利用率）；`short1,2-bal.rep` 小 trace 不用，非得在大 trace 上死磕 |
| **10** | **Proxy Lab** | Lecture 18–21 | **`driver.sh`**、CS:APP **Tiny** 服务器、`port-for-user.pl`、`curl`、`telnet` | **不转发完整 URL → 相对路径的转换**；**漏发 `Host`/`Connection`/`Proxy-Connection`**；`Content-Length` **不精确转发**；**用一把大锁保护整个缓存**（明确不可接受）；**用浏览器测缓存**（浏览器自己会缓存）；**代理遇错就退出**（必须长驻） |
| **11** | **SFS Lab**（CMU 自加） | Lecture 18–19、21–23 | CMU 自研 handout（教材官网**无**） | 这是 CMU Fall 2025/2026 新增的第 9 个 lab（课程代号 **L8**）；**不是**教材官网 11 个 lab 之一。**超级块 / inode 位图 / 目录条目全是共享资源，锁错一处文件系统静默损坏** |

> **两个"取代关系"**：**Buffer Lab**（32 位）已被 **Attack Lab**（64 位）取代；**Arch Lab** 另有 Legacy Y86（32 位）版本。**自学一律优先 64 位版本。**
> **关于 SFS Lab 的定位**：教材官网列表到 Proxy Lab 为止（共 11 个）；SFS 是 CMU 自研的第 9 个课程 lab。本文档把它列在末尾并明确标注这一点。

#### A.15.2 各 lab 的官方评分阈值与关键约束

**① Data Lab**（官方 writeup）

| 项 | 分值 |
|---|---|
| Correctness | 36 |
| **Performance** | 26 |
| Style | 5 |
| **合计** | **67** |

*   13 个编程题（puzzle），难度评级 **1–4 分**；只有**功能与性能都达标**才拿满分。
*   **限制**：整数题只准**直线代码**（**不准循环、不准条件**）；只准用 **8 个运算符** `! ~ & ^ \| + << >>`（部分函数进一步限制）；**不允许使用超过 8 位的常量**。
*   工具：`./btest`（功能正确性，`-f <func>` 测单个，`-1 4 -2 5` 传参）、`./dlc bits.c`（**检查是否违规**）、`./dlc -e bits.c`（**打印每个函数的运算符计数**）、`./driver.pl`（**汇总 btest + dlc 算出成绩**）、`./ishow`/`./fshow`（看位与浮点结构）。
*   **必须每次改完 `bits.c` 就重新 `make`/重建 `btest`**。

**② Bomb Lab**

| 阶段 | 分值 |
|---|---|
| 第 1–4 关 | **每关 10 分** |
| 第 5、6 关 | **每关 15 分** |
| **满分** | **70 分** |
| 扣分 | **每次爆炸 −1/2 分，最多扣 20 分** |

**③ Attack Lab**（官方 Figure 1，满分 100）

| 阶段 | 程序 | 关 | 方法 | 函数 | 分值 |
|---|---|---|---|---|---|
| 1 | `CTARGET` | 1 | **CI**（代码注入） | `touch1` | **10** |
| 2 | `CTARGET` | 2 | CI | `touch2` | **25** |
| 3 | `CTARGET` | 3 | CI | `touch3` | **25** |
| 4 | `RTARGET` | 2 | **ROP** | `touch2` | **35** |
| 5 | `RTARGET` | 3 | ROP | `touch3` | **5** |

*   **关键约束**：只能从 `rtarget` 中 `start_farm` 与 `end_farm` **之间**的地址构造 gadget；**必须在与生成 target 相似的机器上做题**；`hex2raw` 把文本转成攻击字符串。

**④ Arch Lab**（满分 **190**）——Part A **30** + Part B **35** + Part C **100**

*   Part A：30 分（3 个 Y86-64 程序，**各 10 分**）。
*   Part B：iaddq 指令描述 **10** + `y86-code` 回归 **10** + `ptest` 回归 **15** = **35 分**。
*   Part C：描述性注释 **20**（`ncopy.ys` 与 `pipe-full.hcl` 头部各 10）+ **性能 60**。
*   **Part C 性能评分公式**（$c$ = 平均 CPE）：

$$S = \begin{cases} 0, & c > 10.5 \\ 20\cdot(10.5 - c), & 7.50 \le c \le 10.50 \\ 60, & c < 7.50 \end{cases}$$

*   基线 `ncopy` 的平均 CPE **15.18**（范围 14.27–29.00）；**能做到平均 CPE < 9.00**；官方**最佳版本 7.48**。
*   工具：`./benchmark.pl`（测 CPE，**不检查正确性**）、**`./correctness.pl`**（必须用它查正确性）、`-f` 指定别的文件名、`-h` 看全部参数。

**⑤ Cache Lab**（Part A + Part B）

*   Part A 每测例：**正确报告 hits/misses/evictions 各占 1/3 分**（如某测例 3 分，hits 与 misses 对而 evictions 错 → 得 2 分）。评测用法示例：
  `./csim -s 2 -E 4 -b 3 -t traces/trans.trace`、`./csim -s 5 -E 1 -b 5 -t traces/long.trace`。
*   Part B 用 valgrind 抽地址 trace，再用参考模拟器回放，缓存参数固定为 **$(s=5, E=1, b=5)$**（即 **32 组、直接映射、32 字节块 = 1 KB**）：

| 矩阵尺寸 | 满分条件 | 0 分条件 | 满分 |
|---|---|---|---|
| **32 × 32** | $m < 300$ | $m > 600$ | **8 分** |
| **64 × 64** | $m < 1300$ | $m > 2000$ | **8 分** |
| **61 × 67** | $m < 2000$ | $m > 3000$ | **10 分** |

*   分数在阈值之间**线性插值**；**代码必须先正确才给性能分**；**允许对三种尺寸分别写专门代码**。Style **7 分**（助教会检查**非法数组**与**过多局部变量**）。
*   工具：`./test-csim`（Part A 自测）、`./test-trans -M 32 -N 32`（Part B，**并生成 `trace.fi`**）、`./csim-ref -v -s 5 -E 1 -b 5 -t trace.f0`（查看每次访问的 hit/miss/eviction）。

**⑥ Performance Lab**（rotate 50% + smooth 50%）

*   每个内核：**正确性**（**buggy 代码 0 分**；除测试尺寸外**其他尺寸也要正确**；可假定图像边长是 **32 的倍数**）+ **CPE**（达阈值 $S_r$ / $S_s$ 得满分，优于朴素版得部分分；官方常用**线性标度，最低约 40%**）。
*   只允许改 `kernels.c`；**可定义宏、额外全局变量、额外过程**。

**⑦ Shell Lab**（满分 **90**）

| 项 | 分值 |
|---|---|
| **说明**：**16 个 trace 文件，每个 5 分** | **80** |
| Style：注释 **5** + **检查每个系统调用的返回值 5** | **10** |

*   **必须做到**：提示符是 `"tsh> "`；支持**内置命令** `quit`/`jobs`/`bg <job>`/`fg <job>`；JID 用 `%` 前缀（如 `%5`）；`&` 后缀表示**后台**；**回收所有僵尸子进程**；子进程**因未捕获信号终止时要打印消息**（含 PID）；**不需支持管道 `\|` 与重定向 `<` `>`**。
*   **用法**：`./sdriver.pl -t trace01.txt -s ./tsh -a "-p"`（`-a "-p"` 让 shell **不打印提示符**）；`make test01` / `make rtest01`（跑参考 shell）；**`tshref.out` 是参考实现在所有 trace 上的完整输出**——比手工逐个跑方便得多。
*   建议**从 `trace01.txt` 开始逐个推进**，确保输出与参考 shell **完全一致**。

**⑧ Malloc Lab**（满分 **65**）

| 项 | 分值 |
|---|---|
| Correctness | **20** |
| Performance | **35** |
| Style | **10**（`mm_check` 5 + 结构与注释 5） |

$$P = wU + (1-w)\min\!\left(1, \frac{T}{T_{\text{libc}}}\right),\quad w = 0.6,\quad T_{\text{libc}} = 600\ \text{Kops/s}$$

*   详细提分点见 **A.9.12**；`mdriver` 用法见 **A.9.12**。

**⑨ Proxy Lab**（满分 **70**）

| 项 | 分值 |
|---|---|
| BasicCorrectness（自动评分） | **40** |
| Concurrency（自动评分） | **15** |
| Cache（自动评分） | **15** |

*   细节与约束见 **A.12.13**。**`./driver.sh` 必须在 Linux 机器上运行。**

**⑩ SFS Lab**（CMU 自加，课程代号 **L8**）

*   权重 **4%**、**1 个 grace day**；**不在教材官网的 11 个 lab 列表中**。

#### A.15.3 CMU 课程权重与截止日期（Fall 2026）

| 代号 | 名称 | 权重 | Grace Days | 发放 | 截止 | 代码评审登记 |
|---|---|---|---|---|---|---|
| **L0** | C Programming | 2% | 1 | Tue Aug 25 | Tue Sep 01 | — |
| **L1** | Data | **7%** | 1 | Thu Aug 27 | Tue Sep 08 | Thu Sep 10 |
| **L2** | Bomb | 6% | 1 | Thu Sep 03 | Tue Sep 15 | — |
| **L3** | Attack | 4% | 1 | Tue Sep 15 | Thu Sep 24 | — |
| **L4** | Cache | 5% | 2 | Thu Sep 24 | Thu Oct 08 | Sat Oct 10 |
| **L5a** | Malloc (checkpoint) | 4% | 2 | Thu Oct 08 | Tue Oct 27 | Thu Oct 29 |
| **L5b** | Malloc (final) | **7%** | 2 | （同 L5a） | Thu Nov 05 | — |
| **L6** | Shell | **7%** | 2 | Tue Nov 03 | Thu Nov 12 | Sat Nov 14 |
| **L7** | Proxy | 4% | 2 | Thu Nov 12 | Tue Nov 24 | Sun Nov 29 |
| **L8** | SFS | 4% | 1 | Thu Nov 19 | Thu Dec 03 | — |
| | **合计** | **50%** | | | | |

- **通用规则**：所有 lab **23:59 ET / 20:59 PT** 截止；最多可迟交 **3 天**（**L0 与 L8 最多 1 天**）；每人全学期 **5 个 grace day**（自动使用、**不可拒绝**）；**超出限额按每天 15% 计罚**；**不得邮件申请延期**。
- **代码评审（Code Review）**：**L1、L4、L5a、L5b、L6、L7** 需登记，分两部分——① **L4/L5a/L5b/L6/L7** 进行**与助教一对一的代码风格评审**（依据课程 Style Guidelines）；② **L1/L4/L5b/L6/L7** 进行 **10 分钟"帕森斯拼图（Parsons puzzle）"测试**，把打乱的代码片段拼回正确顺序并解释——**考的是"你是否真的理解这段代码为什么这么写"**。
- **考核总览**：**9 个 lab 共 50%**、书面作业 6%、随堂测验 4%、期中 15%、期末 25%。
- **自学建议**：即使没人给你打分，也请**自己执行这两步**——把代码读一遍问"这一行为什么必须在这儿"；把关键函数打乱重排看能否复原。

#### A.15.4 推荐执行顺序（阶段 0 → 阶段 4）

**三个排序原则**：**① 先补齐工具链能力 → ② 再按课程时间线推进 → ③ 把最难的 Malloc 留足时间。**

```
阶段 0：环境与工具（不做 lab，但是所有 lab 的前提）
  ├─ Bootcamp 1：Linux / 命令行 / Git
  ├─ Bootcamp 2：Debugging & GDB
  └─ Bootcamp 3：GCC & Build Automation
        │
        ▼
阶段 1：数据与机器码（Lecture 2–8 期间）
  ① L0 C Programming        （热身：确认 C 与 Linux 工具链没问题）
  ② Data Lab         ★★☆☆☆  （位运算 + 补码 + 浮点，检验 Lecture 2）
  ③ Bomb Lab         ★★★☆☆  （gdb + 汇编，检验 Lecture 3–5；全课程最"好玩"）
  ④ Attack Lab       ★★★★☆  （栈帧 + 代码注入 + ROP，检验 Lecture 5–6）
  并行可选：Buffer Lab（32 位遗留，只想练栈溢出可跳过）
        │
        ▼
阶段 2：存储与性能（Lecture 9–15 期间，期中前后）
  ⑤ Cache Lab        ★★★★☆  （模拟器 + 矩阵转置优化，知识密度最高）
  并行可选：Performance Lab（与 Cache Lab Part B 高度重叠，可二选一）
  ⑥ Arch Lab         ★★★★☆  （Y86-64 + 流水线，需额外读教材 Ch.4，可与 Cache Lab 换序）
        │
        ▼
阶段 3：内存管理（Lecture 11–14 之后，最需要时间的一段）
  ⑦ Malloc Lab       ★★★★★  （全课程最难，务必留 3 周以上；先做 checkpoint）
        │
        ▼
阶段 4：并发与系统（Lecture 16–24 期间）
  ⑧ Shell Lab        ★★★★☆  （进程 + 信号 + 作业控制）
  ⑨ Proxy Lab        ★★★★☆  （socket + 线程 + 缓存 + 同步，综合度最高）
  ⑩ SFS Lab          ★★★★☆  （并发文件系统，CMU 自加）
```

**各阶段的关键依赖**：

```
L0 C Programming ──► 依赖 Lecture 1-2（C 基础、位运算）
L1 Data          ──► 依赖 Lecture 2（补码、位运算、浮点、UB）
L2 Bomb          ──► 依赖 Lecture 3-5（x86-64 汇编、控制流、栈帧）+ GDB
L3 Attack        ──► 依赖 Lecture 5-6（栈帧、缓冲区、ROP）+ 反汇编
L4 Cache         ──► 依赖 Lecture 9-10（缓存组织、局部性、分块）
L5a/L5b Malloc   ──► 依赖 Lecture 11-14（虚拟内存、堆、分配器设计）
L6 Shell         ──► 依赖 Lecture 16-18（进程、信号、系统调用、I/O）
L7 Proxy         ──► 依赖 Lecture 18-21（socket、并发、缓存、同步）
L8 SFS           ──► 依赖 Lecture 18-19、21-24（并发、同步、文件系统）
```

**十条执行建议（踩过坑的人才知道）**：

1. **先做 L0 热身**——只有 2% 权重，但能确认 gcc、make、gdb、提交链路都通。
2. **Data Lab 不要抄答案**——它的价值是逼你理解补码与 IEEE 754 的位级结构；抄了就白学 Lecture 2。
3. **Bomb Lab 是"gdb 训练营"**——`objdump -d` 找到 `phase_1`…`phase_6`，逐个下断点用 `x/s`/`x/d`/`info registers` 推。**不要**用 `strings` 找答案。
4. **Attack Lab 先画栈帧再动手**——把 `getbuf` 栈帧、`buf` 起始地址、返回地址偏移量画清楚，剩下的就是填字节。
5. **Cache Lab Part A 用时间戳数组实现 LRU**，别用链表（trace 规模大时链表太慢）。Part B：32×32 用 8 个 8×8 分块 + 局部变量暂存；64×64 必须用 4×4 子块 + `B` 数组分 8 块的特殊分块法才能拿满分。
6. **Malloc Lab 分两阶段做**（对应 L5a checkpoint / L5b final）：先**隐式空闲链表 + 立即合并**保证正确性，再升级成**分离空闲链表 + 显式链表 + 边界标记优化**追求利用率。**先用 `mdriver` 跑通再优化**。
7. **Shell Lab 的关键是"信号时序"**——`waitpid` 的 `WNOHANG\|WUNTRACED`、`sigprocmask` 屏蔽 `SIGCHLD` 避免竞态、`setpgid` 建立进程组：这三件事做对了，作业控制就成了一半。
8. **Proxy Lab 先做"能用"，再做"并发"，最后做"缓存"**——迭代式代理 → 多线程（每连接一线程）→ 加读者-写者锁保护缓存 + LRU 淘汰。三步混在一起做，会同时面对三个 bug。
9. **每个 lab 第一步永远是读 README**——里面写清了文件清单、`make` 目标、提交要求、以及"**哪些文件不能改**"。**改错文件是零分的常见原因。**
10. **利用官方 handout 自带的测试工具**——`btest`（Data）、`mdriver`（Malloc）、`tracegen`/`test-csim`/`test-trans`（Cache）、`tshref` 参考壳 + `sdriver.pl`（Shell）、`driver.sh`（Proxy）。**它们比任何第三方答案都可靠。**

#### A.15.5 官方 handout 获取地址

教材官网 lab 页面对**自学者免费开放 handout**（只有**解答（solution）需要教师账号**）：

| lab | 页面（writeup） | README | **自学 handout（tar）** |
|---|---|---|---|
| **总入口** | <https://csapp.cs.cmu.edu/3e/labs.html> | — | — |
| Data Lab | <https://csapp.cs.cmu.edu/3e/datalab.pdf> | <https://csapp.cs.cmu.edu/3e/README-datalab> | <https://csapp.cs.cmu.edu/3e/datalab-handout.tar> |
| Bomb Lab | <https://csapp.cs.cmu.edu/3e/bomblab.pdf> | <https://csapp.cs.cmu.edu/3e/README-bomblab> | <https://csapp.cs.cmu.edu/3e/bomb.tar>（已禁用评分服务器） |
| Attack Lab | <https://csapp.cs.cmu.edu/3e/attacklab.pdf> | <https://csapp.cs.cmu.edu/3e/README-attacklab> | <https://csapp.cs.cmu.edu/3e/target1.tar>（**需用 `-q` 选项运行**） |
| Buffer Lab（IA32 遗留） | <https://csapp.cs.cmu.edu/3e/buflab32.pdf> | <https://csapp.cs.cmu.edu/3e/README-buflab32> | <https://csapp.cs.cmu.edu/3e/buflab32-handout.tar> |
| Arch Lab | <https://csapp.cs.cmu.edu/3e/archlab.pdf> | <https://csapp.cs.cmu.edu/3e/README-archlab> | <https://csapp.cs.cmu.edu/3e/archlab-handout.tar> |
| Arch Lab（Legacy 32 位） | <https://csapp.cs.cmu.edu/3e/archlab32.pdf> | <https://csapp.cs.cmu.edu/3e/README-archlab32> | <https://csapp.cs.cmu.edu/3e/archlab32-handout.tar> |
| Cache Lab | <https://csapp.cs.cmu.edu/3e/cachelab.pdf> | <https://csapp.cs.cmu.edu/3e/README-cachelab> | <https://csapp.cs.cmu.edu/3e/cachelab-handout.tar> |
| Performance Lab | <https://csapp.cs.cmu.edu/3e/perflab.pdf> | <https://csapp.cs.cmu.edu/3e/README-perflab> | <https://csapp.cs.cmu.edu/3e/perflab-handout.tar> |
| Shell Lab | <https://csapp.cs.cmu.edu/3e/shlab.pdf> | <https://csapp.cs.cmu.edu/3e/README-shlab> | <https://csapp.cs.cmu.edu/3e/shlab-handout.tar> |
| Malloc Lab | <https://csapp.cs.cmu.edu/3e/malloclab.pdf> | <https://csapp.cs.cmu.edu/3e/README-malloclab> | <https://csapp.cs.cmu.edu/3e/malloclab-handout.tar> |
| Proxy Lab | <https://csapp.cs.cmu.edu/3e/proxylab.pdf> | <https://csapp.cs.cmu.edu/3e/README-proxylab> | <https://csapp.cs.cmu.edu/3e/proxylab-handout.tar> |
| **SFS Lab** | **无（CMU 自研）** | — | **教材官网无 handout**——需通过 CMU 课程页面（Autolab / GitHub Education）获取 |

> **关于"网上答案"的严肃提醒。** CSAPP 的 lab 之所以经典，是因为**做的过程**才是学习本身。抄答案等于把最值钱的部分扔了。

{% endraw %}
