---
title: "附录：C 语言与系统编程核心概念速查表"
collection: course-notes
chapter: true
permalink: /course-notes/uiuc-ece220-computer-systems-programming/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/uiuc-ece220-computer-systems-programming/) · [← l21](/course-notes/uiuc-ece220-computer-systems-programming/l21)

{% raw %}
## 附录：C 语言与系统编程核心概念速查表

> 本速查表按类别汇总全课程的关键概念、代码模板、内存图与调试技巧，供复习与考试时快速检索。
> 所有代码模板均基于 ECE 220 的官方编码规范（4 空格缩进、始终使用花括号、`typedef struct` 风格）。

### 目录

1. [编译与工具链](#1-编译与工具链)
2. [数据类型与内存表示](#2-数据类型与内存表示)
3. [运算符与优先级](#3-运算符与优先级)
4. [指针速查](#4-指针速查)
5. [数组与指针算术速查](#5-数组与指针算术速查)
6. [字符串速查](#6-字符串速查)
7. [作用域与存储期](#7-作用域与存储期)
8. [函数与调用约定速查](#8-函数与调用约定速查)
9. [运行时栈与栈帧](#9-运行时栈与栈帧)
10. [结构体、typedef 与信息隐藏](#10-结构体typedef-与信息隐藏)
11. [动态内存管理](#11-动态内存管理)
12. [动态内存分配器内部机制](#12-动态内存分配器内部机制)
13. [数据结构速查](#13-数据结构速查)
14. [递归速查](#14-递归速查)
15. [排序算法速查](#15-排序算法速查)
16. [文件 I/O 速查](#16-文件-io-速查)
17. [格式化输入输出速查](#17-格式化输入输出速查)
18. [LC-3 汇编速查](#18-lc-3-汇编速查)
19. [C 到 LC-3 翻译模板](#19-c-到-lc-3-翻译模板)
20. [调试速查：GDB](#20-调试速查gdb)
21. [调试速查：Valgrind 与 Sanitizer](#21-调试速查valgrind-与-sanitizer)
22. [错误分类与排查流程图](#22-错误分类与排查流程图)
23. [常见内存错误图鉴](#23-常见内存错误图鉴)
24. [C 编程准则清单](#24-c-编程准则清单)

---

### 1. 编译与工具链

#### 官方编译命令

```bash
# ECE 220 标准编译命令（不要省略任何警告选项）
gcc -g -std=c99 -Wall -Werror -o output_executable \
    -l library1 [-l library2] ... source_file1.c [source_file2.c] ...
```

| 选项 | 作用 | 为什么重要 |
|---|---|---|
| `-g` | 生成调试符号 | GDB 能看到变量名和源码行号；没有它 `gdb` 只能看汇编 |
| `-std=c99` | 使用 C99 标准 | 课程要求的语言子集；保证 `//` 注释、块内声明、`stdint.h` 可用 |
| `-Wall` | 打开所有常见警告 | 未初始化变量、隐式声明、可疑的 `=` 等 |
| `-Werror` | 把警告变成错误 | 强迫你面对每一个警告，而不是忽略它们 |
| `-o out` | 指定输出文件名 | 注意与 `-O0`（优化级别 0）不同，别写混 |
| `-O0` | 关闭优化 | 调试时使用；源码行与执行流一一对应 |
| `-O2` | 开启优化 | 变量可能被优化进寄存器或被删除，GDB 行为会"古怪" |
| `-lm` | 链接数学库 | 用了 `sqrtf` 等函数时必须加，且**必须放在源文件之后** |

#### 调试与净化的编译变体

```bash
# 调试版：关闭优化，带符号
gcc -g -O0 -std=c99 -Wall -Werror -o prog prog.c

# 净化版：AddressSanitizer + UndefinedBehaviorSanitizer
gcc -g -O0 -std=c99 -Wall -fsanitize=address,undefined -o prog prog.c

# 泄漏检测版（Valgrind 不需要特殊编译，但需要 -g）
gcc -g -O0 -std=c99 -Wall -o prog prog.c
valgrind --leak-check=full --show-leak-kinds=all ./prog
```

#### 多文件编译

```bash
gcc -g -std=c99 -Wall -Werror -o prog main.c stack.c util.c -lm
```

头文件用 `#include "my.h"`（双引号，先搜当前目录），系统头文件用 `#include <stdio.h>`（尖括号）。

---

### 2. 数据类型与内存表示

#### 基本类型表（64 位机器，即 EWS 实验环境）

| 类型 | 大小 | 编码 | 最小值 | 最大值 |
|---|---|---|---|---|
| `bool` | 1 字节（实际只需 1 bit，编译器补齐） | 0/1 | `false` | `true` |
| `char` | 8 bit | 有符号整数 | −128 | 127 |
| `unsigned char` | 8 bit | 无符号整数 | 0 | 255 |
| `short` | 16 bit | 有符号整数 | −32 768 | 32 767 |
| `unsigned short` | 16 bit | 无符号整数 | 0 | 65 535 |
| `int` | 32 bit | 有符号整数 | ≈ −2.1×10⁹ | ≈ 2.1×10⁹ |
| `unsigned int` | 32 bit | 无符号整数 | 0 | ≈ 4.3×10⁹ |
| `long` | 64 bit | 有符号整数 | ≈ −9.2×10¹⁸ | ≈ 9.2×10¹⁸ |
| `unsigned long` | 64 bit | 无符号整数 | 0 | ≈ 1.8×10¹⁹ |
| `float` | 32 bit | IEEE 754 单精度 | −3.4×10³⁸ | 3.4×10³⁸（7 位十进制有效数字） |
| `double` | 64 bit | IEEE 754 双精度 | −1.8×10³⁰⁸ | 1.8×10³⁰⁸（16 位十进制有效数字） |

> ⚠️ `int` 和 `long` 的大小**依赖平台**。课程参考明确指出：这些尺寸假设 64 位处理器。
> 在 32 位或 16 位平台（如 LC-3）上，`int` 可能更小。

#### 派生类型表

| 类型 | 运算符 | 示例 | 大小 | 说明 |
|---|---|---|---|---|
| 指向 Z 的指针 | `*` | `int *x;` | 64 bit (8 字节) | 一个存放 Z 类型对象地址的对象 |
| n 个 Z 的数组 | `[]` | `int x[4];` | n × sizeof(Z) | 连续存放的 n 个 Z 类型对象 |
| 结构体 | `struct` | `struct { int x, y; } v;` | 成员大小之和（可能有填充） | 顺序存放的成员对象集合 |

**⚠️ 数组参数例外规则**（官方参考原文）：

```c
void foo(int x[4]);   /* 编译器立即转换成： */
void foo(int *x);
```

> 数组作为函数参数时**立即退化为指针**，因此函数内部 `sizeof(x)` 得到的是**指针大小 8**，
> 而不是数组大小。这就是必须额外传递长度的根本原因。

#### 字面量与后缀

| 后缀 | 示例 | 结果类型 |
|---|---|---|
| `ul` | `42ul` | `unsigned long` |
| `u` | `42u` | `unsigned int` |
| `l` | `42l` | `long` |
| （无） | `42` | `int` |
| `f` | `42.0f` | `float` |
| `.` | `42.0` | `double` |

| 名字 | 示例 | 描述 | 结果类型 |
|---|---|---|---|
| 字符字面量 | `'c'` | 某字符的 ASCII 值 | `int` |
| 字符串字面量 | `"hello"` | NUL 结尾的 ASCII 数组 | `char *` |

**字符串字面量的特殊性**：它**隐式地在静态存储区分配并初始化**一个字符数组，
而指针本身分配在临时空间。唯一例外是用作 `char` 数组初始化器时：

```c
char s[6] = "hello";   /* 等价于： */
char s[6] = { 'h', 'e', 'l', 'l', 'o', '\0' };
```

#### 转义序列

| 序列 | 结果 |
|---|---|
| `\0` | NUL（字符串结束符） |
| `\n` | 换行 |
| `\t` | 制表符 |
| `\\` | 反斜杠 |
| `\"` | 双引号 |
| `\'` | 单引号 |
| `\r` | 回车 |

#### 精确宽度类型（推荐在 ECE 220 中使用）

```c
#include <stdint.h>
int8_t   /*  8 位有符号 */   uint8_t  /*  8 位无符号 */
int16_t  /* 16 位有符号 */   uint16_t /* 16 位无符号 */
int32_t  /* 32 位有符号 */   uint32_t /* 32 位无符号 */
int64_t  /* 64 位有符号 */   uint64_t /* 64 位无符号 */
intptr_t /* 能装下一个指针的整数 */
```

> **为什么 `intptr_t` 存在？** 64 位地址空间出现后，`int`（32 位）**装不下指针**了。
> `sbrk(intptr_t increment)` 的参数就是为此设计的。

---

### 3. 运算符与优先级

#### 按优先级从高到低（常用部分）

| 优先级 | 运算符 | 结合性 | 说明 |
|---|---|---|---|
| 1 | `()` `[]` `->` `.` | 左→右 | 函数调用、下标、成员访问 |
| 2 | `!` `~` `++` `--` `+`(一元) `-`(一元) `*`(解引用) `&`(取地址) `sizeof` `(type)` | 右→左 | 一元运算符 |
| 3 | `*` `/` `%` | 左→右 | 乘除取模 |
| 4 | `+` `-` | 左→右 | 加减（含指针算术） |
| 5 | `<<` `>>` | 左→右 | 移位 |
| 6 | `<` `<=` `>` `>=` | 左→右 | 关系 |
| 7 | `==` `!=` | 左→右 | 相等 |
| 8 | `&` | 左→右 | 按位与 |
| 9 | `^` | 左→右 | 按位异或 |
| 10 | `\|` | 左→右 | 按位或 |
| 11 | `&&` | 左→右 | 逻辑与（**短路**） |
| 12 | `\|\|` | 左→右 | 逻辑或（**短路**） |
| 13 | `?:` | 右→左 | 条件运算符 |
| 14 | `=` `+=` `-=` `*=` `/=` `%=` `&=` `\|=` `^=` `<<=` `>>=` | 右→左 | 赋值 |
| 15 | `,` | 左→右 | 逗号 |

#### 三个最容易记错的点

```c
/* ① 解引用与乘法用同一个符号，靠上下文区分 */
int *A, *B;
int c = (*A) * (*B);      /* 正确、可读 */
int c = *A**B;            /* 能编译，但没人读得懂 —— 别这么写 */

/* ② 后缀 ++ 与 * 的优先级 */
int arr[3] = {1, 2, 3};
int *p = arr;
int x = *p++;             /* 等价于 *(p++)：先取 *p=1，再 p 后移 → x=1, p 指向 arr[1] */
int y = (*p)++;           /* 取 arr[1] 的值 2，再让 arr[1] 变成 3 → y=2, arr[1]=3 */

/* ③ 赋值运算符优先级极低 */
if (a = b) { }            /* 编译通过（除非 -Werror 拦下）：这是赋值，不是比较！ */
if (a == b) { }           /* 正确写法 */
if (3 == a) { }           /* "Yoda 写法"：写错成 3 = a 会直接编译报错 */
```

#### 位运算模板

```c
x & (1 << n)          /* 测试第 n 位是否为 1 */
x |= (1 << n)         /* 置第 n 位为 1 */
x &= ~(1 << n)        /* 清第 n 位为 0 */
x ^= (1 << n)         /* 翻转第 n 位 */
(x >> n) & 1          /* 取出第 n 位 */
x & 0xFFFF            /* 取低 16 位 */
x & (x - 1)           /* 清除最低位的 1（用于判断 2 的幂） */
(x & (x - 1)) == 0    /* x 是 2 的幂（x != 0） */
```

---

### 4. 指针速查

#### 核心规则

```c
int   v = 42;
int  *p = &v;      /* p 的类型是 int*，值是 v 的地址 */
int **q = &p;      /* q 的类型是 int**，值是 p 的地址 */

*p        /* 解引用：得到 v 的值 42 */
&v        /* 取地址：得到 v 的地址（类型 int*） */
**q       /* 等价于 *p，即 v 的值 */
q         /* p 的地址 */
```

**指针类型的读法：从右往左读。**

```c
int  * p;      /* p 是一个指针，指向 int */
int ** p;      /* p 是一个指针，指向（指向 int 的指针） */
char * p;      /* p 是一个指针，指向 char */
```

#### 声明陷阱（课程重点强调）

```c
int *A, B;     /* A 是 int*，但 B 是 int ！！ */
int *A, *B;    /* 两个都是 int* —— 想声明多个指针必须每个都写 * */
```

#### 指针的三条基本事实

1. **指针就是一个内存地址**——一个地址需要多少位取决于内存的可寻址性（64 位机器上是 8 字节）。
2. **编译器知道指针的类型**，因此能正确解读该地址上的比特（这是类型存在的核心意义）。
3. **声明一个指针只为指针本身分配空间**，不为它指向的东西分配空间。
   如果你需要一个 `int` 供指针指向，必须另行声明。

```c
int *p;        /* 只为 p 分配 8 字节，p 未初始化（野指针！） */
int  v;        /* 现在有了一个 int 供 p 指向 */
p = &v;        /* 让 p 指向它 */
*p = 42;       /* 安全地写入 */
```

#### 指针常量与 NULL

```c
int *p = NULL;         /* 空指针：明确表示"不指向任何东西" */
if (p != NULL) { ... } /* 解引用前必须检查 */
if (p) { ... }         /* 等价写法：NULL 在布尔上下文中为假 */
```

> `NULL` 通常定义为 `((void *)0)`。**永远不要解引用 NULL**——
> 在 Linux 上会产生段错误（segmentation fault），因为虚拟地址 0 不映射到任何物理页。

#### 用指针让函数修改调用者的变量（swap 模板）

```c
void swap(int32_t *a, int32_t *b)
{
    int32_t tmp = *a;
    *a = *b;
    *b = tmp;
}

/* 调用：swap(&x, &y); —— 必须传地址，因为 C 是传值调用 */
```

#### 指针的指针：让函数修改调用者的**指针**

```c
/* 模板：分配成功时把新指针写回调用者 */
int32_t alloc_buffer(uint8_t **out, size_t n)
{
    uint8_t *p = malloc(n);
    if (p == NULL) {
        return -1;         /* 失败：不修改 *out，调用者的指针保持原样 */
    }
    *out = p;              /* 成功：写回 */
    return 0;
}

/* 调用 */
uint8_t *buf = NULL;
if (alloc_buffer(&buf, 1024) != 0) {
    fprintf(stderr, "allocation failed\n");
}
```

#### `realloc` 的正确写法（避免失败时泄漏）

```c
/* ❌ 错误：realloc 失败返回 NULL，原指针被覆盖，内存泄漏 */
p = realloc(p, new_size);

/* ✅ 正确：用临时指针接住返回值 */
void *tmp = realloc(p, new_size);
if (tmp == NULL) {
    /* 原内存 p 仍然有效，可以继续使用或在这里 free */
    free(p);
    return -1;
}
p = tmp;
```

---

### 5. 数组与指针算术速查

#### 数组的本质

```c
int region[20];    /* 编译器分配 20 个连续 int，名为 region[0] … region[19] */
```

- `region` 这个表达式的类型是 `int *`，值是 `region[0]` 的地址。
- `region + N` 指向 `region[N]`，这叫**指针算术 (pointer arithmetic)**。
- `region[N]` 与 `*(region + N)` **完全等价**——方括号只是"加法 + 解引用"的简写。

#### 指针算术的步长

```c
int arr[5];
int *p = arr;

p + 1     /* 地址增加 sizeof(int) = 4 字节，不是 1 字节 */
p + 5     /* 增加 5 × 4 = 20 字节 */

/* 一般规则：ptr + n 的新地址 = (char *)ptr + n * sizeof(*ptr) */
```

> 课程幻灯片原话：*"The amount added is the number of addresses required for 5 ints."*
> 如果 `region` 是 `0x12345000`，那么 `region + 5` **不是** `0x12345005`，
> 而是 `0x12345014`（假设 `sizeof(int) == 4`）。

#### 三种等价的遍历写法

```c
int arr[5] = {10, 20, 30, 40, 50};

/* 写法 1：下标 */
for (int i = 0; i < 5; i++) { printf("%d ", arr[i]); }

/* 写法 2：指针 + 偏移 */
for (int i = 0; i < 5; i++) { printf("%d ", *(arr + i)); }

/* 写法 3：移动指针（推荐，最贴近机器） */
for (int *p = arr; p < arr + 5; p++) { printf("%d ", *p); }
```

> `arr + 5` 是 **one-past-the-end** 指针（尾后指针）。C 标准允许**计算**它并用于比较，
> 但**不允许解引用**它。

#### `sizeof` 陷阱

```c
void f(int arr[10])
{
    /* ⚠️ 参数已退化为 int*，这里的 sizeof 是指针大小！ */
    size_t n = sizeof(arr);        /* 8，不是 40 */
    size_t m = sizeof(arr) / sizeof(arr[0]);  /* 8/4 = 2，完全错误！ */
}

int main(void)
{
    int arr[10];
    size_t n = sizeof(arr) / sizeof(arr[0]);  /* 40/4 = 10 ✅ 只在定义所在的函数里有效 */
    f(arr);    /* 必须另行传长度：改为 f(arr, 10) */
    return 0;
}
```

**修复模板**：永远显式传递长度。

```c
void f(int32_t *arr, size_t n)
{
    for (size_t i = 0; i < n; i++) { /* ... */ }
}
```

#### 数组名不是指针变量

| | 数组名 `arr` | 指针变量 `p` |
|---|---|---|
| 有自己的存储空间吗？ | **没有**（它只是首元素地址的别名） | **有**（8 字节） |
| `sizeof` | 整个数组的大小 (n × sizeof(T)) | 指针大小 (8) |
| 可以 `arr = ...` 赋值吗？ | **不可以**（不是左值） | 可以 |
| `&arr` 的类型 | `int (*)[n]`（指向数组的指针） | `int **` |
| `&arr + 1` 的步长 | n × sizeof(T) | 8 |

#### 二维数组的行主序 (row-major) 布局

```c
int m[3][4];    /* 3 行 4 列，共 12 个 int，连续存放 */

/* 元素 m[i][j] 的地址公式： */
/*   base + (i * 4 + j) * sizeof(int)     ← 注意 4 是列数 */
```

```
内存布局（每格 4 字节）：
偏移:    0    4    8   12   16   20   24   28   32   36   40   44
      +----+----+----+----+----+----+----+----+----+----+----+----+
      |[0][0]|[0][1]|[0][2]|[0][3]|[1][0]|[1][1]|[1][2]|[1][3]|[2][0]|...
      +----+----+----+----+----+----+----+----+----+----+----+----+
        └──── 第 0 行 ────┘└──── 第 1 行 ────┘└──── 第 2 行 ────┘
```

#### 三种"多维"声明的区别（高频考点）

```c
int  m1[3][4];      /* 真正的二维数组：12 个 int 连续存放，共 48 字节 */
int *m2[3];         /* 3 个指针的数组，共 24 字节，每行可指向不同长度的内存（锯齿表） */
int (*m3)[4];       /* 一个指针，指向"含 4 个 int 的数组"，步长 16 字节 */
```

```
m1:  [ 12 个 int 连续 ]            m2:  [ p0 | p1 | p2 ]        m3:  p ──►[ int[4] ]
     m1[i] 退化为 int*                  ↓    ↓    ↓                  （只有一行）
     sizeof(m1)   = 48              [4 ints][4 ints][6 ints]      sizeof(m3) = 8
     sizeof(m1[0])= 16              sizeof(m2)   = 24             m3 + 1 步长 16
```

**读法技巧**：从变量名出发，先看右边，再看左边。

```c
int  *m2[3];    /* m2 先与 [3] 结合 → m2 是数组；元素类型是 int* */
int (*m3)[4];   /* 括号强制 m3 先与 * 结合 → m3 是指针；指向 int[4] */
int *f(void);   /* f 先与 () 结合 → f 是函数；返回 int* */
int (*f)(void); /* 括号强制 f 先与 * 结合 → f 是指针；指向"返回 int 的函数" */
```

---

### 6. 字符串速查

#### C 字符串 = NUL 结尾的 char 数组

```c
char s1[] = "hello";        /* 数组：6 字节（含 '\0'），可修改 */
char *s2  = "hello";        /* 指针：只占 8 字节，指向静态存储区的只读字面量 */
```

```
char s1[] = "hello";    ← 在栈上分配 6 字节
栈:  +-----+-----+-----+-----+-----+-----+
     | 'h' | 'e' | 'l' | 'l' | 'o' | \0  |
     +-----+-----+-----+-----+-----+-----+
     s1 是数组名，sizeof(s1) == 6

char *s2 = "hello";     ← 在栈上只分配 8 字节，指向全局数据段
栈:  +------------------+
     | s2 = 0x402008    |  (8 字节)
     +------------------+
              |
              v
全局数据段(只读): +-----+-----+-----+-----+-----+-----+
                  | 'h' | 'e' | 'l' | 'l' | 'o' | \0  |
                  +-----+-----+-----+-----+-----+-----+
```

> ⚠️ `s2[0] = 'H';` 是**未定义行为**（修改字符串字面量）。`s1[0] = 'H';` 是合法的。

#### 标准字符串函数（`#include <string.h>`）

| 函数 | 原型 | 说明 | 陷阱 |
|---|---|---|---|
| `strlen` | `size_t strlen(const char *s)` | 返回**不含 NUL** 的长度 | O(n)；循环里反复调用会变 O(n²) |
| `strcpy` | `char *strcpy(char *dst, const char *src)` | 复制含 NUL | **不检查目标缓冲区大小** → 溢出 |
| `strncpy` | `char *strncpy(char *dst, const char *src, size_t n)` | 最多复制 n 字节 | n 用完时**可能不写 NUL**！ |
| `strcat` | `char *strcat(char *dst, const char *src)` | 追加 | 同样不检查大小 |
| `strncat` | `char *strncat(char *dst, const char *src, size_t n)` | 最多追加 n 字节 | 会自动补 NUL |
| `strcmp` | `int strcmp(const char *a, const char *b)` | 比较，**返回 0 表示相等** | 别写成 `if (strcmp(a,b))` 表示相等 |
| `strncmp` | `int strncmp(const char *a, const char *b, size_t n)` | 比较前 n 字节 | |
| `strchr` | `char *strchr(const char *s, int c)` | 查找字符，返回指针或 NULL | |
| `strstr` | `char *strstr(const char *h, const char *n)` | 查找子串 | |
| `strtok` | `char *strtok(char *s, const char *delim)` | 切分（**会修改原串**） | 非可重入；首参固定时返回 NULL 表示结束 |
| `strtol` | `long strtol(const char *s, char **end, int base)` | 字符串转整数 | 必须检查 `end` 与 `errno` |

#### 安全模板

```c
/* 模板：安全复制（自己检查长度） */
int32_t safe_copy(char *dst, size_t dst_size, const char *src)
{
    size_t need = strlen(src) + 1;    /* +1 给 NUL */
    if (need > dst_size) {
        return -1;                    /* 目标太小，拒绝复制 */
    }
    memcpy(dst, src, need);
    return 0;
}
```

```c
/* 模板：用 snprintf 做有界格式化（最省心的安全拼接） */
char buf[64];
int n = snprintf(buf, sizeof(buf), "value=%d, name=%s", v, name);
if (n < 0 || (size_t)n >= sizeof(buf)) {
    /* 被截断了 */
}
```

#### `strcmp` 的正确用法

```c
if (strcmp(a, b) == 0) { /* 相等 */ }
if (strcmp(a, b) <  0) { /* a 在字典序上小于 b */ }
if (strcmp(a, b) != 0) { /* 不等 */ }

/* ❌ 常见错误：这样比较的是两个指针的地址，永远不为 0（字面量可能相同，但不可依赖） */
if (a == b) { }
```

---

### 7. 作用域与存储期

#### 作用域（Scope）—— 名字在代码的哪里可见

| 种类 | 声明位置 | 可见范围 |
|---|---|---|
| **文件作用域** | 在所有函数之外，加 `static` | 本文件内（推荐，避免全局污染） |
| **全局作用域** | 在所有函数之外，不加 `static` | 整个程序（**应尽量避免**） |
| **函数作用域** | 函数体内 | 该函数内 |
| **块作用域** | `{ }` 之内 | 该块内 |

```c
static int32_t file_counter = 0;   /* 文件作用域：其他文件看不见 */

void f(void)
{
    int32_t local = 0;             /* 函数作用域 */
    {
        int32_t inner = 5;         /* 块作用域：出了这对花括号就消失 */
    }
    /* inner 在这里不存在 */
}
```

> **为什么避免全局变量？** 课程幻灯片给出的理由很实际：
> 想象一个 100 万行、20 个程序员的项目——你无法保证名字不冲突，
> 而且**你还得考虑链接进来的库代码**。收益极小，痛苦极大。
> 用 `static` 把作用域限制在文件内。

#### 存储期（Storage Duration）—— 变量的寿命

| 存储期 | 何时分配/释放 | 存放在哪 | 典型例子 |
|---|---|---|---|
| **automatic**（自动） | 进入块时分配，离开块时释放 | **栈** | 普通局部变量 |
| **static**（静态） | 程序启动时分配，程序结束时释放 | **全局数据段** | 全局变量、`static` 局部变量、字符串字面量 |
| **dynamic**（动态） | 显式 `malloc` 时分配，显式 `free` 时释放 | **堆** | `malloc` 返回的内存 |

```c
#include <stdio.h>

int32_t g = 1;                 /* static storage duration，整个程序生命周期 */

void counter(void)
{
    static int32_t n = 0;      /* static storage duration，但作用域只在本函数内 */
    n++;
    printf("called %d times\n", n);   /* 每次调用都递增，不会重置 */
}

void f(void)
{
    int32_t local = 5;         /* automatic：每次进入 f 都重新分配 */
}
```

#### 内存映射（Memory Map）

```
高地址  ┌──────────────────────────────┐
        │  系统空间 (system space)      │  操作系统保留
        ├──────────────────────────────┤
        │  栈 (stack)                   │  ← 向低地址增长
        │   局部变量、参数、返回地址      │
        │            ⋮                  │
        │            ⋮                  │
        │  堆 (heap)                    │  ← 向高地址增长
        │   malloc/calloc/realloc 分配   │
        ├──────────────────────────────┤ ← break（sbrk 调整这里）
        │  全局数据 (global data)        │
        │   全局/静态变量、字符串字面量    │
        ├──────────────────────────────┤
        │  代码 (code)                  │
        ├──────────────────────────────┤
低地址  │  系统空间 (system space)      │
        └──────────────────────────────┘
```

**记忆要点**：栈和堆**相向增长**，中间的空隙是它们共享的可用空间。
栈溢出（递归太深）和堆耗尽（忘记 `free`）从两端逼近同一个空隙。

---

### 8. 函数与调用约定速查

#### 声明的三种形态

```c
/* 1. 函数原型（声明）：告诉编译器返回类型和参数类型，不含函数体 */
int32_t max_of(int32_t a, int32_t b);

/* 2. 函数定义：包含函数体 */
int32_t max_of(int32_t a, int32_t b)
{
    return (a > b) ? a : b;
}

/* 3. 无参数的函数：用 (void) 而不是 () */
int32_t get_value(void);   /* ✅ 明确表示不接受参数 */
int32_t get_value();       /* ⚠️ 在 C 中表示"参数未指定"，不是"无参数" */
```

#### 参数传递：一切皆传值

```c
void try_to_change(int32_t x)   { x = 99; }        /* 改的是副本，调用者不变 */
void really_change(int32_t *px) { *px = 99; }      /* 改的是调用者的变量 */

int main(void)
{
    int32_t v = 1;
    try_to_change(v);        /* v 仍然是 1 */
    really_change(&v);       /* v 变成 99 */
    return 0;
}
```

**底层原因**：C 的参数通过栈传递（LC-3 上还有 R0–R3），传的是**实参的副本**。
要修改调用者的变量，你必须把**它的地址**（一个副本，但副本的内容是地址）传进去。

#### 数组参数

```c
/* 下面四种写法在编译器眼里完全一样，全都退化为 int32_t* */
void f(int32_t arr[10]);
void f(int32_t arr[]);
void f(int32_t *arr);
void f(int32_t *arr, size_t n);   /* ✅ 推荐：显式带长度 */
```

#### 函数指针（高频考点）

```c
/* 声明一个函数指针 */
int32_t (*cmp)(const void *, const void *);

/* 读法：从 cmp 出发 → 先看 (*cmp) 说明是指针 → 再看 (…) 说明指向函数
        → 返回 int32_t */

/* 赋值：函数名本身就是函数地址 */
cmp = my_compare;      /* ✅ 等价于 cmp = &my_compare; */

/* 调用 */
int32_t r = cmp(&a, &b);        /* ✅ 等价于 (*cmp)(&a, &b); */

/* 常见混淆 */
int32_t  *f(int32_t);           /* f 是函数，返回 int32_t* */
int32_t (*f)(int32_t);          /* f 是指针，指向返回 int32_t 的函数 */
int32_t (*tbl[4])(int32_t);     /* tbl 是数组，每个元素是函数指针（跳转表） */
```

#### 回调与跳转表模板

```c
/* 跳转表：用一个整数索引直接分派到不同函数 */
typedef int32_t (*op_fn)(int32_t, int32_t);

static int32_t op_add(int32_t a, int32_t b) { return a + b; }
static int32_t op_sub(int32_t a, int32_t b) { return a - b; }
static int32_t op_mul(int32_t a, int32_t b) { return a * b; }

static op_fn ops[] = { op_add, op_sub, op_mul };

int32_t dispatch(size_t which, int32_t a, int32_t b)
{
    if (which >= sizeof(ops) / sizeof(ops[0])) {
        return 0;
    }
    return ops[which](a, b);      /* 通过函数指针调用 */
}
```

> **与 LC-3 的联系**：这正是 LC-3 汇编里"跳转表"（`JSRR` 配合一个存地址的数组）的 C 版本。
> 两者都是"用数据（地址）取代一长串条件分支"。

---

### 9. 运行时栈与栈帧

#### LC-3 调用约定寄存器

| 寄存器 | 用途 | 保存责任 |
|---|---|---|
| `R0`–`R3` | 传递参数、返回返回值 | **caller-saved**（调用者负责保存） |
| `R4` | 全局数据指针 (global data pointer) | 全局约定 |
| `R5` | 帧指针 (frame pointer) | 被调用者保存并恢复 |
| `R6` | 栈指针 (stack pointer) | 被调用者保存并恢复 |
| `R7` | 返回地址 (return address) | 被调用者保存并恢复 |

#### LC-3 栈帧布局

> **本节偏移已对照课程真实汇编核实**（`translate.asm` 的 `FIND_ABS`、`538-mt1-review` 讲义）。

```
高地址  ┌──────────────────────────┐
        │  caller 的栈帧            │
        ├──────────────────────────┤
        │  参数                     │  ← R5 + 4, R5 + 5, …
        ├──────────────────────────┤
        │  return value            │  ← R5 + 3   ★
        ├──────────────────────────┤
        │  return address (R7)     │  ← R5 + 2
        ├──────────────────────────┤
        │  previous frame pointer  │  ← R5 + 1
        ├──────────────────────────┤
        │  局部变量（第一个）        │  ← R5 + 0   ← R5 指向这里
        ├──────────────────────────┤
        │  局部变量                 │  ← R5 − 1
        │  …                       │
低地址  └──────────────────────────┘  ← R6 指向栈顶
```

**记忆口诀**：从 `R5` 往上数——`+1` 旧帧指针、`+2` 返回地址、`+3` **返回值**、`+4` 起参数；
`R5+0` 往下是局部变量。

> **为什么返回值必须在 `R5+3`？** 因为调用者返回后要用**一条** `ADD R6, R6, #(nparams + 1)`
> 同时弹出参数**和**返回值。只有返回值紧贴在第一个参数（`R5+4`）下面，这一条指令才成立。

#### 调用序列（caller 侧）

```
1. 把参数压栈（从右往左）        ADD R6, R6, #-1 ; STR R0, R6, #0
2. JSR 到子程序                  JSR SUB
3. 从 R0 取返回值                ; 返回值写在 R0
4. 清理参数占用的栈空间          ADD R6, R6, #n
```

#### 被调用序列（callee 侧）

课程真实代码（`translate.asm` 中的 `FIND_ABS`，`R5 = R6 + 0`，含 1 个局部变量）：

```asm
FIND_ABS
        ; ---- 建立栈帧 ----
        ADD     R6,R6,#-4       ; 4 个位置 = 3 个 linkage + 1 个局部变量
        STR     R5,R6,#1        ; 保存旧帧指针        → R5+1
        ADD     R5,R6,#0        ; 设置帧指针（R5 = R6）
        STR     R7,R5,#2        ; 保存返回地址        → R5+2
        ; ---- 函数体 ----
        LDR     R0,R5,#4        ; 读第一个参数 num    → R5+4
        STR     R0,R5,#0        ; 写局部变量          → R5+0
        STR     R0,R5,#3        ; 写返回值            → R5+3
        ; ---- 拆除栈帧 ----
        LDR     R7,R5,#2        ; 恢复返回地址
        LDR     R5,R5,#1        ; 恢复旧帧指针
        ADD     R6,R6,#3        ; 弹出局部变量与 linkage（返回值除外）
        RET                     ; = JMP R7
```

当函数**没有局部变量**时（`translate.asm` 中的 `MAIN`），帧指针设为 `R5 = R6 - 1`：

```asm
        ADD     R6,R6,#-3       ; 3 个位置，全部是 linkage
        STR     R5,R6,#0        ; → 对应 R5+1
        ADD     R5,R6,#-1       ; R5 = R6 - 1
        STR     R7,R5,#2        ; → R5+2
```

两种写法的**偏移量完全一致**：`R5+1`、`R5+2`、`R5+3`、`R5+4` 的含义不变。

**注意**：`ADD R6,R6,#3` 之后 `R6` 正好指向**返回值**所在位置（`R5+3`），
所以调用者返回后可以直接读 `M[R6]` 取返回值，再用一条 `ADD R6, R6, #(nparams+1)` 弹干净。

#### 为什么参数从右往左压栈？

**因为 C 允许可变参数函数（如 `printf`）。** 被调用者需要能够**先找到第一个参数**
（即格式字符串），才能知道后面还有几个参数、各是什么类型。
把第一个参数放在**固定偏移**处（栈顶附近），就能保证无论传了多少个参数，
它总能被找到。如果从左往右压，第一个参数的位置会随参数个数变化，就找不到了。

```c
printf("%d %s %f\n", i, s, d);
/*      ↑ 第一个参数必须在固定偏移，才能解析出后面三个 */
```

#### 为什么同时需要 R5 和 R6？

`R6`（栈指针）在函数执行过程中会**不断移动**（每次压栈/弹栈都变）。
`R5`（帧指针）在函数建立栈帧后**保持不变**，因此可以用固定的偏移
（局部变量 `R5+0`、`R5-1`…，参数 `R5+4`、`R5+5`…）稳定地访问局部变量和参数。
这就是"用 R5 当基准，用 R6 当游标"。

#### 编译器优化：栈帧可能不存在

课程幻灯片明确指出：**函数内部的栈帧使用不是接口问题，所以编译器可以自由优化。**
编译器可能：

- 把变量放进**寄存器**而不是栈
- **不保存 R7**（如果该函数不调用任何子程序）
- **完全不建立栈帧**

这就是为什么用 GDB 调试优化过的代码（`-O2`）时，变量会显示 `<optimized out>`。
**调试时务必用 `-O0`。**

#### 返回局部变量指针 = 悬垂指针

```c
/* ❌ 未定义行为：返回指向已销毁栈帧的指针 */
int32_t *bad(void)
{
    int32_t local = 42;
    return &local;      /* local 随栈帧销毁而失效 */
}

/* ✅ 正确：用堆分配（调用者负责 free） */
int32_t *good(void)
{
    int32_t *p = malloc(sizeof(int32_t));
    if (p != NULL) {
        *p = 42;
    }
    return p;           /* 调用者必须 free */
}
```

---

### 10. 结构体、typedef 与信息隐藏

#### 定义与访问

```c
typedef struct {
    int32_t x;
    int32_t y;
} point_t;

point_t p   = { 3, 4 };
point_t *pp = &p;

p.x        /* 通过结构体变量访问成员 */
pp->x      /* 通过指针访问成员：-> 就是 (*pp).x 的简写 */
(*pp).x    /* 与上一行完全等价 */
```

#### 内存布局与填充（Padding）

```c
typedef struct {
    char a;      /* 偏移 0，1 字节 */
    /* 3 字节填充 */
    int  b;      /* 偏移 4，4 字节 */
    char c;      /* 偏移 8，1 字节 */
    /* 3 字节尾部填充（为了让数组中的下一个元素对齐） */
} padded_t;

/* sizeof(padded_t) == 12，而不是 1+4+1 = 6 */
```

```
偏移:   0    1    2    3    4    5    6    7    8    9   10   11
      +----+----+----+----+----+----+----+----+----+----+----+----+
      | a  | 填充    |填充| b                   | c  |  填充        |
      +----+----+----+----+----+----+----+----+----+----+----+----+
```

**调整成员顺序可以减小 sizeof**：

```c
typedef struct { char a; char c; int b; } packed_t;   /* sizeof == 8 */
```

> 这个技巧也是"为什么编译期能算出成员偏移"的关键：**偏移在编译期就是常量**，
> 所以 `p->b` 会编译成一条 `LDR R1, R0, #offset` 形式的指令，**运行时没有任何查找开销**。

#### `typedef` 与 `enum`

```c
/* 课程编码规范要求：定义 struct 和 enum 时总是使用 typedef */
typedef struct {
    float origin_x;
    float origin_y;
    float radius;
} circle;

typedef enum {
    APPLE,
    ORANGE,
    BANANA,
} fruit;
```

#### 信息隐藏：不透明类型（Opaque Type / Handle Idiom）

**头文件（接口，`mem220.h`）—— 只有声明，没有定义：**

```c
#if !defined(_MEM220_H)
#define _MEM220_H

#include <stdint.h>
#include <stddef.h>

/* 只前向声明，使用者看到的是一个不完整的类型 */
typedef struct memory_system memory_system_t;

memory_system_t *mem220_create(void);
void             mem220_destroy(memory_system_t *ms);
void            *mem220_allocate(memory_system_t *ms, size_t n_bytes);
void             mem220_free(memory_system_t *ms, void *ptr);

#endif /* !defined(_MEM220_H) */
```

**实现文件（`mem220.c`）—— 只有这里有真正的定义：**

```c
struct memory_system {
    /* 私有表示：使用者完全不知道这些字段的存在 */
    uint8_t *heap_start;
    size_t   total_size;
    /* ... 空闲链表等内部状态 ... */
};
```

**好处：**

1. **封装**：使用者无法依赖内部表示，实现可以自由改变而不破坏调用者。
2. **不变量**：所有修改都必须经过函数，函数可以维护数据结构不变量。
3. **可测试性**：有清晰的接口边界，便于单元测试。
4. **这就是"类"的雏形**：`memory_system_t *` 相当于 `this` 指针，
   每个函数第一个参数相当于一个"方法"。

#### 头文件模板（含 include guard）

```c
#if !defined(_MYMODULE_H)
#define _MYMODULE_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int32_t value;
    char    name[32];
} record_t;

/* 函数声明，每行加注释说明契约 */
int32_t record_init(record_t *rec, const char *name, int32_t value);
void    record_print(const record_t *rec);

#endif /* !defined(_MYMODULE_H) */
```

---

### 11. 动态内存管理

#### 四个函数的契约

| 函数 | 原型 | 成功 | 失败 | 注意 |
|---|---|---|---|---|
| `malloc` | `void *malloc(size_t n)` | 返回未初始化的 n 字节 | 返回 `NULL` | `malloc(0)` 返回值由实现定义，可能为 NULL |
| `calloc` | `void *calloc(size_t n, size_t sz)` | 返回**清零**的 n×sz 字节 | 返回 `NULL` | 会检查 n×sz 是否溢出 |
| `realloc` | `void *realloc(void *p, size_t n)` | 返回调整后的块 | 返回 `NULL`，**原块不变** | 必须用临时指针接返回值 |
| `free` | `void free(void *p)` | 释放 | — | `free(NULL)` 是合法的空操作 |

#### 标准使用模板

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(void)
{
    size_t n = 10;

    /* 1. 分配 */
    int32_t *arr = malloc(n * sizeof(int32_t));
    if (arr == NULL) {                       /* 2. 必须检查 */
        fprintf(stderr, "Out of memory\n");
        return 1;
    }

    /* 3. 使用 */
    for (size_t i = 0; i < n; i++) {
        arr[i] = (int32_t)(i * i);
    }

    /* 4. 释放 */
    free(arr);
    arr = NULL;                              /* 5. 置 NULL，防止悬垂指针 */

    return 0;
}
```

#### `sizeof` 的正确写法

```c
int32_t *a = malloc(10 * sizeof(int32_t));       /* ✅ 推荐：显式类型 */
int32_t *b = malloc(10 * sizeof(*b));            /* ✅ 也推荐：改类型时自动跟着变 */
int32_t *c = malloc(10 * sizeof(int));           /* ⚠️ 在本平台上恰好对，但不通用 */
int32_t *d = malloc(10 * 4);                     /* ❌ 魔法数字，禁止 */
```

#### 动态数组增长（倍增策略）

```c
typedef struct {
    int32_t *data;
    size_t   count;
    size_t   capacity;
} vec_t;

int32_t vec_push(vec_t *v, int32_t value)
{
    if (v->count == v->capacity) {
        size_t new_cap = (v->capacity == 0) ? 4 : v->capacity * 2;
        int32_t *tmp = realloc(v->data, new_cap * sizeof(int32_t));
        if (tmp == NULL) {
            return -1;                 /* 原数据完好，调用者可继续使用 */
        }
        v->data     = tmp;
        v->capacity = new_cap;
    }
    v->data[v->count++] = value;
    return 0;
}
```

**为什么倍增是 O(1) 摊还？** n 次 push 的总复制成本为
`1 + 2 + 4 + … + n/2 < n`，即 O(n)，摊到每次 push 就是 O(1)。
如果每次只加 1，总成本是 `1+2+…+n = O(n²)`，单次就是 O(n)。

#### 倍增策略的代价（课程幻灯片量化）

以 2 倍增长为例：

- **复制成本**：对 N 个元素，累计复制次数 ≤ 2N
- **空间浪费**：约 **38%**（因为平均只用了一半容量）

**删除**：如果不要求顺序，把最后一个元素复制到被删位置，然后 `count--`，是 O(1)。
如果要求保序，删除是 O(n)。

---

### 12. 动态内存分配器内部机制

#### 堆与 break

```
        ┌──────────────────────┐
        │  栈                  │ ← 向低地址增长
        │        ⋮             │
        │        ⋮             │
        │  堆                  │ ← 向高地址增长
        └──────────────────────┘ ← break（可用 sbrk 移动）
        │  全局数据             │
        │  代码                 │
        └──────────────────────┘
```

#### `sbrk` 系统调用

```c
#include <unistd.h>
void *sbrk(intptr_t increment);
```

- 请求把 break 移动 `increment` 字节。
- **返回移动前**的 break 地址（即新分配区域的起始地址）。
- 失败时返回 `(void *)-1`。
- `increment` 可以是负数，用于**收缩**堆。

> 为什么参数是 `intptr_t` 而不是 `int`？因为 64 位地址空间下 `int`（32 位）装不下指针。

#### 分配器的核心数据结构

```
空闲链表（free list）—— 每个空闲块头部藏有元数据：
        ┌────────────┬──────────────────────────┐
        │ size | next│      空闲空间             │
        └────────────┴──────────────────────────┘
             元数据            可供分配

分配后的堆布局：
   ┌────────┬──────┬────────────┬──────┬────────┬──────┐
   │ 元数据  │ 已用 │   空闲      │ 元数据│  已用  │ 空闲 │
   └────────┴──────┴────────────┴──────┴────────┴──────┘
```

**为什么 `free(ptr)` 知道要释放多大？** 因为 `malloc` 在返回给用户的指针**前面**
存了块的元数据（大小、状态）。`free` 往后退几个字节就能读到它。
这也解释了为什么 **`free` 只能释放 `malloc` 返回的原始指针**——
传一个偏移过的指针（如 `free(p + 1)`）会读到错误的元数据并破坏堆。

#### 分配策略

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **first-fit** | 用第一个足够大的空闲块 | 快 | 低地址处产生大量小碎片 |
| **best-fit** | 用最小的足够大的空闲块 | 空间利用率好 | 需要搜索整个空闲表；产生难以利用的微小碎片 |
| **worst-fit** | 用最大的空闲块 | 留下的剩余块较大 | 大块很快被消耗 |

课程讲授的是 **best-fit logarithmic allocator**（对数最佳适配分配器），
它在 20 世纪被广泛使用了几十年。其思想是：把空闲块按**大小的对数**分桶，
这样寻找"最小的足够大"的块时，只需检查比请求大小略大的桶，而不是扫描整个链表。

#### 分裂与合并

- **分裂 (splitting)**：一个空闲块比请求大得多时，切成"分配出去的部分 + 剩余空闲块"。
  若剩余部分太小（小于最小块大小），干脆整块给出去。
- **合并 (coalescing)**：`free` 时，若相邻块也是空闲的，把它们合并成一个大块，
  否则会产生**外部碎片**（总空闲够，但没有一个连续块够大）。

#### 两类碎片

| | 定义 | 成因 | 缓解 |
|---|---|---|---|
| **内部碎片** | 分配出去的块**内部**未被使用的字节 | 对齐、最小块大小、分配粒度 | 减小粒度 |
| **外部碎片** | 空闲空间**总量够**但没有一个足够大的连续块 | 分配/释放的时序 | 合并相邻空闲块 |

---

### 13. 数据结构速查

#### 动态数组 vs 链表

| 维度 | 动态数组 (`vec_t`) | 链表 (`node_t *`) |
|---|---|---|
| 随机访问 `[i]` | **O(1)** | O(n) |
| 头部插入 | O(n) | **O(1)** |
| 尾部插入 | O(1) 摊还 | O(1)（带尾指针） |
| 中间插入/删除 | O(n) | O(1)（已有前驱指针时） |
| 按值查找 | O(n) | O(n) |
| 每元素额外内存 | 无 | **8 字节**（next 指针） |
| 缓存局部性 | **好**（连续） | 差（分散在堆上） |
| 增长能力 | 受连续空间限制 | 不受限制（只要有零散空间） |
| 适用场景 | 频繁随机访问、元素数可预估 | 频繁中间插删、元素数未知且大 |

#### 链表节点定义与核心操作

```c
typedef struct node {
    int32_t      value;
    struct node *next;      /* 自引用：必须是指针，不能是 struct node */
} node_t;
```

```
head
  |
  v
+------+------+    +------+------+    +------+------+
|  10  |  ---|--->|  20  |  ---|--->|  30  | NULL |
+------+------+    +------+------+    +------+------+
 0x1000            0x2040            0x3080
 （节点在堆上分散，靠指针串起来，因此不能做指针算术）
```

**遍历模板**：

```c
for (node_t *p = head; p != NULL; p = p->next) {
    /* 处理 p->value */
}
```

**头插法（O(1)）**：

```c
node_t *push_front(node_t *head, int32_t value)
{
    node_t *n = malloc(sizeof(node_t));
    if (n == NULL) {
        return head;            /* 失败：返回原链表 */
    }
    n->value = value;
    n->next  = head;
    return n;                   /* 新头 */
}
```

**删除（指针的指针写法，消除"删除头结点"的特例）**：

```c
int32_t list_remove(node_t **head, int32_t value)
{
    node_t **link = head;               /* link 指向"指向当前节点的指针" */
    while (*link != NULL) {
        if ((*link)->value == value) {
            node_t *victim = *link;
            *link = victim->next;       /* 让前驱（或 head）跳过它 */
            free(victim);
            return 0;
        }
        link = &(*link)->next;
    }
    return -1;                          /* 未找到 */
}
```

**销毁整个链表（先存 next 再 free）**：

```c
void list_destroy(node_t *head)
{
    while (head != NULL) {
        node_t *next = head->next;      /* ⚠️ 必须先保存！ */
        free(head);
        head = next;
    }
}
```

> **经典 bug**：先 `free(head)` 再读 `head->next` 就是 use-after-free。
> 顺序必须是"存 next → free 当前 → 前进"。

#### 二叉树

```c
typedef struct tree_node {
    int32_t          value;
    struct tree_node *left;
    struct tree_node *right;
} tree_node_t;
```

**BST 不变式**：对每个节点，**左子树所有值 < 节点的值 < 右子树所有值**。

```
              50
            /    \
          30      70
         /  \    /  \
       20   40  60   80

中序遍历 (inorder, 左-根-右): 20 30 40 50 60 70 80   ← 升序！
前序遍历 (preorder, 根-左-右): 50 30 20 40 70 60 80
后序遍历 (postorder, 左-右-根): 20 40 30 60 80 70 50
```

| 遍历 | 顺序 | 典型用途 |
|---|---|---|
| 前序 | 根 → 左 → 右 | 复制树、序列化、打印目录结构 |
| 中序 | 左 → 根 → 右 | **BST 升序输出** |
| 后序 | 左 → 右 → 根 | **释放整棵树**、计算子树聚合值 |

**递归模板**：

```c
void inorder(const tree_node_t *root)
{
    if (root == NULL) {          /* 基本情况：空树 */
        return;
    }
    inorder(root->left);
    printf("%d ", root->value);
    inorder(root->right);
}

void tree_destroy(tree_node_t *root)
{
    if (root == NULL) {
        return;
    }
    tree_destroy(root->left);    /* 必须先释放子树 */
    tree_destroy(root->right);
    free(root);                  /* 最后释放自己（后序） */
}
```

**BST 查找**：

```c
const tree_node_t *bst_find(const tree_node_t *root, int32_t key)
{
    while (root != NULL) {
        if (key == root->value) { return root; }
        root = (key < root->value) ? root->left : root->right;
    }
    return NULL;
}
```

> **复杂度**：O(树高)。平衡树高 ≈ log₂n；但**退化成链表的树高 = n**，
> 此时查找退化为 O(n)。这就是为什么需要平衡树。

---

### 14. 递归速查

#### 递归的两个必要部分

1. **基本情况 (base case)**：不再递归、直接返回的情形。**没有它 = 无限递归 = 栈溢出。**
2. **递归情况 (recursive case)**：把问题化归为**更小的同类问题**。

#### 标准模板

```c
int32_t factorial(int32_t n)
{
    if (n <= 1) {              /* 基本情况 */
        return 1;
    }
    return n * factorial(n - 1);   /* 递归情况 */
}
```

#### 递归的代价：栈帧链

```
factorial(4) 调用过程（栈向低地址增长）：

调用 factorial(4)   →  调用 factorial(3)  →  调用 factorial(2)  →  调用 factorial(1)
                                                                        ↓ 返回 1
                                                                    ↓ 返回 2*1=2
                                                                ↓ 返回 3*2=6
                                                            ↓ 返回 4*6=24

栈帧布局（每一层都有自己独立的一份 n）：
  高地址 ┌─────────────────┐
         │ factorial(4)    │  n=4，等着 3 的结果
         ├─────────────────┤
         │ factorial(3)    │  n=3，等着 2 的结果
         ├─────────────────┤
         │ factorial(2)    │  n=2，等着 1 的结果
         ├─────────────────┤
         │ factorial(1)    │  n=1，直接返回 1  ← 栈顶（R6 指向这里）
  低地址 └─────────────────┘
```

**关键洞察**：递归的"记忆"就是**栈**。每层调用有自己的局部变量和参数副本，
所以递归天然支持"回到上一层继续算"。这也是为什么递归深度受**栈大小**限制
（默认约 8 MB，即约 10 万层简单调用）。

#### 递归 vs 迭代

| 维度 | 递归 | 迭代 |
|---|---|---|
| 代码可读性 | 树、回溯、分治类问题**明显更好** | 线性扫描类问题更好 |
| 空间开销 | O(深度) 栈帧 | O(1) |
| 速度 | 较慢（函数调用开销） | 较快 |
| 栈溢出风险 | **有**（深度受栈大小限制） | 无 |
| 适用 | 自相似结构（树、分形、回溯） | 简单累积、线性遍历 |

#### 尾递归 (Tail Recursion)

递归调用是函数的**最后一个动作**（返回值直接就是递归调用的返回值）时，
编译器**可能**优化成循环，消除栈帧增长。但 **C 标准不保证**这种优化，
所以不要把"不会栈溢出"当作前提。

```c
/* 尾递归形式：把累积量作为参数传下去 */
int32_t fact_tail(int32_t n, int32_t acc)
{
    if (n <= 1) {
        return acc;
    }
    return fact_tail(n - 1, n * acc);   /* 递归调用是最后一步 */
}
```

#### 回溯 (Backtracking) 模板

```c
/*
 * 通用回溯模板：
 *   1. 判断是否已经找到解 / 走不通 → 返回
 *   2. 标记当前位置（做选择）
 *   3. 递归探索所有邻居
 *   4. 撤销标记（回溯）
 */
int32_t solve(grid_t *g, int32_t r, int32_t c)
{
    /* 越界或已访问或不可通行 */
    if (r < 0 || r >= g->rows || c < 0 || c >= g->cols) { return 0; }
    if (g->cell[r][c] != OPEN)                          { return 0; }

    /* 到达目标 */
    if (r == g->goal_r && c == g->goal_c)               { return 1; }

    g->cell[r][c] = VISITED;        /* 做选择：标记 */

    /* 探索四个方向 */
    if (solve(g, r + 1, c) || solve(g, r - 1, c) ||
        solve(g, r, c + 1) || solve(g, r, c - 1)) {
        return 1;                   /* 找到解，逐层返回 */
    }

    g->cell[r][c] = OPEN;           /* 撤销选择：回溯 */
    return 0;
}
```

> **回溯的要点**：第 4 步的"撤销"必须与第 2 步的"标记"严格配对。
> 忘了撤销，算法就只能探索一条路径；撤销错了，会重复访问导致死循环。

#### 递归树与复杂度：朴素 Fibonacci

```c
int32_t fib(int32_t n)
{
    if (n < 2) { return n; }
    return fib(n - 1) + fib(n - 2);
}
```

```
                    fib(5)
                   /      \
              fib(4)        fib(3)
             /     \       /     \
         fib(3)  fib(2)  fib(2)  fib(1)
         /    \   /   \   /   \
     fib(2) fib(1) ...  ...  ...
      /  \
  fib(1) fib(0)

节点总数 ≈ 2^n  →  时间复杂度 O(2^n)（实际约 O(1.618^n)）
注意 fib(3) 被计算了 2 次，fib(2) 被计算了 3 次 —— 大量重复子问题！
```

**修复**：动态规划/记忆化（memoization），把已算过的结果存起来，降到 O(n)。

#### Towers of Hanoi

```
n 个盘子需要 2^n − 1 次移动。
移动 n 个盘子的方法：
   1. 把上面 n−1 个盘子从 A 移到 B（借助 C）
   2. 把最大的盘子从 A 移到 C
   3. 把 n−1 个盘子从 B 移到 C（借助 A）
```

---

### 15. 排序算法速查

#### 总览对比表

| 算法 | 最好 | 平均 | 最坏 | 额外空间 | 稳定 | 特点 |
|---|---|---|---|---|---|---|
| **插入排序** | O(n) | O(n²) | O(n²) | O(1) | ✅ | 近乎有序时极快；小数组最快 |
| **选择排序** | O(n²) | O(n²) | O(n²) | O(1) | ❌ | 交换次数最少（恰好 n−1 次） |
| **冒泡排序** | O(n) | O(n²) | O(n²) | O(1) | ✅ | 加提前退出标志后近乎有序时 O(n) |
| **归并排序** | O(n log n) | O(n log n) | O(n log n) | **O(n)** | ✅ | 稳定；可外排序；需要额外数组 |
| **快速排序** | O(n log n) | O(n log n) | **O(n²)** | O(log n) | ❌ | 实际最快；最坏出现在已排序+糟糕选主元 |
| **堆排序** | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ | 原地且最坏有保证 |
| **计数排序** | O(n+k) | O(n+k) | O(n+k) | O(k) | ✅ | 非比较；仅适合小范围整数 |

> **比较排序的下界**：任何只通过"比较两个元素"来决定顺序的算法，
> 最坏情况下至少要 Ω(n log n) 次比较（决策树有 n! 个叶子，高度 ≥ log₂(n!) ≈ n log n）。
> 这就是为什么计数排序必须"不比较"才能打破这个界。

#### 插入排序模板

```c
void insertion_sort(int32_t *a, size_t n)
{
    for (size_t i = 1; i < n; i++) {
        int32_t key = a[i];        /* 取出待插入的元素 */
        size_t  j   = i;
        while (j > 0 && a[j - 1] > key) {
            a[j] = a[j - 1];       /* 更大的元素整体右移一格 */
            j--;
        }
        a[j] = key;                /* 插到正确位置 */
    }
}
```

#### 归并排序模板

```c
static void merge(int32_t *a, int32_t *tmp, size_t lo, size_t mid, size_t hi)
{
    size_t i = lo, j = mid, k = lo;

    while (i < mid && j < hi) {
        /* 用 <= 保证稳定性：相等时优先取左边的 */
        tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    }
    while (i < mid) { tmp[k++] = a[i++]; }
    while (j < hi)  { tmp[k++] = a[j++]; }

    for (size_t m = lo; m < hi; m++) {
        a[m] = tmp[m];
    }
}

void merge_sort_rec(int32_t *a, int32_t *tmp, size_t lo, size_t hi)
{
    if (hi - lo < 2) {             /* 0 或 1 个元素：已有序 */
        return;
    }
    size_t mid = lo + (hi - lo) / 2;
    merge_sort_rec(a, tmp, lo, mid);
    merge_sort_rec(a, tmp, mid, hi);
    merge(a, tmp, lo, mid, hi);
}

void merge_sort(int32_t *a, size_t n)
{
    int32_t *tmp = malloc(n * sizeof(int32_t));
    if (tmp == NULL) {
        return;                    /* 分配失败：放弃（或回退到插入排序） */
    }
    merge_sort_rec(a, tmp, 0, n);
    free(tmp);
}
```

#### 快速排序模板（Lomuto 分区）

```c
static size_t partition(int32_t *a, size_t lo, size_t hi)
{
    /* hi 位置作为主元 */
    int32_t pivot = a[hi];
    size_t  i     = lo;            /* a[lo..i) 都是 <= pivot */

    for (size_t j = lo; j < hi; j++) {
        if (a[j] <= pivot) {
            int32_t t = a[i]; a[i] = a[j]; a[j] = t;
            i++;
        }
    }
    int32_t t = a[i]; a[i] = a[hi]; a[hi] = t;   /* 主元归位 */
    return i;                      /* 主元的最终下标 */
}

void quick_sort_rec(int32_t *a, size_t lo, size_t hi)
{
    if (hi <= lo || hi - lo < 2) {
        return;
    }
    size_t p = partition(a, lo, hi - 1);
    if (p > lo)     { quick_sort_rec(a, lo, p); }
    if (p + 1 < hi) { quick_sort_rec(a, p + 1, hi); }
}
```

> **主元选择很重要**：总是取最后一个元素作主元，在**已排序**输入上会退化为 O(n²)，且递归深度 O(n)。
> 实践中用"三者取中"（median-of-three）或随机化来避免。

#### `qsort` 与比较函数

```c
#include <stdlib.h>

/* 比较函数的契约：返回 <0 表示 a 应排在 b 前；0 表示等价；>0 表示 a 应排在 b 后 */
int cmp_int_asc(const void *pa, const void *pb)
{
    int32_t a = *(const int32_t *)pa;
    int32_t b = *(const int32_t *)pb;
    if (a < b) { return -1; }
    if (a > b) { return  1; }
    return 0;
    /* ⚠️ 不要写 return a - b; —— 对 int32_t 可能溢出！ */
}

/* 调用 */
int32_t arr[100];
qsort(arr, 100, sizeof(arr[0]), cmp_int_asc);
```

**按结构体不同字段排序**：

```c
typedef struct { char name[32]; int32_t score; } student_t;

static int cmp_by_score(const void *pa, const void *pb)
{
    const student_t *a = pa;
    const student_t *b = pb;
    if (a->score < b->score) { return -1; }
    if (a->score > b->score) { return  1; }
    return strcmp(a->name, b->name);       /* 分数相同时按名字，保证确定性 */
}

qsort(students, n, sizeof(student_t), cmp_by_score);
```

---

### 16. 文件 I/O 速查

#### 标准流

| 流 | 含义 | 缓冲 |
|---|---|---|
| `stdin` | 标准输入（默认键盘 / 管道 / 重定向） | 行缓冲或全缓冲 |
| `stdout` | 标准输出（默认终端 / 管道 / 重定向） | 行缓冲（终端）/ 全缓冲（重定向） |
| `stderr` | 标准错误（**不缓冲**） | 无缓冲 —— 所以崩溃前的错误信息总能显示 |

#### `fopen` 模式

| 模式 | 含义 | 文件不存在时 | 文件存在时 |
|---|---|---|---|
| `"r"` | 只读 | 返回 `NULL` | 从开头读 |
| `"w"` | 只写 | 创建 | **截断为 0 字节** |
| `"a"` | 追加写 | 创建 | 从末尾写 |
| `"r+"` | 读写 | 返回 `NULL` | 从开头读写（不截断） |
| `"w+"` | 读写 | 创建 | 截断 |
| `"a+"` | 读 + 追加 | 创建 | 读从开头，写从末尾 |
| 加 `b` | 二进制模式（`"rb"`, `"wb"` 等） | | Unix 上等同于文本模式 |

#### 常用函数

| 函数 | 用途 | 返回值 |
|---|---|---|
| `fopen(path, mode)` | 打开 | `FILE *` 或 `NULL` |
| `fclose(fp)` | 关闭（**必须**） | 0 成功，`EOF` 失败 |
| `fprintf(fp, fmt, ...)` | 格式化写 | 写入的字符数，负值表示错误 |
| `fscanf(fp, fmt, ...)` | 格式化读 | **成功赋值的项数**（不是字符数！） |
| `fgetc(fp)` / `getc(fp)` | 读一个字符 | `int`（**必须是 int 才能装下 EOF**） |
| `fputc(c, fp)` | 写一个字符 | 写的字符，或 `EOF` |
| `fgets(buf, n, fp)` | 读一行（最多 n−1 字符） | `buf` 或 `NULL`（EOF/错误） |
| `fputs(s, fp)` | 写字符串（不加换行） | 非负值或 `EOF` |
| `fread(p, sz, n, fp)` | 二进制读 | **实际读到的元素个数** |
| `fwrite(p, sz, n, fp)` | 二进制写 | 实际写入的元素个数 |
| `feof(fp)` | 是否已到文件尾 | 非零表示已到尾 |
| `ferror(fp)` | 是否出错 | 非零表示出错 |
| `fflush(fp)` | 强制把缓冲区写出 | 0 成功 |
| `rewind(fp)` / `fseek` | 移动文件位置 | — |
| `remove(path)` | 删除文件 | 0 成功 |
| `perror(s)` | 打印 `s` + `errno` 对应的错误信息 | — |

#### 读取整个文件的模板

```c
#include <stdio.h>
#include <stdlib.h>

int32_t read_whole_file(const char *path, char **out, size_t *out_len)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        perror(path);              /* 打印 "path: No such file or directory" */
        return -1;
    }

    size_t cap  = 4096;
    size_t len  = 0;
    char  *buf  = malloc(cap);
    if (buf == NULL) {
        fclose(fp);
        return -1;
    }

    size_t got;
    while ((got = fread(buf + len, 1, cap - len, fp)) > 0) {
        len += got;
        if (len == cap) {                    /* 缓冲区满了，扩容 */
            char *tmp = realloc(buf, cap * 2);
            if (tmp == NULL) {
                free(buf);
                fclose(fp);
                return -1;
            }
            buf = tmp;
            cap *= 2;
        }
    }

    if (ferror(fp)) {                        /* 区分 EOF 与真错误 */
        free(buf);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *out     = buf;
    *out_len = len;
    return 0;
}
```

#### 逐行读取的模板

```c
char line[256];
while (fgets(line, sizeof(line), fp) != NULL) {
    /* 去掉行尾换行 */
    size_t n = strlen(line);
    if (n > 0 && line[n - 1] == '\n') {
        line[n - 1] = '\0';
    } else if (n == sizeof(line) - 1) {
        /* ⚠️ 行太长，缓冲区被填满且没有读到换行 —— 剩余部分还在文件里！
           必须把这一行剩下的字符读掉，否则会被当成下一行。 */
        int c;
        while ((c = fgetc(fp)) != '\n' && c != EOF) { }
    }
    /* 处理 line */
}
```

#### `while (!feof(f))` 为什么是错的

```c
/* ❌ 错误写法 */
while (!feof(fp)) {
    fgets(line, sizeof(line), fp);
    printf("%s", line);          /* 最后一次循环会重复打印上一行！ */
}

/* ✅ 正确写法：用读取函数的返回值判断 */
while (fgets(line, sizeof(line), fp) != NULL) {
    printf("%s", line);
}
```

**原因**：`feof()` 只有在**尝试读取并撞到文件尾之后**才变成真。
它是"**已经**到过 EOF"的标志，不是"**将要**到 EOF"的预测。
所以 `while (!feof(f))` 会在最后一次多转一圈，此时 `fgets` 什么都没读到、
缓冲区仍是上次的内容，于是重复输出。

#### 命令行重定向与管道

```bash
./prog < input.txt            # stdin 来自文件
./prog > output.txt           # stdout 写入文件（截断）
./prog >> output.txt          # stdout 追加
./prog 2> errors.txt          # stderr 重定向
./prog > out.txt 2>&1         # stdout 和 stderr 都到同一个文件
./prog < in.txt | sort | uniq -c    # 管道：程序的 stdout 接下一个程序的 stdin
```

> **程序设计原则**：让程序默认从 `stdin` 读、向 `stdout` 写、错误写到 `stderr`。
> 这样它就能自然地参与 Unix 管道，无需知道文件名。

#### 解析文本的模板

```c
/* 用 strtok 切分（注意：会破坏原字符串，把分隔符改成 '\0'） */
char *save = NULL;
for (char *tok = strtok_r(line, " \t\n", &save);
     tok != NULL;
     tok = strtok_r(NULL, " \t\n", &save)) {
    /* 处理 tok */
}

/* 用 strtol 安全地把字符串转成整数 */
char *end;
errno = 0;
long v = strtol(tok, &end, 10);
if (end == tok || *end != '\0' || errno == ERANGE) {
    /* 转换失败或溢出 */
}

/* 用 sscanf 从字符串里提取格式化字段（比 strtok 更适合固定格式） */
int y, m, d;
if (sscanf(line, "%d-%d-%d", &y, &m, &d) == 3) {
    /* 成功解析 3 个字段 */
}
```

---

### 17. 格式化输入输出速查

#### `printf` / `scanf` 格式说明符（官方速查表）

| 说明符 | 格式描述 | `printf` 类型 | `scanf` 类型 |
|---|---|---|---|
| `%c` | ASCII 字符 | `char` | `char *` |
| `%d` | 有符号十进制整数 | `int` | `int *` |
| `%f` | 带小数部分的十进制 | `float` | `float *` |
| `%hd` | 有符号十进制整数 | `short` | `short *` |
| `%hhd` | 有符号十进制整数 | `char` | `char *` |
| `%hhu` | 无符号十进制整数 | `unsigned char` | `unsigned char *` |
| `%hhx` | 十六进制 | `unsigned char` | `unsigned char *` |
| `%hu` | 无符号十进制整数 | `unsigned short` | `unsigned short *` |
| `%hx` | 十六进制 | `unsigned short` | `unsigned short *` |
| `%ld` | 有符号十进制整数 | `long` | `long *` |
| `%lf` | 带小数部分的十进制 | `double` | `double *` |
| `%lu` | 无符号十进制整数 | `unsigned long` | `unsigned long *` |
| `%lx` | 十六进制 | `unsigned long` | `unsigned long *` |
| `%p` | 十六进制地址 | `T *`（任意指针） | `T **` |
| `%s` | ASCII 字符串 | `char *` | `char *` |
| `%u` | 无符号十进制整数 | `unsigned int` | `unsigned int *` |
| `%x` | 十六进制 | `unsigned int` | `unsigned int *` |

#### 关键差异：`printf` 传值，`scanf` 传地址

```c
int32_t v;
scanf("%d", &v);          /* ✅ scanf 需要地址 */
printf("%d\n", v);        /* ✅ printf 需要值 */
scanf("%d", v);           /* ❌ 漏了 & —— 编译警告，运行时段错误 */
```

**唯一的例外**：`%s` 在 `scanf` 中传的**已经是数组名**（即地址），所以不加 `&`：

```c
char name[64];
scanf("%s", name);        /* ✅ name 本身就是地址 */
scanf("%s", &name);       /* 类型不同（char(*)[64]）但地址值相同，能跑，不推荐 */
```

#### 三个经典陷阱

```c
/* ① scanf 的 %c 会匹配空格和换行 —— 这通常不是你要的 */
scanf("%c", &ch);         /* ❌ 可能读到上一行残留的 '\n' */
scanf(" %c", &ch);        /* ✅ 格式串前面加一个空格：跳过所有前导空白 */

/* ② scanf 的 %s 不检查缓冲区大小 —— 缓冲区溢出风险 */
char buf[10];
scanf("%s", buf);                 /* ❌ 输入 "verylongstring" 就溢出了 */
scanf("%9s", buf);                /* ✅ 限制宽度为 sizeof(buf)-1 */

/* ③ scanf 的 %f 对应 float*，但 double 必须用 %lf */
float  f;  scanf("%f",  &f);      /* ✅ */
double d;  scanf("%f",  &d);      /* ❌ 类型不匹配，未定义行为 */
double d;  scanf("%lf", &d);      /* ✅ */
```

#### 检查返回值（必须做）

```c
/* printf 返回写入的字符数，负值表示出错 */
int n = printf("value = %d\n", v);
if (n < 0) {
    perror("printf");
}

/* scanf 返回成功赋值的项数，不是字符数！ */
int count = scanf("%d %d", &a, &b);
if (count != 2) {
    /* 可能是输入格式不对，或者提前遇到 EOF */
    fprintf(stderr, "Expected two integers, got %d\n", count);
}
```

#### 常用格式控制

```c
printf("%5d\n",   42);       /* 宽度 5，右对齐："   42" */
printf("%-5d|\n", 42);       /* 左对齐："42   |" */
printf("%05d\n",  42);       /* 补零："00042" */
printf("%.3f\n",  3.14159);  /* 3 位小数："3.142" */
printf("%8.2f\n", 3.14159);  /* 宽度 8，2 位小数："    3.14" */
printf("%x\n",    255);      /* 十六进制："ff" */
printf("%#x\n",   255);      /* 带前缀："0xff" */
printf("%X\n",    255);      /* 大写："FF" */
printf("%e\n",    1234.5);   /* 科学计数法 */
printf("%%\n",    0);        /* 输出一个百分号 */
printf("%zu\n",   sizeof(int)); /* size_t 必须用 %zu */
```

---

### 18. LC-3 汇编速查

#### 寄存器

| 寄存器 | 名称与用途 |
|---|---|
| `R0`–`R3` | 通用；**caller-saved**；用于传参数和返回值 |
| `R4` | 全局数据指针 (global data pointer) |
| `R5` | 帧指针 (frame pointer) |
| `R6` | 栈指针 (stack pointer) |
| `R7` | 返回地址 (return address) |

#### 内存映射 I/O 寄存器

| 地址 | 名称 | 作用 |
|---|---|---|
| `xFE00` | KBSR (Keyboard Status Register) | **bit 15** = 1 表示有新按键可读 |
| `xFE02` | KBDR (Keyboard Data Register) | **bit 7:0** = 按键的 ASCII 码 |
| `xFE04` | DSR (Display Status Register) | **bit 15** = 1 表示可以写下一个字符 |
| `xFE06` | DDR (Display Data Register) | **bit 7:0** = 要显示的字符 |

#### 轮询 I/O 模板

```asm
; ---- 从键盘读一个字符（busy-wait）----
POLL_IN
        LDI     R1, KBSR        ; R1 = 键盘状态寄存器的内容
        BRz     POLL_IN         ; 若 bit15 == 0（无按键），继续轮询
        LDI     R0, KBDR        ; 读入字符到 R0
        RET

KBSR    .FILL   xFE00
KBDR    .FILL   xFE02
```

```asm
; ---- 向显示器写一个字符（busy-wait）----
POLL_OUT
        ST      R0, SAVE_R0     ; 保存要输出的字符（LDI 会破坏 R1，但 R0 需保留）
        LDI     R1, DSR
        BRz     POLL_OUT        ; 若显示器忙，继续等待
        LD      R0, SAVE_R0
        STI     R0, DDR         ; 写字符到显示数据寄存器
        RET

DSR     .FILL   xFE04
DDR     .FILL   xFE06
SAVE_R0 .BLKW   1
```

#### 常用指令

```asm
; ---- 算术与逻辑 ----
ADD     R0, R1, R2              ; R0 = R1 + R2（寄存器相加）
ADD     R0, R1, #5              ; R0 = R1 + 5（立即数，5 位补码，范围 -16..15）
AND     R0, R1, R2              ; R0 = R1 & R2
NOT     R0, R1                  ; R0 = ~R1
; 注意：没有 SUB、MUL、DIV —— 用 ADD 配合取负实现

; ---- 数据搬移 ----
LD      R0, LABEL               ; R0 = M[PC + offset]（PC 相对寻址，直接加载）
LDI     R0, LABEL               ; R0 = M[M[PC + offset]]（间接加载，两次访存）
LDR     R0, R1, #4              ; R0 = M[R1 + 4]（基址 + 偏移）
LEA     R0, LABEL               ; R0 = PC + offset（加载地址本身，不访存）
ST      R0, LABEL               ; M[PC + offset] = R0
STI     R0, LABEL               ; M[M[PC + offset]] = R0（间接存储）
STR     R0, R1, #4              ; M[R1 + 4] = R0

; ---- 控制转移 ----
BRnzp   LABEL                   ; 无条件跳转
BRz     LABEL                   ; 零则跳转（Z=1，即上一次运算结果为 0）
BRnp    LABEL                   ; 非零则跳转
BRn     LABEL                   ; 负数则跳转
BRp     LABEL                   ; 正数则跳转
JMP     R1                      ; 跳转到 R1 中的地址
JSR     LABEL                   ; 子程序调用；R7 = 返回地址（PC 相对，11 位偏移）
JSRR    R1                      ; 子程序调用；R7 = 返回地址，跳转到 R1（5 位寄存器）
RET                             ; = JMP R7，从子程序返回

; ---- TRAP ----
TRAP    x20                     ; GETC：读一个字符到 R0
TRAP    x21                     ; OUT：输出 R0 中的字符
TRAP    x22                     ; PUTS：输出以 R0 为首地址的字符串
TRAP    x23                     ; IN：提示并读入一个字符
TRAP    x24                     ; PUTSP：输出打包字符串（两字符一字节对）
TRAP    x25                     ; HALT：停止机器

; ---- 伪指令 ----
.FILL   value                   ; 在当前位置放一个字
.BLKW   n                       ; 保留 n 个字（Block Word），初值为 0
.STRINGZ "text"                 ; 存放 NUL 结尾的字符串
.ORIG   x3000                   ; 程序起始地址
.END                            ; 源文件结束
```

#### 压栈与弹栈

```asm
; ---- PUSH R0 ----
        ADD     R6, R6, #-1     ; 栈向低地址增长：先移动指针
        STR     R0, R6, #0      ; 再存数据

; ---- POP R0 ----
        LDR     R0, R6, #0      ; 先取数据
        ADD     R6, R6, #1      ; 再移动指针
```

#### 完整的子程序框架

```asm
; =========================================================
; 子程序：ADD3 —— 返回三个参数之和
; 输入：R0, R1, R2
; 输出：R0
; 帧布局：R5+0 = 无局部变量；R5+1 旧帧指针；R5+2 返回地址；R5+3 返回值
; =========================================================
ADD3
        ; ---- 建立栈帧 ----
        ADD     R6,R6,#-3       ; 3 个位置：3 个 linkage（本函数无局部变量）
        STR     R5,R6,#0        ; 保存旧帧指针（对应 R5+1）
        ADD     R5,R6,#-1       ; 设置帧指针：R5 = R6 - 1
        STR     R7,R5,#2        ; 保存返回地址（对应 R5+2）

        ; ---- 函数体 ----
        ADD     R0, R0, R1      ; R0 = R0 + R1
        ADD     R0, R0, R2      ; R0 = R0 + R2
        STR     R0,R5,#3        ; 写返回值（对应 R5+3）

        ; ---- 拆除栈帧 ----
        LDR     R7,R5,#2        ; 恢复返回地址
        LDR     R5,R5,#1        ; 恢复旧帧指针
        ADD     R6,R6,#3        ; 弹出 linkage（返回值留在栈上给调用者读）
        RET                     ; 返回：JMP R7
```

若函数**有 N 个局部变量**，通用形式如下（`N` 必须是编译期常量）：

```asm
; 设 N = 局部变量个数
        ADD     R6,R6,#-(3+N)   ; 一次分配：3 个 linkage + N 个局部变量
        STR     R5,R6,#N        ; 把【旧】帧指针存到 R6+N（即未来的 R5+1）
        ADD     R5,R6,#(N-1)    ; 设置帧指针：R5 = R6 + (N-1)
        STR     R7,R5,#2        ; 保存返回地址 → R5+2
        ; 此时 R5+0, R5-1, …, R5-(N-1) 就是 N 个局部变量
```

**代入验证**（与课程真实代码逐字吻合）：

| N | `ADD R6,R6,#-(3+N)` | `STR R5,R6,#N` | `ADD R5,R6,#(N-1)` | 对应真实代码 |
|---|---|---|---|---|
| 1 | `ADD R6,R6,#-4` | `STR R5,R6,#1` | `ADD R5,R6,#0` | `translate.asm` 的 `FIND_ABS` ✅ |
| 0 | `ADD R6,R6,#-3` | `STR R5,R6,#0` | `ADD R5,R6,#-1` | `translate.asm` 的 `MAIN` ✅ |

> **注意保存顺序**：必须**先用 `R6` 作基址保存旧 `R5`**，然后才 `ADD R5,...` 设置新帧指针。
> 反过来写会把新 `R5` 当成旧值存起来，`RET` 时跳回一个错误的位置。

> **两条不变的规则**：① `R5+1`/`R5+2`/`R5+3` 永远是旧帧指针/返回地址/返回值，
> 与局部变量个数无关；② `R5+4` 起永远是第一个参数。

---

### 19. C 到 LC-3 翻译模板

> **这是 ECE 220 考试的核心能力**（教学目标第 1 条明确要求 "Be able to perform such a
> transformation manually"）。下面给出最常用的翻译模式。

#### 变量声明的翻译

```c
int32_t a;          /* 局部变量 */
int32_t b = 5;
```

```asm
; 假设 R5 指向局部变量底部
; 局部变量占 R5+0、R5-1、R5-2 …（向下增长）；R5+1 起是 linkage，不可用作局部变量！
; a 在 R5+0，b 在 R5-1
        AND     R0, R0, #0      ; R0 = 0（清空寄存器）
        STR     R0, R5, #0      ; a = 0（未初始化的局部变量内容不确定，这里示意）
        ADD     R0, R0, #5      ; R0 = 5
        STR     R0, R5, #-1     ; b = 5
```

#### 赋值的翻译

```c
a = b + c;
```

```asm
        LDR     R0, R5, #0      ; R0 = a
        LDR     R1, R5, #-1     ; R1 = b
        LDR     R2, R5, #-2     ; R2 = c
        ADD     R0, R0, R1      ; R0 = a + b
        ADD     R0, R0, R2      ; R0 = (a + b) + c
        STR     R0, R5, #-3     ; 结果存入临时/目标
```

#### `if` 语句的翻译

```c
if (a < b) {
    x = 1;
} else {
    x = 2;
}
```

```asm
; 局部变量向下分配：a 在 R5+0，b 在 R5-1，x 在 R5-2
        LDR     R0, R5, #0      ; a
        LDR     R1, R5, #-1     ; b
        NOT     R1, R1
        ADD     R1, R1, #1      ; R1 = -b
        ADD     R0, R0, R1      ; R0 = a - b
        BRn     THEN            ; 若 a < b（结果为负）跳到 THEN
        ; ---- else 分支 ----
        AND     R0, R0, #0
        ADD     R0, R0, #2      ; R0 = 2
        STR     R0, R5, #-2     ; x = 2
        BRnzp   ENDIF
THEN
        AND     R0, R0, #0
        ADD     R0, R0, #1      ; R0 = 1
        STR     R0, R5, #-2     ; x = 1
ENDIF
```

> **关键技巧**：LC-3 没有 `SUB` 指令，所以 `a - b` 要写成 `a + (-b)`，
> 而取负是 `NOT` 加 `ADD #1`（补码取负）。

#### `while` 循环的翻译

```c
while (i < n) {
    sum += i;
    i++;
}
```

```asm
; 局部变量：i 在 R5+0，n 在 R5-1，sum 在 R5-2
WHILE
        LDR     R0, R5, #0      ; i
        LDR     R1, R5, #-1     ; n
        NOT     R1, R1
        ADD     R1, R1, #1      ; -n
        ADD     R0, R0, R1      ; i - n
        BRzp    DONE            ; 若 i >= n，退出循环
        ; ---- 循环体 ----
        LDR     R0, R5, #0      ; i
        LDR     R1, R5, #-2     ; sum
        ADD     R1, R1, R0      ; sum + i
        STR     R1, R5, #-2     ; sum = sum + i
        LDR     R0, R5, #0
        ADD     R0, R0, #1      ; i + 1
        STR     R0, R5, #0      ; i = i + 1
        BRnzp   WHILE
DONE
```

#### `for` 循环的翻译

`for (init; cond; update) body;` 等价于 `init; while (cond) { body; update; }`。

#### 数组访问的翻译

```c
int32_t arr[5];
int32_t x = arr[2];
arr[3] = 7;
```

```asm
; 假设 arr 的基址在 R4（全局）或某个寄存器中
; arr[2] → LDR R0, R_base, #2   （字偏移，因为是字寻址）
        LDR     R0, R4, #2      ; R0 = arr[2]
        STR     R0, R5, #0      ; x = arr[2]

        ADD     R1, R1, #7      ; R1 = 7
        STR     R1, R4, #3      ; arr[3] = 7
```

```c
/* 用指针遍历数组（等价形式）*/
for (int32_t *p = arr; p < arr + 5; p++) { sum += *p; }
```

```asm
; p 在 R2 中，sum 在 R3 中，arr 末地址在 R4 中
LOOP
        NOT     R0, R4
        ADD     R0, R0, #1      ; -end
        ADD     R0, R2, R0      ; p - end
        BRzp    LOOP_END        ; p >= end 时退出
        LDR     R0, R2, #0      ; *p
        ADD     R3, R3, R0      ; sum += *p
        ADD     R2, R2, #1      ; p++（字地址 +1）
        BRnzp   LOOP
LOOP_END
```

> **重要区别**：LC-3 是**字寻址**（每个地址一个 16 位字），所以 `p++` 就是地址加 1；
> 而在 64 位机器上按字节寻址，`int*` 的 `p++` 是地址加 4。**指针算术的步长永远等于
> 所指向类型的大小除以可寻址单位**。

#### 函数调用的翻译

```c
int32_t r = add3(1, 2, 3);
```

```asm
; ---- 调用者：把参数放入 R0-R3 ----
        AND     R0, R0, #0
        ADD     R0, R0, #1      ; R0 = 1（第一个参数）
        AND     R1, R1, #0
        ADD     R1, R1, #2      ; R1 = 2
        AND     R2, R2, #0
        ADD     R2, R2, #3      ; R2 = 3
        JSR     ADD3            ; 调用；R7 = 返回地址
        ; R0 中是返回值
        STR     R0, R5, #0      ; r = 返回值
```

> 对**参数多于 4 个**的函数，第 5 个及以后的参数必须**压栈**传递（从右往左压）。

#### 结构体成员访问的翻译

```c
typedef struct { int32_t x; int32_t y; } point_t;
point_t p;
p.y = 7;
```

```asm
; p 是 2 个字的局部变量，占 R5+0 (p.x) 与 R5-1 (p.y)
; 注意：不能用 R5+1 —— 那是保存旧帧指针的位置！
        AND     R0, R0, #0
        ADD     R0, R0, #7
        STR     R0, R5, #-1     ; p.y = 7（基址 + 编译期常量偏移 1）
```

```c
point_t *pp = &p;
int32_t v = pp->y;
```

```asm
; pp 在 R2 中
        LDR     R0, R2, #1      ; R0 = pp->y（基址 = pp 的值，偏移 = 1）
```

> **这就是结构体的全部秘密**：成员访问编译成"**基址 + 编译期常量偏移**"的一次访存。
> 偏移在编译期就确定了，运行时没有任何查表或计算开销。
> `p.y` 和 `pp->y` 的唯一区别是基址来自 R5 偏移还是来自一个指针寄存器。

---

### 20. 调试速查：GDB

#### 启动

```bash
gcc -g -O0 -std=c99 -Wall -Werror -o prog prog.c    # 必须先带 -g 编译

gdb ./prog                       # 无参数
gdb --args ./prog arg1 arg2      # 带命令行参数
gdb -tui ./prog                  # 打开 TUI 界面（显示源码）
```

#### 命令表（官方速查表命令）

| 类别 | 命令 | 简写 | 作用 |
|---|---|---|---|
| **运行** | `run` | `r` | 启动程序（到第一个断点或结束） |
| | `run arg1 arg2` | | 带参数运行 |
| | `set args 4 25` | | 设置运行参数 |
| | `continue` | `c` | 继续到下一个断点 |
| | `quit` | `q` | 退出 GDB |
| **断点** | `break baz` | `b baz` | 在函数 `baz` 入口设断点 |
| | `break qux.c:24` | `b qux.c:24` | 在第 24 行设断点 |
| | `clear qux.c:24` | | 删除该行断点 |
| | `info breakpoints` | | 列出所有断点 |
| | `delete N` | `d N` | 删除 3 号断点 |
| **单步** | `next` | `n` | 执行下一行，**不进入**函数调用 |
| | `step` | `s` | 执行下一行，**进入**函数调用 |
| | `until` | `u` | 运行到当前循环的下一行（跳出循环） |
| | `finish` | `fin` | 运行到当前函数返回 |
| | `nexti` / `stepi` | `ni`/`si` | 单步**一条机器指令**（无源码时用） |
| **查看** | `print corge` | `p corge` | 打印变量的值 |
| | `print/x var` | | 以十六进制打印 |
| | `display corge` | | 每次暂停时自动打印该变量 |
| | `undisplay N` | | 取消自动显示 |
| | `list` | `l` | 显示当前行附近的源码 |
| | `layout src` | | 打开源码窗口 |
| | `backtrace` | `bt` | 显示调用栈（函数调用链） |
| | `frame 1` | `f 1` | 切换到 1 号栈帧的上下文 |
| | `info locals` | | 显示当前帧的所有局部变量 |
| | `info args` | | 显示当前帧的参数 |
| | `info frame` | | 显示当前栈帧的详细信息 |
| | `ptype var` | | 显示变量的类型定义 |
| **内存** | `x/8xb ptr` | | 从 `ptr` 起检查 8 个**字节**，十六进制 |
| | `x/4dw arr` | | 检查 4 个**字**（4 字节），十进制 |
| | `x/s str` | | 以字符串形式检查 |
| | `p *ptr` | | 解引用打印 |
| | `p arr[3]` | | 打印数组元素 |
| | `p &var` | | 打印变量地址 |
| **修改** | `set var x = 5` | | 运行时改变量值 |
| | `watch var` | | 变量一变就暂停 |
| | `call f(1, 2)` | | 手动调用函数 |

#### 读懂 `backtrace`

```
(gdb) bt
#0  0x0000000000401198 in baz () at bar.c:56
#1  0x00000000004011c4 in foo () at bar.c:47
#2  0x00000000004011e9 in main () at bar.c:12
```

- **`#0` 是最内层**（程序现在停在这里），编号越大越靠外层。
- 读法：`main` 在第 12 行调用了 `foo`，`foo` 在第 47 行调用了 `baz`，
  `baz` 停在第 56 行。
- 用 `frame 1` 切到 `foo` 的上下文，才能 `print` `foo` 的局部变量。
- **在某个栈帧里只能访问该帧的变量**——`baz` 里 `print` 不到 `main` 的局部变量。

#### 检查内存的 `x` 命令格式

```
x/[数量][格式][单位] 地址

格式：x=十六进制  d=十进制  u=无符号  o=八进制  t=二进制  c=字符  s=字符串  i=指令
单位：b=字节(1)  h=半字(2)  w=字(4)  g=巨字(8)

例：
x/8xb  ptr      → 8 个字节，十六进制
x/4dw  arr      → 4 个字（4 字节），十进制
x/2gx  &dbl     → 2 个 8 字节，十六进制
x/s    str      → 字符串
x/10i  main     → main 的前 10 条指令（反汇编）
```

#### `.gdbinit` 配置（自动加载断点）

在家目录创建 `~/.gdbinit`，内容：

```
add-auto-load-safe-path /home/user/ece220/mp08
```

在当前工作目录创建 `.gdbinit`，内容示例：

```
set print pretty            # 让结构体打印更易读
b main                      # 在 main 设断点
b list_insert               # 在 list_insert 设断点
set args input.txt          # 设置程序参数
run                         # 启动
```

之后直接 `gdb ./prog` 就会自动应用这些设置，无需每次重打。**这在 MP8 之后会非常省时间。**

#### 示例调试会话（真实的 off-by-one bug）

```c
/* buggy.c */
#include <stdio.h>
int main(void)
{
    int arr[5] = { 10, 20, 30, 40, 50 };
    int sum = 0;
    for (int i = 0; i <= 5; i++) {      /* ⚠️ 应为 i < 5 */
        sum += arr[i];
    }
    printf("sum = %d\n", sum);
    return 0;
}
```

```
$ gcc -g -O0 -std=c99 -Wall -o buggy buggy.c
$ gdb ./buggy
(gdb) b main
Breakpoint 1 at 0x1149: file buggy.c, line 4.
(gdb) r
Breakpoint 1, main () at buggy.c:4
4	    int arr[5] = { 10, 20, 30, 40, 50 };
(gdb) n
5	    int sum = 0;
(gdb) n
6	    for (int i = 0; i <= 5; i++) {
(gdb) p arr
$1 = {10, 20, 30, 40, 50}
(gdb) p sizeof(arr)/sizeof(arr[0])
$2 = 5
(gdb) watch i
Hardware watchpoint 2: i
(gdb) c
...
(gdb) p i
$3 = 5
(gdb) p arr[5]              ← 越界访问！
$4 = 0
```

**诊断**：`arr` 只有 5 个元素（下标 0..4），但循环条件 `i <= 5` 让 `i` 取到 5，
访问了 `arr[5]`——**越界**。**修复**：把 `<=` 改成 `<`。

#### 常用查看技巧

```gdb
# 查看某个指针指向的内容和它的地址
(gdb) p p
$1 = (int *) 0x7fffffffe4a0
(gdb) p *p
$2 = 42
(gdb) p &p
$3 = (int **) 0x7fffffffe498        # p 自己也有地址

# 查看数组的原始字节
(gdb) x/20xb arr
0x7fffffffe4a0: 0x0a  0x00  0x00  0x00  0x14  0x00  0x00  0x00
0x7fffffffe4a8: 0x1e  0x00  0x00  0x00  0x28  0x00  0x00  0x00
0x7fffffffe4b0: 0x32  0x00  0x00  0x00
# 注意：小端序！10 = 0x0a 存成 0a 00 00 00

# 查看结构体
(gdb) p *node
$4 = {value = 42, next = 0x602030}
(gdb) p node->next->value
$5 = 7
```

> **小端序观察**：`10` 的十六进制是 `0x0000000a`，但内存里是 `0a 00 00 00`——
> 低位字节存在低地址。这是 x86-64 的小端序 (little-endian)，与 LC-3 一致。

---

### 21. 调试速查：Valgrind 与 Sanitizer

#### Valgrind Memcheck

```bash
gcc -g -O0 -std=c99 -Wall -o prog prog.c
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./prog
```

| 选项 | 作用 |
|---|---|
| `--leak-check=full` | 详细报告每一处泄漏的完整调用栈 |
| `--show-leak-kinds=all` | 显示全部四种泄漏类别 |
| `--track-origins=yes` | 追踪未初始化值的**来源**（较慢但极有用） |
| `--error-exitcode=1` | 有错误时返回码为 1（便于脚本化测试） |
| `--gen-suppressions=all` | 生成抑制文件（用于忽略库的内部问题） |

#### 读懂泄漏分类

| 类别 | 含义 | 严重性 |
|---|---|---|
| **definitely lost** | 没有任何指针指向这块内存 | **真正的泄漏，必须修** |
| **indirectly lost** | 指向它的指针本身也丢了（通常伴随 definitely lost） | **必须修** |
| **possibly lost** | 只有指向块内部的指针（如 `p+1`），没有指向起始处 | 通常需要检查 |
| **still reachable** | 程序结束时仍有指针指向它，但没 `free` | 常常可接受（如全局缓存） |

**实测输出**（本机 `valgrind-3.19.0`，对一个 `malloc(10 * sizeof(int32_t))` 后未 `free` 的程序）：

```
==1221543==   total heap usage: 1 allocs, 0 frees, 40 bytes allocated
==1221543==
==1221543== 40 bytes in 1 blocks are definitely lost in loss record 1 of 1
==1221543==    at 0x484486F: malloc (vg_replace_malloc.c:381)
==1221543==    by 0x401137: main (leak.c:5)
==1221543==
==1221543== LEAK SUMMARY:
==1221543==    definitely lost: 40 bytes in 1 blocks
==1221543==    indirectly lost: 0 bytes in 0 blocks
==1221543==      possibly lost: 0 bytes in 0 blocks
==1221543==    still reachable: 0 bytes in 0 blocks
```

> 关键信息是 `by 0x401137: main (leak.c:5)`——它**直接指出泄漏发生在 `leak.c` 第 5 行**，
> 即那次 `malloc`。

#### Valgrind 能抓到的错误类型

| 报告 | 含义 |
|---|---|
| `Invalid read of size 4` | 读取了越界或已释放的内存 |
| `Invalid write of size 1` | 写入了越界或已释放的内存 |
| `Conditional jump depends on uninitialised value` | 用未初始化的变量做判断 |
| `Invalid free() / delete / delete[]` | `free` 了非 `malloc` 指针，或重复 `free` |
| `Mismatched free() / delete / delete[]` | C++ 中 `new` 配了 `free` |
| `Source and destination overlap in memcpy` | `memcpy` 区域重叠（应用 `memmove`） |

#### AddressSanitizer（更快，编译期插桩）

```bash
gcc -g -O0 -std=c99 -Wall -fsanitize=address,undefined -o prog prog.c
./prog
```

**优势**：比 Valgrind 快得多（约 2 倍减速 vs 20 倍），且能找到**栈上的**越界访问
（Valgrind 主要覆盖堆）。

> ⚠️ **本笔记编写环境的实测说明**：ASan 需要预留约 15 TB 的**虚拟**地址空间作为影子内存
> （shadow memory），而本机的 `ulimit -v` 为 32 GB，因此 ASan 无法启动，报错为：
>
> ```
> ==ERROR: AddressSanitizer failed to allocate 0xdfff0001000 ...
> ReserveShadowMemoryRange failed while trying to map 0xdfff0001000 bytes. Perhaps you're using ulimit -v
> ```
>
> 若你在 EWS 实验室机器或自己的 Linux 上运行（无此虚拟内存限制），ASan 可以正常工作。
> **本笔记中的实测输出均来自 Valgrind**（本机 `valgrind-3.19.0` 可用）。
> 下面展示的是 ASan 报告的**标准形式**，供你识别其输出结构，而非本机粘贴的输出。

```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000018
WRITE of size 4 at 0x602000000018 thread T0
    #0 0x400b4d in main /tmp/overflow.c:7
    #1 0x7f... in __libc_start_main

0x602000000018 is located 0 bytes to the right of 16-byte region
[0x602000000008,0x602000000018)
allocated by thread T0 here:
    #0 0x7f... in malloc
    #1 0x400a9e in main /tmp/overflow.c:5
```

> **读法**：`heap-buffer-overflow` 说明越界；`0 bytes to the right of 16-byte region`
> 说明你正好写在 16 字节区域的右边界外——典型的 `i <= n` 写成 `i < n` 的错误。
> 两处调用栈分别告诉你**出错位置**和**分配位置**。

#### 其他 Sanitizer

```bash
-fsanitize=undefined        # 未定义行为（有符号溢出、空指针解引用、越界移位、错位对齐）
-fsanitize=leak             # 仅泄漏检测（比 ASan 轻量）
-fsanitize=thread           # 数据竞争（多线程）
-fno-omit-frame-pointer     # 与 -fsanitize=address 配合，让栈更完整
```

#### 三种工具的选择

| 工具 | 速度 | 覆盖范围 | 何时用 |
|---|---|---|---|
| **AddressSanitizer** | 快（~2×） | 堆 + 栈 + 全局变量的越界、UAF、泄漏 | 日常开发首选 |
| **Valgrind** | 慢（~20×） | 堆为主；能追踪未初始化值的来源 | 需要精确的未初始化值溯源时 |
| **手工 `assert` + 打印** | 最快 | 只覆盖你想到的东西 | 快速定位逻辑错误 |

---

### 22. 错误分类与排查流程图

#### 四类错误（课程的错误分类学）

| 类别 | 何时暴露 | 典型表现 | 谁来抓 |
|---|---|---|---|
| **编译期错误** | 编译时 | 语法错误、类型错误、未声明标识符 | 编译器（`-Wall -Werror`） |
| **链接期错误** | 链接时 | `undefined reference to 'foo'` | 链接器 |
| **运行期错误** | 运行时 | 段错误、`abort`、错误结果、随机行为 | GDB、Valgrind、Sanitizer |
| **逻辑错误** | 运行时 | 能跑完，结果不对 | **只有测试能抓** |

> **核心认识**：编译器只能抓第一类。第二类靠链接器。**第三、四类只能靠测试和调试工具。**
> 这就是为什么 "它编译通过了" 完全不能说明程序是对的。

#### 排查流程

```
程序有问题
    │
    ├─ 编译不过？ ──────────────────────────────► 读第一条错误信息（不是最后一条！）
    │                                             加 -Wall -Werror，逐个修
    │
    ├─ 链接不过？ ──────────────────────────────► undefined reference:
    │                                             忘了加 .c 文件 / 忘了 -l库 / 拼错函数名
    │
    ├─ 崩溃（段错误）？ ────────────────────────► 用 ASan 重编译运行
    │   │                                         gcc -g -fsanitize=address
    │   │                                         或 gdb → run → bt → 看 #0 帧
    │   │
    │   └─ 常见原因：野指针、解引用 NULL、数组越界、栈溢出（递归太深）
    │
    ├─ 结果不对？ ──────────────────────────────► 先用 printf 缩小范围
    │   │                                         二分法：在中间打一个 printf，判断错误在前半还是后半
    │   │                                         GDB: b 函数名 → n 单步 → p 变量 → watch 关键变量
    │   │
    │   └─ 边界条件检查：空输入 / 单元素 / 全相同 / 最大值 / 负数
    │
    └─ 行为随机、时对时错？ ────────────────────► 未初始化变量 或 越界写坏了别人的数据
                                                  用 Valgrind --track-origins=yes
                                                  或 ASan + MSan
```

#### 段错误的五个常见原因

```
Segmentation fault（段错误）—— 访问了不属于你的虚拟地址
    │
    ├─ 1. 解引用未初始化的指针       int *p; *p = 5;
    ├─ 2. 解引用 NULL                int *p = NULL; *p = 5;
    ├─ 3. 数组越界写                 arr[100] = 5;   （arr 只有 10 个元素）
    ├─ 4. 使用已释放的内存           free(p); *p = 5;
    ├─ 5. 递归太深导致栈溢出         void f(void){ f(); }
    └─ 6. scanf 忘了 &               scanf("%d", v);   （v 被当作地址）
```

---

### 23. 常见内存错误图鉴

#### ① 内存泄漏 (Memory Leak)

```c
/* ❌ 每次循环泄漏一块内存 */
for (int i = 0; i < 1000; i++) {
    int32_t *p = malloc(sizeof(int32_t));
    *p = i;
    /* 忘了 free(p)，而且 p 出了作用域就丢了 —— 这块内存再也找不回来了 */
}

/* ✅ 修复 */
for (int i = 0; i < 1000; i++) {
    int32_t *p = malloc(sizeof(int32_t));
    if (p == NULL) { return -1; }
    *p = i;
    /* ... 使用 p ... */
    free(p);
    p = NULL;
}
```

```
堆状态随时间变化：
开始:  [ 空闲 ................ ]
第 1 次: [已用][ 空闲 .......... ]
第 2 次: [已用][已用][ 空闲 .... ]
...
第 n 次: [已用][已用][已用]...[已用]  ← 堆耗尽，malloc 返回 NULL
                                  ↑ 累积的"已用"块全部无法访问，永不回收
```

**检测**：`valgrind --leak-check=full ./prog` → 看 "definitely lost"。

#### ② 悬垂指针 / 释放后使用 (Dangling Pointer / Use-After-Free)

```c
/* ❌ 释放后继续使用 */
int32_t *p = malloc(sizeof(int32_t));
*p = 42;
free(p);

printf("%d\n", *p);       /* ❌ use-after-free：内存可能已被重新分配或归还给 OS */
*p = 7;                   /* ❌ 更糟：写坏别人的数据 */

/* ✅ 修复：释放后立刻置 NULL */
free(p);
p = NULL;
/* 之后任何对 p 的解引用都会立即段错误（好过默默破坏数据） */
```

```
时间线：
  p ──► [块 A：42]
  free(p) 后：
  p ──► ???（指针值没变，但块 A 已归还给分配器）
         ↓
  p 仍指向原地址，但该内存可能已被分配给别的变量
         ↓
  解引用 p 会读到垃圾 / 写坏别人的数据 —— 行为不确定
```

**检测**：ASan 会在 UAF 时立即报 `heap-use-after-free` 并给出**释放点**和**使用点**两个栈。

#### ③ 重复释放 (Double Free)

```c
/* ❌ */
int32_t *p = malloc(sizeof(int32_t));
free(p);
free(p);                  /* ❌ 破坏分配器的空闲链表 */

/* ✅ */
free(p);
p = NULL;
free(p);                  /* ✅ free(NULL) 是合法的空操作 */
```

#### ④ 越界访问 (Out-of-Bounds)

```c
int32_t arr[10];

/* ❌ 循环写多了 */
for (int i = 0; i <= 10; i++) { arr[i] = 0; }   /* arr[10] 越界 */

/* ❌ 差一错误 */
int32_t *buf = malloc(10 * sizeof(int32_t));
buf[10] = 1;                                    /* 正好写在块尾之后 */
```

```
堆布局（越界写的破坏）：
  ┌───────────────┬───────────────┬───────────────┐
  │  元数据        │   buf (40字节) │   下一个块     │
  └───────────────┴───────────────┴───────────────┘
                                    ↑ buf[10] 写在这里 —— 踩坏了下一个块的元数据！
                                      后果：下次 free 时段错误或堆损坏
```

**检测**：ASan（立即报错）或 Valgrind（`Invalid write of size 4`）。

#### ⑤ 返回局部变量的地址

```c
/* ❌ 返回指向已销毁栈帧的指针 */
int32_t *bad(void)
{
    int32_t local = 42;
    return &local;         /* local 所在栈帧已失效 */
}

/* ✅ 方案 A：调用者提供缓冲区 */
void good_a(int32_t *out) { *out = 42; }

/* ✅ 方案 B：堆分配，调用者负责 free */
int32_t *good_b(void) { int32_t *p = malloc(sizeof *p); if (p) { *p = 42; } return p; }

/* ✅ 方案 C：static 变量（注意：不可重入！） */
int32_t *good_c(void) { static int32_t v; v = 42; return &v; }
```

#### ⑥ 未初始化的内存

```c
int32_t *p = malloc(10 * sizeof(int32_t));
/* malloc 不保证清零！内容可能是任意值 */
printf("%d\n", p[0]);      /* ❌ 读到垃圾值 */

/* ✅ 用 calloc 清零 */
int32_t *q = calloc(10, sizeof(int32_t));
/* q[0..9] 全是 0 */
```

#### ⑦ 结构体填充未初始化

```c
typedef struct { char a; int b; char c; } s_t;
s_t s;
s.a = 'x'; s.c = 'y';      /* ❌ 忘了 s.b */
fwrite(&s, sizeof(s), 1, fp);   /* 把 b 的垃圾值和填充字节一起写进文件 */
```

#### ⑧ `free` 非堆指针

```c
int32_t arr[10];
free(arr);                 /* ❌ arr 在栈上，不是 malloc 来的 */
free(&arr[2]);             /* ❌ 偏移过的指针，元数据位置错误 */
```

#### 内存错误速查表

| 错误 | 症状 | 检测工具 |
|---|---|---|
| 内存泄漏 | 长时间运行内存持续增长 | Valgrind `--leak-check=full`、ASan |
| 悬垂指针 / UAF | 随机崩溃或错误结果 | ASan、Valgrind |
| 重复释放 | `free(): double free detected` | ASan、Valgrind |
| 越界读写 | 段错误或"幽灵"数据损坏 | ASan、Valgrind |
| 未初始化读取 | 结果不确定、时对时错 | Valgrind `--track-origins=yes`、MSan |
| 返回局部变量地址 | 函数返回后数据变成垃圾 | 编译器 `-Wreturn-local-addr`（在 `-Wall` 里） |
| `realloc` 失败覆盖指针 | 内存泄漏 | 代码审查 + Valgrind |
| 结构体填充未清零 | 二进制输出不确定 | `memset(&s, 0, sizeof(s))` |

---

### 24. C 编程准则清单

#### 内存与指针

1. **每个 `malloc` 都要检查 `NULL`。**
2. **每个 `malloc` 都要有对应的 `free`。** 画一张"所有权图"：谁分配、谁释放。
3. **`free` 之后立刻把指针置 `NULL`。**
4. **`realloc` 必须用临时指针接返回值。**
5. **不要返回指向局部变量的指针。**
6. **数组下标前先想清楚边界**：有效范围是 `0 .. n-1`，不是 `0 .. n`。
7. **`sizeof(arr)` 在函数参数上是指针大小**——必须显式传长度。

#### 字符串

8. **永远不用 `gets`。** 用 `fgets`。
9. **`strcpy`/`strcat` 不检查长度**——用 `snprintf` 或自己检查。
10. **`strcmp` 返回 0 表示相等**。
11. **字符串字面量是只读的**，不要写 `char *s = "abc"; s[0] = 'x';`。

#### 作用域与类型

12. **避免全局变量**，用 `static` 限制在文件内。
13. **用 `int32_t` 等定宽类型**而不是裸 `int`，特别是在需要确定宽度时。
14. **`int *A, B;` 里 `B` 不是指针**——每个指针都写 `*`。
15. **`typedef struct { ... } name_t;`** 遵循课程编码规范。

#### 函数与接口

16. **每个函数只做一件事**，名字说明它做什么。
17. **函数超过一屏就考虑拆分。**
18. **头文件只暴露接口**，把表示藏在 `.c` 里（信息隐藏）。
19. **头文件必须加 include guard。**
20. **检查所有可能失败的函数的返回值**（`malloc`、`fopen`、`scanf`、`fclose`）。

#### 流程与风格

21. **始终使用花括号**，即使只有一条语句。
22. **用 `=` 还是 `==` 想清楚**；把常量写在左边（`if (5 == x)`）可以借助编译器抓错。
23. **不要用 Tab**，用 4 个空格；行宽 ≤ 120 字符。
24. **`switch` 每个 `case` 都要有 `break`**（除非你**故意**要 fall-through 并写了注释）。

#### 测试与调试

25. **编译时永远带 `-g -std=c99 -Wall -Werror`。**
26. **调试时永远用 `-O0`**，`-O2` 会让 GDB 显示的变量和源码对不上。
27. **测试的边界**：空、单元素、全相同、最大值、最小值、NULL。
28. **"它跑通了一次"不是证据。** 写能重复运行的测试。
29. **遇到崩溃先上 ASan**，遇到泄漏上 Valgrind，遇到逻辑错误用 GDB 单步加 `watch`。
30. **`assert` 表达"这里必须为真"的不变量**，不要用它处理用户输入错误。

---

### 附：LC-3 与 x86-64 概念对照表

| 概念 | LC-3 | x86-64（EWS Linux） |
|---|---|---|
| 字长 | 16 位 | 64 位 |
| 寻址单位 | 字（16 位） | 字节（8 位） |
| 通用寄存器数 | 8 个（R0–R7） | 16 个（RAX, RBX, …, R15） |
| 栈指针 | R6 | RSP |
| 帧指针 | R5 | RBP |
| 返回地址 | R7 | 栈上（`call` 自动压栈） |
| 参数传递 | R0–R3，多余的压栈 | RDI, RSI, RDX, RCX, R8, R9（System V），多余的压栈 |
| 返回值 | R0 | RAX |
| 调用指令 | `JSR` / `JSRR` | `call` |
| 返回指令 | `RET`（= `JMP R7`） | `ret` |
| 减法 | 无（用 `NOT` + `ADD #1` 取负） | `sub` |
| 乘法/除法 | 无 | `imul` / `idiv` |
| 指针大小 | 16 位（1 个字） | 64 位（8 字节） |
| 加载 | `LD` / `LDI` / `LDR` / `LEA` | `mov` / `lea`（`lea` 用于算地址） |
| 存储 | `ST` / `STI` / `STR` | `mov` |
| 条件跳转 | `BRnzp`（基于 NZP 三个标志） | `jmp` / `je` / `jne` / `jl` / `jg`（基于 flags 寄存器） |
| 按类型缩放指针算术 | 编译器自动（步长 = sizeof(T) / 2 字） | 编译器自动（步长 = sizeof(T) 字节） |
| 调用约定 | 课程定义的 LC-3 CIS | System V AMD64 ABI |
| 系统调用 | `TRAP` 向量（x20–x25） | `syscall` 指令 + 系统调用号 |
| 动态内存扩展 | 无（LC-3 无堆） | `sbrk` / `mmap` |

> **最该记住的一条**：LC-3 的字寻址与 x86-64 的字节寻址**不影响指针算术的语义**——
> 在两种机器上，`p + 1`（`p` 是 `int*`）都前进"一个 `int` 的距离"。
> 差别只在于这个距离在 LC-3 上是 1 个地址单位（1 个 16 位字），
> 在 x86-64 上是 4 个地址单位（4 个字节）。
> **指针算术的步长永远等于"所指向类型的大小"**，这是跨平台的不变真理。

{% endraw %}
