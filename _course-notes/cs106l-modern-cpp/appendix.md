---
title: "现代 C++ 核心特性速查表（Modern C++ Cheat Sheet）"
collection: course-notes
chapter: true
permalink: /course-notes/cs106l-modern-cpp/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/cs106l-modern-cpp/) · [← l17](/course-notes/cs106l-modern-cpp/l17)

{% raw %}
# 现代 C++ 核心特性速查表（Modern C++ Cheat Sheet）

> 按类别汇总 CS106L 全课程涉及的关键特性与语法。标准标注：C++11 / C++14 / C++17 / C++20 / C++23 / C++26。

## 1. 类型推导（Type Deduction）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| `auto` 变量 | `auto x = expr;` | C++11 | 编译器推导类型；仍是**静态类型**，`auto i = 1; i = "hi";` 编译错误 |
| `auto` 返回类型 | `auto f() { return 42; }` | C++14 | 返回类型由 `return` 推导 |
| 尾置返回类型 | `auto f() -> int;` | C++11 | 显式指定返回类型 |
| 泛型 lambda / 模板参数 | `auto` 形参：`[](auto x){...}` | C++14 | `auto` 形参等价于模板参数 |
| `decltype` | `decltype(expr) y = x;` | C++11 | 不求值地取表达式的声明类型 |
| `decltype(auto)` | `decltype(auto) y = expr;` | C++14 | 保留引用/值语义地推导 |
| 结构化绑定 | `auto [k, v] = map_pair;` | C++17 | 解构 pair/tuple/struct |
| `using` 类型别名 | `using Zeros = std::pair<double,double>;` | C++11 | 替代 C++98 的 `typedef`，可带模板参数 |

## 2. 初始化（Initialization）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| 统一/列表初始化 | `T obj {a, b, c};` | C++11 | 最通用、最安全的初始化方式；禁止窄化转换（narrowing） |
| 成员初始化列表 | `Foo(int x) : x_{x}, y_{} {}` | C++98（`{}` 版 C++11） | 唯一能初始化 `const`/引用成员的方式；避免"先默认构造再赋值"的双重开销 |
| 默认成员初始化器 | `int x_ = 0;` | C++11 | 类内给成员默认值 |
| 默认构造 `= default` | `Foo() = default;` | C++11 | 显式保留编译器生成的版本 |
| 删除函数 | `Foo(const Foo&) = delete;` | C++11 | 禁止拷贝等操作（如 `unique_ptr`） |

## 3. 引用与移动语义（References & Move Semantics）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| 左值引用 | `int& r = x;` | C++98 | 别名；`x = y;` 两侧都可出现 |
| `const` 左值引用 | `const T& r = x;` | C++98 | 只读别名，可绑定临时值 |
| 右值引用 | `T&& r = temporary;` | C++11 | 只能绑定临时对象（右值），用于"窃取"资源 |
| 移动构造/赋值 | `T(T&& other) noexcept;` / `T& operator=(T&& other) noexcept;` | C++11 | 转移资源所有权，通常 O(1) |
| `std::move` | `std::move(x)` | C++11 | 仅把左值**转换**为右值引用（`static_cast<T&&>`），本身不移动任何东西 |
| `std::forward` | `std::forward<T>(x)` | C++11 | 完美转发，保留实参的左右值类别 |
| 移动后状态 | 源对象"有效但未指定" | C++11 | 通常应把源指针置 `nullptr`；只允许对源对象销毁或重新赋值 |

## 4. 特殊成员函数（Special Member Functions）与规则

| 规则 | 内容 | 说明 |
|---|---|---|
| 六大 SMF | 默认构造 `T()`、析构 `~T()`、拷贝构造 `T(const T&)`、拷贝赋值 `T& operator=(const T&)`、移动构造 `T(T&&)`、移动赋值 `T& operator=(T&&)` | 需要时才由编译器隐式生成 |
| **Rule of Zero** | 成员都是自管理类型（`string`、`vector`、智能指针）时，什么都不用写 | 最推荐 |
| **Rule of Three** | 需要自定义析构 ⇒ 通常也要自定义拷贝构造 + 拷贝赋值 | 手工管理资源（如 `new`/`delete`）时 |
| **Rule of Five** | Rule of Three 成立时，通常还应定义移动构造 + 移动赋值 | 否则会退化为拷贝，性能受损 |

## 5. 智能指针与 RAII（Smart Pointers & RAII）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| `std::unique_ptr` | `auto p = std::make_unique<T>(args);` | C++11 | 独占所有权；不可拷贝、可移动；离开作用域自动 `delete` |
| `std::shared_ptr` | `auto p = std::make_shared<T>(args);` | C++11 | 共享所有权，引用计数归零时释放；注意循环引用 |
| `std::weak_ptr` | `std::weak_ptr<T> w = sp;` `w.lock()` | C++11 | 观察者，不增加引用计数，打破循环依赖 |
| RAII 思想 | 资源在**构造**时获取、**析构**时释放 | — | 保证异常安全：析构函数必然被调用 |
| 其他 RAII 例子 | `std::lock_guard`、`std::ifstream/ofstream` | — | 锁、文件等在析构时自动释放 |

## 6. 模板（Templates）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| 模板类 | `template <typename T> class Vector { ... };` | C++98 | 按需实例化，为每种 `T` 生成代码 |
| 模板函数 | `template <typename T> T min(T a, T b);` | C++98 | 显式 `min<int>(...)` 或隐式推导 |
| 非类型模板参数 | `template <size_t N> struct Array;` `std::array<T, N>` | C++11 | 编译期常量作参数；`array` 栈上分配 |
| 变参模板 | `template <typename T, typename... Args>` + 包展开 `args...` | C++11 | 接受任意数量/类型参数，递归实例化 |
| 模板特化 | `template <> struct Foo<int> {...};` | C++98 | 为特定类型提供专门实现 |
| 模板实现位置 | `.h` 底部 `#include "Foo.cpp"`，定义用 `Foo<T>::` | — | 实例化需要完整定义可见 |

## 7. Concepts 与编译期计算（C++20）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| 自定义 concept | `template <typename T> concept Comparable = requires(T a, T b) { {a < b} -> std::convertible_to<bool>; };` | C++20 | 在实例化前约束模板参数，显著改善错误信息 |
| 约束模板 | `template <Comparable T>` 或 `template <typename T> requires Comparable<T>` | C++20 | 简写与完整两种写法等价 |
| 内置 concepts | `std::input_iterator`、`std::range`、`std::convertible_to`、`std::same_as` 等 | C++20 | 标准库自带 |
| `constexpr` | `constexpr size_t fact(size_t n);` | C++11（放宽于 C++14） | "请尽量在编译期求值" |
| `consteval` | `consteval size_t f(size_t n);` | C++20 | "必须在编译期求值" |
| 模板元编程（TMP） | `Factorial<N-1>::value` 递归结构 | C++98 | 传统 TMP；现代用 `constexpr` 更可读 |

## 8. Lambda 与函数对象（Lambdas & Functors）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| Lambda | `auto f = [captures](params) { body };` | C++11 | 编译器展开为匿名 functor 类 |
| 捕获列表 | `[x]`（值）、`[&x]`（引用）、`[=]`、`[&]`、`[&, x]`、`[this]`、`[x = expr]`（初始化捕获 C++14） | C++11/14 | 值捕获复制；引用捕获注意生命周期 |
| 泛型 lambda | `[](auto x) { return x * 2; }` | C++14 | 等价于模板 |
| 函数指针 | `bool(*pred)(char) = isVowel;` | C++98 | 只能指向无捕获的普通函数 |
| 仿函数（functor） | `struct G { bool operator()(int a, int b) const {...} };` | C++98 | 重载 `operator()` 的对象，可有状态 |
| `std::function` | `std::function<bool(int)> f = lambda;` | C++11 | 统一容器类型（可存任何可调用对象），略慢 |
| `std::bind` / `std::ref` | — | C++11 | 部分应用与引用包装（较少用了，lambda 更清晰） |

## 9. STL 容器（Containers）

| 容器 | 头文件 | 特点 | 迭代器类别 |
|---|---|---|---|
| `std::vector<T>` | `<vector>` | 动态数组，随机访问 O(1)，尾部增删摊还 O(1) | 随机访问 |
| `std::deque<T>` | `<deque>` | 双端队列，头尾增删均 O(1) | 随机访问 |
| `std::array<T, N>` | `<array>` | 定长数组，栈上分配，大小编译期已知（C++11） | 随机访问 |
| `std::list<T>` | `<list>` | 双向链表，任意位置插入 O(1)（需迭代器） | 双向 |
| `std::map<K,V>` / `std::set<K>` | `<map>` / `<set>` | 有序（红黑树），需 `K` 有 `operator<`，查找 O(log n) | 双向 |
| `std::unordered_map<K,V>` / `std::unordered_set<K>` | `<unordered_map>` / `<unordered_set>` | 哈希表，需 `std::hash<K>`，平均查找 O(1) | 前向 |
| `std::pair<T1,T2>` | `<utility>` | 两个字段的泛型 struct | — |
| `std::tuple` | `<tuple>` | 任意多个字段（C++11） | — |

## 10. 算法与 Ranges（Algorithms & Ranges）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| 迭代器对算法 | `std::sort(b, e)`、`std::find(b, e, v)`、`std::count_if(b, e, p)`、`std::copy_if(b, e, o, p)`、`std::transform(b, e, o, op)`、`std::unique_copy(b, e, o, p)` | C++98 | `<algorithm>` 通用算法，作用于迭代器区间 `[first, last)` |
| 范围（range）算法 | `std::ranges::find(v, c)`、`std::ranges::sort(v)` | C++20 | 直接传容器；受 concepts 约束，错误信息更好 |
| 视图（view） | `auto v = c \| std::views::filter(p) \| std::views::transform(f);` | C++20 | **惰性**组合，逐元素按需计算；类似 Python 生成器 |
| 物化视图 | `std::ranges::to<std::vector<T>>(view)` | C++23 | 把惰性视图收集成容器 |
| 插入迭代器 | `std::back_inserter(v)` | C++98 | 让算法向容器"推入"输出 |

## 11. 运算符重载（Operator Overloading）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| 成员重载 | `bool operator<(const T& other) const;` | C++98 | 左操作数是 `this` |
| 非成员重载 | `bool operator<(const T& a, const T& b);` | C++98 | 更对称、更惯用；左操作数可为非类类型 |
| `friend` | `friend bool operator<(const T&, const T&);` | C++98 | 非成员函数访问私有成员 |
| 流插入/提取 | `std::ostream& operator<<(std::ostream&, const T&);` | C++98 | 让 `std::cout << obj` 可用；返回流以支持链式 |
| 规则 | 语义应显然（Principle of Least Astonishment）；`==` 与 `!=` 成对（rule of contrariety：`!=` 用 `!(a==b)` 实现）；不可重载 `::` `?:` `.` `.*` `sizeof` `typeid` | — | 运算符的意义必须符合直觉，否则用命名函数 |

## 12. 类型安全与 `std::optional`（C++17）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| `std::optional<T>` | `std::optional<int> o;` / `o = 42;` / `o = std::nullopt;` | C++17 | 可能包含值，也可能为空 |
| 判空与取值 | `o.has_value()`、`o.value()`（空则抛 `std::bad_optional_access`）、`o.value_or(0)`、`bool(o)` / `if (o)` | C++17 | 空 optional 隐式转换为 `false` |
| Monadic 操作 | `o.and_then(f)`（f 返回 optional）、`o.transform(f)`（f 返回值）、`o.or_else(f)` | C++23 | 链式处理"可能失败"的计算 |
| `std::nullopt` vs `nullptr` | `nullopt` 用于 optional；`nullptr` 用于指针 | C++17 | 不要混淆 |
| 类型安全理念 | "Well typed programs cannot go wrong." | — | 用类型系统把"可能失败"写进签名 |

## 13. 并发与杂项（C++11 起）

| 特性 | 语法 | 标准 | 说明 |
|---|---|---|---|
| 线程 | `std::thread t(f, args); t.join();` | C++11 | `<thread>` |
| 互斥锁 | `std::mutex m; std::lock_guard<std::mutex> lk(m);` | C++11 | RAII 管理锁 |
| 原子操作 | `std::atomic<int> counter;` | C++11 | `<atomic>`，无锁编程基础 |
| `nullptr` | `T* p = nullptr;` | C++11 | 类型安全的空指针（替代 `NULL`/`0`） |
| `enum class` | `enum class Color { Red, Green };` | C++11 | 强类型、有作用域的枚举 |
| `override` / `final` | `void update() override;` / `class D final {};` | C++11 | 显式覆写/终结虚函数，防拼写错误 |
| `noexcept` | `void f() noexcept;` | C++11 | 声明不抛异常；移动构造标 `noexcept` 让容器优先移动 |
| range-for | `for (const auto& x : c) {...}` | C++11 | 基于迭代器的语法糖 |
| 字符串字面量 | `u8"..."`、`R"(raw)"`、`"..."s`（`std::string` 字面量） | C++11/14 | 编码与原始字符串 |

---

*笔记完。祝学习愉快！* 🌽

{% endraw %}
