---
title: "附录 A：课程网站调研记录（阶段一成果摘要）"
collection: course-notes
chapter: true
permalink: /course-notes/cs106b-programming-abstractions/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/cs106b-programming-abstractions/) · [← l17](/course-notes/cs106b-programming-abstractions/l17)

{% raw %}
## 附录 A：课程网站调研记录（阶段一成果摘要）

**调研对象**：https://web.stanford.edu/class/cs106b/ （自动指向归档 `https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1268/`，2026 夏季学期）。完整原始记录见同目录 `00_research_inventory.md`；原始 HTML/清洗文本见 `cs106b_pages/`。

### A.1 页面可访问性速览

| 资源 | 状态 | 说明 |
|---|---|---|
| 课程主页 / Syllabus / Honor Code / 教职员页 / 资源大页 | ✅ 公开 | 课程定位、学习目标、先修、评分、教科书、工具 |
| About Lectures / Flat Lecture Index | ✅ 公开 | 讲座组织与小测规则；28 讲索引 |
| **28 个讲座页**（`/lectures/xx-slug/`） | ✅ 公开 | 每页含**当日完整文字讲义/纪要**（Contents 提纲 + 讲解）+ 部分附件（PDF/zip） |
| About Sections + 7 个 Section 页 | ✅ 公开 | 小班习题（答案周五公布） |
| About Assignments + 8 个作业页 | ✅ 公开 | 制度说明；题面公开 |
| Exams 说明页 + 备考建议 + 期末参考表 refsheet.pdf | ✅ 公开 | 含往年真题（期末页） |
| 讲座小测（Canvas） | ❌ 需斯坦福账号 | 每周发布 |
| Ed 讨论区 / Gradescope / Paperless / LaIR | ❌ 需课程注册 | — |
| 讲座录像 | ❌ 仅注册学生 | 教室录制，供 CGOE 远程生观看 |
| 教科书 | ❌ 需购买/图书馆 | Roberts《Programming Abstractions in C++》 |

### A.2 真实 28 讲一览（官方标题与日期）

| 讲 | 日期 | 官方标题 | 讲 | 日期 | 官方标题 |
|---|---|---|---|---|---|
| L01 | 6/22 | Welcome! | L15 | 7/20 | Object-Oriented Programming |
| L02 | 6/23 | C++ Fundamentals | L16 | 7/21 | Pointers and Arrays |
| L03 | 6/24 | C++ Strings | L17 | 7/22 | Dynamic Memory Management |
| L04 | 6/25 | Testing, Vectors, and Grids | L18 | 7/23 | Priority Queues and Binary Heaps |
| L05 | 6/29 | Stacks and Queues | L19 | 7/27 | Introduction to Linked Lists |
| L06 | 6/30 | Sets and Maps | L20 | 7/28 | More Linked Lists |
| L07 | 7/1 | Big-O and Algorithmic Analysis | L21 | 7/29 | Binary Trees, BSTs, and Tree Traversals |
| L08 | 7/6 | Introduction to Recursion | L22 | 7/30 | More on Binary Trees |
| L09 | 7/7 | More Recursion | L23 | 8/3 | Huffman Coding |
| L10 | 7/8 | Recursive Problem Solving | L24 | 8/4 | Hashing |
| L11 | 7/9 | Recursive Backtracking and Enumeration | L25 | 8/5 | Graphs |
| L12 | 7/13 | More Recursive Backtracking | L26 | 8/6 | Dijkstra and A* Shortest Path Algorithms |
| L13 | 7/14 | Sorting Algorithms | L27 | 8/10 | Graph Coding |
| L14 | 7/15 | Problem Solving Day（复习答疑） | L28 | 8/11 | Wrap（期末回顾） |

### A.3 各章内容如何“落地”到公开资料

- **每章概述与概念清单**以官方讲座页的公开文字讲义（Contents 提纲 + 讲解）为事实依据，正文为原创中文讲解。
- **公开附件样例**（讲座页直接链接）：L05 StackViz.zip/QueueViz.zip；L13 sorting-stuff.zip；L15 oop-geocities-quokkas.zip；L16 pointers-worksheet.pdf（+解答）；L18 minheaps-written-notes.pdf；L21 tree-notes.pdf；L22 bst-code.zip、traversal-puzzle.pdf；L26 dijkstra-slides.pdf；L27 graph-algorithms.zip；期末 refsheet.pdf。Stanford C++ 库文档（Vector/Grid/Stack/Queue/Set/Map/HashMap/HashSet/strlib 等）在 web.stanford.edu/dept/cs_edu/resources/cslib_docs/ 公开。
- **受限项**：Canvas 讲座小测与录像、Ed/Gradescope/Paperless/LaIR、A5 等作业的 starter 工程文件需课程身份；正文已标注哪些内容“官方未公开/受限”。

## 附录 B：各章数据记录（lecture_data_records.json 摘要）

下表为阶段一为每“讲（章节）”生成的结构化数据记录概要；完整 JSON（含 key_concepts_raw 全量与 available_public_info/not_public 字段）见同目录 `lecture_data_records.json`。

| 章 | 主题（lecture_topic） | 对应官方讲座 | 公开信息可用性（一句话） |
|---|---|---|---|
| 1 | C++ 基础回顾与 STL 容器入门（C++ Fundamentals & ADT Containers Intro） | L01, L02, L03, L04 | 官方公开 4 个讲座页：每页含当日完整文字讲义（要点提纲+讲解），公开附件 BlankProject.zip、strlib.h/Vector/Grid 文档链接；对应作业 0/1 题面页公开 |
| 2 | 栈与队列（Stacks and Queues: LIFO/FIFO ADTs） | L05 | 官方讲座页完整公开：Stack/Queue 操作表、应用（含后缀表达式）、StackViz.zip/QueueViz.zip 可视化程序公开可下载 |
| 3 | 集合与映射（Sets and Maps: ordered, tree-based containers） | L06 | 官方讲座页完整公开：Set/Map 关键操作与运算符、有序性讨论、去重/词频应用（德古拉全文词频）、Set/Map vs HashSet/HashMap 概念对比 |
| 4 | 算法分析：大 O 记号与运行时间估算（Big-O & Algorithmic Analysis） | L07 | 官方讲座页完整公开：大 O 术语、常见函数族、求和恒等式推导、add/insert 幕后分析、线性/二次/指数/对数增长与运行时间估算示例 |
| 5 | 递归：原理、策略与递归式思维（Recursion: Principles & Strategies） | L08, L09, L10 | 官方 3 个讲座页完整公开：递归入门（阶乘/回文/打印/包装函数/常见陷阱）、二分查找与枚举生成（硬币序列、排列）、递归解题（分形、骰子序列）；对应作业 3（Recursion Etudes）题面页公开 |
| 6 | 递归回溯与枚举（Recursive Backtracking & Enumeration） | L11, L12 | 官方 2 个讲座页完整公开：回溯范式与函数骨架、子集生成/计数、划分问题 isPartitionable、0-1 背包三种递归写法及最好/最坏复杂度讨论；对应作业 4（Recursive Backtracking）题面页 |
| 7 | 排序算法（Sorting: Selection, Insertion, Merge, Quicksort） | L13 | 官方讲座页完整公开：选择/插入/归并排序要点与运行时间对比数据、slides 与代码（sorting-stuff.zip 公开） |
| 8 | 面向对象编程：类、对象与封装（OOP: Classes, Objects & Encapsulation） | L15 | 官方讲座页完整公开：OOP 范式转变、.h/.cpp 分离、Quokka 课堂示例全代码（oop-geocities-quokkas.zip 公开）；对应作业 5（Tone Matrix）题面页公开 |
| 9 | 指针、数组与动态内存管理（Pointers, Arrays & Dynamic Memory） | L16, L17 | 官方 2 个讲座页完整公开：地址/指针语法、&与*的双重语义、数组与指针关系、动态内存与内存图、new/delete 法则、ArrayBasedStack 讲解与练习；L16 附 pointers-worksheet.p |
| 10 | 优先队列与二叉堆（Priority Queues & Binary Heaps） | L18 | 官方讲座页完整公开：树术语、最小堆性质与操作、percolation、最好情形删除、堆排序、数组表示、补充 heapify/maxheap 讨论；手写讲义 minheaps-written-notes.pdf 公开 |
| 11 | 链表（Linked Lists: singly/doubly, tail pointers） | L19, L20 | 官方 2 个讲座页完整公开：链表解剖与内存图、从笨拙到优雅的多版实现（头插/尾插/删除）、尾指针维护、双向链表、用链表实现栈/队列、数组 vs 链表权衡；对应作业 6（Listy Things）题面页公开 |
| 12 | 二叉树、二叉搜索树与树遍历（Binary Trees, BSTs & Traversals） | L21, L22 | 官方 2 个讲座页完整公开：树术语、TreeNode 初探、BST 运行时、遍历算法；L22 覆盖删除三情形、自平衡 BST、遍历应用与代码（tree-notes.pdf、bst-code.zip、traversal-p |
| 13 | 霍夫曼编码：前缀码与编码树（Huffman Coding） | L23 | 官方讲座页完整公开：编码概览、ASCII、紧凑定长/变长编码、编码树构建与解码、最优树构造（讲义源自 A7 handout，作者 Julie Zelenski 等）；对应作业 7（Huffman Coding）题面页公开 |
| 14 | 散列与哈希表（Hashing & Hash Tables） | L24 | 官方讲座页完整公开：学生记录检索问题的多种方案权衡、线性探测（含聚类与表大小）、分离链、运行时分析、好哈希函数性质、斯坦福 HashSet/HashMap 与复杂度表达注意事项；L28 亦再次回顾 HashSet/Has |
| 15 | 图：概念、表示、DFS/BFS 与拓扑排序（Graphs） | L25 | 官方讲座页完整公开：图术语、种类与性质、三种表示法、斯坦福 Graph 类、MST/拓扑排序概念、DFS/BFS 遍历与路径动画（页内含 Prezi 动画链接） |
| 16 | 最短路径：Dijkstra 与 A*（Shortest Paths） | L26, L27 | 官方 2 个讲座页完整公开：Dijkstra 原理与最小 dist 选取的运行时考量、负权问题、Bellman-Ford 补充、A* 与启发式（附外部资源链接）、WeightedGraph 类编码与输入文件（dijkst |
| 17 | 拓展专题：Trie 与并查集（Bonus: Tries & Union-Find） | （延伸内容）官方 | 官方公开信息有限：L24 文字讲义中明确提及 trie（按学号数字逐位走 10 叉树）并建议课外深究；本章主体为面向求职/后续课程的原创扩展笔记 |

---

## C++ STL 容器与算法速查表

> 课程官方使用 Stanford C++ Library（`Vector`、`Grid`、`Stack`、`Queue`、`Set`、`Map`、`HashSet`、`HashMap`、`Lexicon`、`PriorityQueue`…）；本表统一给**标准库（STL）等价物**与常用操作，便于你在任何现代 C++（C++17）环境里动手。A.5 节给出两者对照。

###  高频“基础设施”备忘

```cpp
#include <iostream>      // cin / cout / endl
#include <string>        // std::string
#include <vector>        // std::vector 等容器
#include <algorithm>     // sort / find / reverse ...
#include <numeric>       // accumulate / iota
#include <cctype>        // isalpha / isdigit / tolower ...
#include <sstream>       // 字符串流（按词切分等）
using namespace std;     // 教学代码简化写法（工程上避免）

// 常用写法速记
auto    v = vector<int>{1,2,3};      // 类型推导 + 列表初始化
for (const auto& x : v) { /* 只读遍历，零拷贝 */ }
for (auto& x : v)       { /* 可修改遍历 */ }
const string& s2 = s;   // 引用：不拷贝、不修改（传参首选）
nullptr                  // 空指针字面量（替代 NULL/0）
```

传参铁律：**对象默认按值拷贝**。只读传 `const T&`，要改传 `T&`；小类型（`int`、`char`、`bool`、指针）按值即可。

###  std::string（字符串）

C++ 字符串是**可变**的（区别于 Python/Java 的不可变）字符序列。

| 操作 | 示例 | 说明 |
|---|---|---|
| 长度 | `s.size()` / `s.length()` | O(1) |
| 判空 | `s.empty()` | |
| 访问 | `s[i]`、`s.at(i)` | `at` 越界抛异常；`[]` 不检查 |
| 追加 | `s += c; s.append(t); s.push_back('x')` | |
| 插入/删除 | `s.insert(pos, t)`、`s.erase(pos, len)` | 下标版本 O(n) |
| 子串 | `s.substr(pos, len)` | len 缺省到末尾 |
| 查找 | `s.find(t)`、`s.rfind(t)` | 返回 `size_t`；找不到返回 `string::npos` |
| 比较 | `s == t`、`s < t` | 字典序 |
| C 风格 | `s.c_str()` | 得到 `const char*` |
| 读一行 | `getline(cin, s)` | 含空格整行 |

数值 ↔ 字符串：`to_string(42)`、`stoi(s)`（`stol/stod/...`）。

字符处理（`<cctype>`，参数按 `unsigned char` 转）：`isalpha/isalnum/isdigit/isspace/isupper/islower`、`toupper/tolower`。字符即小整数：`'a'` 是 97，`c - 'a'` 得 0–25。

```cpp
// 词频演示：统计一段文本中每个单词出现次数
#include <iostream>
#include <string>
#include <sstream>
#include <map>
using namespace std;
int main() {
    string text = "to be or not to be";
    map<string,int> freq;                 // 键自动按字典序排好
    istringstream iss(text);              // 按空白切词
    string w;
    while (iss >> w) ++freq[w];           // 首次访问自动插入并置 0，再自增
    for (const auto& [word, cnt] : freq)  // 结构化绑定 (C++17)
        cout << word << ": " << cnt << "\n";
}
// 输出：be: 2  not: 1  or: 1  to: 2
```

###  顺序容器

| 容器 | 底层 | 特点 | 常用操作 |
|---|---|---|---|
| `vector<T>` | 连续数组（自动扩容） | 尾部 O(1)（均摊）；随机访问 O(1)；中间插删 O(n) | `push_back/pop_back`、`back/front`、`size/empty`、`v[i]`、`insert/erase`、`reserve/resize`、`clear` |
| `deque<T>` | 分段连续 | 头尾都 O(1)（均摊） | `push_back/push_front/pop_back/pop_front` + vector 全部 |
| `list<T>` | 双向链表 | 任意位置插删 O(1)（已知位置）；**无随机访问** | `push_back/push_front`、`insert/erase`、`splice` |
| `array<T,N>` | 定长数组 | 编译期定长 | 同 vector（无 push_back） |

**vector 扩容机制（面试高频）**：满时按倍数（常见 2×）申请新块→拷贝/移动旧元素→释放旧块。单次 push_back 最坏 O(n)，但 n 次 push_back 总代价 O(n)，**均摊 O(1)**——这正是 Lecture 4 里“`add` 均摊快、`insert(0,·)` 每次都要搬动所有元素 O(n)”的原因。

```cpp
#include <vector>
vector<int> v{3,1,4,1,5};
v.push_back(9);                 // 3 1 4 1 5 9
v.insert(v.begin() + 2, 100);   // 在下标 2 前插入 → 3 1 100 4 1 5 9
v.erase(v.begin());             // 删第一个 → 1 100 4 1 5 9
int x = v.back(); v.pop_back(); // 取并弹尾部
```

迭代器：`begin()/end()`（`cbegin/cend` 只读），支持 `++/--/*` 与 `it + n`（随机访问容器）。**注意**：对 vector 做插入/删除会使指向其后元素的迭代器失效。

###  容器适配器（栈 / 队列 / 优先队列）

适配器 = 在底层容器上“限量”操作，只暴露一种出入口语义。

| 适配器 | 语义 | 入 | 出/看 | 常用 |
|---|---|---|---|---|
| `stack<T>` | LIFO 后进先出（默认底层 deque） | `push` | `pop`（无返回值）、`top` | `empty/size` |
| `queue<T>` | FIFO 先进先出 | `push`(入队) | `pop`(出队)、`front/back` | |
| `priority_queue<T>` | 最大堆（默认）；`greater<T>` 变最小堆 | `push` | `pop`、`top` | 底层 vector + make_heap |

```cpp
#include <stack>   #include <queue>
stack<int> st;  st.push(1); st.push(2);      // top=2
int t = st.top(); st.pop();                  // 弹出 2

queue<int> q;   q.push(1); q.push(2);        // front=1
int f = q.front(); q.pop();                  // 出 1

priority_queue<int> mx;                      // 默认最大堆
mx.push(3); mx.push(5);                      // top()==5
priority_queue<int, vector<int>, greater<int>> mn; // 最小堆 top()==最小
// 自定义类型：给 operator<，或传比较器/仿函数
```

###  关联容器（有序树 vs 哈希桶）

| 容器 | 底层 | 键序 | 查找/插删复杂度 |
|---|---|---|---|
| `set<T>` / `map<K,V>` | 红黑树（平衡 BST） | 有序（可遍历出升序） | O(log n) |
| `multiset` / `multimap` | 红黑树 | 有序，允许重复键 | O(log n) |
| `unordered_set<T>` / `unordered_map<K,V>` | 哈希表（桶） | 无序 | 平均 O(1)，最坏 O(n) |
| `unordered_multiset/multimap` | 哈希表 | 无序，允许重复 | 平均 O(1) |

```cpp
#include <set>    #include <map>
#include <unordered_set>   #include <unordered_map>
set<int> s{3,1,4}; s.insert(2);          // {1,2,3,4} 自动有序
if (s.count(4)) { /* 存在 */ }            // count∈{0,1}
s.erase(3);

map<string,int> m;
m["alice"] = 1;                           // [] 会“探键”：不存在则插入默认值
m.at("alice");                            // at() 不存在时抛异常（安全版）
auto it = m.find("bob");                  // 找不到 == m.end()
for (const auto& [k, v] : m) { /* 键升序 */ }

unordered_map<string,int> u;              // 无序版：平均更快，无序遍历
```

**取舍**：需要“有序遍历/前驱后继/范围查询”→ `set/map`（BST）；只做查存、不关心顺序、量大 → `unordered_*`（哈希）。这正是 Lecture 3（树版有序）与 Lecture 14（哈希版）的对照。

###  常用算法（`<algorithm>` / `<numeric>`）

| 算法 | 示例 | 说明 |
|---|---|---|
| 排序 | `sort(v.begin(), v.end())` | 升序；`sort(b,e,greater<int>())` 降序 |
| 稳定排序 | `stable_sort(...)` | 相等元素保持原相对顺序 |
| 反转 | `reverse(v.begin(), v.end())` | |
| 线性查找 | `find(v.begin(), v.end(), x)` | 返回迭代器；`==end()` 未找到 |
| 计数 | `count(...)` | |
| 最值 | `min/max(a,b)`；`min_element/max_element(b,e)` | 返回迭代器 |
| 二分查找 | `binary_search(vb,ve,x)`；`lower_bound/upper_bound` | **前提：已排序**，O(log n) |
| 累加 | `accumulate(vb,ve,0)`；`accumulate(vb,ve,0.0)` | 整数/浮点初值要写对 |
| 填值 | `fill(vb,ve,0)`；`iota(vb,ve,0)`（填 0,1,2,…） | |
| 去重 | `sort` 后 `unique` + `erase` | 见下 |
| 全排列 | `next_permutation(vb,ve)` | 返回 bool，配合 do-while 枚举 |
| 变换/遍历 | `transform`、`for_each` | 配合 lambda |

```cpp
#include <algorithm>
vector<int> v{5,1,4,1,3};
sort(v.begin(), v.end());                       // 1 1 3 4 5
v.erase(unique(v.begin(), v.end()), v.end());   // 去重 → 1 3 4 5
bool ok = binary_search(v.begin(), v.end(), 4); // true（已排序）
auto it = lower_bound(v.begin(), v.end(), 3);   // 第一个 >=3 的位置
sort(v.begin(), v.end(), [](int a,int b){ return a > b; }); // lambda 自定义
```

###  其他实用件

- `pair<K,V>`：`make_pair`/`{k,v}`；`tie`/结构化绑定解包。`tuple` 同理。
- 计时（`<chrono>`，Lecture 4 可用来实测运行时间）：
  ```cpp
  auto t0 = chrono::steady_clock::now();
  /* ...被测代码... */
  auto ms = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - t0).count();
  cout << ms << " ms\n";
  ```
- 随机数（`<random>`）：`mt19937 gen(rd()); uniform_int_distribution<int> d(1,6); d(gen);`
- 输入输出流 `cin/cout`：读整数 `cin >> x`；读整行 `getline(cin,s)`；`<iomanip>` 的 `setw/setprecision` 排版。

###  课程 Stanford 库 ↔ STL 对照

| Stanford C++ 库 | 语义 | 本笔记/STL 等价 |
|---|---|---|
| `Vector<T>` | 动态数组 | `std::vector<T>` |
| `Grid<T>` | 二维网格 | `std::vector<std::vector<T>>` |
| `Stack<T>` | LIFO | `std::stack<T>` |
| `Queue<T>` | FIFO | `std::queue<T>` |
| `Set<T>` | 有序集合（平衡 BST） | `std::set<T>` |
| `Map<K,V>` | 有序映射（平衡 BST） | `std::map<K,V>` |
| `HashSet<T>` | 无序集合（哈希） | `std::unordered_set<T>` |
| `HashMap<K,V>` | 无序映射（哈希） | `std::unordered_map<K,V>` |
| `PriorityQueue<T>` | 优先队列 | `std::priority_queue<T>`（注意默认是最大堆，课程常用最小堆语义） |
| `Lexicon` | 词典（trie/哈希实现） | `std::unordered_set<std::string>` 或 Trie（Lecture 17） |
| `strlib`（toLowerCase/…） | 字符串工具 | `std::tolower`/`<algorithm>` 手写小工具 |
| `SimpleTest` | 单元测试 | 自写 `assert`/简单测试函数 |
| `randomInteger(a,b)` | 随机整数 | `<random>` 的 uniform_int_distribution |

> 一句话总结：**学 ADT 先学“语义”（栈/队列/映射…），再学“实现”（数组/链表/树/哈希）**——CS106B 的精髓就是把“用什么”和“怎么造”分开想清楚，本笔记各章正是按这条主线展开的。

---
*笔记完。祝学习愉快——画图、动手、多问为什么。*

{% endraw %}
