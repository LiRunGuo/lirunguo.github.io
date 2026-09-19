---
title: "UIUC ECE 408 应用并行编程"
excerpt: "伊利诺伊大学 UIUC ECE 408 / CS 483 Applied Parallel Programming 系统学习笔记，涵盖 CUDA 编程模型、线程与内存层次、tiling、归并与扫描、性能优化。"
collection: course-notes
permalink: /course-notes/uiuc-ece408-applied-parallel-programming
toc: false
toc_sticky: false
---

本笔记按章节拆分为子页，逐章阅读更快。

伊利诺伊大学 UIUC ECE 408 / CS 483 Applied Parallel Programming 系统学习笔记，涵盖 CUDA 编程模型、线程与内存层次、tiling、归并与扫描、性能优化。

## 章节

- [开篇与课程概览](/course-notes/uiuc-ece408-applied-parallel-programming/l0)
- [Lecture 1: 课程概述、并行计算动机与 GPU 架构导论 (对应 Lab 0: Device Query)](/course-notes/uiuc-ece408-applied-parallel-programming/l1)
- [Lecture 2: CUDA 编程模型基础 —— Kernel、线程层次与内存模型 (对应 Lab 0 / Lab 1)](/course-notes/uiuc-ece408-applied-parallel-programming/l2)
- [Lecture 3: 并行模式一 —— 向量加法：线程映射与内存访问 (对应 Lab 1: Vector Addition)](/course-notes/uiuc-ece408-applied-parallel-programming/l3)
- [Lecture 4: 并行模式二 —— 基础矩阵乘法及其性能瓶颈 (对应 Lab 2: Simple Matrix Multiply)](/course-notes/uiuc-ece408-applied-parallel-programming/l4)
- [Lecture 5: 并行模式三 —— 分块矩阵乘法：共享内存与数据复用 (对应 Lab 3: Tiled Matrix Multiply)](/course-notes/uiuc-ece408-applied-parallel-programming/l5)
- [Lecture 6: 并行模式四 —— 归约：树形归约与 warp 级原语 (对应 Lab 4 / Lab 5: List Reduction)](/course-notes/uiuc-ece408-applied-parallel-programming/l6)
- [Lecture 7: 并行模式五 —— 前缀和 / 扫描：双缓冲与层次化算法 (对应 Lab 5 / Lab 6: Scan)](/course-notes/uiuc-ece408-applied-parallel-programming/l7)
- [Lecture 8: 并行模式六 —— 分块卷积：常量内存与边界处理 (对应 Lab 6 / Lab 4: 3D Convolution)](/course-notes/uiuc-ece408-applied-parallel-programming/l8)
- [Lecture 9: 并行模式七 —— 稀疏矩阵向量乘：压缩格式与负载均衡 (对应 Lab 7 / Lab 8: Sparse Matrix Multiply)](/course-notes/uiuc-ece408-applied-parallel-programming/l9)
- [Lecture 10: 性能分析与优化方法论 —— Roofline、占用率、合并访问与原子操作 (对应全部 Lab)](/course-notes/uiuc-ece408-applied-parallel-programming/l10)
- [Lecture 11: 高级主题 —— 多流、数据传输、张量核心、深度学习与 CUDA 替代方案 (对应最终项目)](/course-notes/uiuc-ece408-applied-parallel-programming/l11)
- [Lecture 12: 最终项目 —— 问题定义、方案设计、性能优化与报告 (对应 Final Project)](/course-notes/uiuc-ece408-applied-parallel-programming/l12)
- [附录：CUDA 并行模式与优化速查表](/course-notes/uiuc-ece408-applied-parallel-programming/appendix)
