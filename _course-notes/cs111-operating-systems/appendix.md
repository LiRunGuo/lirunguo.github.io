---
title: "核心系统调用与 API 速查表（Quick Reference）"
collection: course-notes
chapter: true
permalink: /course-notes/cs111-operating-systems/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/cs111-operating-systems/) · [← l21](/course-notes/cs111-operating-systems/l21)

{% raw %}
# 核心系统调用与 API 速查表（Quick Reference）

## 进程与线程

| API / 系统调用 | 头文件 | 说明 |
| :--- | :--- | :--- |
| `fork()` | `<unistd.h>` | 创建子进程（复制当前进程）；父进程返回子 PID，子进程返回 0 |
| `execvp(path, argv)` | `<unistd.h>` | 用新程序覆盖当前进程的代码/数据；成功不返回 |
| `waitpid(pid, &status, opts)` | `<sys/wait.h>` | 等待子进程结束并回收退出状态（防僵尸） |
| `getpid() / getppid()` | `<unistd.h>` | 获取当前进程/父进程 PID |
| `exit(status)` | `<stdlib.h>` | 终止进程（返回状态给父进程） |
| `std::thread t(f, args)` | `<thread>` | C++ 创建线程；`t.join()` 等待，`t.detach()` 分离 |
| `pthread_create/join` | `<pthread.h>` | POSIX 线程创建/等待（C 语言） |
| `clone(fn, stack, flags, …)` | `<sched.h>` | Linux 底层线程/进程创建（fork 的实现基础） |
| `setuid/setgid` | `<unistd.h>` | 放弃/切换特权（最小权限原则） |

## 同步

| API | 头文件 | 说明 |
| :--- | :--- | :--- |
| `std::mutex` | `<mutex>` | 互斥锁：`lock()` 阻塞获取，`unlock()` 释放，`try_lock()` 非阻塞 |
| `std::lock_guard` / `std::unique_lock` | `<mutex>` | RAII 锁包装（构造加锁、析构解锁）；`unique_lock` 可配合条件变量 |
| `std::condition_variable` | `<condition_variable>` | 条件变量：`wait(lock)` 原子释放锁并睡眠；`notify_one()`/`notify_all()` 唤醒 |
| `std::atomic<T>` | `<atomic>` | 原子类型（如 `std::atomic<bool>`），编译为原子指令（如 `exchange`） |
| `pthread_mutex_lock/unlock` | `<pthread.h>` | POSIX 互斥锁 |
| `pthread_cond_wait/signal` | `<pthread.h>` | POSIX 条件变量 |

## 内存

| API / 系统调用 | 头文件 | 说明 |
| :--- | :--- | :--- |
| `malloc / free`、`new / delete` | `<stdlib.h>` / C++ | 堆内存分配/释放（走空闲链表/slab；不足时向 OS 申请） |
| `mmap(addr, len, prot, flags, fd, off)` | `<sys/mman.h>` | 把文件/匿名区域映射进虚拟地址空间（Assign5 核心） |
| `munmap(addr, len)` | `<sys/mman.h>` | 解除映射 |
| `mprotect(addr, len, prot)` | `<sys/mman.h>` | 修改页的访问权限（PROT_NONE/READ/WRITE，制造伪缺页） |
| `msync(addr, len, flags)` | `<sys/mman.h>` | 把脏页写回文件 |
| `brk/sbrk` | `<unistd.h>` | 扩展/收缩数据段（堆） |
| `std::shared_ptr<T>` | `<memory>` | 引用计数智能指针（注意循环引用需 weak_ptr） |

## 文件与目录

| API / 系统调用 | 头文件 | 说明 |
| :--- | :--- | :--- |
| `open(path, flags, mode)` | `<fcntl.h>` | 打开文件/创建设备描述符；内核逐级查目录、载入 inode |
| `close(fd)` | `<unistd.h>` | 关闭文件描述符 |
| `read(fd, buf, n)` / `write(fd, buf, n)` | `<unistd.h>` | 从当前偏移顺序读写（块缓存/延迟写） |
| `pread/pwrite(fd, buf, n, off)` | `<unistd.h>` | 指定偏移的随机读写（数据库常用） |
| `fsync(fd)` / `fdatasync(fd)` | `<unistd.h>` | 强制把脏块刷到磁盘（耐久性） |
| `lseek(fd, off, whence)` | `<unistd.h>` | 移动文件偏移 |
| `ftruncate(fd, len)` | `<unistd.h>` | 设置文件长度 |
| `stat / fstat / lstat` | `<sys/stat.h>` | 获取文件元数据（inode 信息：大小、权限、nlink；lstat 不跟随符号链接） |
| `link(old, new)` | `<unistd.h>` | 创建硬链接（共享 inode，nlink++） |
| `symlink(target, linkpath)` | `<unistd.h>` | 创建符号链接（内容为路径字符串） |
| `unlink(path)`（`rm`） | `<unistd.h>` | 删除目录项（硬链接计数--；计数归零才释放数据） |
| `mkdir/rmdir/opendir/readdir` | `<sys/stat.h>` `<dirent.h>` | 目录操作 |
| `chmod/chown` | `<sys/stat.h>` | 修改权限/属主（保护机制） |
| `dup/dup2` | `<unistd.h>` | 复制文件描述符（重定向，如 `ls > out`） |
| `pipe(fds)` | `<unistd.h>` | 创建管道（进程间通信，生产者-消费者的系统形态） |
| `fstrim / TRIM` | 命令行 / `ioctl` | 通知 SSD 释放的块（闪存管理） |

## 进程间通信与其它

| API / 系统调用 | 头文件 | 说明 |
| :--- | :--- | :--- |
| `pipe`、`dup2` | `<unistd.h>` | 管道与重定向 |
| `signal / sigaction` | `<csignal>` `<signal.h>` | 信号处理（如捕获 SIGSEGV 模拟缺页） |
| `nice(inc)` / shell `nice -n` | `<unistd.h>` | 调整进程调度优先级（-20..+19） |
| `ptrace` | `<sys/ptrace.h>` | 进程追踪（gdb 的基础） |
| `getuid/setuid` | `<unistd.h>` | 用户身份查询/切换（信任边界） |

---

> **备注**：本笔记中的图表（ASCII 图）根据讲义幻灯片重新绘制；代码示例为教学用途的简化版本。讲义中还有更多深入案例（如 Lecture 5 的 Pipe 逐版本推演、Lecture 9 的链接器三遍扫描实例、Lecture 20 的读块 23/1040 例子），建议结合原始 PDF 学习：https://web.stanford.edu/class/archive/cs/cs111/cs111.1266/

{% endraw %}
