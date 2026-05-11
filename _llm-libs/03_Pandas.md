---
title: "Pandas 数据处理库"
excerpt: "DataFrame、数据读写、清洗、groupby聚合、merge/join、训练数据预处理"
collection: llm-libs
permalink: /llm-libs/03-pandas
category: core
---


## 1. 简介与在 LLM 开发中的作用

Pandas 是 Python 生态中最广泛使用的数据分析和操作库，提供了高性能、易用的数据结构（DataFrame 和 Series）以及丰富的数据分析工具。它构建在 NumPy 之上，能够高效处理表格型、异构型数据。

在 LLM（大语言模型）开发中，Pandas 扮演着不可或缺的角色：

- **训练数据加载与预处理**：LLM 的训练数据通常以 CSV、JSON、Parquet 等格式存储，Pandas 是加载和清洗这些数据的首选工具
- **评估结果分析**：对模型输出进行批量评估、统计指标计算、结果可视化
- **数据集清洗与去重**：去除低质量样本、处理缺失值、去除重复数据，确保训练数据质量
- **特征工程**：文本长度统计、标签分布分析、数据采样与分割
- **Prompt 批量处理**：对大规模 prompt 数据集进行模板化处理和批量生成

---

## 2. 安装方式

```bash
# 基础安装
pip install pandas

# 带有完整依赖的安装（支持 Excel、SQL、HTML 等格式）
pip install pandas[all]

# 常用搭配安装（LLM 开发推荐）
pip install pandas numpy pyarrow fastparquet

# Conda 安装
conda install pandas
```

安装后验证：

```python
import pandas as pd
print(pd.__version__)  # 输出: 2.x.x
```

---

## 3. 核心数据结构与操作

### 3.1 Series — 一维数据结构

Series 是带标签的一维数组，可以存储任意数据类型（整数、字符串、浮点数、Python 对象等）。

```python
import pandas as pd
import numpy as np

# 从列表创建
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64

# 指定索引
s = pd.Series([90, 85, 78], index=['math', 'english', 'physics'], name='scores')
print(s['math'])  # 90

# 从字典创建
s = pd.Series({'a': 1, 'b': 2, 'c': 3})

# 常用属性
print(s.index)    # 索引对象
print(s.values)   # 底层 NumPy 数组
print(s.dtype)    # 数据类型
print(s.shape)    # 形状
print(s.name)     # 名称
```

**Series 常用参数**：

| 参数 | 说明 |
|------|------|
| `data` | 数据源（列表、字典、NumPy 数组等） |
| `index` | 索引标签 |
| `dtype` | 数据类型 |
| `name` | Series 名称 |

### 3.2 DataFrame — 二维表格数据结构

DataFrame 是 Pandas 的核心数据结构，是一个二维的、带标签的表格，每列可以是不同的数据类型。

```python
import pandas as pd
import numpy as np

# 从字典创建
df = pd.DataFrame({
    'prompt': ['翻译以下句子', '总结这段文字', '生成代码'],
    'response': ['Hello -> 你好', '本文主要讲述...', 'def hello():'],
    'score': [0.95, 0.87, 0.91],
    'length': [5, 8, 15]
})

# 从列表的列表创建
df = pd.DataFrame(
    [[1, 'a'], [2, 'b'], [3, 'c']],
    columns=['id', 'label']
)

# 从 NumPy 数组创建
df = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])

# 常用属性
print(df.shape)       # (行数, 列数)
print(df.columns)     # 列名
print(df.index)       # 行索引
print(df.dtypes)      # 每列的数据类型
print(df.values)      # 底层 NumPy 数组
print(df.head(2))     # 前2行
print(df.tail(2))     # 后2行
print(df.info())      # 数据概览
print(df.describe())  # 统计描述
```

**DataFrame 常用参数**：

| 参数 | 说明 |
|------|------|
| `data` | 数据源 |
| `index` | 行索引 |
| `columns` | 列名 |
| `dtype` | 数据类型 |
| `copy` | 是否深拷贝 |

### 3.3 索引与选择

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4', 'Claude-3', 'Llama-3'],
    'accuracy': [0.92, 0.89, 0.85],
    'latency_ms': [1200, 800, 200]
}, index=['run1', 'run2', 'run3'])

# ---- 基础选择 ----
df['model']              # 选择单列，返回 Series
df[['model', 'accuracy']]  # 选择多列，返回 DataFrame

# ---- 标签索引 .loc ----
df.loc['run1']                    # 按行标签选择
df.loc['run1', 'model']          # 指定行和列
df.loc['run1':'run2', 'model']   # 切片（两端包含）
df.loc[df['accuracy'] > 0.86]    # 布尔索引

# ---- 位置索引 .iloc ----
df.iloc[0]            # 第0行
df.iloc[0, 1]         # 第0行第1列
df.iloc[0:2, 0:2]     # 切片（右端不包含）
df.iloc[[0, 2], [0, 2]]  # 花式索引

# ---- 条件过滤 ----
df[df['accuracy'] > 0.86]
df[(df['accuracy'] > 0.85) & (df['latency_ms'] < 1000)]
df.query('accuracy > 0.86 and latency_ms < 1000')

# ---- 设置/修改值 ----
df.loc['run1', 'accuracy'] = 0.93
df['speed_score'] = 1000 / df['latency_ms']  # 新增计算列
```

---

## 4. 数据读取与写入

### 4.1 读取数据

```python
import pandas as pd

# ---- CSV ----
df = pd.read_csv('data.csv')
df = pd.read_csv(
    'data.csv',
    sep=',',           # 分隔符
    header=0,          # 列名所在行，None表示无列名
    index_col=0,       # 用作索引的列
    usecols=['col1', 'col2'],  # 只读取指定列
    dtype={'col1': str},       # 指定列类型
    nrows=1000,        # 只读取前N行
    encoding='utf-8',  # 编码
    na_values=['NA', 'null'],  # 视为缺失值的字符串
    chunksize=10000    # 分块读取（返回迭代器）
)

# ---- JSON ----
df = pd.read_json('data.json')
df = pd.read_json(
    'data.json',
    orient='records',  # JSON格式: records, columns, index, values, split
    lines=True         # 每行一个JSON对象（JSONL格式，LLM数据集常用）
)

# ---- Parquet（列式存储，大数据推荐） ----
df = pd.read_parquet('data.parquet')
df = pd.read_parquet(
    'data.parquet',
    engine='pyarrow',  # 引擎: pyarrow, fastparquet
    columns=['col1', 'col2']  # 只读取指定列
)

# ---- Excel ----
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# ---- SQL ----
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)
df = pd.read_sql_table('table_name', conn)  # 按表名读取
```

### 4.2 写入数据

```python
# ---- CSV ----
df.to_csv('output.csv', index=False, encoding='utf-8')

# ---- JSON ----
df.to_json('output.json', orient='records', force_ascii=False, indent=2)
df.to_json('output.jsonl', orient='records', lines=True)  # JSONL格式

# ---- Parquet ----
df.to_parquet('output.parquet', engine='pyarrow', compression='snappy')

# ---- Excel ----
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
```

**LLM 开发中的数据格式选择**：

| 格式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| CSV | 简单表格数据 | 通用、可读 | 无类型信息、大文件慢 |
| JSON/JSONL | 嵌套结构、对话数据 | 灵活、支持嵌套 | 文件较大 |
| Parquet | 大规模数据集 | 高压缩率、列式读取 | 二进制不可读 |

---

## 5. 数据清洗

### 5.1 缺失值处理

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'text': ['Hello', None, 'World', 'Test', None],
    'score': [0.9, 0.8, np.nan, 0.7, np.nan],
    'label': ['A', 'B', 'A', None, 'B']
})

# ---- 检测缺失值 ----
print(df.isnull())       # 返回布尔DataFrame
print(df.isna())         # 同 isnull
print(df.notnull())      # 非空检测
print(df.isnull().sum()) # 每列缺失值计数
print(df.isnull().any()) # 每列是否有缺失值

# ---- 删除缺失值 ----
df_clean = df.dropna()                    # 删除含缺失值的行
df_clean = df.dropna(how='all')           # 只删除全为缺失值的行
df_clean = df.dropna(subset=['text'])     # 只检查指定列
df_clean = df.dropna(axis=1)              # 删除含缺失值的列
df_clean = df.dropna(thresh=2)            # 保留至少2个非空值的行

# ---- 填充缺失值 ----
df_filled = df.fillna(0)                          # 用常数填充
df_filled = df.fillna({'score': 0.0, 'label': 'Unknown'})  # 每列指定填充值
df_filled = df.fillna(df.mean(numeric_only=True)) # 用均值填充
df_filled = df.fillna(method='ffill')             # 前向填充
df_filled = df.fillna(method='bfill')             # 后向填充
df_filled = df.interpolate()                      # 插值填充
```

**参数说明**：

| 方法 | 参数 | 说明 |
|------|------|------|
| `dropna` | `how` | `'any'`（默认）任一缺失即删除；`'all'` 全部缺失才删除 |
| `dropna` | `subset` | 只检查指定列 |
| `dropna` | `thresh` | 保留至少 N 个非空值的行 |
| `fillna` | `value` | 填充值或字典 |
| `fillna` | `method` | `'ffill'` 前向填充，`'bfill'` 后向填充 |
| `fillna` | `inplace` | 是否原地修改 |

### 5.2 重复值处理

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': ['翻译', '翻译', '总结', '生成'],
    'response': ['Hello', 'Hello', 'World', 'Code']
})

# 检测重复值
print(df.duplicated())                # 返回布尔Series
print(df.duplicated(subset=['prompt']))  # 指定列检测
print(df.duplicated(keep='first'))    # 保留第一次出现
print(df.duplicated(keep='last'))     # 保留最后一次出现
print(df.duplicated(keep=False))      # 标记所有重复

# 删除重复值
df_unique = df.drop_duplicates()                    # 删除完全重复的行
df_unique = df.drop_duplicates(subset=['prompt'])   # 按指定列去重
df_unique = df.drop_duplicates(keep='last')         # 保留最后出现
df_unique = df.drop_duplicates(inplace=True)        # 原地修改

# LLM数据去重：按文本相似度去重（结合embedding）
# 先按文本长度排序，再去除完全重复
df_sorted = df.sort_values('prompt', key=lambda x: x.str.len())
df_dedup = df_sorted.drop_duplicates(subset=['prompt'], keep='first')
```

### 5.3 类型转换

```python
import pandas as pd

df = pd.DataFrame({
    'text_id': ['1', '2', '3'],
    'score': ['0.9', '0.8', '0.7'],
    'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'is_valid': [1, 0, 1]
})

# astype —— 基本类型转换
df['text_id'] = df['text_id'].astype(int)
df['score'] = df['score'].astype(float)
df['is_valid'] = df['is_valid'].astype(bool)

# to_numeric —— 安全数值转换
df['score'] = pd.to_numeric(df['score'], errors='coerce')  # 无法转换的设为NaN
df['score'] = pd.to_numeric(df['score'], errors='raise')   # 无法转换的抛异常

# to_datetime —— 日期转换
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# category 类型（低基数字符串，节省内存）
df['is_valid'] = df['is_valid'].astype('category')

# 转换后验证
print(df.dtypes)
```

---

## 6. 分组聚合

### 6.1 groupby 基础

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4', 'GPT-4', 'Claude-3', 'Claude-3', 'Llama-3', 'Llama-3'],
    'task': ['translation', 'summarization', 'translation', 'summarization',
             'translation', 'summarization'],
    'score': [0.95, 0.88, 0.91, 0.85, 0.82, 0.79],
    'tokens': [150, 300, 120, 280, 80, 200]
})

# 基本分组
grouped = df.groupby('model')          # 按单列分组
grouped = df.groupby(['model', 'task']) # 按多列分组

# 分组后聚合
print(df.groupby('model')['score'].mean())  # 每个模型的平均分
print(df.groupby('model').size())           # 每组的大小
print(df.groupby('model').count())          # 每组非空值计数

# 常用聚合方法
print(df.groupby('model')['score'].sum())
print(df.groupby('model')['score'].max())
print(df.groupby('model')['score'].min())
print(df.groupby('model')['score'].std())
print(df.groupby('model')['score'].median())
print(df.groupby('model')['score'].quantile(0.75))  # 75分位数
```

### 6.2 agg —— 多聚合函数

```python
# 同时计算多个聚合指标
result = df.groupby('model').agg({
    'score': ['mean', 'std', 'max', 'min'],
    'tokens': ['sum', 'mean']
})

# 自定义聚合函数名
result = df.groupby('model').agg(
    avg_score=('score', 'mean'),
    max_score=('score', 'max'),
    total_tokens=('tokens', 'sum'),
    count=('score', 'count')
)

# 使用自定义聚合函数
result = df.groupby('model').agg(
    score_range=('score', lambda x: x.max() - x.min()),
    cv=('score', lambda x: x.std() / x.mean())  # 变异系数
)
```

### 6.3 transform —— 分组变换

`transform` 返回与原 DataFrame 相同长度的结果，用于在组内进行计算并将结果广播回每一行。

```python
# 组内标准化（每个模型内部做标准化）
df['score_normalized'] = df.groupby('model')['score'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# 组内排名
df['score_rank'] = df.groupby('model')['score'].rank(ascending=False)

# 填充组内缺失值
df['score_filled'] = df.groupby('model')['score'].transform(
    lambda x: x.fillna(x.mean())
)
```

### 6.4 apply —— 分组自定义函数

```python
# 选择每组得分最高的行
best_per_model = df.groupby('model').apply(
    lambda x: x.nlargest(1, 'score'),
    include_groups=False
)

# 自定义复杂逻辑
def analyze_group(group):
    return pd.Series({
        'avg_score': group['score'].mean(),
        'total_tokens': group['tokens'].sum(),
        'best_task': group.loc[group['score'].idxmax(), 'task']
    })

result = df.groupby('model').apply(analyze_group)
```

---

## 7. 合并连接

### 7.1 merge —— 数据库风格连接

```python
import pandas as pd

# 模型评估结果表
eval_df = pd.DataFrame({
    'run_id': ['R001', 'R002', 'R003', 'R004'],
    'model': ['GPT-4', 'Claude-3', 'Llama-3', 'GPT-4'],
    'accuracy': [0.92, 0.89, 0.85, 0.93]
})

# 模型配置表
config_df = pd.DataFrame({
    'model': ['GPT-4', 'Claude-3', 'Llama-3', 'Mistral'],
    'params_B': [1750, 175, 70, 7],
    'type': ['proprietary', 'proprietary', 'open', 'open']
})

# 内连接（默认）—— 只保留两表都有的键
result = pd.merge(eval_df, config_df, on='model', how='inner')

# 左连接 —— 保留左表所有行
result = pd.merge(eval_df, config_df, on='model', how='left')

# 右连接 —— 保留右表所有行
result = pd.merge(eval_df, config_df, on='model', how='right')

# 外连接 —— 保留所有行
result = pd.merge(eval_df, config_df, on='model', how='outer')

# 不同列名连接
result = pd.merge(df1, df2, left_on='model_name', right_on='model', how='inner')

# 多键连接
result = pd.merge(df1, df2, on=['model', 'task'], how='inner')
```

**merge 参数说明**：

| 参数 | 说明 |
|------|------|
| `how` | 连接方式：`'inner'`（默认）、`'left'`、`'right'`、`'outer'`、`'cross'` |
| `on` | 连接键（两表列名相同时） |
| `left_on` / `right_on` | 左/右表的连接键（列名不同时） |
| `suffixes` | 重名列后缀，默认 `('_x', '_y')` |
| `indicator` | 是否添加 `_merge` 列标记来源 |

### 7.2 join —— 索引连接

```python
# 基于索引连接
df1 = pd.DataFrame({'A': [1, 2]}, index=['a', 'b'])
df2 = pd.DataFrame({'B': [3, 4]}, index=['a', 'c'])

result = df1.join(df2, how='left')      # 左连接
result = df1.join(df2, how='inner')     # 内连接
result = df1.join(df2, how='outer')     # 外连接

# 多个DataFrame同时join
result = df1.join([df2, df3], how='outer')
```

### 7.3 concat —— 拼接

```python
import pandas as pd

df1 = pd.DataFrame({'model': ['GPT-4'], 'score': [0.92]})
df2 = pd.DataFrame({'model': ['Claude-3'], 'score': [0.89]})
df3 = pd.DataFrame({'latency': [1200, 800]})

# 纵向拼接（行拼接，最常用）
result = pd.concat([df1, df2], axis=0, ignore_index=True)
#   model  score
# 0    GPT-4   0.92
# 1  Claude-3   0.89

# 横向拼接（列拼接）
result = pd.concat([df1, df3], axis=1)

# 带键的拼接（用于区分来源）
result = pd.concat([df1, df2], keys=['exp1', 'exp2'])

# join 参数
result = pd.concat([df1, df2], join='inner')  # 只保留共有列
result = pd.concat([df1, df2], join='outer')   # 保留所有列（默认）
```

**concat 参数说明**：

| 参数 | 说明 |
|------|------|
| `objs` | 要拼接的 DataFrame 列表 |
| `axis` | `0` 纵向拼接（默认），`1` 横向拼接 |
| `ignore_index` | 是否重置索引（默认 False） |
| `join` | `'outer'`（默认，并集）或 `'inner'`（交集） |
| `keys` | 为各部分添加层级索引 |

---

## 8. 在 LLM 开发中的典型使用场景和代码示例

### 8.1 训练数据加载与预处理

```python
import pandas as pd

# 加载 JSONL 格式的指令微调数据集
df = pd.read_json('alpaca_data.jsonl', lines=True)
print(f"数据集大小: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 数据质量检查
print(df.isnull().sum())         # 缺失值统计
print(df.duplicated().sum())     # 重复数据统计

# 清洗数据
df = df.dropna(subset=['instruction', 'output'])  # 删除关键字段缺失的行
df = df.drop_duplicates(subset=['instruction'])   # 按指令去重

# 文本长度分析
df['input_len'] = df['input'].str.len()
df['output_len'] = df['output'].str.len()
df['instruction_len'] = df['instruction'].str.len()

print(f"输入长度分布:\n{df['input_len'].describe()}")
print(f"输出长度分布:\n{df['output_len'].describe()}")

# 过滤异常样本（过长或过短）
df = df[(df['output_len'] >= 10) & (df['output_len'] <= 4096)]

# 保存清洗后的数据
df.to_json('alpaca_clean.jsonl', orient='records', lines=True, force_ascii=False)
```

### 8.2 评估结果分析

```python
import pandas as pd
import numpy as np

# 模拟多个模型在多个任务上的评估结果
results = []
for model in ['GPT-4', 'Claude-3', 'Llama-3']:
    for task in ['math', 'coding', 'writing', 'reasoning']:
        results.append({
            'model': model,
            'task': task,
            'accuracy': np.random.uniform(0.7, 0.98),
            'latency_ms': np.random.uniform(100, 2000)
        })

df = pd.DataFrame(results)

# 各模型平均表现
model_summary = df.groupby('model').agg(
    avg_accuracy=('accuracy', 'mean'),
    avg_latency=('latency_ms', 'mean')
).round(4)
print("各模型平均表现:\n", model_summary)

# 各任务模型排名
df['rank'] = df.groupby('task')['accuracy'].rank(ascending=False)
pivot = df.pivot_table(index='task', columns='model', values='accuracy')
print("任务×模型准确率:\n", pivot.round(4))

# 性价比分析（准确率/延迟）
df['efficiency'] = df['accuracy'] / (df['latency_ms'] / 1000)
print("性价比排名:\n", df.sort_values('efficiency', ascending=False)[['model', 'task', 'efficiency']])
```

### 8.3 数据集分割与采样

```python
import pandas as pd

df = pd.read_json('dataset.jsonl', lines=True)

# 随机采样
sample = df.sample(n=1000, random_state=42)           # 固定数量采样
sample = df.sample(frac=0.1, random_state=42)          # 按比例采样

# 分层采样（保证各类别比例一致）
stratified = df.groupby('category').apply(
    lambda x: x.sample(frac=0.2, random_state=42),
    include_groups=False
).reset_index(drop=True)

# 训练/验证/测试集分割
def train_val_test_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """按比例分割数据集"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    shuffled = df.sample(frac=1, random_state=random_state)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return shuffled.iloc[:train_end], shuffled.iloc[train_end:val_end], shuffled.iloc[val_end:]

train_df, val_df, test_df = train_val_test_split(df)
print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
```

### 8.4 Prompt 批量处理与模板化

```python
import pandas as pd

# 加载原始数据
df = pd.read_csv('qa_data.csv')

# 批量生成 prompt
template = "请根据以下上下文回答问题。\n\n上下文：{context}\n\n问题：{question}\n\n回答："

df['prompt'] = df.apply(
    lambda row: template.format(context=row['context'], question=row['question']),
    axis=1
)

# 按类别统计 prompt 分布
print(df['category'].value_counts())

# 按类别限制数量（平衡数据集）
balanced = df.groupby('category').apply(
    lambda x: x.sample(n=min(len(x), 500), random_state=42),
    include_groups=False
).reset_index(drop=True)

# 导出为推理用的格式
df[['prompt']].to_json('prompts.jsonl', orient='records', lines=True, force_ascii=False)
```

---

## 9. 数学原理

### 9.1 数据结构底层实现

Pandas 的 DataFrame 底层基于 NumPy 的 ndarray 数组。每一列（Series）是一个独立的 ndarray，通过 BlockManager 进行管理。

```
DataFrame
├── Index (行索引，基于 NumPy ndarray)
├── Columns (列索引，Index 对象)
└── BlockManager
    ├── NumericBlock → ndarray[float64] (连续内存)
    ├── ObjectBlock → ndarray[object] (指针数组)
    └── CategoricalBlock → ndarray[int8] + 分类映射
```

这种列式存储的设计使得：
- 同类型列的数据在内存中连续，有利于向量化运算
- 对单列的操作（如 `df['col'].mean()`）缓存友好，效率高
- 不同列可以使用不同的数据类型

### 9.2 分组聚合的数学表达

groupby 操作的数学本质是：

$$\text{agg}(G_k) = f(\{x_i \mid i \in G_k\})$$

其中 $G_k$ 是第 $k$ 个分组，$f$ 是聚合函数（mean、sum 等）。

- **mean**: $\bar{x}_k = \frac{1}{|G_k|} \sum_{i \in G_k} x_i$
- **std**: $s_k = \sqrt{\frac{1}{|G_k|-1} \sum_{i \in G_k} (x_i - \bar{x}_k)^2}$
- **transform**: 将聚合结果 $f(G_k)$ 广播回组内每个元素，保持原长度

### 9.3 merge 的数学本质

merge 操作等价于关系代数中的连接操作：

- **内连接**: $R \bowtie S = \{(r, s) \mid r \in R, s \in S, r.key = s.key\}$
- **左连接**: $R \ltimes S = \{(r, s) \mid r \in R, s \in S, r.key = s.key\} \cup \{(r, \text{NULL}) \mid r \in R, \nexists s \in S: r.key = s.key\}$

Pandas 内部使用 hash join 算法（对连接键建哈希表），时间复杂度约为 $O(|R| + |S|)$。

---

## 10. 代码原理 / 架构原理

### 10.1 整体架构

```
pandas/
├── core/
│   ├── frame.py          # DataFrame 实现
│   ├── series.py         # Series 实现
│   ├── indexes/          # 各种索引类型
│   ├── groupby/          # 分组聚合
│   ├── reshape/          # 数据重塑（pivot、melt）
│   └── internals/        # BlockManager（内存管理）
├── io/                   # 数据读写（csv, json, parquet...）
├── _libs/                # C/Cython 扩展（性能关键路径）
└── api/                  # 公共 API
```

### 10.2 性能关键点

1. **向量化操作**：Pandas 底层调用 NumPy 的向量化运算，避免 Python 层循环。`df['col'] * 2` 实际上是 C 层的数组运算。

2. **BlockManager**：列式存储管理器，将同类型的列组合成连续内存块（Block），提高缓存命中率。

3. **Cython 扩展**：关键路径（如 CSV 解析、分组聚合、索引查找）使用 Cython 实现，接近 C 性能。

4. **Copy-on-Write（CoW，Pandas 2.0+）**：新增的写时复制机制，减少不必要的数据拷贝，提升内存效率。

### 10.3 索引原理

Pandas 的 Index 对象基于哈希表实现快速查找：

- 索引查找 `df.loc[key]` 的时间复杂度为 $O(1)$（哈希查找）
- 位置查找 `df.iloc[pos]` 的时间复杂度为 $O(1)$（数组直接访问）
- 多级索引（MultiIndex）使用元组作为键，构建层级哈希表

---

## 11. 常见注意事项和最佳实践

### 11.1 性能优化

```python
# ❌ 避免：在循环中逐行操作
for i in range(len(df)):
    df.loc[i, 'new_col'] = process(df.loc[i, 'text'])

# ✅ 推荐：向量化操作或 apply
df['new_col'] = df['text'].apply(process)

# ✅ 更好：使用内置的字符串/数值向量化方法
df['text_len'] = df['text'].str.len()
df['score_doubled'] = df['score'] * 2
```

### 11.2 内存优化

```python
# 查看内存使用
print(df.memory_usage(deep=True))

# 使用 category 类型节省内存（低基数字符串）
df['category'] = df['category'].astype('category')

# 使用更小的数值类型
df['score'] = df['score'].astype('float32')  # float64 → float32，内存减半
df['count'] = df['count'].astype('int32')    # int64 → int32

# 分块读取大数据
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

### 11.3 SettingWithCopyWarning

```python
# ❌ 触发警告：链式赋值
df[df['score'] > 0.5]['label'] = 'good'  # 可能在副本上操作

# ✅ 推荐：使用 .loc 一次完成
df.loc[df['score'] > 0.5, 'label'] = 'good'

# ✅ 或使用 .copy() 明确创建副本
subset = df[df['score'] > 0.5].copy()
subset['label'] = 'good'
```

### 11.4 LLM 数据处理最佳实践

1. **始终检查缺失值和重复值**：低质量数据会直接影响模型训练效果
2. **使用 JSONL 格式存储对话数据**：每行一条样本，方便流式读取和处理
3. **大数据使用 Parquet 格式**：比 CSV/JSON 节省大量存储和读取时间
4. **数据分割前先打乱**：避免数据分布偏差
5. **文本长度过滤**：移除过短（可能是噪音）和过长（可能超上下文窗口）的样本
6. **使用 `value_counts()` 检查标签分布**：确保数据集类别均衡
7. **保留原始数据**：清洗时不要修改原始文件，保存为新文件

```python
# LLM 数据清洗完整示例
def clean_llm_dataset(input_path, output_path):
    """清洗 LLM 训练数据的完整流程"""
    # 读取
    df = pd.read_json(input_path, lines=True)

    # 去除关键字段缺失
    df = df.dropna(subset=['instruction', 'output'])

    # 去除重复指令
    df = df.drop_duplicates(subset=['instruction'], keep='first')

    # 计算文本长度
    df['output_len'] = df['output'].str.len()

    # 过滤异常长度
    df = df[(df['output_len'] >= 10) & (df['output_len'] <= 4096)]

    # 类型优化
    df['output_len'] = df['output_len'].astype('int32')

    # 保存
    df.drop(columns=['output_len']).to_json(
        output_path, orient='records', lines=True, force_ascii=False
    )

    print(f"清洗完成: {len(df)} 条数据，已保存至 {output_path}")
    return df
```

---

## 总结

Pandas 是 LLM 开发中数据处理的核心工具，掌握其 DataFrame/Series 操作、数据读写、清洗、分组聚合、合并连接等核心功能，能够高效完成训练数据的加载预处理、评估结果的分析统计、数据集的清洗整理等工作。在实际开发中，应注意向量化操作、内存优化、避免链式赋值等最佳实践，以充分发挥 Pandas 的性能优势。
