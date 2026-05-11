---
title: "PEFT 参数高效微调库（结合源码）"
excerpt: "LoRA/AdaLora/QLoRA数学原理、LoraLayer注入与合并、get_peft_model"
collection: llm-libs
permalink: /llm-libs/08-peft
category: training
---


## 1. 库的简介

PEFT（Parameter-Efficient Fine-Tuning）是 HuggingFace 开发的参数高效微调库，专为在资源受限环境下对大规模语言模型（LLM）进行微调而设计。在大模型时代，全参数微调一个数十亿参数的模型需要巨大的显存和计算资源，而 PEFT 通过只训练极少量的额外参数，实现了接近全参数微调的效果。

PEFT 的核心价值：

- **大幅降低训练成本**：以 LoRA 为例，仅训练不到 1% 的参数即可获得与全参数微调相当的效果
- **消除显存瓶颈**：基础模型权重可以冻结甚至量化为 4-bit，训练时只需保存优化器状态对应的少量参数
- **多适配器管理**：同一基础模型可挂载多个轻量适配器，实现不同任务间的快速切换，无需为每个任务保存一份完整模型
- **与 HuggingFace 生态无缝集成**：与 Transformers、Accelerate、bitsandbytes 等库深度兼容

PEFT 支持的微调方法包括：LoRA、AdaLora、IA³、Prefix Tuning、Prompt Tuning、P-Tuning、LoftQ、VeRA、BOFT、FourierFT、HRA、LoKr、LoHA、OFT 等数十种方法。

---

## 2. 源码架构分析

### 2.1 整体架构

```
src/peft/
├── __init__.py              # 公共API导出
├── peft_model.py            # PeftModel基类及各任务子类
├── config.py                # PeftConfig基类、PeftConfigMixin、PromptLearningConfig
├── auto.py                  # AutoPeftModel自动加载逻辑
├── mapping.py               # PEFT_TYPE_TO_*映射表（配置类、Tuner类、前缀）
├── mapping_func.py          # get_peft_model()核心函数
├── mixed_model.py           # PeftMixedModel混合适配器支持
├── helpers.py               # 辅助函数
│
├── tuners/                  # 所有微调方法的实现
│   ├── tuners_utils.py      # BaseTuner和BaseTunerLayer基类
│   ├── _buffer_dict.py      # BufferDict工具类
│   │
│   ├── lora/                # LoRA实现（最核心）
│   │   ├── __init__.py
│   │   ├── config.py        # LoraConfig及相关子配置(LoftQ, EVA, CorDA等)
│   │   ├── model.py         # LoraModel（继承BaseTuner）
│   │   ├── layer.py         # LoraLayer及各变体层(Linear, Conv2d, Embedding等)
│   │   ├── aqlm.py          # AQLM量化分发
│   │   ├── awq.py           # AWQ量化分发
│   │   ├── gptq.py          # GPTQ量化分发
│   │   ├── hqq.py           # HQQ量化分发
│   │   ├── eetq.py          # EETQ量化分发
│   │   ├── te.py            # TransformerEngine分发
│   │   ├── torchao.py       # TorchAO分发
│   │   ├── tp_layer.py      # Tensor Parallel(Megatron)分发
│   │   └── variants.py      # LoRA变体(DoRA, aLoRA)框架
│   │
│   ├── adalora/             # AdaLora自适应LoRA
│   │   ├── config.py        # AdaLoraConfig
│   │   ├── model.py         # AdaLoraModel
│   │   └── layer.py         # AdaLoraLayer（使用SVD分解）
│   │
│   ├── ia3/                 # IA³ (Infused Adapter by Inhibition)
│   │   ├── config.py        # IA3Config
│   │   ├── model.py         # IA3Model
│   │   └── layer.py         # IA3Layer
│   │
│   ├── prefix_tuning/       # Prefix Tuning
│   ├── prompt_tuning/       # Prompt Tuning
│   ├── p_tuning/            # P-Tuning
│   ├── loftq/               # LoftQ初始化
│   ├── vera/                # VeRA
│   ├── boft/                # BOFT
│   ├── fourierft/           # FourierFT
│   ├── hra/                 # HRA
│   ├── lokr/                # LoKr
│   ├── loha/                # LoHA
│   ├── oft/                 # OFT
│   └── ...                  # 其他微调方法
│
└── utils/                   # 工具函数
    ├── peft_types.py         # PeftType和TaskType枚举，register_peft_method()
    ├── save_and_load.py      # 模型保存/加载工具
    ├── other.py              # 其他辅助函数
    ├── integrations.py       # 与bitsandbytes等的集成
    ├── loftq_utils.py        # LoftQ初始化工具
    ├── merge_utils.py        # 适配器合并算法(task arithmetic, TIES等)
    └── constants.py          # 常量定义
```

**架构关系说明**：

- `PeftModel` 是对基础模型的包装器，内部通过 `PEFT_TYPE_TO_TUNER_MAPPING` 映射到具体的 `BaseTuner` 子类（如 `LoraModel`）
- `PeftConfig` 是配置基类，通过 `PEFT_TYPE_TO_CONFIG_MAPPING` 映射到具体配置类（如 `LoraConfig`）
- `BaseTuner` 负责遍历模型中匹配 `target_modules` 的模块，将其替换为对应的 `BaseTunerLayer` 子类
- 每种微调方法在 `tuners/` 目录下独立实现，通过 `register_peft_method()` 注册到全局映射表

### 2.2 PeftModel 基类

`PeftModel`（定义于 `peft_model.py:94`）继承自 `PushToHubMixin` 和 `torch.nn.Module`，是所有 PEFT 模型的统一入口。

**核心属性**：

```python
class PeftModel(PushToHubMixin, torch.nn.Module):
    def __init__(self, model, peft_config, adapter_name="default",
                 autocast_adapter_dtype=True, low_cpu_mem_usage=False):
        self.active_adapter = adapter_name          # 当前活跃适配器名
        self.peft_type = peft_config.peft_type      # PEFT类型(LoRA/AdaLora等)
        self._is_prompt_learning = peft_config.is_prompt_learning
```

**初始化逻辑的关键分支**：

```python
if self._is_prompt_learning:
    # Prompt Learning方法（Prefix Tuning, Prompt Tuning等）
    # 直接在base_model上添加adapter
    self.base_model = model
    self.add_adapter(adapter_name, peft_config)
else:
    # Tuner类方法（LoRA, IA³等）
    # 通过映射表获取具体的Tuner类，包装基础模型
    cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]
    self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
```

对于 Tuner 类方法，`PeftModel.base_model` 实际上是一个 `LoraModel`（或 `AdaLoraModel` 等），而 `LoraModel.model` 才是原始的 Transformers 模型。调用链为：`PeftModel.forward()` → `LoraModel.forward()` → `model.forward()`。

**from_pretrained 加载逻辑**（`peft_model.py:421`）：

1. 加载 `adapter_config.json` 获取 `PeftConfig`
2. 根据 `task_type` 选择对应的 `PeftModel` 子类（如 `PeftModelForCausalLM`）
3. 实例化 `PeftModel`，注入适配器层
4. 加载适配器权重（`adapter_model.safetensors`）
5. 设置适配器为推理/训练模式

### 2.3 PeftConfig 配置管理

`PeftConfig`（定义于 `config.py`）使用 Python `dataclass` 实现，继承层次为：

```
PeftConfigMixin (PushToHubMixin)
├── PeftConfig                # 所有PEFT方法的基类
│   ├── LoraConfig            # LoRA配置
│   │   └── AdaLoraConfig     # AdaLora配置(继承LoraConfig)
│   ├── IA3Config             # IA³配置
│   ├── LoftQConfig           # LoftQ子配置
│   └── ...                   # 其他Tuner类配置
└── PromptLearningConfig      # Prompt Learning方法基类
    ├── PromptTuningConfig    # Prompt Tuning配置
    ├── PrefixTuningConfig    # Prefix Tuning配置
    └── PTuningConfig         # P-Tuning配置
```

`PeftConfigMixin` 提供 `save_pretrained()` 和 `from_pretrained()` 方法用于配置的序列化和反序列化。加载时通过 `peft_type` 字段自动路由到正确的配置子类。

### 2.4 tuners/ 目录：统一接口设计

所有微调方法都遵循相同的接口规范，由 `BaseTuner`（`tuners_utils.py:234`）和 `BaseTunerLayer` 定义：

**BaseTuner** 的核心接口：

| 方法 | 说明 |
|------|------|
| `inject_adapter()` | 遍历模型模块，替换目标层为适配器层 |
| `_create_and_replace()` | 创建适配器层并替换原始层 |
| `_check_target_module_exists()` | 检查模块名是否匹配 `target_modules` |
| `merge_and_unload()` | 合并适配器权重到基础模型并卸载 |
| `unload()` | 卸载适配器（不合并） |

**BaseTunerLayer** 的核心接口：

| 方法 | 说明 |
|------|------|
| `update_layer()` | 为指定适配器创建/更新参数 |
| `merge()` | 合并适配器权重到基础层 |
| `unmerge()` | 取消合并 |
| `set_adapter()` | 切换活跃适配器 |

每种微调方法只需实现这些接口即可。例如 `LoraModel` 继承 `BaseTuner`，重写 `_create_and_replace()` 来创建 `LoraLayer`；`LoraLayer` 继承 `BaseTunerLayer`，实现 `update_layer()` 来初始化 `lora_A` 和 `lora_B`。

### 2.5 LoRA 的源码实现

#### LoraModel

`LoraModel`（`tuners/lora/model.py:88`）继承自 `BaseTuner`，是 LoRA 方法的入口：

```python
class LoraModel(BaseTuner):
    prefix: str = "lora_"                                    # 参数名前缀
    tuner_layer_cls = LoraLayer                              # 适配器层类
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING  # 默认目标模块映射
```

`_create_and_replace()` 方法（`model.py:178`）的核心流程：

1. 根据当前模块名查找 `rank_pattern` 和 `alpha_pattern`，确定该层实际使用的 `r` 和 `lora_alpha`
2. 如果目标模块已经是 `LoraLayer`（多适配器场景），调用 `update_layer()` 添加新适配器
3. 如果是首次注入，调用 `_create_new_module()` 创建新的 `LoraLayer` 实例
4. 通过 `_replace_module()` 将原始 `nn.Linear` 替换为 `LoraLayer`

**注入流程伪代码**：

```
for key, module in model.named_modules():
    if matches_target_modules(key, target_modules):
        # 创建LoraLayer，包装原始Linear
        lora_layer = LoraLayer(base_layer=module, ...)
        # 在parent中替换module为lora_layer
        setattr(parent, target_name, lora_layer)
```

#### LoraLayer

`LoraLayer`（`tuners/lora/layer.py:104`）继承自 `BaseTunerLayer`，是 LoRA 权重的容器。

**核心属性**：

```python
class LoraLayer(BaseTunerLayer):
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")

    def __init__(self, base_layer, ephemeral_gpu_offload=False):
        self.base_layer = base_layer        # 原始nn.Linear层
        self.r = {}                         # 每个adapter的秩 {adapter_name: r}
        self.lora_alpha = {}                # 每个adapter的alpha
        self.scaling = {}                   # 每个adapter的缩放因子 alpha/r
        self.lora_dropout = nn.ModuleDict({})  # Dropout层
        self.lora_A = nn.ModuleDict({})     # A矩阵: (in_features, r)
        self.lora_B = nn.ModuleDict({})     # B矩阵: (r, out_features)
        self.lora_embedding_A = nn.ParameterDict({})  # Embedding层的A
        self.lora_embedding_B = nn.ParameterDict({})  # Embedding层的B
```

**update_layer() 中的初始化逻辑**（`layer.py:157`）：

```python
def update_layer(self, adapter_name, r, lora_alpha, config):
    # 创建A和B矩阵
    self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
    self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)

    # 计算缩放因子
    if use_rslora:
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)  # Rank-Stabilized
    else:
        self.scaling[adapter_name] = lora_alpha / r              # 标准LoRA

    # 权重初始化
    if init_lora_weights is True:
        nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[adapter_name].weight)  # B初始化为0，保证ΔW=0
```

**权重合并逻辑** — `merge()` 方法（`layer.py:826`）和 `get_delta_weight()` 方法（`layer.py:916`）：

```python
def get_delta_weight(self, adapter):
    weight_A = self.lora_A[adapter].weight   # (r, in_features)
    weight_B = self.lora_B[adapter].weight   # (out_features, r)
    # ΔW = scaling * B @ A
    output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
    return output_tensor

def merge(self, safe_merge=False, adapter_names=None):
    for active_adapter in adapter_names:
        delta_weight = self.get_delta_weight(active_adapter)
        base_layer.weight.data += delta_weight   # W_merged = W_0 + (α/r) * BA
```

---

## 3. 核心类/函数详细说明

### 3.1 PeftModel.from_pretrained()

加载预训练的 PEFT 适配器并挂载到基础模型上。

```python
PeftModel.from_pretrained(
    model,                  # transformers.PreTrainedModel: 基础模型
    model_id,               # str: 适配器路径或HuggingFace Hub ID
    adapter_name="default", # str: 适配器名称
    is_trainable=False,     # bool: 是否可训练（False=推理模式）
    config=None,            # PeftConfig: 手动指定配置（可选）
    autocast_adapter_dtype=True,  # bool: 自动将fp16/bf16适配器权重转为fp32
    ephemeral_gpu_offload=False,  # bool: CPU/GPU临时卸载
    low_cpu_mem_usage=False,      # bool: 在meta device上创建空权重加速加载
)
```

**返回**：`PeftModel` 实例（或其任务子类，如 `PeftModelForCausalLM`）

### 3.2 PeftModel.merge_and_unload()

将适配器权重合并到基础模型中，返回不含 PEFT 层的原始模型。

```python
model.merge_and_unload(
    progressbar=False,      # bool: 是否显示进度条
    safe_merge=False,       # bool: 安全合并模式（检查NaN）
    adapter_names=None,     # List[str]: 要合并的适配器列表，None=所有活跃适配器
)
```

**返回**：合并后的 `torch.nn.Module`（原始模型架构）

**注意**：此方法不是原地操作，需要将返回值赋给变量使用。

### 3.3 get_peft_model()

核心函数，为模型注入 PEFT 适配器。

```python
from peft import get_peft_model

model = get_peft_model(
    model,                      # PreTrainedModel: 基础模型
    peft_config,                # PeftConfig: 适配器配置
    adapter_name="default",     # str: 适配器名称
    mixed=False,                # bool: 是否允许混合不同类型的适配器
    autocast_adapter_dtype=True,# bool: 自动类型转换
    revision=None,              # str: 基础模型的版本
    low_cpu_mem_usage=False,    # bool: 低CPU内存模式
)
```

**内部流程**：

1. 获取模型配置，更新 `peft_config.base_model_name_or_path`
2. 检查模型是否已经被 PEFT 包装过
3. 如果 `mixed=True`，返回 `PeftMixedModel`
4. 如果 `task_type` 匹配已知类型，返回对应的 `PeftModel` 子类（如 `PeftModelForCausalLM`）
5. 否则返回通用的 `PeftModel`

### 3.4 LoraConfig

LoRA 方法的配置类，是最常用的配置。

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    r=8,                        # int: LoRA秩（低秩维度），越大表达力越强但参数越多
    lora_alpha=16,              # int: 缩放因子，实际缩放 = lora_alpha / r
    target_modules=["q_proj", "v_proj"],  # 要替换的模块名列表或正则表达式
                                # 特殊值 "all-linear" 自动选择所有线性层
    lora_dropout=0.0,           # float: Dropout概率
    bias="none",                # "none"/"all"/"lora_only": 偏置处理方式
    task_type=TaskType.CAUSAL_LM,  # 任务类型
    fan_in_fan_out=False,       # bool: Conv1D权重格式(GPT-2)
    init_lora_weights=True,     # True/False/"gaussian"/"pissa"/"olora"/"loftq"/"eva"/"orthogonal"等
    use_rslora=False,           # bool: 使用Rank-Stabilized LoRA，缩放改为alpha/sqrt(r)
    use_dora=False,             # bool: 启用DoRA（权重分解方向+幅度）
    modules_to_save=None,       # List[str]: 额外需要训练和保存的模块（如分类头）
    layers_to_transform=None,   # List[int]/int: 只变换指定层索引
    layers_pattern=None,        # str: 层模式名（配合layers_to_transform使用）
    rank_pattern={},            # dict: 层名→秩的映射，覆盖默认r
    alpha_pattern={},           # dict: 层名→alpha的映射，覆盖默认lora_alpha
    exclude_modules=None,       # List[str]: 排除的模块名
    loftq_config=None,          # LoftQConfig: LoftQ量化初始化配置
    eva_config=None,            # EvaConfig: EVA数据驱动初始化配置
    corda_config=None,          # CordaConfig: CorDA初始化配置
    lora_ga_config=None,        # LoraGAConfig: LoRA-GA梯度近似初始化配置
    layer_replication=None,     # List[Tuple[int,int]]: 层复制范围（扩展模型）
    trainable_token_indices=None,  # 选择性训练特定token的embedding
    lora_bias=False,            # bool: 是否为LoRA B参数启用偏置
    target_parameters=None,     # List[str]: 直接指定参数名（用于MoE等）
    ensure_weight_tying=False,  # bool: 确保权重共享后适配器也共享
)
```

**关键参数详解**：

- **r**：LoRA 的秩，决定了低秩矩阵的维度。常用值 4~64。r=8 时，对于一个 4096×4096 的权重矩阵，LoRA 仅需 `2 * 4096 * 8 = 65,536` 个参数（原矩阵有 16,777,216 个参数）。
- **lora_alpha**：缩放超参数。实际缩放系数为 `lora_alpha / r`。当 `lora_alpha = r` 时缩放为 1；当 `lora_alpha = 2r` 时缩放为 2。实践中常设 `lora_alpha = 2 * r`。
- **target_modules**：决定哪些层被注入 LoRA。对于 LLaMA 系列模型，常用 `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`。设为 `"all-linear"` 可自动选择所有线性层。
- **init_lora_weights**：控制初始化策略，对训练效果影响显著：
  - `True`（默认）：A 用 Kaiming 初始化，B 初始化为零，保证初始时 ΔW=0
  - `"pissa"`：PiSSA 初始化，对原始权重做 SVD 分解，取前 r 个主成分初始化 A、B，收敛更快
  - `"loftq"`：LoftQ 初始化，用 LoRA 残差近似量化误差
  - `"eva"`：数据驱动初始化，利用激活值 SVD

### 3.5 AdaLoraConfig

AdaLora 在 LoRA 基础上自适应分配不同层的秩。

```python
from peft import AdaLoraConfig, TaskType

config = AdaLoraConfig(
    target_r=8,             # int: 目标平均秩（所有层的秩之和 / 层数 ≈ target_r）
    init_r=12,              # int: 初始秩（训练开始时每层的秩）
    tinit=0,                # int: 初始预热步数（不做秩调整）
    tfinal=0,               # int: 最终微调步数（不再调整秩）
    deltaT=1,               # int: 秩调整的时间间隔
    beta1=0.85,             # float: EMA平滑系数（敏感性估计）
    beta2=0.85,             # float: EMA不确定性量化系数
    orth_reg_weight=0.5,    # float: 正交正则化权重
    total_step=1000,        # int: 总训练步数（必须指定）
    rank_pattern=None,      # dict: 预设的秩分配模式

    # 继承自LoraConfig
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
```

**三阶段训练过程**：

1. **初始预热阶段**（0 ~ tinit 步）：不调整秩，预训练适配器
2. **预算分配阶段**（tinit ~ total_step-tfinal 步）：根据重要性评分逐步调整各层秩
3. **最终微调阶段**（total_step-tfinal ~ total_step 步）：固定秩分配，专注微调

**注意**：`r` 参数在 AdaLora 中被忽略，应使用 `init_r` 和 `target_r`。

### 3.6 IA3Config

IA³（Infused Adapter by Inhibition）通过元素级缩放向量实现参数高效微调。

```python
from peft import IA3Config, TaskType

config = IA3Config(
    target_modules=["q_proj", "v_proj", "down_proj"],  # 要替换的模块
    feedforward_modules=["down_proj"],  # 前馈模块（缩放作用于输入而非输出）
    fan_in_fan_out=False,       # Conv1D权重格式
    modules_to_save=None,       # 额外训练模块
    init_ia3_weights=True,      # 是否初始化IA³向量
    task_type=TaskType.CAUSAL_LM,
)
```

**关键参数**：

- **feedforward_modules**：指定哪些模块是前馈模块。对于注意力模块（如 q_proj），缩放向量作用于输出；对于前馈模块（如 down_proj），缩放向量作用于输入。这是 IA³ 的核心设计——不同类型的层使用不同的缩放方向。

### 3.7 PromptTuningConfig

```python
from peft import PromptTuningConfig, TaskType

config = PromptTuningConfig(
    num_virtual_tokens=20,                  # int: 虚拟token数量
    prompt_tuning_init=PromptTuningInit.TEXT,  # "TEXT"/"RANDOM"/"SAMPLE_VOCAB"
    prompt_tuning_init_text="Classify the following:",  # str: 初始化文本（TEXT模式）
    tokenizer_name_or_path="meta-llama/Llama-2-7b",     # str: 分词器路径（TEXT模式）
    task_type=TaskType.CAUSAL_LM,
)
```

**初始化方式**：

- `RANDOM`：随机初始化软提示
- `TEXT`：用指定文本的 embedding 初始化，通常更稳定
- `SAMPLE_VOCAB`：从词表中随机采样 token 初始化

### 3.8 PrefixTuningConfig

```python
from peft import PrefixTuningConfig, TaskType

config = PrefixTuningConfig(
    num_virtual_tokens=20,           # int: 前缀token数量
    encoder_hidden_size=768,         # int: 前缀编码器的隐藏维度
    prefix_projection=False,         # bool: 是否投影前缀embedding
    init_weights=None,               # "zero"或None: 初始化方式
    task_type=TaskType.CAUSAL_LM,
)
```

Prefix Tuning 在每层注意力机制前添加可学习的前缀向量，通过 `encoder_hidden_size` 控制前缀编码器的维度（可小于模型隐藏维度以减少参数）。

### 3.9 TaskType 枚举

```python
from peft import TaskType

class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"                # 文本分类
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"      # 序列到序列语言模型（如T5）
    CAUSAL_LM = "CAUSAL_LM"            # 因果语言模型（如LLaMA, GPT）
    TOKEN_CLS = "TOKEN_CLS"            # Token分类（如NER）
    QUESTION_ANS = "QUESTION_ANS"       # 问答
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"  # 特征提取
```

`TaskType` 决定了使用哪个 `PeftModel` 子类。例如 `CAUSAL_LM` 对应 `PeftModelForCausalLM`，它在 `forward()` 中自动处理 labels 和损失计算。

---

## 4. 数学原理

### 4.1 LoRA

LoRA（Low-Rank Adaptation）的核心思想：预训练权重矩阵的微调增量具有低秩特性，可以用两个低秩矩阵的乘积来近似。

**基本公式**：

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将权重更新参数化为：

$$W = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

**缩放因子**：

$$\Delta W = \frac{\alpha}{r} \cdot BA$$

- $\alpha$ 是超参数 `lora_alpha`，控制更新的整体幅度
- 缩放因子 $\alpha/r$ 使得调整秩 $r$ 时无需同步调整学习率

**前向传播**：

$$h = W_0 x + \frac{\alpha}{r} \cdot BAx$$

训练时，$W_0$ 被冻结（不参与梯度计算），只有 $A$ 和 $B$ 是可训练参数。

**初始化策略**：

- $A$ 使用 Kaiming 均匀初始化（`nn.init.kaiming_uniform_`）
- $B$ 初始化为零矩阵

这样在训练开始时 $\Delta W = BA = 0$，保证模型初始输出与预训练模型完全一致。

**权重合并**：

推理时可以将 LoRA 权重合并回原始权重，消除额外计算：

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} \cdot BA$$

合并后模型结构与原始模型完全相同，没有推理延迟。

**参数量对比**：

以 $d = k = 4096$，$r = 8$ 为例：

| 方法 | 参数量 | 占比 |
|------|--------|------|
| 全参数微调 | $4096 \times 4096 = 16,777,216$ | 100% |
| LoRA | $2 \times 4096 \times 8 = 65,536$ | 0.39% |

**Rank-Stabilized LoRA (rsLoRA)**：

标准 LoRA 的缩放因子 $\alpha/r$ 在增大 $r$ 时会减小，导致高秩时更新幅度不足。rsLoRA 将缩放因子改为：

$$\Delta W = \frac{\alpha}{\sqrt{r}} \cdot BA$$

这在 $r$ 较大时效果更稳定。

### 4.2 AdaLora

AdaLora（Adaptive LoRA）在 LoRA 基础上引入自适应秩分配机制。核心思想：不同层对微调的重要性不同，应分配不同的秩。

**SVD 分解形式**：

AdaLora 将增量矩阵参数化为 SVD 形式而非 BA 形式：

$$\Delta W = \frac{\alpha}{r} \cdot \sum_{i=1}^{r} s_i \cdot u_i v_i^T$$

其中 $s_i$ 是可学习的奇异值，$u_i$ 和 $v_i$ 是左/右奇异向量。

**重要性评分**：

AdaLora 使用基于 SVD 的重要性评分来决定每层的秩：

$$S_i = |s_i \cdot u_i^T E v_i|$$

其中 $E$ 是梯度与输入乘积的移动平均。评分越高的奇异值对损失函数影响越大，应被保留。

**预算约束下的秩分配**：

总参数预算为 $B$（由 `target_r` 控制），每步调整时：
1. 计算所有层所有奇异值的重要性评分
2. 按重要性排序
3. 在预算约束下保留最重要的奇异值
4. 删除不重要的奇异值（降低对应层的秩）

**正交正则化**：

为保证 SVD 分解的有效性，对 $U$ 和 $V$ 施加正交约束：

$$L_{\text{orth}} = \|U^TU - I\|_F^2 + \|V^TV - I\|_F^2$$

### 4.3 QLoRA

QLoRA 不是 PEFT 库中的独立方法，而是 4-bit 量化基础模型 + LoRA 的组合策略。核心创新在于 4-bit NormalFloat 量化。

**NF4 量化**：

NF4（4-bit NormalFloat）是一种信息论最优的量化数据类型：

1. 假设权重服从正态分布 $N(0, \sigma^2)$
2. 计算正态分布的分位数作为量化区间边界
3. 每个权重映射到最近的分位数值

NF4 的量化区间在 0 附近更密集，在两端更稀疏，恰好匹配正态分布权重的分布特性，因此量化误差最小。

**双重量化（Double Quantization）**：

4-bit 量化需要存储每组 64 个权重的缩放因子（fp32），双重量化将这些缩放因子本身再量化为 fp32 + fp8 组合：

- 第一层量化：权重 → 4-bit（每组一个 fp32 缩放因子）
- 第二层量化：fp32 缩放因子 → fp8 + fp32（每 256 组一个 fp32 缩放因子）

这进一步节省了约 0.37 bit/param 的显存。

**组合使用**：

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # 使用NF4量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
    bnb_4bit_use_double_quant=True,       # 启用双重量化
)
```

### 4.4 LoRA+

LoRA+ 是对 LoRA 的改进，核心思想是为 A 和 B 矩阵使用不同的学习率。

**问题**：标准 LoRA 中 A 和 B 使用相同的学习率，但由于 A 和 B 在计算图中的位置不同，它们的梯度尺度不一致，导致训练效率低下。

**解决方案**：

$$\eta_B = \lambda \cdot \eta_A, \quad \lambda > 1$$

其中 $\eta_A$ 和 $\eta_B$ 分别是 A 和 B 的学习率。理论上最优的 $\lambda \approx \sqrt{r}$ 或更大。这使得 B 矩阵能更快地学习到有用的方向，加速收敛。

**在 PEFT 中的使用**：LoRA+ 的不同学习率通常通过自定义优化器分组实现：

```python
# 手动实现LoRA+的学习率分组
lora_a_params = [p for n, p in model.named_parameters() if "lora_A" in n]
lora_b_params = [p for n, p in model.named_parameters() if "lora_B" in n]

optimizer = torch.optim.AdamW([
    {"params": lora_a_params, "lr": 1e-4},
    {"params": lora_b_params, "lr": 1e-3},  # B的学习率是A的10倍
])
```

---

## 5. 代码原理

### 5.1 get_peft_model 的流程

```
get_peft_model(model, peft_config, adapter_name)
│
├── 1. 获取模型配置，更新 peft_config.base_model_name_or_path
│
├── 2. 检查模型是否已被PEFT包装（检查是否有BaseTunerLayer模块）
│
├── 3. 根据task_type选择PeftModel子类
│   ├── CAUSAL_LM → PeftModelForCausalLM
│   ├── SEQ_2_SEQ_LM → PeftModelForSeq2SeqLM
│   ├── SEQ_CLS → PeftModelForSequenceClassification
│   └── ... 或通用 PeftModel
│
├── 4. 实例化PeftModel
│   ├── PeftModel.__init__()
│   │   ├── 判断是否为prompt learning方法
│   │   ├── 非prompt learning: 创建BaseTuner子类实例（如LoraModel）
│   │   │   └── LoraModel.__init__(model, peft_config, adapter_name)
│   │   │       └── BaseTuner.__init__()
│   │   │           └── inject_adapter()  # 注入适配器
│   │   └── prompt learning: 直接添加adapter
│   └── 自动类型转换（autocast_adapter_dtype）
│
└── 5. 返回PeftModel实例
```

### 5.2 LoRA 注入流程

```
LoraModel.inject_adapter(model, adapter_name)
│
├── 1. 准备阶段
│   ├── _prepare_adapter_config(): 如果未指定target_modules，自动推断
│   ├── _prepare_model(): 准备模型结构（如layer_replication）
│   └── _check_new_adapter_config(): 检查配置合法性
│
├── 2. 遍历模型模块
│   for key, module in model.named_modules():
│   │
│   ├── _check_target_module_exists(): 检查是否匹配target_modules
│   ├── _check_target_module_compatiblity(): 检查兼容性（如Mamba）
│   │
│   └── 如果匹配：
│       ├── _create_and_replace()
│       │   ├── 如果已有LoraLayer → update_layer() 添加新adapter
│       │   ├── 否则 → _create_new_module() 创建新的LoraLayer
│       │   │   ├── 根据层类型选择具体的LoraLayer子类
│       │   │   │   ├── nn.Linear → Linear LoraLayer
│       │   │   │   ├── nn.Conv2d → Conv2d LoraLayer
│       │   │   │   ├── nn.Embedding → Embedding LoraLayer
│       │   │   │   └── Conv1D → Conv1D LoraLayer
│       │   │   └── 初始化 lora_A 和 lora_B
│       │   └── _replace_module(): 替换原始模块
│       │       parent.target_name = new_module
│       │
│       └── 记录 targeted_module_names
│
└── 3. 设置可训练参数
    └── _set_trainable(): 冻结非LoRA参数
```

### 5.3 merge_and_unload 流程

```
BaseTuner.merge_and_unload(progressbar, safe_merge, adapter_names)
│
└── _unload_and_optionally_merge(merge=True)
    │
    ├── 1. 检查合并是否允许（_check_merge_allowed）
    │
    ├── 2. 遍历模型的所有非PEFT模块
    │   for key in model.named_modules():
    │       │
    │       ├── 如果模块有 unload_and_optionally_merge_module() → 调用
    │       │
    │       └── 如果模块有 base_layer 属性（即LoraLayer）：
    │           ├── target.merge(safe_merge, adapter_names)
    │           │   ├── get_delta_weight(adapter)
    │           │   │   └── delta = scaling * B @ A    # (α/r) * BA
    │           │   ├── base_layer.weight.data += delta  # W += (α/r) * BA
    │           │   └── 标记为已合并
    │           │
    │           └── _replace_module(): 替换回原始Linear层
    │               parent.target_name = target.get_base_layer()
    │
    ├── 3. 清理模型上的peft_config属性
    │
    └── 4. 返回原始模型（已合并权重）
```

---

## 6. 在LLM开发中的典型使用场景和代码示例

### 6.1 LoRA 微调 LLM

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 加载基础模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置LoRA
lora_config = LoraConfig(
    r=8,                                    # LoRA秩
    lora_alpha=16,                          # 缩放因子
    target_modules=["q_proj", "k_proj",     # 目标模块
                    "v_proj", "o_proj",
                    "gate_proj", "up_proj",
                    "down_proj"],
    lora_dropout=0.05,                      # Dropout
    bias="none",                            # 不训练偏置
    task_type=TaskType.CAUSAL_LM,           # 因果语言模型
)

# 注入LoRA适配器
model = get_peft_model(model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
# 输出示例: trainable params: 13,107,200 || all params: 6,738,415,616 || trainable%: 0.1945

# 训练循环（简化示例）
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 保存适配器
model.save_pretrained("./lora-adapter")
tokenizer.save_pretrained("./lora-adapter")
```

### 6.2 QLoRA（4-bit 量化 + LoRA）

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 配置4-bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit加载
    bnb_4bit_quant_type="nf4",              # NF4量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度
    bnb_4bit_use_double_quant=True,         # 双重量化
)

# 加载4-bit量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# 准备量化模型训练
model = prepare_model_for_kbit_training(model)

# 配置LoRA
lora_config = LoraConfig(
    r=16,                                   # QLoRA通常使用较大的r
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 注入LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 显存占用大幅降低: 7B模型全参数约28GB(fp16), QLoRA约5-6GB
```

**prepare_model_for_kbit_training 的作用**：

1. 冻结基础模型所有参数
2. 将 CastOutputToFloat 模块添加到模型输出，确保 loss 计算在 fp32 下进行
3. 启用梯度检查点以节省显存

### 6.3 多 LoRA 适配器管理

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 添加第一个适配器（摘要任务）
summary_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, summary_config, adapter_name="summary")

# 添加第二个适配器（翻译任务）
translation_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model.add_adapter("translation", translation_config)

# 添加第三个适配器（从预训练加载）
model.load_adapter("./lora-qa-adapter", adapter_name="qa")

# 切换适配器
model.set_adapter("summary")   # 使用摘要适配器
model.set_adapter("translation")  # 切换到翻译适配器

# 禁用适配器（退回基础模型）
with model.disable_adapter():
    outputs = model.generate(**inputs)  # 纯基础模型推理

# 删除适配器
model.delete_adapter("qa")

# 查看所有适配器
print(model.peft_config.keys())  # dict_keys(['summary', 'translation'])
```

### 6.4 合并 LoRA 权重

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# 合并权重（安全模式，检查NaN）
merged_model = model.merge_and_unload(safe_merge=True)

# 保存合并后的完整模型
merged_model.save_pretrained("./merged-model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained("./merged-model")

# 合并后的模型与原始架构完全相同，可直接用于推理
# 无需安装PEFT，也不存在推理延迟
```

**多适配器顺序合并**：

```python
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("./base-model", ...)

# 依次加载并合并多个适配器
model = PeftModel.from_pretrained(base_model, "./adapter-1", adapter_name="adapter_1")
model.load_adapter("./adapter-2", adapter_name="adapter_2")

# 合并所有活跃适配器
merged_model = model.merge_and_unload()

# 或只合并指定适配器
merged_model = model.merge_and_unload(adapter_names=["adapter_1"])
```

### 6.5 AdaLora 自适应微调

```python
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import AdaLoraConfig, get_peft_model, TaskType

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 配置AdaLora
adalora_config = AdaLoraConfig(
    target_r=8,                 # 目标平均秩
    init_r=12,                  # 初始秩
    tinit=100,                  # 预热100步
    tfinal=200,                 # 最后200步固定
    deltaT=10,                  # 每10步调整一次秩
    beta1=0.85,                 # EMA平滑系数
    beta2=0.85,                 # EMA不确定性系数
    orth_reg_weight=0.5,        # 正交正则化权重
    total_step=1000,            # 总训练步数
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, adalora_config)
model.print_trainable_parameters()

# 在训练回调中更新预算分配器
class AdaLoraCallback:
    def __init__(self, model, total_step):
        self.model = model
        self.total_step = total_step

    def on_step_end(self, args, state, control, **kwargs):
        # 更新AdaLora的预算分配器
        if hasattr(self.model, "update_and_allocate"):
            self.model.update_and_allocate(state.global_step)

# 使用自定义Trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./adalora-output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
    ),
    train_dataset=train_dataset,
    callbacks=[AdaLoraCallback(model, total_step=1000)],
)
trainer.train()

# 保存
model.save_pretrained("./adalora-adapter")
```

### 6.6 加载已保存的 PEFT 模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 方式1: 先加载基础模型，再加载适配器
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# 方式2: 加载为可训练模型
model = PeftModel.from_pretrained(
    base_model,
    "./lora-adapter",
    is_trainable=True,  # 启用训练模式
)

# 方式3: 使用AutoPeftModel自动加载
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained(
    "./lora-adapter",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 方式4: 加载量化基础模型+适配器
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
```

---

## 7. 常见注意事项和最佳实践

### 7.1 参数选择

- **r 的选择**：从小值开始尝试（r=4 或 r=8），逐渐增大。r=8 对于大多数任务已足够；复杂任务或数据量大时可尝试 r=16 或 r=32。r>64 通常不再带来显著提升。
- **lora_alpha**：常见做法设为 `2 * r`（如 r=8, alpha=16）。增大 alpha 等效于增大 LoRA 更新的幅度。
- **target_modules**：对于 LLaMA 系列模型，建议同时包含注意力层（q/k/v/o_proj）和 MLP 层（gate/up/down_proj）。仅用 q_proj 和 v_proj 参数量最少但效果可能不够。
- **lora_dropout**：小数据集（<10k）建议 0.05~0.1；大数据集可设为 0。

### 7.2 量化训练注意事项

- 使用 QLoRA 时，`bnb_4bit_compute_dtype` 建议设为 `torch.bfloat16`（Ampere及以上GPU）或 `torch.float16`
- 调用 `prepare_model_for_kbit_training()` 可确保量化模型训练的数值稳定性
- 量化模型不支持 `merge_and_unload()`，需要先反量化再合并

### 7.3 多适配器管理

- 使用 `model.add_adapter()` 添加新适配器时，新适配器默认不可训练，需调用 `model.set_adapter("new_name")` 激活
- 不同适配器可以有不同的 `target_modules`，但同一模块上挂载的所有适配器必须共享相同的层结构
- `PeftMixedModel` 支持混合不同类型的适配器（如同时使用 LoRA 和 IA³），但需要 `mixed=True` 参数

### 7.4 权重合并注意

- `merge_and_unload()` 不是原地操作，必须使用返回值：`model = model.merge_and_unload()`
- 使用 `safe_merge=True` 可检测合并后是否产生 NaN，推荐在关键场景中使用
- 合并后模型体积等于完整模型大小，需确保有足够存储空间
- 如果基础模型使用了量化，需要先反量化再合并

### 7.5 训练技巧

- **学习率**：LoRA 通常使用 1e-4 ~ 3e-4 的学习率，比全参数微调（1e-5 ~ 5e-5）高约一个数量级
- **Epoch 数**：LoRA 通常需要更多 epoch（3~10），因为每步更新的参数量较少
- **梯度检查点**：与 `model.gradient_checkpointing_enable()` 配合使用可进一步节省显存
- **rank_pattern 和 alpha_pattern**：可以为不同层设置不同的秩和 alpha 值，例如对注意力层使用较高秩

### 7.6 保存与加载

- `save_pretrained()` 默认使用 safetensors 格式，推荐保持此设置
- 保存时仅保存适配器权重（几百MB），不保存基础模型（数十GB）
- 加载时需要先加载基础模型，再用 `PeftModel.from_pretrained()` 加载适配器
- `AutoPeftModel` 可自动推断模型类型并加载，但需要 `adapter_config.json` 中包含 `base_model_name_or_path`

### 7.7 常见错误排查

- **`ValueError: Please specify target_modules`**：模型架构不在自动推断映射表中，需手动指定 `target_modules`
- **训练时显存不降**：检查是否正确冻结了基础模型参数，可用 `model.print_trainable_parameters()` 确认
- **合并后效果变差**：检查是否在推理模式下合并（`model.eval()`），以及是否所有适配器都已合并
- **多适配器冲突**：不同适配器的 `target_modules` 最好保持一致，否则可能导致未预期的行为
