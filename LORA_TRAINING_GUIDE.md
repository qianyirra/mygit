# GTA1 LoRA 训练完整指南

## 📋 概述

本指南将帮助您使用LoRA (Low-Rank Adaptation) 技术对GTA1项目进行训练。LoRA是一种高效的参数微调方法，可以在不显著增加模型大小的情况下获得良好的性能。

## 🔧 环境要求

### 1. Python环境
- Python >= 3.10
- CUDA >= 11.8 (推荐CUDA 12.1)

### 2. 必需的依赖包

```bash
# 安装基础深度学习框架
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装Transformers和相关库
pip install transformers>=4.40.0
pip install datasets
pip install accelerate
pip install peft
pip install trl
pip install deepspeed

# 安装视觉处理相关库
pip install qwen-vl-utils
pip install pillow
pip install opencv-python

# 安装数学验证库
pip install math-verify

# 安装优化内核 (可选，用于加速)
pip install liger-kernel
```

### 3. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python -m venv venv_gta1

# 激活虚拟环境 (Windows)
.\venv_gta1\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
source venv_gta1/bin/activate

# 安装依赖
pip install -r requirements_lora.txt
```

## 📁 项目结构说明

```
GTA1/
├── src/
│   ├── grpo_grounding.py        # 主训练脚本
│   └── trainer/
│       ├── grpo_trainer.py      # GRPO训练器
│       └── grpo_config.py      # 训练配置
├── configs/
│   └── lora_config.yaml        # LoRA训练配置
├── preprocessing/
│   ├── inp.json               # 训练数据集
│   └── images/               # 训练图像
└── output/
    └── gta1_lora/           # 训练输出目录
```

## 🚀 快速开始

### 方法1: 使用启动脚本（推荐）

在Windows系统上运行：

```batch
# 激活虚拟环境
.\venv_gta1\Scripts\activate

# 运行LoRA训练
python train_lora.bat
```

### 方法2: 直接使用Python脚本

```bash
# 进入GTA1目录
cd GTA1

# 运行训练（使用配置文件）
python src/grpo_grounding.py --output_dir ./output/gta1_lora --config configs/lora_config.yaml

# 或者直接指定参数
python src/grpo_grounding.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name preprocessing/inp.json \
    --image_root ./preprocessing \
    --output_dir ./output/gta1_lora \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --num_generations 8 \
    --reward_funcs accuracy format \
    --freeze_vision_modules true \
    --bf16 \
    --logging_steps 5
```

## ⚙️ LoRA配置详解

### 核心参数

| 参数 | 推荐值 | 说明 |
|------|----------|------|
| `peft_type` | lora | PEFT类型，使用LoRA |
| `r` | 16 | LoRA的秩（低维矩阵的维度） |
| `lora_alpha` | 32 | LoRA的缩放因子，通常设置为r的2倍 |
| `lora_dropout` | 0.05 | LoRA层的dropout率 |
| `target_modules` | 见配置 | 需要应用LoRA的模块列表 |

### 训练参数

| 参数 | 推荐值 | 说明 |
|------|----------|------|
| `learning_rate` | 1e-5 | LoRA训练的学习率 |
| `num_train_epochs` | 3 | 训练轮数 |
| `per_device_train_batch_size` | 1 | 每个GPU的批处理大小 |
| `gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `max_prompt_length` | 1024 | 提示词的最大长度 |
| `max_completion_length` | 128 | 生成文本的最大长度 |
| `num_generations` | 8 | GRPO的生成候选数 |
| `beta` | 0.04 | KL散度惩罚系数 |

## 📊 训练监控

训练过程中，您可以通过以下方式监控训练进度：

1. **日志输出**: 终端会实时显示训练进度和损失
2. **Checkpoint保存**: 训练会定期保存检查点到`output_dir/checkpoint-*`目录
3. **TensorBoard/WandB**: 配置`report_to`参数可启用可视化监控

```bash
# 启动TensorBoard（如果配置了tensorboard）
tensorboard --logdir ./output/gta1_lora

# 启动WandB（如果配置了wandb）
pip install wandb
wandb login
```

## 🔧 GPU内存优化

如果遇到显存不足，可以尝试以下优化：

### 1. 调整批处理大小
```yaml
per_device_train_batch_size: 1  # 减小批处理大小
gradient_accumulation_steps: 16  # 增加梯度累积步数
```

### 2. 使用DeepSpeed ZeRO
```bash
python src/grpo_grounding.py \
    --deepspeed local_scripts/zero3.json \
    --config configs/lora_config.yaml
```

### 3. 降低精度
```yaml
torch_dtype: float16  # 或 float32（非推荐）
bf16: false
```

### 4. 减少生成候选数
```yaml
num_generations: 4  # 减少GRPO生成候选数（默认8）
```

## 📝 自定义数据集

准备您的训练数据集，格式如下：

```json
[
    {
        "image": "images/1.png",
        "bbox": [38, 166, 961, 218],
        "conversations": [
            {
                "from": "human",
                "value": "Click on the search bar"
            },
            {
                "from": "gpt",
                "value": "any text here"
            }
        ]
    }
]
```

**注意事项**:
- `bbox` 格式为 `[x0, y0, x1, y1]`，坐标范围 `[0, 1000]`
- `image` 是相对于 `image_root` 的相对路径
- `conversations` 格式遵循标准对话格式

## 🎯 模型推理

训练完成后，可以使用以下代码进行推理：

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(model, "./output/gta1_lora")

# 加载处理器
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# 进行推理
import torch
from PIL import Image

image = Image.open("test.png")
inputs = processor(
    text="Describe this UI element",
    images=image,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)
    result = processor.decode(outputs[0], skip_special_tokens=True)
    
print(result)
```

## 📦 模型导出

训练完成后，可以导出完整的模型：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 合并LoRA权重到基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
lora_model = PeftModel.from_pretrained(base_model, "./output/gta1_lora")

# 合并并保存
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./output/gta1_lora_merged")

# 同时保存tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer.save_pretrained("./output/gta1_lora_merged")
```

## ⚠️ 常见问题

### Q1: CUDA out of memory错误
**解决方法**: 
- 减小`per_device_train_batch_size`
- 增加`gradient_accumulation_steps`
- 使用DeepSpeed ZeRO-3
- 启用`gradient_checkpointing: true`

### Q2: 训练非常慢
**解决方法**:
- 检查CUDA是否正确安装: `torch.cuda.is_available()`
- 使用`torch_dtype: bfloat16`降低精度
- 增加`per_device_train_batch_size`（如显存允许）
- 安装`liger-kernel`优化库

### Q3: 模型不收敛
**解决方法**:
- 调整学习率，尝试`1e-6`到`1e-4`范围
- 增加训练轮数
- 检查数据质量
- 调整`lora_alpha`和`r`的比例

### Q4: 训练中断后如何恢复
**解决方法**: 
训练会自动保存checkpoint，使用以下命令恢复：

```bash
python src/grpo_grounding.py \
    --output_dir ./output/gta1_lora \
    --resume_from_checkpoint true
```

## 📚 参考资源

- [GTA1论文](https://arxiv.org/pdf/2507.05791)
- [Qwen2.5-VL模型文档](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [PEFT文档](https://huggingface.co/docs/peft)
- [TRL文档](https://huggingface.co/docs/trl)
- [LoRA论文](https://arxiv.org/abs/2106.09685)

## 💡 技巧建议

1. **从较小的模型开始**: 如果是新任务，先用0.5B或3B模型验证流程
2. **数据质量优先**: 高质量的少量数据比大量低质量数据更有效
3. **监控过拟合**: 观察训练/验证loss，如果差异过大，需要调整
4. **实验不同参数**: LoRA的`r`和`lora_alpha`可以显著影响性能
5. **保存多个checkpoint**: 定期保存，便于回溯和选择最佳模型

## 📄 许可证

本代码遵循Apache 2.0许可证。
