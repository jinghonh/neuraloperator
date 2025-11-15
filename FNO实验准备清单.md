# FNO (Fourier Neural Operator) 实验准备清单

本文档提供了使用 FNO 进行实验所需的完整准备清单，涵盖从环境配置到实验执行的各个环节。

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [硬件要求](#2-硬件要求)
3. [项目安装](#3-项目安装)
4. [数据准备](#4-数据准备)
5. [模型配置](#5-模型配置)
6. [训练设置](#6-训练设置)
7. [评估与可视化](#7-评估与可视化)
8. [实验跟踪](#8-实验跟踪)
9. [常见问题排查](#9-常见问题排查)
10. [快速开始示例](#10-快速开始示例)

---

## 1. 环境准备

### 1.1 Python 环境

- **Python 版本**: Python 3.8 或更高版本（推荐 3.9+）
- **包管理器**: pip 或 conda

### 1.2 虚拟环境（推荐）

```bash
# 使用 venv 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows

# 或使用 conda
conda create -n neuralop python=3.9
conda activate neuralop
```

### 1.3 核心依赖包

根据 `requirements.txt`，需要安装以下核心依赖：

```bash
# 核心依赖
pip install torch torchvision torchaudio  # PyTorch (根据你的CUDA版本选择)
pip install wandb                        # 实验跟踪
pip install ruamel.yaml                  # YAML 配置处理
pip install configmypy                   # 配置类型检查
pip install zencfg                       # 配置管理
pip install tensorly                     # 张量分解
pip install tensorly-torch               # TensorLy PyTorch 后端
pip install torch-harmonics              # 球谐函数
pip install matplotlib                   # 可视化
pip install opt-einsum                  # 优化的 einsum
pip install h5py                         # HDF5 文件支持
pip install zarr                         # Zarr 数组存储
```

### 1.4 安装 NeuralOperator 包

```bash
# 从项目根目录安装（开发模式）
cd /path/to/neuraloperator
pip install -e .[dev]

# 或仅安装基础包
pip install -e .
```

---

## 2. 硬件要求

### 2.1 GPU（推荐）

- **CUDA**: 支持 CUDA 的 GPU（推荐 NVIDIA GPU）
- **显存**: 
  - 小型实验（16x16 分辨率）: 至少 4GB
  - 中型实验（32x32 分辨率）: 至少 8GB
  - 大型实验（64x64+ 分辨率）: 至少 16GB 或更多
- **CUDA 版本**: 根据 PyTorch 版本选择（通常 CUDA 11.8 或 12.1）

### 2.2 CPU（可选）

- 可以在 CPU 上运行小型实验，但训练速度会显著降低
- 推荐至少 8GB RAM

### 2.3 存储空间

- **数据集**: 根据数据集大小，预留 1-10GB 空间
- **模型检查点**: 每个检查点约 10-100MB（取决于模型大小）
- **日志和可视化**: 预留 1-5GB

---

## 3. 项目安装

### 3.1 克隆或确认项目结构

确保项目目录结构正确：

```
neuraloperator/
├── neuralop/          # 主包
│   ├── models/        # 模型定义（包含 fno.py）
│   ├── data/          # 数据集和数据处理
│   ├── training/      # 训练工具
│   └── layers/        # 层定义
├── config/            # 配置文件
├── scripts/           # 训练脚本
├── examples/          # 示例代码
└── requirements.txt   # 依赖列表
```

### 3.2 验证安装

```bash
# 运行测试（可选）
pytest neuralop/models/tests/test_fno.py -v

# 或运行完整测试套件
pytest neuralop -v
```

---

## 4. 数据准备

### 4.1 使用内置数据集（Darcy Flow）

Darcy Flow 是最常用的测试数据集，可以自动下载：

```python
from neuralop.data.datasets import load_darcy_flow_small

# 数据会自动下载到指定目录
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    data_root="~/data/darcy/",  # 数据存储路径
    n_train=1000,               # 训练样本数
    batch_size=32,              # 批次大小
    n_tests=[100, 50],          # 每个测试分辨率的样本数
    test_resolutions=[16, 32],  # 测试分辨率
    test_batch_sizes=[32, 32],  # 测试批次大小
    download=True               # 自动下载
)
```

### 4.2 准备自定义数据

如果使用自定义数据，需要：

1. **数据格式**: 
   - 输入和输出都应该是 PyTorch 张量
   - 形状: `[batch, channels, height, width]` (2D) 或 `[batch, channels, depth, height, width]` (3D)

2. **数据加载器**:
   ```python
   from torch.utils.data import DataLoader, Dataset
   
   class YourDataset(Dataset):
       def __init__(self, ...):
           # 初始化
           pass
       
       def __getitem__(self, idx):
           return {"x": input_tensor, "y": output_tensor}
   
   train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
   ```

3. **数据预处理**:
   - 使用 `neuralop.data.transforms.DataProcessor` 进行归一化
   - 或手动实现预处理管道

### 4.3 数据路径配置

在配置文件中设置数据路径：

```python
# config/darcy_config.py
class DarcyDatasetConfig(ConfigBase):
    folder: str = "~/data/darcy/"  # 修改为你的数据路径
    batch_size: int = 8
    n_train: int = 1000
    train_resolution: int = 16
    # ...
```

---

## 5. 模型配置

### 5.1 基本 FNO 配置

FNO 的关键参数：

```python
from neuralop.models import FNO

model = FNO(
    n_modes=(16, 16),          # Fourier 模式数（每个维度）
    in_channels=1,              # 输入通道数
    out_channels=1,             # 输出通道数
    hidden_channels=64,          # 隐藏层通道数（模型宽度）
    n_layers=4,                 # FNO 层数
    lifting_channel_ratio=2,     # Lifting 层通道比例
    projection_channel_ratio=2, # Projection 层通道比例
)
```

### 5.2 预定义配置

项目提供了多个预定义配置（在 `config/models.py` 中）：

- **FNO_Small2d**: 小型 2D FNO
  - `n_modes=[16, 16]`, `hidden_channels=24`
- **FNO_Medium2d**: 中型 2D FNO
  - `n_modes=[64, 64]`, `hidden_channels=64`
- **FNO_Large2d**: 大型 2D FNO
  - `n_modes=[64, 64]`, `hidden_channels=128`
- **FNO_Medium3d**: 中型 3D FNO
  - `n_modes=[32, 32, 32]`, `hidden_channels=64`

### 5.3 高级配置选项

```python
model = FNO(
    # ... 基本参数 ...
    
    # 归一化
    norm="group_norm",  # 或 "instance_norm", "ada_in", None
    
    # 跳过连接
    fno_skip="linear",  # 或 "identity", "soft-gating", None
    channel_mlp_skip="soft-gating",
    
    # 域填充（用于处理边界）
    domain_padding=0.1,  # 填充百分比
    
    # 精度控制
    fno_block_precision="full",  # 或 "half", "mixed"
    stabilizer="tanh",  # 用于混合精度训练
    
    # 张量分解（减少参数量）
    factorization="Tucker",  # 或 "CP", "TT", None
    rank=0.1,  # 分解秩（0.1 表示约 10% 的参数量）
    
    # 位置编码
    positional_embedding="grid",  # 或 None, GridEmbeddingND
)
```

### 5.4 使用配置文件

```python
# 使用 zencfg 配置系统
from config.darcy_config import Default
from neuralop import get_model

config = Default()
config.model = FNO_Small2d()  # 或自定义配置
model = get_model(config)
```

---

## 6. 训练设置

### 6.1 优化器配置

```python
from neuralop.training import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=5e-3,              # 学习率（通常 1e-3 到 1e-2）
    weight_decay=1e-4      # 权重衰减（L2 正则化）
)
```

### 6.2 学习率调度器

```python
import torch.optim.lr_scheduler as lr_scheduler

# StepLR: 每 N 个 epoch 降低学习率
scheduler = lr_scheduler.StepLR(
    optimizer,
    step_size=60,          # 每 60 个 epoch
    gamma=0.5             # 学习率乘以 0.5
)

# CosineAnnealingLR: 余弦退火
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100             # 周期长度
)

# ReduceLROnPlateau: 基于验证损失
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)
```

### 6.3 损失函数

```python
from neuralop import LpLoss, H1Loss

# L2 损失（仅函数值）
l2loss = LpLoss(d=2, p=2)  # d: 维度, p: Lp 范数的 p

# H1 损失（函数值 + 梯度）
h1loss = H1Loss(d=2)

# 训练时通常使用 H1Loss（对 PDE 问题更合适）
train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}
```

### 6.4 训练器配置

```python
from neuralop import Trainer

trainer = Trainer(
    model=model,
    n_epochs=300,                    # 训练轮数
    device=device,                   # CPU 或 GPU
    data_processor=data_processor,   # 数据预处理器
    mixed_precision=False,           # 混合精度训练
    wandb_log=False,                 # 是否使用 WandB
    eval_interval=5,                 # 每 N 个 epoch 评估一次
    log_output=False,                # 是否记录输出
    use_distributed=False,           # 分布式训练
    verbose=True,                    # 详细输出
)
```

### 6.5 开始训练

```python
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,       # 字典: {resolution: DataLoader}
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,               # 是否使用正则化
    training_loss=train_loss,
    eval_losses=eval_losses,
)
```

### 6.6 使用训练脚本

项目提供了现成的训练脚本：

```bash
# 训练 Darcy Flow
python scripts/train_darcy.py --config config/darcy_config.py

# 可以覆盖配置参数
python scripts/train_darcy.py \
    --config config/darcy_config.py \
    --opt.n_epochs 500 \
    --opt.learning_rate 1e-3 \
    --data.batch_size 16
```

---

## 7. 评估与可视化

### 7.1 模型评估

```python
model.eval()
with torch.no_grad():
    for data in test_loader:
        x = data['x'].to(device)
        y = data['y'].to(device)
        out = model(x)
        error = l2loss(out, y).item()
        # 处理误差...
```

### 7.2 可视化预测结果

```python
import matplotlib.pyplot as plt
import numpy as np

# 获取一个测试样本
data = test_samples[0]
x = data['x'].to(device)
y = data['y'].to(device)

# 预测
with torch.no_grad():
    out = model(x.unsqueeze(0))

# 转换为 numpy
x_np = x[0].cpu().numpy()
y_np = y.squeeze().cpu().numpy()
out_np = out.squeeze().cpu().numpy()

# 绘制
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(x_np, cmap='viridis')
axes[0].set_title('输入')
axes[1].imshow(y_np, cmap='coolwarm')
axes[1].set_title('真实输出')
axes[2].imshow(out_np, cmap='coolwarm')
axes[2].set_title('模型预测')
plt.show()
```

### 7.3 零样本超分辨率

FNO 的一个优势是可以直接在不同分辨率上推理：

```python
# 在 16x16 上训练
train_loader, _, _ = load_darcy_flow_small(
    train_resolution=16, ...
)

# 在 32x32 上测试（无需重新训练！）
test_loader_32, _, _ = load_darcy_flow_small(
    test_resolutions=[32], ...
)

# 直接使用训练好的模型
model.eval()
with torch.no_grad():
    for data in test_loader_32:
        out = model(data['x'].to(device))
        # 评估...
```

---

## 8. 实验跟踪

### 8.1 Weights & Biases (WandB) 设置

1. **获取 API Key**:
   ```bash
   # 登录 WandB
   wandb login
   # 或设置环境变量
   export WANDB_API_KEY="your_api_key"
   ```

2. **在代码中启用**:
   ```python
   import wandb
   
   wandb.init(
       project="fno-experiments",
       name="darcy-fno-small",
       config=config_dict
   )
   
   # 在训练器中启用
   trainer = Trainer(
       ...,
       wandb_log=True,
   )
   ```

3. **配置文件设置**:
   ```python
   # config/wandb.py
   class WandbConfig(ConfigBase):
       log: bool = True
       project: str = "fno-experiments"
       entity: str = "your-entity"
       name: str = None  # 自动生成
   ```

### 8.2 保存和加载检查点

```python
# 保存
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}, 'checkpoint.pth')

# 加载
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 9. 常见问题排查

### 9.1 CUDA 相关错误

**问题**: `CUDA out of memory`

**解决方案**:
- 减小 `batch_size`
- 减小 `hidden_channels` 或 `n_modes`
- 使用混合精度训练: `mixed_precision=True`
- 使用张量分解: `factorization="Tucker", rank=0.1`
- 使用梯度累积

**问题**: `CUDA not available`

**解决方案**:
- 检查 PyTorch 是否正确安装 CUDA 版本
- 验证 GPU 驱动和 CUDA 版本兼容性
- 如果只有 CPU，设置 `device=torch.device("cpu")`

### 9.2 数据加载错误

**问题**: 数据下载失败

**解决方案**:
- 检查网络连接
- 手动下载数据到指定目录
- 检查数据路径权限

**问题**: 数据形状不匹配

**解决方案**:
- 确认输入/输出通道数匹配 `in_channels` 和 `out_channels`
- 检查数据维度顺序: `[batch, channels, height, width]`

### 9.3 训练不收敛

**问题**: 损失不下降

**解决方案**:
- 调整学习率（尝试 1e-4 到 1e-2）
- 检查数据归一化是否正确
- 增加模型容量（`hidden_channels`, `n_layers`）
- 使用不同的损失函数（尝试 H1Loss）
- 检查数据质量

**问题**: 训练过慢

**解决方案**:
- 使用 GPU 而非 CPU
- 减小 `batch_size` 如果受内存限制
- 使用混合精度训练
- 减少 `n_modes` 或 `hidden_channels`

### 9.4 配置错误

**问题**: `n_modes` 太大

**解决方案**:
- `n_modes` 必须小于 `max_resolution // 2`（Nyquist 频率）
- 对于 16x16 分辨率，`n_modes` 应 ≤ 8
- 对于 32x32 分辨率，`n_modes` 应 ≤ 16

---

## 10. 快速开始示例

### 10.1 最小示例

```python
import torch
from neuralop.models import FNO
from neuralop import Trainer, H1Loss, LpLoss
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small

# 1. 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载数据
train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=32,
    n_tests=[100, 50],
    test_resolutions=[16, 32],
    test_batch_sizes=[32, 32],
)
data_processor = data_processor.to(device)

# 3. 创建模型
model = FNO(
    n_modes=(8, 8),
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_layers=4,
).to(device)

# 4. 设置训练组件
optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
train_loss = H1Loss(d=2)
eval_losses = {"h1": train_loss, "l2": LpLoss(d=2, p=2)}

# 5. 训练
trainer = Trainer(
    model=model,
    n_epochs=20,
    device=device,
    data_processor=data_processor,
    wandb_log=False,
    verbose=True,
)
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    training_loss=train_loss,
    eval_losses=eval_losses,
)
```

### 10.2 使用配置文件

```python
from zencfg import make_config_from_cli
from config.darcy_config import Default
from neuralop import get_model, Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small

# 加载配置
config = make_config_from_cli(Default)
config = config.to_dict()

# 加载数据和模型
train_loader, test_loaders, data_processor = load_darcy_flow_small(...)
model = get_model(config)

# 设置训练器并开始训练
# ... (参考 scripts/train_darcy.py)
```

### 10.3 运行完整示例

项目根目录提供了完整示例：

```bash
# 运行完整示例（包含可视化）
python complete_example.py

# 或运行简单示例
python simple_complete_example.py
```

---

## 📚 参考资源

### 文档
- 项目 README: `README.rst`
- API 文档: `doc/source/`
- 示例代码: `examples/`

### 配置文件
- 模型配置: `config/models.py`
- Darcy 配置: `config/darcy_config.py`
- 优化配置: `config/opt.py`

### 训练脚本
- Darcy Flow: `scripts/train_darcy.py`
- 其他 PDE: `scripts/train_*.py`

### 示例
- FNO Darcy: `examples/models/plot_FNO_darcy.py`
- 完整示例: `complete_example.py`

---

## ✅ 检查清单

在开始实验前，确认以下项目：

- [ ] Python 环境已设置（3.8+）
- [ ] 虚拟环境已创建并激活
- [ ] 所有依赖已安装
- [ ] NeuralOperator 包已安装
- [ ] GPU 可用（如需要）或 CPU 配置正确
- [ ] 数据已准备或下载路径已配置
- [ ] 模型配置已设置（`n_modes`, `hidden_channels` 等）
- [ ] 训练参数已配置（学习率、批次大小等）
- [ ] WandB 已配置（如使用）
- [ ] 存储空间充足
- [ ] 已阅读相关文档和示例

---

## 🎯 下一步

1. **运行快速示例**: 先运行 `complete_example.py` 验证环境
2. **调整配置**: 根据你的问题调整模型和训练参数
3. **实验迭代**: 尝试不同的超参数组合
4. **结果分析**: 使用可视化工具分析模型性能
5. **扩展到新问题**: 将 FNO 应用到你的具体问题

---

**祝实验顺利！** 🚀

如有问题，请参考项目文档或提交 Issue。

