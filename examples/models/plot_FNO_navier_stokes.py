"""
训练和可视化 FNO 在二维纳维-斯托克斯方程上的实验
====================================================

本脚本展示了如何：
1. 加载和预处理 Navier-Stokes 数据集
2. 创建 FNO 模型架构
3. 设置训练组件（优化器、调度器、损失函数）
4. 训练模型
5. 评估预测结果和零样本超分辨率
6. 可视化结果

Navier-Stokes 方程描述流体的运动，是流体动力学的核心方程。
输入：初始速度场（2个通道：u和v分量）
输出：后续时间步的速度场（2个通道：u和v分量）
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 配置 matplotlib 以支持中文显示 (macOS)
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',      # 苹方（macOS 默认中文字体）
    'STHeiti',          # 华文黑体
    'Heiti SC',         # 黑体
    'Hiragino Sans GB', # 冬青黑体
    'Arial Unicode MS', # Arial Unicode MS
    'DejaVu Sans'       # 备用字体
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入 neuralop 的核心组件
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# %%
# 加载 Navier-Stokes 数据集
# --------------------------
# Navier-Stokes 数据集包含初始速度场和后续时间步的速度场
# 注意：数据集较大，如果本地没有数据，需要先下载

data_root = Path("./data/navier_stokes/").expanduser()
print(f"\n数据目录: {data_root}")

# 如果数据目录不存在，提示用户
if not data_root.exists():
    print(f"警告: 数据目录 {data_root} 不存在!")
    print("请确保已下载 Navier-Stokes 数据集，或修改 data_root 路径")
    print("数据集通常需要从指定位置下载")
    sys.exit(1)

# 加载数据集（使用较小的分辨率用于演示）
train_loader, test_loaders, data_processor = load_navier_stokes_pt(
    data_root=data_root,
    n_train=1000,              # 训练样本数量（可根据需要调整）
    batch_size=8,               # 批次大小
    train_resolution=64,        # 训练分辨率（64x64，较小以便快速演示）
    n_tests=[200],              # 测试样本数量
    test_resolutions=[64],      # 测试分辨率
    test_batch_sizes=[8],       # 测试批次大小
    encode_input=True,          # 编码输入
    encode_output=True,         # 编码输出
)

data_processor = data_processor.to(device)

print(f"✓ 数据加载完成!")
print(f"  - 训练样本: {len(train_loader.dataset)}")
print(f"  - 训练分辨率: 64x64")
print(f"  - 测试样本数: {len(test_loaders[64].dataset)}")

# 查看一个数据样本
sample_batch = next(iter(train_loader))
print(f"\n数据形状信息:")
print(f"  - 输入形状 (x): {sample_batch['x'].shape}")  # [batch, channels, height, width]
print(f"  - 输出形状 (y): {sample_batch['y'].shape}")
print(f"  - 输入通道数: {sample_batch['x'].shape[1]} (u和v速度分量)")
print(f"  - 输出通道数: {sample_batch['y'].shape[1]} (u和v速度分量)")

# %%
# 创建 FNO 模型
# --------------
# FNO 适用于 Navier-Stokes 方程，因为它能有效处理周期性边界条件

model = FNO(
    n_modes=(16, 16),           # Fourier 模式数量（控制频率空间截断）
    in_channels=2,              # 输入通道数（u和v速度分量）
    out_channels=2,             # 输出通道数（u和v速度分量）
    hidden_channels=32,         # 隐藏层通道数
    projection_channel_ratio=2, # 投影层的通道比例
    n_layers=4,                 # FNO 层数
)
model = model.to(device)

# 统计模型参数
n_params = count_model_params(model)
print(f"\n✓ 模型创建完成!")
print(f"  - 模型参数总数: {n_params:,}")
print(f"  - Fourier 模式: {model.n_modes}")
print(f"  - 隐藏通道数: {model.hidden_channels}")
print(f"  - 网络层数: {model.n_layers}")

# %%
# 创建优化器和调度器
# -------------------
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,              # 学习率（根据配置）
    weight_decay=1e-4     # 权重衰减
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,        # 每100个epoch降低学习率
    gamma=0.5             # 学习率衰减因子
)

print(f"\n✓ 优化器: AdamW")
print(f"  - 学习率: 3e-4")
print(f"  - 权重衰减: 1e-4")
print(f"✓ 调度器: StepLR")
print(f"  - Step size: 100")
print(f"  - Gamma: 0.5")

# %%
# 设置损失函数
# ------------
# H1Loss 包含函数值和梯度信息，适合 PDE 问题
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

print(f"\n✓ 损失函数:")
print(f"  - 训练损失: H1Loss (包含梯度信息)")
print(f"  - 评估损失: H1Loss 和 L2Loss")

# %%
# 训练模型
# --------
print("\n" + "=" * 80)
print("开始训练模型...")
print("=" * 80)

trainer = Trainer(
    model=model,
    n_epochs=50,                # 训练轮数（可根据需要调整）
    device=device,
    data_processor=data_processor,
    wandb_log=False,            # 不使用 Weights & Biases 日志
    eval_interval=10,            # 每10个epoch评估一次
    use_distributed=False,       # 不使用分布式训练
    verbose=True,                # 打印训练进度
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

print("\n✓ 训练完成!")

# %%
# 可视化预测结果
# --------------
# 我们将比较输入、真实输出和模型预测

print("\n" + "=" * 80)
print("可视化预测结果")
print("=" * 80)

model.eval()
test_samples = test_loaders[64].dataset

# 可视化3个样本
fig = plt.figure(figsize=(18, 12))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    
    # 输入（初始速度场）
    x = data["x"].to(device)
    # 真实输出（后续时间步的速度场）
    y = data["y"].to(device)
    # 模型预测
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    
    # 转换为 numpy 数组用于绘图
    x_np = x.cpu().numpy()  # [2, H, W]
    y_np = y.squeeze().cpu().numpy()  # [2, H, W]
    out_np = out.squeeze().cpu().numpy()  # [2, H, W]
    
    # 计算速度幅值（magnitude）
    x_magnitude = np.sqrt(x_np[0]**2 + x_np[1]**2)
    y_magnitude = np.sqrt(y_np[0]**2 + y_np[1]**2)
    out_magnitude = np.sqrt(out_np[0]**2 + out_np[1]**2)
    
    # 绘制输入速度幅值
    ax = fig.add_subplot(3, 4, index * 4 + 1)
    im = ax.imshow(x_magnitude, cmap="viridis", origin="lower")
    if index == 0:
        ax.set_title("输入速度幅值\n(初始时刻)", fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 绘制输入速度场（向量场）
    ax = fig.add_subplot(3, 4, index * 4 + 2)
    # 下采样以便显示向量场
    step = 4
    Y, X = np.meshgrid(np.arange(0, x_magnitude.shape[0], step),
                       np.arange(0, x_magnitude.shape[1], step), indexing='ij')
    ax.imshow(x_magnitude, cmap="gray", alpha=0.5, origin="lower")
    ax.quiver(X, Y, x_np[0][::step, ::step], x_np[1][::step, ::step],
              scale=20, width=0.003, color='red')
    if index == 0:
        ax.set_title("输入速度场\n(向量场)", fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # 绘制真实输出速度幅值
    ax = fig.add_subplot(3, 4, index * 4 + 3)
    im = ax.imshow(y_magnitude, cmap="viridis", origin="lower")
    if index == 0:
        ax.set_title("真实输出速度幅值\n(后续时刻)", fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 绘制模型预测速度幅值
    ax = fig.add_subplot(3, 4, index * 4 + 4)
    im = ax.imshow(out_magnitude, cmap="viridis", origin="lower")
    if index == 0:
        ax.set_title("模型预测速度幅值", fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 计算并显示误差
    error = np.abs(y_magnitude - out_magnitude).mean()
    ax.text(0.5, -0.15, f'平均误差: {error:.4f}', 
            transform=ax.transAxes, ha='center', fontsize=9)

fig.suptitle("FNO 在 Navier-Stokes 方程上的预测结果 (64x64)", 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig("navier_stokes_predictions_64x64.png", dpi=150, bbox_inches='tight')
print("✓ 图像已保存: navier_stokes_predictions_64x64.png")
plt.show()

# %%
# 可视化误差分布
# --------------
print("\n可视化误差分布...")

model.eval()
errors = []
with torch.no_grad():
    for data in test_loaders[64]:
        x = data['x'].to(device)
        y = data['y'].to(device)
        out = model(x)
        error = l2loss(out, y).item()
        errors.append(error)

errors = np.array(errors)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 误差直方图
axes[0].hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0].set_xlabel('L2 误差', fontsize=11)
axes[0].set_ylabel('频数', fontsize=11)
axes[0].set_title('误差分布', fontsize=12, fontweight='bold')
axes[0].axvline(np.mean(errors), color='red', linestyle='--', 
                linewidth=2, label=f'平均值: {np.mean(errors):.6f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 误差统计
stats_text = f"""
统计信息:
  平均值: {np.mean(errors):.6f}
  中位数: {np.median(errors):.6f}
  标准差: {np.std(errors):.6f}
  最小值: {np.min(errors):.6f}
  最大值: {np.max(errors):.6f}
"""
axes[1].text(0.1, 0.5, stats_text, fontsize=11, 
             verticalalignment='center', family='monospace')
axes[1].axis('off')
axes[1].set_title('误差统计', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("navier_stokes_error_distribution.png", dpi=150, bbox_inches='tight')
print("✓ 误差分布图已保存: navier_stokes_error_distribution.png")
plt.show()

# %%
# 可视化单个样本的详细对比
# --------------------------
print("\n可视化单个样本的详细对比...")

# 选择一个样本进行详细可视化
sample_idx = 0
data = test_samples[sample_idx]
data = data_processor.preprocess(data, batched=False)

x = data["x"].to(device)
y = data["y"].to(device)
with torch.no_grad():
    out = model(x.unsqueeze(0))

x_np = x.cpu().numpy()
y_np = y.squeeze().cpu().numpy()
out_np = out.squeeze().cpu().numpy()

# 计算速度幅值和误差
x_magnitude = np.sqrt(x_np[0]**2 + x_np[1]**2)
y_magnitude = np.sqrt(y_np[0]**2 + y_np[1]**2)
out_magnitude = np.sqrt(out_np[0]**2 + out_np[1]**2)
error_magnitude = np.abs(y_magnitude - out_magnitude)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：速度幅值
im1 = axes[0, 0].imshow(x_magnitude, cmap="viridis", origin="lower")
axes[0, 0].set_title("输入速度幅值", fontsize=12, fontweight='bold')
axes[0, 0].axis('off')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

im2 = axes[0, 1].imshow(y_magnitude, cmap="viridis", origin="lower")
axes[0, 1].set_title("真实输出速度幅值", fontsize=12, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

im3 = axes[0, 2].imshow(out_magnitude, cmap="viridis", origin="lower")
axes[0, 2].set_title("模型预测速度幅值", fontsize=12, fontweight='bold')
axes[0, 2].axis('off')
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

# 第二行：u分量和v分量
im4 = axes[1, 0].imshow(y_np[0] - out_np[0], cmap="coolwarm", origin="lower")
axes[1, 0].set_title("u分量误差", fontsize=12, fontweight='bold')
axes[1, 0].axis('off')
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

im5 = axes[1, 1].imshow(y_np[1] - out_np[1], cmap="coolwarm", origin="lower")
axes[1, 1].set_title("v分量误差", fontsize=12, fontweight='bold')
axes[1, 1].axis('off')
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

im6 = axes[1, 2].imshow(error_magnitude, cmap="hot", origin="lower")
axes[1, 2].set_title("速度幅值误差", fontsize=12, fontweight='bold')
axes[1, 2].axis('off')
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

fig.suptitle("Navier-Stokes 方程预测详细对比", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig("navier_stokes_detailed_comparison.png", dpi=150, bbox_inches='tight')
print("✓ 详细对比图已保存: navier_stokes_detailed_comparison.png")
plt.show()

# %%
# 总结
# ----
print("\n" + "=" * 80)
print("训练和可视化完成!")
print("=" * 80)
print("\n关键要点:")
print("  1. ✓ 成功加载了 Navier-Stokes 数据集")
print("  2. ✓ 创建并训练了 Fourier Neural Operator (FNO)")
print("  3. ✓ 模型学习了从初始速度场到后续时间步速度场的映射")
print("  4. ✓ 生成了可视化结果和误差分析")
print("\n生成的文件:")
print("  - navier_stokes_predictions_64x64.png")
print("  - navier_stokes_error_distribution.png")
print("  - navier_stokes_detailed_comparison.png")
print("\nNavier-Stokes 方程:")
print("  • 描述流体的运动（速度场演化）")
print("  • 输入: 初始速度场 (u, v)")
print("  • 输出: 后续时间步的速度场 (u, v)")
print("  • FNO 的优势: 分辨率不变性，可以处理不同分辨率的输入")
print("=" * 80)

