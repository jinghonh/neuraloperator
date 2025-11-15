"""
完整的Neural Operator训练示例
从数据加载到最后的可视化

这个脚本展示了完整的工作流程：
1. 环境设置和依赖导入
2. 数据加载和预处理
3. 模型创建和配置
4. 训练过程
5. 模型评估
6. 结果可视化（包括零样本超分辨率）
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 配置 matplotlib 以支持中文显示 (macOS)
# 设置 macOS 系统支持的中文字体列表
# matplotlib 会自动选择第一个可用的字体
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',      # 苹方（macOS 默认中文字体）
    'STHeiti',          # 华文黑体
    'Heiti SC',         # 黑体
    'Hiragino Sans GB', # 冬青黑体
    'Arial Unicode MS', # Arial Unicode MS
    'DejaVu Sans'       # 备用字体
]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入neuralop的核心组件
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss


def main():
    """主函数"""
    print("=" * 80)
    print("Neural Operator 完整训练示例")
    print("=" * 80)

    # ========================================================================
    # 第1步: 环境设置
    # ========================================================================
    print("\n[步骤 1/6] 环境设置")
    print("-" * 80)

    # 设置设备 (CPU 或 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置随机种子以保证可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    print("随机种子已设置为 42")


    # ========================================================================
    # 第2步: 数据加载和预处理
    # ========================================================================
    print("\n[步骤 2/6] 数据加载和预处理")
    print("-" * 80)

    print("正在加载 Darcy Flow 数据集...")
    print("  - Darcy Flow 方程描述流体在多孔介质中的流动")
    print("  - 输入: 渗透率场 (permeability field)")
    print("  - 输出: 压力场 (pressure field)")

    # 加载小型 Darcy Flow 数据集
    # 这个数据集专门设计用于快速演示，可以在 CPU 上训练
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000,        # 训练样本数量
        batch_size=32,       # 训练批次大小
        n_tests=[100, 50],   # 每个测试分辨率的样本数量
        test_resolutions=[16, 32],  # 测试分辨率 (16x16 和 32x32)
        test_batch_sizes=[32, 32],  # 测试批次大小
    )

    # 将数据处理器移到设备上
    data_processor = data_processor.to(device)

    print(f"✓ 数据加载完成!")
    print(f"  - 训练样本: {len(train_loader.dataset)}")
    print(f"  - 训练批次大小: {train_loader.batch_size}")
    print(f"  - 测试分辨率: {list(test_loaders.keys())}")
    print(f"  - 测试样本数: {[len(loader.dataset) for loader in test_loaders.values()]}")

    # 查看一个数据样本
    sample_batch = next(iter(train_loader))
    print(f"\n数据形状信息:")
    print(f"  - 输入形状 (x): {sample_batch['x'].shape}")  # [batch, channels, height, width]
    print(f"  - 输出形状 (y): {sample_batch['y'].shape}")


    # ========================================================================
    # 第3步: 创建模型
    # ========================================================================
    print("\n[步骤 3/6] 创建 FNO (Fourier Neural Operator) 模型")
    print("-" * 80)

    model = FNO(
        n_modes=(8, 8),              # Fourier 模式数量 (控制频率空间的截断)
        in_channels=1,               # 输入通道数 (渗透率场)
        out_channels=1,              # 输出通道数 (压力场)
        hidden_channels=32,          # 隐藏层通道数
        projection_channel_ratio=2,  # 投影层的通道比例
        n_layers=4,                  # FNO 层数
    )
    model = model.to(device)

    # 统计模型参数
    n_params = count_model_params(model)
    print(f"✓ 模型创建完成!")
    print(f"  - 模型参数总数: {n_params:,}")
    print(f"  - Fourier 模式: {model.n_modes}")
    print(f"  - 隐藏通道数: {model.hidden_channels}")
    print(f"  - 网络层数: {model.n_layers}")


    # ========================================================================
    # 第4步: 设置训练组件
    # ========================================================================
    print("\n[步骤 4/6] 设置训练组件 (优化器、调度器、损失函数)")
    print("-" * 80)

    # 优化器: AdamW (带权重衰减的 Adam)
    optimizer = AdamW(
        model.parameters(),
        lr=8e-3,              # 学习率
        weight_decay=1e-4     # 权重衰减 (L2 正则化)
    )
    print(f"✓ 优化器: AdamW")
    print(f"  - 学习率: 8e-3")
    print(f"  - 权重衰减: 1e-4")

    # 学习率调度器: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30  # 余弦周期
    )
    print(f"✓ 调度器: CosineAnnealingLR")
    print(f"  - T_max: 30")

    # 损失函数
    # H1Loss: 包含函数值和梯度信息，适合 PDE 问题
    # L2Loss: 仅考虑函数值
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    train_loss = h1loss
    eval_losses = {"h1": h1loss, "l2": l2loss}

    print(f"✓ 损失函数:")
    print(f"  - 训练损失: H1Loss (包含梯度信息)")
    print(f"  - 评估损失: H1Loss 和 L2Loss")


    # ========================================================================
    # 第5步: 训练模型
    # ========================================================================
    print("\n[步骤 5/6] 开始训练模型")
    print("-" * 80)

    # 创建训练器
    trainer = Trainer(
        model=model,
        n_epochs=20,                    # 训练轮数
        device=device,
        data_processor=data_processor,
        wandb_log=False,                # 不使用 Weights & Biases 日志
        eval_interval=3,                # 每3个epoch评估一次
        use_distributed=False,          # 不使用分布式训练
        verbose=True,                   # 打印详细信息
    )

    print("开始训练... (这可能需要几分钟)")
    print()

    # 执行训练
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


    # ========================================================================
    # 第6步: 可视化结果
    # ========================================================================
    print("\n[步骤 6/6] 可视化结果")
    print("-" * 80)

    # 设置模型为评估模式
    model.eval()

    # ------------ 6.1: 在训练分辨率 (16x16) 上的预测 ------------
    print("\n6.1 在训练分辨率 (16x16) 上的预测")
    test_samples_16 = test_loaders[16].dataset

    fig = plt.figure(figsize=(15, 5))
    for index in range(3):
        data = test_samples_16[index]
        data = data_processor.preprocess(data, batched=False)
        
        # 输入 (渗透率场)
        x = data["x"].to(device)
        # 真实输出 (压力场)
        y = data["y"].to(device)
        # 模型预测
        with torch.no_grad():
            out = model(x.unsqueeze(0))
        
        # 转换为 numpy 数组用于绘图
        x_np = x[0].cpu().numpy()
        y_np = y.squeeze().cpu().numpy()
        out_np = out.squeeze().cpu().numpy()
        
        # 绘制输入
        ax = fig.add_subplot(3, 3, index * 3 + 1)
        im = ax.imshow(x_np, cmap="viridis")
        if index == 0:
            ax.set_title("输入 (渗透率场)", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 绘制真实输出
        ax = fig.add_subplot(3, 3, index * 3 + 2)
        im = ax.imshow(y_np, cmap="coolwarm")
        if index == 0:
            ax.set_title("真实输出 (压力场)", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 绘制模型预测
        ax = fig.add_subplot(3, 3, index * 3 + 3)
        im = ax.imshow(out_np, cmap="coolwarm")
        if index == 0:
            ax.set_title("模型预测", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 计算并显示误差
        error = np.abs(y_np - out_np).mean()
        ax.text(0.5, -0.15, f'平均误差: {error:.4f}', 
                transform=ax.transAxes, ha='center', fontsize=9)

    fig.suptitle("FNO 在 16x16 分辨率上的预测结果", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("fno_predictions_16x16.png", dpi=150, bbox_inches='tight')
    print("✓ 图像已保存: fno_predictions_16x16.png")
    plt.show()

    # ------------ 6.2: 零样本超分辨率 (32x32) ------------
    print("\n6.2 零样本超分辨率: 16x16 → 32x32")
    print("注意: 模型只在 16x16 分辨率上训练，但可以直接在 32x32 上推理!")

    test_samples_32 = test_loaders[32].dataset

    fig = plt.figure(figsize=(15, 5))
    for index in range(3):
        data = test_samples_32[index]
        data = data_processor.preprocess(data, batched=False)
        
        # 更高分辨率的输入
        x = data["x"].to(device)
        # 更高分辨率的真实输出
        y = data["y"].to(device)
        # 模型在更高分辨率上的预测
        with torch.no_grad():
            out = model(x.unsqueeze(0))
        
        # 转换为 numpy 数组
        x_np = x[0].cpu().numpy()
        y_np = y.squeeze().cpu().numpy()
        out_np = out.squeeze().cpu().numpy()
        
        # 绘制输入
        ax = fig.add_subplot(3, 3, index * 3 + 1)
        im = ax.imshow(x_np, cmap="viridis")
        if index == 0:
            ax.set_title("输入 32x32", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 绘制真实输出
        ax = fig.add_subplot(3, 3, index * 3 + 2)
        im = ax.imshow(y_np, cmap="coolwarm")
        if index == 0:
            ax.set_title("真实输出 32x32", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 绘制模型预测
        ax = fig.add_subplot(3, 3, index * 3 + 3)
        im = ax.imshow(out_np, cmap="coolwarm")
        if index == 0:
            ax.set_title("FNO 预测 32x32", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 计算并显示误差
        error = np.abs(y_np - out_np).mean()
        ax.text(0.5, -0.15, f'平均误差: {error:.4f}', 
                transform=ax.transAxes, ha='center', fontsize=9)

    fig.suptitle("零样本超分辨率: 16x16 → 32x32 (无需重新训练!)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("fno_super_resolution_32x32.png", dpi=150, bbox_inches='tight')
    print("✓ 图像已保存: fno_super_resolution_32x32.png")
    plt.show()

    # ------------ 6.3: 定量评估 ------------
    print("\n6.3 定量评估")
    print("-" * 50)

    model.eval()
    with torch.no_grad():
        # 在 16x16 分辨率上评估
        errors_16 = []
        for data in test_loaders[16]:
            x = data['x'].to(device)
            y = data['y'].to(device)
            out = model(x)
            error = l2loss(out, y).item()
            errors_16.append(error)
        
        # 在 32x32 分辨率上评估 (零样本超分辨率)
        errors_32 = []
        for data in test_loaders[32]:
            x = data['x'].to(device)
            y = data['y'].to(device)
            out = model(x)
            error = l2loss(out, y).item()
            errors_32.append(error)

    print(f"16x16 分辨率 (训练分辨率):")
    print(f"  - 平均 L2 误差: {np.mean(errors_16):.6f}")
    print(f"  - 标准差: {np.std(errors_16):.6f}")

    print(f"\n32x32 分辨率 (零样本超分辨率):")
    print(f"  - 平均 L2 误差: {np.mean(errors_32):.6f}")
    print(f"  - 标准差: {np.std(errors_32):.6f}")

    # 绘制误差分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(errors_16, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('L2 误差', fontsize=11)
    axes[0].set_ylabel('频数', fontsize=11)
    axes[0].set_title('16x16 分辨率误差分布', fontsize=12, fontweight='bold')
    axes[0].axvline(np.mean(errors_16), color='red', linestyle='--', 
                    linewidth=2, label=f'平均值: {np.mean(errors_16):.4f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(errors_32, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_xlabel('L2 误差', fontsize=11)
    axes[1].set_ylabel('频数', fontsize=11)
    axes[1].set_title('32x32 分辨率误差分布 (零样本)', fontsize=12, fontweight='bold')
    axes[1].axvline(np.mean(errors_32), color='red', linestyle='--', 
                    linewidth=2, label=f'平均值: {np.mean(errors_32):.4f}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("error_distribution.png", dpi=150, bbox_inches='tight')
    print("\n✓ 误差分布图已保存: error_distribution.png")
    plt.show()

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("训练和评估完成!")
    print("=" * 80)
    print("\n关键要点:")
    print("  1. ✓ 成功加载了 Darcy Flow 数据集")
    print("  2. ✓ 创建并训练了 Fourier Neural Operator (FNO)")
    print("  3. ✓ 模型在训练分辨率 (16x16) 上表现良好")
    print("  4. ✓ 展示了零样本超分辨率能力 (16x16 → 32x32)")
    print("  5. ✓ 生成了可视化结果和误差分析")
    print("\n生成的文件:")
    print("  - fno_predictions_16x16.png")
    print("  - fno_super_resolution_32x32.png")
    print("  - error_distribution.png")
    print("\nNeural Operator 的优势:")
    print("  • 分辨率不变性: 可以在任意分辨率上推理")
    print("  • 函数空间学习: 学习的是算子，而非点对点映射")
    print("  • 高效性: 参数量远少于传统神经网络")
    print("=" * 80)


if __name__ == "__main__":
    main()
