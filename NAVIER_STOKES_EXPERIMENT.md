# 二维纳维-斯托克斯实验运行与可视化指南

本指南说明如何使用 `scripts/run_navier_stokes_experiment.py` 一键运行二维 Navier-Stokes 实验（训练 + 评估 + 可视化），并解释脚本需要的依赖、可配置参数以及产物。

## 环境准备

1. 建议在隔离环境中安装依赖：
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .[dev]
   ```
2. 脚本会自动从 Zenodo 下载分辨率为 128 或 1024 的 Navier-Stokes 数据集（若本地不存在）。默认路径为 `~/data/navier_stokes`，可通过 `--data-root` 改写。

## 脚本功能概览

`scripts/run_navier_stokes_experiment.py` 流程：

- 读取/下载 Navier-Stokes `.pt` 数据。
- 初始化轻量版 FNO 模型（根据数据通道数自适应输入/输出维度）。
- 使用 H1Loss 训练、StepLR 调度、AdamW 优化，可选混合精度。
- 评估 L2/H1 损失，统计 batch 误差分布。
- 生成预测图与误差直方图，并将指标写入 JSON。

所有输出都会写入 `--output-dir`（默认 `outputs/navier_stokes`）。

## 运行示例

```bash
python scripts/run_navier_stokes_experiment.py \
  --data-root ~/data/navier_stokes \
  --output-dir runs/navier_demo \
  --n-train 512 \
  --n-test 64 \
  --epochs 5 \
  --batch-size 4 \
  --test-batch-size 4 \
  --samples-to-plot 3
```

该命令：

- 下载（如需）并加载 512 个训练样本、64 个测试样本，分辨率 128×128。
- 在 CPU/GPU/MPS（自动探测，可用 `--device` 强制）上训练 5 epoch。
- 在 `runs/navier_demo/` 下生成：
  - `navier_stokes_metrics.json`：记录 L2/H1 平均损失、样本数、参数量等。
  - `navier_stokes_predictions_128.png`：按采样索引展示输入、目标、预测、绝对误差（逐像素幅值）。
  - `navier_stokes_error_hist_128.png`：batch L2 损失直方图。

## 关键参数

- `--train-resolution` / `--test-resolution`：必须是 128 或 1024，并需要与下载的数据匹配。
- `--n-modes NX NY`：FNO 傅里叶模式数，默认 `(16, 16)`，分辨率越高可适当增大。
- `--hidden-channels`、`--n-layers`、`--projection-channel-ratio`：控制模型容量。
- `--mixed-precision`：在支持 CUDA 的设备上开启 `torch.cuda.amp`。
- `--samples-to-plot`：可视化的测试样本个数（脚本自动截断不超过测试集大小）。
- `--seed`：保证可复现的数据顺序和权重初始化。

更多参数可通过 `python scripts/run_navier_stokes_experiment.py --help` 查看。

## 可视化解释

- **预测图 (`navier_stokes_predictions_*.png`)**：每一行对应一个样本，四列分别是输入场、目标场、模型预测和绝对误差。若数据包含多个通道，脚本自动计算前两个通道的速度幅值或在单通道情况下直接渲染该通道。
- **误差直方图 (`navier_stokes_error_hist_*.png`)**：展示每个 batch 的 L2 损失分布，可快速判断训练是否稳定。

## 后续建议

- 提升精度：增大 `--n-train`、`--epochs`、`--hidden-channels` 或切换至更高分辨率（1024）。
- 研究多分辨率泛化：调整 `--train-resolution` 与 `--test-resolution` 组合并观察指标差异。
- 如果需要 WandB 日志或多机训练，可参考 `scripts/train_navier_stokes.py` 进行扩展。
