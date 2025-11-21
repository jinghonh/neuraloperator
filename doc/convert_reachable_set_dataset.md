# `convert_reachable_set_dataset.py` 说明文档

## 概述
`convert_reachable_set_dataset.py` 将 MATLAB 格式的 Reachable Set 数据集（默认 `data/ReachableSetDataset/linearSystemValueFunctionData.mat`）转换为 NeuralOperator 训练可用的 PyTorch `.pt` 格式数据。它会：

- 读取 `.mat` 文件中的 `samples` 条目，逐样本展开每个系统的输入参数（`A`、`B`）、空间场 `dataStack` 及时间戳 `timeStamps`。
- 根据可用网格信息自动构造或补齐二维坐标 (`grid_x`, `grid_y`)；如果 `.mat` 文件未携带，脚本会生成默认的 `[-1, 1]` 区间网格。
- 将每个时刻的场切片拼接为模型输入，通道顺序为 x 坐标、y 坐标、时间、参数展开值，输出为该时刻的场单通道标签，并在整个数据集上打乱后按比例拆分为训练/测试。
- 导出训练集、测试集各自的 `.pt` 文件（包含 `x`, `y`, `metadata`），以及整体 metadata JSON 以便训练脚本使用。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mat-path` | `data/ReachableSetDataset/linearSystemValueFunctionData.mat` | 需要转换的 MATLAB 数据文件。 |
| `--output-dir` | `data/ReachableSetDataset/processed` | 存放输出 `.pt` 与 metadata 的目录。 |
| `--dataset-name` | `linear_value_function` | 输出文件名前缀。 |
| `--resolution-tag` | `default` | 附加于输出文件名的解析度标签。 |
| `--train-fraction` | `0.9` | 初始化总样本后划分为训练集的比例（0~1）。 |
| `--shuffle-seed` | `0` | 样本打乱时的随机种子。 |
| `--max-pairs` | `None` | 限制生成的 `(系统, 时间)` 对数量，默认无上限。 |

参数确保数据在不同训练配置下可以复现且支持自定义输出路径、文件名与样本数量控制。

## 生成的数据结构

### `.pt` 文件结构（训练 / 测试相同）

每个 `.pt` 文件是一个字典：

- `x`: 形状为 `(N, C_in, H, W)`，其中：
  - `N`：样本对数（系统×时间片）。
  - `C_in`：输入通道，包括 `x` 网格、`y` 网格、时间通道、系统参数通道（A、B 向量展平后填充为常量平面）。
  - `H, W`：空间分辨率（由原始 `dataStack` 决定）。
- `y`: 形状为 `(N, 1, H, W)`，对应每个样本时间点的场值切片。
- `metadata`: 与整体 metadata 同步，包含：
  - `grid_shape`: 原始 spatial 维度。
  - `params_per_sample`: 参数通道数（A、B 展平后长度）。
  - `dataset_name`/`resolution_tag`: CLI 指定的名字与标签。
  - `input_channels`/`output_channels`：`x`/`y` 的通道数。
  - `train_samples`/`test_samples`：训练/测试各自的样本量。

### `metadata.json`

保存于同一目录，结构同 `metadata` 字段，便于模型配置读取。

## 具体处理细节

- `grid` 组解析：尝试从 `.mat` 中提取 `gridx`, `gridy` 等字段；若仅有 1D 数据依然可广播匹配空间维度。
- `dataStack` 处理：确保时间维度位于最后（通过 `_move_time_axis` 寻找与 `timeStamps` 长度一致的轴）。
- 参数拼接：把 `A`、`B` 展平成 1D 向量后，为每个时间切片生成与场值相同空间形状的常量通道。
- 样本限制：`--max-pairs` 控制总数，便于生成调试规模更小的子集。
- 打乱与拆分：按随机种子打乱全体样本后根据 `train_fraction` 切分训练/测试，若比例导致空集会抛错。

## 输出文件示例

- `data/ReachableSetDataset/processed/linear_value_function_train_default.pt`
- `data/ReachableSetDataset/processed/linear_value_function_test_default.pt`
- `data/ReachableSetDataset/processed/linear_value_function_metadata.json`

以上文件可直接用于 `neuralop/data/datasets/reachable_set.py` 等训练流程，结合 metadata 统一动力学配置。

