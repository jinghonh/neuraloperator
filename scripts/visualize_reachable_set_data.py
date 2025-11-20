#!/usr/bin/env python
"""可视化 `convert_reachable_set_dataset.py` 输出的 Reachable Set PT 数据。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import torch


def build_argparser() -> argparse.ArgumentParser:
    """返回控制可视化行为的参数解析器。"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default="data/ReachableSetDataset/processed/linear_value_function_train_default.pt",
        help="待可视化的 PT 文件路径。",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="要可视化的样本编号（从 0 开始）。",
    )
    parser.add_argument(
        "--max-input-channels",
        type=int,
        default=5,
        help="最多展示多少个输入通道。",
    )
    parser.add_argument(
        "--colormap",
        default="viridis",
        help="用于绘制场值的 matplotlib colormap。",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="可选的图像保存路径；不指定则显示窗口。",
    )
    return parser


def _get_channel_names(metadata: dict, channel_count: int) -> List[str]:
    """根据 metadata 生成输入通道的可读名称列表。"""

    base = ["grid_x", "grid_y", "time"]
    params_per_sample = metadata.get("params_per_sample")
    if params_per_sample is None:
        params_per_sample = max(0, channel_count - len(base))
    base.extend(f"param_{idx}" for idx in range(params_per_sample))
    return base[:channel_count]


def _chunk_axes(n_plots: int, ncols: int = 3) -> List:
    """根据总图像数构造子图轴集合。"""

    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if isinstance(axes, Iterable):
        axes = list(axes)
    else:
        axes = [axes]
    flat_axes = []
    for ax in axes:
        if isinstance(ax, Iterable):
            flat_axes.extend(list(ax))
        else:
            flat_axes.append(ax)
    return fig, flat_axes


def _render_channels(
    fig,
    axes: List,
    values: Iterable[tuple[str, torch.Tensor]],
    colormap: str,
) -> None:
    """在子图上绘制每个通道/字段。"""

    axes_used = axes[: len(list(values))]
    for ax, (name, tensor) in zip(axes_used, values):
        image = tensor.cpu().numpy()
        im = ax.imshow(image, origin="lower", cmap=colormap)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_used[len(list(values)) :]:
        ax.set_visible(False)


def visualize_sample(
    dataset_path: Path,
    sample_index: int,
    max_input_channels: int,
    colormap: str,
    save_path: Path | None,
) -> None:
    """加载 PT 文件并绘制指定样本的输入/输出。"""

    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} 不存在")
    payload = torch.load(dataset_path, map_location="cpu")
    x = payload["x"]
    y = payload["y"]
    metadata = payload.get("metadata", {})

    total_samples = x.shape[0]
    if not 0 <= sample_index < total_samples:
        raise IndexError(f"sample_index 必须在 0~{total_samples - 1} 之间")

    sample_x = x[sample_index]
    sample_y = y[sample_index]
    channel_names = _get_channel_names(metadata, sample_x.shape[0])

    input_channels = min(len(channel_names), max_input_channels)

    fields = [(name, sample_x[idx]) for idx, name in enumerate(channel_names[:input_channels])]
    fields.append(("y_field", sample_y.squeeze(0) if sample_y.shape[0] == 1 else sample_y[0]))

    fig, axes = _chunk_axes(len(fields), ncols=3)
    _render_channels(fig, axes, fields, colormap)

    metas = metadata.copy()
    metas.setdefault("input_channels", sample_x.shape[0])
    metas.setdefault("output_channels", sample_y.shape[0])
    metas.setdefault("shape", sample_y.shape[1:])
    meta_text = "\n".join(f"{key}: {value}" for key, value in metas.items())
    fig.suptitle(f"Sample {sample_index}\n{meta_text}", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"已保存可视化图像到 {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    visualize_sample(
        dataset_path=args.dataset_path,
        sample_index=args.sample_index,
        max_input_channels=args.max_input_channels,
        colormap=args.colormap,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()

