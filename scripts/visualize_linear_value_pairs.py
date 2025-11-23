#!/usr/bin/env python
"""绘制 ReachableSet 的 linear_value_t1_t2 训练样本。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch


def build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/ReachableSetDataset/processed/linear_value_t1_t2_train_default.pt"),
        help="要可视化的训练 PT 文件路径。",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=[0],
        help="要绘制的样本索引列表，支持多个。",
    )
    parser.add_argument(
        "--colormap",
        default="viridis",
        help="matplotlib 的 colormap 名称。",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="若指定，则将每个样本的图保存到该目录，文件名为 sample_{idx}.png。",
    )
    return parser


def _ensure_dir(path: Path | None) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)


def _plot_field(ax, tensor: torch.Tensor, title: str, cmap: str) -> None:
    image = tensor.detach().cpu().numpy()
    im = ax.imshow(image, origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)


def visualize_samples(
    dataset_path: Path,
    sample_indices: Sequence[int],
    colormap: str,
    save_dir: Path | None,
) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} 不存在，请先运行 data/ReachableSetDataset/processed 生成数据。")

    data = torch.load(dataset_path, map_location="cpu")
    x = data["x"]
    y = data["y"]
    total = x.shape[0]

    if save_dir is not None:
        _ensure_dir(save_dir)

    for sample_index in sample_indices:
        if not 0 <= sample_index < total:
            raise IndexError(f"样本编号 {sample_index} 越界，训练集共有 {total} 个样本。")

        sample_x = x[sample_index]
        sample_y = y[sample_index]

        channel_count = sample_x.shape[0]
        panel_count = channel_count + 2
        fig, axes = plt.subplots(1, panel_count, figsize=(3.5 * panel_count, 3.8))

        if panel_count == 1:
            axes = [axes]

        for idx in range(channel_count):
            _plot_field(axes[idx], sample_x[idx], f"x channel {idx}", colormap)

        _plot_field(axes[channel_count], sample_y.squeeze(0) if sample_y.shape[0] == 1 else sample_y[0], "y target", colormap)

        diff = sample_y[0] - sample_x[0] if sample_x.shape[0] > 0 and sample_y.shape[0] > 0 else torch.zeros_like(sample_y[0])
        _plot_field(axes[channel_count + 1], diff, "y - x0", colormap)

        fig.suptitle(f"linear_value_t1_t2 training sample {sample_index}", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir is not None:
            save_path = save_dir / f"sample_{sample_index}.png"
            plt.savefig(save_path, dpi=150)
            print(f"保存图像到 {save_path}")
            plt.close(fig)
        else:
            plt.show()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    visualize_samples(
        dataset_path=args.dataset_path,
        sample_indices=args.sample_indices,
        colormap=args.colormap,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

