#!/usr/bin/env python
"""分析 linear_value_t1_t2 数据集训练样本的输入分布、差异和简要统计。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch


def _format_stats(name: str, data: Iterable[float]) -> str:
    data = list(data)
    return f"{name}: min {min(data):.6f}, max {max(data):.6f}, mean {sum(data)/len(data):.6f}"


def explain_input(
    dataset_path: Path,
    channel_indices: Iterable[int] | None = None,
    sample_indices: Iterable[int] | None = None,
) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    payload = torch.load(dataset_path, map_location="cpu")
    x = payload["x"]
    n_samples, in_channels, *spatial = x.shape
    total_spatial = 1
    for dim in spatial:
        total_spatial *= dim

    sample_indices = list(sample_indices) if sample_indices is not None else [0, min(1, n_samples - 1)]
    channel_indices = (
        list(channel_indices)
        if channel_indices is not None
        else list(range(min(in_channels, 3)))
    )
    channel_indices = [idx for idx in channel_indices if 0 <= idx < in_channels]
    if not channel_indices:
        channel_indices = [0]
    sample_indices = list(sample_indices) if sample_indices is not None else [0, min(1, n_samples - 1)]
    channel_indices = (
        list(channel_indices)
        if channel_indices is not None
        else list(range(min(in_channels, 3)))
    )
    channel_indices = [idx for idx in channel_indices if 0 <= idx < in_channels]
    if not channel_indices:
        channel_indices = [0]

    print(f"Dataset: {dataset_path}")
    print(f"Samples: {n_samples}, channels: {in_channels}, spatial: {spatial}")

    flat_x = x.view(n_samples, -1)
    norm_per_sample = torch.linalg.norm(flat_x, dim=1).tolist()
    print(_format_stats("L2 norm per sample", norm_per_sample))

    mean_sample = x.mean(dim=0)
    max_diff_from_mean = (x - mean_sample).abs().max().item()
    print(f"Max abs difference from mean sample: {max_diff_from_mean:.6f}")

    max_pairwise = (x[1:] - x[:-1]).abs().max().item() if n_samples > 1 else 0.0
    print(f"Max abs difference between adjacent samples: {max_pairwise:.6f}")

    ref = x[0]
    global_max = (x - ref).abs().amax()
    print(f"Max difference to first sample: {global_max:.6f}")

    for idx in channel_indices:
        channel = x[:, idx]
        channel_flat = channel.view(n_samples, -1)
        channel_means = channel_flat.mean(dim=1)
        stats = {
            "per_sample_mean": channel_means.tolist(),
            "global_mean": channel_flat.mean().item(),
            "global_std": channel_flat.std(unbiased=False).item(),
        }
        print(f"Channel {idx} stats → mean per sample range {min(stats['per_sample_mean']):.6f}~{max(stats['per_sample_mean']):.6f}, "
              f"global mean {stats['global_mean']:.6f}, std {stats['global_std']:.6f}")

    if sample_indices:
        for sample_index in sample_indices:
            if not 0 <= sample_index < n_samples:
                continue
            sample = x[sample_index]
            diffs = (sample - x[0]).abs()
            print(
                f"Sample {sample_index}: max diff to sample 0 {diffs.max().item():.6f}, "
                f"mean diff {(diffs.mean()).item():.6f}"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/ReachableSetDataset/processed/linear_value_t1_t2_train_default.pt"),
        help="PT 数据文件。",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        help="要打印统计的输入通道编号。",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        help="要比较的样本编号（用来判断输入差异是否恒定）。",
    )
    args = parser.parse_args()
    explain_input(
        dataset_path=args.dataset_path,
        channel_indices=args.channels,
        sample_indices=args.samples,
    )


if __name__ == "__main__":
    main()

