#!/usr/bin/env python
"""将 Reachable Set MATLAB 数据集转换为 NeuralOperator 可用的 PT 格式文件。

本脚本读取 `data/ReachableSetDataset/linearSystemValueFunctionData.mat`（或用户指定的路径）中的
`samples`、`grid` 等字段，处理每个系统的 A/B 参数、时间序列以及空间场数据，将输入拼接
为坐标、时间和参数通道，输出为对应的场值切片，最终导出训练/测试的 `.pt` 文件和 metadata
JSON 以供后续训练流程使用。支持手动控制输出目录、样本数量上限、训练集比例和随机种子，可
以 `python scripts/convert_reachable_set_dataset.py --help` 查看所有参数。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch


def _default_grid(spatial_shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """根据格点形状构造默认的 [-1, 1] 网格坐标阵列。

    Parameters
    ----------
    spatial_shape:
        格点的高度和宽度。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        分别表示 x 和 y 坐标的二维阵列。
    """
    height, width = spatial_shape
    x_line = np.linspace(-2.0, 2.0, width, dtype=np.float32)
    y_line = np.linspace(-2.0, 2.0, height, dtype=np.float32)
    grid_x = np.broadcast_to(x_line, (height, width))
    grid_y = np.broadcast_to(y_line[:, None], (height, width))
    return grid_x, grid_y


def _reshape_grid_component(
    component: np.ndarray | None,
    spatial_shape: Sequence[int],
    horizontal: bool,
) -> np.ndarray | None:
    """尝试将网格组件重塑为与 spatial_shape 匹配的二维阵列。

    Parameters
    ----------
    component:
        可选的一维或二维坐标数组。
    spatial_shape:
        期望的 (高度, 宽度)。
    horizontal:
        用于判断 component 表示的是水平方向 (x) 还是竖直方向 (y)。

    Returns
    -------
    np.ndarray | None
        匹配的坐标阵列或无法匹配时返回 None。
    """
    if component is None:
        return None
    height, width = spatial_shape
    arr = np.array(component, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.shape == tuple(spatial_shape):
        return arr
    if arr.ndim == 1:
        if horizontal and arr.shape[0] == width:
            return np.broadcast_to(arr.reshape(1, width), (height, width)).astype(np.float32)
        if not horizontal and arr.shape[0] == height:
            return np.broadcast_to(arr.reshape(height, 1), (height, width)).astype(np.float32)
    return None


def _resolve_dataset_values(dataset: h5py.Dataset) -> List[np.ndarray]:
    """展开 object dtype 的 HDF5 数据集引用，返回可用的 numpy 数组。

    Parameters
    ----------
    dataset:
        可能包含引用的 HDF5 数据集。

    Returns
    -------
    List[np.ndarray]
        将全部解析结果以 float32 形式返回。
    """
    if dataset.dtype != object:
        return [np.array(dataset, dtype=np.float32)]

    resolved = []
    for index in np.ndindex(dataset.shape):
        entry = dataset[index]
        while isinstance(entry, np.ndarray) and entry.size == 1:
            entry = entry.reshape(-1)[0]
        if isinstance(entry, h5py.Reference):
            target = dataset.file[entry]
            if isinstance(target, h5py.Dataset):
                arr = np.array(target)
                if arr.dtype.kind in {"S", "U"}:
                    continue
                resolved.append(arr.astype(np.float32))
    return resolved


def _load_grid_arrays(
    grid_group: h5py.Group | None,
    spatial_shape: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """从 HDF5 grid 分组提取坐标数据，备用则生成默认网格。

    Parameters
    ----------
    grid_group:
        .mat 文件中可选的 grid 组。
    spatial_shape:
        预期的 spatial 维。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        解析后的 (grid_x, grid_y)。
    """
    if grid_group is None:
        return _default_grid(spatial_shape)

    normalized: Dict[str, List[np.ndarray]] = {}
    for key in grid_group.keys():
        obj = grid_group[key]
        if isinstance(obj, h5py.Dataset):
            normalized[key.lower()] = _resolve_dataset_values(obj)

    def _first_match(options: Iterable[str]) -> List[np.ndarray] | None:
        for name in options:
            arrays = normalized.get(name)
            if arrays:
                return arrays
        return None

    x_candidates = _first_match(["x", "gridx", "xs", "xx"])
    y_candidates = _first_match(["y", "gridy", "ys", "yy"])

    grid_x = None
    grid_y = None

    if x_candidates:
        grid_x = _reshape_grid_component(x_candidates[0], spatial_shape, horizontal=True)
        if not y_candidates and len(x_candidates) > 1:
            grid_y = _reshape_grid_component(x_candidates[1], spatial_shape, horizontal=False)

    if y_candidates and grid_y is None:
        grid_y = _reshape_grid_component(y_candidates[0], spatial_shape, horizontal=False)

    if grid_x is None or grid_y is None:
        return _default_grid(spatial_shape)
    return grid_x, grid_y


def _move_time_axis(values: np.ndarray, time_count: int) -> np.ndarray:
    """确保 time 轴在数组的最后一个维度以便统一处理。

    Parameters
    ----------
    values:
        包含空间和时间维的张量。
    time_count:
        时间步数。

    Returns
    -------
    np.ndarray
        时间轴已经移到尾部的数组。

    Raises
    ------
    ValueError
        没有找到匹配的时间维度或维度不足。
    """
    if values.ndim < 3:
        raise ValueError("Expected at least 3 dimensions (HxWxT)")
    for axis, size in enumerate(values.shape):
        if size == time_count:
            if axis == values.ndim - 1:
                return values
            return np.moveaxis(values, axis, -1)
    raise ValueError("Could not align time axis with provided timeStamps")


def _iter_structs(node: h5py.Group | h5py.Dataset) -> Iterable[Dict[str, h5py.Dataset | np.ndarray]]:
    """遍历 samples 分组中的每个样本结构，统一返回字典。

    Parameters
    ----------
    node:
        可以是含引用的 Dataset 或直接的 Group。

    Returns
    -------
    Iterable[Dict[str, h5py.Dataset | np.ndarray]]
        每个样本对应的字段映射。
    """
    if isinstance(node, h5py.Dataset):
        for index in np.ndindex(node.shape):
            ref = node[index]
            if isinstance(ref, h5py.Reference):
                group = node.file[ref]
                if not isinstance(group, h5py.Group):
                    raise TypeError("Expected reference to group entry")
                yield {key: group[key] for key in group.keys()}
            else:
                raise TypeError("Encountered non-reference entry in samples dataset")
    elif isinstance(node, h5py.Group):
        datasets = {name: node[name] for name in node.keys() if isinstance(node[name], h5py.Dataset)}
        if not datasets:
            raise RuntimeError("Samples group does not contain dataset fields")
        lengths = {name: ds.shape[0] for name, ds in datasets.items()}
        n_samples = min(lengths.values())

        def _resolve(entry):
            while isinstance(entry, np.ndarray) and entry.size == 1:
                entry = entry.reshape(-1)[0]
            if isinstance(entry, h5py.Reference):
                return node.file[entry]
            return entry

        for index in range(n_samples):
            sample = {}
            for field, ds in datasets.items():
                sample[field] = _resolve(ds[index])
            yield sample
    else:
        raise TypeError("Unsupported samples container type")


def _stack_example(
    coords: Tuple[np.ndarray, np.ndarray],
    params: np.ndarray,
    time_value: float,
    field_slice: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """将坐标、时间、系统参数与当前场值拼接成一个样本对。

    Parameters
    ----------
    coords:
        网格坐标 (grid_x, grid_y)。
    params:
        展平的 A/B 参数。
    time_value:
        当前时间戳。
    field_slice:
        某一时间步对应的场数据。

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        分别是模型输入 (channels, H, W) 和标签 (1, H, W)。
    """
    field_slice = field_slice.astype(np.float32, copy=False)
    base_channels: List[np.ndarray] = [coords[0], coords[1]]
    base_channels.append(np.full_like(field_slice, time_value, dtype=np.float32))
    for value in params.astype(np.float32):
        base_channels.append(np.full_like(field_slice, value, dtype=np.float32))
    stacked_x = np.stack(base_channels, axis=0)
    stacked_y = field_slice[None, ...]
    return stacked_x, stacked_y


def convert_dataset(args: argparse.Namespace) -> None:
    """将 MATLAB Reachable Set 数据集解析为训练/测试 PT 文件。

    Parameters
    ----------
    args:
        命令行参数，决定输入路径、输出目录、分割比例等。
    """
    mat_path = Path(args.mat_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name
    resolution_tag = args.resolution_tag

    x_records: List[np.ndarray] = []
    y_records: List[np.ndarray] = []
    metadata: Dict[str, object] = {}

    with h5py.File(mat_path, "r") as mat_file:
        samples = mat_file["samples"]
        grid_group = mat_file.get("grid")
        coords = None
        spatial_shape = None

        for sample_idx, sample_group in enumerate(_iter_structs(samples)):
            data_stack = np.array(sample_group["dataStack"])
            time_stamps = np.array(sample_group["timeStamps"]).reshape(-1)
            data_stack = _move_time_axis(data_stack, time_stamps.shape[0])

            if spatial_shape is None:
                spatial_shape = data_stack.shape[:-1]
                coords = _load_grid_arrays(grid_group, spatial_shape)
                metadata["grid_shape"] = list(spatial_shape)

            A = np.array(sample_group["A"])
            B = np.array(sample_group["B"])
            params = np.concatenate([A.ravel(), B.ravel()])

            if "params_per_sample" not in metadata:
                metadata["params_per_sample"] = int(params.shape[0])

            for time_idx, time_value in enumerate(time_stamps):
                x_item, y_item = _stack_example(
                    coords=coords,
                    params=params,
                    time_value=float(time_value),
                    field_slice=data_stack[..., time_idx],
                )
                x_records.append(x_item)
                y_records.append(y_item)

                if args.max_pairs is not None and len(x_records) >= args.max_pairs:
                    break

            if args.max_pairs is not None and len(x_records) >= args.max_pairs:
                break

    if not x_records:
        raise RuntimeError("No samples were created from the MAT file."
                           " Double-check the input path and structure.")

    x_tensor = torch.from_numpy(np.stack(x_records))
    y_tensor = torch.from_numpy(np.stack(y_records))

    total_pairs = x_tensor.shape[0]
    rng = np.random.default_rng(args.shuffle_seed)
    ordering = torch.from_numpy(rng.permutation(total_pairs))
    x_tensor = x_tensor.index_select(0, ordering)
    y_tensor = y_tensor.index_select(0, ordering)

    train_count = int(total_pairs * args.train_fraction)
    if train_count == 0 or train_count == total_pairs:
        raise ValueError("train_fraction resulted in empty split. Adjust the value.")

    train_x = x_tensor[:train_count].clone()
    train_y = y_tensor[:train_count].clone()
    test_x = x_tensor[train_count:].clone()
    test_y = y_tensor[train_count:].clone()

    metadata.update(
        {
            "dataset_name": dataset_name,
            "resolution_tag": resolution_tag,
            "input_channels": int(train_x.shape[1]),
            "output_channels": int(train_y.shape[1]),
            "train_samples": int(train_x.shape[0]),
            "test_samples": int(test_x.shape[0]),
        }
    )

    train_path = out_dir / f"{dataset_name}_train_{resolution_tag}.pt"
    test_path = out_dir / f"{dataset_name}_test_{resolution_tag}.pt"
    meta_path = out_dir / f"{dataset_name}_metadata.json"

    torch.save({"x": train_x, "y": train_y, "metadata": metadata}, train_path)
    torch.save({"x": test_x, "y": test_y, "metadata": metadata}, test_path)

    with meta_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    print(f"Saved {train_path} with {train_x.shape[0]} samples")
    print(f"Saved {test_path} with {test_x.shape[0]} samples")
    print(f"Metadata written to {meta_path}")


def build_argparser() -> argparse.ArgumentParser:
    """构建解析脚本所需参数的 argparse.ArgumentParser。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mat-path",
        default="data/ReachableSetDataset/linearSystemValueFunctionData.mat",
        help="Path to the MATLAB .mat file containing the Reachable Set dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/ReachableSetDataset/processed",
        help="Directory where the PT files should be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default="linear_value_function",
        help="Prefix used for the generated PT files.",
    )
    parser.add_argument(
        "--resolution-tag",
        default="default",
        help="Identifier appended to the PT filenames (e.g. spatial resolution).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.9,
        help="Fraction of samples to keep for training.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed used while shuffling examples before the train/test split.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit on the number of (system, time) pairs to export.",
    )
    return parser


def main() -> None:
    """入口函数，解析参数后执行数据转换流程。"""
    parser = build_argparser()
    args = parser.parse_args()
    convert_dataset(args)


if __name__ == "__main__":  # pragma: no cover
    main()
