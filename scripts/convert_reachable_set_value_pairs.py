#!/usr/bin/env python
"""Convert reachable set value functions into (t=source -> t=target) PT pairs.

This script builds a dataset where the input is only the value function field at
a chosen time (default t=1) and the target is the field at another time (default
t=2). Time selection is done by matching the provided time values against the
`timeStamps` in the MAT file (nearest match within a tolerance) so it works with
grids that have >2 steps between 1 and 10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch


def _find_time_index(
    time_stamps: np.ndarray, target_time: float, tolerance: float
) -> int:
    """Return the index in time_stamps closest to target_time within tolerance."""
    diffs = np.abs(time_stamps - target_time)
    idx = int(np.argmin(diffs))
    if diffs[idx] > tolerance:
        raise ValueError(
            f"Requested time {target_time} not found within tolerance; "
            f"closest is {time_stamps[idx]} (diff {diffs[idx]:.4g})."
        )
    return idx


def _move_time_axis(values: np.ndarray, time_count: int) -> np.ndarray:
    """Ensure the time dimension is last so indexing matches time_stamps."""
    if values.ndim < 3:
        raise ValueError("Expected at least 3 dimensions (HxWxT)")
    for axis, size in enumerate(values.shape):
        if size == time_count:
            if axis == values.ndim - 1:
                return values
            return np.moveaxis(values, axis, -1)
    raise ValueError("Could not align time axis with provided timeStamps")


def _iter_structs(node: h5py.Group | h5py.Dataset) -> Iterable[Dict[str, object]]:
    """Iterate over sample structs under the `samples` node."""
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


def _stack_value_pair(
    prev_field: np.ndarray, next_field: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack a single-channel input/output pair."""
    prev_field = prev_field.astype(np.float32, copy=False)
    next_field = next_field.astype(np.float32, copy=False)
    return prev_field[None, ...], next_field[None, ...]


def convert_dataset(args: argparse.Namespace) -> None:
    """Convert the MAT file into (V_t -> V_{t+step}) PT files."""
    if args.target_time <= args.source_time and args.target_index is None:
        raise ValueError("target_time must be greater than source_time")

    mat_path = Path(args.mat_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_records: List[np.ndarray] = []
    y_records: List[np.ndarray] = []
    metadata: Dict[str, object] = {}

    with h5py.File(mat_path, "r") as mat_file:
        samples = mat_file["samples"]

        for sample_idx, sample_group in enumerate(_iter_structs(samples)):
            data_stack = np.array(sample_group["dataStack"])
            time_stamps = np.array(sample_group["timeStamps"]).reshape(-1)
            time_count = time_stamps.shape[0]
            data_stack = _move_time_axis(data_stack, time_count)

            source_idx = (
                args.source_index
                if args.source_index is not None
                else _find_time_index(time_stamps, args.source_time, args.time_tolerance)
            )
            target_idx = (
                args.target_index
                if args.target_index is not None
                else _find_time_index(time_stamps, args.target_time, args.time_tolerance)
            )

            if target_idx <= source_idx:
                raise ValueError("target index must be greater than source index")
            if max(source_idx, target_idx) >= time_count:
                continue

            if "grid_shape" not in metadata:
                metadata["grid_shape"] = list(data_stack.shape[:-1])

            x_item, y_item = _stack_value_pair(
                prev_field=data_stack[..., source_idx],
                next_field=data_stack[..., target_idx],
            )
            x_records.append(x_item)
            y_records.append(y_item)

            if "source_time" not in metadata:
                metadata.update(
                    {
                        "source_index": int(source_idx),
                        "target_index": int(target_idx),
                        "source_time": float(time_stamps[source_idx]),
                        "target_time": float(time_stamps[target_idx]),
                        "requested_source_time": float(args.source_time),
                        "requested_target_time": float(args.target_time),
                        "time_tolerance": float(args.time_tolerance),
                    }
                )

            if args.max_pairs is not None and len(x_records) >= args.max_pairs:
                break

        if args.max_pairs is not None and len(x_records) >= args.max_pairs:
            pass

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

    dataset_name = args.dataset_name
    resolution_tag = args.resolution_tag

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
    """Construct the CLI argument parser."""
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
        default="linear_value_t1_t2",
        help="Prefix used for the generated PT files.",
    )
    parser.add_argument(
        "--resolution-tag",
        default="default",
        help="Identifier appended to the PT filenames (e.g. spatial resolution).",
    )
    parser.add_argument(
        "--source-time",
        type=float,
        default=1.0,
        help="Time (in the MAT file) to use as input V_t; matched to nearest timeStamp.",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=2.0,
        help="Time to use as output V_{t'}; matched to nearest timeStamp.",
    )
    parser.add_argument(
        "--source-index",
        type=int,
        default=None,
        help="Optional override: explicit time index used as input (0-based).",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="Optional override: explicit time index used as output (0-based).",
    )
    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=0.05,
        help="Maximum allowed |requested_time - matched_time| when selecting indices.",
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
        help="Optional limit on the number of pairs to export.",
    )
    return parser


def main() -> None:
    """Entry point for CLI usage."""
    parser = build_argparser()
    args = parser.parse_args()
    convert_dataset(args)


if __name__ == "__main__":  # pragma: no cover
    main()
