"""Data loader helpers for the Reachable Set linear value function dataset."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

from torch.utils.data import DataLoader

from .pt_dataset import PTDataset


class ReachableSetDataset(PTDataset):
    """Specialization of :class:`PTDataset` for Reachable Set data exports."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        dataset_name: str,
        train_resolution: str,
        n_train: int,
        n_tests: Sequence[int],
        batch_size: int,
        test_batch_sizes: Sequence[int],
        test_resolutions: Sequence[str],
        encode_input: bool,
        encode_output: bool,
        encoding: str,
        channel_dim: int = 1,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            dataset_name=dataset_name,
            n_train=n_train,
            n_tests=list(n_tests),
            batch_size=batch_size,
            test_batch_sizes=list(test_batch_sizes),
            train_resolution=train_resolution,
            test_resolutions=list(test_resolutions),
            encode_input=encode_input,
            encode_output=encode_output,
            encoding=encoding,
            channel_dim=channel_dim,
            channels_squeezed=False,
        )


def load_reachable_set(
    data_root: Union[str, Path],
    dataset_name: str,
    train_resolution: str,
    n_train: int,
    batch_size: int,
    n_tests: List[int],
    test_resolutions: List[str],
    test_batch_sizes: List[int],
    encode_input: bool = True,
    encode_output: bool = True,
    encoding: str = "channel-wise",
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
) -> tuple[DataLoader, dict, object]:
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    if persistent_workers and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")

    dataset = ReachableSetDataset(
        root_dir=data_root,
        dataset_name=dataset_name,
        train_resolution=train_resolution,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
    )

    dataloader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    train_loader = DataLoader(dataset.train_db, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    test_loaders = {}
    for resolution, test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[resolution] = DataLoader(
            dataset.test_dbs[resolution],
            batch_size=test_bsize,
            shuffle=False,
            **dataloader_kwargs,
        )

    return train_loader, test_loaders, dataset.data_processor
