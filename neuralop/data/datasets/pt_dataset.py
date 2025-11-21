from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional

import torch

from .tensor_dataset import TensorDataset
from ..transforms.data_processors import DefaultDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer


class PTDataset:
    """PTDataset is a base Dataset class for our library.
        PTDatasets contain input-output pairs a(x), u(x) and may also
        contain additional information, e.g. function parameters,
        input geometry or output query points.

        datasets may implement a download flag at init, which provides
        access to a number of premade datasets for sample problems provided
        in our Zenodo archive.

    Parameters
    ----------
    root_dir : Union[Path, str]
        root at which to download data files
    dataset_name : str
        prefix of pt data files to store/access
    n_train : int
        number of train instances
    n_tests : List[int]
        number of test instances per test dataset
    batch_size : int
        batch size of training set
    test_batch_sizes : List[int]
        batch size of test sets
    train_resolution : int
        resolution of data for training set
    test_resolutions : List[int]
        resolution of data for testing sets
    encode_input : bool, optional
        whether to normalize inputs in provided DataProcessor,
        by default False
    encode_output : bool, optional
        whether to normalize outputs in provided DataProcessor,
        by default True
    encoding : str, optional
        parameter for input/output normalization. Whether
        to normalize by channel ("channel-wise") or
        by pixel ("pixel-wise"), default "channel-wise"
    input_subsampling_rate : int or List[int], optional
        rate at which to subsample each input dimension, by default None
    output_subsampling_rate : int or List[int], optional
        rate at which to subsample each output dimension, by default None
    channel_dim : int, optional
        dimension of saved tensors to index data channels, by default 1
    channels_squeezed : bool, optional
        If the channels dim is 1, whether that is explicitly kept in the saved tensor.
        If not, we need to unsqueeze it to explicitly have a channel dim.
        Only applies when there is only one data channel, as in our example problems
        Defaults to True
    mmap_files : bool, optional
        If True, use torch.load(..., mmap=True) to avoid materializing the entire PT archive
        in memory. Falls back to the standard load path when unsupported. Defaults to True.

    All datasets are required to expose the following attributes after init:

    train_db: torch.utils.data.Dataset of training examples
    test_db: torch.utils.data.Dataset of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        dataset_name: str,
        n_train: int,
        n_tests: List[int],
        batch_size: int,
        test_batch_sizes: List[int],
        train_resolution: int,
        test_resolutions: List[int],
        encode_input: bool = False,
        encode_output: bool = True,
        encoding="channel-wise",
        input_subsampling_rate=None,
        output_subsampling_rate=None,
        channel_dim=1,
        channels_squeezed=True,
        mmap_files: bool = True,
    ):
        """Initialize the PTDataset.

        See class docstring for detailed parameter descriptions.
        """
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.mmap_files = mmap_files

        # save dataloader properties for later
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes

        train_path = Path(root_dir).joinpath(f"{dataset_name}_train_{train_resolution}.pt")
        train_data = self._load_pt_archive(train_path)

        x_train_raw = train_data["x"]
        y_train_raw = train_data["y"]
        n_train = min(n_train, x_train_raw.shape[0])
        if n_train == 0:
            raise ValueError("Training archive does not contain any samples.")

        x_train = self._prepare_split(
            tensor=x_train_raw,
            n_examples=n_train,
            channel_dim=channel_dim,
            channels_squeezed=channels_squeezed,
            subsampling_rate=input_subsampling_rate,
            dtype=torch.float32,
        )

        y_train = self._prepare_split(
            tensor=y_train_raw,
            n_examples=n_train,
            channel_dim=channel_dim,
            channels_squeezed=channels_squeezed,
            subsampling_rate=output_subsampling_rate,
        )

        del train_data

        # Fit optional encoders to train data
        # Actual encoding happens within DataProcessor
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            input_encoder.fit(x_train)
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                # preserve mean for each channel
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]

            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
        else:
            output_encoder = None

        # Save train dataset
        self._train_db = TensorDataset(
            x_train,
            y_train,
        )

        # create DataProcessor
        self._data_processor = DefaultDataProcessor(
            in_normalizer=input_encoder, out_normalizer=output_encoder
        )

        # load test data
        self._test_dbs = {}
        for res, n_test in zip(test_resolutions, n_tests):
            print(f"Loading test db for resolution {res} with {n_test} samples ")
            test_path = Path(root_dir).joinpath(f"{dataset_name}_test_{res}.pt")
            test_data = self._load_pt_archive(test_path)

            n_test = min(n_test, test_data["x"].shape[0])
            if n_test == 0:
                raise ValueError(f"Test archive for resolution {res} does not contain any samples.")

            x_test = self._prepare_split(
                tensor=test_data["x"],
                n_examples=n_test,
                channel_dim=channel_dim,
                channels_squeezed=channels_squeezed,
                subsampling_rate=input_subsampling_rate,
                dtype=torch.float32,
            )
            y_test = self._prepare_split(
                tensor=test_data["y"],
                n_examples=n_test,
                channel_dim=channel_dim,
                channels_squeezed=channels_squeezed,
                subsampling_rate=output_subsampling_rate,
            )

            del test_data

            test_db = TensorDataset(
                x_test,
                y_test,
            )
            self._test_dbs[res] = test_db

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs

    def _load_pt_archive(self, path: Path):
        """Load a PT archive, enabling mmap when available to avoid full reads."""
        load_kwargs = {"map_location": "cpu"}
        if self.mmap_files:
            load_kwargs["mmap"] = True
        try:
            return torch.load(path.as_posix(), **load_kwargs)
        except TypeError:
            # Older PyTorch builds might not support mmap. Retry without it.
            load_kwargs.pop("mmap", None)
            return torch.load(path.as_posix(), **load_kwargs)

    @staticmethod
    def _normalize_subsampling_rate(rate, n_dims: int, rate_name: str):
        """Convert subsampling specs to a per-dimension list."""
        if not rate:
            rate = 1
        if isinstance(rate, int):
            rates = [rate] * n_dims
        else:
            rates = list(rate)
        if len(rates) != n_dims:
            raise ValueError(
                f"length mismatch between {rate_name} and tensor dimensions: "
                f"expected {n_dims} entries, got {rates}"
            )
        rates = [1 if r in (None, 0) else r for r in rates]
        return rates

    def _prepare_split(
        self,
        tensor: torch.Tensor,
        *,
        n_examples: int,
        channel_dim: int,
        channels_squeezed: bool,
        subsampling_rate,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Apply dtype normalization, optional unsqueeze, and subsampling."""
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        if channels_squeezed:
            tensor = tensor.unsqueeze(channel_dim)
        data_dims = tensor.ndim - 2  # remove batch + channel dims
        rates = self._normalize_subsampling_rate(subsampling_rate, data_dims, "subsampling_rate")

        data_slices = [slice(None, None, rate) for rate in rates]
        indices = [slice(0, n_examples, None)] + data_slices
        indices.insert(channel_dim, slice(None))
        tensor = tensor[tuple(indices)]
        return tensor
