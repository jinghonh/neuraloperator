from typing import Any, List, Optional

from zencfg import ConfigBase

from .distributed import DistributedConfig
from .models import ModelConfig, SimpleFNOConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class ReachableSetOptConfig(OptimizationConfig):
    n_epochs: int = 400
    learning_rate: float = 5e-4
    training_loss: str = "h1"
    weight_decay: float = 1e-4
    scheduler: str = "CosineAnnealingLR"
    scheduler_T_max: int = 200


class ReachableSetDatasetConfig(ConfigBase):
    folder: str = "./data/ReachableSetDataset/processed"
    dataset_name: str = "linear_value_function"
    train_resolution: str = "default"
    batch_size: int = 4
    n_train: int = 100
    n_tests: List[int] = [20]
    test_resolutions: List[str] = ["default"]
    test_batch_sizes: List[int] = [4]
    encode_input: bool = True
    encode_output: bool = True
    encoding: str = "channel-wise"


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = SimpleFNOConfig(
        data_channels=4,
        out_channels=1,
        n_modes=[32, 32],
        hidden_channels=64,
        projection_channel_ratio=4,
    )
    opt: OptimizationConfig = ReachableSetOptConfig()
    data: ReachableSetDatasetConfig = ReachableSetDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
