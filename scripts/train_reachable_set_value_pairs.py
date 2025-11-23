"""Training loop for Reachable Set value-to-value pairs (V_t -> V_{t'})."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import wandb

from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.data.datasets.reachable_set import load_reachable_set
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.utils import get_wandb_api_key, count_model_params
from neuralop.mpu.comm import get_local_rank
from neuralop.training import setup, AdamW

from zencfg import make_config_from_cli
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.reachable_config import Default


DEFAULT_DATASET_NAME = "linear_value_t1_t2"
DEFAULT_RESOLUTION = "default"
config_name = "reachable_value_pairs"


# Pre-process CLI args so list-typed fields can be provided as simple scalars
# by users (e.g. `--data.test_resolutions default` or `--data.n_tests 100`).
# This converts those scalar tokens into JSON array literals (e.g. '["default"]')
# so zencfg/pydantic can parse them as lists instead of raising validation errors.
def _preprocess_cli_list_flags():
    flags = [
        "--data.test_resolutions",
        "--data.n_tests",
        "--data.test_batch_sizes",
    ]
    argv = sys.argv
    # operate in-place on argv list
    for idx, token in enumerate(list(argv)):
        for flag in flags:
            # space-separated form: --flag value
            if token == flag:
                if idx + 1 < len(argv):
                    val = argv[idx + 1]
                    # if it's already a JSON array or quoted, leave it
                    if not (val.startswith("[") or val.startswith('"') or val.startswith("'")):
                        try:
                            parsed = json.loads(val)
                        except Exception:
                            parsed = val
                        new_val = json.dumps(parsed if isinstance(parsed, list) else [parsed])
                        argv[idx + 1] = new_val
            # equals form: --flag=value
            elif token.startswith(flag + "="):
                _, eqval = token.split("=", 1)
                if not (eqval.startswith("[") or eqval.startswith('"') or eqval.startswith("'")):
                    try:
                        parsed = json.loads(eqval)
                    except Exception:
                        parsed = eqval
                    new_eq = json.dumps(parsed if isinstance(parsed, list) else [parsed])
                    argv[idx] = f"{flag}={new_eq}"
    sys.argv = argv


# run preprocessing before zencfg parses CLI args
_preprocess_cli_list_flags()


def _load_dataset_metadata(data_dir: Path, dataset_name: str) -> dict | None:
    metadata_path = data_dir / f"{dataset_name}_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def _apply_metadata_overrides(config, metadata: dict | None):
    if metadata is None:
        return
    if hasattr(config.model, "data_channels"):
        config.model.data_channels = metadata.get("input_channels", config.model.data_channels)
    if hasattr(config.model, "out_channels"):
        config.model.out_channels = metadata.get("output_channels", config.model.out_channels)

    if "grid_shape" in metadata:
        config.data.grid_shape = metadata["grid_shape"]
    if "train_samples" in metadata:
        config.data.n_train = min(config.data.n_train, metadata["train_samples"])
    if "test_samples" in metadata and config.data.n_tests:
        # assume single test split mirrors metadata count when present
        config.data.n_tests[0] = min(config.data.n_tests[0], metadata["test_samples"])


config = make_config_from_cli(Default)
if not getattr(config.data, "dataset_name", None) or config.data.dataset_name == Default().data.dataset_name:
    config.data.dataset_name = DEFAULT_DATASET_NAME
if not getattr(config.data, "train_resolution", None):
    config.data.train_resolution = DEFAULT_RESOLUTION
if not getattr(config.data, "test_resolutions", None):
    config.data.test_resolutions = [DEFAULT_RESOLUTION]

metadata = _load_dataset_metadata(Path(config.data.folder).expanduser(), config.data.dataset_name)
_apply_metadata_overrides(config, metadata)

# Coerce some CLI-provided scalar values into lists so downstream
# code (which expects List[...] types) works when the user passes
# a single value like `--data.n_tests 100` or `--data.test_resolutions default`.
def _ensure_list_and_cast(obj, attr_name, cast_type=str):
    val = getattr(obj, attr_name, None)
    if val is None:
        return
    # If zencfg left the value as a string for a scalar, cast it
    if not isinstance(val, list):
        val = [val]
    # Cast individual entries to requested type when possible
    casted = []
    for entry in val:
        try:
            casted.append(cast_type(entry))
        except Exception:
            # fallback: keep original entry
            casted.append(entry)
    setattr(obj, attr_name, casted)

_ensure_list_and_cast(config.data, "n_tests", int)
_ensure_list_and_cast(config.data, "test_resolutions", str)
_ensure_list_and_cast(config.data, "test_batch_sizes", int)

config = config.to_dict()

# Limit CPU thread usage for BLAS/OpenMP-related libraries
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OPENBLAS_NUM_THREADS", "1"))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("NUMEXPR_NUM_THREADS", "1"))

# Also instruct PyTorch to limit its intra/inter op thread pools
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
torch.set_num_interop_threads(int(os.environ.get("TORCH_INTEROP_THREADS", "1")))

device, is_logger = setup(config)

wandb_init_args = None
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    if config.wandb.name:
        wandb_name = config.wandb.name
    else:
        wandb_name = "_".join(
            str(var)
            for var in [
                config_name,
                config.model.n_layers,
                config.model.n_modes,
                config.model.hidden_channels,
                config.model.factorization,
                config.model.rank,
                config.patching.levels,
                config.patching.padding,
            ]
        )
    wandb_init_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.init(**wandb_init_args)

config.verbose = config.verbose and is_logger
if config.verbose:
    print("##### CONFIG #####\n")
    print(config)

data_dir = Path(config.data.folder).expanduser()
# Read DataLoader tuning params from config (fall back to safe defaults)
dl_cfg = config.get("data", {}) if isinstance(config, dict) else {}
num_workers = dl_cfg.get("num_workers", 0)
pin_memory = dl_cfg.get("pin_memory", False)
persistent_workers = dl_cfg.get("persistent_workers", None)

train_loader, test_loaders, data_processor = load_reachable_set(
    data_root=data_dir,
    dataset_name=config.data.dataset_name,
    train_resolution=config.data.train_resolution,
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    n_tests=config.data.n_tests,
    test_resolutions=config.data.test_resolutions,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
    encoding=config.data.encoding,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
)

model = get_model(config)
model = model.to(device)
if config.patching.levels > 0:
    data_processor = MGPatchingDataProcessor(
        model=model,
        in_normalizer=data_processor.in_normalizer,
        out_normalizer=data_processor.out_normalizer,
        padding_fraction=config.patching.padding,
        stitching=config.patching.stitching,
        levels=config.patching.levels,
        use_distributed=config.distributed.use_distributed,
    )
data_processor = data_processor.to(device)
# Enable non-blocking transfers from pinned memory (if pin_memory=True)
try:
    data_processor.set_non_blocking(True)
except Exception:
    pass

if config.distributed.use_distributed:
    train_db = train_loader.dataset
    train_sampler = DistributedSampler(train_db, rank=get_local_rank())
    train_loader = DataLoader(dataset=train_db, batch_size=config.data.batch_size, sampler=train_sampler)
    for (res, loader), batch_size in zip(test_loaders.items(), config.data.test_batch_sizes):
        test_db = loader.dataset
        test_sampler = DistributedSampler(test_db, rank=get_local_rank())
        test_loaders[res] = DataLoader(
            dataset=test_db,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
        )

optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
if config.opt.training_loss == "l2":
    train_loss = l2loss
elif config.opt.training_loss == "h1":
    train_loss = h1loss
else:
    raise ValueError(
        f"Got training_loss={config.opt.training_loss} but expected one of ['l2', 'h1']"
    )
eval_losses = {"h1": h1loss, "l2": l2loss}

if config.verbose:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    print("\n### Beginning Training...\n")
    sys.stdout.flush()

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log=config.wandb.log,
)

if is_logger:
    n_params = count_model_params(model)
    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()
    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)

trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if config.wandb.log and is_logger:
    wandb.finish()

if dist.is_initialized():
    dist.destroy_process_group()
