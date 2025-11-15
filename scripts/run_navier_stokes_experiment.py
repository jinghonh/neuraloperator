#!/usr/bin/env python3
"""Train and visualize a 2D Navier-Stokes experiment with NeuralOperator.

This utility script downloads the Navier-Stokes dataset (if needed), trains a
lightweight Fourier Neural Operator (FNO) model, evaluates it on the held-out
set, and saves diagnostic visualizations/metrics for quick inspection.

Dataset Download:
    The script will automatically download the dataset from Zenodo if needed.
    If download is slow, you can:
    
    1. Use --skip-download to only download missing files
    2. Use --no-download and manually download from:
       https://zenodo.org/records/12825163
       
       Place the downloaded .tgz files in the --data-root directory.
       The script expects files named: nsforcing_{resolution}.tgz
       (e.g., nsforcing_128.tgz, nsforcing_1024.tgz)
    
    3. The download function has been optimized with:
       - Larger chunk size (1MB) for faster downloads
       - Progress display with speed and ETA
       - Better error handling
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from neuralop import H1Loss, LpLoss, Trainer
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and visualize a 2D Navier-Stokes experiment."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="~/data/navier_stokes",
        help="Directory that stores the Navier-Stokes .pt files (will download if missing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/navier_stokes",
        help="Directory used to store plots and metrics.",
    )
    parser.add_argument(
        "--train-resolution",
        type=int,
        default=128,
        help="Spatial resolution for the training split (allowed: 128 or 1024).",
    )
    parser.add_argument(
        "--test-resolution",
        type=int,
        default=128,
        help="Spatial resolution for evaluation (allowed: 128 or 1024).",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=1024,
        help="Number of training samples to load from disk.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=128,
        help="Number of evaluation samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="Batch size used for evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=100,
        help="Scheduler step size for StepLR.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.5,
        help="Scheduler gamma for StepLR.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        default=(16, 16),
        help="Fourier modes along spatial dimensions.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Hidden channel width inside the FNO blocks.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of Fourier layers.",
    )
    parser.add_argument(
        "--projection-channel-ratio",
        type=int,
        default=2,
        help="Multiplier for the projection layers inside the FNO.",
    )
    parser.add_argument(
        "--samples-to-plot",
        type=int,
        default=2,
        help="How many evaluation samples to visualize.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run on. Use 'auto' to prefer CUDA when available.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training if supported by the device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading dataset if files already exist. "
        "If files are missing, the script will still attempt to download.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Never download dataset. Will fail if files are missing. "
        "Useful when you want to manually download the dataset.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is visible.")
    if device.type == "mps" and not getattr(torch.backends, "mps", None).is_available():
        raise RuntimeError("MPS requested but it is not available.")
    return device


def make_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model(
    n_modes: Sequence[int],
    hidden_channels: int,
    n_layers: int,
    projection_channel_ratio: int,
    in_channels: int,
    out_channels: int,
    device: torch.device,
) -> FNO:
    model = FNO(
        n_modes=tuple(n_modes),
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        projection_channel_ratio=projection_channel_ratio,
        n_layers=n_layers,
    )
    return model.to(device)


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    l2loss: LpLoss,
    h1loss: H1Loss,
) -> Dict[str, float]:
    model.eval()
    agg = {"l2": 0.0, "h1": 0.0, "num_batches": 0}
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            preds = model(x)
            agg["l2"] += l2loss(preds, y).item()
            agg["h1"] += h1loss(preds, y).item()
            agg["num_batches"] += 1
    num_batches = max(agg["num_batches"], 1)
    return {
        "l2": agg["l2"] / num_batches,
        "h1": agg["h1"] / num_batches,
        "num_batches": agg["num_batches"],
    }


def _field_to_magnitude(field: np.ndarray) -> np.ndarray:
    if field.ndim < 3:
        raise ValueError(f"Expected a (C, H, W) field but got shape {field.shape}.")
    if field.shape[0] == 1:
        return field[0]
    return np.linalg.norm(field[:2], axis=0)


def save_prediction_grid(
    model: torch.nn.Module,
    data_processor,
    dataset,
    sample_indices: Iterable[int],
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    sample_indices = list(sample_indices)
    n_rows = len(sample_indices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(4 * 4, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(sample_indices):
        data = dataset[idx]
        data = data_processor.preprocess(data, batched=False)
        x = data["x"].to(device)
        y = data["y"].to(device)
        with torch.no_grad():
            pred = model(x.unsqueeze(0)).squeeze(0)

        x_np = x.cpu().numpy()
        y_np = y.squeeze().cpu().numpy()
        pred_np = pred.cpu().numpy()
        err_np = np.abs(y_np - pred_np)

        panels = [
            ("Input", _field_to_magnitude(x_np)),
            ("Target", _field_to_magnitude(y_np)),
            ("Prediction", _field_to_magnitude(pred_np)),
            ("|Target - Prediction|", _field_to_magnitude(err_np)),
        ]

        for col, (title, image) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(image, origin="lower", cmap="viridis")
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("Navier-Stokes Predictions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_error_histogram(
    errors: List[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=30, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Batch L2 Loss")
    ax.set_ylabel("Frequency")
    ax.set_title("Navier-Stokes Error Distribution")
    ax.axvline(np.mean(errors), color="crimson", linestyle="--", label="Mean")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def collect_batch_errors(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    l2loss: LpLoss,
    device: torch.device,
) -> List[float]:
    errors: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            preds = model(x)
            errors.append(l2loss(preds, y).item())
    return errors


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root).expanduser()
    output_dir = make_output_dir(Path(args.output_dir).expanduser())
    device = resolve_device(args.device)

    print(f"Using device: {device}")
    print(f"Data root: {data_root}")
    print(f"Outputs will be saved to: {output_dir}")

    # 处理下载选项
    download = not args.no_download
    if args.no_download:
        print("Note: --no-download is set. Will not download dataset.")
        print("      If files are missing, the script will fail.")
        print("      To manually download, visit: https://zenodo.org/records/12825163")
    elif args.skip_download:
        print("Note: --skip-download is set. Will only download if files are missing.")

    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        n_train=args.n_train,
        n_tests=[args.n_test],
        batch_size=args.batch_size,
        test_batch_sizes=[args.test_batch_size],
        data_root=data_root,
        train_resolution=args.train_resolution,
        test_resolutions=[args.test_resolution],
        encode_input=True,
        encode_output=True,
        num_workers=args.num_workers,
        download=download,
    )

    sample_batch = next(iter(train_loader))
    in_channels = sample_batch["x"].shape[1]
    out_channels = sample_batch["y"].shape[1]
    print(f"Detected {in_channels} input channels and {out_channels} output channels.")

    data_processor = data_processor.to(device)

    n_modes = tuple(args.n_modes)
    model = build_model(
        n_modes=n_modes,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        projection_channel_ratio=args.projection_channel_ratio,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
    )

    n_params = count_model_params(model)
    print(f"Model parameters: {n_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.lr_gamma,
    )

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    trainer = Trainer(
        model=model,
        n_epochs=args.epochs,
        device=device,
        data_processor=data_processor,
        mixed_precision=args.mixed_precision,
        eval_interval=max(1, min(args.epochs, 5)),
        use_distributed=False,
        verbose=True,
        wandb_log=False,
    )

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=h1loss,
        eval_losses={"h1": h1loss, "l2": l2loss},
    )

    test_loader = test_loaders[args.test_resolution]
    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        l2loss=l2loss,
        h1loss=h1loss,
    )
    metrics_path = output_dir / "navier_stokes_metrics.json"
    metrics_payload = {
        "metrics": metrics,
        "n_parameters": n_params,
        "train_resolution": args.train_resolution,
        "test_resolution": args.test_resolution,
        "epochs": args.epochs,
        "n_train": args.n_train,
        "n_test": args.n_test,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Saved evaluation metrics to {metrics_path}")

    dataset = test_loader.dataset
    sample_indices = list(range(min(args.samples_to_plot, len(dataset))))
    predictions_png = output_dir / f"navier_stokes_predictions_{args.test_resolution}.png"
    save_prediction_grid(
        model=model,
        data_processor=data_processor,
        dataset=dataset,
        sample_indices=sample_indices,
        device=device,
        output_path=predictions_png,
    )
    print(f"Saved qualitative predictions to {predictions_png}")

    batch_errors = collect_batch_errors(
        model=model,
        loader=test_loader,
        l2loss=l2loss,
        device=device,
    )
    histogram_png = output_dir / f"navier_stokes_error_hist_{args.test_resolution}.png"
    save_error_histogram(
        errors=batch_errors,
        output_path=histogram_png,
    )
    print(f"Saved error histogram to {histogram_png}")


if __name__ == "__main__":
    main()
