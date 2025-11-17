"""
加载所有训练状态工件作为模块的片段，
而不限制在默认 Trainer 内部使用。
"""
from typing import Union
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from neuralop.mpu.comm import get_local_rank


def load_training_state(
    save_dir: Union[str, Path],
    save_name: str,
    model: nn.Module,
    optimizer: nn.Module = None,
    scheduler: nn.Module = None,
    regularizer: nn.Module = None,
    map_location: dict = None,
) -> dict:
    """加载模型以及可选的优化器、调度器和正则化器状态。

    此辅助函数镜像了 `save_training_state` 创建的检查点，
    并使得在不与 Trainer 实现紧密耦合的情况下，可以轻松地恢复训练或分析训练好的模型。

    参数
    ----------
    save_dir : Union[str, Path]
        从中加载训练状态（模型、可选的优化器、调度器、正则化器）的目录
    save_name : str
        要加载的模型的名称
    model : nn.Module
        要保存的模型
    optimizer : nn.Module, optional
        要保存的优化器对象，默认为 None
    scheduler : nn.Module, optional
        要保存的调度器对象，默认为 None
    regularizer : nn.Module, optional
        要保存的正则化器对象，默认为 None
    map_location : dict, optional
        映射字典，键为 `{device_from: device_to}`，默认为 None
        字典指示 torch 从 rank `device_from` 上的检查点加载模型
        并将其发送到 `device_to`

    返回
    -------
    训练状态的元组
        ``model, optimizer, scheduler, regularizer, epoch``

    """
    if not map_location:
        if dist.is_initialized():
            # 确保检查点张量被重新映射到本地 GPU
            map_location = {"cuda:0": f"cuda:{get_local_rank()}"}

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    # 可选地加载 epoch，以便恢复的 Trainer 知道从哪里继续
    epoch = None
    manifest_pth = save_dir / "manifest.pt"
    if manifest_pth.exists():
        manifest = torch.load(manifest_pth)
        epoch = manifest.get("epoch")

    if dist.is_initialized():
        # 对于分布式训练，首先在 CPU 上加载检查点，以避免
        # 在多个 CUDA 设备之间复制张量。
        device_id = get_local_rank()
        save_pth = save_dir / f"{save_name}_state_dict.pt"
        model.load_state_dict(
            torch.load(save_pth.absolute().as_posix(), map_location="cpu")
        )
        model = model.to(device=f"cuda:{device_id}")
        torch.cuda.empty_cache()
    else:
        save_pth = save_dir / f"{save_name}_state_dict.pt"
        model.load_state_dict(torch.load(save_pth.absolute().as_posix()))

    # 如果状态存在，则加载优化器
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.pt"
        if optimizer_pth.exists():
            optimizer.load_state_dict(torch.load(optimizer_pth.absolute().as_posix(), map_location=map_location))
        else:
            print(f"警告：请求加载优化器状态，但在 {save_dir} 中不存在已保存的优化器状态。")
    
    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.pt"
        if scheduler_pth.exists():
            scheduler.load_state_dict(torch.load(scheduler_pth.absolute().as_posix(), map_location=map_location))
        else:
            print(f"警告：请求加载调度器状态，但在 {save_dir} 中不存在已保存的调度器状态。")
    
    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.pt"
        if regularizer_pth.exists():
            regularizer.load_state_dict(torch.load(regularizer_pth.absolute().as_posix(), map_location=map_location))
        else:
            print(f"警告：请求加载正则化器状态，但在 {save_dir} 中不存在已保存的正则化器状态。")
    
    return model, optimizer, scheduler, regularizer, epoch


def save_training_state(
    save_dir: Union[str, Path],
    save_name: str,
    model: nn.Module,
    optimizer: nn.Module = None,
    scheduler: nn.Module = None,
    regularizer: nn.Module = None,
    epoch: int = None,
) -> None:
    """将模型、优化器、调度器和正则化器状态持久化到磁盘。

    这镜像了 `load_training_state` 期望的格式，写入一个
    清单映射，该映射捕获了哪些组件可用于恢复
    工作流。

    参数
    ----------
    save_dir : Union[str, Path]
        从中加载训练状态（模型、可选的优化器、调度器、正则化器）的目录
    save_name : str
        要加载的模型的名称
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    manifest = {}

    # 如果模型处于 DDP 模式，则只保存 model.module
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        save_dir.mkdir(exist_ok=True, parents=True)
        model_pth = save_dir / f"{save_name}_state_dict.pt"
        torch.save(model.module.state_dict(), model_pth.as_posix())
    else:
        # 否则保存模型检查点
        model.save_checkpoint(save_dir, save_name)
    # 记录哪些文件存在，以便加载器知道要期望什么
    manifest["model"] = f"{save_name}_state_dict.pt"

    # 跟踪哪些辅助状态存在，以便加载器可以优雅地跳过丢失的文件

    # 如果状态存在，则保存优化器
    if optimizer is not None:
        # 保存优化器状态，以便训练步骤可以继续保持动量/历史记录
        optimizer_pth = save_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_pth)
        manifest["optimizer"] = "optimizer.pt"

    if scheduler is not None:
        # 持久化调度器状态以保持学习率调度连续性
        scheduler_pth = save_dir / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_pth)
        manifest["scheduler"] = "scheduler.pt"

    if regularizer is not None:
        # 记录任何正则化器状态，以便集成或稀疏性惩罚可以恢复
        regularizer_pth = save_dir / "regularizer.pt"
        torch.save(regularizer.state_dict(), regularizer_pth)
        manifest["regularizer"] = "regularizer.pt"

    if epoch is not None:
        manifest["epoch"] = epoch

    torch.save(manifest, save_dir / "manifest.pt")
