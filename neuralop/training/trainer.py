from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings

import torch
from torch.cuda import amp
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
from .training_state import load_training_state, save_training_state


class Trainer:
    """
    一个通用的 Trainer 类，用于在给定的数据集上训练神经算子。

    .. note::
        我们的 Trainer 希望数据集提供键值字典形式的批次，例如：
        ``{'x': x, 'y': y}``，这些键对应于模型和损失函数所期望的参数。
        有关具体细节和示例，请查看 ``neuralop.data.datasets.DarcyDataset``。

    参数
    ----------
    model : nn.Module
        要训练的神经网络模型。
    n_epochs : int
        训练的总轮数。
    wandb_log : bool, 默认为 False
        是否将结果记录到 wandb。
    device : torch.device, 或 str 'cpu' 或 'cuda'
        指定训练设备。
    mixed_precision : bool, 默认为 False
        是否使用 torch.autocast 进行混合精度计算。
    data_processor : DataProcessor 类, 默认为 None
        用于转换数据。如果不是 None，来自加载器的数据首先通过 data_processor.preprocess 进行转换，
        然后从模型获得输出后，再通过 data_processor.postprocess 进行转换。
    eval_interval : int, 默认为 1
        评估模型和记录训练统计数据的频率。
    log_output : bool, 默认为 False
        如果为 True，并且 wandb_log 也为 True，则将输出图像记录到 wandb。
    use_distributed : bool, 默认为 False
        是否使用分布式数据并行 (DDP)。
    verbose : bool, 默认为 False
        是否打印详细信息。
    non_blocking_transfer : bool, 默认为 False
        如果为 True，Trainer 执行的主机到设备的数据复制将使用非阻塞语义（需要固定内存）。
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = "cpu",
        mixed_precision: bool = False,
        data_processor: nn.Module = None,
        eval_interval: int = 1,
        log_output: bool = False,
        use_distributed: bool = False,
        verbose: bool = False,
        non_blocking_transfer: bool = False,
    ):
        """初始化训练器的记录、日志和设备设置。

        准备优化器/调度器持有者、wandb 日志标志、
        设备/混合精度配置，以及一个数据处理器
        引用，用于在训练循环后期调用预处理/后处理。
        """

        self.model = model
        self.n_epochs = n_epochs
        # 仅当有活动的运行时才记录到 wandb
        self.wandb_log = False
        if wandb_available:
            self.wandb_log = wandb_log and wandb.run is not None
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        # 处理 autocast 设备
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            if "cuda" in self.device:
                self.autocast_device_type = "cuda"
            else:
                self.autocast_device_type = "cpu"
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor
        self.non_blocking_transfer = non_blocking_transfer

        # 跟踪用于检查点/恢复的起始轮数
        self.start_epoch = 0

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        eval_modes=None,
        save_every: int = None,
        save_best: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
        max_autoregressive_steps: int = None,
    ):
        """在给定的数据集上训练给定的模型。

        如果提供了设备，模型和数据处理器将在这里加载到设备上。

        参数
        -----------
        train_loader: torch.utils.data.DataLoader
            训练数据加载器
        test_loaders: dict[torch.utils.data.DataLoader]
            测试数据加载器
        optimizer: torch.optim.Optimizer
            训练期间使用的优化器
        scheduler: torch.optim.lr_scheduler
            训练期间使用的学习率调度器
        training_loss: training.losses 函数
            要最小化的成本函数
        eval_losses: dict[Loss]
            在 self.eval() 中使用的损失字典
        eval_modes: dict[str], optional
            从每个加载器的名称到其评估模式的可选映射。

            * 如果是 'single_step'，则预测一个输入-输出对并评估损失。

            * 如果是 'autoregressive'，则使用上一步的输出作为输入，自回归地预测输出，
              步数由批次的 temporal 维度定义。
              这需要特殊批处理的数据，并且数据处理器的 ``.preprocess`` 和
              ``.postprocess`` 都接受 ``idx`` 作为参数。
        save_every: int, optional, 默认为 None
            如果提供，保存检查点的间隔
        save_best: str, optional, 默认为 None
            如果提供，要监控的度量 f"{loader_name}_{loss_name}" 的键，
            并保存评估结果最好的模型。
            覆盖 save_every 并在 eval_interval 上保存。
        save_dir: str | Path, 默认为 "./ckpt"
            如果提供了 save_every 和/或 save_best，则保存训练状态的目录
        resume_from_dir: str | Path, 默认为 None
            如果提供，则从 `resume_from_dir` 中保存的状态恢复训练状态（模型、优化器、正则化器、调度器）
        max_autoregressive_steps : int, 默认为 None
            如果提供，并且数据加载器要以自回归模式进行评估，
            则限制每个 rollout 中执行的自回归步数。

        返回
        -------
        all_metrics: dict
            对于所有 test_loaders，最后一个验证轮次的度量结果字典，
            键为 f"{loader_name}_{loss_name}"

        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            # 当调用者未提供时，默认为 L2 损失
            training_loss = LpLoss(d=2)

        # 如果训练损失在批次中减少，则警告用户
        if hasattr(training_loss, "reduction"):
            if training_loss.reduction == "mean":
                warnings.warn(
                    f"{training_loss.reduction=}. 这意味着损失"
                    "被初始化为在批次维度上取平均。Trainer "
                    "期望损失在批次维度上求和。"
                )

        if eval_losses is None:  # 默认情况下，仅在训练损失上进行评估
            # 至少保留训练损失，以便评估有标量可报告
            eval_losses = dict(l2=training_loss)

        # 累积的 wandb 指标
        self.wandb_epoch_metrics = None

        # 创建默认评估模式
        if eval_modes is None:
            eval_modes = {}

        # 用于检查点的属性
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # 在训练前将核心模块移动到所选设备
        self.model = self.model.to(self.device)

        if self.use_distributed and dist.is_initialized():
            device_id = dist.get_rank()
            # 包装模型以进行多 GPU 同步更新
            self.model = DDP(self.model, device_ids=[device_id], output_device=device_id)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)

        # 确保 save_best 是我们收集的指标
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert (
                self.save_best in metrics
            ), f"错误：期望一个形式为 <loader_name>_<metric> 的指标，但得到 {self.save_best}"
            best_metric_value = float("inf")
            # 当监控一个关键指标时，我们只在该指标改善时保存
            self.save_every = None

        if self.verbose:
            print(f"在 {len(train_loader.dataset)} 个样本上进行训练")
            print(f"在 {[len(loader.dataset) for loader in test_loaders.values()]} 个样本上进行测试"
                f"         在分辨率 {[name for name in test_loaders]} 上。")
            sys.stdout.flush()

        for epoch in range(self.start_epoch, self.n_epochs):
            (
                train_err,
                avg_loss,
                avg_lasso_loss,
                epoch_train_time,
            ) = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time,
            )

            if epoch % self.eval_interval == 0:
                # 定期在所有测试加载器上进行评估并收集指标
                eval_metrics = self.evaluate_all(
                    epoch=epoch,
                    eval_losses=eval_losses,
                    test_loaders=test_loaders,
                    eval_modes=eval_modes,
                    max_autoregressive_steps=max_autoregressive_steps,
                )
                epoch_metrics.update(**eval_metrics)
                # 如果满足条件，则保存检查点
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # 如果设置了 save_every 且未设置 save_best，则保存检查点
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    # 保留定期快照以供恢复
                    self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch 在 train_loader 上训练 self.model
        一个轮次并返回训练指标

        参数
        ----------
        epoch : int
            轮次数
        train_loader : torch.utils.data.DataLoader
            训练样本的数据加载器
        test_loaders : dict
            测试 torch.utils.data.DataLoader 对象的字典

        返回
        -------
        all_errors
            最后一个轮次的所有评估指标的字典
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0

        # 跟踪批次中的训练样本数
        self.n_samples = 0

        for idx, sample in enumerate(train_loader):
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # plateau 调度器期望聚合指标以动态调整
            self.scheduler.step(train_err)
        else:
            # 大多数调度器每轮都步进，而不管指标如何
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(
        self,
        epoch,
        eval_losses,
        test_loaders,
        eval_modes,
        max_autoregressive_steps=None,
    ):
        """evaluate_all 遍历 test_loaders 的整个字典
        以对存储在每个加载器中的整个数据集执行评估。

        参数
        ----------
        epoch : int
            当前训练轮次
        eval_losses : dict[Loss]
            每个对的键为 ``loss_name: loss_obj``。用于每个测试加载器评估的
            完整损失集。
        test_loaders : dict[DataLoader]
            每个测试加载器的键为 ``loader_name: loader``。
        eval_modes : dict[str], optional
            每个测试加载器的键为 ``loader_name: eval_mode``。
            * 如果 ``eval_modes.get(loader_name)`` 没有返回值，
            评估将自动以 ``single_step`` 模式执行。
        max_autoregressive_steps : ``int``, optional
            如果提供，并且其中一个测试加载器的 ``eval_mode == "autoregressive"``，
            则限制每个 rollout 执行的自回归步数。

        返回
        -------
        all_metrics: dict
            为每个加载器收集的评估指标。
        """
        # 在 test_loaders 中的每个加载器上评估并收集指标
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_eval_mode = eval_modes.get(loader_name, "single_step")
            loader_metrics = self.evaluate(
                eval_losses,
                loader,
                log_prefix=loader_name,
                mode=loader_eval_mode,
                max_steps=max_autoregressive_steps,
            )
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics

    def evaluate(
        self,
        loss_dict,
        data_loader,
        log_prefix="",
        epoch=None,
        mode="single_step",
        max_steps=None,
    ):
        """评估模型，在加载器上累积每个损失的指标。

        `errors` 字典通过为每个指标添加 `log_prefix` 前缀来初始化，
        以区分来自不同加载器的指标。

        参数
        ----------
        loss_dict : 函数字典
          每个函数都以一个元组 (prediction, ground_truth) 作为输入
          并返回相应的损失
        data_loader : 要评估的数据加载器
        log_prefix : str, 默认为 ''
            如果不是 ''，则用作输出字典中的前缀
        epoch : int | None
            当前轮次。在记录训练和评估时使用
            默认为 None
        mode : Literal {'single_step', 'autoregression'}
            如果为 'single_step'，则执行标准评估
            如果为 'autoregression'，则循环 `max_steps` 步
        max_steps : int, optional
            自回归 rollout 的最大步数。
            如果为 None，则运行完整的 rollout。
        返回
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        # 确保模型和数据处理器已加载到正确的设备

        self.model = self.model.to(self.device)
        if self.data_processor is not None and self.data_processor.device != self.device:
            # 将处理器张量带到与模型相同的设备
            self.data_processor = self.data_processor.to(self.device)

        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        # 验证评估损失也设置为求和缩减以匹配训练

        # 如果任何评估损失在批次中减少，则警告用户
        for _, eval_loss in loss_dict.items():
            if hasattr(eval_loss, "reduction"):
                if eval_loss.reduction == "mean":
                    warnings.warn(
                        f"{eval_loss.reduction=}. 这意味着损失"
                        "被初始化为在批次维度上取平均。Trainer "
                        "期望损失在批次维度上求和。"
                    )

        self.n_samples = 0
        with torch.no_grad():
            # 在评估期间将梯度钳制为零以节省内存
            for idx, sample in enumerate(data_loader):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                if mode == "single_step":
                    eval_step_losses, outs = self.eval_one_batch(
                        sample, loss_dict, return_output=return_output
                    )
                elif mode == "autoregression":
                    eval_step_losses, outs = self.eval_one_batch_autoreg(
                        sample,
                        loss_dict,
                        return_output=return_output,
                        max_steps=max_steps,
                    )

                for loss_name, val_loss in eval_step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss

        for key in errors.keys():
            errors[key] /= self.n_samples

        # 在最后一个批次上，记录模型输出
        if self.log_output and self.wandb_log:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)

        return errors

    def on_epoch_start(self, epoch):
        """on_epoch_start 在每个训练轮次开始时运行。
        此方法是一个存根，可以在更复杂的情况下被覆盖。

        参数
        ----------

        epoch : int
            轮次的索引

        返回
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample, training_loss):
        """通过模型运行一批输入
           并返回输出的训练损失

        参数
        ----------
        idx : int
            train_loader 中批次的索引
        sample : dict
            包含一批数据的数据字典

        返回
        -------
        loss: float | Tensor
            训练损失的浮点值
        """

        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # 如果不存在预处理器，则将数据加载到设备
            sample = {
                k: v.to(self.device, non_blocking=self.non_blocking_transfer)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        if isinstance(sample["y"], torch.Tensor):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        
        if self.epoch == 0 and idx == 0 and self.verbose and isinstance(out, torch.Tensor):
            print(f"原始输出的形状 {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                # 在相同的设备上下文中进行混合精度评估
                loss += training_loss(out, **sample)
        else:
            loss += training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss

        return loss

    def eval_one_batch(
        self, sample: dict, eval_losses: dict, return_output: bool = False
    ):
        """eval_one_batch 在一个批次上运行推理
        并返回该批次的 eval_losses。

        参数
        ----------
        sample : dict
            数据批次字典
        eval_losses : dict
            命名评估指标的字典
        return_outputs : bool
            是否返回模型输出以供绘图
            默认为 False
        返回
        -------
        eval_step_losses : dict
            键为 "loss_name": step_loss_value 的每个损失名称的字典
        outputs: torch.Tensor | None
            可选地返回批次输出
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # 如果不存在预处理器，则将数据加载到设备
            sample = {
                k: v.to(self.device, non_blocking=self.non_blocking_transfer)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        # 跟踪已处理的真实样本总数
        self.n_samples += sample["y"].size(0)

        out = self.model(**sample)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, **sample)
            eval_step_losses[loss_name] = val_loss

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def eval_one_batch_autoreg(
        self,
        sample: dict,
        eval_losses: dict,
        return_output: bool = False,
        max_steps: int = None,
    ):
        """eval_one_batch 在一个批次上运行推理
        并返回该批次的 eval_losses。

        参数
        ----------
        sample : dict
            数据批次字典
        eval_losses : dict
            命名评估指标的字典
        return_outputs : bool
            是否返回模型输出以供绘图
            默认为 False
        max_steps: int
            要展开的时间步数
            通常是完整的轨迹长度
            如果 max_steps 为 none，则运行到完整长度

            .. note::
                如果未提供 ``max_steps`` 的值，则必须提供一个 data_processor
                来处理展开逻辑。
        返回
        -------
        eval_step_losses : dict
            键为 "loss_name": step_loss_value 的每个损失名称的字典
        outputs: torch.Tensor | None
            可选地返回批次输出


        """
        eval_step_losses = {loss_name: 0.0 for loss_name in eval_losses.keys()}
        # eval_rollout_losses = {loss_name: 0. for loss_name in eval_losses.keys()}

        t = 0
        if max_steps is None:
            max_steps = float("inf")

        # 仅增加一次样本计数（每个批次，而不是每个展开步骤）
        sample_count_incr = False

        while sample is not None and t < max_steps:
            if self.data_processor is not None:
                sample = self.data_processor.preprocess(sample, step=t)
            else:
                # 如果不存在预处理器，则将数据加载到设备
                sample = {
                    k: v.to(self.device, non_blocking=self.non_blocking_transfer)
                    for k, v in sample.items()
                    if torch.is_tensor(v)
                }

            if sample is None:
                break

            # 仅增加一次样本计数
            if not sample_count_incr:
                self.n_samples += sample["y"].shape[0]
                sample_count_incr = True

            out = self.model(**sample)

            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample, step=t)

            for loss_name, loss in eval_losses.items():
                step_loss = loss(out, **sample)
                eval_step_losses[loss_name] += step_loss

            t += 1
        # 在最终展开的所有步骤上取平均
        for loss_name in eval_step_losses.keys():
            eval_step_losses[loss_name] /= t

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def log_training(
        self,
        epoch: int,
        time: float,
        avg_loss: float,
        train_err: float,
        avg_lasso_loss: float = None,
        lr: float = None,
    ):
        """记录单个训练轮次结果的基本方法。


        参数
        ----------
        epoch: int
        time: float
            轮次的训练时间
        avg_loss: float
            每个样本的平均 train_err
        train_err: float
            整个轮次的训练误差
        avg_lasso_loss: float
            来自正则化器的平均 lasso 损失，可选
        lr: float
            当前轮次的学习率
        """
        # 累积要记录到 wandb 的信息
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"

        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=False)

    def log_eval(self, epoch: int, eval_metrics: dict):
        """log_eval 将所有测试加载器上的评估输出
        记录到 stdout 和 wandb

        参数
        ----------
        epoch : int
            当前训练轮次
        eval_metrics : dict
            评估期间收集的指标
            每个 test_loader 的键为 f"{test_loader_name}_{metric}"

        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.4f}, "
            if self.wandb_log:
                values_to_log[metric] = value

        msg = f"评估: " + msg[:-2]  # 去掉最后的逗号和空格
        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=True)

    def resume_state_from_dir(self, save_dir):
        """
        从 `neuralop.training.save_training_state` 创建的 save_dir 恢复训练

        参数
        ------
        save_dir: Union[str, Path]
            保存训练状态的目录
            (参见 neuralop.training.training_state)
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # 检查检查点目录以决定加载哪个模型文件
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        else:
            raise FileNotFoundError(
                "错误：resume_from_dir 期望一个名为 model.pt 或 best_model.pt 的模型\
                                        状态字典。"
            )
        # 返回模型，如果提供则加载其他模块
        
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.regularizer,
            resume_epoch,
        ) = load_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=self.model,
            optimizer=self.optimizer,
            regularizer=self.regularizer,
            scheduler=self.scheduler,
        )

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose:
                    print(f"Trainer 从轮次 {resume_epoch} 恢复")

    def checkpoint(self, save_dir):
        """checkpoint 将当前训练状态保存
        到一个目录中，以便稍后恢复。仅在第一个 GPU 上保存
        训练状态。
        参见 neuralop.training.training_state

        参数
        ----------
        save_dir : str | Path
            保存训练状态的目录
        """
        if comm.get_local_rank() == 0:
            # 只有 rank 0 写入文件以避免在 DDP 中重复
            if self.save_best is not None:
                save_name = "best_model"
            else:
                save_name = "model"
            save_training_state(
                save_dir=save_dir,
                save_name=save_name,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                regularizer=self.regularizer,
                epoch=self.epoch,
            )
            if self.verbose:
                print(f"[Rank 0]: 已将训练状态保存到 {save_dir}")
