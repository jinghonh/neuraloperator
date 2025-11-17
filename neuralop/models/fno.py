from functools import partialmethod
from typing import Tuple, List, Union, Literal

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置警告过滤器，使每个警告只显示一次
import warnings

warnings.filterwarnings("once", category=UserWarning)


from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel


class FNO(BaseModel, name="FNO"):
    """N维傅里叶神经算子 (FNO)。

    FNO 使用傅里叶卷积学习在规则网格上离散化的函数空间之间的映射，
    如 [1]_ 中所述。

    FNO 的关键组成部分是其 SpectralConv 层（请参阅
    ``neuralop.layers.spectral_convolution``），它类似于标准的 CNN
    卷积层，但在频域中操作。

    有关 FNO 架构的更深入介绍，请参阅 :ref:`fno_intro`。

    主要参数
    ----------
    n_modes : Tuple[int, ...]
        在傅里叶层中，沿每个维度保留的模式数。
        FNO 的维度由 len(n_modes) 推断。
        n_modes 必须足够大，但要小于 max_resolution//2（奈奎斯特频率）。
    in_channels : int
        输入函数中的通道数。由具体问题确定。
    out_channels : int
        输出函数中的通道数。由具体问题确定。
    hidden_channels : int
        FNO 的宽度（即通道数）。
        这会显著影响 FNO 的参数数量。
        一个好的起点可以是 64，如果需要更强的表达能力，可以增加。
        相应地更新 lifting_channel_ratio 和 projection_channel_ratio，因为它们与 hidden_channels 成正比。
    n_layers : int, optional
        傅里叶层数。默认为 4。

    其他参数
    ---------------
    lifting_channel_ratio : Number, optional
        提升层通道数与隐藏层通道数的比率。
        FNO 提升块中的提升通道数是
        lifting_channel_ratio * hidden_channels（例如，默认为 2 * hidden_channels）。
    projection_channel_ratio : Number, optional
        投影层通道数与隐藏层通道数的比率。
        FNO 投影块中的投影通道数是
        projection_channel_ratio * hidden_channels（例如，默认为 2 * hidden_channels）。
    positional_embedding : Union[str, nn.Module], optional
        在通过 FNO 之前，应用于原始输入最后通道的位置嵌入。
        选项:
        - "grid": 将具有默认设置的网格位置嵌入附加到原始输入的最后通道。
          假设输入在原点为 [0,0,...] 且边长为 1 的网格上离散化。
        - GridEmbeddingND: 直接使用此模块（详见 :mod:`neuralop.embeddings.GridEmbeddingND`）。
        - GridEmbedding2D: 直接用于 2D 情况。
        - None: 不执行任何操作。
        默认为 "grid"。
    non_linearity : nn.Module, optional
        要使用的非线性激活函数模块。默认为 F.gelu。
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        要使用的归一化层。选项："ada_in", "group_norm", "instance_norm", None。默认为 None。
    complex_data : bool, optional
        数据是否为复数值。如果为 True，则初始化复数值模块。默认为 False。
    use_channel_mlp : bool, optional
        是否在每个 FNO 块后使用 MLP 层。默认为 True。
    channel_mlp_dropout : float, optional
        FNO 块中 ChannelMLP 的 Dropout 参数。默认为 0。
    channel_mlp_expansion : float, optional
        FNO 块中 ChannelMLP 的扩展参数。默认为 0.5。
    channel_mlp_skip : Literal["linear", "identity", "soft-gating", None], optional
        在通道混合 MLP 中使用的跳跃连接类型。选项："linear", "identity", "soft-gating", None。
        默认为 "soft-gating"。
    fno_skip : Literal["linear", "identity", "soft-gating", None], optional
        在 FNO 层中使用的跳跃连接类型。选项："linear", "identity", "soft-gating", None。
        默认为 "linear"。
    resolution_scaling_factor : Union[Number, List[Number]], optional
        按层缩放函数域分辨率的因子。
        选项:
        - None: 不缩放。
        - 单个数字 n: 在每层将分辨率缩放 n 倍。
        - 数字列表 [n_0, n_1,...]: 将第 i 层的分辨率缩放 n_i 倍。
        默认为 None。
    domain_padding : Union[Number, List[Number]], optional
        要使用的填充百分比。
        选项:
        - None: 无填充。
        - 单个数字: 沿所有维度使用的填充百分比。
        - 数字列表 [p1, p2, ..., pN]: 沿每个维度使用的填充百分比。
        默认为 None。
    fno_block_precision : str, optional
        执行频谱卷积的精度模式。
        选项: "full", "half", "mixed"。默认为 "full"。
    stabilizer : str, optional
        是否在 FNO 块中使用稳定器。选项: "tanh", None。默认为 None。
        在 `fno_block_precision='mixed'` 的情况下，稳定器能极大地提高性能。
    max_n_modes : Tuple[int, ...], optional
        训练期间在傅里叶域中使用的最大模式数。
        None 表示使用所有的 n_modes。
        整数元组: 在训练期间逐步增加模式数。
        这可以在训练期间动态更新。
    factorization : str, optional
        要使用的 FNO 层权重的张量分解方法。
        选项: "None", "Tucker", "CP", "TT"。
        tltorch 支持的其他分解方法。默认为 None。
    rank : float, optional
        用于分解的张量秩。默认为 1.0。
        当使用 TFNO（即 factorization 不为 None）时，设置为小于 1.0 的浮点数。
        秩为 0.1 的 TFNO 的参数数量大约是密集 FNO 的 10%。
    fixed_rank_modes : bool, optional
        是否不对某些模式进行分解。默认为 False。
    implementation : str, optional
        分解张量的实现方法。
        选项: "factorized", "reconstructed"。默认为 "factorized"。
    decomposition_kwargs : dict, optional
        张量分解的额外关键字参数（请参阅 `tltorch.FactorizedTensor`）。默认为 {}。
    separable : bool, optional
        是否使用可分离的频谱卷积。默认为 False。
    preactivation : bool, optional
        是否使用 resnet 风格的预激活计算 FNO 前向传播。默认为 False。
    conv_module : nn.Module, optional
        用于 FNOBlock 卷积的模块。默认为 SpectralConv。

    示例
    ---------

    >>> from neuralop.models import FNO
    >>> model = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)
    >>> model
    FNO(
    (positional_embedding): GridEmbeddingND()
    (fno_blocks): FNOBlocks(
        (convs): SpectralConv(
        (weight): ModuleList(
            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)
        )
        )
            ... torch.nn.Module printout truncated ...

    参考文献
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()
        self.n_dim = len(n_modes)

        # n_modes 是一个特殊属性 - 请参阅类的属性以了解其底层机制
        # 更新时，更改应反映在 fno 块中
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # 使用相对于隐藏通道的比率初始化提升和投影通道
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision

        ## 位置嵌入
        # 如果 positional_embedding 为 "grid"，则创建一个 GridEmbeddingND 实例
        # 这将在输入中添加坐标信息，帮助模型理解空间关系
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f"错误：期望 {self.n_dim}-d 位置嵌入，但得到 {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"错误：尝试使用 {positional_embedding} 实例化 FNO 位置嵌入，"
                f"期望的是 'grid' 或 GridEmbeddingND"
            )

        ## 域填充
        # 如果指定了 domain_padding，则创建一个 DomainPadding 实例
        # 这有助于处理周期性边界条件或减少傅里叶变换的边界效应
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        ## 分辨率缩放因子
        # 如果提供了缩放因子，则为每一层都设置
        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        ## FNO 核心块
        # 这是模型的核心，包含一系列傅里叶卷积层
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
        )

        ## 提升层 (Lifting Layer)
        # 将输入从低维空间映射到高维隐空间
        # 如果添加了位置嵌入，则输入通道数需要增加
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # 如果指定了 lifting_channels，则使用一个带隐藏层的 Channel-Mixing MLP
        # 否则，使用一个简单的线性层
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # 如果数据是复数，则将提升层转换为复数值 MLP
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        ## 投影层 (Projection Layer)
        # 将高维隐空间中的数据映射回输出空间
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(self, x, output_shape=None, **kwargs):
        """FNO 的前向传播过程

        1. 应用可选的位置编码

        2. 通过提升层将输入发送到高维潜在空间

        3. 对高维中间函数表示应用可选的域填充

        4. 依次应用 `n_layers` 个傅里叶/FNO 层（谱卷积 + 跳跃连接，非线性激活）

        5. 如果应用了域填充，则移除域填充

        6. 将中间函数表示投影到输出通道

        参数
        ----------
        x : tensor
            输入张量

        output_shape : {tuple, tuple list, None}, 默认为 None
            提供为奇数形状输入指定确切输出形状的选项。

            * 如果为 None，则不指定输出形状

            * 如果为元组，则指定 **最后一个** FNO 块的输出形状

            * 如果为元组列表，则指定每个 FNO 块的确切输出形状
        """
        if kwargs:
            warnings.warn(
                f"FNO.forward() 收到了意外的关键字参数: {list(kwargs.keys())}。"
                "这些参数将被忽略。",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # 如果设置了，则附加空间位置嵌入
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        """获取傅里叶模式的数量"""
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        """设置傅里叶模式的数量，并更新 FNO 块"""
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


def partialclass(new_name, cls, *args, **kwargs):
    """创建一个具有不同默认值的新类

    有关示例，请参阅 neuralop/models/sfno.py 中的球形 FNO 类。

    注意
    -----
    一个明显的替代方法是使用 functools.partial
    >>> new_class = partial(cls, **kwargs)

    问题有两方面：
    1. 该类没有名称，因此必须显式设置：
    >>> new_class.__name__ = new_name

    2. 新类将是一个 functools 对象，不能从中继承。

    相反，在这里，我们动态定义一个新类，继承自现有类。
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )


class TFNO(FNO):
    """Tucker 张量化傅里叶神经算子 (TFNO)。

    TFNO 是一个默认启用 Tucker 分解的 FNO。

    它使用权重的 Tucker 分解，通过直接与分解的因子进行收缩，
    使得前向传播非常高效。

    这导致其参数数量仅为等效密集 FNO 的一小部分。

    参数
    ----------
    factorization : str, optional
        张量分解方法，默认为 "Tucker"
    rank : float, optional
        用于分解的张量秩，默认为 0.1。
        秩为 0.1 的 TFNO 的参数数量大约是密集 FNO 的 10%。

    所有其他参数均继承自 FNO，具有相同的默认值。
    有关完整的参数列表，请参阅 FNO 类的文档字符串。

    示例
    --------
    >>> from neuralop.models import TFNO
    >>> # 创建一个具有默认 Tucker 分解的 TFNO 模型
    >>> model = TFNO(n_modes=(12, 12), in_channels=1, out_channels=1, hidden_channels=64)
    >>>
    >>> # 具有显式分解的等效 FNO 模型：
    >>> model = FNO(n_modes=(12, 12), in_channels=1, out_channels=1, hidden_channels=64,
    ...             factorization="Tucker", rank=0.1)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("factorization", "Tucker")
        kwargs.setdefault("rank", 0.1)
        super().__init__(*args, **kwargs)
