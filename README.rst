.. image:: https://img.shields.io/pypi/v/neuraloperator
   :target: https://pypi.org/project/neuraloperator/
   :alt: PyPI

.. image:: https://github.com/NeuralOperator/neuraloperator/actions/workflows/test.yml/badge.svg
   :target: https://github.com/neuraloperator/neuraloperator/actions/workflows/test.yml


NeuralOperator 是一个基于 PyTorch 的神经算子学习库，涵盖 Fourier Neural Operator、Tensorized Neural Operator 等能力。此 README 的重点在于说明如何使用 `uv` 管理环境，并描述与 Reachable Set 数据集相关的脚本。

========================
环境管理（使用 uv）
========================

本项目推荐使用 [`uv`](https://uv.run) 作为 Python 环境管理工具，它会解析项目根目录下的 `constraints.txt`（即 `@constraints.txt` 文件）并创建可复现的环境。

1. 安装 `uv`：

.. code-block:: shell

   python -m pip install --upgrade pip
   python -m pip install uv

2. 进入仓库并安装依赖（默认读取 `constraints.txt`）：

.. code-block:: shell

   cd neuraloperator
   uv install --constraints constraints.txt

3. 运行脚本或测试时通过 `uv run` 前缀激活隔离环境，例如：

.. code-block:: shell

   uv run python scripts/train_reachable_set.py --help

4. 若需直接进入 shell，可执行 `uv shell`，命令会自动活跃该虚拟环境。

====================
依赖与约束说明
====================

- `@constraints.txt` 指定了基础依赖的版本（目前包含 `numpy>=1.26.4` 与 `torch_harmonics>=0.7`）。
- `uv install --constraints constraints.txt` 会读取该文件并锁定相关版本，避免不同机器之间产生漂移。
- 若新增依赖，请在 `constraints.txt` 中追加后重新执行该命令以刷新环境。

========================
数据转换脚本
========================

- `@scripts/convert_reachable_set_dataset.py`：将 `data/ReachableSetDataset/linearSystemValueFunctionData.mat` 转换成 NeuralOperator 可训练的 `.pt` 文件，并同时生成 metadata（包括网格形状、参数数量等）。推荐通过 `--dataset-name`、`--resolution-tag`、`--max-pairs` 等参数控制输出。

  .. code-block:: shell

     uv run python scripts/convert_reachable_set_dataset.py \
       --dataset-name linear_value_function \
       --resolution-tag default

- `@scripts/convert_reachable_set_value_pairs.py`：针对 value-to-value 的任务（V_t -> V_{t'}），在 MAT 文件中根据时间戳选取输入与目标对，支持 `--source-time`、`--target-time`、`--train-fraction` 等参数，并会输出 train/test `.pt` 文件与 metadata。

  .. code-block:: shell

     uv run python scripts/convert_reachable_set_value_pairs.py \
       --dataset-name linear_value_t1_t2 \
       --resolution-tag default \
       --source-time 1.0 \
       --target-time 2.0

========================
训练脚本（Reachable Set）
========================

- `@scripts/train_reachable_set.py`：使用包含坐标、时间以及系统参数等完整信息的样本训练线性值函数。默认配置来源于 `config/reachable_config.py`，可通过 CLI 或 YAML 覆盖模型结构、patching、调度器等参数。

  .. code-block:: shell

     uv run python scripts/train_reachable_set.py --config config/reachable_config.py

- `@scripts/train_reachable_set_value_pairs.py`：使用 value-to-value 对训练，适用于 `linear_value_t1_t2` 类数据，流程与常规训练脚本一致但专注于时间差异的映射建模。

  .. code-block:: shell

     uv run python scripts/train_reachable_set_value_pairs.py --config config/reachable_config.py

====================
后续建议
====================

- 在训练前先运行对应的转换脚本生成 `.pt` 文件，metadata 会被用来自动调整 `data_channels`、`grid_shape` 等配置项。
- 每次更新依赖后请重新执行 `uv install --constraints constraints.txt` 以保持虚拟环境一致。
- 在 `uv` 环境下运行测试：`uv run pytest neuralop -v`。


===============
Code of Conduct
===============

All participants are expected to uphold the `Code of Conduct <https://github.com/neuraloperator/neuraloperator/blob/main/CODE_OF_CONDUCT.md>`_ to ensure a friendly and welcoming environment for everyone.


=====================
Citing NeuralOperator
=====================

If you use NeuralOperator in an academic paper, please cite [1]_ ::

   @article{kossaifi2025librarylearningneuraloperators,
      author    = {Jean Kossaifi and
                     Nikola Kovachki and
                     Zongyi Li and
                     David Pitt and
                     Miguel Liu-Schiaffini and
                     Valentin Duruisseaux and
                     Robert Joseph George and
                     Boris Bonev and
                     Kamyar Azizzadenesheli and
                     Julius Berner and
                     Anima Anandkumar},
      title     = {A Library for Learning Neural Operators},
      journal   = {arXiv preprint arXiv:2412.10354},
      year      = {2025},
   }

and consider citing [2]_, [3]_::

   @article{kovachki2021neural,
      author    = {Nikola B. Kovachki and
                     Zongyi Li and
                     Burigede Liu and
                     Kamyar Azizzadenesheli and
                     Kaushik Bhattacharya and
                     Andrew M. Stuart and
                     Anima Anandkumar},
      title     = {Neural Operator: Learning Maps Between Function Spaces},
      journal   = {CoRR},
      volume    = {abs/2108.08481},
      year      = {2021},
   }

   @article{berner2025principled,
      author    = {Julius Berner and
                     Miguel Liu-Schiaffini and
                     Jean Kossaifi and
                     Valentin Duruisseaux and
                     Boris Bonev and
                     Kamyar Azizzadenesheli and
                     Anima Anandkumar},
      title     = {Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning},
      journal   = {arXiv preprint arXiv:2506.10973},
      year      = {2025},
   }


.. [1] Kossaifi, J., Kovachki, N., Li, Z., Pitt, D., Liu-Schiaffini, M., Duruisseaux, V., George, R., Bonev, B., Azizzadenesheli, K., Berner, J., and Anandkumar, A., "A Library for Learning Neural Operators", ArXiV, 2025. doi:10.48550/arXiv.2412.10354.

.. [2] Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., and Anandkumar A., "Neural Operator: Learning Maps Between Function Spaces", JMLR, 2021. doi:10.48550/arXiv.2108.08481.

.. [3] Berner, J., Liu-Schiaffini, M., Kossaifi, J., Duruisseaux, V., Bonev, B., Azizzadenesheli, K., and Anandkumar, A., "Principled Approaches for Extending Neural Architectures to Function Spaces for Operator Learning", arXiv preprint arXiv:2506.10973, 2025. https://arxiv.org/abs/2506.10973.
