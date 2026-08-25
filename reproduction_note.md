

# MISC
1. 复现分支是 feature/reproduction


# Dev Methods

使用 `scripts/remote_dev.sh` 做本机开发、远端 GPU 执行：

```bash
REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh push
REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh py train.py --data PEMS08
REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh exec nvidia-smi
REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh tail
REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh pull-output
```

可把配置写进 `.remote-dev.env`：

```bash
REMOTE_HOST=gpu
REMOTE_DIR=~/runs/DiffSTG
PY_RUNNER=uv run python
```

注意：`push` 使用 `rsync --delete` 同步代码，但默认排除 `data/dataset/` 和 `output/`；数据和训练结果应保留在远端，结果用 `pull-output` 拉回本机分析。简单换入口用 `py`，复杂实验组合写成 repo 内脚本后用 `job <script>`。

# 与原文的差异

## 训练期验证开销

原始实现每个 epoch 后都会在验证集上执行一次完整采样评估。该步骤耗时较高，并且会影响学习率调度、best checkpoint 选择和 early stopping，因此不能简单跳过后仍声称结果完全等价。

复现时区分两类运行：
- 调试运行：可降低验证频率或使用少量 batch，目的是检查代码、数据和日志链路。
- 正式运行：保持完整验证与最终 test 流程，用于报告可对比指标。

如需加速正式实验，应明确记录验证频率、checkpoint 选择规则和最终 test 配置，避免把工程加速与论文指标复现混在一起。

## `train.py` 改动

- 新增 `--start_epoch`（默认 0，不改变原行为）：跳过最前面几个 epoch 的验证。之所以安全，是因为 `Metric.best_metrics['epoch']` 初始值是 `np.inf`，跳过期间不会误存 checkpoint、不会误触发 early stop，scheduler 的 patience 计数也只从真正开始验证的 epoch 起算。这个跳过仅限"训练最前面几个 epoch"，中途/全程降低验证频率仍适用上一节的结论（不能视为等价）。
- 修复了一个原仓库自带的 bug：`evals()` 里会调用 `model.eval()`，但训练循环从未调用 `model.train()`，导致第一次验证之后所有训练 batch 实际上都在 eval 模式下跑（UGnet 里的 `nn.Dropout` 一直失效）。现已在每个 epoch 的训练 batch 循环前显式加上 `model.train()`。
