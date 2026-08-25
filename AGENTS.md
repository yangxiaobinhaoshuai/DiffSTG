

# Desc
此 repo 为个人 paper 复现 repo 
要求要跑 3 个 random seed


## Agents demands

1. 注意本实验是在租的 GPU 上运行，注意重要数据保存和节约必要成本
2. 注意保存 checkpoints 和 log，保证数据可用于计算 CRPS ，mae， mape， rmse 等 metrics
3. 所有 doc 性描述一律保持简洁扼要
4. 所有对 code 逻辑的改动都要 reproduction notes 里简要写明，方便回溯



## Remote Dev

使用 `scripts/remote_dev.sh` 作为本机到 GPU host 的 facade。约定本机是唯一代码源，远端只负责运行；`push` 会排除 `.git/`、虚拟环境、`data/dataset/`、`output/`，避免覆盖远端数据和训练结果。

常用命令：
- `REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh push`
- `REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh py train.py --data PEMS08`
- `REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh exec nvidia-smi`
- `REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh tail`
- `REMOTE_HOST=gpu REMOTE_DIR='~/runs/DiffSTG' scripts/remote_dev.sh pull-output`

复杂实验组合写成 repo 内脚本后，用 `scripts/remote_dev.sh job <script>` 在远端执行。
