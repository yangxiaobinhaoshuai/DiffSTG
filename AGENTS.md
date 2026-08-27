

# Desc
此 repo 为个人 paper 复现 repo 
要求要跑 3 个 random seed


## Agents demands

1. 注意本实验是在租的 GPU 上运行，注意重要数据保存和节约必要成本
2. 注意保存 checkpoints 和 log，保证数据可用于计算 CRPS ，mae， mape， rmse 等 metrics
3. 所有 doc 性描述一律保持简洁扼要
4. 所有对 code 逻辑的改动都要 reproduction notes 里简要写明，方便回溯



## Dev & Sync

代码用 git 在多设备间同步。本机和 GPU host 都是 `origin` 的完整 clone，host 上装了 agent，可以直接在 host 上开发和跑实验。

- 代码改动在本机提交，结果在 host 提交；推同一分支前先 `git pull --rebase`。
- `output/metrics/*.csv` 和 `output/log/*.log` 纳入 git，随 pull 到任何设备；checkpoint、forecast pickle 和原始 runner transcript 体积大，留在 host（见 `.gitignore`）。
- 复杂实验组合写成 repo 内脚本，在 host 上用 tmux 直接跑。
- 跑完后脚本会在 host 上自动 commit 结果但**不 push**，需要下次开机手动 `git push`。

`scripts/remote_dev.sh` 保留作为从本机查看 host 的快捷方式（`exec` / `tail` / `tmux` / `pull-output`），但 **`push` 已废弃、不要再用**：它是 `rsync --delete`，会覆盖 host 上还没推送的改动。
