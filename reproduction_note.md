

# MISC
1. 复现分支是 feature/reproduction


# Dev Methods

代码用 git 同步：本机和 GPU host 都是 `origin` 的完整 clone，host 上装了 agent，可以直接在 host 上开发和跑实验。

```bash
git push              # 本机：改完推上去
git pull              # host：拉下来再跑
```

`output/` 的分工（见 `.gitignore`）：

| 进 git | 留在 host |
| --- | --- |
| `output/metrics/*.csv` 最终指标 | `output/model/` checkpoint |
| `output/log/*.log` 每个 seed 的 epoch 级日志 | `output/forecast/` 预测 pickle |
| `output/log/summary_3seeds_*.log` 运行摘要 | `output/log/run_3seeds_*.log` 原始 transcript |

原始 transcript 不进 git 是因为 `train.py` 的进度条用 `\r` 且不换行，整份文件基本是一行，git diff 没法看；它的结构化内容都在 summary log 里。

约定代码改动在本机提交、结果在 host 提交，推同一分支前先 `git pull --rebase`。

> `scripts/remote_dev.sh` 保留作为从本机查看 host 的快捷方式（`exec` / `tail` / `tmux` / `pull-output`），
> 但 **`push` 已废弃、不要再用**：它是 `rsync --delete`，会覆盖 host 上还没推送的改动。

# AutoDL Proxy

```bash
# 在 GPU host 上执行
cd <host 上的 repo 目录>
bash autodl-proxy.sh setup          # 按提示输入 client.json 的 HTTPS URL
source /root/.bashrc

proxy_on auto                       # 开启；auto 会按环境自动降级
proxy_off                           # 关闭
proxy_status                        # 查看状态
proxy_test                          # 测试连通性
proxy_refresh                       # 手动刷新订阅
```

> **Tips:** 若 `sing-box` 下载失败，可在本机从官方 Release 下载对应架构的 `.deb`，执行 `source .remote-dev.env && scp -F /dev/null -P "$REMOTE_PORT" <sing-box.deb> "$REMOTE_USER@$REMOTE_HOSTNAME:/tmp/sing-box.deb"`，再到 host 运行 `dpkg -i /tmp/sing-box.deb`，最后重试 `bash autodl-proxy.sh setup`。

> **git 走不走代理：** `origin` 是 SSH（`git@github.com:...`），`proxy_on` 导出的
> `http_proxy`/`https_proxy` 只对 HTTP(S) 生效，除非代理跑在 TUN 模式，否则 `git push`
> 不受影响、可能直接卡住。host 上若推不动，把 remote 换成 HTTPS 让它走代理：
> `git remote set-url origin https://github.com/yangxiaobinhaoshuai/DiffSTG.git`
> （AutoDL 的学术加速也是同理，只覆盖 HTTP(S)）。

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
- `--start_epoch > 0` 的副作用（需记录在案）：`evals()` 开头的 `setup_seed()` 会重置全局 CPU RNG，而 `train_loader` 每个 epoch 建迭代器时从该 RNG 取 shuffle 种子，所以原实现里从 epoch 1 起每个 epoch 的 batch 顺序都是同一个。跳过前若干 epoch 的验证 = 跳过这次重置，被跳过的 epoch 反而是正常 shuffle。因此 `--start_epoch 20` 不只是省时间，也轻微改变了训练轨迹，不能声称与原文逐 step 等价。



# 3-seed 复现

## 启动

`scripts/run_3seeds_shutdown.sh` 顺序跑 seed `2022/2023/2024`，流程是
preflight → smoke test → 三个 seed → 汇总 → commit 结果 → 关机。

在 host 上执行：

```bash
git pull
tmux new -s diffstg-3seeds
bash scripts/run_3seeds_shutdown.sh
```

按 `Ctrl-B D` detach，之后断开 SSH 也不影响运行。
**注意** attach 状态下按 `Ctrl-C` 会打死训练。

环境变量：`DATASET=PEMS08` `START_EPOCH=20` `GPU=0` `SEEDS="2022 2023 2024"`
`MAX_HOURS=48`（单 seed 墙钟上限，0 关闭）`SKIP_SMOKE=1`（跳过冒烟）
`COMMIT_RESULTS=0`（不自动 commit 结果）`NO_SHUTDOWN=1`（跑完不关机）
`SHUTDOWN_ON_EARLY_FAIL=1`（没跑出结果时也关机，默认不关）。

**关机策略**：满足以下任一条件时**保持开机、也不 commit**，好让你直接看到报错——

- preflight 或冒烟失败（什么都没训练）
- 三个 seed 全失败且总耗时不到 1 小时（系统性问题，不是跑到一半崩的）

其余情况一律关机：那才是无人值守的长跑，不关就空烧 GPU。

## 随时查看状态

在 host 上：

```bash
# 进度：attach 回 tmux（Ctrl-B D 离开）
tmux attach -t diffstg-3seeds

# 进度：只看日志，不进 tmux
tail -F output/log/run_3seeds_*.log      # 实时 batch 级进度
tail -n 30 output/log/summary_3seeds_*.log  # 每个 seed 的起止与耗时

# GPU 负载 / 显存 / 进程
nvidia-smi          # 一次性
nvidia-smi -l 5     # 每 5 秒刷新

# 当前指标表（每个 seed 跑完追加一行）
tail -n 5 output/metrics/DiffSTG.csv

# 单个 seed 的 epoch 级曲线（文件名以 +<seed>.log 结尾；
# 冒烟那份含 +True+，正式那份含 +False+）
ls -lt output/log/

# 磁盘
df -h .
```

从本机远程看（只读命令，不写 host 代码）：

```bash
scripts/remote_dev.sh exec nvidia-smi
scripts/remote_dev.sh exec tail -n 5 output/metrics/DiffSTG.csv
scripts/remote_dev.sh tail 'output/log/run_3seeds_*.log'
scripts/remote_dev.sh tmux diffstg-3seeds     # attach；不要带命令，带命令会重跑一遍
```

判断某个 seed 是否收工：`output/log/` 里出现 `[PEMS08]mae<数值>+...+<seed>.log`，
同时 `output/metrics/DiffSTG.csv` 多出一行。

## 结果在哪

| 内容 | 路径 | 进 git |
| --- | --- | --- |
| 最终指标 mae/rmse/mape/crps/mis（含 `seed`、`start_epoch`、`best_epoch` 列） | `output/metrics/DiffSTG.csv` | ✅ |
| 每个 seed 的 epoch 级训练日志 | `output/log/*+<seed>.log` | ✅ |
| 运行摘要（preflight / 每个 seed 起止与耗时 / 汇总） | `output/log/summary_3seeds_*.log` | ✅ |
| 原始 transcript（实时 batch 级进度） | `output/log/run_3seeds_*.log` | ❌ |
| best-val checkpoint | `output/model/*+<seed>*.dm4stg` | ❌ |
| 预测样本 pickle（前 50 条，用于画图/重算概率指标） | `output/forecast/*+<seed>.pkl` | ❌ |

关机前脚本会在 host 上把打 ✅ 的文件 commit 掉，但**不 push**（关机时机器可能没网，
且推送是对外动作，留给你自己决定）。拿指标回本机：

```bash
# host，无卡模式开机即可
git push

# 本机
git pull
```

打 ❌ 的大文件只在 host 上。确实要拉回本机时，开机后用
`scripts/remote_dev.sh pull-output`（落到 `output_remote/`）。

## 代码逻辑改动

- 新增 `--seed`，seed 与 `start_epoch` 写入 metrics CSV；`--seed` 声明在参数表最后，
  保证它是 `trial_name` 的结尾，三个 seed 的 log/checkpoint/forecast 路径互不覆盖。
- `setup_seed()` 里打开了原本注释掉的 `random.seed(seed)`。
- **修 bug**：`torch.load` 加 `weights_only=False`。torch≥2.6 默认 `weights_only=True`，
  而 checkpoint 存的是整个 `nn.Module`，原代码必然加载失败，且异常被 `except` 吞掉后
  会**静默拿最后一个 epoch 的模型去跑 test**——三天训练换来一组错的指标。现改为直接抛错。
- **修 bug**：`calc_mis()` 的 `torch.quantile` 改为沿 batch 维分块调用（`quantile_over_samples`）。
  该函数对整个张量一次调用，val 阶段 `n_samples=1` 约 714 万元素能过，
  test 阶段 `n_samples=8` 约 5712 万元素会触到 `torch.quantile` 的输入上限
  （`quantile() input tensor is too large`）而崩在最后一步。分块沿 `dim=0` 切、
  只对 `dim=1` 归约，逐元素结果与原来完全一致，顺带压低了峰值内存。
- 新增 `--test_batch_size`（默认 8，原为写死的 64）。采样器会把 batch 扩成
  `batch × n_samples`，64×8=512 是验证阶段峰值显存的 8 倍；改成 8×8=64 与验证阶段持平，
  指标不受影响（batch 切分不改变逐样本的指标计算）。
- 新增 `--epoch`（默认 300，与原值一致），便于冒烟测试和封顶。
- `nni.report_final_result()` 包了 try/except：此时结果已全部落盘，不该因为上报失败让整个 seed 记为失败。
- `metric_lst` 为空时直接报错，避免 `min()` 抛一个看不懂的 `ValueError`。

## 脚本为什么这么写

针对"跑了几天没结果"这一类失败：

- **preflight**：先查数据集文件、`torch.cuda.is_available()`、磁盘、`nvidia-smi`，秒级失败。
- **smoke test**：正式跑之前用 `--is_test --start_epoch 0 --epoch 2` 跑一遍完整链路
  （训练 → 存 checkpoint → `torch.load` → ddim_multi test → CRPS/MIS → 写 pickle → 写 CSV → 改名 log）。
  这些代码只在训练结束那一刻才第一次执行，必须提前用几分钟验证掉。
  `--start_epoch` 必须是 0，否则验证一次都不跑、checkpoint 不存在，`torch.load` 那段测不到。
  冒烟结果因 `is_test=True` 而落在不同的 `trial_name` 上，不会污染正式结果。
- **失败不中断**：某个 seed 挂掉后继续跑下一个，最后统一汇报失败 seed，
  而不是 break 掉剩下两个 seed 然后关机。
- **`MAX_HOURS` 墙钟上限**：卡死的训练不会一直烧 GPU。
- **落盘校验**：每个 seed 结束后比对 CSV 行数，退出码为 0 但没写出结果也算失败。
- **关机前 commit**：机器一关就拉不到文件了，所以指标和日志在关机前先落成一个 commit；
  只 `git add` `output/metrics` 和 `output/log`，大文件靠 `.gitignore` 挡住。
  不自动 push：关机时机器未必有网，且推送是对外动作。
- **没结果就不关机**：关机是为了长跑结束后不空烧 GPU，但"什么都没跑出来"时关机只会把
  报错连同机器一起带走，只能重新开机翻日志。所以 preflight/冒烟失败、或三个 seed 在
  1 小时内全挂，都保持开机且不 commit。
- **关机可验证**：`shutdown -h +1` 留一分钟 `shutdown -c` 的窗口；`shutdown` 是异步返回的，
  所以额外 sleep 5 分钟确认真的关掉了，没关掉才退回 `poweroff`。

已知缺口：没有 resume。训练中途崩溃只能从头再来（best checkpoint 仍在盘上）。
考虑到 early stop 会把实际 epoch 数压到远低于 300，且 resume 逻辑本身有写错、
反过来污染复现结果的风险，暂不引入。
