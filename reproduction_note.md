

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
| `output/metrics/*.csv` 最终指标 | `output/forecast/` 预测 pickle |
| `output/log/*.log` 每个 seed 的 epoch 级日志 | `output/log/run_3seeds_*.log` 原始 transcript |
| `output/log/summary_3seeds_*.log` 运行摘要 | `output/model/*.last.dm4stg` 每 epoch 快照 |
| `output/model/*.dm4stg` best-val checkpoint | |

checkpoint 自 2026-08-30（commit `d85584f`）起纳入 git：4.5 MB/个，而训练不是逐位可复现的
（`cudnn.benchmark` 没关、没有 `use_deterministic_algorithms`），**丢了就只能按 ~9 h/seed 重训**。
它也是换采样协议重测的唯一依赖 —— 本轮的 `ddpm`-200 数字全靠它。

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

原始实现每个 epoch 后在完整验证集（PEMS08 = 3548 个窗口）上做一次采样评估。实测该步
620 s，而训练本身只有 75 s —— 单 epoch 12.2 min 里 89% 花在验证上，跑满 300 epoch 需
61 h，超过单 seed 48 h 的墙钟上限（见「实验记录 / 2026-08-28 中止的一轮」）。

**采用的做法：每个 epoch 照常验证，但只评估固定的 512 个等距窗口**（`--val_subset`，
默认 512，0 表示全量）。**不采用「隔 k 个 epoch 验证一次」**——这份代码有三样东西挂在
「本 epoch 是否验证」上，降频会改掉它们的语义：

| | 隔 k 个 epoch 验证 | 每 epoch 验固定子集 |
| --- | --- | --- |
| `scheduler.step()`（在 `if epoch >= start_epoch` 块内，`train.py:462`） | 每 k epoch 才调一次，`ReduceLROnPlateau(patience=5)` 实际变成 5k 个 epoch | 每 epoch 照常，patience 仍是 5 |
| 早停 `epoch - best_epoch > 10` | 11 个 epoch 里只剩约 11/k 次刷新 best 的机会，系统性提前停 | 仍是 11 次 |
| 训练 batch 顺序（`evals()` 开头 `setup_seed()` 重置 CPU RNG，`train_loader` 每 epoch 的 shuffle 种子取自该 RNG） | 被跳过的 epoch 没有重置，batch 顺序改变 | 每 epoch 照常重置，batch 顺序不变 |

即：子集只改变**测量精度**，降频会改变**实验规则**。

子集尺寸是测出来的，不是拍的。方法：用 epoch 135 的 checkpoint 在全量验证集上跑一次
（`ddpm`-200、`n_samples=1`，与 `evals(mode='Val')` 一致），记下每个窗口的误差，之后各
尺寸子集离线计算；每个尺寸取 8 组不同偏移的等距抽样。

| 子集 | MAE 偏置 \|max\| | 偏置 sd | 单 epoch |
| --- | --- | --- | --- |
| 256 | 0.399 | 0.255 | ~2.5 min |
| **512** | **0.206** | **0.114** | **~3.0 min** |
| 1024 | 0.234 | 0.131 | ~4.5 min |
| 2048 | 0.060 | 0.041 | ~7.7 min |
| 全量 3548（MAE 25.0352） | — | — | 12.2 min |

512 → 1024 偏置没有改善：相邻窗口共享 12 个时间步里的 11 个，误差高度自相关，等距抽样
的有效样本量不随 n 线性增长。真要提精度得上 2048，但那会让单 seed 超过 48 h 上限。故取
**512**（也正是上游作者自己注释掉的那行的尺寸，但改成等距抽样——连续切片只覆盖一天中的
一段，PEMS08 误差随时段波动很大）。

实测确认（`evals(mode='Val')` 同一条代码路径，512 窗口）：**91.3 s**，单 epoch 由 12.2 min
降到 **3.0 min**，跑满 300 epoch 为 **14.9 h/seed、三个 seed 44.6 h**，稳在 `MAX_HOURS=48`
的单 seed 上限内。

那个 0.114 是「换一组窗口」时偏置的抖动；实际只用一组固定窗口，该偏置对 epoch 间比较是
常数。训练中期（LR 调度与早停真正做决策的阶段）val MAE 每 epoch 变化 0.1~0.5，远在子集
分辨力之内；后期每 epoch 只变 0.01，子集分辨不了，但那时选哪个 epoch 当 best 对最终模型
的影响也就 0.01。

**最终指标不受影响**：test 阶段仍是全量 test 集 + `n_samples=8` + 样本
均值，与原文 Table 2 的协议一致（原文原话：*"we report MAE and RMSE of the deterministic
forecasting results by averaging S (set to 8) generated samples"*）。副作用：日志里打印的
val MAE 带一个 ≤0.2 的固定偏移，**不能与全量验证的历史日志直接比较**。

> **2026-08-29 更正**：这一段原本写的是「test 阶段仍是全量 test 集 + `ddim_multi`-40 +
> …… 与原文 Table 2 的协议一致」。采样器那半句是错的——`ddim_multi`-40 不是原文的采样器，
> 见「实验记录 / 2026-08-29」。`val_subset` 本身的结论不受影响。

## `train.py` 改动

- 新增 `--val_subset`（默认 512）：每 epoch 的验证只评估固定的等距子集。理由、实测数据与
  为什么不降频，见上一节。metrics CSV 同步新增 `val_subset` 列（`save2file_meta` 会自动为
  旧行补 -1）。
- `--start_epoch` 默认保持 0，且 `scripts/run_3seeds_shutdown.sh` 的 `START_EPOCH` 默认由
  20 改回 **0**：验证变便宜之后它只省约 1 小时，不值得为此保留一条会改变训练轨迹的偏离。
  下面那条关于 `--start_epoch > 0` 的副作用因此不再适用于正式运行（参数本身保留）。
- 更正一条此前记录过头的说法：`model.train()` 那个修复在**本配置下数值上是 no-op**。UGnet
  里只有 `nn.Dropout` 且 `dropout=0.0`，没有 BatchNorm，train/eval 模式对前向没有影响。
  修复仍然保留（改 `dropout > 0` 时才是对的），但不应把它算作与原文的行为差异。
- 日志的 epoch 列改为**整数 epoch index**。原来打的是 `i / len(train_loader) + epoch`，
  每个 epoch 最后一个 batch 的值是 `epoch + 0.999`，格式化后**进位成下一个 epoch**（code
  epoch 135 记成 `136.0`），和 metrics CSV 里的 `best_epoch` 对不上。现在日志里的数就是
  `best_epoch` 里的数；batch 进度 `[i/N]` 只留在终端，不进日志。**2026-08-29 之前的日志仍是
  旧格式，比较时要减 1。**
- 新增 `--start_epoch`（默认 0，不改变原行为）：跳过最前面几个 epoch 的验证。之所以安全，是因为 `Metric.best_metrics['epoch']` 初始值是 `np.inf`，跳过期间不会误存 checkpoint、不会误触发 early stop，scheduler 的 patience 计数也只从真正开始验证的 epoch 起算。这个跳过仅限"训练最前面几个 epoch"，中途/全程降低验证频率仍适用上一节的结论（不能视为等价）。
- 修复了一个原仓库自带的 bug：`evals()` 里会调用 `model.eval()`，但训练循环从未调用 `model.train()`，导致第一次验证之后所有训练 batch 实际上都在 eval 模式下跑（UGnet 里的 `nn.Dropout` 一直失效）。现已在每个 epoch 的训练 batch 循环前显式加上 `model.train()`。
- `--start_epoch > 0` 的副作用（需记录在案）：`evals()` 开头的 `setup_seed()` 会重置全局 CPU RNG，而 `train_loader` 每个 epoch 建迭代器时从该 RNG 取 shuffle 种子，所以原实现里从 epoch 1 起每个 epoch 的 batch 顺序都是同一个。跳过前若干 epoch 的验证 = 跳过这次重置，被跳过的 epoch 反而是正常 shuffle。因此 `--start_epoch 20` 不只是省时间，也轻微改变了训练轨迹，不能声称与原文逐 step 等价。
- **test 采样协议由 `ddim_multi`-40 改为 `ddpm`-200**（`train.py` 的 `for sample_strategy,
  sample_steps in [...]`）。发布代码的默认值不是原文协议：`ddim_multi` 的 `seq` 从
  t=0.8N 起步，而 `beta_end=0.1` 在那里还留着 15.3% 的干净信号、采样器却只喂纯噪声，
  实测值 3.5 MAE（seed 2022：21.67 → 17.90）。完整推导见下方
  「2026-08-29 发布代码的 test 采样器不是原文协议」。改了之后跑完就是可报告的数字，
  不必再用 `scripts/test_from_checkpoint.py` 逐个补测。`sample_steps` 对 `ddpm` 路径是
  装饰性的（`p_sample_loop` 恒定循环 `self.N` 步），填 200 只为让 CSV 行如实反映协议。
- **修 bug**：`evals()` 的 `mode == 'test'` 分支里，写 forecast pickle 前对切片补
  `.clone()`。`torch.cat(samples, dim=0)[:50]` 返回的是**视图**，pickle 会连同底下整个
  storage（全部 3549 个窗口）一起序列化：逻辑数据 9.0 MB，落盘 522.1 MB（samples 463 MB
  + targets 58 MB），4 个文件合计 2.0 GB。`observed_flag` / `evaluate_flag` 是
  `ones_like` 新分配的，所以不受影响。**不能用 `.contiguous()`** —— 该切片本身连续，
  `contiguous()` 会原样返回同一个视图。修后单个 pickle 约 9 MB。



## 新增数据集：PEMS03 / PEMS04（原文没有）

原文只评测 PEMS08、AIR-BJ、AIR-GZ。为论文需要另加 PEMS03/PEMS04（同为 5 分钟粒度的交通流量，
STSGCN 标准四件套里的两个）；AIR 两个是 PM2.5、不是交通，不纳入。**这不是"复现"，是把
baseline 扩展到自己的评测集** —— 没有已发表数字可对照，PEMS08 的 17.92（对原文 17.68）
是唯一的协议锚点，它证明 pipeline 忠实，PEMS03/04 的数字才因此可信。

超参一律沿用原作者发布值（`N=200`、`beta_end=0.1`、`d_h=32`），**不做数据集特定调优** ——
这些值是原作者在 PEMS08/AIR 上搜出来的，搬过来可能低估 DiffSTG，需在论文里列为 limitation。
但"自己的方法调参、baseline 用默认值"是更严重的不对称，必须避免。

### 数据不进 git，自己下

PEMS03/04 的 `flow.npy` / `adj.npy` **不纳入版本控制**（`.gitignore`）—— 这套数据是 STSGCN
的标准发布版，到处都能拿到，没必要让每个 clone 都背 58 MB。开源时在 README 里说明获取方式即可。

需要的原始文件（放进任意目录，比如 `/root/data_raw/<NAME>/`）：

| 数据集 | 文件 | 说明 |
| --- | --- | --- |
| PEMS04 | `PEMS04.npz`, `PEMS04.csv` | csv 的 ID 已是 0..306 |
| PEMS03 | `PEMS03.npz`, `PEMS03.csv`, **`PEMS03.txt`** | txt 是传感器 ID → 节点下标的映射，**缺了图就是错的** |

来源：STSGCN 仓库 <https://github.com/Davidham3/STSGCN> 的数据包，或 IEEE DataPort 上的
同一份发布版。然后：

```bash
uv run --frozen --no-sync python scripts/convert_pems.py \
  --dataset PEMS04 --raw-dir /root/data_raw/PEMS04 --out-dir data/dataset/PEMS04
```

加 `--dry-run` 只校验不写盘。脚本会打印 shape / NaN / 度分布 / 孤立点并断言全部约定，
输出和下表对不上就说明拿错了版本。

**例外**：`data/dataset/PEMS08/` 的两个文件仍然 tracked —— 它们随上游仓库进来
（commit `b723e1e`，早已在 `origin/main` 上），撤掉要重写共享历史，而收益只是让新 clone
少 70 MB，不值。

`scripts/convert_pems.py` 负责把公开发布版转成 `train.py` 要的 `flow.npy` + `adj.npy`，
校验全部内建。两个不做就会静默出错的点：

- **只存 channel 0、且用 float32。** PEMS04 原始是 `(16992, 307, 3)` float64 = 125 MB。
  `CleanDataset.read_data()` 本来就只取 `[:, :, 0]`，channel 1/2 是 occupancy/speed，用不到；
  流量是整数、远在 float32 精确整数范围内，脚本里断言了转换无损。转换后 21 MB / 37 MB。
- **PEMS03 的边表用的是原始传感器 ID（311903~318844），不是节点下标**，必须经 `PEMS03.txt`
  映射，该文件的行序就是 npz 的节点序。注意它是 CRLF 且末尾无换行，`wc -l` 会少数一行
  （报 357，实为 358），按行数校验会误判。PEMS04 的 ID 已是 0..306，无需映射。

邻接矩阵按 PEMS08 的既有约定构建：**binary 0/1、对称、对角线为 0，距离一律丢弃**。
很多 PEMS pipeline 默认给高斯核加权图，套错会喂给 UGnet 一个和已验证结果不同的算子。
PEMS03 边表里有 1 个自环，已丢弃。

| | V | T | 无向边 | 平均度 | 度域 | 孤立点 |
| --- | --- | --- | --- | --- | --- | --- |
| PEMS08（参照） | 170 | 17,856 | 274 | 3.22 | [1, 9] | 0 |
| PEMS04 | 307 | 16,992 | 340 | 2.21 | [1, 7] | 0 |
| PEMS03 | 358 | 26,208 | 546 | 3.05 | [1, 6] | 0 |

split 与 PEMS08 一致的 6:2:2（`train.py` 的 `default_config` 新增两个分支）。三者训练集的
mean/std 也在同一区间（181~230 / 144~156）。

`--is_test` 冒烟测试两个数据集均端到端通过（训练 → 存 → 载入 best → `ddpm`-200 测试 →
CRPS/MIS → 写 CSV，exit=0）。**注意 2-epoch 冒烟的 MAE 现在是 ~3000 而不是历史记录里的 345**：
那是 C2 换协议造成的，同条件下 PEMS08 也是 2702，不是数据问题。

### 实测成本（本机 4090）

`ugnet.py:55` 的 `einsum("knm,bitm->bitkn", Lk, x)` 是稠密邻接乘法、`O(V²)`，但 batch 8 的
训练路径在 V≤358 时 GPU 没吃满，耗时几乎不随 V 变；采样路径（val batch 64、test 8×8=64）
一吃满 `V²` 就现形。所以成本主要由**采样**决定：

| | 训练 B=8 | 采样 B=64 | 显存峰值 | 训练/seed | test/seed | 3 seeds 合计 |
| --- | --- | --- | --- | --- | --- | --- |
| PEMS08 | 1.00× | 1.00× | 0.92 GB | 9.0 h *(实测)* | 1.4 h *(实测)* | 22.9 h *(实测)* |
| PEMS04 | 1.01× | 2.06× | 1.65 GB | ~14 h | ~2.7 h | **~44 h** |
| PEMS03 | 0.98× | 2.45× | 1.92 GB | ~18 h | ~5.0 h | **~61 h** |
| PEMS07 | 1.79× | 7.41× | 4.73 GB | ~48 h | ~16 h | 超单 seed 墙钟上限，不跑 |

建议先跑 PEMS04（便宜 17 h，且是和 PEMS08 配对出现频率最高的一个），提交结果后再开 PEMS03。


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

环境变量：`DATASET=PEMS08` `START_EPOCH=0` `VAL_SUBSET=512` `GPU=0` `SEEDS="2022 2023 2024"`
`MAX_HOURS=48`（单 seed 墙钟上限，0 关闭）`SKIP_SMOKE=1`（跳过冒烟）
`COMMIT_RESULTS=0`（不自动 commit 结果）`NO_SHUTDOWN=1`（跑完不关机）
`SHUTDOWN_GRACE_MINUTES=30`（关机前真等多少分钟，0 表示立即关机）
`SHUTDOWN_ON_EARLY_FAIL=1`（没跑出结果时也关机，默认不关）。

**取消关机**（倒计时开始后）：tmux 里 `Ctrl-C`，或从任何地方
`touch /root/projects/DiffSTG/output/.cancel_shutdown`。
**不要敲 `shutdown -c`** —— 这台机器上它等于立刻关机。

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

# 上一轮（或当前这轮）是成是败，固定路径
cat output/last_run.json

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

## 从 checkpoint 补测

`train.py` 只在训练循环**正常返回之后**才跑 test。被墙钟上限、断电或人为中止打断的运行，
盘上留着一个完好的 best checkpoint，却一个指标都没有。`scripts/test_from_checkpoint.py`
补这个洞：它加载 checkpoint 并调用**同一个** `evals(mode='test')`，指标由同一条代码路径产出，
不是另写一遍。

```bash
uv run --frozen --no-sync python scripts/test_from_checkpoint.py \
  --ckpt output/model/<trial>.dm4stg --data PEMS08 --gpu 0 --seed 2022 \
  --epoch 300 --best_epoch <best> --start_epoch <se>
```

默认协议就是 `train.py` 最终 test 的协议（全量 test 集、`ddim_multi`-40、`n_samples=8`）。
日志与预测 pickle 带 `.testonly` 后缀——`train.py` 用 `mode="w"` 开日志，沿用原 trial 名会
把训练日志截断。`--best_epoch` 等训练侧信息无法从 checkpoint 反推，需显式传入，会原样写进 CSV。

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
- **AutoDL 上没有"定时关机"这回事**：host 的 `/usr/bin/shutdown` 不是 systemd 的 shutdown，
  是 AutoDL 塞进来的三行脚本，**忽略全部参数**直接 kill supervisord 关掉容器。因此
  `shutdown -h +30` 是立刻关机，`shutdown --help` 是立刻关机，连 `shutdown -c` 也是立刻关机。
  2026-08-27 就因此被误关了两次（一次是脚本的 `shutdown -h +1`，一次是 agent 执行
  `shutdown --help` 探测）。**任何人和 agent 都不要在这台机器上敲关机命令**，
  `.claude/hooks/block-poweroff.sh` 已经把 agent 这条路堵死。
- **`output/last_run.json`（固定路径，纳入 git）**：回答"上次为什么关机"。字段有
  `status`（running/preflight_failed/smoke_failed/seed_failed/success）、`shutdown`
  （pending/skipped/cancelled/poweroff）、`exit_status`、`seeds`/`failed_seeds`、
  `started`/`finished`、`code_commit`/`results_commit`、summary 与 raw log 路径、
  checkpoint 列表。**每个阶段边界都重写一次**（不是只在收尾写一次）：2026-08-27 那次
  raw transcript 停在 seed 2022 的中途、已提交的 summary log 停在"seed=2022 started"，
  但 commit message 里却算出了 `failed: 2022 2023 2024` —— 结论行确实丢了，
  丢失的确切机制没查清，所以不再依赖"跑完写一次"这个单点。
  写法是临时文件 + `mv` + `sync`，中断也不会留下半个 JSON。
  `status` 仍是 `running` 且 `finished` 为 null，就代表上次是被外力打断的
  （控制台关机 / OOM / 被 kill / 余额耗尽），脚本自己关机绝不会留下这个组合。
- **登录即见**：`scripts/last_run_banner.sh` 由 `~/.bashrc` source，交互式 shell 一进来
  就打印一行 `last run` 摘要（success 绿、其余红）。开机 ssh 上来第一眼就知道上次是成还是败。
- **`.last.dm4stg` 兜底 checkpoint**：`train.py` 里 best checkpoint 只在验证变好时写，而
  `--start_epoch 20` 意味着第 20 epoch 之前根本不验证 —— 早期崩溃会两手空空。现在每个 epoch
  都额外写一份 `<trial>.last.dm4stg`（临时文件 + `os.replace`，写一半也不会毁掉上一份）。
  它**不参与** best 选择，测试阶段仍然只加载 best checkpoint，复现指标不受影响。
- **文件名缩短**：`trial_name` 原本是把全部参数值用 `+` 拼起来（~90 字符、无标签），
  加上 `model_file_name()` 的尾巴一共 ~135 字符，`ls` 里完全没法读。现在是
  `PEMS08_UGnet_N200_ss200_h32_bs8_lr0.002_se20_e300_s2022_0588d5`：常看的旋钮带标签，
  末尾 6 位是**全部参数**的 blake2b 摘要，所以扫别的超参也不会撞名（`gpu`/`nni` 不进摘要，
  它们只影响在哪跑、不影响算什么）。冒烟跑加 `test_` 前缀。完整参数照旧在
  `output/metrics/DiffSTG.csv`。mae 标注的那份日志副本改名为 `<trial>.mae345.56.log`。
  改名前后 smoke test 的 mae 都是 345.55786，未扰动结果。
- **关机窗口改成脚本自己 sleep**：跑完后先打印 result 状态、summary 与原始 transcript 路径并
  `sync`，然后在脚本里真等 `SHUTDOWN_GRACE_MINUTES`（默认 30，每 5 分钟播报一次剩余时间，
  0 表示立即关机），到点才调一次 `shutdown -h now`。取消有两条路：tmux 里 Ctrl-C，
  或者 `touch output/.cancel_shutdown`（detach 之后也能用）。取消后脚本退出、机器保持开机。

仓库里两个碍眼的文件：`cache.db` 是 autodl-proxy 的 sing-box 写的 bbolt 缓存
（`experimental.cache_file` 没设 `path`，就落在 `proxy_on` 当时的 cwd 里），与本仓库无关；
`*.orig` 是 `git pull --rebase` 冲突留下的备份。两者都已进 `.gitignore`，
根治办法是给 `/etc/autodl-proxy/*.json` 的 `cache_file` 加绝对 `path`。

已知缺口：没有 resume。训练中途崩溃只能从头再来（best checkpoint 仍在盘上）。
考虑到 early stop 会把实际 epoch 数压到远低于 300，且 resume 逻辑本身有写错、
反过来污染复现结果的风险，暂不引入。


# 实验记录

## 2026-08-28 中止的一轮

第一次 3-seed 正式运行，**人为中止，未产出可用于论文的结果**。留档是因为它测出了「全量验证
跑不完」这个结论，以及一个可对比的中途 checkpoint 成绩。

| | |
| --- | --- |
| 配置 | `START_EPOCH=20` `MAX_HOURS=48`，全量验证（3548 窗口），code commit `978714d` |
| 起止 | 2026-08-28 01:53 → 2026-08-29 02:24（seed 2022 单独跑了 25 h） |
| 进度 | 只跑到 seed 2022 的 code epoch 135（日志显示 `136.0`），2023/2024 未开始 |
| 状态 | `output/last_run.json` 里 `status=aborted` |

**为什么中止**：实测单 epoch 12.2 min，其中验证 620 s、训练仅 75 s。跑满 300 epoch 需 61 h，
超过单 seed 48 h 的墙钟上限——也就是说**三个 seed 一个都到不了 test 阶段**，`timeout` 的
SIGTERM 会在 epoch ~250 处把进程杀掉，`output/metrics/DiffSTG.csv` 一行都不会有。这不是运气
问题，是配置必然，所以停下来改成 `--val_subset 512` 重跑。

**中途 checkpoint 的成绩**（epoch 135 的 best checkpoint，用 `scripts/test_from_checkpoint.py`
按原文协议在全量 test 集上补测：`ddim_multi`-40、`n_samples=8`、样本均值）：

| | MAE | RMSE | MAPE | CRPS | MIS |
| --- | --- | --- | --- | --- | --- |
| 本轮 epoch 135（未收敛） | 26.74 | 37.10 | 18.21 | 0.0946 | 441.37 |
| 原文 Table 2 | **17.68** | **27.13** | — | **0.06** | — |
| SpecSTG 复现的 DiffSTG（PEMS08F） | 18.99 | 28.26 | — | 0.0692 | — |

停止时 val MAE 仍在下降（best 连续落在 epoch 130~135），所以这不是收敛值。协议已逐条核对
与原文一致：split 6:2:2（10713/14284/17856）、`T_h=T_p=12`、batch 8、lr 0.002、8 样本均值。
差距是真实的，不是协议错配，需要跑满 300 epoch 后再判断。

> **2026-08-29 更正**：上面这句话是错的。差距**正是**协议错配——发布代码在 test 阶段
> 写死的 `ddim_multi`-40 不是产出原文 Table 2 的采样器。见下一节。

参考：原文 [arXiv 2301.13629](https://arxiv.org/abs/2301.13629)；
SpecSTG [arXiv 2401.08119](https://arxiv.org/abs/2401.08119)，其中明确写过
*"The validation time of DiffSTG is significantly high because it requires sampling and
prediction during validation."*

**保留的产物**（checkpoint 现已进 git，见上方 `output/` 分工表；`output/forecast/` 仍只在 host 上）：

- `output/log/PEMS08_..._se20_e300_s2022_f62798.log` —— 136 个 epoch 的完整曲线
- `output/model/PEMS08_..._se20_e300_s2022_f62798.dm4stg` —— epoch 135 的 best checkpoint
- `output/log/PEMS08_..._f62798.testonly.log` + CSV 中对应行 —— 上表的补测结果

## 2026-08-29 发布代码的 test 采样器不是原文协议（21.67 → 17.90）

**结论：`train.py` 在 test 阶段写死的 `ddim_multi`-40 不是产出原文 Table 2 的采样器。同一个
checkpoint 换成 `ddpm`-200 后 MAE 由 21.67 降到 17.90，落在原文 17.68 的 0.22 以内。
不需要重训，`beta_end` 保持 0.1。**

| | MAE | RMSE | MAPE | CRPS | MIS |
| --- | --- | --- | --- | --- | --- |
| 原文 Table 2 | 17.68 | 27.13 | — | 0.06 | — |
| **本复现 seed 2022 · `ddpm`-200** | **17.90** | **27.31** | 11.32 | **0.0611** | 215.39 |
| SpecSTG 复现的 DiffSTG（PEMS08F） | 18.99 | 28.26 | — | 0.0692 | — |
| 本复现 seed 2022 · `ddim_multi`-40（发布默认） | 21.67 | 31.37 | 14.19 | 0.0753 | 316.81 |

对照的唯一变量是采样器：同一份 `..._s2022_6788e0.dm4stg`（best_epoch 173）、全量 test 集
3549 窗口、`n_samples=8` 样本均值、同一个 `evals(mode='test')` 代码路径、`beta_end` 由
`test_from_checkpoint.py` 从 checkpoint 读出仍是 0.1。

### 机制

`algorithm/diffstg/model.py:107-109` 的 DDIM 时间表把最大步截在 **0.8N**：

```python
seq = (np.linspace(0, np.sqrt(N * 0.8), timesteps) ** 2)  # N=200 → max t = 160，不是 199
x = torch.randn([B, F, V, T])                             # 纯高斯噪声
```

采样从纯噪声起步，却把它标成 t=160；而训练时 `x_160 = 0.153·x0 + 0.988·ε`，模型**期望那里
还带着 15.3% 的干净信号**。紧接着第一步 `x0_t = (xt - et·√(1-at)) / √at`，其中
`at = ᾱ[160] = 0.0235`，把这一步的预测误差放大 **6.5 倍**。`ddpm` 走 `p_sample_loop`，
从 t=199 起步，那里只剩 2.85%。

残留比例完全由 `beta_end` 决定，而原文明说 β_N 是搜出来的（`N ∈ {50,100,200}`、
`β_N ∈ {0.1,0.2,0.3,0.4}`、`M ∈ {40,100}`）：

| N | β_N | √ᾱ[N-1] | √ᾱ[0.8N]（DDIM 起点残留） |
| --- | --- | --- | --- |
| 200 | **0.1** | 0.0285 | **0.1533** ← 发布默认 |
| 200 | 0.2 | 0.0007 | 0.0231 |
| 200 | 0.3 | 0.0000 | 0.0032 |

发布默认正是这一列里 mismatch 最大的点。但换 `ddpm` 就绕开了起点问题，**没有必要为此重训**。

`ddim_multi` 的 `seq` 另有 3 个重复步（`M=40` 时开头四个 0；`M=100` 时 16 个重复），属于
白跑的迭代，与本条结论无关，记录备查。

### 附带确认

- **模型选择没受影响**：训练期 `config.model.sample_strategy` 一直是 `ddpm`，
  `evals(mode='Val')` 走的就是 `p_sample_loop`，best_epoch 是在正确的采样器下选出来的。
- **`--sample_steps` 对 `ddpm` 路径是装饰性的**：`p_sample_loop` 恒定循环 `self.N` 步，
  该参数只写进 CSV，不影响计算。
- **`pos_w` / `pos_d` 全程未被使用**（`ugnet.py:253` 解包后不引用，`is_label_condition`
  无人读）——这与原文一致：原文的 condition 只有 masked 历史、图 `G` 和噪声步 `n`，不含
  time-of-day / day-of-week 嵌入。**不是**差距来源，也不应擅自加上。
- **潜在 bug（当前无害）**：`model.py:124` 的 `x_masked, _, _ = input` 把 `_` 绑定两次，
  于是 `(x_masked, _, _)` 传下去的其实是 `(x_masked, pos_d, pos_d)`，`pos_w` 丢失；且
  `x_masked` 被复制成 `B×n_samples` 而两个 pos 没有。因为 UGnet 两者都不用，现在是 no-op，
  一旦引入时间嵌入会立刻炸。

### 复现命令

```bash
uv run --frozen --no-sync python scripts/test_from_checkpoint.py \
  --ckpt output/model/PEMS08_UGnet_N200_ss200_h32_bs8_lr0.002_se0_e300_s2022_6788e0.dm4stg \
  --data PEMS08 --gpu 0 --seed 2022 --best_epoch 173 --val_subset 512 \
  --sample_strategy ddpm --sample_steps 200 --n_samples 8 --tag ddpm200
```

全量 test 集耗时 9591 s（与 seed 2023 的训练抢卡；独占约 5000 s，`ddim_multi`-40 为 1012 s）。

### 本轮 3-seed 状态

`run_3seeds_PEMS08_start0_20260829-031556`，code commit `44c0d36`，`START_EPOCH=0`
`VAL_SUBSET=512`。当时 CSV 里每个 seed 由 `train.py` 自动写的那一行是 `ddim_multi`-40，
三个 seed 的 `ddpm`-200 都是事后用上面那条命令从 checkpoint 补测的（2026-08-30 补齐）。
`train.py` 此后已改为直接跑 `ddpm`-200，新的运行不再需要补测。

| seed | 训练耗时 | best_epoch | `ddim_multi`-40 MAE | `ddpm`-200 MAE |
| --- | --- | --- | --- | --- |
| 2022 | 08:59:14 | 173 | 21.67 | **17.90** |
| 2023 | 08:48:23 | 150 | 21.08 | **18.05** |
| 2024 | 05:08:33 | 91 | 21.21 | **17.82** |
| **mean ± sd** | — | — | 21.32 ± 0.31 | **17.92 ± 0.11** |

seed 2024 训练耗时短是因为 best_epoch 只到 91，早停触发得早。

三个 seed 全量 test 集、`n_samples=8` 下的完整指标：

| 协议 | MAE | RMSE | MAPE | CRPS | MIS |
| --- | --- | --- | --- | --- | --- |
| **`ddpm`-200（原文协议）** | **17.92 ± 0.11** | **27.34 ± 0.16** | 11.49 ± 0.18 | **0.0611 ± 0.0004** | 215.12 ± 0.36 |
| `ddim_multi`-40（发布默认） | 21.32 ± 0.31 | 31.04 ± 0.29 | 14.20 ± 0.54 | 0.0740 ± 0.0012 | 309.38 ± 7.59 |
| 原文 DiffSTG | 17.68 | 27.13 | — | ~0.06 | — |
| SpecSTG 复现的 DiffSTG | 18.99 | 28.26 | — | 0.0692 | — |

**结论：`ddpm`-200 下复现成功。** MAE 17.92 比原文 17.68 高 0.24（+1.4%），RMSE 高 0.21
（+0.8%），CRPS 与原文报的 0.06 一致；三个 seed 的区间 [17.82, 18.05] 把原文值夹在外面
0.14，属于正常的实现级差异，明显优于 SpecSTG 复现出的 18.99。

### 修正一条此前基于 2 个 seed 的说法

之前记的「种子 sd 是 0.42 MAE（1.95%）」取自 `ddim_multi`-40 下的 seed 2022/2023 两点。
补齐三个 seed 后：`ddim_multi`-40 是 **0.31**，而**原文协议 `ddpm`-200 只有 0.11（0.6%）**。
错采样器把种子间方差放大了约 3 倍 —— 它从 t=0.8N 起步，那里的预测误差被 1/√ᾱ 放大 6.5 倍，
放大的既是偏差也是方差。

所以「方法间距和种子噪声同量级」这个判断要下调：0.11 相对 DiffSTG–PriSTI 的 0.38 已有
3 倍余量，但仍和 PriSTI–SpecSTG 的 0.24 同量级。**当 baseline 用时报 mean±std 的要求不变**，
只是 3 个 seed 对 0.4 以上的差距够用了。注意 n=3 的 sd 本身估得很糙（自由度 2，
95% CI 大致是点估计的 0.5–3 倍），别把 0.11 当成一个精确数字。
