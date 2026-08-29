# TODO

开机后按 A → B → C → D → E 的顺序做。A 是拿数字,B 是防实例回收,C/D 是收拾技术债。

> **背景**:`train.py` 自动写进 CSV 的那一行用的是 `ddim_multi`-40(≈21.x),**不是原文协议**;
> 要报告的数字是 `ddpm`-200(seed 2022 已实测 17.90)。完整推导见
> `reproduction_note.md` → 「实验记录 / 2026-08-29 发布代码的 test 采样器不是原文协议」。
>
> **时机**:3-seed 那轮预计 2026-08-30 04:00–04:30 之间跑完 seed 2024,然后 30 分钟后自动关机。
> 关机前脚本会自动 commit `output/metrics` 和 `output/log`(**不含** checkpoint)。
> AutoDL 关机保留文件系统,关机本身不丢数据;真正的风险是**实例被回收**,所以 B 组要尽早做完。
> B 组只需要**无卡模式开机**,不烧 GPU。


## A. 补齐三个 seed 的 `ddpm`-200 指标

- [x] ~~**A1. 确认盘上的东西还在**~~ —— 2026-08-30 02:53 已确认：`status=success`、
      `shutdown=poweroff`、`failed_seeds=[]`；三个 best checkpoint 都在（各 4449311 B），
      关机后手动重开机,文件完好。

  ```bash
  python3 -c "import json;d=json.load(open('output/last_run.json'));print(d['status'],d['shutdown'],d['failed_seeds'])"
  # 期望: success poweroff []
  ls -la output/model/*.dm4stg
  tail -n 20 output/log/summary_3seeds_PEMS08_start0_20260829-031556.log
  ```

  `output/model/` 与 `output/forecast/` 是 gitignored、只在 host 上。
  若 checkpoint 不在了,A 组全部作废,只能重训 —— 这也正是 B 组存在的理由。

- [ ] **A2. 补测 seed 2023 与 2024 的 `ddpm`-200**(独占 GPU 每个约 1.4 h,串行约 2.8 h)

  先取 seed 2024 的 `best_epoch`(2023 是 **150**,2022 是 173):

  ```bash
  python3 -c "
  import csv
  for r in csv.DictReader(open('output/metrics/DiffSTG.csv')):
      if r['is_test']=='False': print(r['seed'], r['best_epoch'], r['model.sample_strategy'], r['mae'])
  "
  ```

  然后逐个跑(`--best_epoch` 只写进 CSV,不参与计算,但要填对):

  ```bash
  uv run --frozen --no-sync python scripts/test_from_checkpoint.py \
    --ckpt output/model/PEMS08_UGnet_N200_ss200_h32_bs8_lr0.002_se0_e300_s2023_13a8c7.dm4stg \
    --data PEMS08 --gpu 0 --seed 2023 --best_epoch 150 --val_subset 512 \
    --sample_strategy ddpm --sample_steps 200 --n_samples 8 --tag ddpm200

  uv run --frozen --no-sync python scripts/test_from_checkpoint.py \
    --ckpt output/model/PEMS08_UGnet_N200_ss200_h32_bs8_lr0.002_se0_e300_s2024_51bc02.dm4stg \
    --data PEMS08 --gpu 0 --seed 2024 --best_epoch <填上一步查到的> --val_subset 512 \
    --sample_strategy ddpm --sample_steps 200 --n_samples 8 --tag ddpm200
  ```

  **别并行跑**:抢卡会把单次从 ~5000 s 拖到 ~9600 s,串行反而更快。

- [ ] **A3. 汇总三个 seed 的 mean±std**,填进 `reproduction_note.md` 末尾那张表
      (`ddpm`-200 行,以及 `ddim_multi`-40 行作为「发布默认值」对照)。

  对照基准:原文 17.68 / 27.13 / CRPS 0.06;SpecSTG 复现 18.99 / 28.26 / 0.0692。
  已知 `ddim_multi`-40 下 seed 2022/2023 的样本 sd 是 **0.42 MAE**(1.95%),
  而文献里 DiffSTG–PriSTI 只差 0.38、PriSTI–SpecSTG 只差 0.24 —— **方法间距和种子噪声同量级**。
  所以当 baseline 用时:自己的方法也必须多 seed、报 mean±std;若提升在 0.5 MAE 以内,3 个 seed 可能不够。


## B. 防实例回收:把 best checkpoint 纳入 git

理由(三条,按分量排):

1. **A 组就是这个场景**。换采样器重测靠的全是 checkpoint:有它 1.4 h,没它 9 h/seed 重训。
2. **训练不是逐位可复现的**。`train.py:50` 只设了 `cudnn.deterministic = True`,没关
   `cudnn.benchmark`,也没有 `torch.use_deterministic_algorithms(True)`。换 GPU / 驱动 /
   torch 版本,同一个 seed 也不会给出同一份权重 —— **checkpoint 丢了是永久丢了**,
   这点和日志、CSV 有本质区别。
3. **成本可忽略**。4.45 MB/个 × 3 = 13.3 MB;仓库现在打包才 20.5 MB,而 `flow.npy`
   一个 70 MB 的文件早就在 git 里。GitHub 单文件警告线 50 MB、硬上限 100 MB。
   **不需要 git-lfs,别引入。**

- [x] ~~**B1. 改 `.gitignore`**~~ —— 已加，规则见文件 43-51 行；同时更新了上方那段
      "checkpoints ... stay on the GPU host" 的注释（checkpoint 现在进 git 了）。原文如下：,接在现有 `output/` 规则之后。
      `!output/model/` 必须在 `output/model/*` **之前** —— 目录一旦被排除,里面的文件就再也
      un-ignore 不回来了。

  ```gitignore
  # Best-val checkpoints are tracked: 4.5 MB each, and training is NOT bit-wise
  # reproducible (cudnn.benchmark left on, no use_deterministic_algorithms), so a
  # lost checkpoint cannot be regenerated -- only re-trained at ~9 h/seed. They are
  # what lets a finished run be re-tested under a different sampling protocol
  # (see reproduction_note.md, 2026-08-29).
  !output/model/
  output/model/*
  !output/model/*.dm4stg
  # per-epoch crash snapshot; superseded by the best checkpoint once a run finishes
  output/model/*.last.dm4stg
  # smoke-test artifacts
  output/model/test_*.dm4stg
  ```

- [x] ~~**B2. 确认只纳入了 3 个 best checkpoint**~~ —— 实际是 **4 个，17 MB**：除三个 seed 外
      还纳入了 `..._se20_e300_s2022_f62798.dm4stg`（2026-08-28 中止那轮 epoch 135 的 checkpoint，
      即 reproduction_note 里 26.74 那行的来源）。多 4.45 MB，换那条历史记录可复算，值。
      `.last.dm4stg` 与 `test_*` 已确认被规则命中排除。(不是 `.last`,不是冒烟的 `test_*`)

  ```bash
  git status --short output/model/     # 期望恰好 3 个 ?? 行,都是 *_s202{2,3,4}_*.dm4stg
  git check-ignore -v output/model/PEMS08_UGnet_N200_ss200_h32_bs8_lr0.002_se0_e300_s2022_6788e0.last.dm4stg
  ```

- [x] ~~**B3. commit + push**~~ —— 见下方 commit。(无卡模式开机即可,不烧 GPU)

  ```bash
  git add .gitignore output/model/*.dm4stg
  git commit -m "Track best-val checkpoints; they cannot be regenerated"
  git push
  ```

  推不动的话见 `reproduction_note.md` 的「AutoDL Proxy」一节:`origin` 是 SSH,
  `proxy_on` 的 `http_proxy` 对它无效,必要时把 remote 换成 HTTPS。

  **注意**:`torch.save(model)` 存的是整个 `nn.Module` 而不是 `state_dict`,加载依赖
  `algorithm/diffstg/` 的类定义和 torch 版本。`uv.lock` 在 git 里,环境是锁定的,现在没问题;
  但以后若重构 `UGnet` / `DiffSTG`,老 checkpoint 可能加载不了。


## C. 代码修复

- [ ] **C1. 修 forecast pickle 大 58 倍的 bug**

  `train.py` 的 `evals()` 里 `if mode == 'test'` 那段:

  ```python
  samples = torch.cat(samples, dim=0)[:50]      # 现在
  samples = torch.cat(samples, dim=0)[:50].clone()   # 改成这样
  targets = torch.cat(targets, dim=0)[:50].clone()
  ```

  切片返回的是**视图**,pickle 会把底下整个 storage(全部 3549 个窗口)一起序列化。
  实测:逻辑数据 9.0 MB,落盘 522.1 MB(samples storage 463 MB + targets storage 58 MB);
  `observed_flag` / `evaluate_flag` 是 `ones_like` 新分配的,所以它们正常。

  **不能用 `.contiguous()`** —— 这个切片本身就是连续的,`contiguous()` 会原样返回同一个视图,
  必须 `.clone()`。修完 4 个真实 pickle 由 2.0 GB 降到 36 MB。

- [ ] **C2. 把 `train.py` 的 test 协议由 `ddim_multi`-40 改成 `ddpm`-200**

  `train.py:520` 的 `for sample_strategy, sample_steps in [('ddim_multi', 40)]:`。
  这样以后跑完就直接是可报告的数字,不用再补测。

- [ ] **C3. C1/C2 都要在 `reproduction_note.md` 的「`train.py` 改动」一节记明理由**
      (AGENTS.md 第 4 条)。


## D. 清理

- [ ] **D1. 删冒烟测试的 checkpoint 与 pickle**(mae 345 的两 epoch 模型,8.9 MB + 2.6 MB,无保留价值)

  ```bash
  rm -f output/model/test_*.dm4stg output/forecast/test_*.pkl
  ```

- [ ] **D2. C1 修完后,重新生成或直接删掉现存的超大 pickle**(4 个 × 522 MB = 2.0 GB,
      磁盘现在 19G/30G)。它们是**派生数据**,有 checkpoint 就能在 1.4 h 内重生成,
      而且只存了 test 集**开头连续 50 个窗口**,覆盖不到一天里的其它时段,画图价值本来就有限。


## E. 收尾

- [ ] **E1. 更新 `reproduction_note.md` 末尾的 3-seed 表格**(填上三个 seed 的 `ddpm`-200 与 mean±std)

- [ ] **E2. 最终 commit + push**

  ```bash
  git add TODO.md reproduction_note.md train.py .gitignore output/metrics output/log
  git commit
  git push
  ```


## 暂不做

- **不要重训 `beta_end`。** β_N=0.1 只在 `ddim_multi` 的 0.8N 截断下才是坏的;换 `ddpm`
  后起点问题消失,17.90 已落在原文 17.68 的 0.22 以内(且在 0.42 的种子 sd 之内)。

- **不要给 UGnet 加 time-of-day / day-of-week 嵌入。** `pos_w` / `pos_d` 未被使用是**忠于原文**
  的 —— 原文的 condition 只有 masked 历史、图 `G` 和噪声步 `n`。加上去就不是复现了。

- **`model.py:124` 的 `x_masked, _, _ = input` 把 `_` 绑定两次**(实际传下去的是
  `(x_masked, pos_d, pos_d)`,`pos_w` 丢失;且 `x_masked` 被复制成 `B×n_samples` 而两个 pos 没有)。
  因为 UGnet 两者都不用,当前是 no-op,**先不动**;将来真要引入时间嵌入,必须先修这里。

- **forecast pickle 暂不进 git。** 派生数据,可重生成,不属于"丢了就没了"。
  等真要出图时再挑一个进。

- **不要引入 git-lfs。** 13.3 MB 用普通 git 就够。
