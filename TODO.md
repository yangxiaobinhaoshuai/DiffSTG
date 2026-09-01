# TODO

大步骤和训练进度看 [PROGRESS.md](PROGRESS.md)；改动理由和实验记录看
[reproduction_note.md](reproduction_note.md)。这里只放细碎环节。

## 下一步：PEMS03 三 seed（~61 h）

最后一个数据集。`train.py` 的 test 阶段直接跑 `ddpm`-200，**跑完就是可报告的数字，不用补测**。

```bash
cd /root/projects/DiffSTG
DATASET=PEMS03 \
  setsid nohup bash scripts/run_3seeds_shutdown.sh \
  > output/log/run_pems03.out 2>&1 < /dev/null &
```

- [ ] 起跑，确认 `PPID=1` 且 GPU 上来了
- [ ] 跑完核对 CSV 里三行 `PEMS03` + `ddpm`/200，算 mean±std
- [ ] 更新 `PROGRESS.md` 的进度表和结果表

**必须 `setsid`。** 2026-08-30 seed 2024 的补测就是只挂在 tmux 里，shell 一关整个进程被带走，
日志 0 字节、GPU 空转 7 小时。tmux 不是守护进程。

这轮**不加 `NO_SHUTDOWN=1`** —— 跑完没有后续实验，让它自动关机（默认 30 min 缓冲，
`touch output/.cancel_shutdown` 可取消）。默认 `MAX_HOURS=48`（单 seed 墙钟上限），
PEMS03 估算 ~18 h 训练 + ~5 h test/seed，够。

关机后再开机时记得 `git push`：结果由脚本在 host 上 commit，但从不 push。

## 收尾

- [ ] 三数据集结果汇总表 -> `PROGRESS.md` + 论文
- [ ] `git push`（结果在 host 提交，push 手动做；`git status -sb` 看积压几个）
- [ ] 删 `/root/data_raw/`（48 MB 原始上传件，转换完就没用了）
- [ ] 开源前：README 补数据获取说明（获取方式已写在 `reproduction_note.md`）

## 长期不做

- **不要重训 `beta_end`。** β_N=0.1 只在 `ddim_multi` 的 0.8N 截断下才是坏的；换 `ddpm`
  后起点问题消失，17.92 已落在原文 17.68 的 0.24 以内。

- **不要给 UGnet 加 time-of-day / day-of-week 嵌入。** `pos_w` / `pos_d` 未被使用是**忠于原文**
  的 —— 原文的 condition 只有 masked 历史、图 `G` 和噪声步 `n`。加上去就不是复现了。

- **不要为 PEMS03/04 做数据集特定调优。** 超参一律沿用原作者发布值。"自己的方法调参、
  baseline 用默认值"是答辩时最容易被打的不对称。

- **不跑 PEMS07。** V=883，采样路径实测 7.41x，单 seed ~48 h 正好顶到墙钟上限，test 还要 16 h。

- **不做 AIR-BJ / AIR-GZ。** 是 PM2.5 不是交通；且数据原作者仓库没给，拿不到和原文同一版
  就没有可比性。另注：`AIR_GZ` 的 config split 是 `int(8760*10/12)` 和 `int(8160*11/12)`，
  算出来是 83%/2%/15% 而非 6:2:2，那个 `8160` 大概率是 `8760` 的笔误 —— 真要跑得先修。

- **`model.py:124` 的 `x_masked, _, _ = input` 把 `_` 绑定两次**（实际传下去的是
  `(x_masked, pos_d, pos_d)`，`pos_w` 丢失）。因为 UGnet 两者都不用，当前是 no-op，
  **先不动**；将来真要引入时间嵌入，必须先修这里。

- **不要引入 git-lfs。** checkpoint 4.5 MB/个用普通 git 就够；数据集已 gitignored。
