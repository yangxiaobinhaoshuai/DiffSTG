#!/usr/bin/env python
"""Run the paper's test protocol on an existing checkpoint, without training.

Why this exists: train.py only evaluates on the test set *after* the training
loop returns normally. A run that is killed first -- wall-clock cap, power-off,
operator abort -- leaves a perfectly good best checkpoint on disk and no metrics
at all. This script closes that gap: it loads the checkpoint and calls the very
same evals(mode='test') that train.py calls, so the numbers are produced by one
code path, not by a re-implementation.

It never touches the training artefacts: the log and the forecast pickle get a
'.testonly' suffix, because train.py opens its log with mode='w' and reusing the
original trial name would truncate the training log.

Usage:
  uv run --frozen --no-sync python scripts/test_from_checkpoint.py \
      --ckpt output/model/<trial>.dm4stg --data PEMS08 --gpu 0 --seed 2022 \
      --epoch 300 --best_epoch 135 --start_epoch 20
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.model import save2file, CSV_HEAD
from train import default_config, evals, setup_seed
from utils.common_utils import dict_merge, dir_check, unfold_dict
from utils.eval import Metric


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True, help="path to a *.dm4stg checkpoint")
    p.add_argument("--data", type=str, default="PEMS08")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2022)
    # Recorded verbatim in the CSV row; they describe the run that produced the
    # checkpoint and cannot be recovered from the checkpoint itself.
    p.add_argument("--epoch", type=int, default=300, help="max epochs of the training run")
    p.add_argument("--best_epoch", type=int, default=-1, help="epoch the checkpoint came from")
    p.add_argument("--start_epoch", type=int, default=0)
    p.add_argument("--val_subset", type=int, default=0,
                   help="val_subset the training run used (0 = full split)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.002)
    # Test protocol -- defaults match train.py's final evaluation exactly.
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--sample_strategy", type=str, default="ddim_multi")
    p.add_argument("--sample_steps", type=int, default=40)
    p.add_argument("--test_batch_size", type=int, default=8)
    p.add_argument("--tag", type=str, default="testonly", help="suffix for log/forecast files")
    return p.parse_args()


def main():
    args = get_args()
    setup_seed(args.seed)
    torch.set_num_threads(2)

    config = default_config(args.data, args.gpu)
    config.is_test = False
    config.nni = False
    config.seed = args.seed
    config.lr = args.lr
    config.batch_size = args.batch_size
    config.epoch = args.epoch
    config.start_epoch = args.start_epoch
    config.val_subset = args.val_subset
    config.n_samples = args.n_samples
    config.test_batch_size = args.test_batch_size
    config.mask_ratio = 0.0

    trial = os.path.basename(args.ckpt).replace(".dm4stg", "")
    config.trial_name = f"{trial}.{args.tag}"
    config.log_path = f"{config.PATH_LOG}{config.trial_name}.log"
    config.forecast_path = f"{config.PATH_FORECAST}{config.trial_name}.pkl"
    config.model_path = args.ckpt
    dir_check(config.log_path)
    dir_check(config.forecast_path)
    config.logger.open(config.log_path, mode="w")

    # weights_only=False: the checkpoint is a pickled nn.Module (see train.py).
    model = torch.load(args.ckpt, map_location=config.device, weights_only=False)
    model = model.to(config.device)

    # Adopt the checkpoint's own model config. default_config()'s values are only
    # placeholders that train.py overwrites from the CLI, so trusting them here
    # would record hyper-parameters the run never used (beta_end 0.02 vs the 0.1
    # actually baked into this checkpoint) while sampling correctly uses the
    # pickled buffers -- a silently wrong results row.
    for k, v in model.config.items():
        if k not in ('A', 'device'):
            config.model[k] = v
    config.model.sample_strategy = args.sample_strategy
    config.model.sample_steps = args.sample_steps
    model.set_sample_strategy(config.model.sample_strategy)
    model.set_ddim_sample_steps(config.model.sample_steps)

    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj
    print(f"loaded  : {args.ckpt}")
    print(f"model   : N={config.model.N} beta_end={config.model.beta_end} "
          f"beta_schedule={config.model.beta_schedule} d_h={config.model.d_h}")
    print(f"protocol: {config.model.sample_strategy}-{config.model.sample_steps}, "
          f"n_samples={config.n_samples}, full test split")

    # Fail now, not after ~20 min of sampling: save2file_meta indexes params[k] for
    # every column, so one missing knob would KeyError at the very last step.
    missing = sorted(set(CSV_HEAD) - set(unfold_dict(config))
                     - {'mae', 'rmse', 'mape', 'crps', 'mis', 'time', 'log_time',
                        'best_epoch', 'model'})
    if missing:
        raise SystemExit(f"config is missing CSV columns: {missing}")

    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.test_batch_size, shuffle=False)
    print(f"test windows: {len(test_dataset)}")

    metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
    evals(model, test_loader, args.best_epoch, metrics_test, config, clean_data, mode='test')
    config.logger.write(f"Final results in test:{metrics_test}\n", is_terminal=True)

    params = unfold_dict(config)
    params = dict_merge([params, metrics_test.to_dict()])
    params['best_epoch'] = args.best_epoch
    params['model'] = config.model.epsilon_theta
    save2file(params)

    m = metrics_test.metrics
    print(f"\nMAE {m['mae']:.4f}  RMSE {m['rmse']:.4f}  MAPE {m['mape']:.4f}  "
          f"CRPS {m['crps']:.4f}  MIS {m['mis']:.4f}")
    print(f"row appended to output/metrics/DiffSTG.csv")


if __name__ == "__main__":
    main()
