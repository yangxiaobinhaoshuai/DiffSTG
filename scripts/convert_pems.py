#!/usr/bin/env python
"""Convert a raw PEMS03/PEMS04 release into the layout train.py expects.

The repo wants two files per dataset:

  flow.npy  (T, V, C)  -- CleanDataset.read_data() uses channel 0 only
  adj.npy   (V, V)     -- binary, symmetric, zero diagonal

Both conventions are read off the PEMS08 files that shipped with the repo and
that produced the validated 17.92 MAE, so a converted dataset is byte-comparable
in structure to the one the reproduction was checked against.

Two things this handles that a naive converter gets wrong:

1. **Only channel 0 is kept, as float32.** PEMS04 ships (T, V, 3) float64
   (flow / occupancy / speed) = 125 MB, which is over GitHub's 100 MB hard limit
   -- and `data/dataset/*/flow.npy` is tracked in git. Channel 0 as float32 is
   21 MB and loses nothing: read_data() slices [:, :, 0] anyway, and the values
   are integer traffic counts well inside float32's exact-integer range.

2. **PEMS03's edge list uses raw sensor IDs, not node indices.** They must be
   mapped through PEMS03.txt, whose line order *is* the node ordering of the npz.
   Skipping this silently produces a wrong graph. PEMS04's IDs are already
   0-indexed, so it needs no mapping.

Usage:
  uv run --frozen --no-sync python scripts/convert_pems.py --dataset PEMS04 \
      --raw-dir /root/data_raw/PEMS04 --out-dir data/dataset/PEMS04
"""
import argparse
import csv
import os
import sys

import numpy as np

# (nodes, time steps) each release must match, per the STSGCN benchmark.
EXPECTED = {'PEMS03': (358, 26208), 'PEMS04': (307, 16992)}


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(EXPECTED))
    p.add_argument("--raw-dir", required=True, help="dir holding <NAME>.npz, <NAME>.csv [, <NAME>.txt]")
    p.add_argument("--out-dir", required=True, help="dir to write flow.npy and adj.npy into")
    p.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")
    return p.parse_args()


def load_id_map(path, n_nodes):
    """Line order in <NAME>.txt is the node ordering of the npz.

    The file is CRLF-terminated with no trailing newline, so a line count is one
    short of the truth; splitting on whitespace is the robust read.
    """
    ids = open(path, 'rb').read().decode().split()
    ids = [int(x) for x in ids if x.strip()]
    if len(ids) != n_nodes:
        raise SystemExit(f"{path}: {len(ids)} ids for {n_nodes} nodes")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate sensor ids")
    return {sid: i for i, sid in enumerate(ids)}


def build_adj(csv_path, n_nodes, id_map):
    """Binary, symmetric, zero-diagonal -- the PEMS08 convention.

    Distances are deliberately discarded: the shipped PEMS08 adj.npy is 0/1, and
    a Gaussian-kernel weighted graph (what many PEMS pipelines default to) would
    feed UGnet a different operator than the reproduction was validated on.
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    rows = list(csv.DictReader(open(csv_path)))
    cost_col = 'distance' if 'distance' in rows[0] else 'cost'
    self_loops = unknown = 0
    for r in rows:
        a, b = int(r['from']), int(r['to'])
        if id_map is not None:
            if a not in id_map or b not in id_map:
                unknown += 1
                continue
            a, b = id_map[a], id_map[b]
        if not (0 <= a < n_nodes and 0 <= b < n_nodes):
            raise SystemExit(f"{csv_path}: node index out of range: {a},{b}")
        if a == b:
            self_loops += 1          # dropped: PEMS08's adj has a zero diagonal
            continue
        adj[a, b] = adj[b, a] = 1.0
    return adj, len(rows), self_loops, unknown, cost_col


def main():
    args = get_args()
    name = args.dataset
    n_nodes, n_steps = EXPECTED[name]

    npz_path = os.path.join(args.raw_dir, f"{name}.npz")
    csv_path = os.path.join(args.raw_dir, f"{name}.csv")
    txt_path = os.path.join(args.raw_dir, f"{name}.txt")
    for p in (npz_path, csv_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")

    data = np.load(npz_path)['data']
    print(f"=== {name} ===")
    print(f"npz          : shape={data.shape} dtype={data.dtype}")
    if data.shape[0] != n_steps or data.shape[1] != n_nodes:
        raise SystemExit(f"expected (T={n_steps}, V={n_nodes}, C), got {data.shape}")

    flow = data[:, :, 0:1]
    if np.isnan(flow).any():
        # read_data() has no nan_to_num on the PEMS branch -- a NaN would reach the loss.
        raise SystemExit(f"{npz_path}: channel 0 contains {int(np.isnan(flow).sum())} NaNs")
    if not np.array_equal(flow, np.round(flow)):
        print("  warn: channel 0 is not integral; float32 may not be exact")
    flow32 = flow.astype(np.float32)
    if not np.array_equal(flow32.astype(np.float64), flow.astype(np.float64)):
        raise SystemExit("float32 cast is lossy for this data -- keep float64")
    print(f"flow (ch0)   : shape={flow32.shape} dtype=float32 "
          f"min={flow32.min():.1f} max={flow32.max():.1f} mean={flow32.mean():.2f} "
          f"zeros={float((flow32 == 0).mean()) * 100:.2f}%  float32 无损 ✅")

    id_map = None
    if os.path.exists(txt_path):
        id_map = load_id_map(txt_path, n_nodes)
        print(f"id map       : {txt_path} -> {len(id_map)} sensor ids mapped to 0..{n_nodes - 1}")
    else:
        print("id map       : none (csv ids assumed 0-indexed)")

    adj, n_edges, self_loops, unknown, cost_col = build_adj(csv_path, n_nodes, id_map)
    if unknown:
        raise SystemExit(f"{csv_path}: {unknown} edges reference ids absent from {txt_path}")
    deg = (adj != 0).sum(1)
    print(f"csv          : {n_edges} rows (cost col '{cost_col}'), self-loops dropped: {self_loops}")
    print(f"adj          : V={n_nodes} nonzero={int((adj != 0).sum())} "
          f"undirected={int((adj != 0).sum()) // 2} avg_deg={deg.mean():.2f} "
          f"deg[{int(deg.min())},{int(deg.max())}] isolated={int((deg == 0).sum())}")

    assert np.array_equal(adj, adj.T), "adj not symmetric"
    assert np.array_equal(np.unique(adj), np.array([0., 1.])), "adj not binary"
    assert np.diag(adj).sum() == 0, "adj has a nonzero diagonal"
    print("adj checks   : binary ✅  symmetric ✅  zero-diagonal ✅")
    if (deg == 0).any():
        # asym_adj guards division with +1e-6, so this degrades rather than crashes.
        print(f"  warn: {int((deg == 0).sum())} isolated node(s) will receive no spatial signal")

    if args.dry_run:
        print("dry-run: nothing written")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    fp = os.path.join(args.out_dir, "flow.npy")
    ap = os.path.join(args.out_dir, "adj.npy")
    np.save(fp, flow32)
    np.save(ap, adj)
    print(f"wrote        : {fp} ({os.path.getsize(fp) / 1e6:.1f} MB)")
    print(f"wrote        : {ap} ({os.path.getsize(ap) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
