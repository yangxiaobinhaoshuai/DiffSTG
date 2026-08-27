#!/usr/bin/env bash
# 3-seed reproduction runner, meant to be run on the GPU host inside tmux:
# preflight -> smoke test -> seeds -> summary -> commit results -> shutdown.
#
# Env knobs:
#   DATASET=PEMS08          dataset name
#   START_EPOCH=20          skip validation before this epoch
#   GPU=0                   cuda device index
#   SEEDS="2022 2023 2024"  seeds to run, in order
#   MAX_HOURS=48            per-seed wall clock limit; 0 disables
#   SKIP_SMOKE=0            1 skips the --is_test end-to-end check
#   COMMIT_RESULTS=1        0 skips the host-local git commit of output/
#   NO_SHUTDOWN=0           1 keeps the host up after the run
#   SHUTDOWN_ON_EARLY_FAIL=0  1 powers off even when nothing usable came out
#                             (preflight/smoke failed, or every seed failed within
#                             an hour). Default keeps the host up to stay debuggable.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$repo_root" || exit 1

dataset="${DATASET:-PEMS08}"
start_epoch="${START_EPOCH:-20}"
gpu="${GPU:-0}"
max_hours="${MAX_HOURS:-48}"
[[ "$max_hours" == "0" ]] && max_hours=8760
read -r -a seeds <<< "${SEEDS:-2022 2023 2024}"
read -r -a py_runner <<< "${PY_RUNNER:-uv run --frozen --no-sync python}"

csv="output/metrics/DiffSTG.csv"
run_stamp="$(date +%Y%m%d-%H%M%S)"
run_tag="${dataset}_start${start_epoch}_${run_stamp}"
# run_log is the raw transcript (gitignored: train.py's \r progress makes it one
# huge line). summary_log holds only the lines below and is tracked by git.
run_log="output/log/run_3seeds_${run_tag}.log"
summary_log="output/log/summary_3seeds_${run_tag}.log"

mkdir -p output/log output/model output/forecast output/metrics || exit 1
exec > >(tee -a "$run_log") 2>&1

now() { date +%Y-%m-%dT%H:%M:%S%z; }
hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
# say: to the tmux pane + raw transcript + the tracked summary log
say() { printf "$@" | tee -a "$summary_log"; }

timeout_bin="$(command -v timeout || command -v gtimeout || true)"
# run_with_timeout <hours> <cmd...>; runs unbounded if coreutils timeout is absent
run_with_timeout() {
  local hours="$1"
  shift
  if [[ -n "$timeout_bin" ]]; then
    "$timeout_bin" -k 2m "${hours}h" "$@"
  else
    "$@"
  fi
}

say '[INFO] dataset=%s start_epoch=%s gpu=%s max_hours=%s\n' \
  "$dataset" "$start_epoch" "$gpu" "$max_hours"
say '[INFO] seeds=%s\n' "${seeds[*]}"
say '[INFO] runner=%s\n' "${py_runner[*]}"
say '[INFO] commit=%s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
say '[INFO] raw_log=%s\n' "$run_log"
say '[INFO] started at %s\n' "$(now)"

# ---- preflight: fail in seconds, not in days -------------------------------
say '\n[INFO] === preflight ===\n'
preflight_ok=1

for f in "data/dataset/${dataset}/flow.npy" "data/dataset/${dataset}/adj.npy"; do
  if [[ -f "$f" ]]; then
    say '[OK] %s\n' "$f"
  else
    say '[FAIL] missing dataset file: %s\n' "$f"
    preflight_ok=0
  fi
done

if [[ -n "$timeout_bin" ]]; then
  say '[OK] timeout=%s\n' "$timeout_bin"
else
  say '[WARN] coreutils timeout not found; MAX_HOURS will not be enforced\n'
fi

df -h . || true
nvidia-smi || true

if ! "${py_runner[@]}" -c '
import sys, torch
print(f"[OK] torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"n_gpu={torch.cuda.device_count()}")
sys.exit(0 if torch.cuda.is_available() else 1)
'; then
  say '[FAIL] python/torch/cuda preflight failed\n'
  if [[ ! -d .venv ]]; then
    say '[HINT] no .venv in %s; on a fresh host run `uv sync` first\n' "$repo_root"
  else
    say '[HINT] if CUDA is unavailable the instance is probably booted in no-GPU mode;\n'
    say '[HINT] power it off and start it again with a GPU attached.\n'
  fi
  preflight_ok=0
fi

if (( ! preflight_ok )); then
  say '\n[FAIL] preflight failed; nothing was trained.\n'
  run_status=1
else
  # ---- smoke test: exercises save -> load -> ddim test -> CRPS/MIS -> csv ---
  # Needs --start_epoch 0, otherwise validation never runs, no checkpoint is
  # written, and the torch.load path stays untested.
  run_status=0
  if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
    say '\n[INFO] === smoke test skipped (SKIP_SMOKE=1) ===\n'
  else
    say '\n[INFO] === smoke test (--is_test) ===\n'
    if run_with_timeout 1 "${py_runner[@]}" train.py \
      --data "$dataset" --gpu "$gpu" \
      --is_test --start_epoch 0 --epoch 2 --seed "${seeds[0]}"; then
      say '[OK] smoke test passed at %s\n' "$(now)"
    else
      say '[FAIL] smoke test failed (status %s); skipping the real run.\n' "$?"
      run_status=1
    fi
  fi
fi

# ---- seeds -----------------------------------------------------------------
failed=()
seeds_ran=0
if (( run_status == 0 )); then
  seeds_ran=1
  say '\n[INFO] === seeds ===\n'
  for seed in "${seeds[@]}"; do
    say '\n[INFO] seed=%s started at %s\n' "$seed" "$(now)"
    rows_before="$(wc -l < "$csv" 2>/dev/null || echo 0)"
    seed_start=$SECONDS

    run_with_timeout "$max_hours" "${py_runner[@]}" train.py \
      --data "$dataset" \
      --gpu "$gpu" \
      --start_epoch "$start_epoch" \
      --seed "$seed"
    seed_status=$?

    elapsed=$((SECONDS - seed_start))
    rows_after="$(wc -l < "$csv" 2>/dev/null || echo 0)"

    if (( seed_status == 0 )) && (( rows_after > rows_before )); then
      say '[OK] seed=%s done at %s, elapsed=%s\n' "$seed" "$(now)" "$(hms "$elapsed")"
    else
      if (( seed_status == 124 )); then
        say '[FAIL] seed=%s hit the %sh limit\n' "$seed" "$max_hours"
      elif (( seed_status != 0 )); then
        say '[FAIL] seed=%s exited with status %s\n' "$seed" "$seed_status"
      else
        say '[FAIL] seed=%s exited 0 but wrote no row to %s\n' "$seed" "$csv"
      fi
      say '[FAIL] seed=%s at %s, elapsed=%s; continuing with the next seed.\n' \
        "$seed" "$(now)" "$(hms "$elapsed")"
      failed+=("$seed")
    fi
  done
  (( ${#failed[@]} )) && run_status=1
fi

# ---- summary ---------------------------------------------------------------
say '\n[INFO] === summary ===\n'
say '[INFO] finished at %s, total elapsed=%s\n' "$(now)" "$(hms "$SECONDS")"
if (( ! seeds_ran )); then
  say '[FAIL] seeds never started; preflight or smoke test blocked the run.\n'
elif (( ${#failed[@]} )); then
  say '[FAIL] failed seeds: %s (of %s)\n' "${failed[*]}" "${seeds[*]}"
else
  say '[OK] all seeds completed: %s\n' "${seeds[*]}"
fi
say '[INFO] last rows of %s:\n' "$csv"
tail -n $(( ${#seeds[@]} + 1 )) "$csv" 2>/dev/null | tee -a "$summary_log" \
  || say '[FAIL] %s missing\n' "$csv"
ls -lh output/model output/forecast 2>/dev/null || true

# "early failure" = nothing usable came out AND it happened fast enough that you
# are probably still watching. Drives both the commit and the shutdown decision:
# no commit noise, and the host stays up so the error remains readable.
early_fail=0
if (( ! seeds_ran )); then
  early_fail=1
elif (( ${#failed[@]} == ${#seeds[@]} )) && (( SECONDS < 3600 )); then
  early_fail=1
fi

# ---- commit results on the host (push is manual) ---------------------------
# Only output/metrics/*.csv and the tracked logs are staged; .gitignore keeps
# checkpoints, forecast pickles and the raw transcript out.
if (( early_fail )); then
  say '\n[INFO] no usable result; nothing committed.\n'
elif [[ "${COMMIT_RESULTS:-1}" != "1" ]]; then
  say '\n[INFO] COMMIT_RESULTS=0, results left uncommitted under output/.\n'
elif ! git rev-parse --git-dir >/dev/null 2>&1; then
  say '\n[WARN] not a git repo; results left uncommitted under output/.\n'
else
  say '\n[INFO] === commit results ===\n'
  git add -A -- output/metrics output/log
  if git diff --cached --quiet; then
    say '[WARN] nothing to commit under output/.\n'
  else
    msg_status="ok"
    (( ${#failed[@]} )) && msg_status="failed: ${failed[*]}"
    (( ! seeds_ran )) && msg_status="seeds never started"
    if [[ -n "$(git config user.email 2>/dev/null)" ]]; then
      git commit -q -m "results: 3seeds ${run_tag}" -m "seeds: ${seeds[*]} (${msg_status})"
    else
      git -c user.name='diffstg-runner' -c user.email='diffstg-runner@localhost' \
        commit -q -m "results: 3seeds ${run_tag}" -m "seeds: ${seeds[*]} (${msg_status})"
    fi
    commit_status=$?
    if (( commit_status == 0 )); then
      say '[OK] results committed as %s on %s\n' \
        "$(git rev-parse --short HEAD)" "$(git rev-parse --abbrev-ref HEAD)"
      say '[INFO] not pushed by design; run `git push` after powering the host back on.\n'
    else
      say '[FAIL] git commit failed; results are still on disk under output/.\n'
    fi
  fi
fi

say '[INFO] checkpoints and forecast pickles stay on this host (gitignored).\n'

# ---- shutdown --------------------------------------------------------------
# Only power off once the real seeds have started. A preflight/smoke failure
# means nothing was trained, you are almost certainly still at the keyboard,
# and shutting down would take the error off the screen along with the host.
sync
if [[ "${NO_SHUTDOWN:-0}" == "1" ]]; then
  say '[INFO] NO_SHUTDOWN=1, host stays up.\n'
  exit "$run_status"
fi

# A long run always powers off, even a failed one; see early_fail above.
if (( early_fail )) && [[ "${SHUTDOWN_ON_EARLY_FAIL:-0}" != "1" ]]; then
  say '\n[FAIL] ==================================================\n'
  say '[FAIL] no usable result -- host stays up so you can read the error above.\n'
  say '[FAIL] full transcript: %s\n' "$run_log"
  say '[FAIL] set SHUTDOWN_ON_EARLY_FAIL=1 to power off even in this case.\n'
  say '[FAIL] ==================================================\n'
  exit "$run_status"
fi

say '[INFO] shutting down; run `shutdown -c` within 1 minute to cancel.\n'
accepted=0
if command -v shutdown >/dev/null 2>&1; then
  if shutdown -h +1 || shutdown -h now; then
    accepted=1
  fi
fi
if (( accepted )); then
  # shutdown is async: if it really worked this sleep never returns.
  sleep 300
  say '[WARN] shutdown was accepted but the host is still up after 5 min.\n'
fi
if command -v poweroff >/dev/null 2>&1; then
  poweroff
  sleep 120
fi
say '[FAIL] automatic shutdown failed; power the host off manually to stop billing.\n'
exit 1
