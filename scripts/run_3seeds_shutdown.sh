#!/usr/bin/env bash
# 3-seed reproduction runner, meant to be run on the GPU host inside tmux:
# preflight -> smoke test -> seeds -> summary -> commit results -> shutdown.
#
# Env knobs:
#   DATASET=PEMS08          dataset name
#   START_EPOCH=0           skip validation before this epoch (0 = original behaviour)
#   VAL_SUBSET=512          per-epoch validation windows, evenly spaced (0 = full split)
#   GPU=0                   cuda device index
#   SEEDS="2022 2023 2024"  seeds to run, in order
#   MAX_HOURS=48            per-seed wall clock limit; 0 disables
#   SKIP_SMOKE=0            1 skips the --is_test end-to-end check
#   COMMIT_RESULTS=1        0 skips the host-local git commit of output/
#   NO_SHUTDOWN=0           1 keeps the host up after the run
#   SHUTDOWN_GRACE_MINUTES=30  real countdown before power off; 0 powers off at once
#   SHUTDOWN_ON_EARLY_FAIL=0  1 powers off even when nothing usable came out
#                             (preflight/smoke failed, or every seed failed within
#                             an hour). Default keeps the host up to stay debuggable.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
cd "$repo_root" || exit 1

dataset="${DATASET:-PEMS08}"
start_epoch="${START_EPOCH:-0}"
val_subset="${VAL_SUBSET:-512}"
gpu="${GPU:-0}"
max_hours="${MAX_HOURS:-48}"
shutdown_grace_minutes="${SHUTDOWN_GRACE_MINUTES:-30}"
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
# state_file is the fixed-path answer to "why did the host go down last time?".
# It is tracked by git (see .gitignore) so it also reaches the laptop on pull.
state_file="output/last_run.json"

mkdir -p output/log output/model output/forecast output/metrics || exit 1
exec > >(tee -a "$run_log") 2>&1

now() { date +%Y-%m-%dT%H:%M:%S%z; }
hms() { printf '%02d:%02d:%02d' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
# say: to the tmux pane + raw transcript + the tracked summary log
say() { printf "$@" | tee -a "$summary_log"; }

# ---- last_run.json ---------------------------------------------------------
# Rewritten at every phase boundary rather than once at the end: on 2026-08-27
# both the raw transcript and the committed summary log lost their tails around
# the power-off, so a single end-of-run write is not something to rely on.
# run_state:      running | preflight_failed | smoke_failed | seed_failed | success
#                 ('aborted' is never written by this script -- it is set by hand
#                  when an operator stops a run on purpose, so that a leftover
#                  'running' keeps meaning 'killed by something we did not choose')
# shutdown_state: pending | skipped | cancelled | poweroff
run_state="running"
shutdown_state="pending"
started_at="$(now)"
head_commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
results_commit=""

json_escape() { printf '%s' "${1-}" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
json_str() { printf '"%s"' "$(json_escape "${1-}")"; }
# json_or_null <value>: quoted string, or null when empty
json_or_null() { if [[ -n "${1-}" ]]; then json_str "$1"; else printf 'null'; fi; }
json_arr() {
  local out="" x
  for x in "$@"; do out="${out:+$out, }$(json_str "$x")"; done
  printf '[%s]' "$out"
}

# write_state: temp file + rename + sync, so an interrupted write can never
# leave a truncated or half-parsed state file behind.
write_state() {
  local finished="" tmp="${state_file}.tmp"
  [[ "$run_state" != "running" ]] && finished="$(now)"
  {
    printf '{\n'
    printf '  "schema": 1,\n'
    printf '  "status": %s,\n' "$(json_str "$run_state")"
    printf '  "shutdown": %s,\n' "$(json_str "$shutdown_state")"
    printf '  "exit_status": %s,\n' "${run_status:-null}"
    printf '  "run_tag": %s,\n' "$(json_str "$run_tag")"
    printf '  "dataset": %s,\n' "$(json_str "$dataset")"
    printf '  "start_epoch": %s,\n' "$start_epoch"
    printf '  "val_subset": %s,\n' "$val_subset"
    printf '  "seeds": %s,\n' "$(json_arr "${seeds[@]}")"
    printf '  "failed_seeds": %s,\n' "$(json_arr ${failed[@]+"${failed[@]}"})"
    printf '  "started": %s,\n' "$(json_str "$started_at")"
    printf '  "finished": %s,\n' "$(json_or_null "$finished")"
    printf '  "code_commit": %s,\n' "$(json_str "$head_commit")"
    printf '  "results_commit": %s,\n' "$(json_or_null "$results_commit")"
    printf '  "summary_log": %s,\n' "$(json_str "$repo_root/$summary_log")"
    printf '  "raw_log": %s,\n' "$(json_str "$repo_root/$run_log")"
    printf '  "checkpoints": %s\n' "$(json_arr $(ls -1 output/model/*.dm4stg 2>/dev/null))"
    printf '}\n'
  } > "$tmp"
  mv -f "$tmp" "$state_file"
  sync
}

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

say '[INFO] dataset=%s start_epoch=%s val_subset=%s gpu=%s max_hours=%s\n' \
  "$dataset" "$start_epoch" "$val_subset" "$gpu" "$max_hours"
say '[INFO] seeds=%s\n' "${seeds[*]}"
say '[INFO] runner=%s\n' "${py_runner[*]}"
say '[INFO] commit=%s\n' "$(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
say '[INFO] raw_log=%s\n' "$run_log"
say '[INFO] started at %s\n' "$(now)"
failed=()
seeds_ran=0
write_state

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
  run_state="preflight_failed"
  write_state
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
      --is_test --start_epoch 0 --epoch 2 --val_subset "$val_subset" --seed "${seeds[0]}"; then
      say '[OK] smoke test passed at %s\n' "$(now)"
    else
      say '[FAIL] smoke test failed (status %s); skipping the real run.\n' "$?"
      run_status=1
      run_state="smoke_failed"
      write_state
    fi
  fi
fi

# ---- seeds -----------------------------------------------------------------
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
      --val_subset "$val_subset" \
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
    write_state
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
  run_state="seed_failed"
else
  say '[OK] all seeds completed: %s\n' "${seeds[*]}"
  run_state="success"
fi
write_state
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
# Staged: output/metrics/*.csv, the tracked logs, the state file and the best-val
# checkpoints (output/model/*.dm4stg -- tracked since d85584f, and unreproducible
# once lost). .gitignore keeps the .last snapshots, the smoke-test models, the
# forecast pickles and the raw transcript out.
if (( early_fail )); then
  say '\n[INFO] no usable result; nothing committed.\n'
elif [[ "${COMMIT_RESULTS:-1}" != "1" ]]; then
  say '\n[INFO] COMMIT_RESULTS=0, results left uncommitted under output/.\n'
elif ! git rev-parse --git-dir >/dev/null 2>&1; then
  say '\n[WARN] not a git repo; results left uncommitted under output/.\n'
else
  say '\n[INFO] === commit results ===\n'
  write_state
  git add -A -- output/metrics output/log output/model "$state_file"
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
      results_commit="$(git rev-parse --short HEAD)"
      say '[OK] results committed as %s on %s\n' \
        "$results_commit" "$(git rev-parse --abbrev-ref HEAD)"
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
if [[ "${NO_SHUTDOWN:-0}" == "1" ]]; then
  say '[INFO] NO_SHUTDOWN=1, host stays up.\n'
  shutdown_state="skipped"
  write_state
  exit "$run_status"
fi

# A long run always powers off, even a failed one; see early_fail above.
if (( early_fail )) && [[ "${SHUTDOWN_ON_EARLY_FAIL:-0}" != "1" ]]; then
  say '\n[FAIL] ==================================================\n'
  say '[FAIL] no usable result -- host stays up so you can read the error above.\n'
  say '[FAIL] full transcript: %s\n' "$run_log"
  say '[FAIL] set SHUTDOWN_ON_EARLY_FAIL=1 to power off even in this case.\n'
  say '[FAIL] ==================================================\n'
  shutdown_state="skipped"
  write_state
  exit "$run_status"
fi

if [[ ! "$shutdown_grace_minutes" =~ ^[0-9]+$ ]]; then
  say '[FAIL] SHUTDOWN_GRACE_MINUTES must be a non-negative integer; host stays up.\n'
  shutdown_state="skipped"
  write_state
  exit "$run_status"
fi

# This host's /usr/bin/shutdown is an AutoDL wrapper that ignores every argument
# and kills supervisord on the spot, so `shutdown -h +30` powers off NOW and
# `shutdown -c` does not cancel anything -- it powers the container off too.
# The grace window therefore has to be a real sleep here, and the cancel has to
# be something other than `shutdown -c`: Ctrl-C in this pane, or the cancel file
# (which also works after you detached from tmux).
cancel_file="output/.cancel_shutdown"
rm -f "$cancel_file"

say '[INFO] result status=%s; local summary=%s; raw transcript=%s\n' \
  "$run_status" "$summary_log" "$run_log"
say '[INFO] powering off in %s minute(s).\n' "$shutdown_grace_minutes"
say '[INFO] to cancel: Ctrl-C in this pane, or `touch %s/%s`.\n' "$repo_root" "$cancel_file"
say '[INFO] do NOT run `shutdown -c` here: shutdown ignores its arguments on this\n'
say '[INFO] host and powers the container off instead of cancelling.\n'
# Flush the final status, metrics and errors before the countdown starts.
sync

shutdown_cancelled=0
trap 'shutdown_cancelled=1' INT TERM
remaining=$(( shutdown_grace_minutes * 60 ))
while (( remaining > 0 )); do
  [[ -f "$cancel_file" ]] && shutdown_cancelled=1
  (( shutdown_cancelled )) && break
  if (( remaining % 300 == 0 )) || (( remaining <= 60 )); then
    say '[INFO] powering off in %s\n' "$(hms "$remaining")"
  fi
  chunk=30
  (( remaining < chunk )) && chunk=$remaining
  sleep "$chunk"
  remaining=$(( remaining - chunk ))
done
trap - INT TERM

if (( shutdown_cancelled )); then
  say '\n[INFO] shutdown cancelled; host stays up. Remember it is still billing.\n'
  rm -f "$cancel_file"
  shutdown_state="cancelled"
  write_state
  exit "$run_status"
fi

say '[INFO] powering off now at %s\n' "$(now)"
shutdown_state="poweroff"
write_state
if command -v shutdown >/dev/null 2>&1; then
  shutdown -h now
  sleep 120
fi
if command -v poweroff >/dev/null 2>&1; then
  poweroff
  sleep 120
fi
say '[FAIL] automatic shutdown failed; power the host off manually to stop billing.\n'
exit 1
