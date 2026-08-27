#!/usr/bin/env bash
# Print one line about how the last 3-seed run ended. Sourced from ~/.bashrc so
# that an ssh login after a power-off answers "err or success?" without digging
# through output/log/. Safe to source in any shell: never exits, never fails.
_diffstg_last_run_banner() {
  local repo="${DIFFSTG_REPO:-/root/projects/DiffSTG}"
  local state="$repo/output/last_run.json"
  [[ -r "$state" ]] || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  python3 - "$state" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
R, G, Y, B, N = '\033[31m', '\033[32m', '\033[33m', '\033[1m', '\033[0m'
status, sd = d.get('status', '?'), d.get('shutdown', '?')
# status still "running" with no finish time means nothing wrote the ending:
# the box went down under the run (console shutdown, OOM, kill, out of credit).
if status == 'running' and not d.get('finished'):
    colour, head = R, 'INTERRUPTED'
elif status == 'success':
    colour, head = G, 'SUCCESS'
else:
    colour, head = R, status.upper()
seeds, failed = d.get('seeds') or [], d.get('failed_seeds') or []
bits = [f"seeds={' '.join(map(str, seeds))}"]
if failed:
    bits.append(f"failed={' '.join(map(str, failed))}")
bits.append(f"shutdown={sd}")
if d.get('results_commit'):
    bits.append(f"commit={d['results_commit']}")
n_ckpt = len(d.get('checkpoints') or [])
bits.append(f"ckpt={n_ckpt}")
print(f"{B}[last run]{N} {colour}{head}{N}  {d.get('run_tag','?')}  " + '  '.join(bits))
print(f"           started {d.get('started','?')}  finished {d.get('finished') or Y+'never'+N}")
print(f"           details: {d.get('summary_log','?')}")
PY
}
case $- in *i*) _diffstg_last_run_banner ;; esac
