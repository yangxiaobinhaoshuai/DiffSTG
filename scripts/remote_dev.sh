#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/remote_dev.sh <command> [args...]

Commands:
  push                 Sync local source to remote, excluding data/output/cache files.
  pull-output          Sync remote output directory back to local output mirror.
  exec <cmd...>        Run a command in the remote repo directory.
  py <file> [args...]  Run a Python entry file with PY_RUNNER.
  job <file> [args...] Run a repo script on remote with bash.
  ssh                  Open an interactive shell in the remote repo directory.
  tail [pattern]       Follow remote logs; default pattern is output/log/*.log.
  tmux <name> [cmd...] Create/attach a remote tmux session in the repo directory.
  status               Print resolved local/remote paths.

Config:
  REMOTE_HOSTNAME      Remote SSH hostname or IP. Required.
  REMOTE_PORT          Remote SSH port. Required.
  REMOTE_USER          Remote SSH user. Required.
  REMOTE_DIR           Remote repo directory. Default: ~/runs/<local repo name>.
  LOCAL_DIR            Local repo directory. Default: current working directory.
  OUTPUT_DIR           Remote output directory inside repo. Default: output.
  LOCAL_OUTPUT_DIR     Local mirror for remote output. Default: output_remote.
  PY_RUNNER            Python launcher used by py. Default: uv run python.
  RSYNC_EXCLUDES       Extra rsync exclude args, e.g. "--exclude wandb/".

Optional:
  Put the variables above in .remote-dev.env; this script will source it.
  SSH commands use -F /dev/null and do not read ~/.ssh/config.
  Keep REMOTE_DIR free of spaces; shell-style ~/... paths are supported.
EOF
}

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$root_dir/.remote-dev.env" ]]; then
  # shellcheck disable=SC1091
  source "$root_dir/.remote-dev.env"
fi

LOCAL_DIR="${LOCAL_DIR:-$root_dir}"
repo_name="$(basename "$LOCAL_DIR")"
REMOTE_HOSTNAME="${REMOTE_HOSTNAME:-}"
REMOTE_PORT="${REMOTE_PORT:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_DIR="${REMOTE_DIR:-~/runs/$repo_name}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-output_remote}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
RSYNC_EXCLUDES="${RSYNC_EXCLUDES:-}"

if [[ "$REMOTE_DIR" == "$HOME/"* ]]; then
  REMOTE_DIR="~/${REMOTE_DIR#"$HOME"/}"
fi

require_remote() {
  if [[ -z "$REMOTE_HOSTNAME" || -z "$REMOTE_PORT" || -z "$REMOTE_USER" ]]; then
    echo "REMOTE_HOSTNAME, REMOTE_PORT and REMOTE_USER are required in .remote-dev.env." >&2
    exit 2
  fi
  if [[ ! "$REMOTE_PORT" =~ ^[0-9]+$ ]]; then
    echo "REMOTE_PORT must be numeric." >&2
    exit 2
  fi
  if (( 10#$REMOTE_PORT < 1 || 10#$REMOTE_PORT > 65535 )); then
    echo "REMOTE_PORT must be between 1 and 65535." >&2
    exit 2
  fi
}

remote_target() {
  printf '%s@%s' "$REMOTE_USER" "$REMOTE_HOSTNAME"
}

ssh_remote() {
  ssh -F /dev/null -p "$REMOTE_PORT" "$(remote_target)" "$@"
}

remote_cd() {
  printf 'cd %s' "$REMOTE_DIR"
}

quote_args() {
  printf '%q ' "$@"
}

run_remote() {
  ssh_remote "$(remote_cd) && $(quote_args "$@")"
}

cmd="${1:-}"
case "$cmd" in
  ""|-h|--help|help)
    usage
    ;;

  status)
    require_remote
    printf 'LOCAL_DIR=%s\n' "$LOCAL_DIR"
    printf 'REMOTE_HOSTNAME=%s\n' "$REMOTE_HOSTNAME"
    printf 'REMOTE_PORT=%s\n' "$REMOTE_PORT"
    printf 'REMOTE_USER=%s\n' "$REMOTE_USER"
    printf 'SSH_COMMAND=ssh -F /dev/null -p %s %s\n' "$REMOTE_PORT" "$(remote_target)"
    printf 'REMOTE_DIR=%s\n' "$REMOTE_DIR"
    printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
    printf 'LOCAL_OUTPUT_DIR=%s\n' "$LOCAL_OUTPUT_DIR"
    printf 'PY_RUNNER=%s\n' "$PY_RUNNER"
    ;;

  push)
    require_remote
    if ssh_remote "mkdir -p $REMOTE_DIR" && \
      rsync -az --delete -e "ssh -F /dev/null -p $REMOTE_PORT" \
        --exclude '.git/' \
        --exclude '.venv/' \
        --exclude 'venv/' \
        --exclude '__pycache__/' \
        --exclude '.pytest_cache/' \
        --exclude '.mypy_cache/' \
        --exclude '.ruff_cache/' \
        --exclude '.remote-dev.env' \
        --exclude "$OUTPUT_DIR/" \
        --exclude "$LOCAL_OUTPUT_DIR/" \
        --exclude 'data/dataset/' \
        $RSYNC_EXCLUDES \
        "$LOCAL_DIR/" "$(remote_target):$REMOTE_DIR/"; then
      printf '[OK] push complete: %s -> %s:%s/\n' \
        "$LOCAL_DIR" "$(remote_target)" "$REMOTE_DIR"
    else
      push_status=$?
      printf '[FAIL] push failed (exit %s): %s -> %s:%s/\n' \
        "$push_status" "$LOCAL_DIR" "$(remote_target)" "$REMOTE_DIR" >&2
      exit "$push_status"
    fi
    ;;

  pull-output)
    require_remote
    mkdir -p "$LOCAL_DIR/$LOCAL_OUTPUT_DIR"
    rsync -az -e "ssh -F /dev/null -p $REMOTE_PORT" \
      "$(remote_target):$REMOTE_DIR/$OUTPUT_DIR/" "$LOCAL_DIR/$LOCAL_OUTPUT_DIR/"
    ;;

  exec)
    require_remote
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "exec requires a remote command." >&2
      exit 2
    fi
    run_remote "$@"
    ;;

  py)
    require_remote
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "py requires an entry file. Example: $0 py train.py --data PEMS08" >&2
      exit 2
    fi
    ssh_remote "$(remote_cd) && $PY_RUNNER $(quote_args "$@")"
    ;;

  job)
    require_remote
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "job requires a repo script. Example: $0 job scripts/run_seeds.sh" >&2
      exit 2
    fi
    run_remote bash "$@"
    ;;

  ssh)
    require_remote
    ssh -F /dev/null -p "$REMOTE_PORT" -t "$(remote_target)" \
      "$(remote_cd) && exec \"\$SHELL\" -l"
    ;;

  tail)
    require_remote
    shift
    pattern="${1:-$OUTPUT_DIR/log/*.log}"
    ssh_remote "$(remote_cd) && tail -F $pattern"
    ;;

  tmux)
    require_remote
    shift
    session="${1:-remote-dev}"
    shift || true
    if [[ "$#" -eq 0 ]]; then
      ssh -F /dev/null -p "$REMOTE_PORT" -t "$(remote_target)" \
        "$(remote_cd) && tmux new -A -s $(printf '%q' "$session")"
    else
      inner_cmd="$(quote_args "$@")"
      ssh -F /dev/null -p "$REMOTE_PORT" -t "$(remote_target)" \
        "$(remote_cd) && tmux new -A -s $(printf '%q' "$session") $(printf '%q' "$inner_cmd")"
    fi
    ;;

  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
