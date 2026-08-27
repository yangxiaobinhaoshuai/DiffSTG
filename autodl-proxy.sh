#!/usr/bin/env bash
# AutoDL Ubuntu proxy bootstrap and diagnostics tool.
# Uses a sing-box client.json subscription without exposing subscription secrets.

set -Eeuo pipefail
IFS=$'\n\t'

VERSION="1.1.0"
TOOL_NAME="autodl-proxy"
PROFILE_NAME="managed-vless-reality-v1"
DERIVATION_VERSION="2"

# 持久状态统一放在 /etc：订阅、派生配置和运行模式都属于 root-only 数据。
# /run 只保存可以在重启后丢失的 PID；/var/log 保存非 systemd 模式日志。
INSTALL_PATH="/usr/local/bin/${TOOL_NAME}"
STATE_DIR="/etc/${TOOL_NAME}"
BACKUP_DIR="${STATE_DIR}/backups"
SOURCE_CONFIG="${STATE_DIR}/client.source.json"
MIXED_CONFIG="${STATE_DIR}/client.mixed.json"
TUN_CONFIG="${STATE_DIR}/client.tun.json"
ACTIVE_CONFIG="${STATE_DIR}/config.active.json"
URL_FILE="${STATE_DIR}/subscription.url"
MODE_FILE="${STATE_DIR}/active-mode"
DERIVATION_FILE="${STATE_DIR}/derivation.version"
SHELL_HELPER="${STATE_DIR}/shell.sh"
RUNTIME_DIR="/run/${TOOL_NAME}"
PID_FILE="${RUNTIME_DIR}/sing-box.pid"
LOG_DIR="/var/log/${TOOL_NAME}"
LOG_FILE="${LOG_DIR}/sing-box.log"
SYSTEMD_UNIT="/etc/systemd/system/${TOOL_NAME}.service"
ROOT_HOME="/root"
BASHRC="${ROOT_HOME}/.bashrc"
BASH_PROFILE="${ROOT_HOME}/.bash_profile"
MIXED_PORT=7890
MIXED_HTTP="http://127.0.0.1:${MIXED_PORT}"
MIXED_SOCKS="socks5h://127.0.0.1:${MIXED_PORT}"
BEGIN_MARKER="# >>> autodl-proxy managed block >>>"
END_MARKER="# <<< autodl-proxy managed block <<<"

if [[ -t 1 && "${NO_COLOR:-}" != "1" ]]; then
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_RED=$'\033[31m'
  C_BLUE=$'\033[34m'
  C_RESET=$'\033[0m'
else
  C_GREEN=""
  C_YELLOW=""
  C_RED=""
  C_BLUE=""
  C_RESET=""
fi

say() { printf '%s\n' "$*"; }
info() { printf '%s[INFO]%s %s\n' "$C_BLUE" "$C_RESET" "$*"; }
ok() { printf '%s[ OK ]%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '%s[WARN]%s %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die() { printf '%s[FAIL]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

require_root() {
  [[ "$(id -u)" -eq 0 ]] || die "该命令需要 root。AutoDL 默认使用 root；也可执行 sudo -i 后重试。"
}

# 所有创建/覆盖配置的函数都先写临时文件，再使用同目录 mv 原子替换。
# 这样下载中断或磁盘写入失败时不会留下半份 JSON。
ensure_dirs() {
  install -d -m 0750 "$STATE_DIR" "$BACKUP_DIR"
  install -d -m 0755 "$RUNTIME_DIR" "$LOG_DIR"
}

safe_tmpdir() {
  mktemp -d "${TMPDIR:-/tmp}/${TOOL_NAME}.XXXXXX"
}

atomic_write_text() {
  local target="$1"
  local mode="$2"
  local text_value="$3"
  local tmp_file
  tmp_file="$(mktemp "${target}.tmp.XXXXXX")"
  chmod "$mode" "$tmp_file"
  printf '%s\n' "$text_value" > "$tmp_file"
  mv -f "$tmp_file" "$target"
}

atomic_install_file() {
  local source_file="$1"
  local target_file="$2"
  local mode="${3:-0600}"
  local tmp_file
  tmp_file="$(mktemp "${target_file}.tmp.XXXXXX")"
  install -m "$mode" "$source_file" "$tmp_file"
  mv -f "$tmp_file" "$target_file"
}

mode_read() {
  if [[ -r "$MODE_FILE" ]]; then
    tr -d '[:space:]' < "$MODE_FILE"
  else
    printf 'off'
  fi
}

mode_write() {
  ensure_dirs
  atomic_write_text "$MODE_FILE" 0644 "$1"
}

systemd_available() {
  # AutoDL 多数实例是容器，存在 systemctl 命令并不代表 PID 1 真的是 systemd。
  # show-environment 成功才把 systemd 当作可用的进程管理器。
  [[ -d /run/systemd/system ]] && have systemctl && systemctl show-environment >/dev/null 2>&1
}

sing_box_bin() {
  command -v sing-box 2>/dev/null || true
}

ensure_systemd_unit() {
  # 使用独立的 autodl-proxy.service，避免覆盖 sing-box 软件包自带的 unit。
  # unit 始终读取 config.active.json，模式切换只需原子替换活动配置后重启。
  local sb_bin tmp_file
  sb_bin="$(sing_box_bin)"
  [[ -n "$sb_bin" ]] || die "找不到 sing-box，请先运行 ${TOOL_NAME} install。"
  tmp_file="$(mktemp "${SYSTEMD_UNIT}.tmp.XXXXXX")"
  cat > "$tmp_file" <<EOF
[Unit]
Description=AutoDL managed sing-box proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${sb_bin} run -c ${ACTIVE_CONFIG}
Restart=on-failure
RestartSec=3s
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF
  chmod 0644 "$tmp_file"
  if [[ ! -f "$SYSTEMD_UNIT" ]] || ! cmp -s "$tmp_file" "$SYSTEMD_UNIT"; then
    mv -f "$tmp_file" "$SYSTEMD_UNIT"
    systemctl daemon-reload
  else
    rm -f "$tmp_file"
  fi
}

pid_is_managed() {
  # 非 systemd 模式下，kill 前同时核对 PID 存活和命令行中的配置路径，
  # 防止陈旧 PID 文件误杀用户的其他进程。
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -Fq "${ACTIVE_CONFIG}"
  else
    return 0
  fi
}

managed_running() {
  local pid
  if systemd_available; then
    systemctl is-active --quiet "${TOOL_NAME}.service"
    return
  fi
  [[ -r "$PID_FILE" ]] || return 1
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  pid_is_managed "$pid"
}

stop_managed() {
  local pid i
  if systemd_available; then
    systemctl stop "${TOOL_NAME}.service" >/dev/null 2>&1 || true
  fi
  if [[ -r "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if pid_is_managed "$pid"; then
      kill "$pid" 2>/dev/null || true
      for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
      done
      if pid_is_managed "$pid"; then
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$PID_FILE"
  fi
}

start_managed() {
  # 普通 Ubuntu 使用 systemd；AutoDL 容器没有 init 时退化为 nohup + PID 文件。
  # 两条路径使用同一 ACTIVE_CONFIG，便于 on/off/status 保持一致。
  local sb_bin pid
  ensure_dirs
  [[ -s "$ACTIVE_CONFIG" ]] || die "活动配置不存在。请先运行 configure。"
  sb_bin="$(sing_box_bin)"
  [[ -n "$sb_bin" ]] || die "找不到 sing-box，请先运行 install。"
  if systemd_available; then
    ensure_systemd_unit
    systemctl enable "${TOOL_NAME}.service" >/dev/null 2>&1
    systemctl restart "${TOOL_NAME}.service"
  else
    : > "$LOG_FILE"
    nohup "$sb_bin" run -c "$ACTIVE_CONFIG" >> "$LOG_FILE" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    chmod 0644 "$PID_FILE"
    sleep 0.5
    pid_is_managed "$pid" || {
      tail -n 20 "$LOG_FILE" >&2 || true
      die "sing-box 后台进程启动失败。"
    }
  fi
}

mixed_port_listening() {
  have ss || return 1
  ss -ltnH 2>/dev/null | awk -v port="$MIXED_PORT" '$4 ~ (":" port "$") { found=1 } END { exit !found }'
}

wait_for_process() {
  local i
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    : "$i"
    if managed_running; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

wait_for_mixed() {
  local i
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    if managed_running && mixed_port_listening; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

show_recent_log() {
  if systemd_available && have journalctl; then
    journalctl -u "${TOOL_NAME}.service" --output cat -n 30 --no-pager 2>/dev/null || true
  elif [[ -r "$LOG_FILE" ]]; then
    tail -n 30 "$LOG_FILE" || true
  fi
}

self_install() {
  # 复制到 /usr/local/bin 后，用户不依赖当前目录即可调用 autodl-proxy。
  require_root
  ensure_dirs
  local current
  current="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  if [[ ! -f "$INSTALL_PATH" ]] || ! cmp -s "$current" "$INSTALL_PATH"; then
    install -m 0755 "$current" "$INSTALL_PATH"
    ok "命令已安装：${INSTALL_PATH}"
  elif [[ ! -x "$INSTALL_PATH" ]]; then
    chmod 0755 "$INSTALL_PATH"
    ok "已修复全局命令的可执行权限：${INSTALL_PATH}"
  fi
}

version_ge() {
  # version_ge 1.12.0 1.10.0
  local first
  first="$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)"
  [[ "$first" == "$2" ]]
}

sing_box_version_number() {
  local sb_bin
  sb_bin="$(sing_box_bin)"
  [[ -n "$sb_bin" ]] || return 1
  "$sb_bin" version 2>/dev/null | awk '/sing-box version/ {print $3; exit}'
}

cap_net_admin_available() {
  local cap_hex
  if have capsh; then
    capsh --has-p=cap_net_admin >/dev/null 2>&1
  elif [[ -r /proc/self/status ]]; then
    # CAP_NET_ADMIN is bit 12.
    cap_hex="$(awk '/CapEff:/ {print $2; exit}' /proc/self/status)"
    (( (16#${cap_hex} & 4096) != 0 ))
  else
    return 1
  fi
}

tun_config_present() {
  [[ -s "$TUN_CONFIG" ]] && jq -e '.inbounds[]? | select(.type == "tun")' "$TUN_CONFIG" >/dev/null 2>&1
}

probe_tun_capability() {
  # 仅检查设备文件还不够：容器可能挂载了 /dev/net/tun，却没有 CAP_NET_ADMIN。
  # 临时创建并立即删除一个 TUN 接口，是启用全局路由前最可靠的能力探测。
  local probe_name
  [[ -c /dev/net/tun ]] || return 1
  cap_net_admin_available || return 1
  have ip || return 1
  probe_name="adtp${$}"
  probe_name="${probe_name:0:14}"
  if ip tuntap add dev "$probe_name" mode tun >/dev/null 2>&1; then
    ip link delete "$probe_name" >/dev/null 2>&1 || true
    return 0
  fi
  return 1
}

redact_stream() {
  # doctor/journal 输出只能用于排障，不应成为新的秘密泄漏源。
  # 这里遮蔽 UUID、订阅路径和常见凭据字段；完整 JSON 从不直接打印。
  sed -E \
    -e 's/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/<redacted-uuid>/g' \
    -e 's#(https?://[^/[:space:]]+/)[^[:space:]]+#\1<redacted-path>#g' \
    -e 's/(private_key|public_key|password|uuid|short_id|token|psk)["=: ]+[^, }[:space:]]+/\1=<redacted>/Ig'
}

usage() {
  cat <<'EOF'
AutoDL Proxy Tool

Usage:
  autodl-proxy setup [--reconfigure]
  autodl-proxy check
  autodl-proxy sources on|off|status
  autodl-proxy install
  autodl-proxy configure
  autodl-proxy refresh
  autodl-proxy on [auto|mixed|tun]
  autodl-proxy off
  autodl-proxy status [--json]
  autodl-proxy test
  autodl-proxy doctor
  autodl-proxy env current|on|off
  autodl-proxy run -- <command> [args...]
  autodl-proxy shell install|reload|remove|status

Global shell helpers installed by `shell install` / `setup`:
  proxy_on [auto|mixed|tun]
  proxy_off
  proxy_status
  proxy_test
  proxy_refresh
  proxy_doctor

Subscription refresh is always manual. No cron job or timer is created.
Accepted subscription profile: managed-vless-reality-v1 (this is intentionally not a generic sing-box manager).
`setup` is the full personal-device bootstrap: it intentionally configures pip/conda/npm sources and Bash helpers.
For selective changes, call install/configure/on/sources/shell separately.
EOF
}

print_kv() {
  printf '  %-22s %s\n' "$1" "$2"
}

cmd_check() {
  # check 只做宿主机事实采集，不修改网络；真正的 TUN 创建测试留到 proxy_on。
  local os_name="unknown" os_version="unknown" init_name="unknown"
  local cpu_model="unknown" cuda_version="not found" container_hint="none detected"
  local tun_state="unavailable" cap_state="missing" nft_state="unavailable"
  local systemd_state="unavailable" verdict="mixed only"

  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    os_name="${NAME:-unknown}"
    os_version="${VERSION_ID:-unknown}"
  fi
  if [[ -r /proc/1/comm ]]; then
    init_name="$(tr -d '[:space:]' < /proc/1/comm)"
  fi
  if [[ -r /proc/1/cgroup ]]; then
    container_hint="$(awk -F/ '/docker|containerd|kubepods|lxc/ {print $NF; found=1; exit} END {if (!found) print "none detected"}' /proc/1/cgroup)"
  fi
  if have lscpu; then
    cpu_model="$(lscpu 2>/dev/null | awk -F: '/Model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
  elif [[ -r /proc/cpuinfo ]]; then
    cpu_model="$(awk -F: '/model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  fi
  [[ -n "$cpu_model" ]] || cpu_model="unknown"

  systemd_available && systemd_state="available"
  [[ -c /dev/net/tun ]] && tun_state="device present"
  cap_net_admin_available && cap_state="available"
  if have nft && nft list tables >/dev/null 2>&1; then
    nft_state="available"
  fi
  if [[ "$tun_state" == "device present" && "$cap_state" == "available" ]]; then
    verdict="TUN candidate (will probe safely during proxy_on)"
  fi
  if have nvcc; then
    cuda_version="$(nvcc --version 2>/dev/null | awk '/release/ {gsub(/,/, "", $5); print $5; exit}')"
  elif [[ -r /usr/local/cuda/version.json ]] && have jq; then
    cuda_version="$(jq -r '.cuda.version // "unknown"' /usr/local/cuda/version.json 2>/dev/null || true)"
  fi

  say "AutoDL host check"
  print_kv "OS" "${os_name} ${os_version}"
  print_kv "Kernel / arch" "$(uname -r) / $(uname -m)"
  print_kv "PID 1 / container" "${init_name} / ${container_hint}"
  print_kv "systemd" "$systemd_state"
  print_kv "CPU" "${cpu_model}"
  print_kv "CPU cores" "$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 'unknown')"
  if have free; then
    print_kv "Memory" "$(free -h | awk '/^Mem:/ {print $2 " total, " $7 " available"}')"
  fi
  if have df; then
    print_kv "System disk" "$(df -hP / | awk 'NR==2 {print $2 " total, " $4 " free (" $5 " used)"}')"
    if [[ -d /root/autodl-tmp ]]; then
      print_kv "AutoDL data disk" "$(df -hP /root/autodl-tmp | awk 'NR==2 {print $2 " total, " $4 " free (" $5 " used)"}')"
    fi
  fi
  if have nvidia-smi; then
    say "  GPU(s):"
    if ! nvidia-smi --query-gpu=index,name,memory.total,driver_version,temperature.gpu \
      --format=csv,noheader 2>/dev/null | sed 's/^/    /'; then
      nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null | sed 's/^/    /' || warn "nvidia-smi 查询失败"
    fi
  else
    print_kv "GPU" "nvidia-smi not found"
  fi
  print_kv "CUDA toolkit" "$cuda_version"
  if have ip; then
    print_kv "Default route" "$(ip route show default 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -r /etc/resolv.conf ]]; then
    print_kv "DNS" "$(awk '/^nameserver/ {printf "%s ", $2}' /etc/resolv.conf)"
  fi
  print_kv "/dev/net/tun" "$tun_state"
  print_kv "CAP_NET_ADMIN" "$cap_state"
  print_kv "nftables" "$nft_state"
  print_kv "Proxy capability" "$verdict"
  if mixed_port_listening; then
    warn "127.0.0.1:${MIXED_PORT} 已被占用："
    ss -ltnp 2>/dev/null | awk -v port="$MIXED_PORT" '$4 ~ (":" port "$") {print "    " $0}' || true
  else
    print_kv "Port ${MIXED_PORT}" "free"
  fi
}

PIP_CONFIG="${ROOT_HOME}/.config/pip/pip.conf"
CONDA_CONFIG="${ROOT_HOME}/.condarc"
NPM_CONFIG="${ROOT_HOME}/.npmrc"
SOURCE_BACKUP_DIR="${BACKUP_DIR}/sources"

backup_source_file_once() {
  # 只保留“首次启用国内源之前”的基线。重复 sources on 不会覆盖原始备份，
  # 因而 sources off 始终能恢复用户最初的配置，而不是上一次工具生成的配置。
  local key="$1" path="$2"
  install -d -m 0700 "$SOURCE_BACKUP_DIR"
  if [[ -e "${SOURCE_BACKUP_DIR}/${key}.saved" || -e "${SOURCE_BACKUP_DIR}/${key}.absent" ]]; then
    return
  fi
  if [[ -e "$path" ]]; then
    cp -p "$path" "${SOURCE_BACKUP_DIR}/${key}.saved"
  else
    : > "${SOURCE_BACKUP_DIR}/${key}.absent"
    chmod 0600 "${SOURCE_BACKUP_DIR}/${key}.absent"
  fi
}

restore_source_file() {
  local key="$1" path="$2"
  if [[ -e "${SOURCE_BACKUP_DIR}/${key}.saved" ]]; then
    install -d -m 0755 "$(dirname "$path")"
    cp -p "${SOURCE_BACKUP_DIR}/${key}.saved" "$path"
  elif [[ -e "${SOURCE_BACKUP_DIR}/${key}.absent" ]]; then
    rm -f "$path"
  else
    warn "${path} 没有原始备份，保持不变。"
  fi
}

sources_on() {
  # 只修改 root 用户的 pip/conda/npm 配置；刻意不替换 apt 源，避免不同
  # Ubuntu 版本、CUDA 仓库和 AutoDL 自带源之间出现兼容问题。
  require_root
  ensure_dirs
  local configured=0
  local pip_target="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
  local npm_target="https://registry.npmmirror.com"
  local current_value

  if have python3 && python3 -m pip --version >/dev/null 2>&1; then
    backup_source_file_once pip "$PIP_CONFIG"
    current_value="$(awk -F= '/^[[:space:]]*index-url/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$PIP_CONFIG" 2>/dev/null || true)"
    if [[ "$current_value" != "$pip_target" ]]; then
      install -d -m 0755 "$(dirname "$PIP_CONFIG")"
      PIP_CONFIG_FILE="$PIP_CONFIG" python3 -m pip config set global.index-url "$pip_target" >/dev/null
      chmod 0600 "$PIP_CONFIG"
      ok "pip 已切换到清华 TUNA"
    else
      ok "pip 已是清华 TUNA，跳过写入。"
    fi
    configured=1
  else
    warn "未找到可用的 python3 -m pip，跳过 pip。"
  fi

  if have conda; then
    backup_source_file_once conda "$CONDA_CONFIG"
    if [[ -r "$CONDA_CONFIG" ]] && \
      grep -Eq '^[[:space:]]*show_channel_urls:[[:space:]]*true' "$CONDA_CONFIG" && \
      grep -Fq 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main' "$CONDA_CONFIG" && \
      grep -Fq 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r' "$CONDA_CONFIG" && \
      grep -Fq 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2' "$CONDA_CONFIG" && \
      grep -Eq '^[[:space:]]*conda-forge:[[:space:]]*https://mirrors\.tuna\.tsinghua\.edu\.cn/anaconda/cloud' "$CONDA_CONFIG" && \
      grep -Eq '^[[:space:]]*pytorch:[[:space:]]*https://mirrors\.tuna\.tsinghua\.edu\.cn/anaconda/cloud' "$CONDA_CONFIG"; then
      ok "conda 已是清华 TUNA，跳过写入。"
    else
      conda config --file "$CONDA_CONFIG" --set show_channel_urls true >/dev/null
      conda config --file "$CONDA_CONFIG" --remove-key default_channels >/dev/null 2>&1 || true
      conda config --file "$CONDA_CONFIG" --add default_channels \
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2" >/dev/null
      conda config --file "$CONDA_CONFIG" --add default_channels \
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r" >/dev/null
      conda config --file "$CONDA_CONFIG" --add default_channels \
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" >/dev/null
      conda config --file "$CONDA_CONFIG" --set custom_channels.conda-forge \
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >/dev/null
      conda config --file "$CONDA_CONFIG" --set custom_channels.pytorch \
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud" >/dev/null
      chmod 0600 "$CONDA_CONFIG"
      ok "conda 已切换到清华 TUNA"
    fi
    configured=1
  else
    warn "未找到 conda，跳过 conda。"
  fi

  if have npm; then
    backup_source_file_once npm "$NPM_CONFIG"
    current_value="$(npm config --userconfig "$NPM_CONFIG" get registry 2>/dev/null || true)"
    if [[ "$current_value" != "$npm_target" && "$current_value" != "${npm_target}/" ]]; then
      npm config --userconfig "$NPM_CONFIG" set registry "$npm_target" >/dev/null
      chmod 0600 "$NPM_CONFIG"
      ok "npm 已切换到 npmmirror"
    else
      ok "npm 已是 npmmirror，跳过写入。"
    fi
    configured=1
  else
    warn "未找到 npm，跳过 npm。"
  fi

  [[ "$configured" -eq 1 ]] || warn "没有发现可配置的 pip、conda 或 npm。"
}

sources_off() {
  require_root
  restore_source_file pip "$PIP_CONFIG"
  restore_source_file conda "$CONDA_CONFIG"
  restore_source_file npm "$NPM_CONFIG"
  ok "已恢复首次修改前的 pip、conda、npm 配置。"
}

sources_status() {
  say "Development source status"
  if [[ -r "$PIP_CONFIG" ]]; then
    print_kv "pip index" "$(awk -F= '/^[[:space:]]*index-url/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$PIP_CONFIG")"
  else
    print_kv "pip index" "no managed user config"
  fi
  if [[ -r "$CONDA_CONFIG" ]]; then
    print_kv "conda config" "$CONDA_CONFIG"
    awk '/mirrors\.tuna\.tsinghua\.edu\.cn/ {print "    " $0}' "$CONDA_CONFIG" || true
  else
    print_kv "conda config" "not found"
  fi
  if have npm; then
    print_kv "npm registry" "$(npm config --userconfig "$NPM_CONFIG" get registry 2>/dev/null || true)"
  else
    print_kv "npm registry" "npm not found"
  fi
}

cmd_sources() {
  case "${1:-status}" in
    on) sources_on ;;
    off) sources_off ;;
    status) sources_status ;;
    *) die "用法：${TOOL_NAME} sources on|off|status" ;;
  esac
}

cmd_install() {
  # sing-box 二进制仅来自官方 APT 仓库。官方源不可用时明确失败，
  # 不自动使用 GitHub 加速站或未知镜像下载可执行文件。
  require_root
  [[ "$(uname -s)" == "Linux" ]] || die "install 仅支持 Linux。"
  have apt-get || die "当前系统没有 apt-get；本工具面向 Ubuntu/Debian。"
  local installed_now=0 pkg
  local missing_packages=()

  for pkg in ca-certificates curl jq iproute2 netcat-openbsd libcap2-bin nftables procps pciutils; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -Fq 'install ok installed'; then
      missing_packages+=("$pkg")
    fi
  done
  if [[ "${#missing_packages[@]}" -gt 0 ]]; then
    info "安装 ${#missing_packages[@]} 个缺少的基础依赖……"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing_packages[@]}"
  else
    ok "基础诊断依赖已齐全，跳过 apt-get update/install。"
  fi

  if ! have sing-box; then
    installed_now=1
    info "添加 sing-box 官方 APT 仓库……"
    install -d -m 0755 /etc/apt/keyrings /etc/apt/sources.list.d
    local tmp_key tmp_source
    tmp_key="$(mktemp /etc/apt/keyrings/sagernet.asc.tmp.XXXXXX)"
    if ! curl -fsSL --connect-timeout 15 --max-time 60 \
      https://sing-box.app/gpg.key -o "$tmp_key"; then
      rm -f "$tmp_key"
      die "无法访问 sing-box 官方密钥。请从官方 Release 下载对应架构的 .deb 上传到实例后安装。"
    fi
    chmod 0644 "$tmp_key"
    mv -f "$tmp_key" /etc/apt/keyrings/sagernet.asc

    tmp_source="$(mktemp /etc/apt/sources.list.d/sagernet.sources.tmp.XXXXXX)"
    cat > "$tmp_source" <<'EOF'
Types: deb
URIs: https://deb.sagernet.org/
Suites: *
Components: *
Enabled: yes
Signed-By: /etc/apt/keyrings/sagernet.asc
EOF
    chmod 0644 "$tmp_source"
    mv -f "$tmp_source" /etc/apt/sources.list.d/sagernet.sources
    if ! apt-get update || ! DEBIAN_FRONTEND=noninteractive apt-get install -y sing-box; then
      die "sing-box 官方 APT 安装失败。未使用第三方二进制；请查看 https://sing-box.sagernet.org/installation/package-manager/"
    fi
  fi
  have sing-box || die "sing-box 安装后仍不可用。"
  if [[ "$installed_now" -eq 1 ]] && systemd_available; then
    # The package service may auto-start before this tool has installed a validated config.
    systemctl disable --now sing-box.service >/dev/null 2>&1 || true
  fi
  ok "$(sing-box version 2>/dev/null | head -n 1)"
}

validate_subscription_url() {
  # URL 会作为长期凭据保存。默认强制 HTTPS，并拒绝可破坏 curl config
  # 语法的引号、反斜杠和换行，防止配置注入。
  local url="$1"
  [[ "$url" == https://* ]] || {
    if [[ "${AUTODL_PROXY_ALLOW_HTTP:-0}" != "1" || "$url" != http://* ]]; then
      die "订阅地址必须使用 HTTPS。测试环境确需 HTTP 时可临时设置 AUTODL_PROXY_ALLOW_HTTP=1。"
    fi
  }
  [[ "$url" != *$'\n'* && "$url" != *$'\r'* && "$url" != *' '* && "$url" != *$'\t'* ]] \
    || die "订阅地址包含非法空白字符。"
  [[ "$url" != *\"* && "$url" != *\\* ]] || die "订阅地址包含不支持的字符。"
}

download_subscription() {
  # 不执行 `curl "$url"`：那会让完整 token 出现在 /proc/<pid>/cmdline。
  # URL 写入 0600 临时 curl 配置，curl 进程参数中只出现配置文件路径。
  local url="$1" output="$2" curl_cfg
  curl_cfg="$(mktemp "${TMPDIR:-/tmp}/${TOOL_NAME}.curl.XXXXXX")"
  chmod 0600 "$curl_cfg"
  printf 'url = "%s"\n' "$url" > "$curl_cfg"
  if ! curl --config "$curl_cfg" --fail --location --silent \
    --connect-timeout 15 --max-time 90 --output "$output"; then
    rm -f "$curl_cfg" "$output"
    return 1
  fi
  rm -f "$curl_cfg"
  [[ -s "$output" ]]
}

sing_box_check_quiet() {
  local config="$1" check_log
  check_log="$(mktemp "${TMPDIR:-/tmp}/${TOOL_NAME}.check.XXXXXX")"
  if ! sing-box check -c "$config" > "$check_log" 2>&1; then
    warn "sing-box check 未通过："
    redact_stream < "$check_log" | tail -n 12 >&2
    rm -f "$check_log"
    return 1
  fi
  rm -f "$check_log"
}

collect_protected_cidrs() {
  # TUN 改写默认路由前，保护当前 SSH 对端和 VPS 节点地址走原系统路由。
  # 即使透明代理配置有误，watchdog 仍有机会通过现有连接完成回滚。
  local source_json="$1" host ip list_file host_file
  list_file="$(mktemp "${TMPDIR:-/tmp}/${TOOL_NAME}.cidrs.XXXXXX")"
  host_file="$(mktemp "${TMPDIR:-/tmp}/${TOOL_NAME}.hosts.XXXXXX")"

  if [[ -n "${SSH_CONNECTION:-}" ]]; then
    printf '%s\n' "${SSH_CONNECTION%% *}" >> "$host_file"
  fi
  jq -r '.outbounds[]? | select(.type == "vless" or .type == "trojan") | .server // empty' \
    "$source_json" 2>/dev/null >> "$host_file" || true

  while IFS= read -r host; do
    [[ -n "$host" ]] || continue
    if [[ "$host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      printf '%s/32\n' "$host" >> "$list_file"
    elif [[ "$host" == *:* ]]; then
      printf '%s/128\n' "$host" >> "$list_file"
    elif have getent; then
      while IFS= read -r ip; do
        [[ -n "$ip" ]] || continue
        if [[ "$ip" == *:* ]]; then
          printf '%s/128\n' "$ip"
        else
          printf '%s/32\n' "$ip"
        fi
      done < <(getent ahosts "$host" 2>/dev/null | awk '{print $1}' | sort -u) >> "$list_file"
    fi
  done < "$host_file"
  rm -f "$host_file"

  if [[ -s "$list_file" ]]; then
    sort -u "$list_file" | jq -Rsc 'split("\n") | map(select(length > 0))'
  else
    printf '[]\n'
  fi
  rm -f "$list_file"
}

validate_source_shape() {
  # sing-box check 负责 schema；这里额外检查本工具依赖的业务约定，
  # 避免“JSON 合法但只有 direct”之类配置被误当成代理配置启用。
  local source_json="$1"
  if ! jq -e '(.inbounds | type == "array") and (.outbounds | type == "array")' "$source_json" >/dev/null 2>&1; then
    warn "订阅不符合 ${PROFILE_NAME}：inbounds/outbounds 必须是数组。"
    return 1
  fi
  if ! jq -e --argjson port "$MIXED_PORT" \
    '[.inbounds[]? | select(.type == "mixed" and .listen_port == $port)] | length > 0' \
    "$source_json" >/dev/null 2>&1; then
    warn "订阅不符合 ${PROFILE_NAME}：缺少监听 ${MIXED_PORT} 的 mixed inbound。"
    return 1
  fi
  if ! jq -e '[.outbounds[]? | select(.type == "vless")] | length > 0' "$source_json" >/dev/null 2>&1; then
    warn "订阅不符合 ${PROFILE_NAME}：缺少 VLESS outbound。"
    return 1
  fi
  if ! jq -e '[.outbounds[]? | select(.type == "selector" and .tag == "proxy")] | length > 0' \
    "$source_json" >/dev/null 2>&1; then
    warn "订阅不符合 ${PROFILE_NAME}：缺少 tag=proxy 的 selector outbound。"
    return 1
  fi
  if ! jq -e '.route.final == "proxy"' "$source_json" >/dev/null 2>&1; then
    warn "订阅不符合 ${PROFILE_NAME}：route.final 必须显式为 proxy，防止意外直连。"
    return 1
  fi
  return 0
}

derive_configs() {
  # 永不原地编辑订阅：source 是可审计基线，mixed/tun 是本机运行副本。
  # mixed 删除 tun inbound；tun 只在当前版本/权限允许时增加安全增强字段。
  local source_json="$1" mixed_json="$2" tun_json="$3"
  local protected version enable_redirect=false modern=false

  jq -e . "$source_json" >/dev/null 2>&1 || die "订阅响应不是合法 JSON。"
  validate_source_shape "$source_json" || die "订阅未通过 ${PROFILE_NAME} 契约校验；本工具不接受通用 sing-box 模板。"
  sing_box_check_quiet "$source_json" || die "原始 client.json 与当前 sing-box 不兼容。"

  jq '.inbounds |= map(select(.type != "tun"))' "$source_json" > "$mixed_json"
  jq -e '[.inbounds[]? | select(.type == "mixed")] | length > 0' "$mixed_json" >/dev/null \
    || die "无法生成 mixed-only 配置。"
  sing_box_check_quiet "$mixed_json" || die "生成的 mixed-only 配置无效。"

  protected="$(collect_protected_cidrs "$source_json")"
  version="$(sing_box_version_number || true)"
  if [[ -n "$version" ]] && version_ge "$version" "1.10.0"; then
    modern=true
  fi
  if [[ "$modern" == true ]] && have nft && cap_net_admin_available && nft list tables >/dev/null 2>&1; then
    enable_redirect=true
  fi

  if jq -e '.inbounds[]? | select(.type == "tun")' "$source_json" >/dev/null 2>&1; then
    if [[ "$modern" == true ]]; then
      jq --argjson protected "$protected" --argjson redirect "$enable_redirect" '
        .inbounds |= map(
          if .type == "tun" then
            .route_exclude_address = (((.route_exclude_address // []) + $protected) | unique)
            | if $redirect then .auto_redirect = true else . end
          else . end
        )
      ' "$source_json" > "$tun_json"
    else
      cp "$source_json" "$tun_json"
      warn "sing-box 版本低于 1.10，TUN 派生配置不添加 auto_redirect/route_exclude_address。"
    fi
    # auto_redirect 或 route_exclude_address 可能与旧模板中的字段冲突。
    # 增强配置校验失败时退回服务端原始 TUN，不影响 mixed 的可用性。
    if ! sing_box_check_quiet "$tun_json"; then
      warn "TUN 安全增强与当前配置冲突，将保留订阅原始 TUN；mixed 模式不受影响。"
      cp "$source_json" "$tun_json"
      sing_box_check_quiet "$tun_json" || die "订阅原始 TUN 配置无效。"
    fi
  else
    cp "$source_json" "$tun_json"
    warn "订阅中没有 TUN inbound；auto 模式将使用 mixed。"
  fi
  chmod 0600 "$source_json" "$mixed_json" "$tun_json"
}

stage_subscription() {
  # 所有下载、解析和 sing-box 校验都在 staging 目录完成；调用方只在
  # 整组文件全部通过后才安装，因此三个运行配置不会出现版本错配。
  local url="$1" staging_dir="$2"
  validate_subscription_url "$url"
  if ! download_subscription "$url" "${staging_dir}/client.source.json"; then
    die "订阅下载失败。URL 已隐藏；请检查订阅状态、DNS 和网络连通性。"
  fi
  derive_configs "${staging_dir}/client.source.json" \
    "${staging_dir}/client.mixed.json" "${staging_dir}/client.tun.json"
}

install_staged_configs() {
  local staging_dir="$1"
  ensure_dirs
  atomic_install_file "${staging_dir}/client.source.json" "$SOURCE_CONFIG" 0600
  atomic_install_file "${staging_dir}/client.mixed.json" "$MIXED_CONFIG" 0600
  atomic_install_file "${staging_dir}/client.tun.json" "$TUN_CONFIG" 0600
  atomic_write_text "$DERIVATION_FILE" 0644 "$DERIVATION_VERSION"
}

staged_subscription_is_current() {
  local staging_dir="$1"
  [[ -s "$SOURCE_CONFIG" && -s "$MIXED_CONFIG" && -s "$TUN_CONFIG" ]] || return 1
  [[ -r "$DERIVATION_FILE" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$DERIVATION_FILE")" == "$DERIVATION_VERSION" ]] || return 1
  cmp -s "${staging_dir}/client.source.json" "$SOURCE_CONFIG" || return 1
  # 不直接比较 TUN 派生文件，因为其中包含每次 SSH 登录动态保护的对端地址；
  # 但仍重新做 schema 检查，避免损坏的本地副本被“未变化”判断掩盖。
  sing_box_check_quiet "$MIXED_CONFIG" || return 1
  sing_box_check_quiet "$TUN_CONFIG"
}

backup_existing_configs() {
  # configure 可用于更换订阅。覆盖前按时间戳保留旧配置和旧 URL，
  # 且备份目录为 0700，避免备份机制反而扩大凭据可见范围。
  local backup_stamp backup_path candidate found=0
  backup_stamp="$(date +%Y%m%d-%H%M%S).$$"
  backup_path="${BACKUP_DIR}/configs/${backup_stamp}"
  for candidate in "$SOURCE_CONFIG" "$MIXED_CONFIG" "$TUN_CONFIG" "$URL_FILE" "$DERIVATION_FILE"; do
    if [[ -e "$candidate" ]]; then
      if [[ "$found" -eq 0 ]]; then
        install -d -m 0700 "$backup_path"
        found=1
      fi
      install -m 0600 "$candidate" "${backup_path}/$(basename "$candidate")"
    fi
  done
}

backup_existing_url() {
  local backup_stamp backup_path
  [[ -e "$URL_FILE" ]] || return 0
  backup_stamp="$(date +%Y%m%d-%H%M%S).$$"
  backup_path="${BACKUP_DIR}/urls/${backup_stamp}"
  install -d -m 0700 "$backup_path"
  install -m 0600 "$URL_FILE" "${backup_path}/$(basename "$URL_FILE")"
}

save_subscription_url_if_changed() {
  local url="$1" current_url=""
  [[ -r "$URL_FILE" ]] && current_url="$(cat "$URL_FILE")"
  if [[ "$current_url" != "$url" ]]; then
    backup_existing_url
    atomic_write_text "$URL_FILE" 0600 "$url"
    return 0
  fi
  return 1
}

activate_mode_config() {
  local requested_mode="$1" selected
  case "$requested_mode" in
    mixed) selected="$MIXED_CONFIG" ;;
    tun) selected="$TUN_CONFIG" ;;
    *) die "内部错误：未知配置模式 ${requested_mode}" ;;
  esac
  [[ -s "$selected" ]] || die "${selected} 不存在，请先 configure。"
  atomic_install_file "$selected" "$ACTIVE_CONFIG" 0600
}

query_ip_direct() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    curl --noproxy '*' -fsS --connect-timeout 5 --max-time 10 https://api.ipify.org 2>/dev/null \
    || env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl --noproxy '*' -fsS --connect-timeout 5 --max-time 10 \
      https://www.cloudflare.com/cdn-cgi/trace 2>/dev/null | awk -F= '/^ip=/ {print $2; exit}'
}

query_ip_mixed() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    curl --proxy "$MIXED_HTTP" -fsS --connect-timeout 5 --max-time 12 https://api.ipify.org 2>/dev/null \
    || env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl --proxy "$MIXED_HTTP" -fsS --connect-timeout 5 --max-time 12 \
      https://www.cloudflare.com/cdn-cgi/trace 2>/dev/null | awk -F= '/^ip=/ {print $2; exit}'
}

health_mixed_quiet() {
  # HTTP 和 socks5h 都通过才算 mixed 可用；socks5h 同时验证 DNS 经代理解析。
  managed_running && mixed_port_listening && \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl --proxy "$MIXED_HTTP" -fsS -o /dev/null --connect-timeout 5 --max-time 12 \
      https://www.gstatic.com/generate_204 && \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl --proxy "$MIXED_SOCKS" -fsS -o /dev/null --connect-timeout 5 --max-time 12 \
      https://api.ipify.org
}

health_tun_quiet() {
  # TUN 测试显式清除 shell 代理变量并使用 --noproxy，确保成功来自透明路由，
  # 而不是遗留的 HTTP_PROXY 让测试产生假阳性。
  local dns_ok=1
  if have timeout; then
    timeout 10 getent hosts api.ipify.org >/dev/null 2>&1 || dns_ok=0
  else
    getent hosts api.ipify.org >/dev/null 2>&1 || dns_ok=0
  fi
  managed_running && [[ "$dns_ok" -eq 1 ]] && \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl --noproxy '*' -fsS -o /dev/null --connect-timeout 5 --max-time 15 \
      https://www.gstatic.com/generate_204 && \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl --noproxy '*' -fsS -o /dev/null --connect-timeout 5 --max-time 15 \
      https://api.ipify.org
}

prompt_subscription_url() {
  local url
  [[ -r /dev/tty ]] || die "configure 需要交互终端，以便隐藏输入订阅 URL。"
  printf '请输入完整 client.json HTTPS URL（输入不回显）：' >&2
  IFS= read -r -s url < /dev/tty
  printf '\n' >&2
  [[ -n "$url" ]] || die "订阅 URL 不能为空。"
  printf '%s' "$url"
}

LAST_CONFIG_CHANGED=0

apply_subscription_url() {
  # failure_policy=reuse-existing 仅供重复 setup：订阅站临时不可达时，
  # 已有设备继续使用通过校验的本地配置；configure/reconfigure 仍严格失败。
  local url="$1" failure_policy="${2:-strict}" staging_dir
  LAST_CONFIG_CHANGED=0

  staging_dir="$(safe_tmpdir)"
  if ! (stage_subscription "$url" "$staging_dir"); then
    rm -rf "$staging_dir"
    if [[ "$failure_policy" == "reuse-existing" && -s "$SOURCE_CONFIG" && -s "$MIXED_CONFIG" ]] && \
      validate_source_shape "$SOURCE_CONFIG" && sing_box_check_quiet "$MIXED_CONFIG"; then
      warn "订阅暂时无法刷新，重复 setup 将继续使用现有已验证配置。"
      return 0
    fi
    die "配置失败，现有可用配置未被修改。"
  fi

  if staged_subscription_is_current "$staging_dir"; then
    if save_subscription_url_if_changed "$url"; then
      ok "订阅内容未变化，仅更新了 root-only URL。"
    else
      ok "订阅配置与 URL 均未变化，跳过备份、写入和服务重启。"
    fi
    rm -rf "$staging_dir"
    LAST_CONFIG_CHANGED=0
    return 0
  fi

  backup_existing_configs
  install_staged_configs "$staging_dir"
  atomic_write_text "$URL_FILE" 0600 "$url"
  rm -rf "$staging_dir"
  LAST_CONFIG_CHANGED=1

  ok "订阅已保存并通过 ${PROFILE_NAME} / sing-box 校验。"
}

cmd_configure() {
  # 只从 /dev/tty 无回显读取 URL，避免 shell history、命令参数和普通 stdin
  # 日志记录订阅凭据。首次 setup 也会走完全相同的路径。
  require_root
  ensure_dirs
  have curl || die "缺少 curl，请先运行 install。"
  have jq || die "缺少 jq，请先运行 install。"
  have sing-box || die "缺少 sing-box，请先运行 install。"

  local url
  url="$(prompt_subscription_url)"
  apply_subscription_url "$url" strict

  if [[ "$LAST_CONFIG_CHANGED" -eq 1 ]] && managed_running; then
    warn "代理进程仍使用旧的活动副本；请执行 proxy_on 重新验证并切换到新订阅。"
  fi

  if [[ ! -s "${STATE_DIR}/baseline-ip" ]]; then
    atomic_write_text "${STATE_DIR}/baseline-ip" 0644 "$(query_ip_direct || printf 'unknown')"
  fi
  ok "秘密 URL 未输出到日志。"
}

rollback_configs_from() {
  local backup_dir="$1"
  atomic_install_file "${backup_dir}/client.source.json" "$SOURCE_CONFIG" 0600
  atomic_install_file "${backup_dir}/client.mixed.json" "$MIXED_CONFIG" 0600
  atomic_install_file "${backup_dir}/client.tun.json" "$TUN_CONFIG" 0600
  if [[ -e "${backup_dir}/derivation.version" ]]; then
    atomic_install_file "${backup_dir}/derivation.version" "$DERIVATION_FILE" 0644
  else
    rm -f "$DERIVATION_FILE"
  fi
}

restart_exact_mode() {
  # refresh 后保持用户原来的运行模式，不在后台擅自把 mixed 升级为 TUN。
  local running_mode="$1"
  stop_managed
  activate_mode_config "$running_mode"
  (start_managed) || return 1
  if [[ "$running_mode" == "mixed" ]]; then
    wait_for_mixed && health_mixed_quiet
  else
    wait_for_process && health_tun_quiet
  fi
}

cmd_refresh() {
  # 手动更新采用事务式流程：
  # 1) staging 中校验新订阅；2) 备份旧三件套；3) 重启并做实际联网测试；
  # 4) 任一步失败就恢复旧配置并再次验证。这里不会创建 cron/timer。
  require_root
  [[ -r "$URL_FILE" ]] || die "尚未保存订阅 URL，请先运行 configure。"
  [[ -s "$SOURCE_CONFIG" && -s "$MIXED_CONFIG" && -s "$TUN_CONFIG" ]] \
    || die "现有配置不完整，请先运行 configure。"

  local url staging_dir backup_dir previous_mode was_running=0
  url="$(cat "$URL_FILE")"
  staging_dir="$(safe_tmpdir)"
  backup_dir="$(safe_tmpdir)"
  previous_mode="$(mode_read)"
  managed_running && was_running=1

  info "手动刷新订阅（URL 已隐藏）……"
  if ! (stage_subscription "$url" "$staging_dir"); then
    rm -rf "$staging_dir" "$backup_dir"
    die "刷新失败，仍使用原配置。"
  fi

  if staged_subscription_is_current "$staging_dir"; then
    rm -rf "$staging_dir" "$backup_dir"
    ok "订阅内容未变化，跳过备份、写入和服务重启。"
    return 0
  fi

  install -m 0600 "$SOURCE_CONFIG" "${backup_dir}/client.source.json"
  install -m 0600 "$MIXED_CONFIG" "${backup_dir}/client.mixed.json"
  install -m 0600 "$TUN_CONFIG" "${backup_dir}/client.tun.json"
  if [[ -e "$DERIVATION_FILE" ]]; then
    install -m 0644 "$DERIVATION_FILE" "${backup_dir}/derivation.version"
  fi
  install_staged_configs "$staging_dir"

  if [[ "$was_running" -eq 1 && ( "$previous_mode" == "mixed" || "$previous_mode" == "tun" ) ]]; then
    info "新配置已校验，重启 ${previous_mode} 并进行健康检查……"
    if ! restart_exact_mode "$previous_mode"; then
      warn "新配置运行测试失败，正在恢复上一版本。"
      stop_managed
      rollback_configs_from "$backup_dir"
      if ! restart_exact_mode "$previous_mode"; then
        mode_write off
        rm -rf "$staging_dir" "$backup_dir"
        die "旧配置恢复后仍无法启动，请运行 doctor 查看诊断。"
      fi
      mode_write "$previous_mode"
      rm -rf "$staging_dir" "$backup_dir"
      die "订阅未生效，已成功恢复上一版本。"
    fi
    mode_write "$previous_mode"
  fi

  rm -rf "$staging_dir" "$backup_dir"
  ok "订阅已手动刷新。未创建 cron 或定时任务。"
}

unmanaged_port_conflict() {
  mixed_port_listening && ! managed_running
}

start_mixed_verified() {
  # mixed 端口若属于其他进程，只报告冲突，绝不为了“自动修复”而 kill 用户进程。
  if unmanaged_port_conflict; then
    warn "127.0.0.1:${MIXED_PORT} 已被非本工具进程占用，拒绝终止该进程。"
    ss -ltnp 2>/dev/null | awk -v port="$MIXED_PORT" '$4 ~ (":" port "$") {print "    " $0}' >&2 || true
    return 1
  fi
  stop_managed
  activate_mode_config mixed
  (start_managed) || return 1
  if ! wait_for_mixed; then
    warn "mixed 监听端口未就绪。"
    show_recent_log | redact_stream >&2
    return 1
  fi
  if ! health_mixed_quiet; then
    warn "mixed HTTP/SOCKS 连通性测试失败。"
    show_recent_log | redact_stream >&2
    return 1
  fi
  mode_write mixed
  return 0
}

rollback_to_mixed() {
  # mixed 是所有失败路径的安全落点：它不改默认路由，最不容易中断 SSH。
  stop_managed
  activate_mode_config mixed
  if (start_managed) && wait_for_mixed && health_mixed_quiet; then
    mode_write mixed
    return 0
  fi
  mode_write off
  return 1
}

active_mode_is_healthy() {
  local current_mode
  current_mode="$(mode_read)"
  managed_running || return 1
  case "$current_mode" in
    mixed)
      [[ -s "$ACTIVE_CONFIG" && -s "$MIXED_CONFIG" ]] || return 1
      cmp -s "$ACTIVE_CONFIG" "$MIXED_CONFIG" || return 1
      health_mixed_quiet
      ;;
    tun)
      [[ -s "$ACTIVE_CONFIG" && -s "$TUN_CONFIG" ]] || return 1
      cmp -s "$ACTIVE_CONFIG" "$TUN_CONFIG" || return 1
      health_tun_quiet
      ;;
    *) return 1 ;;
  esac
}

cmd_tun_watchdog() {
  # watchdog 由 nohup 独立运行。即使 SSH 断开导致主 setup/on 进程退出，
  # 60 秒内没有 commit 文件也会主动停止 TUN 并恢复 mixed。
  local commit_file="${1:-}"
  [[ -n "$commit_file" && "$commit_file" == "${RUNTIME_DIR}/tun-commit."* ]] || exit 2
  sleep 60
  if [[ ! -e "$commit_file" ]]; then
    warn "TUN 安全确认超时，watchdog 正在恢复 mixed。"
    rollback_to_mixed || warn "watchdog 无法恢复 mixed，请运行 doctor。"
  fi
}

cmd_on() {
  # 安全顺序固定为 mixed -> 联网验证 -> TUN 能力探测 -> 带 watchdog 的 TUN。
  # auto 允许降级；显式 tun 失败会返回非零，但仍保留已经验证过的 mixed。
  require_root
  ensure_dirs
  local requested="${1:-auto}" commit_file watchdog_pid runner
  case "$requested" in
    auto|mixed|tun) ;;
    *) die "用法：${TOOL_NAME} on [auto|mixed|tun]" ;;
  esac
  [[ -s "$MIXED_CONFIG" ]] || die "尚未配置订阅，请先运行 configure。"

  if active_mode_is_healthy; then
    case "$requested:$(mode_read)" in
      auto:mixed|auto:tun|mixed:mixed|tun:tun)
        ok "现有 $(mode_read) 服务、活动配置和联网检查均正常，跳过重启。"
        return 0
        ;;
    esac
  fi

  info "先启动 mixed 并验证 HTTP、SOCKS5h 与出口连通性……"
  if ! start_mixed_verified; then
    stop_managed
    mode_write off
    die "mixed 基础验证失败；为保护 SSH，未尝试 TUN。"
  fi
  ok "mixed proxy 可用：127.0.0.1:${MIXED_PORT}"

  if [[ "$requested" == "mixed" ]]; then
    ok "当前模式：mixed"
    return 0
  fi
  if ! tun_config_present; then
    if [[ "$requested" == "tun" ]]; then
      warn "订阅没有 TUN inbound，已保留可用的 mixed 模式。"
      return 1
    fi
    warn "订阅没有 TUN inbound，auto 已降级为 mixed。"
    return 0
  fi
  if ! probe_tun_capability; then
    if [[ "$requested" == "tun" ]]; then
      warn "实例缺少可用的 /dev/net/tun 或 CAP_NET_ADMIN，已保留 mixed。"
      return 1
    fi
    warn "AutoDL 容器不具备完整 TUN 权限，auto 已降级为 mixed。"
    return 0
  fi

  info "TUN 能力探测通过，启动 60 秒自动回滚保护……"
  commit_file="${RUNTIME_DIR}/tun-commit.$$"
  rm -f "$commit_file"
  runner="$INSTALL_PATH"
  [[ -x "$runner" ]] || runner="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  # watchdog 必须在停止 mixed、改写活动配置之前启动。
  nohup "$runner" _tun-watchdog "$commit_file" >> "$LOG_FILE" 2>&1 < /dev/null &
  watchdog_pid=$!

  stop_managed
  activate_mode_config tun
  if (start_managed) && wait_for_process && health_tun_quiet; then
    mode_write tun
    : > "$commit_file"
    kill "$watchdog_pid" >/dev/null 2>&1 || true
    wait "$watchdog_pid" 2>/dev/null || true
    rm -f "$commit_file"
    ok "当前模式：TUN（系统透明代理，配置内仍按 split 规则分流）"
    return 0
  fi

  : > "$commit_file"
  kill "$watchdog_pid" >/dev/null 2>&1 || true
  wait "$watchdog_pid" 2>/dev/null || true
  rm -f "$commit_file"
  warn "TUN 健康检查失败，正在恢复 mixed。"
  show_recent_log | redact_stream >&2
  rollback_to_mixed || die "TUN 与 mixed 均无法恢复，请运行 doctor。"
  if [[ "$requested" == "tun" ]]; then
    die "TUN 未启用；mixed 已恢复并保持可用。"
  fi
  warn "auto 已安全降级为 mixed。"
}

cmd_off() {
  require_root
  stop_managed
  if systemd_available; then
    systemctl disable "${TOOL_NAME}.service" >/dev/null 2>&1 || true
  fi
  mode_write off
  ok "代理已停止。当前 shell 请使用 proxy_off，或执行：eval \"\$(${TOOL_NAME} env off)\""
}

query_geo_direct() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    curl --noproxy '*' -fsS --connect-timeout 5 --max-time 10 https://ipwho.is/ 2>/dev/null || true
}

query_geo_mixed() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    curl --proxy "$MIXED_HTTP" -fsS --connect-timeout 5 --max-time 12 https://ipwho.is/ 2>/dev/null || true
}

format_geo() {
  local json="$1"
  if [[ -n "$json" ]] && jq -e '.success == true' >/dev/null 2>&1 <<< "$json"; then
    jq -r '[.country, .region, .city, (.connection.isp // ""), (.connection.asn // "")] | map(select(. != "")) | join(" / ")' <<< "$json"
  else
    printf 'location unavailable'
  fi
}

cmd_status() {
  # system IP 强制绕过应用层代理；mixed IP 强制指定本机 mixed 端口。
  # 两者并列展示，才能区分“代理进程存在”和“当前命令确实走了代理”。
  local json_output=0 current_mode running=false default_ip="unavailable" mixed_ip="unavailable"
  local default_geo="" mixed_geo="" sb_version="not installed" config_time="unknown" baseline="unknown"
  [[ "${1:-}" == "--json" ]] && json_output=1
  if [[ "$json_output" -eq 1 ]]; then
    have jq || die "status --json 需要 jq，请先运行 install。"
  fi
  current_mode="$(mode_read)"
  managed_running && running=true
  default_ip="$(query_ip_direct || true)"
  [[ -n "$default_ip" ]] || default_ip="unavailable"
  default_geo="$(query_geo_direct)"
  if [[ "$running" == true ]] && mixed_port_listening; then
    mixed_ip="$(query_ip_mixed || true)"
    [[ -n "$mixed_ip" ]] || mixed_ip="unavailable"
    mixed_geo="$(query_geo_mixed)"
  fi
  if have sing-box; then
    sb_version="$(sing_box_version_number || sing-box version 2>/dev/null | head -n 1)"
  fi
  if [[ -e "$SOURCE_CONFIG" ]]; then
    config_time="$(stat -c '%y' "$SOURCE_CONFIG" 2>/dev/null | cut -d. -f1 || printf 'unknown')"
  fi
  [[ -r "${STATE_DIR}/baseline-ip" ]] && baseline="$(cat "${STATE_DIR}/baseline-ip")"

  if [[ "$json_output" -eq 1 ]]; then
    jq -n \
      --arg mode "$current_mode" --argjson running "$running" \
      --arg profile "$PROFILE_NAME" \
      --arg default_ip "$default_ip" --arg mixed_ip "$mixed_ip" \
      --arg default_location "$(format_geo "$default_geo")" \
      --arg mixed_location "$(format_geo "$mixed_geo")" \
      --arg sing_box_version "$sb_version" --arg config_updated "$config_time" \
      --arg baseline_ip "$baseline" \
      '{profile:$profile,mode:$mode,running:$running,default_egress:{ip:$default_ip,location:$default_location},mixed_egress:{ip:$mixed_ip,location:$mixed_location},baseline_ip:$baseline_ip,sing_box_version:$sing_box_version,config_updated:$config_updated}'
    return
  fi

  say "AutoDL proxy status"
  print_kv "Profile" "$PROFILE_NAME"
  print_kv "Mode" "$current_mode"
  print_kv "Process" "$running"
  print_kv "sing-box" "$sb_version"
  print_kv "Config updated" "$config_time"
  print_kv "Recorded baseline IP" "$baseline"
  print_kv "Current system IP" "$default_ip"
  print_kv "System location" "$(format_geo "$default_geo")"
  print_kv "Explicit mixed IP" "$mixed_ip"
  print_kv "Mixed location" "$(format_geo "$mixed_geo")"
  if [[ "$current_mode" == "mixed" ]]; then
    say "  提示：系统 IP 是直连出口；使用 proxy_on 导出的环境或 --proxy 后才显示 mixed 出口。"
  elif [[ "$current_mode" == "tun" ]]; then
    say "  提示：系统请求已进入 TUN，但 client.json 的国内/私网规则仍可能直连。"
  fi
}

test_vps_ports() {
  # 输出仅包含节点 tag 和端口，不打印 server/UUID/Reality 参数。
  local failures=0 tag host port
  [[ -s "$SOURCE_CONFIG" ]] || return 0
  say "VPS TCP reachability"
  while IFS=$'\t' read -r tag host port; do
    [[ -n "$host" && -n "$port" ]] || continue
    if have nc && nc -z -w 4 "$host" "$port" >/dev/null 2>&1; then
      printf '  [OK]   %s tcp/%s\n' "$tag" "$port"
    else
      printf '  [FAIL] %s tcp/%s\n' "$tag" "$port"
      failures=$((failures + 1))
    fi
  done < <(jq -r '.outbounds[]? | select(.type == "vless") | [.tag, .server, (.server_port|tostring)] | @tsv' "$SOURCE_CONFIG")
  return "$failures"
}

cmd_test() {
  local failures=0 ip_value
  say "Proxy connectivity test"

  if getent hosts www.gstatic.com >/dev/null 2>&1; then
    ok "DNS resolution"
  else
    warn "DNS resolution failed"
    failures=$((failures + 1))
  fi

  ip_value="$(query_ip_direct || true)"
  if [[ -n "$ip_value" ]]; then
    ok "system egress: ${ip_value}"
  else
    warn "system egress query failed"
    failures=$((failures + 1))
  fi

  if managed_running && mixed_port_listening; then
    if env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl --proxy "$MIXED_HTTP" -fsS -o /dev/null --connect-timeout 5 --max-time 12 \
      https://www.gstatic.com/generate_204; then
      ok "mixed HTTP proxy"
    else
      warn "mixed HTTP proxy failed"
      failures=$((failures + 1))
    fi
    if env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      curl --proxy "$MIXED_SOCKS" -fsS -o /dev/null --connect-timeout 5 --max-time 12 \
      https://api.ipify.org; then
      ok "mixed SOCKS5h proxy"
    else
      warn "mixed SOCKS5h proxy failed"
      failures=$((failures + 1))
    fi
    ip_value="$(query_ip_mixed || true)"
    [[ -n "$ip_value" ]] && ok "mixed egress: ${ip_value}"
  else
    warn "mixed listener is not running"
    failures=$((failures + 1))
  fi

  test_vps_ports || failures=$((failures + $?))
  [[ "$failures" -eq 0 ]] || die "检测完成：${failures} 项失败。运行 doctor 查看详情。"
  ok "所有代理检测通过。"
}

emit_env_on() {
  # 独立脚本不能直接修改父 shell，因此 env 子命令输出一段可 eval 的代码。
  # 第一次启用时保存用户原有变量；重复 source/.bashrc 不会覆盖这份基线。
  cat <<'EOF' | sed -e "s|@MIXED_HTTP@|${MIXED_HTTP}|g" -e "s|@MIXED_SOCKS@|${MIXED_SOCKS}|g"
if [[ ${_AUTODL_PROXY_ENV_ACTIVE:-0} != 1 ]]; then
  if [[ ${HTTP_PROXY+x} ]]; then export _AUTODL_PROXY_HAD_HTTP_PROXY=1 _AUTODL_PROXY_OLD_HTTP_PROXY="$HTTP_PROXY"; else export _AUTODL_PROXY_HAD_HTTP_PROXY=0 _AUTODL_PROXY_OLD_HTTP_PROXY=""; fi
  if [[ ${HTTPS_PROXY+x} ]]; then export _AUTODL_PROXY_HAD_HTTPS_PROXY=1 _AUTODL_PROXY_OLD_HTTPS_PROXY="$HTTPS_PROXY"; else export _AUTODL_PROXY_HAD_HTTPS_PROXY=0 _AUTODL_PROXY_OLD_HTTPS_PROXY=""; fi
  if [[ ${ALL_PROXY+x} ]]; then export _AUTODL_PROXY_HAD_ALL_PROXY=1 _AUTODL_PROXY_OLD_ALL_PROXY="$ALL_PROXY"; else export _AUTODL_PROXY_HAD_ALL_PROXY=0 _AUTODL_PROXY_OLD_ALL_PROXY=""; fi
  if [[ ${http_proxy+x} ]]; then export _AUTODL_PROXY_HAD_http_proxy=1 _AUTODL_PROXY_OLD_http_proxy="$http_proxy"; else export _AUTODL_PROXY_HAD_http_proxy=0 _AUTODL_PROXY_OLD_http_proxy=""; fi
  if [[ ${https_proxy+x} ]]; then export _AUTODL_PROXY_HAD_https_proxy=1 _AUTODL_PROXY_OLD_https_proxy="$https_proxy"; else export _AUTODL_PROXY_HAD_https_proxy=0 _AUTODL_PROXY_OLD_https_proxy=""; fi
  if [[ ${all_proxy+x} ]]; then export _AUTODL_PROXY_HAD_all_proxy=1 _AUTODL_PROXY_OLD_all_proxy="$all_proxy"; else export _AUTODL_PROXY_HAD_all_proxy=0 _AUTODL_PROXY_OLD_all_proxy=""; fi
  if [[ ${NO_PROXY+x} ]]; then export _AUTODL_PROXY_HAD_NO_PROXY=1 _AUTODL_PROXY_OLD_NO_PROXY="$NO_PROXY"; else export _AUTODL_PROXY_HAD_NO_PROXY=0 _AUTODL_PROXY_OLD_NO_PROXY=""; fi
  if [[ ${no_proxy+x} ]]; then export _AUTODL_PROXY_HAD_no_proxy=1 _AUTODL_PROXY_OLD_no_proxy="$no_proxy"; else export _AUTODL_PROXY_HAD_no_proxy=0 _AUTODL_PROXY_OLD_no_proxy=""; fi
fi
export HTTP_PROXY="@MIXED_HTTP@"
export HTTPS_PROXY="@MIXED_HTTP@"
export ALL_PROXY="@MIXED_SOCKS@"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" all_proxy="$ALL_PROXY"
export NO_PROXY="127.0.0.1,localhost,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16${_AUTODL_PROXY_HAD_NO_PROXY:+${_AUTODL_PROXY_OLD_NO_PROXY:+,${_AUTODL_PROXY_OLD_NO_PROXY}}}"
export no_proxy="$NO_PROXY"
export _AUTODL_PROXY_ENV_ACTIVE=1
EOF
}

emit_env_off() {
  # 只恢复/删除本工具接管过的变量，不假设用户启用代理前一定是空值。
  cat <<'EOF'
if [[ ${_AUTODL_PROXY_ENV_ACTIVE:-0} == 1 ]]; then
  if [[ ${_AUTODL_PROXY_HAD_HTTP_PROXY:-0} == 1 ]]; then export HTTP_PROXY="$_AUTODL_PROXY_OLD_HTTP_PROXY"; else unset HTTP_PROXY; fi
  if [[ ${_AUTODL_PROXY_HAD_HTTPS_PROXY:-0} == 1 ]]; then export HTTPS_PROXY="$_AUTODL_PROXY_OLD_HTTPS_PROXY"; else unset HTTPS_PROXY; fi
  if [[ ${_AUTODL_PROXY_HAD_ALL_PROXY:-0} == 1 ]]; then export ALL_PROXY="$_AUTODL_PROXY_OLD_ALL_PROXY"; else unset ALL_PROXY; fi
  if [[ ${_AUTODL_PROXY_HAD_http_proxy:-0} == 1 ]]; then export http_proxy="$_AUTODL_PROXY_OLD_http_proxy"; else unset http_proxy; fi
  if [[ ${_AUTODL_PROXY_HAD_https_proxy:-0} == 1 ]]; then export https_proxy="$_AUTODL_PROXY_OLD_https_proxy"; else unset https_proxy; fi
  if [[ ${_AUTODL_PROXY_HAD_all_proxy:-0} == 1 ]]; then export all_proxy="$_AUTODL_PROXY_OLD_all_proxy"; else unset all_proxy; fi
  if [[ ${_AUTODL_PROXY_HAD_NO_PROXY:-0} == 1 ]]; then export NO_PROXY="$_AUTODL_PROXY_OLD_NO_PROXY"; else unset NO_PROXY; fi
  if [[ ${_AUTODL_PROXY_HAD_no_proxy:-0} == 1 ]]; then export no_proxy="$_AUTODL_PROXY_OLD_no_proxy"; else unset no_proxy; fi
  unset _AUTODL_PROXY_ENV_ACTIVE _AUTODL_PROXY_HAD_HTTP_PROXY _AUTODL_PROXY_OLD_HTTP_PROXY
  unset _AUTODL_PROXY_HAD_HTTPS_PROXY _AUTODL_PROXY_OLD_HTTPS_PROXY _AUTODL_PROXY_HAD_ALL_PROXY _AUTODL_PROXY_OLD_ALL_PROXY
  unset _AUTODL_PROXY_HAD_http_proxy _AUTODL_PROXY_OLD_http_proxy _AUTODL_PROXY_HAD_https_proxy _AUTODL_PROXY_OLD_https_proxy
  unset _AUTODL_PROXY_HAD_all_proxy _AUTODL_PROXY_OLD_all_proxy _AUTODL_PROXY_HAD_NO_PROXY _AUTODL_PROXY_OLD_NO_PROXY
  unset _AUTODL_PROXY_HAD_no_proxy _AUTODL_PROXY_OLD_no_proxy
fi
EOF
}

cmd_env() {
  case "${1:-current}" in
    on) emit_env_on ;;
    off) emit_env_off ;;
    current)
      if managed_running && [[ "$(mode_read)" == "mixed" ]] && mixed_port_listening; then
        emit_env_on
      else
        emit_env_off
      fi
      ;;
    *) die "用法：${TOOL_NAME} env current|on|off" ;;
  esac
}

cmd_run() {
  # 为单条命令注入代理变量，不污染调用者当前 shell，适合训练/下载脚本。
  [[ "${1:-}" == "--" ]] && shift
  [[ "$#" -gt 0 ]] || die "用法：${TOOL_NAME} run -- <command> [args...]"
  if ! managed_running || ! mixed_port_listening; then
    die "mixed proxy 未运行，请先执行 proxy_on。"
  fi
  exec env \
    HTTP_PROXY="$MIXED_HTTP" HTTPS_PROXY="$MIXED_HTTP" ALL_PROXY="$MIXED_SOCKS" \
    http_proxy="$MIXED_HTTP" https_proxy="$MIXED_HTTP" all_proxy="$MIXED_SOCKS" \
    NO_PROXY="127.0.0.1,localhost,::1" no_proxy="127.0.0.1,localhost,::1" \
    "$@"
}

write_shell_helper() {
  # .bashrc 只 source 这个稳定入口；快捷函数负责在 on/off 后立即 eval，
  # 新终端则仅依据本地进程状态加载环境，不联网、不刷新、不启动服务。
  local tmp_file
  ensure_dirs
  tmp_file="$(mktemp "${SHELL_HELPER}.tmp.XXXXXX")"
  cat > "$tmp_file" <<'EOF'
# Managed by autodl-proxy. Do not place subscription secrets in this file.
case ":${PATH}:" in
  *:/usr/local/bin:*) ;;
  *) export PATH="/usr/local/bin:${PATH}" ;;
esac

proxy_on() {
  command autodl-proxy on "${1:-auto}" || return $?
  eval "$(command autodl-proxy env current)"
}

proxy_off() {
  command autodl-proxy off || return $?
  eval "$(command autodl-proxy env off)"
}

proxy_status() { command autodl-proxy status "$@"; }
proxy_test() { command autodl-proxy test "$@"; }
proxy_refresh() { command autodl-proxy refresh "$@"; }
proxy_doctor() { command autodl-proxy doctor "$@"; }

# New shells inherit proxy variables only when the managed mixed service is live.
# This reads local state; it does not refresh the subscription or start a process.
if command -v autodl-proxy >/dev/null 2>&1; then
  eval "$(command autodl-proxy env current)"
fi
EOF
  chmod 0644 "$tmp_file"
  if [[ -f "$SHELL_HELPER" ]] && cmp -s "$tmp_file" "$SHELL_HELPER"; then
    rm -f "$tmp_file"
  else
    mv -f "$tmp_file" "$SHELL_HELPER"
  fi
}

remove_managed_block() {
  # 通过精确起止标记删除托管块，避免修改用户在 .bashrc 中的其他内容。
  local file="$1" tmp_file file_mode
  [[ -e "$file" ]] || return 0
  tmp_file="$(mktemp "${file}.tmp.XXXXXX")"
  awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" '
    $0 == begin {
      if (pending && previous != "") print previous
      pending=0
      skip=1
      next
    }
    $0 == end { skip=0; next }
    !skip {
      if (pending) print previous
      previous=$0
      pending=1
    }
    END { if (pending) print previous }
  ' "$file" > "$tmp_file"
  file_mode="$(stat -c '%a' "$file" 2>/dev/null || printf '644')"
  chmod "$file_mode" "$tmp_file"
  mv -f "$tmp_file" "$file"
}

install_managed_block() {
  # 先剔除旧托管块再追加最新版，从而保证 setup/shell install 可重复执行。
  local file="$1" tmp_file file_mode backup_stamp
  install -d -m 0700 "${BACKUP_DIR}/shell"
  [[ -e "$file" ]] || : > "$file"
  if ! grep -Fq "$BEGIN_MARKER" "$file"; then
    backup_stamp="$(date +%Y%m%d-%H%M%S).$$"
    cp -p "$file" "${BACKUP_DIR}/shell/$(basename "$file").${backup_stamp}"
  fi
  tmp_file="$(mktemp "${file}.tmp.XXXXXX")"
  awk -v begin="$BEGIN_MARKER" -v end="$END_MARKER" '
    $0 == begin {
      if (pending && previous != "") print previous
      pending=0
      skip=1
      next
    }
    $0 == end { skip=0; next }
    !skip {
      if (pending) print previous
      previous=$0
      pending=1
    }
    END { if (pending) print previous }
  ' "$file" > "$tmp_file"
  {
    printf '\n%s\n' "$BEGIN_MARKER"
    printf '[ -r "%s" ] && source "%s"\n' "$SHELL_HELPER" "$SHELL_HELPER"
    printf '%s\n' "$END_MARKER"
  } >> "$tmp_file"
  file_mode="$(stat -c '%a' "$file" 2>/dev/null || printf '644')"
  chmod "$file_mode" "$tmp_file"
  if cmp -s "$tmp_file" "$file"; then
    rm -f "$tmp_file"
  else
    mv -f "$tmp_file" "$file"
  fi
}

shell_install() {
  # /usr/local/bin 提供全局命令，.bashrc 提供能修改“当前 shell”的包装函数。
  # 两层缺一不可：普通子进程无法把 HTTP_PROXY 反向写回父 shell。
  require_root
  self_install
  write_shell_helper
  install_managed_block "$BASHRC"
  if [[ -e "$BASH_PROFILE" ]] && \
    ! grep -Eq '(^|[[:space:]])(source|\.)[[:space:]]+.*\.bashrc' "$BASH_PROFILE"; then
    install_managed_block "$BASH_PROFILE"
  fi
  ok "已安装 Bash 快捷函数：proxy_on / proxy_off / proxy_status / proxy_test / proxy_refresh / proxy_doctor"
  say "刷新当前终端：source ${BASHRC}"
}

shell_remove() {
  require_root
  remove_managed_block "$BASHRC"
  remove_managed_block "$BASH_PROFILE"
  rm -f "$SHELL_HELPER"
  ok "已移除 autodl-proxy 的 Bash 托管区块；原始备份保留在 ${BACKUP_DIR}/shell。"
  say "刷新当前终端：source ${BASHRC}"
}

shell_status() {
  print_kv "Global command" "$(command -v autodl-proxy 2>/dev/null || printf 'not installed')"
  if [[ -r "$SHELL_HELPER" ]]; then
    print_kv "Shell helper" "$SHELL_HELPER"
  else
    print_kv "Shell helper" "not installed"
  fi
  if [[ -r "$BASHRC" ]] && grep -Fq "$BEGIN_MARKER" "$BASHRC"; then
    print_kv ".bashrc integration" "installed"
  else
    print_kv ".bashrc integration" "not installed"
  fi
  print_kv "Subscription timer" "disabled (manual proxy_refresh only)"
}

cmd_shell() {
  case "${1:-status}" in
    install) shell_install ;;
    reload)
      say "独立脚本无法修改父 shell；请在当前终端执行："
      say "source ${BASHRC}"
      ;;
    remove) shell_remove ;;
    status) shell_status ;;
    *) die "用法：${TOOL_NAME} shell install|reload|remove|status" ;;
  esac
}

cmd_doctor() {
  # doctor 只打印结构性信息和经过 redact_stream 的日志，不输出完整配置。
  say "===== AutoDL proxy doctor (secrets redacted) ====="
  cmd_check
  say
  shell_status
  say
  sources_status
  say
  if [[ -s "$SOURCE_CONFIG" ]] && have jq; then
    say "Config summary"
    print_kv "profile" "$PROFILE_NAME"
    print_kv "derivation" "$(tr -d '[:space:]' < "$DERIVATION_FILE" 2>/dev/null || printf 'legacy')"
    print_kv "route.final" "$(jq -r '.route.final // "missing"' "$SOURCE_CONFIG")"
    say "  Outbounds (type/tag only):"
    jq -r '.outbounds[]? | [.type, .tag] | @tsv' "$SOURCE_CONFIG" | sed 's/^/    /'
    say "  Inbounds (type/tag only):"
    jq -r '.inbounds[]? | [.type, .tag] | @tsv' "$SOURCE_CONFIG" | sed 's/^/    /'
    if sing_box_check_quiet "$SOURCE_CONFIG"; then
      print_kv "sing-box check" "passed"
    else
      print_kv "sing-box check" "failed"
    fi
  else
    warn "尚无订阅配置。"
  fi
  say
  say "Recent service log (redacted)"
  show_recent_log | redact_stream | tail -n 80
  say "===== end doctor ====="
}

cmd_setup() {
  # setup 是可交互的一站式入口；每个子步骤仍可单独执行和重试。
  # source .bashrc 必须由调用 setup 的父 shell 在 setup 成功后执行。
  local reconfigure=0 url
  case "${1:-}" in
    "") ;;
    --reconfigure) reconfigure=1 ;;
    *) die "用法：${TOOL_NAME} setup [--reconfigure]" ;;
  esac

  require_root
  self_install
  cmd_check
  say
  sources_on
  cmd_install
  if [[ "$reconfigure" -eq 1 || ! -r "$URL_FILE" ]]; then
    cmd_configure
  else
    url="$(cat "$URL_FILE")"
    info "复用已保存的 root-only 订阅 URL；使用 --reconfigure 可显式更换。"
    apply_subscription_url "$url" reuse-existing
  fi
  shell_install
  cmd_on auto
  say
  ok "AutoDL proxy setup 完成。"
  say "当前终端立即加载快捷函数：source ${BASHRC}"
  say "以后手动刷新订阅：proxy_refresh  或  ${TOOL_NAME} refresh"
  say "未创建 cron job 或 systemd timer。"
}

main() {
  # 内部 _tun-watchdog 仅供独立回滚进程调用，不属于日常用户接口。
  local command_name="${1:-help}"
  [[ "$#" -gt 0 ]] && shift || true
  case "$command_name" in
    setup) cmd_setup "$@" ;;
    check) cmd_check "$@" ;;
    sources) cmd_sources "$@" ;;
    install) self_install; cmd_install "$@" ;;
    configure) cmd_configure "$@" ;;
    refresh) cmd_refresh "$@" ;;
    on) cmd_on "$@" ;;
    off) cmd_off "$@" ;;
    status) cmd_status "$@" ;;
    test) cmd_test "$@" ;;
    doctor) cmd_doctor "$@" ;;
    env) cmd_env "$@" ;;
    run) cmd_run "$@" ;;
    shell) cmd_shell "$@" ;;
    _tun-watchdog) cmd_tun_watchdog "$@" ;;
    help|-h|--help) usage ;;
    version|-V|--version) printf '%s %s\n' "$TOOL_NAME" "$VERSION" ;;
    *) usage >&2; die "未知命令：${command_name}" ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
