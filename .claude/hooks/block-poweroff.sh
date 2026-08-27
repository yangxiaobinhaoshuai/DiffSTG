#!/usr/bin/env bash
# AutoDL 的 /usr/bin/shutdown 不是 systemd 的 shutdown：它忽略全部参数，
# 直接 kill supervisord 关掉容器。所以 `shutdown --help` / `shutdown -c`
# 一样会关机。任何形式的关机命令都必须由人来敲，agent 不许碰。
#
# 只拦命令位置上的关机命令（行首，或 ; && || | ( ` 换行 之后，可带 sudo/env），
# 这样 `grep -n shutdown script.sh` 这类读操作不会被误伤。
input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null)"
if printf '%s' "$cmd" \
  | grep -qE '(^|[;&|(`$]|&&|\|\||[[:space:]]&[[:space:]])[[:space:]]*((sudo|env|nohup|exec|xargs|timeout)[[:space:]]+)*(/usr/bin/|/usr/sbin/|/sbin/|/bin/)?(shutdown|poweroff|halt|reboot)([[:space:]]|$|;|&|\|)'; then
  echo "BLOCKED: on this host shutdown/poweroff/halt/reboot kill the container immediately and ignore every argument (yes, even --help and -c). Never run them; ask the user to do it." >&2
  exit 2
fi
exit 0
