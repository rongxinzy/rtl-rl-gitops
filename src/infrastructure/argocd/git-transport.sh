#!/bin/bash
run_bounded() {
  local seconds=$1 group watch timer rc
  shift
  /usr/bin/setsid "$@" &
  group=$!
  (
    sleep "$seconds" & timer=$!
    trap 'kill "$timer" 2>/dev/null; exit 0' TERM
    wait "$timer"
    kill -TERM -- "-$group" 2>/dev/null
    sleep 2 & timer=$!
    wait "$timer"
    kill -KILL -- "-$group" 2>/dev/null
  ) >/dev/null 2>&1 &
  watch=$!
  wait "$group"; rc=$?
  # Always reap remaining helpers even if the main git exits before its children.
  kill -KILL -- "-$group" 2>/dev/null || true
  kill -TERM "$watch" 2>/dev/null || true
  wait "$watch" 2>/dev/null || true
  return "$rc"
}
# Public GitHub HTTPS only; never substitute credentials or disable TLS checks.
case "${1-}" in fetch|clone|ls-remote) ;; *) exec /usr/bin/git "$@" ;; esac
is_github=false
for arg in "$@"; do case "$arg" in https://github.com/*) is_github=true ;; esac; done
if [ "${1-}" = fetch ]; then
  remote=$(/usr/bin/git config --get remote.origin.url 2>/dev/null || true)
  case "$remote" in https://github.com/*) is_github=true ;; esac
fi
[ "$is_github" = true ] || exec /usr/bin/git "$@"
last=124
# All addresses were verified in GitHub's official /meta web list on 2026-09-13.
# Revalidate this list as part of platform maintenance. TLS hostname stays github.com.
for ip in 20.201.28.151 20.27.177.119 20.205.243.166; do
  run_bounded 2 /usr/bin/bash -c 'exec 3<>/dev/tcp/$1/443' -- "$ip" 2>/dev/null || continue
  run_bounded 18 /usr/bin/git -c "http.curloptResolve=github.com:443:$ip" -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=10 "$@"
  last=$?
  [ "$last" -eq 0 ] && exit 0
  echo "GitHub transport attempt failed; trying next verified endpoint" >&2
done
exit "$last"
