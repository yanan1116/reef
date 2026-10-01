#!/usr/bin/env bash
# Mount, read-only, everything the evaluation host .36 reads from other machines (sshfs, user-level).
# .36 is for inference and evaluation only: it keeps base models locally (~/.cache/huggingface,
# ~/models) and reads code, the venv and LoRA adapters in place, so it never holds a second copy.
#   .29:/home/yanan/agents -> .36:/home/yanan/agents          code + .venv-finqa, same absolute paths
#   .29:/mnt/disk1t        -> .36:/home/yanan/mnt/dot29-disk1t single-table run adapters
#   .16:/works/yanan       -> .36:/home/yanan/mnt/dot16-works  multi-table run adapters, read through .29's
#                             own read-only mount (scripts/mount_dot16_works.sh): .36 has no key on .16
# Idempotent: a live readable mount is left alone, a stale one is remounted.
# usage (from .29): ssh 10.225.68.36 bash -s < exp_scripts/scripts/mount_eval_host_36.sh
#        unmount:   fusermount -u <mount point>
set -euo pipefail
OPTS=ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,BatchMode=yes,follow_symlinks,idmap=user
# kernel_cache/auto_cache + long attr/entry timeouts: the venv's ~100k files import in ~20 s warm.
FAST=Compression=no,Ciphers=aes128-gcm@openssh.com,kernel_cache,auto_cache,attr_timeout=60,entry_timeout=60

mount_ro() {
  local remote=$1 point=$2
  mkdir -p "$point"
  if mountpoint -q "$point"; then
    if timeout 20 ls "$point" >/dev/null 2>&1; then echo "[mount] $point already mounted"; return 0; fi
    echo "[mount] $point is stale; remounting" >&2
    fusermount -uz "$point"
  fi
  if [ -n "$(ls -A "$point" 2>/dev/null)" ]; then
    echo "[mount] expected $point to be empty before mounting $remote; it holds files (a local copy?)" >&2
    exit 1
  fi
  sshfs "$remote" "$point" -o "$OPTS,$FAST"
  timeout 20 ls "$point" >/dev/null || { echo "[mount] expected $point readable after mounting $remote" >&2; exit 1; }
  echo "[mount] $remote -> $point (read-only)"
}

mount_ro 10.225.68.29:/home/yanan/agents /home/yanan/agents
mount_ro 10.225.68.29:/mnt/disk1t /home/yanan/mnt/dot29-disk1t
mount_ro 10.225.68.29:/home/yanan/mnt/dot16-works /home/yanan/mnt/dot16-works
