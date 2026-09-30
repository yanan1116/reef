#!/usr/bin/env bash
# Mount .16's run-data disk (/works/yanan) read-only on this host (.29) with sshfs, so evaluation
# reads the adapters the training copier keeps on .16 in place instead of copying them here.
# Idempotent: exits 0 when the mount is already live. User-level FUSE; no root needed.
#
# usage: mount_dot16_works.sh        (unmount: fusermount -u /home/yanan/mnt/dot16-works)
set -euo pipefail
MOUNT=/home/yanan/mnt/dot16-works
REMOTE=10.225.68.16:/works/yanan

mkdir -p "$MOUNT"
if mountpoint -q "$MOUNT"; then
  if timeout 20 ls "$MOUNT" >/dev/null 2>&1; then
    echo "[mount] $MOUNT already mounted and readable"
    exit 0
  fi
  echo "[mount] $MOUNT is mounted but unreadable (stale connection); remounting" >&2
  fusermount -uz "$MOUNT"
fi
# ro: evaluation must never write into the training host's run data.
# reconnect + keepalives: survive short network drops; Compression off + AES-GCM: full-speed reads.
sshfs "$REMOTE" "$MOUNT" -o ro,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,BatchMode=yes \
  -o Compression=no,Ciphers=aes128-gcm@openssh.com,follow_symlinks,idmap=user
timeout 20 ls "$MOUNT" >/dev/null || { echo "[mount] expected $MOUNT to be readable after mounting $REMOTE" >&2; exit 1; }
echo "[mount] $REMOTE -> $MOUNT (read-only)"
