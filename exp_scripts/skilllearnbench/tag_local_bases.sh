#!/usr/bin/env bash
# Pull every base image the prepared tasks build FROM once, and tag it slb-local.invalid/<image>, so task builds
# resolve their base locally instead of on Docker Hub (prepare_tasks.py local_base_dockerfile; user, 2026-10-20).
#
#   tag_local_bases.sh TASKS_DIR [TASKS_DIR ...]
set -eu
for dir in "$@"; do cat "$dir"/*/environment/Dockerfile; done | grep '^FROM' | sed -E 's#^FROM +(--platform=[^ ]+ +)?##; s#^slb-local\.invalid/##' | awk '{print $1}' | sort -u |
while read -r image; do
  docker image inspect "$image" > /dev/null 2>&1 || docker pull -q --platform linux/amd64 "$image" > /dev/null
  docker tag "$image" "slb-local.invalid/$image"
  echo "tagged $image -> slb-local.invalid/$image ($(docker image inspect -f '{{.Id}}' "$image" | cut -c1-19))"
done
