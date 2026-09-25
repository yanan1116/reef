#!/usr/bin/env bash
# Build the SAO training image from this reef fork checkout (branch sao), with
# the same slime base digest as every earlier SAO image, so the only variable is
# the Reef source. The fork's commits carry what the old two-stage build patched
# in (configurable SGLang health timeout, Reef's runtime dependencies), so this
# is a single docker/Dockerfile.reef build with no patch step.
#
# Tag: reef:sao-<last commit touching Reef source>, i.e. excluding exp_scripts/,
# so editing experiment scripts never implies a rebuild. start_stack.sh and
# run_formal.sh derive the same default tag.
set -euo pipefail
BASE_DIGEST=sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c
REEF_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GIT=(git -C "$REEF_ROOT" -c safe.directory='*')
SRC_COMMIT=$("${GIT[@]}" log -1 --format=%h -- . ':(exclude)exp_scripts')
BRANCH=$("${GIT[@]}" rev-parse --abbrev-ref HEAD)
TAG=reef:sao-$SRC_COMMIT
LOG=${LOG:-/home/yanan/reef-sao/logs/docker-build-$SRC_COMMIT.log}
mkdir -p "$(dirname "$LOG")"
echo "[build] $(date '+%F %T %Z') $REEF_ROOT branch=$BRANCH reef-src=$SRC_COMMIT base=slimerl/slime@$BASE_DIGEST -> $TAG" | tee "$LOG"
# Only Reef source must be committed; exp_scripts/ is excluded from the image by .dockerignore.
if [ -n "$("${GIT[@]}" status --porcelain -- . ':(exclude)exp_scripts')" ]; then
  echo "[build] Reef source has uncommitted changes; commit them first so the tag names what was built" | tee -a "$LOG"
  exit 1
fi
cd "$REEF_ROOT"
docker build -f docker/Dockerfile.reef --build-arg SLIME_IMAGE_TAG="latest@$BASE_DIGEST" \
  --label reef.commit="$SRC_COMMIT" --label reef.branch="$BRANCH" -t "$TAG" . >>"$LOG" 2>&1
docker run --rm --entrypoint python3 "$TAG" -c "import reef, tomli_w, sqlalchemy, alembic, reef_eval, reef_client; print('reef imports OK')" | tee -a "$LOG"
echo "[build] DONE $(date '+%F %T %Z') -> $TAG" | tee -a "$LOG"
