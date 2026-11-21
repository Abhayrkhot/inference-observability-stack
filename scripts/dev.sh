#!/usr/bin/env bash
# Run a command in the inferobs-dev container (host networking, repo mounted at /src).
#   scripts/dev.sh 'pytest -q'
# Env: IOS_IMAGE (default inferobs-dev:py3.12). Build it with: docker build -f Dockerfile.dev -t inferobs-dev:py3.12 .
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IOS_IMAGE:-inferobs-dev:py3.12}
sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
git diff --quiet HEAD -- . ':!results' 2>/dev/null || sha="$sha-dirty"
tty=(); [ -t 0 ] && [ -t 1 ] && tty=(-it)
# IOS_DOCKER=1 mounts the host Docker socket (fault experiments start/stop/pause containers).
dock=(); [ "${IOS_DOCKER:-0}" = 1 ] && dock=(-v /var/run/docker.sock:/var/run/docker.sock --group-add "$(stat -c %g /var/run/docker.sock)")
exec docker run --rm "${tty[@]}" "${dock[@]}" --network host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e IOS_GIT_SHA="$sha" -e IOS_HOST="$(hostname -s)" \
  -v "$PWD":/src -w /src "$IMAGE" bash -c 'mkdir -p "$HOME" && eval "$1"' _ "$*"
