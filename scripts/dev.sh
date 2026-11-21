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
exec docker run --rm "${tty[@]}" --network host \
  --user "$(id -u):$(id -g)" -e HOME=/tmp/home -e IOS_GIT_SHA="$sha" -e IOS_HOST="$(hostname -s)" \
  -v "$PWD":/src -w /src "$IMAGE" bash -c 'mkdir -p "$HOME" && eval "$1"' _ "$*"
