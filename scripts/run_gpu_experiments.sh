#!/usr/bin/env bash
# Run the vLLM experiments on this machine's GPU (the box: RTX 3050 Ti Laptop, 4 GB).
#
# The GPU is shared with another project's queue, so each chunk takes the same flock
# (GPU_LOCK, default ~/qildata/logs/gpu.lock) and waits until no other GPU job is running.
#   scripts/run_gpu_experiments.sh            # chunks a and b
#   scripts/run_gpu_experiments.sh a          # trace, capacity, saturation, agreement, overhead
#   scripts/run_gpu_experiments.sh b          # faults (needs results/$TAG/capacity.json from chunk a)
set -euo pipefail
cd "$(dirname "$0")/.."
LOCK=${GPU_LOCK:-$HOME/qildata/logs/gpu.lock}
TAG=${TAG:-box-3050ti}
MODEL=${MODEL:-qwen2.5-0.5b-instruct}
C="docker compose -f deploy/compose.yaml -f deploy/compose.vllm.yaml"
CF="$C -f deploy/compose.faults.yaml"
dev() { IOS_DOCKER=1 scripts/dev.sh "$@"; }
log() { echo "== $(date -Is) load $(cut -d' ' -f1 /proc/loadavg) $*"; }

wait_gpu_free() {
  while docker ps -q --filter ancestor=qil-dev:latest --filter ancestor=qil-ref:latest | grep -q . ||
        [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    sleep 30
  done
}

stack_down() {
  docker compose -f deploy/compose.yaml -f deploy/compose.sim.yaml -f deploy/compose.faults.yaml \
    down --remove-orphans >/dev/null 2>&1 || true
  $CF down --remove-orphans >/dev/null 2>&1 || true
  docker rm -f inferobs-gpu-burner >/dev/null 2>&1 || true
}

up() {  # $1 = compose command
  INFEROBS_ACCESS_LOG_GAPS=1 $1 up -d --build >/dev/null 2>&1
  for _ in $(seq 1 180); do
    curl -fsS localhost:28001/health >/dev/null 2>&1 && curl -fsS localhost:28000/health >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "engine did not become healthy"; $1 logs --tail 50 engine; return 1
}

chunk_a() {
  log "chunk a: stack up"
  stack_down
  up "$C"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  dev "python -m inferobs exp trace --model $MODEL --tag $TAG"
  log "capacity probe"
  dev "python scripts/capacity.py --model $MODEL --out results/$TAG/capacity.json"
  RATES=$(dev "python scripts/capacity.py --rates-from results/$TAG/capacity.json")
  log "saturation at $RATES"
  dev "python -m inferobs exp saturation --model $MODEL --rates $RATES --stage-s 75 --tag $TAG"
  RATE=$(dev "python scripts/capacity.py --steady-from results/$TAG/capacity.json")
  log "agreement at $RATE req/s"
  dev "python -m inferobs exp agreement --model $MODEL --rate $RATE --duration 300 --tag $TAG"
  log "overhead"
  dev "python -m inferobs exp overhead --model $MODEL --levels 1,8,32 --duration 60 --repeats 3 --tag $TAG"
  log "chunk a done"
}

chunk_b() {
  log "chunk b: stack up with faults overlay"
  stack_down
  up "$CF"
  RATE=$(dev "python scripts/capacity.py --steady-from results/$TAG/capacity.json")
  log "faults at $RATE req/s"
  dev "python -m inferobs exp faults --engine-kind vllm --model $MODEL --rate $RATE --repeats 2 --tag $TAG"
  log "chunk b done"
}

run_locked() {  # $1 = chunk function
  exec 9>"$LOCK"
  log "waiting for GPU lock $LOCK"
  flock 9
  wait_gpu_free
  rc=0
  "$1" || rc=$?
  stack_down
  flock -u 9
  return $rc
}

case "${1:-all}" in
  a) run_locked chunk_a ;;
  b) run_locked chunk_b ;;
  all) run_locked chunk_a && run_locked chunk_b ;;
  *) echo "usage: $0 [a|b|all]"; exit 2 ;;
esac
