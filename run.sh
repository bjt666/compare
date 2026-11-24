#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if (( $# < 2 )); then
  echo 'Usage: bash run.sh {unet|deeplabv3plus|unetformer|pugnet|mseonet|all} /absolute/path/to/dataset [seed] [train flags...]' >&2
  exit 2
fi

model="$1"
dataset="$2"
shift 2
seed=42
if (( $# )) && [[ "$1" =~ ^[0-9]+$ ]]; then
  seed="$1"
  shift
fi

bench_python="${BENCH_VENV:-.venv}/bin/python"
if [[ ! -x "$bench_python" ]]; then
  echo '请先执行 bash setup.sh。' >&2
  exit 2
fi

models=("$model")
if [[ "$model" == all ]]; then
  models=(unet deeplabv3plus unetformer pugnet mseonet)
fi
for name in "${models[@]}"; do
  "$bench_python" train.py --model "$name" --data-root "$dataset" --seed "$seed" "$@"
done
