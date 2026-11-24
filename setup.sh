#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

venv_dir="${BENCH_VENV:-.venv}"
if [[ ! -x "$venv_dir/bin/python" ]]; then
  "${PYTHON_BIN:-python}" -m venv --system-site-packages "$venv_dir"
fi
bench_python="$venv_dir/bin/python"
"$bench_python" -m pip install numpy==1.26.4

if ! "$bench_python" -c 'import torch, torchvision; assert torch.__version__.split("+")[0] == "2.1.2" and torchvision.__version__.split("+")[0] == "0.16.2" and torch.version.cuda == "11.8"'; then
  "$bench_python" -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    'torch==2.1.2+cu118' 'torchvision==0.16.2+cu118'
fi

"$bench_python" -m pip install -r requirements.txt

if [[ "${SKIP_ASSETS:-0}" != 1 ]]; then
  "$bench_python" prepare_assets.py "$@"
fi

"$bench_python" -c 'import torch, torchvision; print("torch", torch.__version__, "torchvision", torchvision.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU not visible")'
