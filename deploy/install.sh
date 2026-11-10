#!/usr/bin/env bash
# 一键安装：创建虚拟环境、安装依赖、安装与显卡匹配的 PyTorch。
# 用法： bash deploy/install.sh            # 自动检测 GPU
#        TORCH_INDEX=cpu bash deploy/install.sh   # 强制 CPU 版
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
$PYTHON -c 'import sys; assert sys.version_info >= (3, 10), "需要 Python 3.10 及以上"'
[ -d .venv ] || $PYTHON -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ -z "${TORCH_INDEX:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    # RTX PRO 6000 Blackwell（以及 RTX 50 系）需要 CUDA 12.8 构建的 PyTorch（>= 2.7）
    TORCH_INDEX=cu128
  else
    TORCH_INDEX=cpu
  fi
fi
pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

python - <<'PY'
import torch, xgboost, lightgbm, optuna
print("torch", torch.__version__, "| CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
    x = torch.randn(1024, 1024, device="cuda"); (x @ x).sum().item()   # 实际跑一次，确认驱动与构建匹配
    print("GPU 计算自检通过")
print("xgboost", xgboost.__version__, "| lightgbm", lightgbm.__version__, "| optuna", optuna.__version__)
PY

[ -f config/local.yaml ] || cp config/gpu_server.example.yaml config/local.yaml
echo
echo "安装完成。下一步："
echo "  1) 按需编辑 config/local.yaml"
echo "  2) make test        # 自检"
echo "  3) make data        # 下载历史数据"
echo "  4) make train       # 全自动训练"
echo "  5) make web / make live   # 控制台 与 实时预测"
