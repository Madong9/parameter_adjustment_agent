#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 自动加载本机私密配置。.env 已被 Git 忽略，不会随代码上传。
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

export PYTHONUTF8=1
exec conda run --no-capture-output -n rl_agent python -m rl_training_agent desktop
