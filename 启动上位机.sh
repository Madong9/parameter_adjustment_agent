#!/usr/bin/env bash
set -euo pipefail

# 使用相对路径转交给 Agent 内的真正启动器，支持项目整体移动。
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT_DIR/rl_agent/启动上位机.sh"
