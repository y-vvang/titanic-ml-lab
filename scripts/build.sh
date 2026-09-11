#!/bin/bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

# Vercel 构建镜像的 Python 由 uv 管理（PEP 668 externally-managed），禁止 pip install；
# api/py/** 函数的依赖由 Vercel Python builder 从 requirements.txt 安装，仅在本地装。
if [ -z "${VERCEL:-}" ]; then
  pip install -r requirements.txt 2>&1 | tail -5
fi

echo "Building the Next.js project..."
pnpm next build

echo "Build completed successfully!"
