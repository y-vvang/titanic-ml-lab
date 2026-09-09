#!/usr/bin/env python3
"""
Titanic ML Engine — CLI 包装（本地开发用）

核心逻辑在 api/py/shared/ml_engine.py（与 Vercel Python Functions 共享）。
用法: python3 scripts/ml_engine.py <command> <config_json>
命令: info | explore | clean | preview | train | predict
train 输出多行 JSON 事件流（供 SSE 转发），其余命令输出单行 JSON。
"""

import sys
import os
import json

# 核心引擎位于 api/py/shared/（serverless 函数也从此处导入）
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "api", "py", "shared"
))
import ml_engine  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(ml_engine.json_safe_dumps({"error": "Usage: ml_engine.py <command> <config_json>"}))
        sys.exit(1)

    command = sys.argv[1]
    try:
        config = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(ml_engine.json_safe_dumps({"error": f"Invalid config JSON: {e}"}))
        sys.exit(1)

    if command == "train":
        # 逐行输出事件流（与旧版行为一致）
        def emit(event):
            print(ml_engine.json_safe_dumps(event), flush=True)
        ml_engine.run_train(config, emit)
    else:
        payload = ml_engine.run_command(command, config)
        print(ml_engine.json_safe_dumps(payload, ensure_ascii=False, default=str))
        if "error" in payload and str(payload.get("error", "")).startswith("Unknown command"):
            sys.exit(1)


if __name__ == "__main__":
    main()
