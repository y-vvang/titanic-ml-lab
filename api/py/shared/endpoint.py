"""Vercel Python Functions 通用 handler 工厂

api/py/<command>/index.py 通过本工厂生成处理对应 ML 引擎命令的 HTTP 入口：
    from endpoint import make_handler
    handler = make_handler("train")

请求/响应契约：
- 入参：POST JSON body = 引擎命令的 config（GET 视为空 config）
- 出参：info/explore/clean/preview/predict 返回引擎结果 dict；
        train 返回 {"events": [...]}（全部事件一次性返回，
        由 Next.js train 路由转成 SSE 流，前端契约不变）
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

# 本文件与 ml_engine.py 同在 api/py/shared/，将其加入 sys.path
_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)

import ml_engine  # noqa: E402


def make_handler(command: str):
    """生成处理指定 ML 引擎命令的 BaseHTTPRequestHandler 子类"""

    class Handler(BaseHTTPRequestHandler):
        def _execute(self, config):
            if command == "train":
                # Python 函数不支持逐事件流式输出，一次性返回事件数组
                return {"events": ml_engine.run_train_collect(config)}
            return ml_engine.run_command(command, config)

        def _respond(self, payload, status=200):
            body = ml_engine.json_safe_dumps(
                payload, ensure_ascii=False, default=str
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
                config = json.loads(raw) if raw.strip() else {}
                self._respond(self._execute(config))
            except Exception as e:  # noqa: BLE001 — 引擎内部错误统一转 500
                self._respond({"error": str(e)}, status=500)

        def do_GET(self):
            self._handle()

        def do_POST(self):
            self._handle()

        def log_message(self, format, *args):  # noqa: A002
            # 静默默认访问日志，避免污染函数日志
            pass

    return Handler


# ── 兜底入口 ──
# Vercel 可能把 api/ 下任意 .py 注册为函数入口；本文件是共享工具模块而非 API，
# 定义 404 handler 仅用于避免 "no handler" 构建错误，正常不会被请求到。
class handler(BaseHTTPRequestHandler):
    def _not_found(self):
        body = b'{"error": "not found"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._not_found()

    def do_POST(self):
        self._not_found()

    def log_message(self, format, *args):  # noqa: A002
        pass
