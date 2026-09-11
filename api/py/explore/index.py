"""数据探索：GET/POST /api/py/explore — config 恒为空"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from endpoint import make_handler  # noqa: E402

_base = make_handler("explore")


# 入口必须是顶层 `class handler` 类定义（Vercel 静态入口检测不认赋值形式），
# 原因详见 endpoint.py 模块注释。
class handler(_base):
    pass
