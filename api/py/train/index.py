"""模型训练：POST /api/py/train — config = 完整训练配置。

一次性返回 {"events": [...]}（log/lc_progress/learning_curve/model_structure/result），
由 Next.js train 路由转成 SSE 流，前端契约不变。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from endpoint import make_handler  # noqa: E402

_base = make_handler("train")


# 入口必须是顶层 `class handler` 类定义（Vercel 静态入口检测不认赋值形式），
# 原因详见 endpoint.py 模块注释。
class handler(_base):
    pass
