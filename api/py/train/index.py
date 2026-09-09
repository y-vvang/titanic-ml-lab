"""模型训练：POST /api/py/train — config = 完整训练配置。

一次性返回 {"events": [...]}（log/lc_progress/learning_curve/model_structure/result），
由 Next.js train 路由转成 SSE 流，前端契约不变。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from endpoint import make_handler  # noqa: E402

handler = make_handler("train")
