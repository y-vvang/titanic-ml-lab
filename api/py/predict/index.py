"""模型预测：POST /api/py/predict — config = {passenger, modelInfo, cleanConfig, hyperparams}。

预测通过重训实现（与 CLI 行为一致），见 ml_engine.predict。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from endpoint import make_handler  # noqa: E402

handler = make_handler("predict")
