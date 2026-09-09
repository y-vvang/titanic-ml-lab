"""数据探索：GET/POST /api/py/explore — config 恒为空"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from endpoint import make_handler  # noqa: E402

handler = make_handler("explore")
