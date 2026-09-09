"""数据清洗：POST /api/py/clean — config = cleanConfig"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from endpoint import make_handler  # noqa: E402

handler = make_handler("clean")
