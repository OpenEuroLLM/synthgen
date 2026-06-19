"""Shared logging setup."""
from __future__ import annotations

import logging
import os
import sys


_DEFAULT_FMT = "%(asctime)s %(levelname)s %(name)s | %(message)s"


def get_logger(name: str = "synthgen") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = os.environ.get("SYNTHGEN_LOG", "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FMT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
