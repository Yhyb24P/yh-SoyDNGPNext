"""Training protocols and task types.

Protocols are physically separated: each Protocol member has its own split
function in splits.py. There is no shared "train_fraction" knob that
silently switches protocol behavior.
"""
from enum import Enum


class Protocol(Enum):
    PACKAGE_LEGACY = "package_legacy"
    PAPER_REPRO = "paper_repro"


class Task(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
