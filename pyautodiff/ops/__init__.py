"""
Operations module - all differentiable operations.
"""

from pyautodiff.ops.base import Op
from pyautodiff.ops.math import add, sub, mul, neg, exp, log
from pyautodiff.ops.linalg import matmul, solve, logdet
from pyautodiff.ops.reduction import summation, inner, logsumexp

__all__ = [
    "Op",
    # Math ops
    "add", "sub", "mul", "neg", "exp", "log",
    # Linear algebra
    "matmul", "solve", "logdet",
    # Reductions
    "summation", "inner", "logsumexp",
]
