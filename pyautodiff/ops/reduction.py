"""
Reduction operations (sum, inner product, logsumexp).
"""

import numpy as np
from typing import Tuple
from pyautodiff.ops.base import Op


summation = Op(
    "sum",
    1,
    lambda x: np.array(np.sum(x)),
    lambda g, x: (np.ones_like(x) * g,)
)
"""Sum all elements: z = sum(x)"""


inner = Op(
    "inner",
    2,
    lambda x, y: np.array(np.sum(x * y)),
    lambda g, x, y: (g * y, g * x)
)
"""Inner product: z = <x, y> = sum(x * y)"""


def _logsumexp_forward(x: np.ndarray) -> np.ndarray:
    """Numerically stable logsumexp."""
    xf = x.ravel()
    m = np.max(xf)
    return np.array(m + np.log(np.sum(np.exp(xf - m))))


def _logsumexp_backward(g: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray]:
    """Backward pass for logsumexp (softmax gradient)."""
    xf = x.ravel()
    m = np.max(xf)
    exps = np.exp(xf - m)
    softmax = exps / np.sum(exps)
    return (g * softmax.reshape(x.shape),)


logsumexp = Op(
    "logsumexp",
    1,
    _logsumexp_forward,
    _logsumexp_backward
)
"""Log-sum-exp: z = log(sum(exp(x)))"""
