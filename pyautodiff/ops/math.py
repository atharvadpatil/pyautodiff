"""
Element-wise math operations.
"""

import numpy as np
from pyautodiff.ops.base import Op, _reduce_like


add = Op(
    "add",
    2,
    lambda x, y: x + y,
    lambda g, x, y: (_reduce_like(g, x), _reduce_like(g, y))
)
"""Element-wise addition: z = x + y"""


sub = Op(
    "sub",
    2,
    lambda x, y: x - y,
    lambda g, x, y: (_reduce_like(g, x), _reduce_like(-g, y))
)
"""Element-wise subtraction: z = x - y"""


mul = Op(
    "mul",
    2,
    lambda x, y: x * y,
    lambda g, x, y: (_reduce_like(g * y, x), _reduce_like(g * x, y))
)
"""Element-wise multiplication: z = x * y"""


neg = Op(
    "neg",
    1,
    lambda x: -x,
    lambda g, x: (-g,)
)
"""Negation: z = -x"""


exp = Op(
    "exp",
    1,
    lambda x: np.exp(x),
    lambda g, x: (g * np.exp(x),)
)
"""Element-wise exponential: z = exp(x)"""


log = Op(
    "log",
    1,
    lambda x: np.log(x),
    lambda g, x: (g / x,)
)
"""Element-wise natural logarithm: z = log(x)"""
