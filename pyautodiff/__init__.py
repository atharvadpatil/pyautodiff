"""
PyAutoDiff - A lightweight automatic differentiation library.
"""

from pyautodiff.tensor import Tensor
from pyautodiff.autograd import backward, grad

__version__ = "0.1.0"
__all__ = ["Tensor", "backward", "grad"]
