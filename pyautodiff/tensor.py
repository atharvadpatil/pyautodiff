"""
Tensor class - the core computation graph node.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional, Union

if TYPE_CHECKING:
    from pyautodiff.ops.base import Op

ArrayLike = Union[np.ndarray, list, float, int]


class Tensor:
    """
    A tensor that tracks operations for automatic differentiation.
    
    Tensors form nodes in a computation graph. Each tensor stores:
    - Its value (numpy array)
    - The operation that created it
    - Parent tensors (inputs to the operation)
    - Accumulated gradient (after backward pass)
    
    Example:
        >>> x = Tensor([1.0, 2.0, 3.0])
        >>> y = Tensor([2.0, -1.0, 0.5])
        >>> z = x * y
        >>> z.backward()
        >>> print(x.grad)  # dz/dx
    """
    
    def __init__(
        self, 
        data: ArrayLike,
        _op: Optional["Op"] = None,
        _parents: Tuple["Tensor", ...] = ()
    ):
        """
        Create a tensor.
        
        Args:
            data: The tensor data (will be converted to numpy array).
            _op: (Internal) The operation that produced this tensor.
            _parents: (Internal) Parent tensors in the computation graph.
        """
        self._value = np.asarray(data, dtype=np.float64)
        self._op = _op
        self._parents = _parents
        self._grad: Optional[np.ndarray] = None
    
    @property
    def value(self) -> np.ndarray:
        """The tensor's data as a numpy array."""
        return self._value
    
    @property
    def data(self) -> np.ndarray:
        """Alias for value."""
        return self._value
    
    @property
    def grad(self) -> Optional[np.ndarray]:
        """The accumulated gradient, or None if backward() hasn't been called."""
        return self._grad
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor."""
        return self._value.shape
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._value.ndim
    
    @property
    def dtype(self):
        """Data type of the tensor."""
        return self._value.dtype
    
    @property
    def op(self) -> Optional["Op"]:
        """The operation that created this tensor, or None for leaf tensors."""
        return self._op
    
    @property
    def parents(self) -> Tuple["Tensor", ...]:
        """Parent tensors in the computation graph."""
        return self._parents
    
    @property
    def is_leaf(self) -> bool:
        """True if this is a leaf tensor (not created by an operation)."""
        return self._op is None
    
    def backward(self, grad_output: Optional[ArrayLike] = None) -> None:
        """
        Compute gradients via backpropagation.
        
        Args:
            grad_output: Gradient of the loss w.r.t. this tensor.
                         Defaults to ones for scalar outputs.
        """
        from pyautodiff.autograd import backward
        backward(self, grad_output)
    
    def zero_grad(self) -> None:
        """Reset the gradient to None."""
        self._grad = None
    
    # -------------------- Operator Overloading --------------------
    
    def __add__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        from pyautodiff.ops import add
        other = other if isinstance(other, Tensor) else Tensor(other)
        return add(self, other)
    
    def __radd__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) + self
    
    def __sub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        from pyautodiff.ops import sub
        other = other if isinstance(other, Tensor) else Tensor(other)
        return sub(self, other)
    
    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) - self
    
    def __mul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        from pyautodiff.ops import mul
        other = other if isinstance(other, Tensor) else Tensor(other)
        return mul(self, other)
    
    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) * self
    
    def __neg__(self) -> "Tensor":
        from pyautodiff.ops import neg
        return neg(self)
    
    def __matmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        from pyautodiff.ops import matmul
        other = other if isinstance(other, Tensor) else Tensor(other)
        return matmul(self, other)
    
    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return Tensor(other) @ self
    
    # -------------------- Convenience Methods --------------------
    
    def sum(self) -> "Tensor":
        """Sum all elements."""
        from pyautodiff.ops import summation
        return summation(self)
    
    def exp(self) -> "Tensor":
        """Element-wise exponential."""
        from pyautodiff.ops import exp
        return exp(self)
    
    def log(self) -> "Tensor":
        """Element-wise natural logarithm."""
        from pyautodiff.ops import log
        return log(self)
    
    # -------------------- Magic Methods --------------------
    
    def __repr__(self) -> str:
        grad_str = ", grad=..." if self._grad is not None else ""
        return f"Tensor({self._value}{grad_str})"
    
    def __str__(self) -> str:
        return str(self._value)
    
    def __len__(self) -> int:
        return len(self._value)
    
    def __getitem__(self, key):
        return self._value[key]
    
    def numpy(self) -> np.ndarray:
        """Return the tensor data as a numpy array."""
        return self._value.copy()
    
    def item(self) -> float:
        """Return a scalar tensor's value as a Python float."""
        return float(self._value)
