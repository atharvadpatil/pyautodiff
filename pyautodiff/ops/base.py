"""
Base Op class for defining differentiable operations.
"""

from dataclasses import dataclass
import numpy as np
from typing import Callable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyautodiff.tensor import Tensor


@dataclass(frozen=True)
class Op:
    """
    A differentiable operation with forward and backward functions.
    
    An Op defines:
    - name: Human-readable name for debugging
    - num_inputs: Number of input tensors
    - forward_fn: Computes the output from input arrays
    - backward_fn: Computes gradients w.r.t. inputs given output gradient
    
    Example:
        >>> add_op = Op("add", 2, lambda x, y: x + y, lambda g, x, y: (g, g))
    """
    name: str
    num_inputs: int
    forward_fn: Callable[..., np.ndarray]
    backward_fn: Callable[..., Tuple[np.ndarray, ...]]
    
    def __call__(self, *inputs: "Tensor") -> "Tensor":
        """
        Apply the operation to create a new tensor in the computation graph.
        
        Args:
            *inputs: Input tensors.
            
        Returns:
            A new Tensor with this operation as its creator.
        """
        from pyautodiff.tensor import Tensor
        
        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"{self.name}: expected {self.num_inputs} inputs, got {len(inputs)}"
            )
        
        # Validate all inputs are Tensors
        for i, v in enumerate(inputs):
            if not isinstance(v, Tensor):
                raise TypeError(
                    f"{self.name}: input {i} must be Tensor, got {type(v).__name__}"
                )
        
        # Compute forward pass
        input_values = [t.value for t in inputs]
        output_value = self.forward(*input_values)
        
        # Create output tensor with graph connection
        return Tensor(output_value, _op=self, _parents=tuple(inputs))
    
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass.
        
        Args:
            *xs: Input numpy arrays.
            
        Returns:
            Output numpy array.
        """
        xs_ = [np.asarray(x, dtype=np.float64) for x in xs]
        if len(xs_) != self.num_inputs:
            raise ValueError(
                f"{self.name}: expected {self.num_inputs} inputs, got {len(xs_)}"
            )
        return self.forward_fn(*xs_)
    
    def backward(
        self, 
        grad_output: np.ndarray, 
        *xs: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Compute gradients w.r.t. inputs.
        
        Args:
            grad_output: Gradient of the loss w.r.t. this op's output.
            *xs: Input numpy arrays (from forward pass).
            
        Returns:
            Tuple of gradients, one per input.
        """
        xs_ = [np.asarray(x, dtype=np.float64) for x in xs]
        grad_output_ = np.asarray(grad_output, dtype=np.float64)
        
        if len(xs_) != self.num_inputs:
            raise ValueError(
                f"{self.name} backward: expected {self.num_inputs} inputs, got {len(xs_)}"
            )
        
        grads = self.backward_fn(grad_output_, *xs_)
        
        if not isinstance(grads, tuple) or len(grads) != self.num_inputs:
            raise ValueError(
                f"{self.name} backward must return tuple of length {self.num_inputs}"
            )
        
        return grads


def _reduce_like(grad: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Reduce gradient to match the shape of x (handles broadcasting).
    """
    g = np.asarray(grad, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    
    # Handle dimension mismatch
    while x_arr.ndim < g.ndim:
        x_arr = np.expand_dims(x_arr, axis=0)
    
    # Sum over broadcast dimensions
    axes = tuple(
        i for i, (sx, sg) in enumerate(zip(x_arr.shape, g.shape))
        if sx == 1 and sg > 1
    )
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    
    return g.reshape(np.asarray(x).shape)
