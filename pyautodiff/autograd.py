"""
Automatic differentiation via reverse-mode backpropagation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Iterable, Callable, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pyautodiff.tensor import Tensor

ArrayLike = Union[np.ndarray, list, float, int]


def get_topological_order(output: "Tensor") -> List["Tensor"]:
    """
    Get tensors in topological order (dependencies before dependents).
    
    Args:
        output: The output tensor to trace back from.
        
    Returns:
        List of tensors in topological order.
    """
    visited = set()
    topo_order = []
    
    def dfs(node: "Tensor") -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.parents:
            dfs(parent)
        topo_order.append(node)
    
    dfs(output)
    return topo_order


def backward(
    output: "Tensor",
    grad_output: Optional[ArrayLike] = None
) -> None:
    """
    Compute gradients via reverse-mode automatic differentiation.
    
    This populates the .grad attribute of all tensors in the computation graph.
    
    Args:
        output: The tensor to differentiate.
        grad_output: Gradient of the loss w.r.t. output.
                     Defaults to 1.0 for scalar outputs.
    """
    # Initialize output gradient
    if grad_output is None:
        if output.value.size != 1:
            raise ValueError(
                "grad_output must be specified for non-scalar outputs"
            )
        grad_output = np.ones_like(output.value, dtype=np.float64)
    else:
        grad_output = np.asarray(grad_output, dtype=np.float64)
    
    # Get computation graph in topological order
    topo_order = get_topological_order(output)
    
    # Gradient accumulator: id(tensor) -> gradient
    grad_map: Dict[int, np.ndarray] = {id(output): grad_output}
    
    # Backward pass (reverse topological order)
    for node in reversed(topo_order):
        g_out = grad_map.get(id(node))
        if g_out is None:
            continue
        
        # Store gradient on tensor
        node._grad = g_out
        
        # Skip leaf nodes (no parents to propagate to)
        if node.op is None:
            continue
        
        # Compute gradients for parent tensors
        parent_values = [p.value for p in node.parents]
        local_grads = node.op.backward(g_out, *parent_values)
        
        # Accumulate gradients
        for parent, local_grad in zip(node.parents, local_grads):
            pid = id(parent)
            if pid in grad_map:
                grad_map[pid] = grad_map[pid] + local_grad
            else:
                grad_map[pid] = local_grad


def compute_gradients(
    output: "Tensor",
    wrt: Iterable["Tensor"],
    grad_output: Optional[ArrayLike] = None
) -> List[np.ndarray]:
    """
    Compute gradients w.r.t. specific tensors without storing on .grad.
    
    Args:
        output: The output tensor.
        wrt: Tensors to compute gradients for.
        grad_output: Gradient of loss w.r.t. output.
        
    Returns:
        List of gradients corresponding to wrt tensors.
    """
    # Initialize output gradient
    if grad_output is None:
        grad_output = np.ones_like(output.value, dtype=np.float64)
    else:
        grad_output = np.asarray(grad_output, dtype=np.float64)
    
    topo_order = get_topological_order(output)
    grad_map: Dict[int, np.ndarray] = {id(output): grad_output}
    
    for node in reversed(topo_order):
        g_out = grad_map.get(id(node))
        if g_out is None or node.op is None:
            continue
        
        parent_values = [p.value for p in node.parents]
        local_grads = node.op.backward(g_out, *parent_values)
        
        for parent, local_grad in zip(node.parents, local_grads):
            pid = id(parent)
            grad_map[pid] = grad_map.get(pid, 0) + local_grad
    
    wrt_list = list(wrt)
    return [grad_map.get(id(v), np.zeros_like(v.value)) for v in wrt_list]


def grad(func: Callable[..., "Tensor"]) -> Callable[..., List[np.ndarray]]:
    """
    Create a function that computes gradients of func w.r.t. its inputs.
    
    This is a higher-order function that transforms a function into
    a gradient function.
    
    Args:
        func: A function that takes Tensors and returns a scalar Tensor.
        
    Returns:
        A function that returns gradients w.r.t. all inputs.
        
    Example:
        >>> def f(x, y):
        ...     return (x * y).sum()
        >>> grad_f = grad(f)
        >>> dx, dy = grad_f(Tensor([1, 2]), Tensor([3, 4]))
    """
    from pyautodiff.tensor import Tensor
    
    def grad_fn(*args) -> List[np.ndarray]:
        # Convert inputs to Tensors if needed
        input_tensors = [
            arg if isinstance(arg, Tensor) else Tensor(arg)
            for arg in args
        ]
        
        # Forward pass
        output = func(*input_tensors)
        
        # Backward pass
        return compute_gradients(output, input_tensors)
    
    return grad_fn
