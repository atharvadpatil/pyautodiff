# PyAutoDiff

A lightweight automatic differentiation library built from scratch using pure Python and NumPy.

## Features

- **Reverse-mode autodiff**: Efficient gradient computation via backpropagation
- **Clean API**: PyTorch-inspired tensor operations with operator overloading
- **Computation graph**: Automatic graph construction and topological sorting
- **Rich operations**: Math, linear algebra, and reduction operations

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from pyautodiff import Tensor, grad

# Create tensors
x = Tensor([1.0, 2.0, 3.0])
y = Tensor([2.0, -1.0, 0.5])

# Build computation graph with operators
z = (x * y).sum()

# Compute gradients
z.backward()
print("dz/dx:", x.grad)
print("dz/dy:", y.grad)
```

### Using the `grad` function

```python
from pyautodiff import Tensor, grad
from pyautodiff.ops import matmul, solve

def func(x, y, A):
    return (solve(A, x) * matmul(A, y)).sum()

# Get gradient function
grad_func = grad(func)

# Compute gradients for all inputs
dx, dy, dA = grad_func(x, y, A)
```

## Operations

| Category | Operations |
|----------|------------|
| **Math** | `add`, `sub`, `mul`, `neg`, `exp`, `log` |
| **Linear Algebra** | `matmul`, `solve`, `logdet` |
| **Reductions** | `sum`, `inner`, `logsumexp` |

## Examples

See the `examples/` directory for more usage examples:
- `basic_autodiff.py` - Simple gradient computations

## License
MIT License
