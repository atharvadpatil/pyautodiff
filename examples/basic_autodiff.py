"""
Basic automatic differentiation example.

This demonstrates the core functionality of pyautodiff:
- Creating tensors
- Building computation graphs with operations
- Computing gradients via backpropagation
"""

import numpy as np
from pyautodiff import Tensor, grad
from pyautodiff.ops import matmul, solve, inner


def example_basic_gradients():
    """Basic gradient computation with operator overloading."""
    print("=" * 60)
    print("Example 1: Basic Gradient Computation")
    print("=" * 60)
    
    # Create tensors
    x = Tensor([1.0, 2.0, 3.0])
    y = Tensor([2.0, -1.0, 0.5])
    
    # Build computation: z = sum(x * y)
    z = (x * y).sum()
    
    print(f"x = {x.value}")
    print(f"y = {y.value}")
    print(f"z = sum(x * y) = {z.value}")
    
    # Backpropagate
    z.backward()
    
    print(f"\ndz/dx = {x.grad}")
    print(f"dz/dy = {y.grad}")
    
    # Verify: d/dx[sum(x*y)] = y, d/dy[sum(x*y)] = x
    print(f"\nVerification: dz/dx should equal y: {np.allclose(x.grad, y.value)}")
    print(f"Verification: dz/dy should equal x: {np.allclose(y.grad, x.value)}")


def example_chain_rule():
    """Demonstrate automatic chain rule application."""
    print("\n" + "=" * 60)
    print("Example 2: Chain Rule with exp and log")
    print("=" * 60)
    
    x = Tensor([1.0, 2.0, 3.0])
    
    # z = sum(exp(log(x))) = sum(x)
    z = x.log().exp().sum()
    
    print(f"x = {x.value}")
    print(f"z = sum(exp(log(x))) = {z.value}")
    
    z.backward()
    
    print(f"\ndz/dx = {x.grad}")
    print("(Should be all ones since exp(log(x)) = x)")


def example_matrix_operations():
    """Linear algebra operations."""
    print("\n" + "=" * 60)
    print("Example 3: Matrix Operations")
    print("=" * 60)
    
    A = Tensor([[2.0, 1.0],
                [1.0, 3.0]])
    x = Tensor([1.0, 2.0])
    y = Tensor([1.0, 1.0])
    
    # z = <Ax, y> = y^T A x
    Ax = matmul(A, x)
    z = inner(Ax, y)
    
    print(f"A =\n{A.value}")
    print(f"x = {x.value}")
    print(f"y = {y.value}")
    print(f"z = <Ax, y> = {z.value}")
    
    z.backward()
    
    print(f"\ndz/dA =\n{A.grad}")
    print(f"dz/dx = {x.grad}")
    print(f"dz/dy = {y.grad}")


def example_grad_function():
    """Using the grad() higher-order function."""
    print("\n" + "=" * 60)
    print("Example 4: Using grad() Function")
    print("=" * 60)
    
    # Define a function
    def f(x, y):
        return (x * x + y * y).sum()  # sum(x^2 + y^2)
    
    # Get gradient function
    grad_f = grad(f)
    
    # Compute gradients
    x_val = np.array([1.0, 2.0, 3.0])
    y_val = np.array([4.0, 5.0, 6.0])
    
    dx, dy = grad_f(x_val, y_val)
    
    print(f"f(x, y) = sum(x^2 + y^2)")
    print(f"x = {x_val}")
    print(f"y = {y_val}")
    print(f"\ndf/dx = {dx}  (should be 2*x)")
    print(f"df/dy = {dy}  (should be 2*y)")


def example_linear_solve():
    """Solve linear system with gradients."""
    print("\n" + "=" * 60)
    print("Example 5: Linear System Solve")
    print("=" * 60)
    
    A = Tensor([[4.0, 1.0],
                [1.0, 3.0]])
    b = Tensor([1.0, 2.0])
    
    # x = A^{-1} b
    x = solve(A, b)
    loss = x.sum()
    
    print(f"A =\n{A.value}")
    print(f"b = {b.value}")
    print(f"x = solve(A, b) = {x.value}")
    
    loss.backward()
    
    print(f"\nd(sum(x))/dA =\n{A.grad}")
    print(f"d(sum(x))/db = {b.grad}")


if __name__ == "__main__":
    example_basic_gradients()
    example_chain_rule()
    example_matrix_operations()
    example_grad_function()
    example_linear_solve()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
