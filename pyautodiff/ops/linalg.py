"""
Linear algebra operations.
"""

import numpy as np
from typing import Tuple
from pyautodiff.ops.base import Op


def _matmul_backward(g: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backward pass for matrix multiplication."""
    # 2D @ 2D
    if x.ndim == 2 and y.ndim == 2:
        dX = g @ y.T
        dY = x.T @ g
        return (dX, dY)
    
    # 2D @ 1D (matrix-vector)
    if x.ndim == 2 and y.ndim == 1:
        dX = np.outer(g, y)
        dY = x.T @ g
        return (dX, dY)
    
    # 1D @ 2D (vector-matrix)
    if x.ndim == 1 and y.ndim == 2:
        dX = y @ g
        dY = np.outer(x, g)
        return (dX, dY)
    
    # 1D @ 1D (dot product)
    if x.ndim == 1 and y.ndim == 1:
        return (g * y, g * x)
    
    raise ValueError(f"Unhandled matmul shapes: {x.shape} @ {y.shape}")


matmul = Op(
    "matmul",
    2,
    lambda x, y: x @ y,
    _matmul_backward
)
"""Matrix multiplication: z = x @ y"""


def _solve_forward(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve linear system Ax = b for x."""
    return np.linalg.solve(A, b)


def _solve_backward(g: np.ndarray, A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backward pass for linear solve."""
    x = np.linalg.solve(A, b)
    gb = np.linalg.solve(A.T, g)
    gA = -np.outer(gb, x) if x.ndim == 1 else -gb @ x.T
    return (gA, gb)


solve = Op(
    "solve",
    2,
    _solve_forward,
    _solve_backward
)
"""Solve linear system: x = A^{-1} b"""


def _logdet_forward(A: np.ndarray) -> np.ndarray:
    """Compute log determinant of a positive definite matrix."""
    sign, ld = np.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError("logdet: matrix must have positive determinant")
    return np.array(ld)


def _logdet_backward(g: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray]:
    """Backward pass for log determinant."""
    return (g * np.linalg.inv(A).T,)


logdet = Op(
    "logdet",
    1,
    _logdet_forward,
    _logdet_backward
)
"""Log determinant of a positive definite matrix: z = log|A|"""
