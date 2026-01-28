import numpy as np

def dot_product_native(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError("Dimensions mismatch")

    total = 0.0
    for i in range(len(a)):
        total += float(a[i] * b[i])
    return total

def dot_product_vectorized(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_prod = dot_product_vectorized(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    eps = 1e-9
    if norm_a < eps or norm_b < eps:
        return 0.0

    return float(dot_prod / (norm_a * norm_b))
