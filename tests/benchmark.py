import numpy as np
import time
from src.tensor_systems.linear_algebra import(
    dot_product_native,
    dot_product_vectorized
)

def benchmark_dot_product():
    size = 1_000_000
    a = np.random.rand(size)
    b = np.random.rand(size)

    start = time.time()
    dot_product_native(a, b)
    native_duration = time.time() - start

    start = time.time()
    dot_product_vectorized(a, b)
    vec_duration = time.time() - start

    print(f"\n[Performance Report] Size: {size}")
    print(f"Native Python loop: {native_duration:.4f}s")
    print(f"Numpy Vectorized: {vec_duration:.4f}s")
    print(f"Speedup: {native_duration / vec_duration:.1f}x")

if __name__ == "__main__":
    benchmark_dot_product()