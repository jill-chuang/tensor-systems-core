import pytest
import numpy as np
from src.tensor_systems.linear_algebra import (
    dot_product_native,
    dot_product_vectorized,
    cosine_similarity
)

@pytest.fixture
def sample_vectors():
    return {
        "v1": np.array([1, 2, 3], dtype=float),
        "v2": np.array([4, 5, 6], dtype=float),
        "zero": np.array([0, 0, 0], dtype=float),
        "opposite": np.array([-1, -2, -3], dtype=float)
    }

def test_dot_product_consistency(sample_vectors):
    v1, v2 = sample_vectors["v1"], sample_vectors["v2"]
    res_native = dot_product_native(v1, v2)
    res_vec = dot_product_vectorized(v1, v2)

    assert res_native == pytest.approx(res_vec)

def test_cosine_similarity_edge_cases(sample_vectors):
    v1 = sample_vectors["v1"]
    opposite = sample_vectors["opposite"]
    zero = sample_vectors["zero"]

    assert cosine_similarity(v1, v1) == pytest.approx(1.0)

    assert cosine_similarity(v1, opposite) == pytest.approx(-1.0)

    assert cosine_similarity(v1, zero) == pytest.approx(0.0)

def test_dimension_mismatch():
    a = np.array([1, 2])
    b = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Dimensions mismatch"):
        dot_product_native(a, b)