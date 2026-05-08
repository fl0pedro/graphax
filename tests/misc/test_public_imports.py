def test_sparse_tensor_importable():
    from graphax.sparse import SparseTensor
    assert SparseTensor is not None


def test_sparse_tensor_zeros_like_importable():
    from graphax.sparse import sparse_tensor_zeros_like
    assert callable(sparse_tensor_zeros_like)


def test_sparse_tensor_zeros_like_top_level():
    from graphax import sparse_tensor_zeros_like
    assert callable(sparse_tensor_zeros_like)
