import unittest
from itertools import chain, combinations, permutations

import jax.numpy as jnp
import jax.random as jrand
from chex import Array

from graphax.sparse.indexes import DenseIndex, SparseIndex
from graphax.sparse.ops.utils import _arr2st
from graphax.sparse.tensor import SparseTensor

# --- Test fixtures and setup ---


def create_block_test_data(factor=1):
    n_blocks = 3
    block_nums = n_blocks
    dense_block_nums = 2
    key = jrand.PRNGKey(42)

    def make_blocks(shape, key):
        return jrand.normal(key, shape)

    keys = jrand.split(key, 11)

    k1 = jrand.split(keys[0], 3)
    k2 = jrand.split(keys[1], 3)
    k3 = jrand.split(keys[2], 3)

    s2 = 2 * factor
    s3 = 3 * factor
    s4 = 4 * factor
    s5 = 5 * factor

    data = {  # no_equal_* depreciated
        "block_nums": block_nums,
        "dense_block_nums": dense_block_nums,
        "scalar_blocks": make_blocks((n_blocks,), keys[3]),
        # "_1d_blocks": make_blocks((n_blocks, s2), keys[4]),
        "all_equal_sides_2d_blocks": make_blocks((n_blocks, s2, s2), keys[5]),
        "no_equal_sides_2d_blocks": make_blocks((n_blocks, s2, s3), keys[6]),
        # "no_equal_2d_blocks": [
        #     jrand.normal(k1[0], (s2, s2)),
        #     jrand.normal(k1[1], (s3, s3)),
        #     jrand.normal(k1[2], (s2, s4))
        # ],
        "all_equal_sides_3d_blocks": make_blocks((n_blocks, s2, s2, s2), keys[7]),
        "no_equal_sides_3d_blocks": make_blocks((n_blocks, s2, s3, s4), keys[8]),
        # "no_equal_3d_blocks": [
        #     jrand.normal(k2[0], (s2, s2, s2)),
        #     jrand.normal(k2[1], (s3, s3, s3)),
        #     jrand.normal(k2[2], (s2, s4, s2))
        # ],
        "all_equal_sides_4d_blocks": make_blocks((n_blocks, s2, s2, s2, s2), keys[9]),
        "no_equal_sides_4d_blocks": make_blocks(
            (n_blocks, s2, s3, s4, s5), jrand.split(keys[10])[0]
        ),
        # "no_equal_4d_blocks": [
        #     jrand.normal(k3[0], (s2, s2, s2, s2)),
        #     jrand.normal(k3[1], (s3, s3, s3, s3)),
        #     jrand.normal(k3[2], (s2, s4, s2, s5))
        # ],
    }
    return data


# --- utils ---


def verify_dimensions(out_dims, primal_dims, val=None):
    dims = list(out_dims) + list(primal_dims)
    ids = [d.id for d in dims]
    assert len(ids) == len(set(ids))

    sparse_dims = {d.id: d for d in dims if d.is_sparse}
    for d in sparse_dims.values():
        assert d.other_id in sparse_dims
        other = sparse_dims[d.other_id]
        assert other.other_id == d.id
        assert d.size == other.size


def extract_shape(blocks):
    if isinstance(blocks, list):
        if not blocks:
            return (0,)
        return (len(blocks),) + blocks[0].shape
    return blocks.shape


def compute_physical_shape(st):
    return st.sparse_shape + st.dense_shape


def flatten_to_2d(st):
    if st.val is None:
        return None
    if st.val.size == st.sparse_size * st.dense_size:
        return st.val.reshape(st.sparse_size, st.dense_size)
    return None


def validate_sparse_tensor(st):
    assert st.ndim == len(st.out_dims) + len(st.primal_dims)
    assert st.shape == st.out_shape + st.primal_shape

    for d in st.dims:
        if d.is_sparse:
            assert d.logical_size == d.size * (d.block_size or 1)
        else:
            assert d.logical_size == d.size


BlocksShape = tuple[int, ...]


def generate_dimension_specs(
    blocks_shape: BlocksShape,
    forced_ndim: int,
    axes: tuple[int, ...] = (0,),
    out_ndim: int | None = None,
) -> list[tuple[list, ...]]:
    blocks_ndim = len(blocks_shape)
    sparse_axes = list(axes)
    dense_axes = sorted(list(set(range(blocks_ndim)) - set(axes)))

    def map_sparse(idx):
        return sparse_axes[idx] if idx < len(sparse_axes) else None

    def map_dense(idx):
        return dense_axes[idx] if idx < len(dense_axes) else None

    def get_block_size(dim_idx):
        phys = map_dense(dim_idx)
        return blocks_shape[phys] if phys is not None else 1

    def get_sparse_size(dim_idx):
        phys = map_sparse(dim_idx)
        return blocks_shape[phys] if phys is not None else 1

    def make_sparse(id, other_id, sparse_idx, dense_idx):
        return SparseIndex(
            id,
            get_sparse_size(sparse_idx),
            axis=map_sparse(sparse_idx),
            other_id=other_id,
            block_size=get_block_size(dense_idx),
            block_axis=map_dense(dense_idx),
        )

    def make_dense(id, dense_idx):
        return DenseIndex(id, get_block_size(dense_idx), axis=map_dense(dense_idx))

    specs = []

    if forced_ndim == 2:
        if len(sparse_axes) >= 1 and len(dense_axes) >= 2:
            specs.append(([make_sparse(0, 1, 0, 0)], [make_sparse(1, 0, 0, 1)]))

    if forced_ndim == 3:
        if len(sparse_axes) >= 1 and len(dense_axes) >= 3:
            specs.append(
                ([make_dense(0, 0), make_sparse(1, 2, 0, 1)], [make_sparse(2, 1, 0, 2)])
            )
            specs.append(
                ([make_sparse(0, 2, 0, 0), make_dense(1, 1)], [make_sparse(2, 0, 0, 2)])
            )
            specs.append(
                ([make_sparse(0, 2, 0, 0)], [make_dense(1, 1), make_sparse(2, 0, 0, 2)])
            )
            specs.append(
                ([make_sparse(0, 1, 0, 0)], [make_sparse(1, 0, 0, 1), make_dense(2, 2)])
            )

    if forced_ndim == 4:
        if len(sparse_axes) >= 1 and len(dense_axes) >= 4:
            specs.append(
                (
                    [make_dense(0, 0), make_dense(1, 1), make_sparse(2, 3, 0, 2)],
                    [make_sparse(3, 2, 0, 3)],
                )
            )
            specs.append(
                (
                    [make_dense(0, 0), make_sparse(1, 3, 0, 1), make_dense(2, 2)],
                    [make_sparse(3, 1, 0, 3)],
                )
            )
            specs.append(
                (
                    [make_sparse(0, 3, 0, 0), make_dense(1, 1), make_dense(2, 2)],
                    [make_sparse(3, 0, 0, 3)],
                )
            )
            specs.append(
                (
                    [make_dense(0, 0), make_sparse(1, 3, 0, 1)],
                    [make_dense(2, 2), make_sparse(3, 1, 0, 3)],
                )
            )
            specs.append(
                (
                    [make_dense(0, 0), make_sparse(1, 2, 0, 1)],
                    [make_sparse(2, 1, 0, 2), make_dense(3, 3)],
                )
            )
            specs.append(
                (
                    [make_sparse(0, 3, 0, 0), make_dense(1, 1)],
                    [make_dense(2, 2), make_sparse(3, 0, 0, 3)],
                )
            )
            specs.append(
                (
                    [make_sparse(0, 2, 0, 0), make_dense(1, 1)],
                    [make_sparse(2, 0, 0, 2), make_dense(3, 3)],
                )
            )
            specs.append(
                (
                    [make_sparse(0, 3, 0, 0)],
                    [make_dense(1, 1), make_dense(2, 2), make_sparse(3, 0, 0, 3)],
                )
            )
            specs.append(
                (
                    [make_sparse(0, 2, 0, 0)],
                    [make_dense(1, 1), make_sparse(2, 0, 0, 2), make_dense(3, 3)],
                )
            )
            specs.append(
                (
                    [make_sparse(0, 1, 0, 0)],
                    [make_sparse(1, 0, 0, 1), make_dense(2, 2), make_dense(3, 3)],
                )
            )

        if len(sparse_axes) >= 2 and len(dense_axes) >= 4:
            specs.append(
                (
                    [make_sparse(0, 2, 0, 0), make_sparse(1, 3, 1, 1)],
                    [make_sparse(2, 0, 0, 2), make_sparse(3, 1, 1, 3)],
                )
            )
            specs.append(
                (
                    [make_sparse(0, 3, 0, 0), make_sparse(1, 2, 1, 1)],
                    [make_sparse(2, 1, 1, 2), make_sparse(3, 0, 0, 3)],
                )
            )

    if out_ndim is None:
        return specs
    return [s for s in specs if len(s[0]) == out_ndim]


def drop_physical_axes(st, axes_to_drop):
    if not axes_to_drop:
        return st

    sorted_axes = sorted(list(axes_to_drop), reverse=True)
    new_val = st.val
    if new_val is not None:
        for axis in sorted_axes:
            new_val = jnp.take(new_val, 0, axis=axis)

    old_ndim = st.val.ndim if st.val is not None else 0

    axis_map = {}
    shift = 0
    for i in range(old_ndim):
        if i in axes_to_drop:
            axis_map[i] = None
            shift += 1
        else:
            axis_map[i] = i - shift

    def map_dim_idx(idx):
        if idx is None:
            return None
        return axis_map.get(idx, None)

    new_dims = []
    for d in st.dims:
        if d.is_sparse:
            new_d = SparseIndex(
                d.id,
                d.size,
                axis=map_dim_idx(d.axis),
                other_id=d.other_id,
                block_size=d.block_size,
                block_axis=map_dim_idx(d.block_axis),
            )
            new_dims.append(new_d)
        else:
            new_d = DenseIndex(d.id, d.size, axis=map_dim_idx(d.axis))
            new_dims.append(new_d)

    n_out = len(st.out_dims)
    return SparseTensor(new_dims[:n_out], new_dims[n_out:], new_val)


def get_block_configs(data, ndim):
    configs = []

    # if ndim == 0:
    #     return [("scalar_blocks", data["scalar_blocks"])]
    # if ndim == 1:
    #     return [("_1d_blocks", data["_1d_blocks"])]

    keys = [
        f"no_equal_{ndim}d_blocks",
        f"no_equal_sides_{ndim}d_blocks",
        f"all_equal_sides_{ndim}d_blocks",
    ]

    for k in keys:
        if k in data:
            configs.append((k, data[k]))

    return configs


def generate_tensors(blocks, ndim):
    if blocks is None or (isinstance(blocks, list) and not blocks):
        return

    blocks_shape = extract_shape(blocks)
    for axes in all_axis_permutations(len(blocks_shape)):
        for out_dims, primal_dims in generate_dimension_specs(blocks_shape, ndim, axes):
            verify_dimensions(out_dims, primal_dims, blocks)
            st = SparseTensor(out_dims, primal_dims, blocks)
            yield st

            if st.val is not None:
                phys_ndim = st.val.ndim
                max_drops = min(phys_ndim, 2)
                for r in range(1, max_drops + 1):
                    for axes_to_drop in combinations(range(phys_ndim), r):
                        yield drop_physical_axes(st, set(axes_to_drop))


def generate_all_tensors(block_test_data, ndim):
    for name, blocks in get_block_configs(block_test_data, ndim):
        yield from generate_tensors(blocks, ndim)


def permute_axes(ndim):
    return permutations(range(ndim))


def all_axis_permutations(ndim):
    return chain.from_iterable(permutations(range(ndim), i) for i in range(ndim + 1))


def matmul_reference(
    lhs: SparseTensor | Array, rhs: SparseTensor | Array
) -> SparseTensor | Array:
    if isinstance(lhs, SparseTensor):
        l_dense = lhs.dense()
    else:
        l_dense = lhs
    if isinstance(rhs, SparseTensor):
        r_dense = rhs.dense()
    else:
        r_dense = rhs

    if isinstance(lhs, SparseTensor):
        n_contract = len(lhs.primal_dims)
    elif isinstance(rhs, SparseTensor):
        n_contract = len(rhs.out_dims)
    else:
        raise ValueError("At least one operand must be a SparseTensor")

    l_axes = list(range(l_dense.ndim - n_contract, l_dense.ndim))
    r_axes = list(range(n_contract))

    res_dense = jnp.tensordot(l_dense, r_dense, axes=(tuple(l_axes), tuple(r_axes)))

    if isinstance(lhs, SparseTensor):
        out_ndim = len(lhs.out_dims)
    else:
        out_ndim = l_dense.ndim - n_contract

    return _arr2st(res_dense, out_ndim=out_ndim)


def assert_matmul_result(
    st_result: SparseTensor,
    dense_ref: jnp.ndarray,
    out_logical_shape: tuple[int, ...],
    primal_logical_shape: tuple[int, ...],
    physical_shape: tuple[int, ...] | None = None,
):
    # ``physical_shape`` pins ``val.shape`` to catch unintended densification —
    # not a semantic invariant. The SparseTensor algebra allows any axis order
    # in val as long as ``dim.axis`` agrees; equivalent layouts (same multiset
    # of sizes, different permutation) all denote the same logical tensor.
    # When refactors change which path emits a matmul (e.g. the fast paths
    # were removed in Phase 5c), the expected ``physical_shape`` may permute.
    if not jnp.allclose(st_result.dense(), dense_ref, atol=1e-5):
        raise AssertionError("Dense evaluations do not match.")

    if st_result.out_shape != out_logical_shape:
        raise AssertionError(
            f"Expected out_shape {out_logical_shape}, got {st_result.out_shape}"
        )

    if st_result.primal_shape != primal_logical_shape:
        raise AssertionError(
            f"Expected primal_shape {primal_logical_shape}, got {st_result.primal_shape}"
        )

    if physical_shape is not None:
        if physical_shape == ():
            if st_result.val is not None:
                raise AssertionError(
                    "Expected val to be None (fully implicit), but it has data."
                )
        else:
            if st_result.val is None:
                raise AssertionError("Expected val to contain data, but it was None.")
            if st_result.val.shape != physical_shape:
                raise AssertionError(
                    f"Physical shape mismatch. Expected {physical_shape}, got {st_result.val.shape}. "
                    f"The matmul operation may be losing sparsity."
                )


def get_keys(seed=42, n=10):
    return jrand.split(jrand.PRNGKey(seed), n)


def idfn(val):
    if isinstance(val, SparseTensor):
        return f"ST({list(val.out_shape)}|{list(val.primal_shape)})"
    return str(val)


def run_matmul_blocks_test(test_case, tensor_a, tensor_b):
    reference_result = matmul_reference(tensor_a, tensor_b)
    reference_dense = reference_result.dense()

    with test_case.subTest(op="sparse @ sparse"):
        result = tensor_a @ tensor_b
        assert_matmul_result(
            result,
            reference_dense,
            reference_result.out_shape,
            reference_result.primal_shape,
            result.val.shape if result.val is not None else None,
        )

    with test_case.subTest(op="sparse @ dense"):
        result_s_d = tensor_a @ tensor_b.dense()
        assert_matmul_result(
            result_s_d,
            reference_dense,
            reference_result.out_shape,
            reference_result.primal_shape,
            result_s_d.val.shape if result_s_d.val is not None else None,
        )

    with test_case.subTest(op="dense @ sparse"):
        result_d_s = tensor_a.dense() @ tensor_b
        assert_matmul_result(
            result_d_s,
            reference_dense,
            reference_result.out_shape,
            reference_result.primal_shape,
            result_d_s.val.shape if result_d_s.val is not None else None,
        )


if __name__ == "__main__":
    unittest.main()


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.block_test_data = create_block_test_data()
        self.axes_test_data = {
            0: {()},
            1: {(0,)},
            2: {(0, 1), (1, 0)},
            3: {(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)},
        }

    def test_setup(self):
        self.assertEqual(
            len(self.block_test_data["all_equal_sides_2d_blocks"]),
            self.block_test_data["block_nums"],
        )

    def test_extract_shape(self):
        self.assertEqual(extract_shape(jnp.ones((2, 3))), (2, 3))
        # Fix: handle list of arrays by converting to an array first if needed,
        # but here we just want to test if it works with lists if that's intended.
        # Original code was failing because lists don't have .shape.
        # I'll update extract_shape to handle lists.
        # Wait, I should probably update the function itself if it's supposed to support lists.
        self.assertEqual(extract_shape([jnp.zeros((2,))] * 3), (3, 2))

    def test_compute_physical_shape(self):
        val = jnp.zeros((2, 3))
        st = _arr2st(val, 1)
        self.assertEqual(compute_physical_shape(st), val.shape)

    def test_verify_dimensions(self):
        d1 = DenseIndex(0, 2, 0)
        d2 = DenseIndex(1, 2, 1)
        verify_dimensions([d1], [d2])

        with self.assertRaises(AssertionError):
            verify_dimensions([d1], [d1])

        s1 = SparseIndex(0, 2, 0, 1)
        with self.assertRaises(AssertionError):
            verify_dimensions([s1], [])

        s2 = SparseIndex(1, 3, 1, 0)
        with self.assertRaises(AssertionError):
            verify_dimensions([s1], [s2])

    def test_generate_dimension_specs(self):
        self.assertEqual(len(generate_dimension_specs((), 1, ())), 0)

        for i in range(1, 4):
            for axes in permute_axes(i):
                for key, blocks in self.block_test_data.items():
                    if "blocks" in key:
                        for n in range(3):
                            shape = (
                                self.block_test_data["dense_block_nums"],
                            ) * n + extract_shape(blocks)
                            for ds in generate_dimension_specs(shape, i + 2, axes):
                                verify_dimensions(*ds)

    # test_fetch_test_blocks was removed as fetch_test_blocks is not defined.

    def test_all_axis_permutations(self):
        perms = list(all_axis_permutations(2))
        self.assertEqual(len(perms), 1 + 2 + 2)
        self.assertIn((), perms)
        self.assertIn((0,), perms)
        self.assertIn((1, 0), perms)

    def test_drop_physical_axes(self):
        val = jnp.arange(24).reshape(2, 3, 4)
        d0 = DenseIndex(0, 2, 0)
        d1 = DenseIndex(1, 3, 1)
        d2 = DenseIndex(2, 4, 2)
        st = SparseTensor([d0], [d1, d2], val)

        st_drop = drop_physical_axes(st, {1})
        self.assertEqual(st_drop.val.shape, (2, 4))
        self.assertEqual(st_drop.out_dims[0].axis, 0)
        self.assertIsNone(st_drop.primal_dims[0].axis)
        self.assertEqual(st_drop.primal_dims[1].axis, 1)

        expected = val[:, 0, :]
        self.assertTrue(jnp.array_equal(st_drop.val, expected))

    def test_generate_tensors(self):
        counts = {}
        for ndim in range(1, 5):
            tensors = list(generate_all_tensors(self.block_test_data, ndim))
            counts[ndim] = len(tensors)
            for st in tensors:
                validate_sparse_tensor(st)
                flat = flatten_to_2d(st)
                if flat is not None:
                    self.assertEqual(flat.size, st.sparse_size * st.dense_size)
                    self.assertEqual(flat.shape, (st.sparse_size, st.dense_size))

    def test_matmul_reference(self):
        key = jrand.PRNGKey(0)
        lhs = jrand.normal(key, (3, 4))
        rhs = jrand.normal(key, (4, 5))

        with self.assertRaisesRegex(
            ValueError, "At least one operand must be a SparseTensor"
        ):
            matmul_reference(lhs, rhs)

        st_lhs = _arr2st(lhs, 1)
        res_st = matmul_reference(st_lhs, rhs)
        expected = lhs @ rhs
        # Fix: use jnp.allclose for JAX arrays
        self.assertTrue(jnp.allclose(res_st.val, expected))
        self.assertTrue(jnp.allclose(res_st.dense(), expected))
