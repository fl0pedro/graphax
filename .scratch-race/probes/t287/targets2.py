"""Targets chosen to REACH the structures the classes exist for: a convolution
(the band), a gather / embedding (a set), a reduce (replication), a reshape and
transpose chain (a permuted diagonal), attention (batched contraction)."""
import jax, jax.numpy as jnp

K = jax.random.PRNGKey(7)


def _n(shape, i):
    return jax.random.normal(jax.random.split(K, 12)[i], shape)


def conv1d():
    x, w = _n((1, 1, 16), 0), _n((1, 1, 3), 1)

    def conv1d(x, w):
        y = jax.lax.conv_general_dilated(x, w, (1,), "SAME")
        return jnp.sum(jnp.tanh(y) ** 2)
    return conv1d, (x, w), (0, 1)


def embedding():
    tab = _n((10, 4), 2)
    idx = jnp.array([0, 3, 3, 7, 1, 9])

    def embedding(tab):
        e = jnp.take(tab, idx, axis=0)
        return jnp.sum(jnp.tanh(e) ** 2)
    return embedding, (tab,), (0,)


def reduce_mean():
    x = _n((6, 5), 3)

    def reduce_mean(x):
        m = jnp.mean(x, axis=0)
        return jnp.sum(jnp.tanh(m) * jnp.sum(x, axis=0))
    return reduce_mean, (x,), (0,)


def reshape_chain():
    x = _n((12,), 4)

    def reshape_chain(x):
        y = jnp.tanh(x).reshape(3, 4).T.reshape(-1)
        return jnp.sum(y * jnp.exp(x))
    return reshape_chain, (x,), (0,)


def attention():
    q, k, v = _n((4, 8), 5), _n((4, 8), 6), _n((4, 8), 7)

    def attention(q, k, v):
        a = jax.nn.softmax(q @ k.T / 2.83, axis=-1)
        return jnp.sum(jnp.tanh(a @ v))
    return attention, (q, k, v), (0, 1, 2)


def layernorm():
    x, g = _n((5, 6), 8), _n((6,), 9)

    def layernorm(x, g):
        mu = jnp.mean(x, axis=-1, keepdims=True)
        sd = jnp.sqrt(jnp.mean((x - mu) ** 2, axis=-1, keepdims=True) + 1e-5)
        return jnp.sum(jnp.tanh((x - mu) / sd * g))
    return layernorm, (x, g), (0, 1)


def slice_concat():
    x = _n((12,), 10)

    def slice_concat(x):
        a, b = x[:6], x[6:]
        return jnp.sum(jnp.tanh(jnp.concatenate([b, a])) * x)
    return slice_concat, (x,), (0,)


TARGETS2 = {"conv1d": conv1d, "embedding": embedding,
            "reduce_mean": reduce_mean, "reshape_chain": reshape_chain,
            "attention": attention, "layernorm": layernorm,
            "slice_concat": slice_concat}
