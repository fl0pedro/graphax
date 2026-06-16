"""MNIST vision models for the vertex-elimination / approx pipeline: a Vision
Transformer, a dilated convolution, and a soft Mixture-of-Experts.

Each is an ``(x, y, *weights) -> per-element squared error`` function in the same
shape as ``alphagrad.approx.common.examples._neural_network`` -- ``x`` is a flat
``(28*28,)`` MNIST image, ``y`` a ``(ncls,)`` label, output ``(ncls,)`` -- so the
approx registry exposes them as ``ViT`` / ``ConvNet`` / ``MoE`` (and ``Vmapped*``
for the batched variants) and ``maybe_scalar_loss`` reduces them to the MSE.

graphax's N-D elimination differentiates all three exactly: the convolution is a
real ``lax.conv_general_dilated`` (a single vertex, not im2col), the MoE batches
its experts with one ``vmap`` and routes them densely (no top-k gather), and the
ViT uses plain matmul attention over a prepended CLS token. Weights come from
``vit_weights`` / ``conv_weights`` / ``moe_weights``.
"""
import jax
import jax.numpy as jnp
import jax.random as jrand

IMG = 28                                   # MNIST image side
PATCH = 7                                   # ViT/MoE patch side
PATCH_DIM = PATCH * PATCH                    # 49: flattened patch length
N_PATCHES = (IMG // PATCH) ** 2              # 16: patches per image
KERNEL = 3                                   # conv kernel side
CONV_DILATION = (2, 2)                       # dilated 3x3 -> 5x5 receptive field
CONV_OUT = IMG - (KERNEL - 1) * CONV_DILATION[0]  # 24: VALID conv output side


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
# graphax differentiates jax.nn.gelu / jax.nn.softmax directly (erfc + the
# stop_gradient'd max-subtraction are handled), so we reuse them rather than
# re-deriving the kernels.
def gelu(x):
    return jax.nn.gelu(x, approximate=False)


def softmax(x, axis):
    return jax.nn.softmax(x, axis=axis)


def layer_norm(x, gamma, beta):
    # feature-major: normalise over axis 0; gamma/beta are (features, 1)
    mu = jnp.mean(x, axis=0, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=0, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * gamma + beta


def squared_error(pred, y):
    return 0.5 * (pred - y) ** 2


def glorot(key, shape, gain=1.0):
    # fan from the last two (matrix) axes so stacked weights (E, m, n) init right
    fan = shape[-1] + (shape[-2] if len(shape) > 1 else 1)
    return jrand.normal(key, shape) * (gain * jnp.sqrt(2.0 / fan))


def patchify(img):
    """flat ``(IMG*IMG,)`` image -> ``(PATCH_DIM, N_PATCHES)`` feature-major patches."""
    n = IMG // PATCH
    g = img.reshape(IMG, IMG).reshape(n, PATCH, n, PATCH).transpose(0, 2, 1, 3)
    return g.reshape(n * n, PATCH * PATCH).T


# --------------------------------------------------------------------------
# 1. Vision Transformer  (patch-embed -> 1 encoder block -> CLS -> classifier)
# --------------------------------------------------------------------------
def _attention(x, WQ, WK, WV):
    q, k, v = WQ @ x, WK @ x, WV @ x                # (d, S)
    p = softmax(q.T @ k / jnp.sqrt(x.shape[0]), axis=1)   # (S, S) token-token
    return v @ p.T                                   # (d, S)


def _encoder_block(x, WQ, WK, WV, W1, b1, W2, b2, g0, c0, g1, c1):
    h = x + _attention(layer_norm(x, g0, c0), WQ, WK, WV)
    return h + W2 @ gelu(W1 @ layer_norm(h, g1, c1) + b1) + b2


def vit_forward(Xp, PE, E, cls, WQ, WK, WV, W1, b1, W2, b2, g0, c0, g1, c1, Wc, bc):
    """patches Xp ``(PATCH_DIM, N_PATCHES)`` -> logits ``(ncls, 1)``."""
    x = jnp.concatenate([cls, E @ Xp], axis=1) + PE   # prepend CLS, add pos-embed
    x = _encoder_block(x, WQ, WK, WV, W1, b1, W2, b2, g0, c0, g1, c1)
    return Wc @ x[:, 0:1] + bc                        # classify the CLS token


def ViT(x, y, PE, E, cls, WQ, WK, WV, W1, b1, W2, b2, g0, c0, g1, c1, Wc, bc):
    """flat MNIST image x ``(784,)`` + label y ``(ncls,)`` -> squared error ``(ncls,)``.

    Weights are explicit (not ``*weights``) so ``jax.vmap`` over a batch can give a
    per-argument ``in_axes``, matching the ``_neural_network`` convention."""
    logits = vit_forward(patchify(x), PE, E, cls, WQ, WK, WV, W1, b1, W2, b2,
                         g0, c0, g1, c1, Wc, bc)
    return squared_error(logits[:, 0], y)


def vit_weights(key, d=16, hidden=None, ncls=10):
    hidden = hidden or 4 * d
    S = N_PATCHES + 1
    k = jrand.split(key, 9)
    return (
        glorot(k[0], (d, S)) * 0.02,            # PE  positional embedding
        glorot(k[1], (d, PATCH_DIM)),           # E   patch embedding
        glorot(k[2], (d, 1)) * 0.02,            # cls token
        glorot(k[3], (d, d)), glorot(k[4], (d, d)), glorot(k[5], (d, d)),  # WQ WK WV
        glorot(k[6], (hidden, d)), jnp.zeros((hidden, 1)),                  # W1 b1
        glorot(k[7], (d, hidden)), jnp.zeros((d, 1)),                       # W2 b2
        jnp.ones((d, 1)), jnp.zeros((d, 1)),    # g0 c0  layer-norm 0
        jnp.ones((d, 1)), jnp.zeros((d, 1)),    # g1 c1  layer-norm 1
        glorot(k[8], (ncls, d)), jnp.zeros((ncls, 1)),                      # Wc bc
    )


# --------------------------------------------------------------------------
# 2. Convolution: a real dilated lax.conv_general_dilated (single vertex)
# --------------------------------------------------------------------------
def conv_forward(x, Wker, bk, Wc, bc):
    """flat MNIST image x ``(784,)`` -> logits ``(ncls, 1)`` via a dilated conv."""
    o = jax.lax.conv_general_dilated(
        x.reshape(1, 1, IMG, IMG), Wker, window_strides=(1, 1), padding="VALID",
        rhs_dilation=CONV_DILATION, dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return Wc @ gelu(o + bk).reshape(-1, 1) + bc      # bk: (1, Cout, 1, 1)


def ConvNet(x, y, Wker, bk, Wc, bc):
    return squared_error(conv_forward(x, Wker, bk, Wc, bc)[:, 0], y)


def conv_weights(key, Cout=8, ncls=10):
    k = jrand.split(key, 2)
    return (glorot(k[0], (Cout, 1, KERNEL, KERNEL)), jnp.zeros((1, Cout, 1, 1)),  # Wker bk
            glorot(k[1], (ncls, Cout * CONV_OUT * CONV_OUT)), jnp.zeros((ncls, 1)))  # Wc bc


# --------------------------------------------------------------------------
# 3. Mixture of Experts, dense / soft routing over all experts (vmapped)
# --------------------------------------------------------------------------
def moe_forward(tokens, Wg, Wc, bc, W1s, b1s, W2s, b2s):
    """tokens ``(d, N_PATCHES)`` -> logits ``(ncls, 1)``; experts batched on axis 0."""
    gate = softmax(Wg @ tokens, axis=0)                     # (E, tokens)
    expert = lambda W1, b1, W2, b2: W2 @ gelu(W1 @ tokens + b1) + b2
    outs = jax.vmap(expert)(W1s, b1s, W2s, b2s)             # (E, d, tokens)
    out = jnp.sum(outs * gate[:, None, :], axis=0)          # gate-weighted -> (d, tokens)
    return Wc @ jnp.sum(out, axis=1, keepdims=True) + bc    # sum-pool tokens -> logits


def MoE(x, y, E, Wg, Wc, bc, W1s, b1s, W2s, b2s):
    """flat MNIST image x ``(784,)`` -> squared error; E embeds patches into tokens."""
    return squared_error(
        moe_forward(E @ patchify(x), Wg, Wc, bc, W1s, b1s, W2s, b2s)[:, 0], y)


def moe_weights(key, d=16, num_experts=4, hidden=None, ncls=10):
    hidden = hidden or 2 * d
    E = num_experts
    k = jrand.split(key, 6)
    return (
        glorot(k[0], (d, PATCH_DIM)),                       # E   patch -> token embed
        glorot(k[1], (E, d)),                               # Wg  gate
        glorot(k[2], (ncls, d)), jnp.zeros((ncls, 1)),      # Wc bc
        glorot(k[3], (E, hidden, d)), jnp.zeros((E, hidden, 1)),  # W1s b1s (stacked)
        glorot(k[4], (E, d, hidden)), jnp.zeros((E, d, 1)),       # W2s b2s (stacked)
    )
