import jax
import jax.nn as jnn
import jax.numpy as jnp


def superspike_surrogate(beta=10.):

    @jax.custom_jvp
    def heaviside_with_super_spike_surrogate(x):
        return jnp.heaviside(x, 1)

    @heaviside_with_super_spike_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_super_spike_surrogate(x)
        tangent_out = 1./(beta*jnp.abs(x)+1.) * x_dot
        return primal_out, tangent_out
    
    return heaviside_with_super_spike_surrogate


surrogate = superspike_surrogate()


def lif(U, I, S, a, b, threshold):
    U_next = a*U + (1.-a)*I
    I_next = b*I + (1.-b)*S
    S_next = surrogate(U_next - threshold)
    
    return U_next, I_next, S_next


def lif_cb(U, I, S, a, b, threshold):
    """Current-based LIF step: input current reaches the membrane the SAME step
    (I_next computed first, then U_next from I_next), so the weight gradient is
    non-zero even in a single-step (online) window. Same surrogate as ``lif``."""
    I_next = b * I + (1. - b) * S
    U_next = a * U + (1. - a) * I_next
    S_next = surrogate(U_next - threshold)
    return U_next, I_next, S_next


# From Bellec et al. e-prop paper
def ada_lif(U, a, S, alpha, beta, rho, threshold):
    U_next = alpha*U + S    
    A_th = threshold + beta*a
    S_next = jnn.sigmoid(U_next - A_th) # this needs to have spiking behavior jnp.heaviside(U_next - A_th, 1) # 
    a_next = rho*a - S_next
    
    return U_next, a_next, S_next


# Single SNN forward pass
def LIF_SNN(S_in, S_target, U1, U2, U3, I1, I2, I3, W1, W2, W3, alpha, beta, thresh):
    i1 = W1 @ S_in
    U1, a1, s1 = lif(U1, I1, i1, alpha, beta, thresh)
    i2 = W2 @ s1
    U2, a2, s2 = lif(U2, I2, i2, alpha, beta, thresh)
    i3 = W3 @ s2
    U3, a3, s3 = lif(U3, I3, i3, alpha, beta, thresh)
    return .5*(s3 - S_target)**2, U1, U2, U3, a1, a2, a3


# Single SNN forward pass 
def ADALIF_SNN(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh):
    i1 = W1 @ S_in
    U1, a1, s1 = ada_lif(U1, a1, i1, alpha, beta, rho, thresh)
    i2 = W2 @ s1
    U2, a2, s2 = ada_lif(U2, a2, i2, alpha, beta, rho, thresh)
    i3 = W3 @ s2
    U3, a3, s3 = ada_lif(U3, a3, i3, alpha, beta, rho, thresh)
    return .5*(s3 - S_target)**2, U1, U2, U3, a1, a2, a3

import os as _os
def ADALIF_SNN_SEQ(S_in_seq, S_target, U1, U2, U3, a1, a2, a3,
                   W1, W2, W3, alpha, beta, rho, thresh):
    """Temporal 3-layer ADAPTIVE LIF over a spike window ``S_in_seq`` (N, n_in).

    The loop is unrolled COMPLETELY -- every one of the N steps is in the
    differentiated graph. This is deliberately NOT the truncated-BPTT scheme
    LIF_SNN_SHD implements: there is no detached warm-up and no window, so the
    Jacobian is exact for the whole sequence. Use N=1 for the single-step
    ("one loop") case and N=T for the fully-unrolled ("multi loop / state")
    case; nothing in between is offered, because a partial window is exactly
    the approximation we are trying not to introduce here.

    Same 3-layer shape as :func:`ADALIF_SNN`; weights are args 8/9/10.
    """
    N = int(S_in_seq.shape[0])
    loss = 0.0
    for t in range(N):
        i1 = W1 @ S_in_seq[t]
        U1, a1, s1 = ada_lif(U1, a1, i1, alpha, beta, rho, thresh)
        i2 = W2 @ s1
        U2, a2, s2 = ada_lif(U2, a2, i2, alpha, beta, rho, thresh)
        i3 = W3 @ s2
        U3, a3, s3 = ada_lif(U3, a3, i3, alpha, beta, rho, thresh)
        loss = loss + jnp.mean(0.5 * (s3 - S_target) ** 2)
    return loss / N


def _snn_trunc():
    """ALPHAGRAD_SNN_TRUNC: unset->None (full BPTT unroll over all T);
    0 (or <0)->online (single step in the graph, recurrent carry is a leaf);
    N>0->truncated window of N steps unrolled into the grad/Jacobian graph."""
    v = _os.environ.get("ALPHAGRAD_SNN_TRUNC", None)
    if v is None or v == "":
        return None
    return int(v)


def LIF_SNN_SHD(S_in_seq, S_target, U1, U2, U3, I1, I2, I3,
                W1, W2, W3, alpha, beta, thresh):
    """Temporal 3-layer LIF over a spike WINDOW ``S_in_seq`` of shape (N, n_in).
    N is the REVERSE/Jacobian truncation window (set by the args builder from
    ALPHAGRAD_SNN_TRUNC). The recurrent carry ENTERING the window (U*, I*) is
    precomputed by a FULL forward pass over the earlier T-N timesteps (detached),
    so forward activations reflect the whole sequence while ONLY these N steps are
    differentiated. The elimination graph graphax sees is therefore a constant
    base + N * per-step block (truncated BPTT): online (N=1) is smallest, full
    (N=T) largest. Returns the scalar mean readout loss (differentiate W @ 8,9,10)."""
    N = int(S_in_seq.shape[0])
    loss = 0.0
    for t in range(N):
        i1 = W1 @ S_in_seq[t]
        U1, I1, s1 = lif_cb(U1, I1, i1, alpha, beta, thresh)
        i2 = W2 @ s1
        U2, I2, s2 = lif_cb(U2, I2, i2, alpha, beta, thresh)
        i3 = W3 @ s2
        U3, I3, s3 = lif_cb(U3, I3, i3, alpha, beta, thresh)
        loss = loss + jnp.mean(0.5 * (s3 - S_target) ** 2)
    return loss / N
