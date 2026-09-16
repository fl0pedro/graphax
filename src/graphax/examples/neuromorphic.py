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

#: The ``jax.named_scope`` prefix every unrolled time step of a temporal model
#: in this module carries. One step copy ``t`` of the elimination graph is
#: exactly the equations whose ``source_info.name_stack`` is ``"snn_step_<t>"``,
#: and everything outside such a scope is the BASE (the readout that closes the
#: loss). alphagrad's ``common/temporal_order.py`` reads this name to give every
#: vertex its step index; the two sides must never spell it differently, so the
#: name lives here, next to the loop that emits it, and is imported there.
SNN_STEP_SCOPE = "snn_step"


def snn_step_scope(t: int):
    """The named scope of time step ``t``. Wrap ONE unrolled step body in it."""
    return jax.named_scope(f"{SNN_STEP_SCOPE}_{int(t)}")


#: The ``jax.named_scope`` that wraps the CARRIED-JACOBIAN attachment block --
#: the real-time-recurrent-learning (RTRL) boundary of the gradient window. It
#: is entered INSIDE ``snn_step_scope(0)``, so a vertex of this block carries
#: step 0 for the temporal order constraint (it is the earliest work there is)
#: while still being findable by its own name for a probe or a test.
SNN_CARRY_SCOPE = "snn_carry_jac"


class snn_carry_scope:
    """The named scope of the carried-Jacobian attachment block.

    It enters ``snn_step_scope(0)`` FIRST and its own name second, so every
    equation of the block reads ``snn_step_0/snn_carry_jac`` on its name stack:
    alphagrad's step-tag reader gives it step 0 -- the earliest work in the
    window, which is where real-time recurrent learning starts -- and a probe
    can still pick the block out by its own name.
    """

    def __enter__(self):
        self._outer = snn_step_scope(0)
        self._outer.__enter__()
        self._inner = jax.named_scope(SNN_CARRY_SCOPE)
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        self._inner.__exit__(*exc)
        return self._outer.__exit__(*exc)


#: WHICH CARRIED JACOBIAN BLOCKS A THREE-LAYER SHD TARGET TAKES, in the order
#: the varargs of :func:`LIF_SNN_SHD` / :func:`ADALIF_SNN_SHD` carry them.
#:
#: Each entry is ``(state slot, weight slot)``. The state slot indexes the SIX
#: carried states in the order the signature lists them -- ``U1, U2, U3`` then
#: ``a1, a2, a3`` (``I1, I2, I3`` for the LIF twin) -- and the weight slot
#: indexes ``W1, W2, W3``. Block ``(s, w)`` holds ``d state_s / d W_w`` at the
#: step the window starts, so its shape is ``state_shape + W_w.shape``.
#:
#: WHY TWELVE AND NOT EIGHTEEN. The three layers are feed-forward in space, so
#: ``W2`` and ``W3`` cannot reach layer 1 and ``W3`` cannot reach layer 2: six
#: of the eighteen blocks are STRUCTURALLY zero and are not carried. The other
#: twelve are the exact influence matrix of one RTRL step, so a plan built on
#: them reproduces the gradient through the WHOLE prefix, not a truncation.
#:
#: THE BLOCK DIAGONAL of this set -- ``(U1,W1) (a1,W1) (U2,W2) (a2,W2)
#: (U3,W3) (a3,W3)`` -- is what e-prop keeps; the other six are the
#: cross-layer coupling Zenke and Neftci's block-diagonal approximation drops.
SHD_CARRY_BLOCKS: tuple[tuple[int, int], ...] = (
    (0, 0),                  # U1 <- W1
    (1, 0), (1, 1),          # U2 <- W1, W2
    (2, 0), (2, 1), (2, 2),  # U3 <- W1, W2, W3
    (3, 0),                  # a1 (I1) <- W1
    (4, 0), (4, 1),          # a2 (I2) <- W1, W2
    (5, 0), (5, 1), (5, 2),  # a3 (I3) <- W1, W2, W3
)

#: The block-diagonal subset of :data:`SHD_CARRY_BLOCKS`: the within-layer
#: blocks, which is exactly what an e-prop eligibility trace carries.
SHD_CARRY_DIAGONAL_BLOCKS: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (2, 2), (3, 0), (4, 1), (5, 2),
)


def attach_carried_jacobians(states, weights, carried):
    """Give every carried state an EDGE to the weights, valued by ``carried``.

    ``states`` are the six carried states in signature order and ``weights``
    are ``(W1, W2, W3)``. ``carried`` is either empty -- backpropagation
    through time, nothing is emitted and the graph is the one this module
    always built -- or the FIFTEEN entries

        ``(W1_ref, W2_ref, W3_ref, J_0, ..., J_11)``

    where ``W*_ref`` is a bit-for-bit copy of the matching weight that is NOT
    differentiated, and ``J_k`` is the block :data:`SHD_CARRY_BLOCKS` names at
    position ``k``.

    The returned states have the SAME VALUES, bit for bit, and a Jacobian with
    respect to the weights that is exactly the carried one.

    HOW, and why this shape. The attachment is

        ``dW_w   = W_w - W_w_ref``
        ``state_s <- state_s + tensordot(J_sw, dW_w)``

    ``W_w - W_w_ref`` is EXACTLY zero, so no forward value moves and a
    backpropagation-through-time run and a real-time-recurrent-learning run of
    the same recording see the same loss to the last bit. What it does move is
    the GRAPH: the subtraction is a vertex whose edge from ``W_w`` is the
    identity, the contraction is a vertex whose edge from it is ``J_sw``, and
    the addition is the carried state the window's first step reads.
    Eliminating the contraction vertex is ONE RTRL STEP -- it multiplies the
    carried Jacobian by the state-to-state Jacobian the step body supplies --
    and that face is where an e-prop-like approximation lives.

    WHY A REFERENCE ARGUMENT AND NOT ``stop_gradient``. Both give an exactly
    zero delta. ``stop_gradient`` would add three equations that carry no edge
    at all, and an edge-free equation is still a VERTEX the policy has to
    choose and eliminate for nothing. The reference weights sit outside
    ``argnums``, so graphax's forward pruning gives them no edge and they add
    no vertex.
    """
    if not carried:
        return tuple(states)
    n = len(SHD_CARRY_BLOCKS)
    if len(carried) != 3 + n:
        raise ValueError(
            f"a carried-Jacobian attachment needs three reference weights and "
            f"then exactly {n} blocks, in the order of "
            f"graphax.examples.neuromorphic.SHD_CARRY_BLOCKS, so {3 + n} "
            f"entries; got {len(carried)}. Six of the eighteen (state, weight) "
            f"pairs are structurally zero and are the ones left out, so a "
            f"shorter tuple is an approximation, not a saving.")
    refs, blocks = carried[:3], carried[3:]
    out = list(states)
    with snn_carry_scope():
        # One delta per weight, shared by every block that reads it. The value
        # is exactly 0.0 and the edge to the weight is the identity.
        deltas = [W - R for W, R in zip(weights, refs)]
        for (s, w), J in zip(SHD_CARRY_BLOCKS, blocks):
            want = tuple(out[s].shape) + tuple(weights[w].shape)
            if tuple(J.shape) != want:
                raise ValueError(
                    f"carried block (state {s}, weight {w}) has shape "
                    f"{tuple(J.shape)}; d state / d W is {want}")
            out[s] = out[s] + jnp.tensordot(J, deltas[w], 2)
    return tuple(out)


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
        with snn_step_scope(t):
            i1 = W1 @ S_in_seq[t]
            U1, a1, s1 = ada_lif(U1, a1, i1, alpha, beta, rho, thresh)
            i2 = W2 @ s1
            U2, a2, s2 = ada_lif(U2, a2, i2, alpha, beta, rho, thresh)
            i3 = W3 @ s2
            U3, a3, s3 = ada_lif(U3, a3, i3, alpha, beta, rho, thresh)
            loss = loss + jnp.mean(0.5 * (s3 - S_target) ** 2)
    return loss / N


def LIF_SNN_SHD(S_in_seq, S_target, U1, U2, U3, I1, I2, I3,
                W1, W2, W3, alpha, beta, thresh, *carried):
    """Temporal 3-layer LIF over a spike WINDOW ``S_in_seq`` of shape (N, n_in).
    N is the GRADIENT WINDOW (set by the args builder from alphagrad's
    ``--target-grad-window``). The recurrent carry ENTERING the window (U*, I*) is
    precomputed by a FULL forward pass over the earlier T-N timesteps (detached),
    so forward activations reflect the whole sequence while ONLY these N steps are
    differentiated. The elimination graph graphax sees is therefore a constant
    base + N * per-step block (truncated BPTT): online (N=1) is smallest, full
    (N=T) largest. Returns the scalar mean readout loss (differentiate W @ 8,9,10).

    ``carried`` is EMPTY for backpropagation through time and holds the twelve
    :data:`SHD_CARRY_BLOCKS` for real-time recurrent learning; see
    :func:`attach_carried_jacobians`."""
    N = int(S_in_seq.shape[0])
    U1, U2, U3, I1, I2, I3 = attach_carried_jacobians(
        (U1, U2, U3, I1, I2, I3), (W1, W2, W3), carried)
    loss = 0.0
    for t in range(N):
        with snn_step_scope(t):
            i1 = W1 @ S_in_seq[t]
            U1, I1, s1 = lif_cb(U1, I1, i1, alpha, beta, thresh)
            i2 = W2 @ s1
            U2, I2, s2 = lif_cb(U2, I2, i2, alpha, beta, thresh)
            i3 = W3 @ s2
            U3, I3, s3 = lif_cb(U3, I3, i3, alpha, beta, thresh)
            loss = loss + jnp.mean(0.5 * (s3 - S_target) ** 2)
    return loss / N


def ADALIF_SNN_SHD(S_in_seq, S_target, U1, U2, U3, a1, a2, a3,
                   W1, W2, W3, alpha, beta, rho, thresh, *carried):
    """The ADAPTIVE-LIF twin of :func:`LIF_SNN_SHD`, same contract.

    Temporal 3-layer adaptive LIF over a spike WINDOW ``S_in_seq`` of shape
    ``(N, n_in)``. ``N`` is the GRADIENT WINDOW: the steps BEFORE it are run by
    the args builder as a detached forward pass and reach this function only as
    the carry ``(U*, a*)``, so the elimination graph is a constant BASE plus
    ``N`` per-step blocks -- exactly the truncated-BPTT shape LIF_SNN_SHD has.

    The cell is :func:`ada_lif` (Bellec et al. e-prop), the same cell
    :func:`ADALIF_SNN` and :func:`ADALIF_SNN_SEQ` use, so the adaptation state
    ``a*`` replaces LIF's synaptic current ``I*`` and one extra decay ``rho``
    joins the signature. The WEIGHTS STAY AT ARGS 8/9/10, so ``--argnums
    8,9,10`` is the same on every member of this family.

    Every step body sits in its own :func:`snn_step_scope`, so the elimination
    graph's vertices carry their step index and the temporal order constraint
    can read it. Returns the scalar mean readout loss.

    ``carried`` SELECTS THE TEMPORAL RULE. Empty (the default) is
    backpropagation through time: the carry enters as a constant and the
    gradient is the truncated one. The twelve :data:`SHD_CARRY_BLOCKS` make it
    real-time recurrent learning: the carried influence matrix enters as a
    given edge from the weights to the carried state, so eliminating that
    vertex is one RTRL step and the gradient is exact through the WHOLE
    prefix. See :func:`attach_carried_jacobians`.
    """
    N = int(S_in_seq.shape[0])
    U1, U2, U3, a1, a2, a3 = attach_carried_jacobians(
        (U1, U2, U3, a1, a2, a3), (W1, W2, W3), carried)
    loss = 0.0
    for t in range(N):
        with snn_step_scope(t):
            i1 = W1 @ S_in_seq[t]
            U1, a1, s1 = ada_lif(U1, a1, i1, alpha, beta, rho, thresh)
            i2 = W2 @ s1
            U2, a2, s2 = ada_lif(U2, a2, i2, alpha, beta, rho, thresh)
            i3 = W3 @ s2
            U3, a3, s3 = ada_lif(U3, a3, i3, alpha, beta, rho, thresh)
            loss = loss + jnp.mean(0.5 * (s3 - S_target) ** 2)
    return loss / N
