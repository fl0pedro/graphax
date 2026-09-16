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
                W1, W2, W3, alpha, beta, thresh):
    """Temporal 3-layer LIF over a spike WINDOW ``S_in_seq`` of shape (N, n_in).
    N is the GRADIENT WINDOW (set by the args builder from alphagrad's
    ``--target-grad-window``). The recurrent carry ENTERING the window (U*, I*) is
    precomputed by a FULL forward pass over the earlier T-N timesteps (detached),
    so forward activations reflect the whole sequence while ONLY these N steps are
    differentiated. The elimination graph graphax sees is therefore a constant
    base + N * per-step block (truncated BPTT): online (N=1) is smallest, full
    (N=T) largest. Returns the scalar mean readout loss (differentiate W @ 8,9,10).

"""
    N = int(S_in_seq.shape[0])
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
                   W1, W2, W3, alpha, beta, rho, thresh):
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


# ===========================================================================
# THE RECURRENT SHD STEP (owner ruling 2026-09-16)
#
# ONE recurrent step is the whole elimination graph. Its inputs are the
# weights, the carried state ``s_(t-1)`` and the input frame ``x_t``; its
# outputs are ``s_t`` and the step loss. Temporal credit does NOT enter as
# more step copies. It enters as EDGES WITH GIVEN VALUES, computed numerically
# outside the graph over the whole untouched recording:
#
#   tbptt  no temporal edge. ``s_(t-1)`` is a constant. Truncated, spatial
#          only. The baseline.
#   bptt   the FUTURE feeds in. An edge from ``s_t`` to the loss carries the
#          adjoint ``lambda_(t+1) = dL_(>t)/ds_t`` from a detached backward
#          pass over the suffix. The graph's gradient is the exact per-step
#          contribution of full backpropagation through time.
#   rtrl   the PAST feeds in. An edge ``W -> s_(t-1)`` carries
#          ``J_(t-1) = ds_(t-1)/dW`` from a detached pass over the prefix.
#          Eliminating that vertex multiplies ``J_(t-1)`` through
#          ``A_t = ds_t/ds_(t-1)``, which is one step of real-time recurrent
#          learning, and the gradient is exact through the whole prefix.
#
# THE MODEL is the recurrent spiking network of Zenke and Neftci (arXiv
# 2010.11931). Its architecture, surrogate and weight init are the ones of
# Zenke's public SpyTorch tutorial 4, which is the SHD network of that paper
# (github.com/fzenke/spytorch, notebooks/SpyTorchTutorial4.ipynb):
# 700 inputs, a RECURRENTLY connected hidden layer, 20 outputs, a leaky
# non-spiking readout, ``SurrGradSpike.scale = 100``, and
# ``std = 0.2 / sqrt(fan_in)``. The threshold adaptation is Bellec et al.'s
# ALIF, which is the adaptive unit Zenke and Neftci write their e-prop-like
# rule for. The decay constants belong to the caller, because the time step
# here is the SHD loader's 10 ms bin and not Zenke's 1 ms.
#
# WHY THE RECURRENT WEIGHTS MATTER. With ``V`` present, ``A_t`` has
# off-diagonal terms: hidden unit i at step t sees hidden unit j at step t-1.
# So the carried Jacobian is NOT block diagonal, and e-prop's drop of that
# coupling is a real approximation the policy can find. Without ``V`` the
# block diagonal is exact and there is nothing to learn.
# ===========================================================================

#: The five carried state components, in the order the signature lists them.
RSNN_STATE_NAMES: tuple[str, ...] = ("S", "I", "U", "a", "Uo")

#: The three weight matrices, in the order the signature lists them.
RSNN_WEIGHT_NAMES: tuple[str, ...] = ("W", "V", "Wo")

#: Which ``(state, weight)`` blocks of the carried Jacobian are carried, in
#: the order the varargs hold them. The readout weight ``Wo`` feeds NOTHING
#: back, so the four blocks ``(S, Wo) (I, Wo) (U, Wo) (a, Wo)`` are
#: structurally zero and are not carried. The other eleven are the whole
#: influence matrix of one RTRL step.
RSNN_CARRY_BLOCKS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, 1),          # S  <- W, V
    (1, 0), (1, 1),          # I  <- W, V
    (2, 0), (2, 1),          # U  <- W, V
    (3, 0), (3, 1),          # a  <- W, V
    (4, 0), (4, 1), (4, 2),  # Uo <- W, V, Wo
)

#: The four blocks that cannot be non-zero. A builder asserts they are zero
#: rather than assuming it.
RSNN_ZERO_BLOCKS: tuple[tuple[int, int], ...] = (
    (0, 2), (1, 2), (2, 2), (3, 2),
)

#: Zenke's surrogate slope (``SurrGradSpike.scale`` in SpyTorch tutorial 4).
RSNN_SURROGATE_SCALE = 100.0


def superspike_sq_surrogate(scale: float = RSNN_SURROGATE_SCALE):
    """Zenke's SuperSpike surrogate: ``1 / (scale |x| + 1)^2`` on the backward.

    This is the one SpyTorch uses (``SurrGradSpike``), squared denominator and
    ``scale = 100``. :func:`superspike_surrogate` above is the UNSQUARED form
    with ``beta = 10`` and is kept for the targets that already use it.
    """
    @jax.custom_jvp
    def theta(x):
        return jnp.heaviside(x, 1.0)

    @theta.defjvp
    def theta_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        return theta(x), x_dot / (scale * jnp.abs(x) + 1.0) ** 2

    return theta


rsnn_surrogate = superspike_sq_surrogate()


def rsnn_cell(x, S, I, U, a, Uo, W, V, Wo,
              a_syn, a_mem, a_out, rho, beta_a, thresh):
    """ONE step of the recurrent adaptive-LIF network. Returns ``s_t``.

    ``S`` are the hidden spikes of the previous step, ``I`` the synaptic
    current, ``U`` the membrane, ``a`` the adaptation variable and ``Uo`` the
    readout membrane. The recurrent term is ``V @ S``: that is the coupling
    that makes ``d s_t / d s_(t-1)`` dense, and it is the term e-prop drops.
    """
    I_n = a_syn * I + W @ x + V @ S
    U_n = a_mem * U + (1.0 - a_mem) * I_n - thresh * S
    A_th = thresh + beta_a * a
    S_n = rsnn_surrogate(U_n - A_th)
    a_n = rho * a + S_n
    Uo_n = a_out * Uo + (1.0 - a_out) * (Wo @ S_n)
    return S_n, I_n, U_n, a_n, Uo_n


def attach_rsnn_past(states, weights, given):
    """RTRL: give every carried state an edge to the weights, valued by ``J``.

    ``given`` is ``(W_ref, V_ref, Wo_ref, J_0, ..., J_10)`` where ``J_k`` is
    the block :data:`RSNN_CARRY_BLOCKS` names at position ``k``. Each reference
    weight is a bit-for-bit copy of its weight and sits OUTSIDE ``argnums``, so
    ``W - W_ref`` is exactly zero (no forward value moves, and a run under any
    rule sees the same loss to the last bit) and its edge to the weight is the
    identity. A `stop_gradient` would give the same zero delta but would add an
    edge-free VERTEX the policy has to eliminate for nothing.
    """
    refs, blocks = given[:3], given[3:]
    out = list(states)
    with snn_carry_scope():
        deltas = [W - R for W, R in zip(weights, refs)]
        for (s, w), J in zip(RSNN_CARRY_BLOCKS, blocks):
            want = tuple(out[s].shape) + tuple(weights[w].shape)
            if tuple(J.shape) != want:
                raise ValueError(
                    f"carried block ({RSNN_STATE_NAMES[s]}, "
                    f"{RSNN_WEIGHT_NAMES[w]}) has shape {tuple(J.shape)}; "
                    f"d state / d W is {want}")
            out[s] = out[s] + jnp.tensordot(J, deltas[w], 2)
    return tuple(out)


def attach_rsnn_future(loss, next_states, given):
    """BPTT: give ``s_t`` an edge to the future loss, valued by the adjoint.

    ``given`` is the five adjoints ``lambda_(t+1) = dL_(>t)/ds_t``, one per
    component of :data:`RSNN_STATE_NAMES`, from a detached backward pass over
    the suffix of the recording. The returned scalar is
    ``L_t + <lambda_(t+1), s_t>``, whose gradient with respect to the weights
    is EXACTLY the per-step contribution full backpropagation through time
    makes at step ``t``.
    """
    if len(given) != len(RSNN_STATE_NAMES):
        raise ValueError(
            f"a future-adjoint attachment needs one adjoint per state "
            f"component ({len(RSNN_STATE_NAMES)}), got {len(given)}")
    with snn_carry_scope():
        for lam, s in zip(given, next_states):
            if tuple(lam.shape) != tuple(s.shape):
                raise ValueError(
                    f"adjoint shape {tuple(lam.shape)} does not match the "
                    f"state it multiplies, {tuple(s.shape)}")
            loss = loss + jnp.sum(lam * s)
    return loss


#: How many varargs each temporal rule passes. The length IS the selector, so
#: the three are kept distinct by construction and a length outside this table
#: raises rather than choosing a rule by accident.
RSNN_GIVEN_LENGTHS: dict[int, str] = {
    0: "tbptt",
    len(RSNN_STATE_NAMES): "bptt",
    3 + len(RSNN_CARRY_BLOCKS): "rtrl",
}


def RSNN_SHD(x, y, S, I, U, a, Uo, W, V, Wo,
             a_syn, a_mem, a_out, rho, beta_a, thresh, *given):
    """ONE recurrent step of the SHD network, plus its step loss.

    Weights are args 7, 8 and 9 (``W``, ``V``, ``Wo``); ``V`` is among them,
    so the recurrent coupling is learned and ``d s_t / d s_(t-1)`` has real
    off-diagonal terms.

    ``given`` selects the temporal rule by its LENGTH
    (:data:`RSNN_GIVEN_LENGTHS`):

      0 entries   tbptt. The carried state is a constant.
      5 entries   bptt. The five future adjoints; see
                  :func:`attach_rsnn_future`.
      14 entries  rtrl. Three reference weights and the eleven carried
                  Jacobian blocks; see :func:`attach_rsnn_past`.

    The loss is the softmax cross entropy of the leaky readout membrane
    against the one-hot label ``y``. It is a PER-STEP loss on purpose: the
    whole point of this target is that the sequence loss decomposes as
    ``sum_t L_t``, so one step is a complete object and the three rules
    differ only in which given quantity meets it.
    """
    rule = RSNN_GIVEN_LENGTHS.get(len(given))
    if rule is None:
        raise ValueError(
            f"RSNN_SHD got {len(given)} extra arguments; the temporal rule is "
            f"selected by that count and the legal counts are "
            f"{sorted(RSNN_GIVEN_LENGTHS)} "
            f"({', '.join(RSNN_GIVEN_LENGTHS[k] for k in sorted(RSNN_GIVEN_LENGTHS))})")
    weights = (W, V, Wo)
    if rule == "rtrl":
        S, I, U, a, Uo = attach_rsnn_past((S, I, U, a, Uo), weights, given)
    with snn_step_scope(0):
        nxt = rsnn_cell(x, S, I, U, a, Uo, W, V, Wo,
                        a_syn, a_mem, a_out, rho, beta_a, thresh)
        loss = jnp.sum(-y * jax.nn.log_softmax(nxt[4]))
    if rule == "bptt":
        loss = attach_rsnn_future(loss, nxt, given)
    return loss
