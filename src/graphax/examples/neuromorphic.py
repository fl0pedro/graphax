from typing import NamedTuple

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

#: Which ``(state, weight)`` blocks of the carried Jacobian are carried. The
#: readout weight ``Wo`` feeds NOTHING back, so the four blocks ``(S, Wo)
#: (I, Wo) (U, Wo) (a, Wo)`` are structurally zero and are not carried. The
#: other eleven are the whole influence matrix of one RTRL step. The varargs
#: hold them STACKED per weight, see :data:`RSNN_CARRY_STACKS`.
RSNN_CARRY_BLOCKS: tuple[tuple[int, int], ...] = (
    (0, 0), (0, 1),          # S  <- W, V
    (1, 0), (1, 1),          # I  <- W, V
    (2, 0), (2, 1),          # U  <- W, V
    (3, 0), (3, 1),          # a  <- W, V
    (4, 0), (4, 1), (4, 2),  # Uo <- W, V, Wo
)

#: The four hidden state components, the ones a hidden weight reaches.
RSNN_HIDDEN_STATES: tuple[int, ...] = (0, 1, 2, 3)

#: How the varargs hold the eleven blocks (owner ruling 2026-09-24, Q31a):
#: ``((states, weight), ...)``, one entry per stacked tensor. The four hidden
#: blocks of one weight are ONE tensor with a leading stack axis of extent 4,
#: in the state order ``S, I, U, a``; the three readout blocks stay single.
#: Five entries, so the attachment contracts five tensors instead of eleven.
RSNN_CARRY_STACKS: tuple[tuple[tuple[int, ...], int], ...] = (
    (RSNN_HIDDEN_STATES, 0),   # (4, h, .., n_in)  S, I, U, a <- W
    (RSNN_HIDDEN_STATES, 1),   # (4, h, .., h)     S, I, U, a <- V
    ((4,), 0), ((4,), 1), ((4,), 2),   # Uo <- W, V, Wo
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


#: The dtype a QUANTIZED carry is stored in. bfloat16 shares float32's
#: exponent range, which is the reason CONTEXT.md gives for it being the only
#: narrow dtype in this project.
RSNN_CARRY_QUANT_DTYPE = jnp.bfloat16


class CarryContainer(NamedTuple):
    """WHICH CONTAINER a carried-Jacobian block arrived in.

    OWNER RULING, 2026-09-16. The container follows the PLAN's approximation
    on the carried-Jacobian face, for all four action classes and every
    combination of them. The three flags below are the three classes that
    change how a block is STORED; ``Skip`` is not one of them, because a
    skipped carry face carries no block at all (the rule becomes truncated
    backpropagation through time and the given tuple is empty).

    ``diag``    the state axis is diagonal, so the block is stored at the
                shape of the WEIGHT it differentiates. This is the
                eligibility trace of e-prop: one number per synapse instead
                of one per (unit, synapse) pair.
    ``reduce``  the LAST axis of the weight -- the presynaptic index -- is
                IMPLICIT. One value is stored and read as a broadcast, and
                the contraction expands it only where it has to (CONTEXT.md,
                Reduce and Implicit axis). The stored axis has extent 1.
    ``quant``   the block is stored in :data:`RSNN_CARRY_QUANT_DTYPE`.

    ``name`` is the canonical spelling used in records, logs and tests:
    ``exact`` for no flag at all, otherwise the set flags joined by ``+`` in
    the order ``diag``, ``reduce``, ``quant``.
    """

    diag: bool = False
    reduce: bool = False
    quant: bool = False

    @property
    def name(self) -> str:
        parts = [n for n, on in
                 (("diag", self.diag), ("reduce", self.reduce),
                  ("quant", self.quant)) if on]
        return "+".join(parts) if parts else "exact"

    def block_shape(self, state_shape, weight_shape) -> tuple:
        """The shape a block of this container has for one (state, weight)."""
        w = tuple(weight_shape)
        if self.reduce:
            w = w[:-1] + (1,)
        return w if self.diag else tuple(state_shape) + w

    def dtype(self, base_dtype):
        return RSNN_CARRY_QUANT_DTYPE if self.quant else base_dtype


#: Every container a carried block can arrive in, by canonical name. The
#: ``compact`` and ``dense`` spellings of the first design are gone: a
#: container is now a SET of classes, because the plan can request more than
#: one on the same face.
RSNN_CARRY_CONTAINERS: tuple[str, ...] = tuple(
    CarryContainer(d, r, q).name
    for d in (False, True) for r in (False, True) for q in (False, True))


def carry_container_from_name(name) -> CarryContainer:
    """``"diag+quant"`` -> :class:`CarryContainer`, or raise."""
    text = str(name)
    if text == "exact":
        return CarryContainer()
    parts = [p for p in text.split("+") if p]
    known = ("diag", "reduce", "quant")
    for p in parts:
        if p not in known:
            raise ValueError(
                f"carry container {text!r} names {p!r}, which is not one of "
                f"{list(known)}. The legal names are "
                f"{list(RSNN_CARRY_CONTAINERS)}.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"carry container {text!r} names a class twice")
    return CarryContainer("diag" in parts, "reduce" in parts,
                          "quant" in parts)


def rsnn_carry_container(block, state_shape, weight_shape, given_shape,
                         given_dtype=None, n_stacked: int = 1) -> CarryContainer:
    """Which container a carried block arrived in, or raise.

    The SHAPE decides ``diag`` and ``reduce`` and the DTYPE decides ``quant``,
    and none of the four shapes is ambiguous: a state axis is never empty and
    the presynaptic axis of this model is never 1 (it is 700, 128 or 128).

    ``n_stacked > 1`` means ``given_shape`` is a STACK of that many same-shaped
    blocks (:data:`RSNN_CARRY_STACKS`): the leading axis is the stack axis,
    not a state axis, and the container is read off the shape behind it.
    """
    s, w = block
    got = tuple(given_shape)
    if n_stacked > 1:
        if not got or got[0] != n_stacked:
            raise ValueError(
                f"carried stack ({RSNN_STATE_NAMES[s]}.., "
                f"{RSNN_WEIGHT_NAMES[w]}) has shape {got}; a stack of "
                f"{n_stacked} blocks leads with an axis of extent {n_stacked}.")
        got = got[1:]
    want = {}
    for d in (False, True):
        for r in (False, True):
            c = CarryContainer(d, r, False)
            want[c.block_shape(state_shape, weight_shape)] = c
    c = want.get(got)
    if c is None:
        lines = ", ".join(
            f"{k} = {v.name}" for k, v in want.items())
        raise ValueError(
            f"carried block ({RSNN_STATE_NAMES[s]}, {RSNN_WEIGHT_NAMES[w]}) "
            f"has shape {got}. The containers this attachment can read are "
            f"{lines}. Nothing else is a container: a block of another shape "
            f"is a producer that disagrees with the plan that asked for it.")
    quant = (given_dtype is not None
             and jnp.dtype(given_dtype) == jnp.dtype(RSNN_CARRY_QUANT_DTYPE))
    return CarryContainer(c.diag, c.reduce, quant)


def _is_sparse_tensor(x) -> bool:
    from graphax.sparse.tensor import SparseTensor
    return isinstance(x, SparseTensor)


def _pure_pair(a, b) -> bool:
    return (a.is_sparse and b.is_sparse and a.other_id == b.id
            and b.other_id == a.id and a.block_size is None
            and b.block_size is None and a.axis is not None
            and a.axis == b.axis)


def _scaled_val(st):
    from graphax.sparse.dtype_compute import _scaled_mul
    v = jnp.asarray(1.0, st.dtype) if st.val is None else st.val
    return _scaled_mul(v, st.scalar_mult)


def _arrange(w, phys, sizes):
    # ``w`` with axis ``phys[k]`` (or a broadcast when None) as output axis k.
    if w.ndim == 0:
        phys = [None] * len(phys)
    used = [a for a in phys if a is not None]
    if len(set(used)) != len(used):
        raise ValueError(f"a stored axis is read twice: {phys}")
    extra = [a for a in range(w.ndim) if a not in used]
    if any(int(w.shape[a]) != 1 for a in extra):
        raise ValueError(
            f"the stored row has axes {extra} of shape {w.shape} that no "
            f"dim of the container reads")
    w = jnp.transpose(w, used + extra).reshape([w.shape[a] for a in used])
    it = iter(range(len(used)))
    w = w.reshape([w.shape[next(it)] if a is not None else 1 for a in phys])
    return jnp.broadcast_to(w, tuple(sizes))


def _refuse(st, why):
    raise ValueError(
        f"the state row's stored class {tuple(st.dims)} (val "
        f"{None if st.val is None else tuple(st.val.shape)}) is not one the "
        f"container reads without expansion: {why}")


def _block_diagonal(st, m_size):
    # lead + (n, m): the entries s == r of the logical lead + (n,) + (r, m)
    # row, read off the stored class. A pure pair (s, r) stores the diagonal
    # on its shared axis; dense s and r hold it on their diagonal; an
    # implicit s or r is uniform along it, so the stored value is the
    # diagonal. ``m_size`` 1 keeps an implicit m unexpanded (the reduce class).
    from graphax.sparse.ops.dense import _expand_implicit_blocks
    st = _expand_implicit_blocks(st)
    lead, ds = st.out_dims[:-1], st.out_dims[-1]
    dr, dm = st.primal_dims
    if any(d.is_sparse for d in lead) or dm.is_sparse:
        _refuse(st, "a batch or presynaptic dim is paired")
    v = _scaled_val(st)
    sizes = [d.logical_size for d in lead] + [ds.logical_size, m_size]
    if _pure_pair(ds, dr):
        phys = [d.axis for d in lead] + [ds.axis, dm.axis]
    elif (ds.is_sparse and dr.is_sparse and ds.other_id == dr.id
          and dr.other_id == ds.id and ds.block_size is not None
          and ds.block_size == dr.block_size and ds.axis == dr.axis
          and ds.block_axis is not None and dr.block_axis is not None
          and v.ndim):
        # A BLOCKED PAIR: N blocks of B x B on the block axes, the block
        # index on the shared outer axis (implicit when N is 1). The diagonal
        # of every block, then the block index merged in front of it.
        a, b = int(ds.block_axis), int(dr.block_axis)
        N, B = int(ds.size), int(ds.block_size)
        v = jnp.diagonal(v, axis1=a, axis2=b)
        rest = [ax for ax in range(v.ndim + 1) if ax not in (a, b)]

        def new(ax):
            return None if ax is None else rest.index(int(ax))
        diag = len(rest)
        if ds.axis is None:
            v = jnp.broadcast_to(v[..., None, :], v.shape[:-1] + (N, B))
            outer = diag
            diag = diag + 1
        else:
            outer = new(ds.axis)
        keep = [k for k in range(v.ndim) if k not in (outer, diag)]
        perm = keep + [outer, diag]
        shp = [int(v.shape[k]) for k in perm]
        v = jnp.transpose(v, perm).reshape(shp[:-2] + [shp[-2] * shp[-1]])

        def moved(ax):
            return None if ax is None else keep.index(ax)
        phys = ([moved(new(d.axis)) for d in lead]
                + [len(keep), moved(new(dm.axis))])
    elif not ds.is_sparse and not dr.is_sparse:
        if ds.axis is not None and dr.axis is not None and v.ndim:
            a, b = int(ds.axis), int(dr.axis)
            v = jnp.diagonal(v, axis1=a, axis2=b)
            rest = [ax for ax in range(v.ndim + 1) if ax not in (a, b)]

            def new(ax):
                return None if ax is None else rest.index(int(ax))
            phys = [new(d.axis) for d in lead] + [len(rest), new(dm.axis)]
        else:
            s_axis = ds.axis if ds.axis is not None else dr.axis
            phys = [d.axis for d in lead] + [s_axis, dm.axis]
    else:
        _refuse(st, "the state and row dims are paired in another form")
    if m_size == 1 and dm.axis is not None:
        # the reduce class: the mean over a stored presynaptic axis
        v = jnp.mean(v, axis=phys[-1], keepdims=True)
    return _arrange(v, phys, sizes)


def _dense_rows(st, i_size):
    # The logical lead + (n,) + (r, m) row with every dim spelled out, for
    # rows with no pair: the readout against a hidden weight and the exact
    # container of a dense row. ``i_size`` 1 keeps an implicit m unexpanded.
    from graphax.sparse.ops.dense import _expand_implicit_blocks
    st = _expand_implicit_blocks(st)
    if any(d.is_sparse for d in st.dims):
        return None
    dm = st.primal_dims[-1]
    v = _scaled_val(st)
    phys = [d.axis for d in st.dims]
    sizes = [d.logical_size for d in st.dims[:-1]] + [i_size]
    if i_size == 1 and dm.axis is not None:
        v = jnp.mean(v, axis=phys[-1], keepdims=True)
    return _arrange(v, phys, sizes)


def _project_sparse(st, weight, c: CarryContainer):
    m = st.primal_dims[-1].logical_size
    m_size = 1 if c.reduce else m
    if c.diag:
        if int(st.out_dims[-1].logical_size) != int(weight.shape[0]):
            _refuse(st, "the diag container's block diagonal pairs the state "
                        "axis with the weight's first axis, and they differ")
        J = _block_diagonal(st, m_size)
    else:
        J = _dense_rows(st, m_size)
        if J is None:
            # THE DENSE CONTAINER of a paired row: the store is dense by
            # definition, so this is the one place a pair is expanded.
            J = st.dense()
            if c.reduce:
                J = jnp.mean(J, axis=-1, keepdims=True)
    if c.quant:
        J = J.astype(RSNN_CARRY_QUANT_DTYPE)
    return J


def _project_dense(J, weight, c: CarryContainer):
    n_lead = J.ndim - 3
    if c.diag:
        if int(J.shape[n_lead]) != int(weight.shape[0]):
            raise ValueError(
                f"the diag container's block diagonal pairs the state axis "
                f"({J.shape[n_lead]}) with the weight's first axis "
                f"({weight.shape[0]}), and they differ")
        D = jnp.diagonal(J, axis1=n_lead, axis2=n_lead + 1)
        J = jnp.moveaxis(D, -1, -2)
    if c.reduce:
        J = jnp.mean(J, axis=-1, keepdims=True)
    if c.quant:
        J = J.astype(RSNN_CARRY_QUANT_DTYPE)
    return J


def _spike_trace(s_row, weight, c: CarryContainer, like):
    # The block diagonal of the plan's row of the new spikes against
    # ``weight``, before any narrowing: the hidden S block's own projection.
    if s_row is None:
        return jnp.zeros(jnp.shape(like), weight.dtype)
    plain = CarryContainer(True, c.reduce, False)
    if _is_sparse_tensor(s_row):
        return _project_sparse(s_row, weight, plain)
    return _project_dense(s_row, weight, plain)


def project_rsnn_carry(rows, container, weights, state_shapes=None, *,
                       given=None, a_out=None):
    # THE CONTAINER'S PROJECTION OF A PLAN'S STATE ROWS (owner ruling
    # 2026-09-24, Q28a), applied on the STORED class of a SparseTensor row
    # (the two-Diag plan stores (h, n_in) per hidden row) and on a dense row
    # as it is; a None row is a path the plan deleted.
    # THE READOUT TRACE AGAINST A HIDDEN WEIGHT in the diag container is the
    # leaky filter of the carried spike trace, f = a_out f + (1 - a_out) e_S,
    # read off the given f and the plan's S row: the readout row itself
    # (n_out, h, n_in) is never formed and never read.
    c = (container if isinstance(container, CarryContainer)
         else carry_container_from_name(container))
    out = []
    for k, (ss, w) in enumerate(RSNN_CARRY_STACKS):
        blocks = []
        for s in ss:
            if c.diag and len(ss) == 1 and w != 2:
                if given is None or a_out is None:
                    raise ValueError(
                        "the diag container's readout trace against a hidden "
                        "weight is the filter of the given trace and the "
                        "plan's S row; pass given= and a_out=")
                prev = given[k]
                eS = _spike_trace(rows[0][w], weights[w], c, prev)
                J = a_out * prev + (1.0 - a_out) * eS
                if c.quant:
                    J = J.astype(RSNN_CARRY_QUANT_DTYPE)
                blocks.append(J)
                continue
            J = rows[s][w]
            if J is None:
                if state_shapes is None:
                    raise ValueError(
                        f"the plan's program has no row for "
                        f"({RSNN_STATE_NAMES[s]}, {RSNN_WEIGHT_NAMES[w]}) and "
                        f"no state_shapes were given to size its zero")
                J = jnp.zeros(tuple(state_shapes[s]) + tuple(weights[w].shape))
            if _is_sparse_tensor(J):
                blocks.append(_project_sparse(J, weights[w], c))
            else:
                blocks.append(_project_dense(J, weights[w], c))
        if len(ss) == 1:
            out.append(blocks[0])
        else:
            out.append(jnp.stack(blocks,
                                 axis=blocks[0].ndim - (2 if c.diag else 3)))
    return tuple(out)


def rsnn_given_container(states, weights, given) -> CarryContainer:
    # The one container the five given stacks arrived in, read by shape and
    # dtype; a leading batch axis is the one the states carry.
    if len(given) != len(RSNN_CARRY_STACKS):
        raise ValueError(
            f"a carried-Jacobian tuple has one tensor per stack "
            f"({len(RSNN_CARRY_STACKS)}), got {len(given)}")
    found = set()
    for (ss, w), J in zip(RSNN_CARRY_STACKS, given):
        s = states[ss[0]]
        lead = len(jnp.shape(s)) - 1
        found.add(rsnn_carry_container(
            (ss[0], w), jnp.shape(s)[lead:], jnp.shape(weights[w]),
            jnp.shape(J)[lead:], J.dtype, n_stacked=len(ss)))
    if len(found) != 1:
        raise ValueError(
            f"the given stacks arrived in {sorted(c.name for c in found)}; a "
            f"plan's carry is one container")
    return found.pop()


def rsnn_zero_carry(container, weights, lead=(), dtype=jnp.float32):
    # The carry a recording starts from, in the container's shapes and dtype.
    c = (container if isinstance(container, CarryContainer)
         else carry_container_from_name(container))
    W, V, Wo = weights
    state_shape = {0: (W.shape[0],), 1: (W.shape[0],), 2: (W.shape[0],),
                   3: (W.shape[0],), 4: (Wo.shape[0],)}
    out = []
    for ss, w in RSNN_CARRY_STACKS:
        stack = (len(ss),) if len(ss) > 1 else ()
        shape = (tuple(lead) + stack
                 + c.block_shape(state_shape[ss[0]], weights[w].shape))
        out.append(jnp.zeros(shape, c.dtype(dtype)))
    return tuple(out)


def attach_rsnn_past(states, weights, given):
    """RTRL: give every carried state an edge to the weights, valued by ``J``.

    ``given`` is the five stacked tensors :data:`RSNN_CARRY_STACKS` names, in
    that order: the four hidden blocks against ``W`` as one ``(4, ..)`` tensor,
    the four against ``V`` likewise, then the three readout blocks. The delta
    each stack is contracted with is ``W - stop_gradient(W)``: exactly zero in
    value (no forward value moves, and a run under any rule sees the same loss
    to the last bit) and the identity as an edge to the weight. The
    ``stop_gradient`` is one edge-free vertex per weight; that price replaces
    the three reference weights the tuple used to lead with (owner ruling
    2026-09-24, Q31a: 434 kB of arguments per program).

    ONE CONTRACTION PER STACK. The stacked hidden contraction gives the
    ``(4, h)`` rows of the four states at once and is split back into the
    four states inside this scope; eleven contraction chains become five.

    EACH BLOCK ARRIVES IN ITS OWN CONTAINER (owner ruling 2026-09-16). The
    carry is produced by the rule the plan describes, run over the whole
    prefix, and it is stored in the container that rule implies. The container
    is a SET of the action classes the plan put on the carried-Jacobian face
    (:class:`CarryContainer`), and every combination is readable here:

    * no class -- the full ``state x weight`` block, ``d s / d W`` entry for
      entry. The exact carry, 225.74 MB over the eleven blocks.
    * ``diag`` -- the BLOCK DIAGONAL, at the shape of the weight. This is the
      eligibility trace: for a hidden block, ``J[j', j, i]`` is
      ``delta(j', j) * e[j, i]``, so one number per synapse instead of one per
      (unit, synapse) pair, and the contraction with ``dW`` collapses from a
      rank-2 tensordot to a row sum. For the two READOUT blocks against a
      hidden weight the readout is not a recurrence at all but a leaky filter
      of the hidden traces through a CONSTANT ``Wo``, so the exact block
      factorises as ``J[m, j, i] = Wo[m, j] * f[j, i]`` and the compact
      form carries ``f`` -- exact, not approximated, and 128 times smaller.
      The factor is restored from ``stop_gradient(Wo)``, so it adds no edge
      to ``Wo``.
    * ``reduce`` -- the presynaptic axis is IMPLICIT. One value is stored for
      the whole axis and the contraction NEVER broadcasts it back into the
      physical grid (CONTEXT.md, ruling D1): ``sum_i J[.., j, 0] * dW[j, i]``
      is ``J[.., j, 0] * sum_i dW[j, i]``, so the expansion becomes a sum on
      the OTHER operand and one axis of work disappears.
    * ``quant`` -- the block is stored narrow. The contraction promotes back
      to the weight's dtype, so the precision that was lost is the precision
      the recursion carried, which is the point.

    Mixed containers are legal: the plan decides per face, and one block may
    be diagonal while another is not.
    """
    if len(given) != len(RSNN_CARRY_STACKS):
        raise ValueError(
            f"a carried-Jacobian attachment needs one tensor per stack "
            f"({len(RSNN_CARRY_STACKS)}), got {len(given)}")
    out = list(states)
    with snn_carry_scope():
        held = [jax.lax.stop_gradient(W) for W in weights]
        deltas = [W - H for W, H in zip(weights, held)]
        for (ss, w), J in zip(RSNN_CARRY_STACKS, given):
            c = rsnn_carry_container(
                (ss[0], w), out[ss[0]].shape, weights[w].shape, J.shape,
                J.dtype, n_stacked=len(ss))
            dW = deltas[w]
            if c.reduce:
                # THE IMPLICIT AXIS IS NEVER BROADCAST. Move the sum over the
                # presynaptic index onto ``dW``, which stores it, and drop the
                # stored axis of extent 1 from ``J``.
                dW = jnp.sum(dW, axis=-1)
                J = J[..., 0]
            if not c.diag:
                rows = jnp.tensordot(J, dW, dW.ndim)
                if len(ss) == 1:
                    out[ss[0]] = out[ss[0]] + rows
                    continue
                for k, s in enumerate(ss):
                    out[s] = out[s] + jax.lax.index_in_dim(rows, k, 0, False)
                continue
            # DIAG. ``row[j] = sum_i J[.., j, i] * dW[j, i]``: the block
            # diagonal's contraction, one row sum per block. Under ``reduce``
            # the presynaptic axis is already gone from both operands and
            # the row sum degenerates to an elementwise product. Each block
            # of a stack is read from its own slot, so the elimination sees
            # one (h, n_in) trace per state and never a slice of a stacked
            # row.
            if len(ss) == 1:
                rows = J * dW if c.reduce else jnp.sum(J * dW, axis=-1)
                if out[ss[0]].shape[0] != weights[w].shape[0]:
                    # The readout against a hidden weight: restore the
                    # constant ``Wo`` factor the compact form left out.
                    rows = held[2] @ rows
                out[ss[0]] = out[ss[0]] + rows
                continue
            for k, s in enumerate(ss):
                Jk = J[k]
                out[s] = out[s] + (Jk * dW if c.reduce
                                   else jnp.sum(Jk * dW, axis=-1))
    return tuple(out)


def attach_rsnn_future(loss, next_states, given):
    """BPTT: give ``s_t`` an edge to the future loss, valued by the adjoint.

    ``given`` is the five adjoints ``lambda_(t+1) = dL_(>t)/ds_t``, one per
    component of :data:`RSNN_STATE_NAMES`, from a detached backward pass over
    the suffix of the recording. The returned scalar is
    ``L_t + <lambda_(t+1), s_t>``, whose gradient with respect to the weights
    is EXACTLY the per-step contribution full backpropagation through time
    makes at step ``t``.

    THE CONTAINER FOLLOWS THE PLAN HERE TOO (owner ruling 2026-09-16), and an
    adjoint has only the two classes that a vector can express:

    * ``reduce`` -- the state axis is IMPLICIT, so the adjoint arrives as one
      number of extent 1 and the inner product becomes that number times the
      SUM of the state. Nothing is broadcast back into the grid.
    * ``quant`` -- the adjoint is stored narrow.

    ``diag`` does not change the STORE of an adjoint, only its VALUE: on the
    suffix it means the state-to-state Jacobian is replaced by its block
    diagonal at every step, which is what the producer does before it hands
    the five numbers over. ``Skip`` means no adjoint at all, and then the rule
    is truncated backpropagation through time and ``given`` is empty.
    """
    if len(given) != len(RSNN_STATE_NAMES):
        raise ValueError(
            f"a future-adjoint attachment needs one adjoint per state "
            f"component ({len(RSNN_STATE_NAMES)}), got {len(given)}")
    with snn_carry_scope():
        for lam, s in zip(given, next_states):
            if tuple(lam.shape) == tuple(s.shape):
                loss = loss + jnp.sum(lam * s)
            elif tuple(lam.shape) == (1,):
                # The implicit state axis: one stored value, and the sum that
                # would have expanded it moves onto the state instead.
                loss = loss + lam[0] * jnp.sum(s)
            else:
                raise ValueError(
                    f"adjoint shape {tuple(lam.shape)} is neither the state "
                    f"it multiplies, {tuple(s.shape)}, nor the reduced "
                    f"container (1,) that an implicit state axis stores.")
    return loss


#: How many varargs each temporal rule passes. ``bptt`` passes one adjoint per
#: state component and ``rtrl`` one tensor per carried stack, five each, so
#: the count alone no longer tells them apart: :func:`rsnn_given_rule` reads
#: the RANK behind the count. An adjoint is a vector (or the ``(1,)`` of an
#: implicit state axis); a carried block is at least rank 2 in every
#: container. A count outside this table raises rather than choosing a rule
#: by accident.
RSNN_GIVEN_COUNTS: dict[str, int] = {
    "tbptt": 0,
    "bptt": len(RSNN_STATE_NAMES),
    "rtrl": len(RSNN_CARRY_STACKS),
}


def rsnn_given_rule(given) -> str:
    """The temporal rule ``given`` selects, or raise."""
    n = len(given)
    if n == 0:
        return "tbptt"
    if n != len(RSNN_STATE_NAMES):
        raise ValueError(
            f"RSNN_SHD got {n} extra arguments; the temporal rule is selected "
            f"by that count and the legal counts are "
            f"{sorted(set(RSNN_GIVEN_COUNTS.values()))} "
            f"({', '.join(f'{k}: {v}' for k, v in RSNN_GIVEN_COUNTS.items())})")
    ranks = {int(jnp.ndim(g)) for g in given}
    if ranks == {1}:
        return "bptt"
    if min(ranks) >= 2:
        return "rtrl"
    raise ValueError(
        f"RSNN_SHD got {n} extra arguments of ranks {sorted(ranks)}; the five "
        f"future adjoints are all vectors and the five carried stacks are all "
        f"of rank 2 or more, and this tuple is neither.")


def RSNN_SHD(x, y, S, I, U, a, Uo, W, V, Wo,
             a_syn, a_mem, a_out, rho, beta_a, thresh, *given):
    """ONE recurrent step of the SHD network, plus its step loss.

    Weights are args 7, 8 and 9 (``W``, ``V``, ``Wo``); ``V`` is among them,
    so the recurrent coupling is learned and ``d s_t / d s_(t-1)`` has real
    off-diagonal terms.

    ``given`` selects the temporal rule by its LENGTH and, at five, by the
    rank of what it holds (:func:`rsnn_given_rule`):

      0 entries   tbptt. The carried state is a constant.
      5 vectors   bptt. The five future adjoints; see
                  :func:`attach_rsnn_future`.
      5 tensors   rtrl. The eleven carried Jacobian blocks, stacked per
                  weight; see :func:`attach_rsnn_past`.

    The loss is the softmax cross entropy of the leaky readout membrane
    against the one-hot label ``y``. It is a PER-STEP loss on purpose: the
    whole point of this target is that the sequence loss decomposes as
    ``sum_t L_t``, so one step is a complete object and the three rules
    differ only in which given quantity meets it.

    Under ``rtrl`` the return is ``(loss, S, I, U, a, Uo)``: the first output
    is the scalar loss, the rest is the carried state, whose Jacobian rows
    with respect to the weights ARE the next carry ``J_t = A_t J_(t-1) + F_t``
    (owner ruling 2026-09-24, Q27b). ``tbptt`` and ``bptt`` return the loss.
    """
    rule = rsnn_given_rule(given)
    weights = (W, V, Wo)
    if rule == "rtrl":
        S, I, U, a, Uo = attach_rsnn_past((S, I, U, a, Uo), weights, given)
    with snn_step_scope(0):
        nxt = rsnn_cell(x, S, I, U, a, Uo, W, V, Wo,
                        a_syn, a_mem, a_out, rho, beta_a, thresh)
        loss = jnp.sum(-y * jax.nn.log_softmax(nxt[4]))
    if rule == "bptt":
        loss = attach_rsnn_future(loss, nxt, given)
    if rule == "rtrl":
        # THE PLAN PRODUCES ITS OWN CARRY (owner ruling 2026-09-24, Q27b):
        # the Jacobian rows of s_t are the next carried value.
        return (loss,) + tuple(nxt)
    return loss


#: The number of step copies in the two-copy window arm.
RSNN_W2_COPIES = 2


def RSNN_SHD_W2(x0, x1, y, S, I, U, a, Uo, W, V, Wo,
                a_syn, a_mem, a_out, rho, beta_a, thresh):
    """TWO recurrent steps of the SHD network, joined by the temporal edge.

    THE FOURTH SNN ARM (owner ruling 2026-09-16). There is no given edge and
    no temporal rule here. The window holds two copies of the step body, the
    state carried from the first copy to the second IS the temporal edge, and
    it is an ordinary edge of the graph -- so the policy picks the direction
    of the temporal credit itself by choosing where in the elimination order
    the second copy's vertices go. Eliminating the later copy first is
    backpropagation through time; eliminating the earlier copy first is real
    time recurrent learning; and the policy may also interleave them, which is
    neither and is the reason this arm exists.

    ``x0`` and ``x1`` are the input frames of steps ``t`` and ``t+1`` and the
    loss is ``L_t + L_(t+1)``, so the temporal edge carries real credit. The
    label is one per recording, as it is everywhere else in this family.

    The weights sit at slots 8, 9 and 10 (``W``, ``V``, ``Wo``); the
    ``infer_argnums`` table names them. Each copy is wrapped in its own
    :func:`snn_step_scope`, so ``--fixed-temporal-order`` can pin the order
    across the copies -- but this arm runs it FREE, which is the point.
    """
    state = (S, I, U, a, Uo)
    loss = 0.0
    for u, x in enumerate((x0, x1)):
        with snn_step_scope(u):
            state = rsnn_cell(x, *state, W, V, Wo,
                              a_syn, a_mem, a_out, rho, beta_a, thresh)
            loss = loss + jnp.sum(-y * jax.nn.log_softmax(state[4]))
    return loss
