# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM post-sampling commit: eager vs staged vs CUDA-graph replay.

The graph path must be bit-identical to the eager chain it captures. Every GPU
scenario runs two worlds built from the same seed: a reference world committed
by the eager chain and a candidate world committed in staged or graph mode,
through ``RecoverSSMState.commit_step`` (all groups' plan, conv compaction,
final and boundary folds AND the real align-postprocess kernel). After every
step it compares bitwise: every layer's fp16 checkpoint and conv state, every
context's plan buffers, the running-state column and the accepted counts.

Inputs follow the engine: per-group persistent builder buffers (spec state
indices, query offsets, block table) rewritten in place each step, fresh
num_sampled / idx_mapping / num_computed_tokens tensors each step, distinct
block tables per group, crossing / exact-boundary / non-crossing rows, accepted
lengths 0..K+1, draft-less (T=1) rows, null-state rows.

Graphs are captured only by the warmup (record_step with for_capture=True, as
the engine's FULL forward captures call it); serving never captures. The warmup
must leave persistent state untouched, and an invalidated capture must raise and
leave the process usable.

Negative controls run a faulty candidate through the same harness and assert
the comparison FAILS: a graph whose staged num_computed is never refreshed, a
chain without the align postprocess, swapped group mappings, and a chain without
the boundary fold.

CPU tests (no GPU) cover mode parsing, env registration outside the compile
factors, the worker RPC names, the frozen commit context, the eligibility rules
and the stage/replay control flow with stand-in graphs."""

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.gdn_recoverssm import (
    GDNRecoverSSMCommitMetadata,
    GDNRecoverSSMMetadata,
)
from vllm.v1.worker.gpu.model_states import recoverssm_commit_graph as rcg

import vllm as _vllm  # noqa: E402

# The patched source root (this file ships outside it, in the release repo).
REPO = pathlib.Path(_vllm.__file__).resolve().parents[1]
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _restore_mode():
    saved = rcg._mode
    yield
    rcg._mode = saved


# --------------------------------------------------------------------------- CPU


def test_parse_commit_mode():
    for v, want in [
        (None, "eager"),
        ("", "eager"),
        ("0", "eager"),
        (" Eager ", "eager"),
        ("staged", "staged"),
        ("1", "graph"),
        ("GRAPH", "graph"),
    ]:
        assert rcg.parse_commit_mode(v) == want
    for bad in ("2", "true", "cuda_graph", "on"):
        with pytest.raises(ValueError, match="unknown RecoverSSM commit mode"):
            rcg.parse_commit_mode(bad)


def test_env_default_and_runtime_switch(monkeypatch):
    monkeypatch.setenv(rcg.ENV_NAME, "graph")
    rcg._mode = None
    assert rcg.get_commit_mode() == "graph"
    assert rcg.set_commit_mode("0") == "eager"
    assert rcg.get_commit_mode() == "eager"
    # The env is only the default: a later env change does not override the switch.
    monkeypatch.setenv(rcg.ENV_NAME, "staged")
    assert rcg.get_commit_mode() == "eager"
    with pytest.raises(ValueError):
        rcg.set_commit_mode("bogus")
    assert rcg.get_commit_mode() == "eager"


def test_env_registered_but_not_a_compile_factor(monkeypatch):
    import vllm.envs as envs

    assert rcg.ENV_NAME in envs.environment_variables
    monkeypatch.delenv(rcg.ENV_NAME, raising=False)
    base = envs.compile_factors()
    assert rcg.ENV_NAME not in base
    monkeypatch.setenv(rcg.ENV_NAME, "graph")
    assert envs.compile_factors() == base


def test_worker_exposes_the_switch_rpc():
    src = (REPO / "vllm/v1/worker/gpu_worker.py").read_text(encoding="utf-8")
    worker = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.ClassDef) and n.name == "Worker"
    )
    names = {n.name for n in worker.body if isinstance(n, ast.FunctionDef)}
    assert {"set_recoverssm_commit_mode", "get_recoverssm_commit_stats"} <= names


def _cpu_meta(n, *, nc=None, bt=None, si=None, qsl=None, ctx=None, req=None):
    m = GDNRecoverSSMMetadata.__new__(GDNRecoverSSMMetadata)
    m.num_spec_decodes = n
    m.recoverssm_commit = GDNRecoverSSMCommitMetadata(
        state_indices=si
        if si is not None
        else torch.zeros(8, 5, dtype=torch.int32)[:, :1],
        query_start_loc=qsl if qsl is not None else torch.zeros(9, dtype=torch.int32),
        request_indices=req,
        block_table=bt,
        num_computed_tokens=nc,
        block_size=8 if bt is not None else None,
    )
    m.recoverssm_context = ctx or SimpleNamespace(commit_lens=torch.zeros(8))
    return m


def test_eligibility_rules_cpu():
    g = rcg.RecoverSSMCommitGraphs(lambda *a, **k: None)
    ns = torch.ones(4, dtype=torch.int32)
    im = torch.arange(4)
    sidx = torch.zeros(8, dtype=torch.int32)
    acc = torch.ones(8, dtype=torch.int32)
    bt = torch.zeros(8, 6, dtype=torch.int32)

    def plan(step, ns=ns, im=im, sidx=sidx):
        return g._plan(step, ns, im, sidx, acc)

    nc = torch.zeros(4, dtype=torch.int32)
    ctx = SimpleNamespace(commit_lens=torch.zeros(8))
    # Persistent per-group builder buffers (spec state indices, query offsets).
    si = [torch.zeros(8, 5, dtype=torch.int32)[:, :1] for _ in range(3)]
    qsl = [torch.zeros(9, dtype=torch.int32) for _ in range(3)]

    def metas(nc_of, bt=bt):
        return [
            _cpu_meta(4, nc=nc_of(), bt=bt, ctx=ctx, si=si[g], qsl=qsl[g])
            for g in range(3)
        ]

    ok = metas(lambda: nc)
    assert plan(ok) == "device"  # CPU tensors never replay
    g._require_cuda = False
    p = plan(ok)
    assert isinstance(p, rcg._Plan) and p.n == 4
    assert p.nc_slots == (0, 0, 0)  # one shared num_computed tensor: one copy
    # Distinct (fresh) num_computed per group: three slots, same key shape.
    p3 = plan(metas(nc.clone))
    assert p3.nc_slots == (0, 1, 2)
    # Fresh num_computed tensors are staged, so they do not enter the key ...
    assert plan(metas(nc.clone)).key == p3.key
    # ... but a moved pass-through tensor (block table, state indices) does.
    assert plan(metas(nc.clone, bt=bt.clone())).key != p3.key
    moved = metas(nc.clone)
    moved[1].recoverssm_commit.state_indices = si[1].clone()
    assert plan(moved).key != p3.key
    # Ineligible shapes of step.
    assert plan([SimpleNamespace(recoverssm_commit=None)]) == "metadata_type"
    assert plan([_cpu_meta(3, nc=nc, bt=bt, ctx=ctx)]) == "not_uniform_decode"
    assert (
        plan([_cpu_meta(4, nc=nc, bt=bt, ctx=ctx, req=torch.arange(4))])
        == "request_indices"
    )
    nocommit = _cpu_meta(4, nc=nc, bt=bt, ctx=ctx)
    nocommit.recoverssm_commit = None
    assert plan([nocommit]) == "no_commit"
    assert plan([]) == "empty_step"
    assert plan(ok, sidx=None) == "state_indices"
    big = SimpleNamespace(commit_lens=torch.zeros(2))
    assert plan([_cpu_meta(4, nc=nc, bt=bt, ctx=big)]) == "capacity"
    assert plan(ok, ns=torch.ones(2, dtype=torch.int32)) == "num_sampled"


def test_eager_mode_never_plans():
    calls = []
    g = rcg.RecoverSSMCommitGraphs(lambda *a, **k: calls.append(1))
    rcg.set_commit_mode("eager")
    assert not g.run(
        [object()],
        torch.ones(1, dtype=torch.int32),
        torch.arange(1),
        state_indices=None,
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
    )
    assert calls == []


def test_eager_boot_captures_nothing():
    rcg._mode = "eager"
    g = rcg.RecoverSSMCommitGraphs(lambda *a, **k: None)
    assert not g.capture_at_warmup
    assert not g.capture_warmup(
        [object()], state_indices=None, num_accepted_tokens=torch.ones(1)
    )


def test_context_is_frozen():
    import dataclasses

    from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
        GDNRecoverSSMCommitContext,
    )

    nb, hv, v, k, t = 4, 2, 8, 4, 5
    ctx = GDNRecoverSSMCommitContext.from_tensors(
        [torch.zeros(nb, 6, 3 + t - 1)],
        [torch.zeros(nb, hv, v, k)],
        [torch.zeros(nb, hv, t, v + k + 1)],
        spec_query_len=t,
        max_num_reqs=4,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.final_state_indices = torch.zeros(4, dtype=torch.int32)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.conv_history_len = 7  # type: ignore[misc]
    assert len(rcg.context_bound_tensors(ctx)) == 15


class _FakeGraph:
    """CPU stand-in for a captured graph: replay re-runs the launch with the
    capture-time arguments, i.e. it reads the same tensors (addresses) the
    capture saw, whatever the current step passed in."""

    def __init__(self, fn):
        self.replay = fn


def _fake_ctx(cap):
    def z(*s):
        return torch.zeros(*s, dtype=torch.int64)

    return SimpleNamespace(
        conv_states=(torch.zeros(4, 6, 7),),
        conv_state_base_addrs=z(1),
        conv_state_block_strides=z(1),
        conv_state_dim_strides=z(1),
        conv_state_token_strides=z(1),
        checkpoints=(torch.zeros(4, 2, 8, 4),),
        state_base_addrs=z(1),
        state_block_strides=z(1),
        replays=(torch.zeros(4, 2, 5, 13),),
        replay_base_addrs=z(1),
        replay_block_strides=z(1),
        commit_lens=torch.zeros(cap, dtype=torch.int32),
        final_state_indices=torch.zeros(cap, dtype=torch.int32),
        boundary_state_indices=torch.zeros(cap, dtype=torch.int32),
        boundary_recovery_lens=torch.zeros(cap, dtype=torch.int32),
        conv_history_len=3,
        spec_query_len=5,
        out=torch.zeros(cap, dtype=torch.int64),
    )


def _cpu_world(n_groups=3, cap=8):
    groups = [
        SimpleNamespace(
            ctx=_fake_ctx(cap),
            si=torch.zeros(cap, 5, dtype=torch.int32),
            qsl=torch.zeros(cap + 1, dtype=torch.int32),
            bt=torch.zeros(cap, 6, dtype=torch.int32),
        )
        for _ in range(n_groups)
    ]
    return SimpleNamespace(groups=groups, sidx=torch.zeros(cap, dtype=torch.int32))


def _cpu_launch(
    step, ns, im, *, state_indices, num_accepted_tokens, num_computed_tokens=None
):
    for i, m in enumerate(step):
        c, n = m.recoverssm_commit, m.num_spec_decodes
        nc = c.num_computed_tokens
        if num_computed_tokens is not None and num_computed_tokens[i] is not None:
            nc = num_computed_tokens[i]
        si = c.state_indices[:n, 0].long()
        m.recoverssm_context.out[:n] = si * 1000 + ns[:n] * 100 + nc[:n] * 10 + im[:n]
        ok = im[:n] >= 0
        state_indices[im[:n][ok]] = (nc[:n] + ns[:n] + si)[ok].to(state_indices.dtype)


def _cpu_metas(world, n, si_vals, nc, fresh_si=False):
    metas = []
    for gr in world.groups:
        si = torch.zeros_like(gr.si) if fresh_si else gr.si
        si[:n, 0] = si_vals
        metas.append(
            _cpu_meta(n, nc=nc.clone(), bt=gr.bt, si=si[:, :1], qsl=gr.qsl, ctx=gr.ctx)
        )
    return metas


def _cpu_ab(ns_seq, modes, *, warm=(), patch=None, fresh_si=False, mutate=None):
    """Eager chain vs RecoverSSMCommitGraphs.run (stand-in graphs captured at
    'warmup') on two CPU worlds with the same inputs; first differing step."""
    rcg._mode = "graph"  # boot mode: warmup captures
    g = rcg.RecoverSSMCommitGraphs(_cpu_launch)
    g._require_cuda = False
    g._capture_graph = lambda dev, step, ns, im, args: _FakeGraph(
        lambda: _cpu_launch(step, ns, im, **args)
    )
    if patch:
        patch(g)
    worlds = [_cpu_world(), _cpu_world()]
    acc = torch.ones(8, dtype=torch.int32)
    for n in warm:
        zeros = torch.zeros(n, dtype=torch.int32)
        assert g.capture_warmup(
            _cpu_metas(worlds[1], n, zeros, zeros),
            state_indices=worlds[1].sidx,
            num_accepted_tokens=acc,
        )
    # The warm-up launch (0 accepted, idx -1) must not touch persistent state.
    assert torch.equal(worlds[0].sidx, worlds[1].sidx)
    if mutate:
        mutate(worlds[1])
    for i, (n, mode) in enumerate(zip(ns_seq, modes, strict=True)):
        outs = []
        for w, wmode in zip(worlds, ("eager", mode)):
            rng = torch.Generator().manual_seed(100 + i)
            si_vals = torch.randint(1, 99, (n,), generator=rng, dtype=torch.int32)
            nc = torch.randint(0, 50, (n,), generator=rng, dtype=torch.int32)
            metas = _cpu_metas(w, n, si_vals, nc, fresh_si)
            ns = torch.randint(0, 6, (n,), generator=rng, dtype=torch.int32)
            im = torch.randperm(8, generator=rng)[:n]
            rcg.set_commit_mode(wmode)
            handled = w is worlds[1] and g.run(
                metas, ns, im, state_indices=w.sidx, num_accepted_tokens=acc
            )
            if not handled:
                _cpu_launch(
                    metas, ns, im, state_indices=w.sidx, num_accepted_tokens=acc
                )
            outs.append([gr.ctx.out[:n].clone() for gr in w.groups] + [w.sidx.clone()])
        if any(not torch.equal(a, b) for a, b in zip(*outs)):
            return i, g
    return None, g


def _delta(before, key):
    after = rcg.commit_stats()
    if key in ("replays", "captures", "staged_steps", "eager_steps"):
        return after[key] - before[key]
    return after["fallbacks"].get(key, 0) - before["fallbacks"].get(key, 0)


def test_graph_control_flow_cpu():
    before = rcg.commit_stats()
    ns = [4, 4, 2, 3, 4, 2, 1, 4]
    modes = ["graph", "graph", "staged", "graph", "eager", "graph", "graph", "graph"]
    bad, g = _cpu_ab(ns, modes, warm=[4, 2])
    assert bad is None
    # Replays at 0, 1, 5, 7; 2 is staged, 4 eager, 3 and 6 have no graph.
    assert _delta(before, "replays") == 4
    assert _delta(before, "captures") == 2
    assert _delta(before, "not_captured") == 2
    assert g.num_graphs == 2
    g.reset()
    assert g.num_graphs == 0


def test_graph_control_flow_cpu_moved_builder_buffers_never_replay():
    before = rcg.commit_stats()
    bad, _ = _cpu_ab([4] * 6, ["graph"] * 6, warm=[4], fresh_si=True)
    assert bad is None
    assert _delta(before, "replays") == 0


def test_graph_control_flow_cpu_rebound_context_tensor_refuses_replay():
    before = rcg.commit_stats()

    def rebind(world):  # the stand-in context is not frozen: rebind directly
        ctx = world.groups[1].ctx
        ctx.final_state_indices = ctx.final_state_indices.clone()

    bad, g = _cpu_ab([4] * 3, ["graph"] * 3, warm=[4], mutate=rebind)
    assert bad is None
    assert _delta(before, "replays") == 0
    assert _delta(before, "binding_moved") == 1
    assert g.num_graphs == 0


def test_graph_control_flow_cpu_negative_stale_staging():
    def patch(g):
        real = g._stage

        def stale(plan, ns, im):
            keys = [k for k in g._staging if k[0].startswith("num_computed")]
            keep = [g._staging[k].clone() for k in keys] if g._graphs else []
            out = real(plan, ns, im)
            for k, v in zip(keys, keep):
                g._staging[k].copy_(v)
            return out

        g._stage = stale

    bad, _ = _cpu_ab([4] * 5, ["graph"] * 5, warm=[4], patch=patch)
    assert bad == 0  # the first replay


# --------------------------------------------------------------------------- GPU

T = 5  # spec_query_len: K=4 drafts
HV, V, K = 2, 32, 16
CONV_DIM, HIST = 64, 3
NB, BS, WIDTH = 200, 8, 12
GROUPS, LAYERS = 3, 16
MAX_REQS = 16
NULL = 0

# (num_computed % BS, accepted, query_len): non-crossing, exact boundary,
# crossing, accepted 0, accepted > window (clamped), draft-less windows.
ROW_CASES = [
    (0, 3, T),
    (5, 3, T),  # ends exactly on the boundary
    (6, 4, T),  # crosses after 2
    (7, 5, T),  # crosses after 1, full window
    (2, 0, T),  # nothing accepted
    (4, 5, T),  # ends exactly on the boundary, full window
    (7, 1, 1),  # draft-less, ends exactly on the boundary
    (3, 3, 1),  # draft-less, accepted clamped to the window
    (1, 2, T),
]


class World:
    """One rank's RecoverSSM state for GROUPS groups of LAYERS GDN layers.
    ``boot_mode`` is the commit mode the process boots with (it decides whether
    the warmup captures commit graphs)."""

    def __init__(self, seed: int, *, align: bool = True, boot_mode: str = "eager"):
        from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
            GDNRecoverSSMCommitContext,
        )
        from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

        g = torch.Generator(device="cuda").manual_seed(seed)
        cpu = torch.Generator().manual_seed(seed)
        self.align = align
        self.groups = []
        for _ in range(GROUPS):
            ck = [
                (torch.randn(NB, HV, V, K, generator=g, device="cuda") * 0.1).half()
                for _ in range(LAYERS)
            ]
            cv = [
                torch.randn(
                    NB, CONV_DIM, HIST + T - 1, generator=g, device="cuda"
                ).bfloat16()
                for _ in range(LAYERS)
            ]
            rp = [
                torch.randn(NB, HV, T, V + K + 1, generator=g, device="cuda") * 0.5
                for _ in range(LAYERS)
            ]
            for r in rp:  # decay in (0, 1], like the verify's record
                r[..., V + K].uniform_(0.5, 1.0, generator=g)
            ctx = GDNRecoverSSMCommitContext.from_tensors(
                cv, ck, rp, spec_query_len=T, max_num_reqs=MAX_REQS
            )
            # Distinct block tables per group, disjoint rows, ids > NULL.
            perm = torch.randperm(NB - 1, generator=cpu)[: MAX_REQS * WIDTH] + 1
            bt = perm.reshape(MAX_REQS, WIDTH).to(torch.int32).cuda()
            si = torch.zeros(MAX_REQS, T, dtype=torch.int32, device="cuda")
            qsl = torch.zeros(MAX_REQS + 1, dtype=torch.int32, device="cuda")
            self.groups.append(
                SimpleNamespace(ck=ck, cv=cv, rp=rp, ctx=ctx, bt=bt, si=si, qsl=qsl)
            )
        self.state_idx = torch.zeros(MAX_REQS, dtype=torch.int32, device="cuda")
        self.num_acc = torch.ones(MAX_REQS, dtype=torch.int32, device="cuda")
        rcg._mode = boot_mode
        self.rs = RecoverSSMState()

    def snapshot(self, n: int):
        """Persistent state, plus the first n entries of every plan buffer (the
        scratch the last commit wrote)."""
        out = []
        for gr in self.groups:
            out += [t.clone() for t in gr.ck + gr.cv]
            out += [
                gr.ctx.commit_lens[:n].clone(),
                gr.ctx.final_state_indices[:n].clone(),
                gr.ctx.boundary_state_indices[:n].clone(),
                gr.ctx.boundary_recovery_lens[:n].clone(),
            ]
        return out + [self.state_idx.clone(), self.num_acc.clone()]

    def _metas(self, n, padded, *, src, qsl_vals, nc, req=None, fresh=False):
        metas = {}
        for gi, gr in enumerate(self.groups):
            si, qsl = gr.si, gr.qsl
            if fresh:  # PIECEWISE/eager forward: no staging
                si = torch.zeros_like(si)
                qsl = torch.zeros_like(qsl)
            si[:n, 0] = src(gr)
            si[:n, 1:] = 7  # columns the commit must ignore
            si[n:padded] = NULL
            qsl[: n + 1] = qsl_vals
            qsl[n + 1 : padded + 1] = qsl_vals[-1]
            m = GDNRecoverSSMMetadata.__new__(GDNRecoverSSMMetadata)
            m.num_spec_decodes = n
            m.recoverssm_commit = GDNRecoverSSMCommitMetadata(
                state_indices=si[:padded, :1],
                query_start_loc=qsl[: padded + 1],
                request_indices=req,
                block_table=gr.bt[:padded] if self.align else None,
                # compute_num_computed_tokens(): a fresh tensor per group per step
                num_computed_tokens=nc.clone() if self.align else None,
                block_size=BS if self.align else None,
            )
            m.recoverssm_context = gr.ctx
            metas[f"group{gi}"] = m
        groups = [[SimpleNamespace(layer_names=[name])] for name in metas]
        return metas, groups

    def warmup(self, ns):
        """What the FULL forward captures hand record_step: dummy uniform
        decode metadata (null state rows, full windows, zero computed)."""
        for n in ns:
            zeros = torch.zeros(n, dtype=torch.int32, device="cuda")
            metas, groups = self._metas(
                n,
                n,
                src=lambda gr, z=zeros: z,
                qsl_vals=torch.arange(n + 1, dtype=torch.int32, device="cuda") * T,
                nc=zeros,
            )
            self.rs.record_step(
                metas,
                groups,
                for_capture=True,
                state_indices=self.state_idx if self.align else None,
                num_accepted_tokens=self.num_acc,
            )

    def commit(self, inp, mode, *, fresh_builder_buffers=False):
        """Stage one step's metadata like the builder and run commit_step."""
        n, padded = inp["n"], inp["padded"]
        rows = torch.arange(n, device="cuda")

        def src(gr):
            s = gr.bt[rows, inp["nc"][:n].long() // BS]
            return torch.where(inp["null_row"][:n], torch.zeros_like(s), s)

        metas, groups = self._metas(
            n,
            padded,
            src=src,
            qsl_vals=inp["qsl"],
            nc=inp["nc"],
            req=inp.get("request_indices"),
            fresh=fresh_builder_buffers,
        )
        self.rs.record_step(metas, groups, for_capture=False)
        rcg.set_commit_mode(mode)
        self.rs.commit_step(
            inp["num_sampled"].clone(),  # the sampler's new_empty
            inp["idx_mapping"].clone(),  # async_tensor_h2d
            state_indices=self.state_idx if self.align else None,
            num_accepted_tokens=self.num_acc,
        )


def make_step(rng: torch.Generator, n: int, *, num_reqs=None, padded=None):
    num_reqs = n if num_reqs is None else num_reqs
    shift = int(torch.randint(len(ROW_CASES), (1,), generator=rng))
    nc, acc, ql, null_row = [], [], [], []
    for r in range(num_reqs):
        off, a, q = ROW_CASES[(r + shift) % len(ROW_CASES)]
        blk = int(torch.randint(0, WIDTH - 2, (1,), generator=rng))
        nc.append(blk * BS + off)
        acc.append(a)
        ql.append(q)
        null_row.append(r == 3 and n > 4)  # a null-state row inside the batch
    qsl = [0]
    for q in ql[:n]:
        qsl.append(qsl[-1] + q)
    slots = torch.randperm(MAX_REQS, generator=rng)[:num_reqs]
    return dict(
        n=n,
        padded=padded or n,
        nc=torch.tensor(nc, dtype=torch.int32, device="cuda"),
        num_sampled=torch.tensor(acc, dtype=torch.int32, device="cuda"),
        idx_mapping=slots.to(torch.int64).cuda(),
        qsl=torch.tensor(qsl, dtype=torch.int32, device="cuda"),
        null_row=torch.tensor(null_row, device="cuda"),
    )


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if not torch.equal(x.view(torch.uint8), y.view(torch.uint8)):
            return i
    return None


def run_ab(
    steps, cand_modes, *, warm=(), seed=0, align=True, cand=None, boot="graph", **kw
):
    """Reference world (eager boot, eager commits) vs a candidate world booted in
    ``boot`` mode and warmed up for the request counts ``warm``; returns the
    first mismatching step (None if every step matched bitwise) and the
    candidate. The warmup itself must leave persistent state untouched."""
    ref = World(seed, align=align, boot_mode="eager")
    cand = cand or World(seed, align=align, boot_mode=boot)
    cand.warmup(warm)
    torch.cuda.synchronize()
    assert first_mismatch(ref.snapshot(0), cand.snapshot(0)) is None
    for i, (inp, mode) in enumerate(zip(steps, cand_modes, strict=True)):
        ref.commit(inp, "eager")
        cand.commit(inp, mode, **kw)
        torch.cuda.synchronize()
        n = inp["n"]
        if first_mismatch(ref.snapshot(n), cand.snapshot(n)) is not None:
            return i, cand
    return None, cand


def _steps(ns, seed=1, **kw):
    rng = torch.Generator().manual_seed(seed)
    return [make_step(rng, n, **kw) for n in ns]


@needs_cuda
@pytest.mark.parametrize("n", [1, 2, 4, 5, 8])
def test_graph_matches_eager_every_step(n):
    before = rcg.commit_stats()
    bad, cand = run_ab(_steps([n] * 8), ["graph"] * 8, warm=[n])
    assert bad is None
    assert cand.rs._commit_graphs.num_graphs == 1
    assert _delta(before, "captures") == 1
    assert _delta(before, "replays") == 8


@needs_cuda
def test_warmup_twice_per_size_captures_once():
    # Every FULL capture builds its metadata twice (warmup + capture forward).
    before = rcg.commit_stats()
    bad, _ = run_ab(_steps([2, 4]), ["graph"] * 2, warm=[4, 4, 2, 2, 1])
    assert bad is None
    assert _delta(before, "captures") == 3
    assert _delta(before, "replays") == 2


@needs_cuda
def test_graph_matches_eager_with_padded_builder_buffers():
    # Graph-padded batch: the builder buffers carry NULL rows beyond n.
    before = rcg.commit_stats()
    steps = [dict(s, padded=8) for s in _steps([5] * 6)]
    bad, _ = run_ab(steps, ["graph"] * 6, warm=[5])
    assert bad is None
    assert _delta(before, "replays") == 6


@needs_cuda
def test_staged_matches_eager_and_never_replays():
    before = rcg.commit_stats()
    bad, _ = run_ab(_steps([4] * 5), ["staged"] * 5, warm=[4], boot="staged")
    assert bad is None
    assert _delta(before, "replays") == 0
    assert _delta(before, "staged_steps") == 5


@needs_cuda
def test_mode_flips_mid_stream():
    modes = ["graph", "graph", "eager", "graph", "staged", "graph", "eager"]
    modes += ["eager", "graph", "staged", "graph"]
    before = rcg.commit_stats()
    bad, cand = run_ab(_steps([4] * len(modes)), modes, warm=[4])
    assert bad is None
    assert _delta(before, "replays") == modes.count("graph")
    assert cand.rs._commit_graphs.num_graphs == 1


@needs_cuda
def test_batch_sizes_captured_and_not():
    ns = [1, 2, 3, 4, 6, 8, 3, 1, 4, 8, 6, 2]
    before = rcg.commit_stats()
    bad, cand = run_ab(_steps(ns), ["graph"] * len(ns), warm=[1, 2, 4, 8])
    assert bad is None
    assert cand.rs._commit_graphs.num_graphs == 4
    assert _delta(before, "replays") == sum(n in (1, 2, 4, 8) for n in ns)
    assert _delta(before, "not_captured") == sum(n in (3, 6) for n in ns)


@needs_cuda
def test_ineligible_steps_run_eager():
    rng = torch.Generator().manual_seed(3)
    steps = [make_step(rng, 4) for _ in range(3)]
    mixed = make_step(rng, 3, num_reqs=5)  # 3 spec rows + 2 prefill rows
    remapped = dict(
        make_step(rng, 4),
        request_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda"),
    )
    steps += [mixed, remapped, make_step(rng, 4)]
    before = rcg.commit_stats()
    bad, _ = run_ab(steps, ["graph"] * len(steps), warm=[3, 4])
    assert bad is None
    assert _delta(before, "not_uniform_decode") == 1
    assert _delta(before, "request_indices") == 1
    assert _delta(before, "replays") == 4


@needs_cuda
def test_moved_builder_buffers_never_replay():
    # Fresh spec state indices / query offsets (a forward without a FULL graph):
    # the key differs from the warmup capture, so the step runs staged-eager.
    before = rcg.commit_stats()
    bad, _ = run_ab(
        _steps([4] * 6), ["graph"] * 6, warm=[4], fresh_builder_buffers=True
    )
    assert bad is None
    assert _delta(before, "replays") == 0


@needs_cuda
def test_non_align_mode():
    before = rcg.commit_stats()
    bad, cand = run_ab(_steps([4] * 6), ["graph"] * 6, warm=[4], align=False)
    assert bad is None
    assert cand.rs._commit_graphs.num_graphs == 1
    assert _delta(before, "replays") == 6


@needs_cuda
def test_serving_never_captures(monkeypatch):
    def refuse(self, *a, **k):
        raise AssertionError("capture during serving")

    monkeypatch.setattr(torch.cuda.CUDAGraph, "capture_begin", refuse)
    before = rcg.commit_stats()
    bad, cand = run_ab(_steps([4] * 6), ["graph"] * 6)  # booted graph, no warmup
    assert bad is None
    assert cand.rs._commit_graphs.num_graphs == 0
    assert _delta(before, "replays") == 0
    assert _delta(before, "not_captured") == 6


@needs_cuda
def test_invalidated_capture_propagates_and_process_recovers():
    cand = World(0, boot_mode="graph")
    graphs = cand.rs._commit_graphs
    real = graphs._launch

    def illegal_during_capture(*a, **k):
        if torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()  # not permitted while a global capture runs
        real(*a, **k)

    graphs._launch = illegal_during_capture
    with pytest.raises(Exception):  # noqa: B017 - RuntimeError or AcceleratorError
        cand.warmup([4])
    assert graphs.num_graphs == 0
    graphs._launch = real
    # The process is usable afterwards: allocation, eager and staged commits, a
    # new capture, and replays that match the eager reference bitwise.
    big = torch.ones(1 << 22, device="cuda")
    assert float(big.sum()) == float(1 << 22)
    before = rcg.commit_stats()
    bad, _ = run_ab(_steps([4] * 4), ["graph"] * 4, warm=[4], cand=cand)
    assert bad is None
    assert _delta(before, "replays") == 4


@needs_cuda
def test_rebound_context_tensor_refuses_replay():
    import dataclasses

    cand = World(0, boot_mode="graph")
    cand.warmup([4])
    ctx = cand.groups[0].ctx
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.final_state_indices = ctx.final_state_indices.clone()  # type: ignore[misc]
    # Bypass the freeze to simulate a re-bound buffer: the replay must refuse.
    object.__setattr__(ctx, "final_state_indices", ctx.final_state_indices.clone())
    before = rcg.commit_stats()
    bad, _ = run_ab(_steps([4] * 3), ["graph"] * 3, cand=cand)
    assert bad is None
    assert _delta(before, "binding_moved") == 1
    assert _delta(before, "replays") == 0


@needs_cuda
def test_proof_line_and_reset(monkeypatch):
    lines = []
    monkeypatch.setattr(rcg.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    bad, cand = run_ab(_steps([2] * 4), ["graph"] * 4, warm=[2])
    assert bad is None
    prefix = "[recoverssm-commit] mode=graph captured="
    proof = [ln for ln in lines if ln.startswith(prefix)]
    assert len(proof) == 1 and "captured=1" in proof[0]
    cand.rs.reset()
    assert cand.rs._commit_graphs.num_graphs == 0


# Negative controls: each faulty candidate must make the comparison fail.


@needs_cuda
def test_negative_stale_staging_is_detected(monkeypatch):
    cand = World(0, boot_mode="graph")
    graphs = cand.rs._commit_graphs
    real_stage = graphs._stage

    def stage_without_num_computed(plan, ns, im):
        # Bug under test: once a graph exists, num_computed is not re-staged.
        keys = [k for k in graphs._staging if k[0].startswith("num_computed")]
        keep = [graphs._staging[k].clone() for k in keys] if graphs._graphs else []
        out = real_stage(plan, ns, im)
        for k, v in zip(keys, keep):
            graphs._staging[k].copy_(v)
        return out

    monkeypatch.setattr(graphs, "_stage", stage_without_num_computed)
    bad, _ = run_ab(_steps([4] * 6), ["graph"] * 6, warm=[4], cand=cand)
    assert bad is not None


@needs_cuda
def test_negative_missing_postprocess_is_detected():
    cand = World(0, boot_mode="graph")

    def chain_without_postprocess(step, ns, im, *, num_computed_tokens=None, **_):
        for i, m in enumerate(step):
            m.commit_recoverssm_state(ns, num_computed_tokens=num_computed_tokens[i])

    cand.rs._commit_graphs._launch = chain_without_postprocess
    bad, _ = run_ab(_steps([4] * 6), ["graph"] * 6, warm=[4], cand=cand)
    assert bad is not None


@needs_cuda
def test_negative_swapped_group_mapping_is_detected():
    cand = World(0, boot_mode="graph")
    real = cand.rs._commit_graphs._launch

    def chain_with_swapped_indices(step, *a, **k):
        # Group 1 commits with group 0's state indices (one shared mapping).
        m0, m1 = step[0], step[1]
        c0, c1 = m0.recoverssm_commit, m1.recoverssm_commit
        m1.recoverssm_commit = GDNRecoverSSMCommitMetadata(
            c0.state_indices,
            c1.query_start_loc,
            c1.request_indices,
            c1.block_table,
            c1.num_computed_tokens,
            c1.block_size,
        )
        try:
            real(step, *a, **k)
        finally:
            m1.recoverssm_commit = c1

    cand.rs._commit_graphs._launch = chain_with_swapped_indices
    bad, _ = run_ab(_steps([4] * 6), ["graph"] * 6, warm=[4], cand=cand)
    assert bad is not None


class _NoBoundaryWorld(World):
    """Candidate whose chain (and therefore its captured graph) lacks the
    boundary fold launch."""

    def _no_boundary(self, fn, *a, **k):
        from vllm.model_executor.layers.mamba.gdn import recoverssm_gdn as mod

        real = mod._commit_gdn_state_kernel

        class _Skip:
            def __getitem__(self, grid):
                def launch(*args, BOUNDARY, **kwargs):
                    if not BOUNDARY:
                        real[grid](*args, BOUNDARY=BOUNDARY, **kwargs)

                return launch

        mod._commit_gdn_state_kernel = _Skip()
        try:
            fn(*a, **k)
        finally:
            mod._commit_gdn_state_kernel = real

    def warmup(self, ns):
        self._no_boundary(super().warmup, ns)

    def commit(self, inp, mode, **kw):
        self._no_boundary(super().commit, inp, mode, **kw)


@needs_cuda
def test_negative_missing_boundary_fold_is_detected():
    cand = _NoBoundaryWorld(0, boot_mode="graph")
    bad, _ = run_ab(_steps([4] * 6), ["graph"] * 6, warm=[4], cand=cand)
    assert bad is not None
