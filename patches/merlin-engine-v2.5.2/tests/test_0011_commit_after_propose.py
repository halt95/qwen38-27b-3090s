# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE: the RecoverSSM GDN commit and the
mamba align postprocess (the model state's postprocess "tail") are enqueued
after the MTP proposal instead of before it, on the same stream.

CPU: the tail runs later with the same arguments and in the same internal
order (commit, then the align postprocess that reads the accepted counts the
commit resets), the accepted-count scatter the drafter-side state needs still
runs before the proposal, and lifecycle guards fire. The runner wiring order is
checked on the source. GPU (skipped without CUDA): the real commit kernels and
the real fused align postprocess give bit-identical GDN/conv state, state
indices and accepted counts whether the tail runs before or after a
proposal-like workload (scratch work + the real set_draft_tokens copy path),
for identical inputs; a control that overwrites a commit input during the
"proposal" flips the comparison only after the unmodified comparison passes.
The real MTP propose / draft FULL replay needs a model and is not covered by
this CPU/GPU unit test (an end-to-end engine test is needed for it)."""

import ast
import random
from types import SimpleNamespace

import pytest
import torch

import vllm.triton_utils as _tu

if getattr(_tu.tl, "constexpr", None) is None:  # triton-less host (CPU tests)
    _tu.tl.constexpr = lambda x: x

from vllm.v1.worker.gpu.model_states import mamba_hybrid as mh  # noqa: E402
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (  # noqa: E402
    MambaHybridModelState,
)


class _Log:
    def __init__(self):
        self.events: list[tuple] = []


class _Kernel:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __getitem__(self, grid):
        return lambda *args, **kw: self.log.events.append((self.name, args))


class _RecoverSSM:
    def __init__(self, log):
        self.log = log
        self._step = ("recorded-step",)

    def commit_step(
        self, num_sampled, idx_mapping, *, state_indices, num_accepted_tokens
    ):
        self.log.events.append(
            (
                "commit_step",
                (num_sampled, idx_mapping, state_indices, num_accepted_tokens),
            )
        )
        self._step = None


class _AlignCtx:
    def __init__(self, log):
        self.log = log

    def run_fused_postprocess_align(self, *args):
        self.log.events.append(("align_postprocess", args))


_BUFS = (torch.ones(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32))


def _state(log, monkeypatch):
    monkeypatch.setattr(mh, "_scatter_num_accepted_kernel", _Kernel(log, "scatter"))
    monkeypatch.setattr(mh, "_fill_num_accepted_kernel", _Kernel(log, "fill"))
    s = MambaHybridModelState.__new__(MambaHybridModelState)
    # Both arms share the state's buffers, so call arguments compare by identity.
    s.num_accepted_tokens_gpu, s._mamba_state_idx_gpu = _BUFS
    s._align_mode = True
    s._mamba_ctx = _AlignCtx(log)
    s.recoverssm = _RecoverSSM(log)
    s._deferred_tail = None
    s._deferred_tail_error = None
    return s


def _args():
    idx = torch.tensor([3, 1, 4], dtype=torch.int32)
    ns = torch.tensor([2, 5, 1], dtype=torch.int32)
    nct = torch.arange(8, dtype=torch.int32)
    return idx, ns, nct


def _same_events(a, b):
    assert [e[0] for e in a] == [e[0] for e in b]
    for (_, x), (_, y) in zip(a, b):
        assert len(x) == len(y)
        for u, v in zip(x, y):
            if isinstance(u, torch.Tensor):
                assert u is v  # the very same tensors, read later
            else:
                assert u == v


def test_deferred_tail_runs_same_calls_after_flush(monkeypatch):
    now, later = _Log(), _Log()
    idx, ns, nct = _args()
    s_now = _state(now, monkeypatch)
    s_now.postprocess_state(idx, ns, nct)
    s_later = _state(later, monkeypatch)
    s_later.postprocess_state(idx, ns, nct, defer_tail=True)
    # Before the proposal only the accepted-count scatter has run.
    assert [e[0] for e in later.events] == ["scatter"]
    s_later.flush_deferred_postprocess()
    assert s_later._deferred_tail is None
    _same_events(now.events, later.events)
    assert [e[0] for e in later.events] == [
        "scatter",
        "commit_step",
        "align_postprocess",
    ]
    # Flushing again is a no-op.
    s_later.flush_deferred_postprocess()
    assert len(later.events) == 3


def test_pending_tail_blocks_next_postprocess(monkeypatch):
    s = _state(_Log(), monkeypatch)
    idx, ns, nct = _args()
    s.postprocess_state(idx, ns, nct, defer_tail=True)
    with pytest.raises(RuntimeError, match="never flushed"):
        s.postprocess_state(idx, ns, nct)


def test_rerecorded_step_fails_the_flush_and_poisons(monkeypatch):
    log = _Log()
    s = _state(log, monkeypatch)
    idx, ns, nct = _args()
    s.postprocess_state(idx, ns, nct, defer_tail=True)
    s.recoverssm._step = ("another-step",)
    with pytest.raises(RuntimeError, match="recorded step changed"):
        s.flush_deferred_postprocess()
    # The marker survives the failure: no tail ran, and the runner is poisoned.
    assert s._deferred_tail is not None
    assert [e[0] for e in log.events] == ["scatter"]
    with pytest.raises(RuntimeError, match="failed"):
        s.check_deferred_postprocess()
    with pytest.raises(RuntimeError, match="failed"):
        s.flush_deferred_postprocess()
    with pytest.raises(RuntimeError, match="failed"):
        s.postprocess_state(idx, ns, nct)


def test_tail_launch_failure_poisons(monkeypatch):
    s = _state(_Log(), monkeypatch)
    idx, ns, nct = _args()

    def boom(*a, **k):
        raise RuntimeError("launch failed")

    s.recoverssm.commit_step = boom
    s.postprocess_state(idx, ns, nct, defer_tail=True)
    with pytest.raises(RuntimeError, match="launch failed"):
        s.flush_deferred_postprocess()
    assert s._deferred_tail is not None
    with pytest.raises(RuntimeError, match="launch failed"):
        s.check_deferred_postprocess()


def test_proposal_exception_blocks_the_next_step(monkeypatch):
    """The proposal raised, so the flush never ran: the check the runner makes
    at the top of execute_model (before any request mutation, metadata build or
    forward) must refuse to continue."""
    s = _state(_Log(), monkeypatch)
    idx, ns, nct = _args()
    s.postprocess_state(idx, ns, nct, defer_tail=True)
    with pytest.raises(RuntimeError, match="never flushed"):
        s.check_deferred_postprocess()
    # A clean flush clears it.
    s.flush_deferred_postprocess()
    s.check_deferred_postprocess()


def _method_calls(path: str, cls: str, method: str) -> list[str]:
    # Parsed from the source file: importing the runner needs the CUDA stack.
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    fn = next(
        f
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == cls
        for f in c.body
        if isinstance(f, ast.FunctionDef) and f.name == method
    )
    names = []
    body = ast.Module(body=fn.body, type_ignores=[])  # not the decorators
    for node in ast.walk(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.append((node.lineno, node.col_offset, node.func.attr))
    return [n for _, _, n in sorted(names)]


def test_runner_wiring_order():
    """postprocess (post_update + scatter) before the proposal; the flush after
    the proposal and the draft-token copies, before the KV connector."""
    import vllm.v1.worker.gpu as gpu_pkg

    path = gpu_pkg.__path__[0] + "/model_runner.py"
    calls = _method_calls(path, "GPUModelRunner", "sample_tokens")
    i_post = calls.index("postprocess_sampled")
    i_propose = calls.index("propose")
    i_set = calls.index("set_draft_tokens")
    i_flush = calls.index("flush_deferred_postprocess")
    i_kv = len(calls) - 1 - calls[::-1].index("post_forward")
    assert i_post < i_propose < i_set < i_flush < i_kv
    assert calls.count("flush_deferred_postprocess") == 1


def test_runner_checks_pending_tail_before_mutating_state():
    import vllm.v1.worker.gpu as gpu_pkg

    path = gpu_pkg.__path__[0] + "/model_runner.py"
    calls = _method_calls(path, "GPUModelRunner", "execute_model")
    assert calls[0] == "check_deferred_postprocess"
    assert calls.index("check_deferred_postprocess") < calls.index("update_requests")


class _LogStub:
    def __init__(self):
        self.info_calls, self.warning_calls = [], []

    def info(self, *a):
        self.info_calls.append(a)

    def warning(self, *a):
        self.warning_calls.append(a)


def _spec(kind):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    if kind == "mamba":
        return MambaSpec(block_size=16, shapes=((4, 8),), dtypes=(torch.float16,))
    return FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
    )


def _runner(groups="fa", pp=None, pcp=None, rssm=True):
    if groups == "missing":
        spec = SimpleNamespace()
    elif groups == "empty":
        spec = SimpleNamespace(attn_groups=[[], []])
    else:
        spec = SimpleNamespace(
            attn_groups=[[], [SimpleNamespace(kv_cache_spec=_spec(groups))]]
        )
    ms = SimpleNamespace(
        recoverssm=object() if rssm else None, supports_deferred_postprocess=True
    )
    return SimpleNamespace(speculator=spec, model_state=ms, pp=pp, pcp=pcp)


@pytest.mark.parametrize(
    "kw, ok, reason",
    [
        ({}, True, "active"),
        ({"groups": "missing"}, False, "not initialized"),
        ({"groups": "empty"}, False, "not initialized"),
        ({"groups": "mamba"}, False, "Mamba/GDN"),
        ({"pp": object()}, False, "pipeline"),
        ({"pcp": object()}, False, "context parallelism"),
        ({"rssm": False}, False, "RecoverSSM is off"),
    ],
)
def test_eligibility(kw, ok, reason):
    from vllm.v1.worker.gpu.recoverssm_order import (
        commit_after_propose_eligibility,
    )

    r = _runner(**kw)
    got, why = commit_after_propose_eligibility(
        r.speculator, r.model_state, r.pp, r.pcp
    )
    assert got is ok and reason in why
    assert commit_after_propose_eligibility(None, r.model_state, None, None) == (
        False,
        "no speculator",
    )


def test_eligibility_cache_follows_the_inventory():
    from vllm.v1.worker.gpu.recoverssm_order import CommitAfterProposeGate

    log = _LogStub()
    gate = CommitAfterProposeGate(log)
    r = _runner()

    def ask():
        return gate.eligible(r.speculator, r.model_state, r.pp, r.pcp)

    assert ask() is True
    assert ask() is True and len(log.info_calls) == 1  # cached
    # The drafter's set_attn ran again (KV cache re-initialization) and now
    # reports a Mamba group: the cached verdict must not survive.
    r.speculator.attn_groups = [[SimpleNamespace(kv_cache_spec=_spec("mamba"))]]
    assert ask() is False
    assert "Mamba" in log.warning_calls[-1][1]
    r.speculator.attn_groups = [[SimpleNamespace(kv_cache_spec=_spec("fa"))]]
    assert ask() is True
    r.pp = object()
    assert ask() is False
    r.pp = None
    del r.speculator.attn_groups
    assert ask() is False


def test_default_off_and_not_a_compile_factor(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE", raising=False)
    assert envs.VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE is False
    monkeypatch.setenv("VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE", "1")
    assert envs.VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE is True
    assert "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE" not in envs.compile_factors()


# --------------------------------------------------------------------- GPU


class _Buf:
    def __init__(self, n, dtype, device):
        self.cpu = torch.zeros(n, dtype=dtype)
        self.gpu = torch.zeros(n, dtype=dtype, device=device)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n=None):
        if n is None:
            self.gpu.copy_(self.cpu)
            return self.gpu
        self.gpu[:n].copy_(self.cpu[:n])
        return self.gpu[:n]


def _gpu_setup(seed):
    """Real RecoverSSM metadata (three GDN groups, align mode, FULL staging) from
    the builder harness, real commit contexts, a REAL MambaSpecDecodeGPUContext
    for the fused align postprocess, random bounded checkpoints/records/conv
    states. Every random input comes from one seeded generator, so two calls
    with the same seed build identical arms."""
    import test_0011_metadata_reuse as H
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_copy_spec,
        get_temporal_copy_spec,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

    saved = H.HV, H.V, H.K, H.CONV_DIM
    H.HV, H.V, H.K, H.CONV_DIM = 2, 32, 32, 64  # Triton-sized heads
    device = torch.device("cuda")
    try:
        groups, builders, kv_cfg, names = H._builders("align", True, device, True, 16)
        rng = random.Random(seed)
        b = H._batch(rng, "align", 16, device, uniform=True)
        shapes = ((H.SPEC_QUERY_LEN + 2, H.CONV_DIM), (H.HV, H.V, H.K))
    finally:
        H.HV, H.V, H.K, H.CONV_DIM = saved
    md = H._build(groups, kv_cfg, b, device, capture=False)
    g = torch.Generator(device="cuda").manual_seed(seed)
    fc = builders[0].vllm_config.compilation_config.static_forward_context
    for layer in fc.values():
        for t in layer.kv_cache:
            # Bounded values: no inf/NaN, so torch.equal is exact.
            t.copy_((torch.rand(t.shape, generator=g, device="cuda") * 0.5).to(t.dtype))
    rs = RecoverSSMState()
    rs.record_step(
        md,
        [[SimpleNamespace(layer_names=n)] for n in names],
        for_capture=False,
    )

    n = b.n
    max_reqs = H.MAX_REQS
    idx = torch.randperm(max_reqs, generator=g, device="cuda")[:n].to(torch.int32)
    ns = torch.randint(
        1, H.SPEC_QUERY_LEN + 1, (n,), generator=g, device="cuda", dtype=torch.int32
    )
    # Post-step computed counts per req slot (as post_update leaves them).
    seq = b.seq_lens.to(torch.int32)
    qlen = b.qsl_gpu.diff().to(torch.int32)
    nct = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    nct[idx.long()] = seq - qlen + ns
    state_idx = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    state_idx[idx.long()] = torch.clamp(seq - qlen - 1, min=0) // 16

    # The real align-postprocess context over the same layers and block tables.
    spec = MambaSpec(
        block_size=16,
        shapes=shapes,
        dtypes=(torch.float32, torch.float16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=H.NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(list(nm), spec) for nm in names],
    )
    copy_funcs = {
        MambaAttentionBackendEnum.GDN_ATTN: (
            get_conv_copy_spec,
            get_temporal_copy_spec,
        )
    }
    persistent = []
    for t in b.tables:
        p = torch.zeros(max_reqs, t.shape[1], dtype=torch.int32, device=device)
        p[idx.long()] = t[:n]
        persistent.append(p)
    ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=max_reqs,
        kv_cache_config=kv_cache_config,
        copy_funcs=copy_funcs,
        device=device,
        make_buffer=lambda k, dtype: _Buf(k, dtype, device),
    )
    ctx.initialize_from_forward_context(
        kv_cache_config,
        fc,
        copy_funcs,
        list(b.tables),
        persistent_block_tables=persistent,
    )

    s = MambaHybridModelState.__new__(MambaHybridModelState)
    s.max_num_reqs = max_reqs
    s.num_accepted_tokens_gpu = torch.ones(max_reqs, dtype=torch.int32, device=device)
    s._mamba_state_idx_gpu = state_idx
    s._align_mode = True
    s._mamba_ctx = ctx
    s.recoverssm = rs
    s._deferred_tail = None
    s._deferred_tail_error = None
    inputs = [idx, ns, nct, state_idx] + list(b.tables)
    return s, fc, builders, idx, ns, nct, [t.clone() for t in inputs]


def _snapshot(s, fc):
    out = [t.clone() for layer in fc.values() for t in layer.kv_cache]
    out += [s.num_accepted_tokens_gpu.clone(), s._mamba_state_idx_gpu.clone()]
    return out


def _proposal_like(builders, bad: bool):
    """Legal: drafter-owned work (scratch arithmetic) plus the real
    DraftTokensHandler.set_draft_tokens copy-stream path. Control (bad):
    overwrite a commit input -- the first group's staged spec state indices --
    which only the deferred commit then reads."""
    from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

    scratch = torch.randn(1 << 16, device="cuda")
    for _ in range(4):
        scratch = scratch * 1.0001 + 1
    handler = DraftTokensHandler(torch.device("cuda"))
    batch = SimpleNamespace(req_ids=["a", "b"], has_structured_output_reqs=True)
    handler.set_draft_tokens(batch, torch.ones(2, 4, dtype=torch.int64, device="cuda"))
    if bad:
        # -1 keeps the ids in range (a 1 becomes the null block, skipped).
        builders[0].spec_state_indices_tensor.sub_(1)
    return handler


def _arm(seed, defer: bool, bad: bool):
    s, fc, builders, idx, ns, nct, inputs = _gpu_setup(seed)
    if defer:
        s.postprocess_state(idx, ns, nct, defer_tail=True)
        _proposal_like(builders, bad)
        s.flush_deferred_postprocess()
    else:
        s.postprocess_state(idx, ns, nct)
        _proposal_like(builders, bad)
    torch.cuda.synchronize()
    return inputs, _snapshot(s, fc)


def _same_inputs(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.equal(x, y), "the two arms were built from different inputs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("seed", range(6))
def test_gpu_commit_after_proposal_is_bit_identical(seed):
    in_ref, ref = _arm(seed, defer=False, bad=False)
    in_out, out = _arm(seed, defer=True, bad=False)
    _same_inputs(in_ref, in_out)
    for a, b in zip(ref, out):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_control_overwriting_a_commit_input_flips():
    flipped = 0
    for seed in range(6):
        # The unmodified comparison of the same seed must pass first, so a flip
        # below is attributable to the overwrite alone.
        in_a, a = _arm(seed, defer=False, bad=False)
        in_b, b = _arm(seed, defer=True, bad=False)
        _same_inputs(in_a, in_b)
        assert all(torch.equal(x, y) for x, y in zip(a, b))
        in_ref, ref = _arm(seed, defer=False, bad=True)
        in_out, out = _arm(seed, defer=True, bad=True)
        _same_inputs(in_ref, in_out)
        flipped += any(not torch.equal(x, y) for x, y in zip(ref, out))
    assert flipped > 0


def test_worker_runtime_switch():
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except ImportError as e:  # the worker module needs the CUDA Python stack
        pytest.skip(f"gpu_worker not importable here: {e}")
    runner = SimpleNamespace(
        _recoverssm_commit_after_propose=False,
        _recoverssm_deferred_steps=7,
        _recoverssm_can_defer=lambda: True,
    )
    worker = SimpleNamespace(model_runner=runner)
    out = Worker.set_recoverssm_commit_after_propose(worker, True)
    assert runner._recoverssm_commit_after_propose is True
    assert out == {"commit_after_propose": True, "eligible": True, "deferred_steps": 7}
    out = Worker.set_recoverssm_commit_after_propose(worker, False)
    assert runner._recoverssm_commit_after_propose is False
    assert out["eligible"] is None
    with pytest.raises(RuntimeError, match="V2 model runner"):
        Worker.set_recoverssm_commit_after_propose(
            SimpleNamespace(model_runner=SimpleNamespace()), True
        )
